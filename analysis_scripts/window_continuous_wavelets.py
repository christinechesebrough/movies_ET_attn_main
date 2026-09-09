#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Window the continuous z-scored wavelets (TODO A10).

Supplies the reduction step that `wavelet_extract_windows.py` was meant to
perform but omits: it computes the 236 window boundaries, writes them, and
never averages within them, leaving 15.8 GB per recording where ~31 MB was
intended.

Input HDF5 (per recording), from wavelet_extract_windows.py:
    {WAVELET_CONTINUOUS_Z}/{vid}/{pat}/*_log_robust_z_wavelets_10s_windows.h5
        pow_tf_log_z         : channels x freqs x samples  (CONTINUOUS despite
                               the filename; verified correct in TODO B3b)
        robust_median        : channels x freqs
        robust_sd            : channels x freqs
        window_start_samples : window boundaries, already computed
        window_end_samples
        labels_ip, freqs_tf

Output HDF5 (per recording):
        pow_log_mean    : channels x freqs x windows
        pow_log_median  : channels x freqs x windows
        pow_log_sd      : channels x freqs x windows
        frac_bad        : windows            fraction flagged bad by 1 s QC
        n_bad_qc_wins   : windows            count of bad 1 s QC windows
        robust_median   : channels x freqs   carried forward
        robust_sd       : channels x freqs   carried forward
        window_* , labels_ip, freqs_tf

WHY LOG POWER RATHER THAN Z:
    The input is z-scored per (channel, frequency) across the whole recording.
    That is an affine transform with constants fixed per channel-frequency, so
    it is exactly invertible - log = z * robust_sd + robust_median - and
    windowing COMMUTES with it (verified to 1e-15):

        mean_win(z)   = (mean_win(log)   - m) / s
        median_win(z) = (median_win(log) - m) / s
        sd_win(z)     =  sd_win(log) / s

    Storing log power therefore loses nothing and gains a lot: absolute scale
    is preserved, and any alternative normalisation (per condition, per
    subject, baseline-relative) stays available. Storing z instead would
    freeze one normalisation choice at the windowing step, and z has its grand
    mean pinned at ~0 by construction - so a state contrast is forced to be
    symmetric about zero and "is power elevated overall" becomes unaskable.

STATISTICS - READ THIS:
    Windows are 10 s with a 2.5 s step, so **75% overlap**: adjacent windows
    share three quarters of their samples. The 236 windows are NOT 236
    independent observations. A 599 s recording contains 60 non-overlapping
    10 s epochs.

        for plotting    use all 236 windows (smooth, no visual stepping)
        for statistics  use every 4th window -> 59 independent observations

    Every 4th window is exactly non-overlapping (step 2.5 s x 4 = the 10 s
    window length). Treating all 236 as independent inflates df ~4x and makes
    p-values anticonservative.

    Full frequency resolution (76 bins) is retained deliberately: it permits
    frequency-resolved cluster-based permutation testing rather than assuming
    the effect lives inside a canonical band.

Deliberately NOT done here:
    No z-scoring, no bad-window exclusion (frac_bad is recorded so exclusion or
    weighting is an analysis-time choice), no band averaging, no aggregation
    across recordings.
"""

import os
import sys
import glob
import h5py
import numpy as np
import pandas as pd

_src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
if _src not in sys.path:
    sys.path.insert(0, _src)

from paths import find                    # noqa: E402

# =============================================================================
# PARAMETERS
# =============================================================================
CONT_Z_DIR = find('wavelet_continuous_z')
BAD_WIN_DIR = find('movies_bad_windows')
OUT_DIR = os.path.join(os.path.dirname(CONT_Z_DIR or '.'), 'wavelet_windowed_10s')

CHANNEL_BATCH = 8          # channels held in memory at once; 8 x 76 x 359428
                           # float64 ~ 1.7 GB. Lower it if memory is tight.

vids = ['despicable_me_english']     # hand-edited knob, as elsewhere in this repo
pats = None                          # None = every patient found

OVERWRITE = False


# =============================================================================
# HELPERS
# =============================================================================

def _real(paths):
    return [p for p in paths if not os.path.basename(p).startswith('._')]


def load_bad_mask(pat, vid, n_samples, fs=600.0):
    """
    Per-sample bad mask from the 1 s QC files.

    QC granularity is 1 s (599-600 rows) and maps cleanly onto 600 Hz samples.
    Returns None when no QC file exists for this recording, in which case
    frac_bad is written as NaN rather than a misleading zero.
    """
    hits = _real(glob.glob(os.path.join(BAD_WIN_DIR or '', pat, f'*{vid}*qc*.csv')))
    if not hits:
        return None
    qc = pd.read_csv(hits[0])
    if 'bad' not in qc.columns:
        return None
    mask = np.zeros(n_samples, dtype=bool)
    for _, row in qc[qc['bad'].astype(bool)].iterrows():
        a = int(row['start_time_sec'] * fs)
        b = int(row['end_time_sec'] * fs)
        mask[max(a, 0):min(b, n_samples)] = True
    return mask


def window_stats(block, starts, ends):
    """
    mean / median / sd within each window for one channel batch.

    block : (n_ch, n_freq, n_samples) log power
    returns three arrays of (n_ch, n_freq, n_windows)
    """
    n_ch, n_freq = block.shape[:2]
    n_win = len(starts)
    out_mean = np.empty((n_ch, n_freq, n_win), dtype=np.float32)
    out_med = np.empty_like(out_mean)
    out_sd = np.empty_like(out_mean)
    for w, (a, b) in enumerate(zip(starts, ends)):
        seg = block[:, :, a:b]
        out_mean[:, :, w] = seg.mean(axis=2)
        out_med[:, :, w] = np.median(seg, axis=2)
        out_sd[:, :, w] = seg.std(axis=2)
    return out_mean, out_med, out_sd


def process_recording(in_path, out_dir=OUT_DIR, overwrite=OVERWRITE):
    base = os.path.basename(in_path).replace(
        '_log_robust_z_wavelets_10s_windows.h5', '')
    with h5py.File(in_path, 'r') as h:
        pat = str(h.attrs.get('pat', 'UNKNOWN'))
        vid = str(h.attrs.get('vid', 'UNKNOWN'))
        run_label = str(h.attrs.get('run_label', 'run-01'))
        fs = float(h.attrs.get('fs', 600.0))
        starts = h['window_start_samples'][:]
        ends = h['window_end_samples'][:]
        n_ch, n_freq, n_samples = h['pow_tf_log_z'].shape

        out_path = os.path.join(out_dir, vid, pat, f'{base}_windowed_10s.h5')
        if os.path.exists(out_path) and not overwrite:
            return out_path, 'skipped'

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        n_win = len(starts)

        with h5py.File(out_path, 'w') as o:
            for name in ('pow_log_mean', 'pow_log_median', 'pow_log_sd'):
                o.create_dataset(name, shape=(n_ch, n_freq, n_win),
                                 dtype=np.float32, compression='gzip',
                                 compression_opts=4)

            med_all = h['robust_median'][:]
            sd_all = h['robust_sd'][:]

            for c0 in range(0, n_ch, CHANNEL_BATCH):
                c1 = min(c0 + CHANNEL_BATCH, n_ch)
                z = h['pow_tf_log_z'][c0:c1, :, :].astype(np.float64)
                # undo the z-scoring: exactly invertible, verified in A10 notes
                logp = z * sd_all[c0:c1, :, None] + med_all[c0:c1, :, None]
                del z
                m, md, sd = window_stats(logp, starts, ends)
                del logp
                o['pow_log_mean'][c0:c1] = m
                o['pow_log_median'][c0:c1] = md
                o['pow_log_sd'][c0:c1] = sd
                del m, md, sd

            # --- per-window QC fraction -------------------------------------
            bad = load_bad_mask(pat, vid, n_samples, fs)
            if bad is None:
                frac = np.full(n_win, np.nan, dtype=np.float32)
                nbad = np.full(n_win, -1, dtype=np.int32)
            else:
                frac = np.array([bad[a:b].mean() for a, b in zip(starts, ends)],
                                dtype=np.float32)
                nbad = np.array([int(round(bad[a:b].sum() / fs))
                                 for a, b in zip(starts, ends)], dtype=np.int32)
            o.create_dataset('frac_bad', data=frac)
            o.create_dataset('n_bad_qc_wins', data=nbad)

            o.create_dataset('robust_median', data=med_all)
            o.create_dataset('robust_sd', data=sd_all)
            o.create_dataset('labels_ip', data=h['labels_ip'][:])
            o.create_dataset('freqs_tf', data=h['freqs_tf'][:])
            for k in ('window_start_samples', 'window_end_samples',
                      'window_start_sec', 'window_end_sec', 'window_centers_sec'):
                if k in h:
                    o.create_dataset(k, data=h[k][:])

            o.attrs['pat'] = pat
            o.attrs['vid'] = vid
            o.attrs['run_label'] = run_label
            o.attrs['fs_original'] = fs
            o.attrs['window_sec'] = 10
            o.attrs['overlap_sec'] = 7.5
            o.attrs['step_sec'] = 2.5
            o.attrs['n_windows'] = n_win
            o.attrs['representation'] = 'log10_wavelet_power_windowed'
            o.attrs['normalization'] = 'none - use robust_median/robust_sd to z-score'
            o.attrs['independent_window_stride'] = 4
            o.attrs['independence_note'] = (
                'Windows overlap 75%. For statistics use every 4th window '
                '(stride 4) for non-overlapping, independent observations. '
                'All windows may be used for plotting.')
            o.attrs['source_file'] = in_path
    return out_path, 'written'


def find_inputs(vid, pats=None):
    hits = _real(sorted(glob.glob(os.path.join(CONT_Z_DIR, vid, '*', '*.h5'))))
    if pats:
        hits = [h for h in hits if os.path.basename(os.path.dirname(h)) in pats]
    return hits


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    print(f'input  : {CONT_Z_DIR}')
    print(f'output : {OUT_DIR}\n')
    for vid in vids:
        inputs = find_inputs(vid, pats)
        print(f'{vid}: {len(inputs)} recordings')
        for p in inputs:
            try:
                out, status = process_recording(p)
                sz = os.path.getsize(out) / 1e6 if os.path.exists(out) else 0
                print(f'   {status:8s} {sz:8.1f} MB  {os.path.basename(out)[:62]}')
            except Exception as exc:
                print(f'   FAILED {os.path.basename(p)[:52]}: '
                      f'{type(exc).__name__}: {exc}')
    print('\nDone.')
