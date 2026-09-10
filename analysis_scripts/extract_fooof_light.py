#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fast aperiodic + per-band oscillation presence, one row per channel x window.

A deliberately cheap alternative to the full peak-parameterisation refit
(TODO C0). It answers two questions and no more:

    1. what is the aperiodic background in this window?   (offset, knee, exponent)
    2. is there an oscillation in theta / alpha / beta / gamma, and where?

WHY NOT JUST LOWER max_n_peaks

    Peak detection via Gaussians is where the cost is, and capping it low
    introduces a BAND-DEPENDENT bias. Measured on 9,994 channel-windows, the
    fraction of true peaks (cap 12) still detected at a lower cap:

        cap   theta   alpha    beta   gamma
          6   73.4%   80.8%   93.2%   94.4%
          8   86.2%   90.2%   97.6%   99.1%
         10   96.4%   97.5%   99.6%    100%

    Beta and gamma peaks are more numerous and taller, so they consume the peak
    budget and crowd theta out first. The loss is worst in exactly the bands
    this project cares about, and it would be state-dependent if beta/gamma
    structure varies with attention - a detection bias inside the contrast.

WHAT THIS DOES INSTEAD

    Fit the aperiodic model with a SMALL peak budget (peaks are still fit, so
    they are removed before the final aperiodic refit and do not drag the
    slope), then read per-band oscillatory content directly off the FLATTENED
    spectrum (observed - aperiodic). No Gaussian is fit per band, so there is
    no cap and no crowding: every band is measured independently.

    Per band it stores the maximum of the flattened spectrum, the frequency at
    which it occurs, and the band mean. These are CONTINUOUS and always
    defined. Presence is then a threshold applied to a stored column, which
    means the threshold can be changed later WITHOUT REFITTING.

    `Has_{band}` is provided as a convenience using PRESENCE_THRESHOLD, but the
    continuous columns are authoritative. Do not treat Has_* as the measurement.

BINARY PRESENCE IS THE WRONG READOUT FOR BETA AND GAMMA - MEASURED

    Prevalence of at least one peak per band, over 9,994 channel-windows:

        theta   44.1%      alpha   73.8%
        beta    98.6%      gamma   99.5%

    Beta and gamma are present in essentially EVERY window, so as binary
    variables they have almost no dynamic range and cannot differentiate
    states. Agreement between a flattened-spectrum threshold and the full
    Gaussian fit reflects only that shared base rate (gamma F1 0.998 at a 100%
    false-positive rate is not accuracy, it is prevalence).

    Theta is the band whose presence genuinely varies - and it is also the
    hardest to threshold reliably (best F1 ~0.71 at cap 6).

    So: use the CONTINUOUS columns. {band}_FlatMax and Periodic_{band} have
    full dynamic range in every band, need no threshold, and are defined in
    every window. Reserve presence for theta, where the base rate makes it
    meaningful, and quote it with its detection uncertainty.

WHY max_n_peaks = 6

    Peaks are still fit so that they are removed before the final aperiodic
    refit. Accuracy of the aperiodic parameters against a cap-20 reference,
    with throughput (pinned, single thread):

        cap    r(exponent)  median |d exp|   r(f_knee)   fits/sec
          0        0.980         0.124         0.965        1152
          3        0.968         0.149         0.962         196
          4        0.975         0.122         0.971         139
          6        0.988         0.070         0.986          80
          8        0.994         0.038         0.991          52
         20        1.000         0.000         1.000          ~25

    Cap 6 is the knee of that curve. Its median exponent error (0.070) is about
    a quarter of the natural window-to-window spread of the exponent within a
    channel (0.264), so it is small relative to the signal being measured, at
    3x the speed of cap 8 and 
    ~3x the speed of cap 20.

WHAT THIS DELIBERATELY DOES NOT DO

    - no per-band Gaussian parameters (true CF/bandwidth of a fitted peak)
    - no delta: at Welch n_fft = 2 s there are ~5 usable bins below 3.5 Hz,
      which cannot support any low-frequency estimate. Delta needs the 30 s
      pass in `extract_fooof_knee.py` (PASS='low').
    - no windowing, z-scoring, attention-label merge, or event reconstruction
    - it does not write into rolling_fooof_*_26Jun26

APERIODIC MODE

    'knee', not 'fixed'. These spectra have a knee inside 1-57 Hz in 95.1% of
    windows (median 5.19 Hz); a fixed fit biases the exponent in a way that
    tracks knee position at r = -0.833, making it a confound rather than an
    offset. See CLAUDE.md.

PERFORMANCE

    BLAS threads MUST be pinned to 1. FOOOF is thousands of tiny curve_fit
    calls; an unpinned numpy spawns a thread pool per call and thrashes.
    Measured: 8.6 fits/sec unpinned vs ~200 fits/sec pinned - a 23x difference.
    The env vars at the top of this file do that, and must be set BEFORE numpy
    is imported (they are).

    Parallelism is by recording, via env vars, so the script stays a plain
    top-level script that runs unchanged in Spyder:

        FOOOF_N_WORKERS=8 FOOOF_WORKER=0 python3 analysis_scripts/extract_fooof_light.py

    See launch_fooof_light.sh for the fan-out.

Output (one CSV per recording):
    rolling_fooof_light/{pat}/{entry_id}_{vid}_fooof_light.csv
"""

import os

# Must precede numpy. See PERFORMANCE above.
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')

import sys
import re
import time
import numpy as np
import pandas as pd
import mne
from mne.time_frequency import psd_array_welch
from fooof import FOOOF

mne.set_log_level('ERROR')

_here = os.path.dirname(os.path.abspath(__file__))
_src = os.path.join(os.path.dirname(_here), 'src')
if _src not in sys.path:
    sys.path.insert(0, _src)

from paths import MOVIE_DATA, p                       # noqa: E402
from channel_metadata import load_channel_metadata     # noqa: E402

# =============================================================================
# CONFIG
# =============================================================================
vids = ['despicable_me_english', 'despicable_me_hungarian', 'inscapes']
REF = 'avg'
OVERWRITE = False

WINDOW_SEC = 10.0
STEP_SEC = 2.5                # matches the Tier 2 window grid
N_FFT_SEC = 2.0               # 0.5 Hz resolution, ~9 Welch segments

FIT_LO, FIT_HI = 1.0, 57.0
APERIODIC_MODE = 'knee'
MAX_N_PEAKS = 6               # LIGHT_CAP
PEAK_WIDTH_LIMITS = [1, 10]
MIN_PEAK_HEIGHT = 0.1
PEAK_THRESHOLD = 2.0

# Bands read off the flattened spectrum. Delta is absent by design - see above.
BANDS = {'theta': (3.5, 7.5), 'alpha': (7.5, 13.5),
         'beta': (13.5, 30.5), 'gamma': (30.5, 57.0)}

# Convenience only. The continuous {band}_FlatMax column is authoritative and
# the threshold can be changed downstream without refitting.
PRESENCE_THRESHOLD = 0.20     # LIGHT_THRESH

PREP_DIR = p(MOVIE_DATA, 'movies_prep_standard')
OUT_ROOT = p(MOVIE_DATA, 'rolling_fooof_light')

WORKER = int(os.environ.get('FOOOF_WORKER', 0))
N_WORKERS = int(os.environ.get('FOOOF_N_WORKERS', 1))


# =============================================================================
# HELPERS
# =============================================================================
def _run_label(f):
    m = re.search(r'run[-_]?(\d+)', f, flags=re.IGNORECASE)
    return f'run-{int(m.group(1)):02d}' if m else 'run-01'


def _ses_label(f):
    m = re.search(r'ses[-_]?(\d+)', f, flags=re.IGNORECASE)
    return f'ses-{int(m.group(1)):02d}' if m else ''


def list_recordings(vid_list=None, prep_dir=PREP_DIR, ref=REF):
    """Every (vid, pat, fif_path). Deterministic order, so worker slices are stable."""
    vid_list = vid_list or vids
    out = []
    for vid in vid_list:
        for pat in sorted(os.listdir(prep_dir)):
            nd = os.path.join(prep_dir, pat, 'Neural_prep')
            if not os.path.isdir(nd):
                continue
            for f in sorted(os.listdir(nd)):
                if (f.endswith('.fif') and ref in f and 'referenced' in f
                        and 'aic' not in f and vid.lower() in f.lower()
                        and not f.startswith('._')):
                    out.append((vid, pat, os.path.join(nd, f)))
    return out


def load_lfp(pat, fif_path):
    raw = mne.io.read_raw(fif_path, preload=False)
    bads = [c for c in raw.info.get('bads', []) if c in raw.ch_names]
    if bads:
        raw.drop_channels(bads)
    labels = raw.ch_names
    meta = None
    try:
        meta = load_channel_metadata(pat)
        keep = set(meta['label'].astype(str))
        idx = np.array([l in keep for l in labels])
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f'      no usable correspondence sheet ({type(exc).__name__}); '
              f'keeping all channels', flush=True)
        idx = np.ones(len(labels), dtype=bool)
    lfp = raw.get_data()[idx, :]
    labels = [l for l, k in zip(labels, idx) if k]
    nonflat = np.std(lfp, axis=1) > 1e-6
    lfp = lfp[nonflat]
    labels = [l for l, k in zip(labels, nonflat) if k]
    return lfp, labels, float(raw.info['sfreq']), meta


def knee_freq(ap):
    if len(ap) < 3:
        return np.nan
    _, knee, exponent = ap
    if knee <= 0 or exponent <= 0:
        return np.nan
    return float(knee ** (1.0 / exponent))


def fit_window(psd_ch, freqs, bands=BANDS):
    """
    One channel-window. Returns a dict: aperiodic parameters plus, per band,
    the flattened-spectrum maximum, its frequency, and the band mean.
    """
    fm = FOOOF(peak_width_limits=PEAK_WIDTH_LIMITS,
               max_n_peaks=MAX_N_PEAKS,
               min_peak_height=MIN_PEAK_HEIGHT,
               aperiodic_mode=APERIODIC_MODE,
               peak_threshold=PEAK_THRESHOLD,
               verbose=False)
    fm.fit(freqs, psd_ch)
    ap = fm.aperiodic_params_
    f = fm.freqs
    flat = fm.power_spectrum - fm._ap_fit

    out = {
        'Aperiodic_Offset': float(ap[0]),
        'Aperiodic_Knee': float(ap[1]) if len(ap) > 2 else np.nan,
        'Aperiodic_Exponent': float(ap[-1]),
        'Knee_Freq_Hz': knee_freq(ap),
        'FOOOF_R2': float(fm.r_squared_),
        'FOOOF_Error': float(fm.error_),
        'N_Peaks_Fit': int(fm.peak_params_.shape[0]),
    }
    for name, (lo, hi) in bands.items():
        m = (f >= lo) & (f < hi)
        if not m.any():
            out[f'{name}_FlatMax'] = np.nan
            out[f'{name}_CF'] = np.nan
            out[f'Periodic_{name}'] = np.nan
            out[f'Has_{name}'] = False
            continue
        seg = flat[m]
        i = int(np.argmax(seg))
        out[f'{name}_FlatMax'] = float(seg[i])
        out[f'{name}_CF'] = float(f[m][i])
        out[f'Periodic_{name}'] = float(np.mean(seg))
        out[f'Has_{name}'] = bool(seg[i] >= PRESENCE_THRESHOLD)
    return out


def window_bounds(n_samp, fs, window_sec=WINDOW_SEC, step_sec=STEP_SEC):
    """(win_index, start_sample, end_sample) on the shared 10 s / 2.5 s grid."""
    n = int(np.floor((n_samp / fs - window_sec) / step_sec)) + 1
    w = int(round(window_sec * fs))
    return [(i, int(round(i * step_sec * fs)), int(round(i * step_sec * fs)) + w)
            for i in range(max(n, 0))
            if int(round(i * step_sec * fs)) + w <= n_samp]


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':

    recs = list_recordings()
    if N_WORKERS > 1:
        recs = [r for i, r in enumerate(recs) if i % N_WORKERS == WORKER]
    tag = f'[w{WORKER}/{N_WORKERS}] ' if N_WORKERS > 1 else ''

    print(f'{tag}{len(recs)} recording(s)  |  {APERIODIC_MODE} aperiodic, '
          f'fit {FIT_LO}-{FIT_HI} Hz, max_n_peaks {MAX_N_PEAKS}, '
          f'{WINDOW_SEC:.0f}s/{STEP_SEC:.1f}s windows', flush=True)
    print(f'{tag}-> {OUT_ROOT}\n', flush=True)

    t_start = time.time()
    n_done = 0

    for vid, pat, fif_path in recs:
        fname = os.path.basename(fif_path)
        ses, run = _ses_label(fname), _run_label(fname)
        entry_id = f'{pat}_{ses}_{run}' if ses else f'{pat}_{run}'

        pat_dir = os.path.join(OUT_ROOT, pat)
        os.makedirs(pat_dir, exist_ok=True)
        out_csv = os.path.join(pat_dir, f'{entry_id}_{vid}_fooof_light.csv')
        if not OVERWRITE and os.path.exists(out_csv):
            print(f'{tag}{entry_id} {vid}  exists, skipped', flush=True)
            continue

        t0 = time.time()
        try:
            lfp, labels, fs, meta = load_lfp(pat, fif_path)
        except Exception as exc:
            print(f'{tag}{entry_id} {vid}  LOAD FAILED: '
                  f'{type(exc).__name__}: {exc}', flush=True)
            continue
        if lfp.size == 0 or not labels:
            print(f'{tag}{entry_id} {vid}  no channels, skipped', flush=True)
            continue

        region = {}
        if meta is not None:
            region = dict(zip(meta['label'].astype(str), meta['Y17_Atlas']))

        bounds = window_bounds(lfp.shape[1], fs)
        if not bounds:
            print(f'{tag}{entry_id} {vid}  shorter than one window, skipped',
                  flush=True)
            continue
        n_fft = int(round(fs * N_FFT_SEC))

        rows = []
        for win_idx, s0, s1 in bounds:
            try:
                psd, freqs = psd_array_welch(
                    lfp[:, s0:s1], sfreq=fs, fmin=FIT_LO, fmax=FIT_HI,
                    n_fft=n_fft, n_overlap=n_fft // 2, average='mean')
            except Exception as exc:
                print(f'{tag}  PSD failed, window {win_idx}: {exc}', flush=True)
                continue
            base = dict(Patient=pat, Session=ses, Run=run, Video=vid,
                        Entry_ID=entry_id, Window_Index=win_idx,
                        Window_Start_Sec=s0 / fs, Window_End_Sec=s1 / fs,
                        Window_Center_Sec=(s0 + s1) / (2 * fs),
                        FOOOF_Fit_Range_Low=FIT_LO, FOOOF_Fit_Range_High=FIT_HI)
            for ci, label in enumerate(labels):
                try:
                    res = fit_window(psd[ci], freqs)
                except Exception:
                    continue
                rows.append({**base, 'Channel': label,
                             'Region': region.get(label, None), **res})

        if not rows:
            print(f'{tag}{entry_id} {vid}  no fits, nothing written', flush=True)
            continue

        df = pd.DataFrame(rows)
        df.to_csv(out_csv, index=False, float_format='%.6g')
        n_done += 1
        dt = time.time() - t0
        print(f'{tag}{entry_id} {vid}  {len(labels)}ch x {len(bounds)}win '
              f'= {len(df):,} fits in {dt:.0f}s ({len(df)/dt:.0f}/s)  '
              f'R2 {df.FOOOF_R2.mean():.3f}  f_knee {df.Knee_Freq_Hz.median():.1f}Hz  '
              f'theta {100*df.Has_theta.mean():.0f}% alpha {100*df.Has_alpha.mean():.0f}%',
              flush=True)

    print(f'\n{tag}done: {n_done} recording(s) in '
          f'{(time.time()-t_start)/60:.1f} min', flush=True)
