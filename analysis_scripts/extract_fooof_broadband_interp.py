#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Broadband (1-150 Hz) aperiodic + per-band oscillation presence, one row per
channel x window, with the notch-filter dips INTERPOLATED out of the spectrum
before fitting.

A copy of `extract_fooof_light.py` (same windows, same light peak budget, same
flattened-spectrum band readout) with two changes:

    1. the fit range is broadband, FIT_LO-FIT_HI = 1-150 Hz, instead of 1-57
    2. the line-noise notches inside that range are located per recording and
       interpolated before the fit

WHY INTERPOLATE

    The preprocessed .fif files are notch filtered (2 Hz notch width, so a
    ~3 Hz wide dip of ~2 log10 units) at 60/120/180 Hz, plus per-patient
    extras (104, 117, 142, ...). Fitting FOOOF across an un-treated notch is
    unusable: measured on NS127_02 english, a 1-150 Hz knee fit gave R2 0.86,
    a nonsense knee, and every "peak" above 70 Hz. The same spectrum with the
    notches interpolated fit at R2 0.999 with alpha/beta peaks where expected.

    fooof 1.1 refuses non-equidistant frequencies, so bins cannot simply be
    dropped. Interpolation is the approach the FOOOF authors recommend for
    line noise (fooof.utils.data.interpolate_spectrum). The version here is
    the same idea - linear in log-log space between the averages of BUFFER
    bins strictly outside the range on either side - vectorised over channels.
    (fooof's own helper includes the last in-range bin in its right-hand
    buffer, which drags the interpolant down at the right edge; that is why
    it is not called directly.)

WHERE THE NOTCH LIST COMES FROM - THE DATA, NOT THE CODE

    The notch table in `movies_ieeg_preprocess_batch.py` cannot be trusted as
    a record of what was applied: in its current form the per-patient branches
    for NS135 / NS151 never reach the filter call, yet the .fif files on disk
    clearly carry those patients' extra notches. So the notches are DETECTED
    per recording from the channel-median Welch spectrum of the whole
    recording: any bin more than NOTCH_MIN_DEPTH log10 units below a local
    log-log line fitted to its neighbours (NOTCH_FIT_INNER..NOTCH_FIT_OUTER Hz
    away, so the dip itself is excluded from the fit). Contiguous flagged
    bins form one notch; each is widened to +/- INTERP_HALF_WIDTH Hz around
    its centre. The search starts at NOTCH_SEARCH_LO Hz because below ~20 Hz
    the curvature of a large alpha peak trips the detector and no line-noise
    notch exists there.

    Every detected notch is written to a per-recording table alongside the
    fit CSV, and compared against EXPECTED_NOTCHES at run time. A mismatch is
    printed, not fatal - the data are authoritative.

WHY FIT_HI = 150 IS SAFE (measured 2026-09-15)

    The preprocessing low-pass is 170 Hz, and on paper MNE's default
    transition band (0.25 * h_freq) would start attenuating near 149 Hz. It
    does not show in the data: on NS127_02 / NS135 / NS151 / NS174_03
    (english and inscapes) the channel-median spectrum stays within
    +/- 0.06 log10 of its own 90-135 Hz log-log trend at every bin from 140
    to 170 Hz. The first real feature above the fit range is the 180 Hz
    notch. So the fit runs to 150 Hz; the PSD is computed to FIT_HI +
    PSD_PAD_HZ so a notch near the ceiling still has a right-hand buffer.

PEAK BUDGET

    MAX_N_PEAKS is raised from 6 to 8 because the range is 2.5x wider and
    broad HFA bumps compete for the budget. As in the light script, the
    Gaussians are only there so peaks are removed before the aperiodic refit;
    the band readout comes from the flattened spectrum and is not capped.

WHAT THIS DELIBERATELY DOES NOT DO

    - no change to the .fif files: interpolation is on the Welch spectrum only.
      The wavelet HFA band and bandpass gamma/HFA still integrate over the dips.
    - no per-band Gaussian parameters
    - no delta (see extract_fooof_light.py)
    - no windowing, z-scoring, attention-label merge, or event reconstruction
    - it does not write into rolling_fooof_light or rolling_fooof_*_26Jun26

PERFORMANCE

    BLAS threads pinned to 1 (must precede numpy import). Parallelism by
    recording via FOOOF_N_WORKERS / FOOOF_WORKER, as in the light script.

Output (per recording):
    rolling_fooof_broadband_interp/{pat}/{entry_id}_{vid}_fooof_broadband.csv
    rolling_fooof_broadband_interp/{pat}/{entry_id}_{vid}_notch_table.csv
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
PATIENTS = None               # hand list of patients; None -> INCLUSION_FILES
# Recording set = union of the neural inclusion list (42, what every
# attention-state contrast reads) and the eye/label set (49, what has
# attention labels). Matched on (video, patient, run). Recordings in these
# files with no `avg`-referenced .fif are printed and skipped (NS140_02
# english, 2026-09-15). Set to () to run every recording in PREP_DIR.
INCLUSION_FILES = (
    ('descriptives_power_10s/included_recordings.csv', 'patient'),
    ('attention_labels_10s/attention_labels_10s_pooled.csv', 'pat'),
)
REF = 'avg'
OVERWRITE = False

WINDOW_SEC = 10.0
STEP_SEC = 2.5                # matches the Tier 2 window grid
N_FFT_SEC = 2.0               # 0.5 Hz resolution, ~9 Welch segments

FIT_LO, FIT_HI = 1.0, 150.0
APERIODIC_MODE = 'knee'
MAX_N_PEAKS = 8
PEAK_WIDTH_LIMITS = [1, 10]
MIN_PEAK_HEIGHT = 0.1
PEAK_THRESHOLD = 2.0

# --- notch detection + interpolation ---
INTERPOLATE = os.environ.get('FOOOF_INTERPOLATE', '1') == '1'
                              # False -> identical pipeline, no interpolation
                              # (the before/after reference, *_nointerp.csv)
INTERP_HALF_WIDTH = 3.0       # Hz either side of the notch centre. 2 Hz left a
                              # spurious 61.5 Hz peak from the dip's shoulder.
INTERP_BUFFER = 3             # bins averaged on each side, strictly outside
NOTCH_SEARCH_LO = 20.0        # Hz; see header
NOTCH_MIN_DEPTH = 0.5         # log10 units below the local 1/f line
NOTCH_FIT_INNER = 3.0         # local line fitted to bins INNER..OUTER Hz away
NOTCH_FIT_OUTER = 8.0
PSD_PAD_HZ = 10.0             # PSD computed to FIT_HI + pad so a notch near the
                              # top of the range still has a right-hand buffer

# What the preprocessing was meant to apply. Cross-check only; detection wins.
EXPECTED_NOTCHES = {
    'default': (60, 120),
    'NS135': (60, 104, 120),
    'NS151': (60, 104, 117),
    'NS174_03': (60, 97, 120, 142),
}

# Bands read off the flattened spectrum. Gamma keeps the light script's
# definition so the two are comparable; HFA is the new broadband band.
BANDS = {'theta': (3.5, 7.5), 'alpha': (7.5, 13.5),
         'beta': (13.5, 30.5), 'gamma': (30.5, 57.0),
         'hfa': (57.0, 150.0)}

PRESENCE_THRESHOLD = 0.20

PREP_DIR = p(MOVIE_DATA, 'movies_prep_standard')
OUT_ROOT = p(MOVIE_DATA, 'rolling_fooof_broadband_interp')

WORKER = int(os.environ.get('FOOOF_WORKER', 0))
N_WORKERS = int(os.environ.get('FOOOF_N_WORKERS', 1))


# =============================================================================
# HELPERS (shared with extract_fooof_light.py)
# =============================================================================
def _run_label(f):
    m = re.search(r'run[-_]?(\d+)', f, flags=re.IGNORECASE)
    return f'run-{int(m.group(1)):02d}' if m else 'run-01'


def _ses_label(f):
    m = re.search(r'ses[-_]?(\d+)', f, flags=re.IGNORECASE)
    return f'ses-{int(m.group(1)):02d}' if m else ''


def list_recordings(vid_list=None, prep_dir=PREP_DIR, ref=REF, patients=None):
    """Every (vid, pat, fif_path). Deterministic order, so worker slices are stable."""
    vid_list = vid_list or vids
    out = []
    for vid in vid_list:
        for pat in sorted(os.listdir(prep_dir)):
            if patients is not None and pat not in patients:
                continue
            nd = os.path.join(prep_dir, pat, 'Neural_prep')
            if not os.path.isdir(nd):
                continue
            for f in sorted(os.listdir(nd)):
                if (f.endswith('.fif') and ref in f and 'referenced' in f
                        and 'aic' not in f and vid.lower() in f.lower()
                        and not f.startswith('._')):
                    out.append((vid, pat, os.path.join(nd, f)))
    return out


def load_inclusion(files=INCLUSION_FILES, root=MOVIE_DATA):
    """Set of (video, patient, run-label) from the union of the inclusion files."""
    keep = set()
    for rel, pat_col in files:
        df = pd.read_csv(p(root, rel), usecols=['video', pat_col, 'run']).drop_duplicates()
        keep |= set(zip(df['video'], df[pat_col], df['run']))
    return keep


def select_recordings(recs, keep):
    """Filter list_recordings() output to `keep`; report members with no file."""
    have = {(vid, pat, _run_label(os.path.basename(fp))): (vid, pat, fp)
            for vid, pat, fp in recs}
    missing = sorted(k for k in keep if k not in have)
    if missing:
        print('inclusion entries with no matching .fif (skipped):', flush=True)
        for k in missing:
            print('   ', k, flush=True)
    return [have[k] for k in sorted(keep) if k in have]


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


def window_bounds(n_samp, fs, window_sec=WINDOW_SEC, step_sec=STEP_SEC):
    """(win_index, start_sample, end_sample) on the shared 10 s / 2.5 s grid."""
    n = int(np.floor((n_samp / fs - window_sec) / step_sec)) + 1
    w = int(round(window_sec * fs))
    return [(i, int(round(i * step_sec * fs)), int(round(i * step_sec * fs)) + w)
            for i in range(max(n, 0))
            if int(round(i * step_sec * fs)) + w <= n_samp]


# =============================================================================
# NOTCH DETECTION + INTERPOLATION
# =============================================================================
def local_loglog_residual(freqs, log_psd, inner=NOTCH_FIT_INNER,
                          outer=NOTCH_FIT_OUTER):
    """
    For each bin, fit a straight line in log-log space to the bins that are
    between `inner` and `outer` Hz away (the bin's own neighbourhood is left
    out so a dip does not pull its own baseline down) and return
    log_psd - line. Negative = below the local 1/f trend.
    """
    lf = np.log10(freqs)
    resid = np.full_like(log_psd, np.nan)
    for i in range(len(freqs)):
        d = np.abs(freqs - freqs[i])
        m = (d >= inner) & (d <= outer)
        if m.sum() < 4:
            continue
        A = np.vstack([lf[m], np.ones(m.sum())]).T
        c = np.linalg.lstsq(A, log_psd[m], rcond=None)[0]
        resid[i] = log_psd[i] - (c[0] * lf[i] + c[1])
    return resid


def detect_notches(freqs, psd, search_lo=NOTCH_SEARCH_LO,
                   min_depth=NOTCH_MIN_DEPTH, half_width=INTERP_HALF_WIDTH):
    """
    Locate notch-filter dips in a (channels x freqs) PSD.

    Returns a DataFrame with one row per notch: centre_hz, lo_hz, hi_hz
    (the interpolation range), depth_log10 (deepest residual), n_bins.
    """
    log_med = np.log10(np.median(psd, axis=0))
    resid = local_loglog_residual(freqs, log_med)
    # Ignore the last few bins: the local fit has no right-hand neighbours
    # there and a notch that close to the top of the PSD has no buffer to
    # interpolate from anyway (it is above FIT_HI by construction of the pad).
    search_hi = freqs[-1] - (NOTCH_FIT_INNER + half_width)
    flagged = (resid < -min_depth) & (freqs >= search_lo) & (freqs <= search_hi)
    idx = np.flatnonzero(flagged)
    rows = []
    if idx.size:
        # split into runs of contiguous bins (allow a 1-bin gap)
        breaks = np.flatnonzero(np.diff(idx) > 2)
        for run in np.split(idx, breaks + 1):
            centre = float(np.round(np.mean(freqs[run]) * 2) / 2)   # 0.5 Hz grid
            rows.append(dict(centre_hz=centre,
                             lo_hz=centre - half_width,
                             hi_hz=centre + half_width,
                             depth_log10=float(np.nanmin(resid[run])),
                             n_bins=int(run.size)))
    tab = pd.DataFrame(rows, columns=['centre_hz', 'lo_hz', 'hi_hz',
                                      'depth_log10', 'n_bins'])
    # merge overlapping ranges
    if len(tab) > 1:
        tab = tab.sort_values('lo_hz').reset_index(drop=True)
        merged = [tab.iloc[0].to_dict()]
        for _, r in tab.iloc[1:].iterrows():
            if r.lo_hz <= merged[-1]['hi_hz']:
                merged[-1]['hi_hz'] = max(merged[-1]['hi_hz'], r.hi_hz)
                merged[-1]['depth_log10'] = min(merged[-1]['depth_log10'],
                                                r.depth_log10)
                merged[-1]['n_bins'] += r.n_bins
            else:
                merged.append(r.to_dict())
        tab = pd.DataFrame(merged)
    return tab


def interpolate_notches(freqs, psd, ranges, buffer=INTERP_BUFFER):
    """
    Replace psd (channels x freqs, linear power) inside each [lo, hi] range
    with a straight line in log-log space drawn between the mean of `buffer`
    bins immediately BELOW lo and the mean of `buffer` bins immediately
    ABOVE hi. Bins inside the range never enter the buffers.

    Ranges without a full buffer on both sides are skipped and reported.
    Returns (psd_interp, skipped_ranges).
    """
    out = np.array(psd, dtype=float, copy=True)
    lf = np.log10(freqs)
    skipped = []
    for lo, hi in ranges:
        inside = np.flatnonzero((freqs >= lo) & (freqs <= hi))
        if inside.size == 0:
            continue
        i0, i1 = inside[0], inside[-1]
        if i0 - buffer < 0 or i1 + buffer + 1 > len(freqs):
            skipped.append((lo, hi))
            continue
        left = slice(i0 - buffer, i0)
        right = slice(i1 + 1, i1 + buffer + 1)
        x0, x1 = lf[left].mean(), lf[right].mean()
        y0 = np.log10(out[:, left]).mean(axis=1)
        y1 = np.log10(out[:, right]).mean(axis=1)
        slope = (y1 - y0) / (x1 - x0)
        out[:, inside] = 10 ** (y0[:, None] + slope[:, None] * (lf[inside] - x0))
    return out, skipped


def compare_to_expected(pat, tab, tol=1.5):
    """Print how detected notches line up with EXPECTED_NOTCHES. Not fatal."""
    exp = [f for f in EXPECTED_NOTCHES.get(pat, EXPECTED_NOTCHES['default'])
           if NOTCH_SEARCH_LO <= f <= FIT_HI + PSD_PAD_HZ]
    det = list(tab.centre_hz) if len(tab) else []
    missing = [f for f in exp if not any(abs(f - c) <= tol for c in det)]
    extra = [c for c in det if not any(abs(f - c) <= tol for f in exp)]
    msg = f'      notches detected: {det}'
    if missing:
        msg += f'  | expected but NOT found: {missing}'
    if extra:
        msg += f'  | found but not expected: {extra}'
    print(msg, flush=True)


# =============================================================================
# FIT
# =============================================================================
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
    fm.fit(freqs, psd_ch, [FIT_LO, FIT_HI])
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
        'Peak_CFs': ';'.join(f'{c:.1f}' for c in fm.peak_params_[:, 0]),
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


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':

    recs = list_recordings(patients=PATIENTS)
    if PATIENTS is None and INCLUSION_FILES:
        recs = select_recordings(recs, load_inclusion())
    if N_WORKERS > 1:
        recs = [r for i, r in enumerate(recs) if i % N_WORKERS == WORKER]
    tag = f'[w{WORKER}/{N_WORKERS}] ' if N_WORKERS > 1 else ''
    suffix = 'fooof_broadband' if INTERPOLATE else 'fooof_broadband_nointerp'

    print(f'{tag}{len(recs)} recording(s)  |  {APERIODIC_MODE} aperiodic, '
          f'fit {FIT_LO}-{FIT_HI} Hz, max_n_peaks {MAX_N_PEAKS}, '
          f'{WINDOW_SEC:.0f}s/{STEP_SEC:.1f}s windows, '
          f'interpolate={INTERPOLATE}', flush=True)
    print(f'{tag}-> {OUT_ROOT}\n', flush=True)

    t_start = time.time()
    n_done = 0

    for vid, pat, fif_path in recs:
        fname = os.path.basename(fif_path)
        ses, run = _ses_label(fname), _run_label(fname)
        entry_id = f'{pat}_{ses}_{run}' if ses else f'{pat}_{run}'

        pat_dir = os.path.join(OUT_ROOT, pat)
        os.makedirs(pat_dir, exist_ok=True)
        out_csv = os.path.join(pat_dir, f'{entry_id}_{vid}_{suffix}.csv')
        notch_csv = os.path.join(pat_dir, f'{entry_id}_{vid}_notch_table.csv')
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
        psd_hi = min(FIT_HI + PSD_PAD_HZ, fs / 2)

        # ---- notch detection on the whole-recording spectrum ----
        psd_rec, f_rec = psd_array_welch(
            lfp, sfreq=fs, fmin=FIT_LO, fmax=psd_hi,
            n_fft=n_fft, n_overlap=n_fft // 2, average='mean')
        notch_tab = detect_notches(f_rec, psd_rec)
        notch_tab.insert(0, 'Entry_ID', entry_id)
        notch_tab.insert(1, 'Video', vid)
        notch_tab.to_csv(notch_csv, index=False, float_format='%.4g')
        compare_to_expected(pat, notch_tab)
        ranges = [(r.lo_hz, r.hi_hz) for r in notch_tab.itertuples()]
        ranges_str = ';'.join(f'{lo:g}-{hi:g}' for lo, hi in ranges)

        rows = []
        for win_idx, s0, s1 in bounds:
            try:
                psd, freqs = psd_array_welch(
                    lfp[:, s0:s1], sfreq=fs, fmin=FIT_LO, fmax=psd_hi,
                    n_fft=n_fft, n_overlap=n_fft // 2, average='mean')
            except Exception as exc:
                print(f'{tag}  PSD failed, window {win_idx}: {exc}', flush=True)
                continue
            if INTERPOLATE and ranges:
                psd, skipped = interpolate_notches(freqs, psd, ranges)
                if skipped and win_idx == 0:
                    print(f'{tag}  WARNING ranges without buffer, not '
                          f'interpolated: {skipped}', flush=True)
            base = dict(Patient=pat, Session=ses, Run=run, Video=vid,
                        Entry_ID=entry_id, Window_Index=win_idx,
                        Window_Start_Sec=s0 / fs, Window_End_Sec=s1 / fs,
                        Window_Center_Sec=(s0 + s1) / (2 * fs),
                        FOOOF_Fit_Range_Low=FIT_LO, FOOOF_Fit_Range_High=FIT_HI,
                        Notch_Interpolated=bool(INTERPOLATE and ranges),
                        Notch_Ranges_Hz=ranges_str)
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
              f'exp {df.Aperiodic_Exponent.median():.2f}  '
              f'theta {100*df.Has_theta.mean():.0f}% alpha {100*df.Has_alpha.mean():.0f}%',
              flush=True)

    print(f'\n{tag}done: {n_done} recording(s) in '
          f'{(time.time()-t_start)/60:.1f} min', flush=True)
