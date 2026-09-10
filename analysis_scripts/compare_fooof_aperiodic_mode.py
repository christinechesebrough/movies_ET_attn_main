#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Is `aperiodic_mode='fixed'` the right aperiodic model for these spectra?

WHY THIS GATES EVERYTHING ELSE IN THE FOOOF BRANCH

    `extract_all_fooof.py` fits FOOOF with aperiodic_mode='fixed', i.e. a
    single straight line in log-log space, over 1-57 Hz. Real iEEG spectra
    usually bend somewhere in the low frequencies (a "knee"). Fitting one
    straight line through a bend has two consequences, and both land on
    measures this project depends on:

    1. THE EXPONENT IS BIASED, and the bias depends on where the bend sits.
       The exponent is a primary dependent measure for the attention-state
       contrast. If knee position varies with arousal or attention - which is
       exactly the kind of thing that plausibly does - then the bias is not
       constant noise, it is a CONFOUND IN THE CONTRAST.

    2. THE RESIDUAL CARRIES SYSTEMATIC LOW-FREQUENCY CURVATURE. FOOOF fits
       Gaussians to whatever the aperiodic model leaves behind, so a misfit
       bend is absorbed as spurious delta/theta "peaks", and it contaminates
       any band-limited periodic power computed from the flattened spectrum.

    So this comparison decides whether the existing exponents are usable as
    they stand, and whether low-frequency peaks are oscillations or artefacts.

WHAT IS COMPARED

    A  fixed, 1-57 Hz     the current production setting
    B  knee,  1-57 Hz     same range, aperiodic bend modelled explicitly
    C  fixed, 3-57 Hz     the other standard remedy: fit above the bend
                          instead of modelling it

    A and B are directly comparable (same frequency support). C is fit on a
    different support, so its R^2 is NOT comparable to A/B; it is included
    because it is the practical alternative if a knee is present, and because
    exponent_C vs exponent_A shows how much the low end was dragging the slope.

HOW TO READ THE OUTPUT

    knee frequency        f_knee = knee ** (1 / exponent), in Hz. This is the
                          decisive number. If it lands INSIDE the fit range,
                          'fixed' is misspecified for that window. If it sits
                          at or below the 1 Hz floor, 'fixed' was fine.
    delta_r2 = B - A      how much explanatory power the knee buys.
    exponent A vs B vs C  the size of the slope bias.
    low-frequency bias    mean residual (observed - aperiodic fit) below 8 Hz.
                          Systematically POSITIVE under 'fixed' means the
                          straight line runs under the data at the low end, and
                          that excess is what gets fit as delta/theta peaks.
    peak counts           delta/theta peaks under each model. A large drop from
                          A to B means those peaks were aperiodic misfit.

    Secondary, free from the same fits: whether max_n_peaks binds. The
    production data truncates at exactly 8 peaks despite max_n_peaks=12, which
    is a hard cliff rather than a natural distribution; the per-fit peak counts
    recorded here show whether that is FOOOF's post-hoc overlap dropping or the
    cap itself.

Input:
    Preprocessed continuous FIFs under movies_prep_standard/{pat}/Neural_prep/
    Correspondence sheets, via src/channel_metadata.py

Output:
    aperiodic_mode_comparison_fits.csv   one row per (recording, channel,
                                         window) with all three fits side by
                                         side
    aperiodic_mode_comparison_peaks.csv  one row per detected peak, tagged by
                                         model
    plus a printed report

This script does NOT modify the pipeline, refit production data, or write
anything into the rolling_fooof_* trees. It samples, measures and reports.
"""

import os
import sys
import re
import glob
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

from paths import MOVIE_DATA, p                      # noqa: E402
from channel_metadata import load_channel_metadata    # noqa: E402

# =============================================================================
# CONFIG
# =============================================================================
VIDS = ['despicable_me_english', 'despicable_me_hungarian', 'inscapes']
REF = 'avg'

N_RECORDINGS_PER_VID = 2      # recordings sampled per video
N_CHANNELS = 50               # channels sampled per recording
N_WINDOWS = 40                # windows sampled per recording, evenly spread

# Welch settings: identical to extract_all_fooof.py, so the comparison is
# about the aperiodic model and nothing else.
WINDOW_LEN_SEC = 10.0
STEP_SEC = 2.5
N_FFT_SEC = 2.0
N_OVERLAP_SEC = 1.0

FIT_LO, FIT_HI = 1.0, 57.0    # production range
FIT_LO_TRUNC = 3.0            # config C: fit above the expected knee

PEAK_WIDTH_LIMITS = [1, 10]
MAX_N_PEAKS = 12
MIN_PEAK_HEIGHT = 0.1
PEAK_THRESHOLD = 2.0

LOWFREQ_EDGE = 8.0            # "low frequency" for the residual-bias readout

BANDS = {'delta': (1.0, 3.5), 'theta': (3.5, 7.5), 'alpha': (7.5, 13.5),
         'beta': (13.5, 30.5), 'gamma': (30.5, 57.0)}

OUT_DIR = p(MOVIE_DATA, 'fooof_diagnostics')
SEED = 0

PREP_DIR = p(MOVIE_DATA, 'movies_prep_standard')


# =============================================================================
# HELPERS
# =============================================================================
def _run_label(fname):
    m = re.search(r'run[-_]?(\d+)', fname, flags=re.IGNORECASE)
    return f'run-{int(m.group(1)):02d}' if m else 'run-01'


def _ses_label(fname):
    m = re.search(r'ses[-_]?(\d+)', fname, flags=re.IGNORECASE)
    return f'ses-{int(m.group(1)):02d}' if m else ''


def find_recordings(vid, n, prep_dir=PREP_DIR):
    """Return up to n (pat, fif_path) for this video, spread across patients."""
    hits = []
    for pat in sorted(os.listdir(prep_dir)):
        nd = os.path.join(prep_dir, pat, 'Neural_prep')
        if not os.path.isdir(nd):
            continue
        fs = [f for f in os.listdir(nd)
              if f.endswith('.fif') and REF in f and 'referenced' in f
              and 'aic' not in f and vid.lower() in f.lower()
              and not f.startswith('._')]
        if fs:
            hits.append((pat, os.path.join(nd, sorted(fs)[0])))
    # spread across the patient list rather than taking the first n
    if len(hits) <= n:
        return hits
    idx = np.linspace(0, len(hits) - 1, n).astype(int)
    return [hits[i] for i in idx]


def load_lfp(pat, fif_path):
    """Load, drop bads and flat channels, restrict to correspondence contacts."""
    raw = mne.io.read_raw(fif_path, preload=False)
    bads = [c for c in raw.info.get('bads', []) if c in raw.ch_names]
    if bads:
        raw.drop_channels(bads)
    labels = raw.ch_names

    try:
        meta = load_channel_metadata(pat)
        keep = set(meta['label'].astype(str))
        idx = np.array([lab in keep for lab in labels])
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f'    no usable correspondence sheet ({exc.__class__.__name__});'
              f' using all channels')
        idx = np.ones(len(labels), dtype=bool)

    lfp = raw.get_data()[idx, :]
    labels = [l for l, k in zip(labels, idx) if k]

    nonflat = np.std(lfp, axis=1) > 1e-6
    lfp = lfp[nonflat]
    labels = [l for l, k in zip(labels, nonflat) if k]
    return lfp, labels, float(raw.info['sfreq'])


def make_fm(mode, lo, hi):
    return FOOOF(peak_width_limits=PEAK_WIDTH_LIMITS,
                 max_n_peaks=MAX_N_PEAKS,
                 min_peak_height=MIN_PEAK_HEIGHT,
                 aperiodic_mode=mode,
                 peak_threshold=PEAK_THRESHOLD,
                 verbose=False)


def knee_freq(ap_params):
    """f_knee = knee ** (1/exponent), the bend location in Hz."""
    if len(ap_params) < 3:
        return np.nan
    _, knee, exponent = ap_params
    if knee <= 0 or exponent <= 0:
        return np.nan
    return float(knee ** (1.0 / exponent))


def fit_one(fm, freqs, psd, tag, low_edge=LOWFREQ_EDGE):
    """Fit and return (summary dict, list of peak dicts)."""
    fm.fit(freqs, psd)
    ap = fm.aperiodic_params_
    f = fm.freqs
    resid = fm.power_spectrum - fm._ap_fit          # flattened spectrum
    lo_mask = f < low_edge
    out = {
        f'{tag}_offset': float(ap[0]),
        f'{tag}_exponent': float(ap[-1]),
        f'{tag}_knee': float(ap[1]) if len(ap) > 2 else np.nan,
        f'{tag}_knee_freq': knee_freq(ap),
        f'{tag}_r2': float(fm.r_squared_),
        f'{tag}_error': float(fm.error_),
        f'{tag}_n_peaks': int(fm.peak_params_.shape[0]),
        f'{tag}_resid_lo_mean': float(np.mean(resid[lo_mask])),
        f'{tag}_resid_lo_max': float(np.max(resid[lo_mask])),
    }
    for name, (blo, bhi) in BANDS.items():
        m = (f >= blo) & (f < bhi)
        out[f'{tag}_periodic_{name}'] = float(np.mean(resid[m])) if m.any() else np.nan
    peaks = [{'model': tag, 'cf': float(pk[0]), 'amp': float(pk[1]),
              'bw': float(pk[2]), 'peak_index': i}
             for i, pk in enumerate(fm.peak_params_)]
    return out, peaks


# =============================================================================
# RUN
# =============================================================================
if __name__ == '__main__':

    rng = np.random.default_rng(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    fit_rows, peak_rows = [], []

    for vid in VIDS:
        recs = find_recordings(vid, N_RECORDINGS_PER_VID)
        print(f'\n{vid}: {len(recs)} recording(s)')
        for pat, fif_path in recs:
            fname = os.path.basename(fif_path)
            run = _run_label(fname)
            ses = _ses_label(fname)
            print(f'  {pat} {ses} {run}')

            try:
                lfp, labels, fs = load_lfp(pat, fif_path)
            except Exception as exc:
                print(f'    LOAD FAILED: {type(exc).__name__}: {exc}')
                continue

            n_ch, n_samp = lfp.shape
            win_samp = int(round(WINDOW_LEN_SEC * fs))
            max_start = n_samp - win_samp
            if max_start <= 0 or n_ch == 0:
                print('    too short / no channels, skipped')
                continue

            ch_idx = rng.choice(n_ch, size=min(N_CHANNELS, n_ch), replace=False)
            ch_idx.sort()
            starts = np.linspace(0, max_start, N_WINDOWS).astype(int)

            print(f'    {len(ch_idx)} channels x {len(starts)} windows '
                  f'@ {fs:.0f} Hz')

            for w, s0 in enumerate(starts):
                seg = lfp[ch_idx, s0:s0 + win_samp]
                try:
                    psd, freqs = psd_array_welch(
                        seg, sfreq=fs, fmin=FIT_LO, fmax=FIT_HI,
                        n_fft=int(fs * N_FFT_SEC),
                        n_overlap=int(fs * N_OVERLAP_SEC),
                        average='mean')
                except Exception as exc:
                    print(f'    PSD failed w{w}: {exc}')
                    continue

                for j, ci in enumerate(ch_idx):
                    base = dict(Patient=pat, Session=ses, Run=run, Video=vid,
                                Channel=labels[ci], Window_Index=w,
                                Window_Start_Sec=s0 / fs)
                    row = dict(base)
                    ok = True
                    for tag, mode, lo in (('fixed', 'fixed', FIT_LO),
                                          ('knee', 'knee', FIT_LO),
                                          ('trunc', 'fixed', FIT_LO_TRUNC)):
                        m = freqs >= lo
                        try:
                            summ, pks = fit_one(make_fm(mode, lo, FIT_HI),
                                                freqs[m], psd[j, m], tag)
                        except Exception:
                            ok = False
                            break
                        row.update(summ)
                        for pk in pks:
                            peak_rows.append({**base, **pk})
                    if ok:
                        fit_rows.append(row)

    fits = pd.DataFrame(fit_rows)
    peaks = pd.DataFrame(peak_rows)

    if fits.empty:
        print('\nno fits produced'); sys.exit(1)

    fits.to_csv(os.path.join(OUT_DIR, 'aperiodic_mode_comparison_fits.csv'),
                index=False, float_format='%.6g')
    peaks.to_csv(os.path.join(OUT_DIR, 'aperiodic_mode_comparison_peaks.csv'),
                 index=False, float_format='%.6g')
    print(f'\nwrote {len(fits):,} fits and {len(peaks):,} peaks to {OUT_DIR}')
