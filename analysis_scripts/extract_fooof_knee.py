#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rolling-window FOOOF refit with a KNEE aperiodic model (TODO C0).

Replaces the aperiodic model used by `extract_all_fooof.py`. That script is
left untouched: its June 2026 outputs stay as a reference for comparison.

WHY A REFIT IS NEEDED  (measured, see CLAUDE.md)

    `extract_all_fooof.py` uses aperiodic_mode='fixed' over 1-57 Hz. Across
    10,000 fits these spectra have a spectral knee INSIDE that range in 95.1%
    of windows, median 5.19 Hz. Fitting one straight line through that bend:

      - biases the exponent, and the bias tracks knee position at r = -0.833,
        so it is a CONFOUND in the attention-state contrast, not an offset.
        corr(fixed exponent, knee exponent) is only 0.46.
      - leaves a systematically positive residual below 8 Hz (98.3% of fits),
        which suppresses delta and inflates theta/alpha. Delta periodic power
        flips sign (-0.029 fixed -> +0.196 knee); delta peaks rise 73.7%.
      - wastes peak budget on the bend, pushing 30.1% of fits into the
        max_n_peaks cap, which then discards real peaks.

    R^2 does NOT reveal any of this (0.9722 fixed vs 0.9742 knee) because
    nearly all spectral variance is in the overall slope. Model choice here is
    made on the knee estimate and the residual, not on fit quality.

TWO PASSES, BECAUSE ONE WINDOW LENGTH CANNOT SERVE THE WHOLE SPECTRUM

    'main'  10 s windows, fit 1-57 Hz
            Matches the existing 10 s / 2.5 s window grid used by
            `lowpass_power_to_windows.py` and the Tier 2 CSVs, so results merge
            onto the attention labels without resampling.

    'low'   30 s windows, fit 0.5-30 Hz
            Frequency resolution at 10 s (Welch n_fft = 2 s -> 0.5 Hz) leaves
            only ~5 usable bins below 3.5 Hz. A Gaussian has three free
            parameters, so delta cannot be estimated there whatever the
            aperiodic model. A 30 s window supports n_fft = 8 s (0.125 Hz),
            giving ~25 bins below 3.5 Hz.

            This is not a free improvement - it is the frequency/time tradeoff
            being spent differently. Delta genuinely resolves on a ~15-20 s
            timescale (see the wavelet analysis in CLAUDE.md), so the coarser
            temporal resolution does not discard information that was there.

            Low-pass windows are CENTRED on the same window centres as 'main'
            (centre +/- 15 s), so the two passes join on Window_Index. Windows
            whose 30 s span runs off either end of the recording are skipped,
            which costs the first and last ~5 windows.

OUTPUTS  (per recording, mirroring the old schema so downstream reshapers work)

    *_fooof_aperiodic_windows_{pass}.csv   one row per channel x window
        Aperiodic_Offset, Aperiodic_Knee, Aperiodic_Exponent, Knee_Freq_Hz
        FOOOF_R2, FOOOF_Error, N_Peaks_Total, Hit_Peak_Cap
        Periodic_{band}   <- band-limited periodic power, see below
    *_fooof_peaks_long_{pass}.csv          one row per detected peak

    Periodic_{band} is the mean of the flattened spectrum (observed minus the
    fitted aperiodic component) over that band. It needs NO detection decision
    and is defined in every window, unlike peak presence or count, so it is the
    measure to use for "is this band-power effect oscillatory or aperiodic".
    Peak columns remain for characterising WHICH oscillation, not how much.

    Knee_Freq_Hz = knee ** (1/exponent), the bend location in Hz. Carried as a
    measure in its own right: it is interpretable as a characteristic timescale
    and may itself vary with state.

This script does NOT window, z-score, merge attention labels, or write into the
rolling_fooof_*_26Jun26 trees. It emits per-recording CSVs only.
"""

import os
import sys
import re
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
# CONFIG  — edit these, run the cells below in Spyder
# =============================================================================
PASS = 'main'                 # 'main' or 'low'

vids = ['despicable_me_english', 'despicable_me_hungarian', 'inscapes']
REF = 'avg'
OVERWRITE = False             # skip recordings whose output already exists

# --- pass presets ------------------------------------------------------------
PASSES = {
    'main': dict(
        window_sec=10.0,
        n_fft_sec=2.0,        # 0.5 Hz resolution, ~9 Welch segments
        fit_lo=1.0, fit_hi=57.0,
        peak_width_limits=[1, 10],
        max_n_peaks=20,       # SWEEP_MAX_N_PEAKS
    ),
    'low': dict(
        window_sec=30.0,
        n_fft_sec=8.0,        # 0.125 Hz resolution, ~6 Welch segments
        fit_lo=0.5, fit_hi=30.0,
        peak_width_limits=[0.5, 8],
        max_n_peaks=12,
    ),
}

APERIODIC_MODE = 'knee'
MIN_PEAK_HEIGHT = 0.1
PEAK_THRESHOLD = 2.0

STEP_SEC = 2.5                # window grid step, matches Tier 2
BANDS = {'delta': (1.0, 3.5), 'theta': (3.5, 7.5), 'alpha': (7.5, 13.5),
         'beta': (13.5, 30.5), 'gamma': (30.5, 57.0)}

PREP_DIR = p(MOVIE_DATA, 'movies_prep_standard')
OUT_ROOT = p(MOVIE_DATA, 'rolling_fooof_knee')


# =============================================================================
# HELPERS
# =============================================================================
def _run_label(f):
    m = re.search(r'run[-_]?(\d+)', f, flags=re.IGNORECASE)
    return f'run-{int(m.group(1)):02d}' if m else 'run-01'


def _ses_label(f):
    m = re.search(r'ses[-_]?(\d+)', f, flags=re.IGNORECASE)
    return f'ses-{int(m.group(1)):02d}' if m else ''


def list_recordings(vid, prep_dir=PREP_DIR, ref=REF):
    """Every (pat, fif_path) for this video. Trim the result to subset."""
    out = []
    for pat in sorted(os.listdir(prep_dir)):
        nd = os.path.join(prep_dir, pat, 'Neural_prep')
        if not os.path.isdir(nd):
            continue
        for f in sorted(os.listdir(nd)):
            if (f.endswith('.fif') and ref in f and 'referenced' in f
                    and 'aic' not in f and vid.lower() in f.lower()
                    and not f.startswith('._')):
                out.append((pat, os.path.join(nd, f)))
    return out


def load_lfp(pat, fif_path):
    """Drop bads and flat channels; restrict to correspondence contacts."""
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
        print(f'      no usable correspondence sheet '
              f'({type(exc).__name__}); keeping all channels')
        idx = np.ones(len(labels), dtype=bool)
    lfp = raw.get_data()[idx, :]
    labels = [l for l, k in zip(labels, idx) if k]
    nonflat = np.std(lfp, axis=1) > 1e-6
    if (~nonflat).any():
        print(f'      dropping {int((~nonflat).sum())} flat channel(s)')
    lfp = lfp[nonflat]
    labels = [l for l, k in zip(labels, nonflat) if k]
    return lfp, labels, float(raw.info['sfreq']), meta


def knee_freq(ap):
    """f_knee = knee ** (1/exponent). NaN when the fit is degenerate."""
    if len(ap) < 3:
        return np.nan
    _, knee, exponent = ap
    if knee <= 0 or exponent <= 0:
        return np.nan
    return float(knee ** (1.0 / exponent))


def window_bounds(cfg, n_samp, fs, step_sec=STEP_SEC):
    """
    (win_index, start_sample, end_sample) on the shared 10 s / 2.5 s grid.

    Longer passes are CENTRED on the same window centres as the main pass, so
    every pass shares one Window_Index and the tables join without resampling.
    Windows whose span falls outside the recording are dropped, which for the
    30 s pass costs roughly the first and last 5 indices.
    """
    base_win = PASSES['main']['window_sec']
    half = cfg['window_sec'] / 2.0
    n_grid = int(np.floor((n_samp / fs - base_win) / step_sec)) + 1
    out = []
    for i in range(max(n_grid, 0)):
        centre = i * step_sec + base_win / 2.0
        s = int(round((centre - half) * fs))
        e = s + int(round(cfg['window_sec'] * fs))
        if s >= 0 and e <= n_samp:
            out.append((i, s, e))
    return out


def fit_window(psd_ch, freqs, cfg, bands=BANDS):
    """Fit one channel-window. Returns (summary dict, list of peak dicts)."""
    fm = FOOOF(peak_width_limits=cfg['peak_width_limits'],
               max_n_peaks=cfg['max_n_peaks'],
               min_peak_height=MIN_PEAK_HEIGHT,
               aperiodic_mode=APERIODIC_MODE,
               peak_threshold=PEAK_THRESHOLD,
               verbose=False)
    fm.fit(freqs, psd_ch)
    ap = fm.aperiodic_params_
    f = fm.freqs
    flat = fm.power_spectrum - fm._ap_fit          # flattened spectrum
    n_pk = int(fm.peak_params_.shape[0])

    summ = {
        'Aperiodic_Offset': float(ap[0]),
        'Aperiodic_Knee': float(ap[1]) if len(ap) > 2 else np.nan,
        'Aperiodic_Exponent': float(ap[-1]),
        'Knee_Freq_Hz': knee_freq(ap),
        'FOOOF_R2': float(fm.r_squared_),
        'FOOOF_Error': float(fm.error_),
        'N_Peaks_Total': n_pk,
        'Hit_Peak_Cap': bool(n_pk >= cfg['max_n_peaks']),
    }
    for name, (lo, hi) in bands.items():
        m = (f >= lo) & (f < hi)
        summ[f'Periodic_{name}'] = float(np.mean(flat[m])) if m.any() else np.nan

    peaks = [{'Peak_Index_In_Window': i, 'Peak_CF_Hz': float(pk[0]),
              'Peak_Amplitude': float(pk[1]), 'Peak_Bandwidth_Hz': float(pk[2])}
             for i, pk in enumerate(fm.peak_params_)]
    return summ, peaks


# =============================================================================
# MAIN  — run as a cell in Spyder, or `python3 analysis_scripts/extract_fooof_knee.py`
# =============================================================================
if __name__ == '__main__':

    cfg = PASSES[PASS]
    out_dir = os.path.join(OUT_ROOT, PASS)
    os.makedirs(out_dir, exist_ok=True)

    print(f'PASS {PASS!r}: {cfg["window_sec"]:.0f} s windows, '
          f'Welch n_fft {cfg["n_fft_sec"]:.0f} s, '
          f'fit {cfg["fit_lo"]}-{cfg["fit_hi"]} Hz, '
          f'{APERIODIC_MODE} aperiodic, max_n_peaks {cfg["max_n_peaks"]}')
    print(f'-> {out_dir}\n')

    for vid in vids:
        recs = list_recordings(vid)
        print(f'{vid}: {len(recs)} recording(s)')

        for pat, fif_path in recs:
            fname = os.path.basename(fif_path)
            ses, run = _ses_label(fname), _run_label(fname)
            entry_id = f'{pat}_{ses}_{run}' if ses else f'{pat}_{run}'

            pat_dir = os.path.join(out_dir, pat)
            os.makedirs(pat_dir, exist_ok=True)
            win_csv = os.path.join(
                pat_dir, f'{entry_id}_{vid}_fooof_aperiodic_windows_{PASS}.csv')
            pk_csv = os.path.join(
                pat_dir, f'{entry_id}_{vid}_fooof_peaks_long_{PASS}.csv')
            if not OVERWRITE and os.path.exists(win_csv) and os.path.exists(pk_csv):
                print(f'  {entry_id}  exists, skipped')
                continue

            print(f'  {entry_id}')
            try:
                lfp, labels, fs, meta = load_lfp(pat, fif_path)
            except Exception as exc:
                print(f'      LOAD FAILED: {type(exc).__name__}: {exc}')
                continue
            if lfp.size == 0:
                print('      no channels, skipped')
                continue

            region = {}
            if meta is not None:
                region = dict(zip(meta['label'].astype(str), meta['Y17_Atlas']))

            bounds = window_bounds(cfg, lfp.shape[1], fs)
            n_fft = int(round(fs * cfg['n_fft_sec']))
            print(f'      {len(labels)} channels x {len(bounds)} windows '
                  f'@ {fs:.0f} Hz')
            if not bounds:
                print('      recording shorter than one window, skipped')
                continue

            win_rows, pk_rows = [], []
            for win_idx, s0, s1 in bounds:
                seg = lfp[:, s0:s1]
                try:
                    psd, freqs = psd_array_welch(
                        seg, sfreq=fs, fmin=cfg['fit_lo'], fmax=cfg['fit_hi'],
                        n_fft=n_fft, n_overlap=n_fft // 2, average='mean')
                except Exception as exc:
                    print(f'      PSD failed, window {win_idx}: {exc}')
                    continue

                base = dict(Patient=pat, Session=ses, Run=run, Video=vid,
                            Entry_ID=entry_id, Pass=PASS,
                            Window_Index=win_idx,
                            Window_Start_Sec=s0 / fs, Window_End_Sec=s1 / fs,
                            Window_Center_Sec=(s0 + s1) / (2 * fs),
                            Start_Sample=s0, End_Sample=s1,
                            FOOOF_Fit_Range_Low=cfg['fit_lo'],
                            FOOOF_Fit_Range_High=cfg['fit_hi'])

                for ci, label in enumerate(labels):
                    row = dict(base, Channel=label,
                               Region=region.get(label, None))
                    try:
                        summ, peaks = fit_window(psd[ci], freqs, cfg)
                    except Exception:
                        continue
                    win_rows.append({**row, **summ})
                    for pk in peaks:
                        pk_rows.append({**row, **pk,
                                        'Aperiodic_Exponent': summ['Aperiodic_Exponent'],
                                        'Knee_Freq_Hz': summ['Knee_Freq_Hz'],
                                        'FOOOF_R2': summ['FOOOF_R2']})

            if not win_rows:
                print('      no fits produced, nothing written')
                continue

            wdf = pd.DataFrame(win_rows)
            pd.DataFrame(pk_rows).to_csv(pk_csv, index=False, float_format='%.6g')
            wdf.to_csv(win_csv, index=False, float_format='%.6g')
            print(f'      {len(wdf):,} fits, {len(pk_rows):,} peaks   '
                  f'R2 {wdf.FOOOF_R2.mean():.4f}  '
                  f'f_knee {wdf.Knee_Freq_Hz.median():.2f} Hz  '
                  f'at cap {100*wdf.Hit_Peak_Cap.mean():.1f}%')

    print('\ndone')
