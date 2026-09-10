#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tier 1 -> Tier 2 for the re-extracted bandpass power (rescaled, epsilon-fixed).

Robustly z-scores each contact over the whole recording, then robustly averages
within overlapping 10 s windows. One output CSV per (recording x band).

    Tier 1  full_raw_log_power_rescale/{band}/{band}_power_log_{band}_{movie}/
            {pat}/{pat}_{movie}_{run}_{band}_cortical_power_log.csv
            4 atlas metadata rows + 179,710 samples @ 300 Hz (599 s)
       |
    Tier 2  windowed_power_10s_rescale/
            windowed_robustz_power_log_{movie}_{band}/{pat}/...csv
            4 atlas metadata rows + 236 windows

WHY ROBUST, NOT MEAN/SD

    Both reductions here use median-based estimators rather than mean/SD.

    Per-contact normalisation:  z = (x - median) / (1.4826 * MAD)
        The 1.4826 makes the scale match SD for Gaussian data, so the values
        stay on a familiar scale. MAD has a 50% breakdown point, so a handful
        of artifact samples cannot inflate the scale and flatten the whole
        contact's dynamic range - which is exactly the failure mode that
        matters here, because bad windows are INCLUDED in extraction and are
        filtered only downstream (see CLAUDE.md).

    Within-window reduction:  20% trimmed mean (WINDOW_STAT)
        A 10 s window is 3,000 samples at 300 Hz. An IED lasts ~100-200 ms,
        i.e. 1-2% of the window, and shifts a plain mean noticeably. Artifact
        windows are NOT excluded before this step (CLAUDE.md), so the window
        statistic has to tolerate them.

        Measured on one english recording per band, mean vs median per window:

            band    r(mean,med)  r(mean,trim20)  |diff|>0.25 SD  sd(mean)  sd(med)
            delta      0.925         0.969           17.9%         0.261    0.270
            theta      0.936         0.974            6.9%         0.296    0.306
            alpha      0.941         0.977            4.8%         0.324    0.340
            gamma      0.931         0.969            0.1%         0.174    0.174
            HFA        0.930         0.968            1.4%         0.223    0.219

        Mean and median agree at only r ~ 0.93, so this is a real choice, and
        it bites hardest in DELTA - the band the current claims rest on.

        Note sd(median) >= sd(mean) in the low bands. The median is therefore
        not simply stripping artifact variance; it tracks something with at
        least as much window-to-window variance. Within a 10 s delta window
        the 3,000 samples span only ~10-30 cycles and are heavily
        autocorrelated, so the median has none of its usual efficiency
        advantage there.

        The 20% trimmed mean is the default because it keeps the mean's
        interpretation and efficiency (r = 0.97) while having a 20% breakdown
        point - an order of magnitude more tolerance than an IED needs. The
        median discards far more information than the artifact problem
        requires.

    WINDOW_STAT accepts 'median', 'trim20' or 'mean'; 'mean' with
    ROBUST_Z = False reproduces `lowpass_power_to_windows.py`. It can be set
    per run without editing this file:  WINDOW_STAT=median python3 ...
    Output directory names carry the statistic, so variants never collide.

WINDOW GRID
    10 s window, 7.5 s overlap, 2.5 s step, at fs = 300 Hz (the power CSVs are
    decimated 600 -> 300 in extraction). 236 windows, matching the existing
    Tier 2 grid, so these join the attention labels without resampling.

METADATA ROWS
    Whatever the input carries is carried through unchanged (4 rows here:
    DK / Y7 / Y17 / AparcAseg). The old chain sometimes had a 5th `network`
    row inserted into Tier 1 by `define_custom_network_atlas.py`; it is absent
    from this extraction and is NOT invented here.

This script does NOT reshape to long format, merge eye/attention data, exclude
bad windows, or modify the old `lowpass_power_to_windows.py` chain, whose
outputs are preserved as validation references.

Parallel by recording, so it stays a plain top-level script:
    FOOOF_N_WORKERS=8 FOOOF_WORKER=0 python3 analysis_scripts/window_bandpass_power_robust.py
"""

import os
import re
import sys
import time
import glob
import numpy as np
import pandas as pd
from scipy.stats import trim_mean

_src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
if _src not in sys.path:
    sys.path.insert(0, _src)

# =============================================================================
# PARAMETERS
# =============================================================================
IN_ROOT = '/media/christine/Data/Movie_data/full_raw_log_power_rescale'
OUT_ROOT = '/media/christine/Samsung/Movie_data/windowed_power_10s_rescale'

fs_lfp = 300                     # power CSVs are decimated 600 -> 300
WINDOW_SEC = 10
OVERLAP_SEC = 7.5
STEP_SEC = WINDOW_SEC - OVERLAP_SEC
WINDOW_SAMPLES = int(WINDOW_SEC * fs_lfp)     # 3000
STEP_SAMPLES = int(STEP_SEC * fs_lfp)         # 750

ROBUST_Z = True                  # median/MAD per contact (False -> mean/SD)
WINDOW_STAT = 'trim20'           # 'median' | 'trim20' | 'mean'
TRIM_PROPORTION = 0.20           # tail fraction cut from EACH end for 'trim20'
MAD_SCALE = 1.4826               # makes MAD comparable to SD for Gaussian data

METHOD = 'power_log'
OVERWRITE = False
MIN_MTIME_AGE_SEC = 120          # skip inputs written within this many seconds:
                                 # they may still be being extracted right now
EXPECTED_MIN_SAMPLES = 100_000   # sanity floor; short files are reported

WINDOW_STAT = os.environ.get('WINDOW_STAT', WINDOW_STAT)

WORKER = int(os.environ.get('FOOOF_WORKER', 0))
N_WORKERS = int(os.environ.get('FOOOF_N_WORKERS', 1))


# =============================================================================
# HELPERS
# =============================================================================
def extract_pat_id(name):
    m = re.match(r'((?:NS|LH)\d+(?:_\d+)?)', name)
    return m.group(1) if m else None


def extract_run_label(fname):
    m = re.search(r'run[-_]?(\d+)', fname, flags=re.IGNORECASE)
    return f'run-{int(m.group(1)):02d}' if m else 'run-01'


def find_inputs(in_root=IN_ROOT):
    """
    Every Tier 1 power CSV on disk, as (band, movie, pat, run, path).

    Derived from the filesystem rather than a hardcoded entry list, because
    this is a complete pass over everything that has been extracted.
    """
    out = []
    for path in sorted(glob.glob(os.path.join(
            in_root, '*', '*_power_log_*', '*', '*power_log.csv'))):
        if os.path.basename(path).startswith('._'):
            continue
        parts = path.split(os.sep)
        band = parts[-4]
        movie_dir = parts[-3]
        pat = parts[-2]
        m = re.match(rf'{re.escape(band)}_power_log_{re.escape(band)}_(.+)$',
                     movie_dir)
        if not m:
            continue
        out.append((band, m.group(1), pat,
                    extract_run_label(os.path.basename(path)), path))
    return out


def robust_z(a, axis=1, scale=MAD_SCALE):
    """(x - median) / (scale * MAD), per row. Zero-scale rows pass through."""
    med = np.nanmedian(a, axis=axis, keepdims=True)
    mad = np.nanmedian(np.abs(a - med), axis=axis, keepdims=True)
    sd = scale * mad
    sd = np.where(sd <= 0, np.nan, sd)
    return (a - med) / sd, med.squeeze(axis), sd.squeeze(axis)


def window_reduce(pow_dat, window_samples=WINDOW_SAMPLES,
                  step_samples=STEP_SAMPLES, stat=None):
    """
    Slide over (n_elec, n_samples); reduce each window to one value per
    electrode. Uses a strided view so no data is copied.
    """
    n_elec, n_samp = pow_dat.shape
    n_win = (n_samp - window_samples) // step_samples + 1
    if n_win <= 0:
        return np.empty((n_elec, 0)), np.empty(0)
    s_elec, s_samp = pow_dat.strides
    view = np.lib.stride_tricks.as_strided(
        pow_dat, shape=(n_elec, n_win, window_samples),
        strides=(s_elec, s_samp * step_samples, s_samp), writeable=False)
    stat = stat or WINDOW_STAT
    if stat == 'median':
        red = np.nanmedian(view, axis=2)
    elif stat == 'mean':
        red = np.nanmean(view, axis=2)
    elif stat == 'trim20':
        red = trim_mean(view, TRIM_PROPORTION, axis=2)
    else:
        raise ValueError(f'unknown WINDOW_STAT {stat!r}')
    starts = np.arange(n_win) * step_samples
    centers = (starts + window_samples / 2.0) / fs_lfp
    return red, centers


def process_one(band, movie, pat, run, path, out_root=OUT_ROOT):
    """Window one Tier 1 file. Returns a status string."""
    z_tag = 'robustz' if ROBUST_Z else 'z'
    red_tag = WINDOW_STAT
    out_dir = os.path.join(
        out_root, f'windowed_{z_tag}_{METHOD}_{movie}_{band}', pat)
    out_name = (f'{pat}_{run}_{movie}_{band}_{METHOD}_{z_tag}_'
                f'rolling_{red_tag}_{WINDOW_SEC}s.csv')
    out_path = os.path.join(out_dir, out_name)
    if not OVERWRITE and os.path.exists(out_path):
        return 'exists'

    age = time.time() - os.path.getmtime(path)
    if age < MIN_MTIME_AGE_SEC:
        return f'SKIP too fresh ({age:.0f}s old, may still be writing)'

    raw = pd.read_csv(path, low_memory=False)
    atlas_rows = raw[raw['Atlas'].notna()]
    n_meta = len(atlas_rows)
    data_rows = raw.iloc[n_meta:]

    elec_cols = [c for c in raw.columns if c not in ('SubID', 'Atlas')]
    if not elec_cols:
        return 'SKIP no electrode columns'

    vals = data_rows[elec_cols].apply(pd.to_numeric, errors='coerce').to_numpy(
        dtype=np.float64).T                     # (n_elec, n_samples)
    n_samp = vals.shape[1]
    short = n_samp < EXPECTED_MIN_SAMPLES

    if ROBUST_Z:
        z, _, sd = robust_z(vals, axis=1)
        n_flat = int(np.isnan(sd).sum())
    else:
        mu = np.nanmean(vals, axis=1, keepdims=True)
        sd_ = np.nanstd(vals, axis=1, keepdims=True)
        sd_ = np.where(sd_ <= 0, np.nan, sd_)
        z = (vals - mu) / sd_
        n_flat = int(np.isnan(sd_).sum())

    red, centers = window_reduce(z)
    if red.shape[1] == 0:
        return 'SKIP too short for one window'

    df = pd.DataFrame(red.T, columns=elec_cols,
                      index=[f'{t:.3f}s' for t in centers])
    df.insert(0, 'SubID', pat)
    df.insert(0, 'Atlas', np.nan)
    df = df.reindex(columns=raw.columns)
    out = pd.concat([atlas_rows, df], ignore_index=True)

    os.makedirs(out_dir, exist_ok=True)
    tmp = out_path + '.tmp'
    out.to_csv(tmp)
    os.replace(tmp, out_path)                   # atomic: no partial file

    note = ''
    if n_flat:
        note += f' [{n_flat} flat contact(s) -> NaN]'
    if short:
        note += f' [SHORT: {n_samp:,} samples]'
    return (f'{red.shape[0]} elec x {red.shape[1]} win, '
            f'{n_meta} meta rows{note}')


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':

    items = find_inputs()
    if N_WORKERS > 1:
        items = [x for i, x in enumerate(items) if i % N_WORKERS == WORKER]
    tag = f'[w{WORKER}/{N_WORKERS}] ' if N_WORKERS > 1 else ''

    print(f'{tag}{len(items)} Tier 1 file(s)  |  '
          f'z={"median/MAD" if ROBUST_Z else "mean/SD"}, '
          f'window={WINDOW_STAT}, '
          f'{WINDOW_SEC}s/{STEP_SEC}s @ {fs_lfp} Hz', flush=True)
    print(f'{tag}-> {OUT_ROOT}\n', flush=True)

    t0 = time.time()
    n_ok = n_skip = n_err = 0
    for band, movie, pat, run, path in items:
        label = f'{pat} {run} {movie} {band}'
        try:
            t1 = time.time()
            status = process_one(band, movie, pat, run, path)
        except Exception as exc:
            n_err += 1
            print(f'{tag}{label}  ERROR {type(exc).__name__}: {exc}', flush=True)
            continue
        if status == 'exists':
            n_skip += 1
            continue
        if status.startswith('SKIP'):
            n_skip += 1
            print(f'{tag}{label}  {status}', flush=True)
            continue
        n_ok += 1
        print(f'{tag}{label}  {status}  ({time.time()-t1:.0f}s)', flush=True)

    print(f'\n{tag}done: {n_ok} written, {n_skip} skipped, {n_err} errors '
          f'in {(time.time()-t0)/60:.1f} min', flush=True)
