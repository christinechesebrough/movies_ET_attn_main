#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LFP Power Condensing Script
Loops over frequency bands, movies, and recordings.
For each recording: loads power CSV, normalizes per electrode,
applies a rolling average (10s window, 7.5s overlap @ 600 Hz),
and saves a condensed CSV (electrodes × windows).
"""

import os
import re
import numpy as np
import pandas as pd

# =============================================================================
# PARAMETERS
# =============================================================================

fs_lfp            = 600          # Hz — samples per second in the power data
normalize         = False        # z-score each electrode before windowing
machine_path      = 'media/christine'
method            = 'power_log'

# Rolling window settings
WINDOW_SEC        = 10         # window length in seconds
OVERLAP_SEC       = 7.5          # overlap in seconds
STEP_SEC          = WINDOW_SEC - OVERLAP_SEC   # 2.5 s

WINDOW_SAMPLES    = int(WINDOW_SEC  * fs_lfp)  # 6 000
STEP_SAMPLES      = int(STEP_SEC    * fs_lfp)  # 1 500

freq_bands = ['HFA','delta','beta','gamma','alpha','theta']
movies     = ['despicable_me_english']#inscapes

# =============================================================================
# ENTRY-ID TABLES  (add/remove entries here)
# =============================================================================

MOVIE_ENTRIES = {
    'despicable_me_english': [
        'NS127_02_ses-02_run-01',
        'NS135_ses-01_run-01',
        'NS136_ses-01_run-01',
        'NS137_ses-01_run-01',
        'NS138_ses-01_run-01',
        'NS140_ses-01_run-01',
        'NS153_ses-01_run-01',
        'NS155_02_ses-02_run-01',
        'NS164_ses-01_run-01',
        'NS174_02_ses-02_run-01',
        'NS174_03_ses-03_run-01',
        'NS178_ses-01_run-01',
        'NS190_ses-01_run-01',
        'NS190_ses-01_run-02',
        'NS191_ses-01_run-01',
        'NS193_ses-01_run-01',
        'NS193_ses-01_run-02',
        'NS194_ses-01_run-01',
        'NS205_ses-01_run-01',
    ],
    'inscapes': [
        'NS127_02_ses-02_run-01',
        'NS135_ses-01_run-01',
        'NS136_ses-01_run-01',
        'NS137_ses-01_run-01',
        'NS138_ses-01_run-01',
        'NS140_ses-01_run-01',
        'NS140_02_ses-02_run-01',
        'NS151_ses-01_run-01',
        'NS153_ses-01_run-01',
        'NS155_ses-01_run-01',
        'NS155_02_ses-02_run-01',
        'NS164_ses-01_run-01',
        'NS178_ses-01_run-01',
        'NS205_ses-01_run-01',
        'NS210_ses-01_run-01',
    ],
      'despicable_me_hungarian':[
        'LH010_ses-01_run-01',
        # 'NS127_02_ses-02_run-01',
        # 'NS135_ses-01_run-01',
        # 'NS136_ses-01_run-01', 
        # 'NS137_ses-01_run-01',
        # 'NS138_ses-01_run-01', 
        # 'NS140_ses-01_run-01',
        # 'NS140_02_ses-02_run-01',
        # 'NS145_ses-02_run-01', 
        # 'NS154_ses-01_run-01',
        # 'NS164_ses-01_run-01', 
        # 'NS174_02_ses-02_run-01',
        # 'NS174_03_ses-03_run-01',
        # 'NS178_ses-01_run-01'
    ]
}


# =============================================================================
# HELPERS
# =============================================================================

def extract_pat_id(entry: str) -> str:
    """Extract patient ID like 'NS127', 'NS127_02', 'LH010', or 'LH010_02' from an entry string."""
    m = re.match(r'((?:NS|LH)\d+(?:_\d+)?)', entry)
    if not m:
        raise ValueError(f"Could not extract patient ID from: {entry}")
    return m.group(1)


def extract_run_label(fname: str) -> str:
    """Extract run label like 'run-01' from a filename; fallback to 'run-01'."""
    m = re.search(r'run[-_]?(\d+)', fname, flags=re.IGNORECASE)
    return f"run-{int(m.group(1)):02d}" if m else "run-01"


def pick_file(files, *, contains_any=None, contains_all=None,
              endswith=None, startswith_not=None):
    """Filter a list of filenames by multiple criteria."""
    out = files
    if startswith_not is not None:
        out = [f for f in out if not f.startswith(startswith_not)]
    if endswith is not None:
        out = [f for f in out if f.endswith(endswith)]
    if contains_any is not None:
        out = [f for f in out if any(k in f for k in contains_any)]
    if contains_all is not None:
        out = [f for f in out if all(k in f for k in contains_all)]
    return out


def rolling_average_windows(pow_dat: np.ndarray,
                             window_samples: int,
                             step_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Slide a window across pow_dat (n_electrodes × n_samples) and compute
    the mean of each electrode within each window.

    Returns
    -------
    condensed : np.ndarray, shape (n_electrodes, n_windows)
        Mean power per electrode per window.
    window_centers_sec : np.ndarray, shape (n_windows,)
        Center time (in seconds) of each window.
    """
    n_electrodes, n_samples = pow_dat.shape
    starts = range(0, n_samples - window_samples + 1, step_samples)

    means   = []
    centers = []
    for start in starts:
        end = start + window_samples
        means.append(pow_dat[:, start:end].mean(axis=1))   # (n_electrodes,)
        centers.append((start + end) / 2.0 / fs_lfp)       # seconds

    condensed          = np.column_stack(means)   # (n_electrodes, n_windows)
    window_centers_sec = np.array(centers)
    return condensed, window_centers_sec

#%%
# =============================================================================
# MAIN LOOP
# =============================================================================

for freq_band in freq_bands:

    freq_range_map = {
        'HFA':         '62 - 150 Hz',
        'alpha':       '8 - 13 Hz',
        'gamma':   '31- 59 Hz',
        'theta': '4 - 7 Hz',
        'beta': '13 - 30 Hz',
        'delta': '1-3 Hz'
    }
    freq_range = freq_range_map.get(freq_band, 'unknown Hz')

    for movie in movies:

        entry_ids = MOVIE_ENTRIES.get(movie, [])
        if not entry_ids:
            print(f"No entries defined for movie '{movie}' — skipping.")
            continue

        data_dir = (f'/{machine_path}/Samsung/Movie_data/full_raw_log_power_1Apr26/{method}_{freq_band}_{movie}_26Mar26')

        for rec in entry_ids:
            pat = extract_pat_id(rec)
            run = extract_run_label(rec)

            pat_dir = os.path.join(data_dir, pat)
            if not os.path.isdir(pat_dir):
                print(f"[SKIP] Directory not found: {pat_dir}")
                continue

            # ------------------------------------------------------------------
            # Find the power CSV for this recording
            # ------------------------------------------------------------------
            all_csvs = [
                f for f in os.listdir(pat_dir)
                if method in f and f.endswith('.csv') and not f.startswith('._')
            ]

            if not all_csvs:
                print(f"[SKIP] No CSV files found for {pat} in {pat_dir}")
                continue

            # Patients with multiple runs need run-specific file selection
            if pat in ('NS190', 'NS193'):
                candidates = pick_file(all_csvs, contains_any=[run])
                if not candidates:
                    print(f"[SKIP] No CSV matching {run} for {pat}")
                    continue
                lfp_file = candidates[0]
            else:
                lfp_file = all_csvs[0]

            full_lfp_path = os.path.join(pat_dir, lfp_file)

            # ------------------------------------------------------------------
            # Load CSV  (first 4 rows = atlas metadata, rest = data)
            # ------------------------------------------------------------------
            try:
                raw = pd.read_csv(full_lfp_path)
                print(f"Loaded: {full_lfp_path}  |  shape {raw.shape}")
            except Exception as e:
                print(f"[ERROR] Reading {full_lfp_path}: {e}")
                continue

            atlas_rows = raw.head(4)                          # kept for reference
            data_rows  = raw.iloc[4:].reset_index(drop=True)

            # ------------------------------------------------------------------
            # Identify electrode columns  (start with L or R, rest alphanumeric)
            # ------------------------------------------------------------------
            electrode_cols = [
                col for col in raw.columns
                if col and col[0] in ('L', 'R','A','P') and col[1:].isalnum() and col not in ['Atlas']
            ]

            if not electrode_cols:
                print(f"[SKIP] No electrode columns found for {pat}")
                continue

            # ------------------------------------------------------------------
            # Convert to numeric and optionally z-score per electrode
            # ------------------------------------------------------------------
            data_rows[electrode_cols] = data_rows[electrode_cols].apply(
                pd.to_numeric, errors='coerce'
            )

            if normalize:
                mu  = data_rows[electrode_cols].mean()
                sig = data_rows[electrode_cols].std().replace(0, np.nan)
                data_rows[electrode_cols] = (data_rows[electrode_cols] - mu) / sig

            # pow_dat: (n_electrodes × n_samples) — each row is one electrode
            pow_dat = data_rows[electrode_cols].to_numpy().T   # shape: (n_elec, n_samples)
            pow_dat = np.nan_to_num(pow_dat, nan=0.0)

            print(f"  pow_dat shape: {pow_dat.shape}  "
                  f"({pow_dat.shape[0]} electrodes × {pow_dat.shape[1]} samples)")

            # ------------------------------------------------------------------
            # Rolling average condensing
            # ------------------------------------------------------------------
            condensed, window_centers = rolling_average_windows(
                pow_dat, WINDOW_SAMPLES, STEP_SAMPLES
            )
            print(f"  condensed shape: {condensed.shape}  "
                  f"({condensed.shape[0]} electrodes × {condensed.shape[1]} windows)")

            col_labels = [f"{t:.3f}s" for t in window_centers]
            out_df = pd.DataFrame(
                condensed,
                index=electrode_cols,
                columns=col_labels
            )
            out_df.index.name = 'electrode'
            
            df = out_df.T

            # Add SubID as first column
            df.insert(0, 'SubID', pat)

            # Add the columns atlas_rows expects
            df.insert(0, 'Atlas', np.nan)
            
            atlas_cols = list(atlas_rows.columns)
            df_cols = list(df.columns)
            
            # Remove non-electrode columns for comparison
            meta_cols = ['SubID', 'Atlas']
            atlas_elec = [c for c in atlas_cols if c not in meta_cols]
            df_elec = [c for c in df_cols if c not in meta_cols]
            
            missing_in_df = set(atlas_elec) - set(df_elec)
            extra_in_df   = set(df_elec) - set(atlas_elec)
            
            if missing_in_df:
                raise ValueError(f"[ALIGNMENT ERROR] Missing electrodes in df: {sorted(missing_in_df)}")
            
            if extra_in_df:
                raise ValueError(f"[ALIGNMENT ERROR] Extra electrodes in df: {sorted(extra_in_df)}")
            
            # Optional: enforce exact order match (strict)
            if atlas_elec != df_elec:
                print("[WARNING] Electrode order mismatch — reordering df to match atlas_rows")
                        
            
            # Reorder df to match atlas_rows exactly
            df = df.reindex(columns=atlas_rows.columns)
            
            # Now concatenate
            df_w_atlas = pd.concat([atlas_rows, df], ignore_index=True)
            
            if normalize == True:
                z_val = 'z'
                normed_val = 'normed'
            else:
                z_val = ''
                normed_val = 'unnormed'
                
            # Save
            out_fname = f"{pat}_{run}_{movie}_{freq_band}_{method}_{z_val}_rolling_avg_{WINDOW_SEC}s.csv"
            # out_dir = (f'/{machine_path}/Samsung/Movie_data/'
            #           f'windowed_{z_val}_{method}_{movie}_{freq_band}_1Apr26')
            out_dir = f'/{machine_path}/Samsung/Movie_data/windowed_power_{WINDOW_SEC}s/windowed_{normed_val}_{method}_{movie}_{freq_band}_1Apr26'

            pat_out_dir = os.path.join(out_dir,pat)
            if not os.path.exists(pat_out_dir):
                os.makedirs(pat_out_dir)
                
            out_path  = os.path.join(pat_out_dir,out_fname)
            df_w_atlas.to_csv(out_path)
            print(f"Saved: {out_path}")



print("Done.")