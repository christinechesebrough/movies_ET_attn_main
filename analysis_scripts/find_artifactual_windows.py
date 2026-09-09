#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 03:23:00 2026

@author: christinechesebrough
"""
import os,re,sys
import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation
import matplotlib.pyplot as plt


def extract_pat_id(entry: str) -> str:
    """
    Extract patient ID like 'NS127' or 'NS127_02' from an entry string.
    """
    m = re.match(r'(NS\d+(?:_\d+)?)', entry)
    if not m:
        raise ValueError(f"Could not extract patient ID from: {entry}")
    return m.group(1)


def extract_run_label(fname: str) -> str:
    """
    Try to extract a run label like 'run-01' or 'run-1' from a filename.
    Fallback: 'run-01'.
    """
    m = re.search(r'run[-_]?(\d+)', fname, flags=re.IGNORECASE)
    if m:
        return f"run-{int(m.group(1)):02d}"
    return "run-01"

def pick_file(files, *, contains_any=None, contains_all=None, endswith=None, startswith_not=None):
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

machine_path = 'Volumes'
movie = 'despicable_me_hungarian'
freq_band = 'HFA'
method = 'power_log'
win_len = 10

#data_dir = f'/{machine_path}/Samsung/Movie_data/1secEpochs_for_review_power_log_{movie}_{freq_band}_1Apr26'

data_dir = '/Volumes/Samsung/Movie_data/1secEpochs_for_review_power_log_despicable_me_hungarian_HFA_1Apr26'

if movie == 'inscapes':
    entry_ids = [
                'NS127_02_ses-02_run-01',
                #'NS135_ses-01_run-01',
                # 'NS136_ses-01_run-01', 
                'NS137_ses-01_run-01',
                'NS138_ses-01_run-01', 
                'NS140_ses-01_run-01',
                'NS140_02_ses-02_run-01',                # 'NS140_02_ses-02_run-01',

                'NS151_ses-01_run-01', 
                'NS153_ses-01_run-01',
                'NS155_ses-01_run-01',
                'NS155_02_ses-02_run-01', 
                 'NS164_ses-01_run-01',
                  'NS178_ses-01_run-01', 
                'NS205_ses-01_run-01',
                'NS210_ses-01_run-01'
                ] 
    
if movie == 'despicable_me_english':
    entry_ids = ['NS127_02_ses-02_run-01', 
                 'NS135_ses-01_run-01',
                   'NS136_ses-01_run-01', 
                   'NS137_ses-01_run-01',
                   'NS138_ses-01_run-01', 
                   'NS140_ses-01_run-01',
                  # 'NS140_02_ses-02_run-01', 
                   'NS153_ses-01_run-01',
                   'NS155_02_ses-02_run-01', 
                   'NS164_ses-01_run-01',
                   #'NS166_ses-01_run-01', 
                   'NS174_02_ses-02_run-01',
                   'NS174_03_ses-03_run-01', 
                   'NS178_ses-01_run-01',
                   'NS190_ses-01_run-01',
                   'NS190_ses-01_run-02',
                   'NS191_ses-01_run-01', 
                   'NS193_ses-01_run-01',
                   'NS193_ses-01_run-02',
                   'NS194_ses-01_run-01',
                   'NS205_ses-01_run-01']
    
if movie == 'despicable_me_hungarian':
    entry_ids = [
       # 'LH010_ses-01_run-01',
        'NS127_02_ses-02_run-01',
        'NS135_ses-01_run-01',
        'NS136_ses-01_run-01', 
        'NS137_ses-01_run-01',
        'NS138_ses-01_run-01', 
        'NS140_ses-01_run-01',
        'NS140_02_ses-02_run-01',
      #  'NS145_ses-02_run-01', 
        'NS154_ses-01_run-01',
        'NS164_ses-01_run-01', 
        'NS174_02_ses-02_run-01',
        'NS174_03_ses-03_run-01',
        'NS178_ses-01_run-01'
    ]
    
    
# Loop through patients
for rec in entry_ids:
    pat = extract_pat_id(rec)
    run = extract_run_label(rec)
    
    pat_dir = os.path.join(data_dir, pat)
    # Find the CSV files containing atlas info and lfp data
    lfp_files = [
        filename for filename in os.listdir(pat_dir)
        if method in filename and filename.endswith('.csv') and not filename.startswith("._")
    ]
    
    if not lfp_files:
        print(f"No LFP files found for patient {pat} in {pat_dir}")
        continue
    
    if pat in ['NS190', 'NS193']:
        matches = pick_file(
            lfp_files,
            contains_any=[run]
        )
    
        if len(matches) == 0:
            print(f"No matching file found for {pat} {run} in {pat_dir}")
            continue
        elif len(matches) > 1:
            print(f"Multiple matching files found for {pat} {run}: {matches}")
            lfp_file = matches[0]
        else:
            lfp_file = matches[0]
    else:
        lfp_file = lfp_files[0]
    
    full_lfp_path = os.path.join(pat_dir, lfp_file)
    
    try:
        lfp_values = pd.read_csv(full_lfp_path)
        print(f"Processed LFP data for: {full_lfp_path}")
        
    except Exception as e:
        print(f"Error reading {full_lfp_path}: {e}")
        continue

    atlas_rows = lfp_values.head(4)
    data_rows = lfp_values.iloc[4:].reset_index(drop=True)
    
    df = lfp_values
    
    # drop metadata rows
    data = df.iloc[4:].copy()
    
    # drop non-channel columns
    non_channel_cols = ['Unnamed: 0', 'SubID', 'Atlas']
    channel_cols = [c for c in data.columns if c not in non_channel_cols]

    # convert to float
    X = data[channel_cols].astype(float).values  # shape: (windows × channels)
    
    med = np.median(X, axis=0)
    mad = median_abs_deviation(X, axis=0, scale='normal')
    
    # avoid divide-by-zero
    mad[mad == 0] = 1
    
    Z = (X - med) / mad
    
    threshold = 4  # start here
    
    frac_high = (Z > threshold).mean(axis=1)   # fraction of channels
    global_mean = Z.mean(axis=1)              # average across channels
    global_max = Z.max(axis=1)                # optional
    
    sync = np.std(Z, axis=1)
    mask = (frac_high > 0.1) & (global_mean > 1)
    
    # optional padding
    pad_seconds = 3   # set to 1.5 or 2
    pad_n = int(np.ceil(pad_seconds))   # 1.5 -> 2, 2 -> 2
    
    mask_padded = mask.copy()
    for shift in range(1, pad_n + 1):
        mask_padded[shift:] |= mask[:-shift]
        mask_padded[:-shift] |= mask[shift:]
        
    mask_df = pd.DataFrame({
        'window_idx': np.arange(len(frac_high)),
        'time_sec': np.arange(len(frac_high)),
        'frac_high': frac_high,
        'global_mean': global_mean,
        'mask': mask.astype(int),
        'mask_padded': mask_padded.astype(int)
    })
    
    mask_path = full_lfp_path.replace('.csv', '_global_HFA_mask.csv')
    mask_df.to_csv(mask_path, index=False)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    
    axes[0].plot(frac_high, linewidth=1.5)
    axes[0].axhline(0.1, linestyle='--', alpha=0.5)
    axes[0].set_ylabel('frac_high')
    axes[0].set_ylim(0, .5)
    axes[0].set_title(f'mask candidates for {pat} {run} {movie}')
    
    axes[1].plot(global_mean, linewidth=1.5)
    axes[1].axhline(2, linestyle='--', alpha=0.5)
    axes[1].set_ylabel('global_mean')
    axes[1].set_xlabel('Window')
    
    for ax in axes:
        bad_idx = np.where(mask_padded)[0]
        for i in bad_idx:
            ax.axvspan(i - 0.5, i + 0.5, alpha=0.15)
    
    plt.tight_layout()
    plot_path = full_lfp_path.replace('.csv', '_global_HFA_mask_plot.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Masked {mask.mean()*100:.1f}% of windows")
    print(f"Saved mask to {mask_path}")
    print(f"Saved plot to {plot_path}")
        
    mask_df['time_sec'] = mask_df['window_idx']
    
    mask_df.to_csv(full_lfp_path.replace('.csv', '_mask.csv'), index=False)

    # choose which 1 s mask to propagate
    mask_1s = mask_padded.astype(bool)   # or mask.astype(bool)
    
    movie_duration = 599.0
    short_bin_width = 1.0
    
    long_win_len = win_len
    long_step = 2.5
    
    # 1 s bin edges
    sec_idx = np.arange(len(mask_1s))
    sec_start = sec_idx.astype(float)
    sec_end = sec_start + short_bin_width
    
    # long-window starts
    long_starts = np.arange(0, movie_duration - long_win_len + 1e-9, long_step)
    
    rows = []
    
    for i, start in enumerate(long_starts):
        end = start + long_win_len
    
        # overlap in seconds between each 1 s bin and this 10 s window
        overlap_sec = np.maximum(0, np.minimum(sec_end, end) - np.maximum(sec_start, start))
    
        # total contaminated overlap
        bad_overlap_sec = overlap_sec[mask_1s].sum()
    
        # mark bad if ANY overlap with a masked 1 s bin
        mask = bad_overlap_sec > 0
    
        rows.append([i, start, end, bad_overlap_sec, int(mask)])
    
    long_mask_df = pd.DataFrame(
        rows,
        columns=[f'window_idx_{long_win_len}s', 'start_sec', 'end_sec', 'bad_overlap_sec', f'mask_{long_win_len}s']
    )
    
    # save all long-window QC info
    long_mask_path = full_lfp_path.replace('.csv', f'_{long_win_len}s_window_artifact_mask.csv')
    long_mask_df.to_csv(long_mask_path, index=False)
    
    # save only the bad long-window indices
    bad_idx_df = long_mask_df.loc[long_mask_df[f'mask_{long_win_len}s'] == 1, [f'window_idx_{long_win_len}s']].copy()
    bad_idx_path = full_lfp_path.replace('.csv', f'_{long_win_len}s_bad_window_indices.csv')
    bad_idx_df.to_csv(bad_idx_path, index=False)
    
    print(f"Saved long-window mask to: {long_mask_path}")
    print(f"Saved bad {long_win_len} s window indices to: {bad_idx_path}")
    print(f"Flagged {bad_idx_df.shape[0]} of {long_mask_df.shape[0]} long windows")