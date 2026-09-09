#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Eye Movement Measures Computation Script

This script computes derived eye movement measures from preprocessed gaze data,
aggregates them into a DataFrame for PCA analysis, and optionally saves per-patient
data files.

Based on: compute_eye_measures.py and prePCA_agg_norm.py

@author: christinechesebrough
Created: 2025
"""

import os
import glob
import math
from itertools import compress
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# CONFIGURATION
# ============================================================================

# Data directories
drive = 'Samsung'
data_dir = f'/Volumes/{drive}/Movie_data/movies_prep_standard'
isc_dir = f'/Volumes/{drive}/Movie_data/data/isc'
fig_dir = f'/Volumes/{drive}/Movie_data/eye_measures_unified'
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

# Video/movie identifier
vid = 'inscapes'  # Options: 'inscapes', 'despicable_me_english', 'dme'

# Patient list
if vid == 'dme':
    patients = ["NS190", 'NS191']
elif vid == 'despicable_me_english':
    patients = ['NS127_02', 'NS135', 'NS136', 'NS137', 'NS138', 'NS140', 
                'NS153', 'NS154', 'NS164', 'NS166', 'NS174_02']
elif vid == 'inscapes':
    patients = ['NS127_02', 'NS135', 'NS136', 'NS137', 'NS138', 'NS140', 
                'NS140_02', 'NS154', 'NS164', 'NS166', 'NS178', 'NS205', 
                'NS210', 'NS211']

patients.sort()

# Processing flags
save_per_patient_npz = True  # Save updated npz files per patient
save_per_patient_csv = True  # Save per-patient CSV files
normalize_per_patient = True  # Normalize measures per patient
normalize_across_patients = True  # Normalize across all patients for PCA

# Rolling window parameters
fs_eye = 300  # Eye tracking sampling rate
window_duration = 10.0  # Window size in seconds
overlap_duration = 7.5  # Overlap in seconds

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def find_eye_prep_file(eye_prep_dir, vid, pat=None):
    """
    Find eye prep npz file with flexible naming.
    
    Parameters:
    -----------
    eye_prep_dir : str
        Directory containing eye prep files
    vid : str
        Video identifier (must be in filename)
    pat : str, optional
        Patient ID (for special cases)
    
    Returns:
    --------
    str : Full path to eye prep npz file
    """
    # Special cases for specific patients
    if pat == "NS190":
        return '/Volumes/Samsung/Movie_data/movies_prep_standard/NS190/Eye_prep/NS190_ses-01_task-despicable_me_english_run-02_et_prep.npz'
    elif pat == "NS191":
        return '/Volumes/Samsung/Movie_data/movies_prep_standard/NS191/Eye_prep/NS191_ses-01_task-despicable_me_english_run-01_et_prep_updated.npz'
    
    # Flexible search pattern
    pattern = os.path.join(eye_prep_dir, f'*{vid}*et_prep*.npz')
    candidates = [
        f for f in glob.glob(pattern)
        if not os.path.basename(f).startswith("._")
        and os.path.basename(f).endswith('.npz')
    ]
    
    if not candidates:
        raise FileNotFoundError(f"No eye prep npz file found in {eye_prep_dir} for video {vid}")
    
    return candidates[0]


def find_vergence_file(eye_prep_dir, vid, pat=None):
    """
    Find vergence CSV file with flexible naming.
    
    Parameters:
    -----------
    eye_prep_dir : str
        Directory containing vergence files
    vid : str
        Video identifier (must be in filename)
    pat : str, optional
        Patient ID (for special cases)
    
    Returns:
    --------
    str : Full path to vergence CSV file
    """
    # Special cases for specific patients
    if pat == 'NS190':
        return '/Volumes/Samsung/Movie_data/movies_prep_standard/NS190/Eye_prep/NS190_ses-dme02_behavior+ecephys.csv'
    elif pat == "NS191":
        return "/Volumes/Samsung/Movie_data/movies_prep_standard/NS191/Eye_prep/NS191_ses-dme01_behavior+ecephys.csv"
    
    # Flexible search pattern
    pattern = os.path.join(eye_prep_dir, f'*{vid}*et_prep*.csv')
    candidates = [
        f for f in glob.glob(pattern)
        if not os.path.basename(f).startswith("._")
        and os.path.basename(f).endswith('.csv')
    ]
    
    if not candidates:
        raise FileNotFoundError(f"No vergence CSV file found in {eye_prep_dir} for video {vid}")
    
    return candidates[0]


def find_blink_file(eye_prep_dir, vid):
    """
    Find blink events CSV file with flexible naming.
    
    Parameters:
    -----------
    eye_prep_dir : str
        Directory containing blink files
    vid : str
        Video identifier (must be in filename)
    
    Returns:
    --------
    str : Full path to blink CSV file
    """
    pattern = os.path.join(eye_prep_dir, f'*{vid}*blink*.csv')
    candidates = [
        f for f in glob.glob(pattern)
        if not os.path.basename(f).startswith("._")
        and os.path.basename(f).endswith('.csv')
    ]
    
    if not candidates:
        raise FileNotFoundError(f"No blink events CSV file found in {eye_prep_dir} for video {vid}")
    
    return candidates[0]


def load_patient_data(pat_dir, vid, pat):
    """
    Load all eye tracking data files for a patient.
    
    Parameters:
    -----------
    pat_dir : str
        Patient directory path
    vid : str
        Video identifier
    pat : str
        Patient ID
    
    Returns:
    --------
    dict : Dictionary with keys 'eye_data', 'verg_data', 'blink_data'
    """
    eye_prep_dir = os.path.join(pat_dir, 'Eye_prep')
    
    # Load eye prep npz (only once!)
    eye_file = find_eye_prep_file(eye_prep_dir, vid, pat)
    eye_data = np.load(eye_file, allow_pickle=True)
    
    # Load vergence CSV
    verg_file = find_vergence_file(eye_prep_dir, vid, pat)
    verg_data = pd.read_csv(verg_file)
    
    # Load blink events CSV
    blink_file = find_blink_file(eye_prep_dir, vid)
    blink_data = pd.read_csv(blink_file)
    
    return {
        'eye_data': eye_data,
        'verg_data': verg_data,
        'blink_data': blink_data,
        'eye_file': eye_file
    }


def compute_rolling_measures(patient_data, window_params, isc_aligned_time):
    """
    Compute all rolling eye movement measures for a single patient.
    
    Parameters:
    -----------
    patient_data : dict
        Dictionary containing loaded patient data
    window_params : dict
        Dictionary with window parameters (window_samples, step_size_samples, num_steps)
    isc_aligned_time : array
        Time array aligned to ISC data
    
    Returns:
    --------
    dict : Dictionary with all computed rolling measures
    """
    eye_data = patient_data['eye_data']
    verg_data = patient_data['verg_data']
    blink_data = patient_data['blink_data']
    
    # Extract data
    t = eye_data['t_gaze']
    xy = eye_data['xy']
    t_pupil = eye_data['t_pupil']
    pupil = eye_data['pupil']
    saccade_onset_t = eye_data['saccade_onset_t']
    fixation_t = eye_data['fixation_t']
    saccade_pos = eye_data['saccade_pos']
    
    # Movie time boundaries
    t_start = t_pupil[0]
    t_end = t_pupil[-1]
    
    # Extract vergence (using vis_fd_interp as standard)
    verg_t = verg_data['time'].values
    vergence = verg_data['vis_fd_interp'].values  # STANDARD: vis_fd_interp
    
    # Extract blink data
    blink_onset = blink_data['start_time'].values
    blink_end = blink_data['end_time'].values
    blink_duration = blink_data['duration'].values
    
    # Cut all data around movie times
    idx_vid_verg = np.logical_and(verg_t > t_start, verg_t < t_end)
    verg_t = verg_t[idx_vid_verg]
    vergence = vergence[idx_vid_verg]
    verg_t = verg_t - verg_t[0]  # Zero out time
    
    idx_vid = np.logical_and(t > t_start, t < t_end)
    t = t[idx_vid]
    xy = xy[idx_vid]
    t = t - t[0]  # Zero out time
    
    # Align saccade/fixation/blink times
    saccade_onset_t = saccade_onset_t - t_start
    fixation_t = fixation_t - t_start
    blink_onset = blink_onset - t_start
    blink_end = blink_end - t_start
    t_pupil = t_pupil - t_start
    
    # Verify timing alignment
    if not np.allclose(t, verg_t, atol=1e-6, rtol=1e-5):
        print(f"WARNING: Gaze and vergence timings are not identical for patient!")
    
    # Window parameters
    window_samples = window_params['window_samples']
    step_size_samples = window_params['step_size_samples']
    num_steps = window_params['num_steps']
    
    # Initialize output arrays
    measures = {
        'verg_sliding': np.zeros(num_steps),
        'verg_sliding_std': np.zeros(num_steps),
        'abs_sliding_vergence': np.zeros(num_steps),
        'saccade_rate_sliding': np.zeros(num_steps),
        'rolling_dispersion': np.zeros(num_steps),
        'rolling_dispersion_std': np.zeros(num_steps),
        'rolling_blink_rate': np.zeros(num_steps),
        'rolling_blink_duration': np.zeros(num_steps),
        'rolling_pupil_avg': np.zeros(num_steps),
        'rolling_pupil_std': np.zeros(num_steps),
    }
    
    # ========================================================================
    # Compute rolling vergence
    # ========================================================================
    for i in range(num_steps):
        window_start_idx = int(i * step_size_samples)
        window_end_idx = int(window_start_idx + window_samples)
        
        window_data = vergence[window_start_idx:window_end_idx]
        measures['verg_sliding'][i] = np.mean(window_data)
        measures['verg_sliding_std'][i] = np.std(window_data)
    
    measures['abs_sliding_vergence'] = np.abs(measures['verg_sliding'])
    
    # ========================================================================
    # Compute rolling saccade rate
    # ========================================================================
    is_saccade_onset = np.zeros(len(t), dtype=bool)
    for onset_time in saccade_onset_t:
        idx = np.argmin(np.abs(t - onset_time))
        is_saccade_onset[idx] = True
    saccade_onset_int = is_saccade_onset.astype(int)
    
    for i in range(num_steps):
        window_start_idx = int(i * step_size_samples)
        window_end_idx = int(window_start_idx + window_samples)
        measures['saccade_rate_sliding'][i] = np.sum(saccade_onset_int[window_start_idx:window_end_idx])
    
    # ========================================================================
    # Compute saccade dispersions
    # ========================================================================
    assert len(saccade_onset_t) == len(fixation_t), "Mismatch in number of saccades and fixations"
    
    saccade_dispersions = []
    for saccade_time, fixation_time in zip(saccade_onset_t, fixation_t):
        saccade_idx = np.argmin(np.abs(t - saccade_time))
        saccade_pos_xy = xy[saccade_idx]
        
        fixation_idx = np.argmin(np.abs(t - fixation_time))
        fixation_pos_xy = xy[fixation_idx]
        
        if np.isnan(saccade_pos_xy).any() or np.isnan(fixation_pos_xy).any():
            dispersion = np.nan
        else:
            dispersion = math.dist(saccade_pos_xy, fixation_pos_xy)
        
        saccade_dispersions.append(dispersion)
    
    # Also compute dispersion from saccade_pos array (additional measure)
    saccade_dispersions_dist = []
    for row in saccade_pos:
        xy1 = row[0:2]
        xy2 = row[2:4]
        distance = np.linalg.norm(xy1 - xy2)
        saccade_dispersions_dist.append(distance)
    
    # Align dispersions to time array
    saccade_dispersion_aligned = np.full_like(t, np.nan, dtype=float)
    for i, (saccade_time, dispersion) in enumerate(zip(saccade_onset_t, saccade_dispersions)):
        idx = np.argmin(np.abs(t - saccade_time))
        saccade_dispersion_aligned[idx] = dispersion
    
    # Compute rolling dispersion
    for i in range(num_steps):
        window_start_idx = int(i * step_size_samples)
        window_end_idx = int(window_start_idx + window_samples)
        
        window_values = saccade_dispersion_aligned[window_start_idx:window_end_idx]
        measures['rolling_dispersion'][i] = np.nanmean(window_values)
        measures['rolling_dispersion_std'][i] = np.nanstd(window_values)
    
    # ========================================================================
    # Compute rolling blink rate and duration
    # ========================================================================
    is_blink_onset = np.zeros(len(t), dtype=bool)
    for blink_time in blink_onset:
        idx = np.argmin(np.abs(t - blink_time))
        is_blink_onset[idx] = True
    blink_onset_int = is_blink_onset.astype(int)
    
    blink_duration_aligned = np.full_like(t, np.nan, dtype=float)
    for blink_time, duration in zip(blink_onset, blink_duration):
        idx = np.argmin(np.abs(t - blink_time))
        blink_duration_aligned[idx] = duration
    
    for i in range(num_steps):
        window_start_idx = int(i * step_size_samples)
        window_end_idx = int(window_start_idx + window_samples)
        
        measures['rolling_blink_rate'][i] = np.sum(blink_onset_int[window_start_idx:window_end_idx])
        
        window_values = blink_duration_aligned[window_start_idx:window_end_idx]
        measures['rolling_blink_duration'][i] = np.nanmean(window_values)
    
    # ========================================================================
    # Compute rolling pupil measures
    # ========================================================================
    fs_pupil = len(pupil) / (t_end - t_start)
    target_num_steps = len(isc_aligned_time)
    total_samples = len(pupil)
    
    step_size_samples_pupil = int(total_samples / (target_num_steps + 1))
    window_samples_pupil = int(2 * step_size_samples_pupil)
    
    for i in range(num_steps):
        window_start_idx = int(i * step_size_samples_pupil)
        window_end_idx = int(window_start_idx + window_samples_pupil)
        
        if window_end_idx > total_samples:
            window_end_idx = total_samples
        
        measures['rolling_pupil_avg'][i] = np.mean(pupil[window_start_idx:window_end_idx])
        measures['rolling_pupil_std'][i] = np.std(pupil[window_start_idx:window_end_idx])
    
    # Store additional data for npz output
    measures['saccade_dispersions'] = saccade_dispersions
    measures['saccade_dispersions_dist'] = saccade_dispersions_dist
    measures['fixation_int'] = np.zeros(len(t), dtype=int)
    measures['saccade_onset_int'] = saccade_onset_int
    
    # Create fixation_int array
    is_fixation_onset = np.zeros(len(t), dtype=bool)
    for onset_time in fixation_t:
        idx = np.argmin(np.abs(t - onset_time))
        is_fixation_onset[idx] = True
    measures['fixation_int'] = is_fixation_onset.astype(int)
    
    return measures


def normalize_measures(df, columns_to_normalize, per_patient=True):
    """
    Normalize measures using z-score normalization.
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame with measures to normalize
    columns_to_normalize : list
        List of column names to normalize
    per_patient : bool
        If True, normalize per patient. If False, normalize across all patients.
    
    Returns:
    --------
    DataFrame : Normalized DataFrame
    """
    df_norm = df.copy()
    
    if per_patient:
        # Normalize per patient
        for pat in df_norm['Patient'].unique():
            pat_mask = df_norm['Patient'] == pat
            for col in columns_to_normalize:
                if col in df_norm.columns:
                    # Replace NaNs with 0 before normalization
                    df_norm.loc[pat_mask, col] = df_norm.loc[pat_mask, col].fillna(0)
                    # Z-score normalization
                    mean_val = df_norm.loc[pat_mask, col].mean()
                    std_val = df_norm.loc[pat_mask, col].std()
                    if std_val > 0:
                        df_norm.loc[pat_mask, col] = (df_norm.loc[pat_mask, col] - mean_val) / std_val
    else:
        # Normalize across all patients
        for col in columns_to_normalize:
            if col in df_norm.columns:
                # Replace NaNs with 0 before normalization
                df_norm[col] = df_norm[col].fillna(0)
                # Z-score normalization
                mean_val = df_norm[col].mean()
                std_val = df_norm[col].std()
                if std_val > 0:
                    df_norm[col] = (df_norm[col] - mean_val) / std_val
    
    return df_norm


# ============================================================================
# MAIN PROCESSING
# ============================================================================

# Load ISC data (once, not per patient)
print(f"Loading ISC data for {vid}...")
if vid == 'despicable_me_english' or vid == 'dme':
    isc_data = np.load(os.path.join(isc_dir, 'isc_gaze_position_time.npz'))
elif vid == 'inscapes':
    isc_data = np.load('/Volumes/Samsung/Movie_data/ISC_inscapes_updated_new/inscapes_isc_gaze_position_time_updated.npz')

time_isc = isc_data['time_isc']
patients_isc = isc_data['patients']
isc_time = isc_data['isc_time_gaze']
fs_isc = 1 / np.mean(np.diff(time_isc))

# Calculate window parameters
window_samples = int(window_duration * fs_eye)
overlap_samples = int(overlap_duration * fs_eye)
step_size_samples = window_samples - overlap_samples

# Calculate number of steps (will be adjusted per patient based on data length)
# Use ISC time as reference
isc_aligned_time = np.arange(window_samples / 2, 
                             len(time_isc) * int(fs_eye / fs_isc) - window_samples / 2 + 1, 
                             step_size_samples) / fs_eye

# Initialize storage for all patients
all_eye_data = []
all_patient_measures = {}

print(f"\nProcessing {len(patients)} patients...")
print("=" * 80)

# Process each patient
for pat in patients:
    print(f"\nProcessing patient: {pat}")
    pat_dir = os.path.join(data_dir, pat)
    fig_patient_dir = os.path.join(fig_dir, pat)
    if not os.path.exists(fig_patient_dir):
        os.makedirs(fig_patient_dir)
    
    try:
        # Load patient data (only once!)
        patient_data = load_patient_data(pat_dir, vid, pat)
        eye_data = patient_data['eye_data']
        
        # Get ISC for this patient
        idx_pat = np.in1d(patients_isc, pat)
        if np.sum(idx_pat) > 0:
            isc_pat = isc_time[idx_pat]
            isc_pat_t = np.squeeze(isc_pat.T)
        else:
            print(f"WARNING: No ISC data found for {pat}")
            isc_pat_t = np.zeros(len(isc_aligned_time))
        
        # Calculate actual number of steps based on data length
        # Use vergence data length as reference
        verg_t = patient_data['verg_data']['time'].values
        t_start = eye_data['t_pupil'][0]
        t_end = eye_data['t_pupil'][-1]
        idx_vid = np.logical_and(verg_t > t_start, verg_t < t_end)
        data_length = np.sum(idx_vid)
        
        num_steps = int(np.floor((data_length - window_samples) / step_size_samples) + 1)
        
        window_params = {
            'window_samples': window_samples,
            'step_size_samples': step_size_samples,
            'num_steps': num_steps
        }
        
        # Compute rolling measures
        measures = compute_rolling_measures(patient_data, window_params, isc_aligned_time)
        
        # Store measures
        all_patient_measures[pat] = measures
        
        # Create patient DataFrame
        patient_df = pd.DataFrame({
            'Patient': [pat] * num_steps,
            'Time': isc_aligned_time[:num_steps],
            'Saccade_Rate': measures['saccade_rate_sliding'],
            'Vergence': measures['verg_sliding'],
            'Vergence_Std': measures['verg_sliding_std'],
            'Abs_Vergence': measures['abs_sliding_vergence'],
            'Saccade_Dispersion': measures['rolling_dispersion'],
            'Saccade_Dispersion_Std': measures['rolling_dispersion_std'],
            'Blink_Rate': measures['rolling_blink_rate'],
            'Blink_Duration': measures['rolling_blink_duration'],
            'Pupil_Avg': measures['rolling_pupil_avg'],
            'Pupil_Std': measures['rolling_pupil_std'],
            'ISC': isc_pat_t[:num_steps] if len(isc_pat_t) >= num_steps else np.pad(isc_pat_t, (0, num_steps - len(isc_pat_t)), 'constant'),
        })
        
        # Normalize per patient if requested
        if normalize_per_patient:
            columns_to_norm = [
                'Saccade_Rate', 'Vergence', 'Vergence_Std', 'Abs_Vergence',
                'Saccade_Dispersion', 'Saccade_Dispersion_Std',
                'Blink_Rate', 'Blink_Duration'
            ]
            patient_df = normalize_measures(patient_df, columns_to_norm, per_patient=True)
        
        # Save per-patient CSV if requested
        if save_per_patient_csv:
            patient_csv_file = os.path.join(fig_patient_dir, f'eye_measures_{vid}_{pat}.csv')
            patient_df.to_csv(patient_csv_file, index=False)
            print(f"  Saved per-patient CSV: {patient_csv_file}")
        
        # Save updated npz file if requested
        if save_per_patient_npz:
            et_data_dict = dict(eye_data)
            et_data_dict['fixation_int'] = measures['fixation_int']
            et_data_dict['saccade_onset_int'] = measures['saccade_onset_int']
            et_data_dict['saccade_dispersion'] = measures['saccade_dispersions']
            et_data_dict['saccade_dispersion_dist'] = measures['saccade_dispersions_dist']
            
            eye_file = patient_data['eye_file']
            et_file_new = eye_file.replace('.npz', '_updated.npz')
            np.savez(et_file_new, **et_data_dict)
            print(f"  Saved updated npz: {et_file_new}")
        
        # Append to combined data
        all_eye_data.append(patient_df)
        
        print(f"  ✓ Completed {pat}")
        
    except Exception as e:
        print(f"  ✗ Error processing {pat}: {e}")
        continue

# ============================================================================
# AGGREGATE ALL PATIENTS
# ============================================================================

if len(all_eye_data) > 0:
    print(f"\n{'='*80}")
    print("Aggregating data from all patients...")
    
    # Concatenate all patient DataFrames
    final_eye_df = pd.concat(all_eye_data, ignore_index=True)
    
    # Save raw aggregated data
    raw_csv_file = os.path.join(fig_dir, f'eye_measures_raw_{vid}.csv')
    final_eye_df.to_csv(raw_csv_file, index=False)
    print(f"Saved raw aggregated data: {raw_csv_file}")
    
    # Normalize across all patients if requested
    if normalize_across_patients:
        columns_to_norm = [
            'Saccade_Rate', 'Vergence', 'Vergence_Std', 'Abs_Vergence',
            'Saccade_Dispersion', 'Saccade_Dispersion_Std',
            'Blink_Rate', 'Blink_Duration', 'Pupil_Avg', 'Pupil_Std', 'ISC'
        ]
        normed_eye_df = normalize_measures(final_eye_df, columns_to_norm, per_patient=False)
        
        # Save normalized data
        normed_csv_file = os.path.join(fig_dir, f'eye_measures_normalized_{vid}.csv')
        normed_eye_df.to_csv(normed_csv_file, index=False)
        print(f"Saved normalized aggregated data: {normed_csv_file}")
    
    # Create correlation heatmap
    print("\nCreating correlation heatmap...")
    correlation_matrix = final_eye_df.iloc[:, 2:].corr()  # Exclude Patient and Time columns
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", 
                cbar=True, square=True)
    plt.title(f'Correlation Heatmap of Eye Movement Features - {vid}', fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    
    heatmap_file = os.path.join(fig_dir, f'correlation_heatmap_{vid}.png')
    plt.savefig(heatmap_file, dpi=300)
    plt.close()
    print(f"Saved correlation heatmap: {heatmap_file}")
    
    print(f"\n{'='*80}")
    print(f"Processing complete! Processed {len(all_eye_data)} patients.")
    print(f"Output directory: {fig_dir}")
    
else:
    print("\nNo patient data was successfully processed!")


