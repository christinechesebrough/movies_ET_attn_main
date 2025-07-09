#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 16:22:55 2025

@author: christinechesebrough
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 20:40:27 2024

@author: max

To do:   
    ISC and variabiltiy across different frequency bands

"""
import os
from itertools import compress
import numpy as np
import pandas as pd 
from scipy import interpolate, signal
from tqdm import tqdm
import matplotlib.pyplot as plt
#from utils import interp_nans, compute_isc, time_resolved_isc, plot_isc_pat
    

# Updated directory paths to match preprocess_movies_newieeg2nwb.py
data_dir = '/Volumes/Samsung/Movie_data/movies_new_prep'
data_dir_old = '/Volumes/Samsung/Movie_data/movies_prep_standard'  # Older data directory
fs_dir = '/Volumes/Samsung/anatomy'
eloc_dir = '/Volumes/Samsung/Movie_data/data/electrode_localization'

vid = 'despicable_me_english'

compute_ISC = False
plot_gaze_summary = True


# Patient lists from old/new directories

if vid == 'despicable_me_english':
    old_data_patients = ['NS127_02','NS135','NS136','NS137','NS138','NS140','NS140_02','NS145',
     'NS151','NS151_02','NS153','NS154','NS164','NS166','NS174_02','NS174_03']
    new_data_patients = ['NS155_02','NS178','NS189_03','NS190','NS191','NS193','NS194','NS201_02','NS203','NS204','NS205','NS206']

if vid == 'inscapes':
    old_data_patients = ['NS127_02','NS135','NS136','NS137','NS138','NS140','NS140_02','NS144',
     'NS154','NS164','NS174_02']
    new_data_patients = ['NS155_02','NS178','NS205','NS211']


#old_data_patients = ['NS127_02','NS128_02','NS135','NS136','NS137','NS138','NS140','NS140_02','NS142','NS144','NS144_02','NS145','NS148_02','NS149',
# 'NS151','NS151_02','NS153','NS154','NS155','NS164','NS166','NS167','NS170','NS174_02','NS174_03']

#old_data_patients = []

#new_data_patients = ['NS155_02','NS178','NS189','NS190','NS205','NS201_02','NS204','NS206','NS192','NS193','NS194','NS191','NS189_03',
# 'NS203','NS155','NS210','NS211'] 

#new_data_patients = []


# Print patient configuration
print("\n=== PATIENT DATA SOURCE CONFIGURATION ===")
print(f"Old data patients: {old_data_patients}")
print(f"New data patients: {new_data_patients}")
print("Note: Patients not in either list will use the default data source")


# Create results directory 
results_dir = f'/Volumes/Samsung/Movie_data/ISC_{vid}_newsubs'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Updated search string to match the actual file naming convention
search_string = f'{vid}_et_prep.npz'

if len(old_data_patients) > 0 or len(new_data_patients) > 0:
    # Get patients from both directories
    old_dir_patients = os.listdir(data_dir_old) if os.path.exists(data_dir_old) else []
    new_dir_patients = os.listdir(data_dir) if os.path.exists(data_dir) else []
    
    # Combine and filter based on patient lists
    all_patients = set(old_dir_patients + new_dir_patients)
    patients = []
    
    for pat in all_patients:
        if pat.startswith('NS'):
            # Include patients that are in either list
            if pat in old_data_patients or pat in new_data_patients:
                patients.append(pat)
    
    patients.sort()
    print(f"Found {len(patients)} patients from mixed data sources: {patients}")


# Show which patients will be processed from which source
print(f"\n=== PATIENT PROCESSING SUMMARY ===")
print(f"Total patients to process: {len(patients)}")
print(f"Patients from old data: {[p for p in patients if p in old_data_patients]}")
print(f"Patients from new data: {[p for p in patients if p in new_data_patients]}")
print(f"Patients using default source: {[p for p in patients if p not in old_data_patients and p not in new_data_patients]}")


#aic_sheet = 'aic_contacts.xlsx'
#aic_table = pd.read_excel('{:s}/{:s}'.format(eloc_dir, aic_sheet))
#aic_patients = np.unique(aic_table.patient)

# Eyetracking sampling rate
fs_eye = 300

# Downsampling factor
dsf = 3

# Time resolved ISC
window = 1000
overlap = 750

# Number of permutations
n_perm = 100


 #%% Utils
 
def interp_nans(data):
    
    s = np.arange(len(data))    
    idx_nan = np.isnan(data)
    
    f = interpolate.interp1d(s[np.invert(idx_nan)], data[np.invert(idx_nan)], 
                             fill_value='extrapolate')
    data[idx_nan] = f(np.where(idx_nan)[0])
    
    return data

def compute_isc(xy):
    
    corr_x = np.corrcoef(xy[:,0,:].T)
    corr_y = np.corrcoef(xy[:,1,:].T)

    corr = np.mean(np.concatenate((np.expand_dims(corr_x, 2), 
                                   np.expand_dims(corr_y, 2)), 
                                  axis=2), 
                   axis=2)

    corr[np.eye(corr.shape[0]) == 1] = np.nan

    isc = np.nanmean(corr, axis=0)
    
    return isc

def time_resolved_isc(data, fs, window, overlap):
    
    step = window - overlap

    time_isc = np.arange(window/2,len(data)-window/2,step) / fs

    n_pat = data.shape[1]
    x_win = [None] * n_pat

    for ip in range(n_pat):
        
        ts = data[:,ip]
        
        shape = (ts.size - window + 1, window)
        strides = ts.strides * 2
        ts_win = np.lib.stride_tricks.as_strided(ts, shape=shape, strides=strides)
        
        x_win[ip] = ts_win[:-1:step, :]

    corr_time = np.empty((n_pat, x_win[ip].shape[0]))    
    for ip in range(n_pat):
        corr_pat = np.empty((n_pat, x_win[ip].shape[0]))
        for jp in range(n_pat):
            corr_mat = np.corrcoef(x_win[ip], x_win[jp])
            corr_pat[jp,:] = np.diag(corr_mat[:len(x_win[ip]), len(x_win[ip]):])
            
        corr_time[ip,:] = np.mean(corr_pat, axis=0)
        
    return corr_time, time_isc

def plot_isc_pat(isc, title_str):
    
    mean_isc = np.mean(isc)
    
    plt.figure()
    plt.hist(isc, color='tab:Gray', ec='k')

    ylim = plt.ylim()

    plt.plot([mean_isc, mean_isc], ylim, 'r', linewidth=2)
    plt.ylim(ylim)

    plt.xlabel('ISC')
    plt.ylabel('# of Patients')
    plt.legend(['Mean ISC', 'ISC for each patient'])
    plt.title(title_str)

    plt.grid()
    plt.tight_layout()
    
# Downstream processing functions

def calculate_gaze_measures(xy_gaze, t_gaze, saccade_onset_t, saccade_pos, pupil_data, t_pupil):
    """
    Calculate various gaze measures from preprocessed data
    
    Parameters:
    -----------
    xy_gaze : array
        Gaze position data (x, y coordinates)
    t_gaze : array
        Time vector for gaze data
    saccade_onset_t : array
        Saccade onset timestamps
    saccade_pos : array
        Saccade positions (x, y coordinates)
    pupil_data : array
        Processed pupil diameter data
    t_pupil : array
        Time vector for pupil data
    
    Returns:
    --------
    dict : Dictionary containing various gaze measures
    """
    
    # Calculate gaze velocity and speed
    gaze_velocity = np.gradient(xy_gaze, t_gaze, axis=0)
    gaze_speed = np.linalg.norm(gaze_velocity, axis=1)
    
    # Calculate gaze dispersion (spatial spread)
    gaze_dispersion = np.std(xy_gaze, axis=0)
    
    # Calculate saccade measures
    saccade_measures = {}
    if len(saccade_pos) > 1:
        sacc_amplitudes = np.linalg.norm(np.diff(saccade_pos, axis=0), axis=1)
        saccade_measures = {
            'mean_amplitude': np.mean(sacc_amplitudes),
            'max_amplitude': np.max(sacc_amplitudes),
            'saccade_rate': len(saccade_onset_t) / (t_gaze[-1] - t_gaze[0]),
            'total_saccades': len(saccade_onset_t)
        }
    
    # Calculate pupil measures
    pupil_measures = {
        'mean_diameter': np.mean(pupil_data),
        'std_diameter': np.std(pupil_data),
        'min_diameter': np.min(pupil_data),
        'max_diameter': np.max(pupil_data)
    }
    
    # Calculate fixation measures (time between saccades)
    if len(saccade_onset_t) > 1:
        fixation_durations = np.diff(saccade_onset_t)
        fixation_measures = {
            'mean_fixation_duration': np.mean(fixation_durations),
            'median_fixation_duration': np.median(fixation_durations),
            'total_fixation_time': np.sum(fixation_durations)
        }
    else:
        fixation_measures = {}
    
    return {
        'gaze_speed_mean': np.mean(gaze_speed),
        'gaze_speed_max': np.max(gaze_speed),
        'gaze_dispersion_x': gaze_dispersion[0],
        'gaze_dispersion_y': gaze_dispersion[1],
        'saccade_measures': saccade_measures,
        'pupil_measures': pupil_measures,
        'fixation_measures': fixation_measures
    }

def calculate_inter_subject_correlation(xy_data_list, t_data_list):
    """
    Calculate inter-subject correlation (ISC) for gaze data
    
    Parameters:
    -----------
    xy_data_list : list
        List of gaze position arrays for each subject
    t_data_list : list
        List of time vectors for each subject
    
    Returns:
    --------
    isc_x : array
        ISC for x-coordinate
    isc_y : array
        ISC for y-coordinate
    t_isc : array
        Time vector for ISC
    """
    
    # Find common time range
    t_min = max([t[0] for t in t_data_list])
    t_max = min([t[-1] for t in t_data_list])
    
    # Resample all data to common time grid
    fs_common = 30  # 30 Hz for ISC calculation
    t_common = np.arange(t_min, t_max, 1/fs_common)
    
    xy_resampled = []
    for xy, t in zip(xy_data_list, t_data_list):
        # Interpolate to common time grid
        f_x = interpolate.interp1d(t, xy[:,0], fill_value='extrapolate')
        f_y = interpolate.interp1d(t, xy[:,1], fill_value='extrapolate')
        
        xy_resampled.append(np.column_stack([f_x(t_common), f_y(t_common)]))
    
    # Calculate ISC for each coordinate
    n_subjects = len(xy_resampled)
    n_timepoints = len(t_common)
    
    isc_x = np.zeros(n_timepoints)
    isc_y = np.zeros(n_timepoints)
    
    for t in range(n_timepoints):
        # Extract data for current timepoint
        x_data = np.array([xy[t,0] for xy in xy_resampled])
        y_data = np.array([xy[t,1] for xy in xy_resampled])
        
        # Calculate correlation matrix
        if n_subjects > 1:
            corr_x = np.corrcoef(x_data)
            corr_y = np.corrcoef(y_data)
            
            # Remove diagonal (self-correlation)
            corr_x[np.eye(n_subjects) == 1] = np.nan
            corr_y[np.eye(n_subjects) == 1] = np.nan
            
            # Calculate mean correlation
            isc_x[t] = np.nanmean(corr_x)
            isc_y[t] = np.nanmean(corr_y)
    
    return isc_x, isc_y, t_common

def plot_gaze_measures_comparison(measures_list, patient_names):
    """
    Plot comparison of gaze measures across patients
    
    Parameters:
    -----------
    measures_list : list
        List of gaze measures dictionaries for each patient
    patient_names : list
        List of patient names
    """
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Extract measures for plotting
    gaze_speeds = [m['gaze_speed_mean'] for m in measures_list]
    saccade_rates = [m['saccade_measures'].get('saccade_rate', 0) for m in measures_list]
    pupil_means = [m['pupil_measures']['mean_diameter'] for m in measures_list]
    
    # Plot 1: Mean gaze speed
    axes[0,0].bar(patient_names, gaze_speeds, color='skyblue')
    axes[0,0].set_title('Mean Gaze Speed')
    axes[0,0].set_ylabel('Speed (pixels/s)')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Saccade rate
    axes[0,1].bar(patient_names, saccade_rates, color='lightcoral')
    axes[0,1].set_title('Saccade Rate')
    axes[0,1].set_ylabel('Saccades/s')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Plot 3: Mean pupil diameter
    axes[0,2].bar(patient_names, pupil_means, color='lightgreen')
    axes[0,2].set_title('Mean Pupil Diameter')
    axes[0,2].set_ylabel('Diameter (z-score)')
    axes[0,2].tick_params(axis='x', rotation=45)
    
    # Plot 4: Gaze dispersion
    disp_x = [m['gaze_dispersion_x'] for m in measures_list]
    disp_y = [m['gaze_dispersion_y'] for m in measures_list]
    x_pos = np.arange(len(patient_names))
    width = 0.35
    
    axes[1,0].bar(x_pos - width/2, disp_x, width, label='X dispersion', color='orange')
    axes[1,0].bar(x_pos + width/2, disp_y, width, label='Y dispersion', color='purple')
    axes[1,0].set_title('Gaze Dispersion')
    axes[1,0].set_ylabel('Dispersion (pixels)')
    axes[1,0].set_xticks(x_pos)
    axes[1,0].set_xticklabels(patient_names, rotation=45)
    axes[1,0].legend()
    
    # Plot 5: Saccade amplitudes (if available)
    sacc_amps = []
    for m in measures_list:
        if 'mean_amplitude' in m['saccade_measures']:
            sacc_amps.append(m['saccade_measures']['mean_amplitude'])
        else:
            sacc_amps.append(0)
    
    axes[1,1].bar(patient_names, sacc_amps, color='gold')
    axes[1,1].set_title('Mean Saccade Amplitude')
    axes[1,1].set_ylabel('Amplitude (pixels)')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    # Plot 6: Fixation durations (if available)
    fix_durs = []
    for m in measures_list:
        if 'mean_fixation_duration' in m['fixation_measures']:
            fix_durs.append(m['fixation_measures']['mean_fixation_duration'])
        else:
            fix_durs.append(0)
    
    axes[1,2].bar(patient_names, fix_durs, color='pink')
    axes[1,2].set_title('Mean Fixation Duration')
    axes[1,2].set_ylabel('Duration (s)')
    axes[1,2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
def find_patient_directories_with_file(root_dir, search_string):
    matching_directories = []

    # Traverse the directory tree
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Check if any file contains the search string
        for filename in filenames:
            if search_string in filename:
                # Get the parent directory above the patient directory
                parent_directory = os.path.dirname(os.path.normpath(dirpath))
                matching_directories.append(parent_directory)
                break  # Stop searching this directory once a match is found

    return matching_directories

#%%

matching_directories = find_patient_directories_with_file(data_dir, search_string)

print(matching_directories)

#%% Collect all data
xy = [None] * len(patients)
gaze_var = [None] * len(patients)

xy_perm = [None] * len(patients)
gaze_var_perm = [None] * len(patients)

# Initialize DataFrame to store statistics
stats_data = []

for i, pat in enumerate(patients):
    
    # Determine data source for this patient
    if pat in old_data_patients:
        current_data_dir = data_dir_old
        data_source = 'old'
        print(f"Loading {pat} from OLD data source")
    elif pat in new_data_patients:
        current_data_dir = data_dir
        data_source = 'new'
        print(f"Loading {pat} from NEW data source")
    else:
        # Default behavior - use the global data_dir setting
        current_data_dir = data_dir
        data_source = 'default'
        print(f"Loading {pat} from DEFAULT data source")
    
    pat_dir = '{:s}/{:s}'.format(current_data_dir, pat)
    
    # Check if patient directory exists in the specified data source
    if not os.path.exists(pat_dir):
        print(f"Patient directory {pat_dir} not found, skipping {pat}")
        continue
    
    # Check if Eye_prep directory exists
    eye_prep_dir = '{:s}/Eye_prep/'.format(pat_dir)
    if not os.path.exists(eye_prep_dir):
        print(f"Eye_prep directory not found for patient {pat} in {current_data_dir}, skipping...")
        continue
    
    et_files = os.listdir(eye_prep_dir)
    
    # Look for files containing the video name and et_prep.npz
    # Handle different file naming conventions between old and new data
    if data_source == 'old':
        # Old data format: sub-NS153_ses-01_task-despicable_me_english_run-1_et_prep.npz
        et_file = [file for file in et_files if vid in file and 'et_prep.npz' in file and 'task-' in file]
    else:
        # New data format: NS155_ses-01_task-inscapes_run-1_et_prep.npz
        et_file = [file for file in et_files if vid in file and 'et_prep.npz' in file]
    
    if not et_file:
        print(f"No matching et_prep.npz files found for patient {pat} and video {vid}")
        print(f"Available files: {et_files}")
        print(f"Searched in directory: {eye_prep_dir}")
        continue
    
    et_file = et_file[0]  # Take the first matching file
    
  #  if not np.in1d(pat, aic_patients)[0]:
  #      continue
    
    print('Loading data for patient {:s} ...'.format(pat))
    
    et_data = np.load('{:s}/Eye_prep/{:s}'.format(pat_dir, et_file))
    
    # Print available keys in the npz file for debugging
    print(f"Available keys in {et_file}: {list(et_data.keys())}")
    
    t = et_data['t_gaze']
    
    # Interpolate missing data
    x = interp_nans(et_data['xy'][:,0])
    y = interp_nans(et_data['xy'][:,1])
       
    xy_pat = np.vstack((x,y)).T
    
    # Interpolate gaps in time axis
    t_res = np.arange(t[0], t[-1], 1/fs_eye)
    f = interpolate.interp1d(t, xy_pat.T)
    xy_pat = f(t_res).T
    t = t_res

    # Compute gaze variation
    xy_pat -= np.mean(xy_pat, axis=0)
    
    env = np.abs(signal.hilbert(xy_pat, axis=0))
    gaze_var_pat = np.sqrt(np.sum(env**2, axis=1))
    
    # Cut around movie    
    idx_mov = np.logical_and(t >= et_data['t_pupil'][0], t < et_data['t_pupil'][-1])
    
    xy[i] = xy_pat[idx_mov,:]
    gaze_var[i] = gaze_var_pat[idx_mov]
    
    # Demonstrate access to saved gaze data from preprocessing
    print(f"\n=== Gaze Data Analysis for Patient {pat} ===")
    
    # Access all the saved data from preprocessing
    try:
        saccade_onset_t = et_data['saccade_onset_t']
        print(f"Found saccade_onset_t with {len(saccade_onset_t)} saccades")
    except KeyError:
        print("WARNING: saccade_onset_t not found in data file")
        saccade_onset_t = np.array([])
    
    try:
        fixation_t = et_data['fixation_t']
        print(f"Found fixation_t with {len(fixation_t)} fixations")
    except KeyError:
        print("WARNING: fixation_t not found in data file")
        fixation_t = np.array([])
    
    try:
        saccade_pos = et_data['saccade_pos']
        print(f"Found saccade_pos with shape {saccade_pos.shape}")
    except KeyError:
        print("WARNING: saccade_pos not found in data file")
        saccade_pos = np.array([])
    
    try:
        xy_gaze = et_data['xy']  # Original gaze data
        print(f"Found xy_gaze with shape {xy_gaze.shape}")
    except KeyError:
        print("ERROR: xy_gaze not found in data file - this is required!")
        continue
    
    try:
        pupil_data = et_data['pupil']
        print(f"Found pupil_data with shape {pupil_data.shape}")
    except KeyError:
        print("WARNING: pupil_data not found in data file")
        pupil_data = np.array([])
    
    try:
        t_pupil = et_data['t_pupil']
        print(f"Found t_pupil with {len(t_pupil)} timepoints")
    except KeyError:
        print("WARNING: t_pupil not found in data file")
        t_pupil = np.array([])
    
    try:
        t_gaze = et_data['t_gaze']
        print(f"Found t_gaze with {len(t_gaze)} timepoints")
    except KeyError:
        print("ERROR: t_gaze not found in data file - this is required!")
        continue
    
    # Handle missing fs_gaze key (common in older data)
    try:
        fs_gaze = et_data['fs_gaze']
    except KeyError:
        # Calculate fs_gaze from time data
        if len(t_gaze) > 1:
            fs_gaze = 1 / np.mean(np.diff(t_gaze))
            print(f"Calculated fs_gaze from time data: {fs_gaze:.2f} Hz")
        else:
            fs_gaze = 300  # Default fallback
            print(f"Could not calculate fs_gaze, using default: {fs_gaze} Hz")
    
    print(f"Gaze data shape: {xy_gaze.shape}")
    print(f"Number of saccades detected: {len(saccade_onset_t)}")
    print(f"Number of fixations detected: {len(fixation_t)}")
    print(f"Pupil data shape: {pupil_data.shape if len(pupil_data) > 0 else 'Not available'}")
    print(f"Gaze sampling rate: {fs_gaze} Hz")
    if len(t_pupil) > 0:
        print(f"Pupil time range: {t_pupil[0]:.2f}s to {t_pupil[-1]:.2f}s")
    else:
        print("Pupil time range: Not available")
    print(f"Gaze time range: {t_gaze[0]:.2f}s to {t_gaze[-1]:.2f}s")
    
    if plot_gaze_summary:
        # Plot the gaze data with saccades and fixations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Gaze trajectory with saccades
        axes[0,0].plot(xy_gaze[:,0], xy_gaze[:,1], 'b-', alpha=0.5, label='Gaze trajectory')
        if len(saccade_pos) > 0:
            axes[0,0].scatter(saccade_pos[:,0], saccade_pos[:,1], c='red', s=20, label='Saccade onsets')
        axes[0,0].set_xlabel('X position (pixels)')
        axes[0,0].set_ylabel('Y position (pixels)')
        axes[0,0].set_title(f'Gaze Trajectory with Saccades - {pat}')
        axes[0,0].legend()
        axes[0,0].invert_yaxis()  # Screen coordinates typically have y=0 at top
        
        # Plot 2: Gaze position over time
        axes[0,1].plot(t_gaze, xy_gaze[:,0], 'b-', label='X position')
        axes[0,1].plot(t_gaze, xy_gaze[:,1], 'g-', label='Y position')
        if len(saccade_onset_t) > 0 and len(saccade_pos) > 0:
            axes[0,1].scatter(saccade_onset_t, saccade_pos[:,0], c='red', s=10, label='Saccades')
        axes[0,1].set_xlabel('Time (s)')
        axes[0,1].set_ylabel('Position (pixels)')
        axes[0,1].set_title(f'Gaze Position Over Time - {pat}')
        axes[0,1].legend()
        
        # Plot 3: Pupil diameter over time
        if len(pupil_data) > 0 and len(t_pupil) > 0:
            axes[1,0].plot(t_pupil, pupil_data, 'purple')
            axes[1,0].set_xlabel('Time (s)')
            axes[1,0].set_ylabel('Pupil diameter (z-score)')
            axes[1,0].set_title(f'Pupil Diameter Over Time - {pat}')
        else:
            axes[1,0].text(0.5, 0.5, 'No pupil data available', ha='center', va='center', transform=axes[1,0].transAxes)
            axes[1,0].set_title(f'Pupil Data - {pat} (Not Available)')
        
        # Plot 4: Saccade amplitude histogram
        if len(saccade_pos) > 1:
            sacc_amplitudes = np.linalg.norm(np.diff(saccade_pos, axis=0), axis=1)
            axes[1,1].hist(sacc_amplitudes, bins=30, alpha=0.7, color='orange')
            axes[1,1].set_xlabel('Saccade amplitude (pixels)')
            axes[1,1].set_ylabel('Count')
            axes[1,1].set_title(f'Saccade Amplitude Distribution - {pat}')
        else:
            axes[1,1].text(0.5, 0.5, 'No saccade data available', ha='center', va='center', transform=axes[1,1].transAxes)
            axes[1,1].set_title(f'Saccade Data - {pat} (Not Available)')
        
        plt.tight_layout()
        plt.show()
    
    # Calculate and display some basic statistics
    print(f"\n=== Basic Statistics for {pat} ===")
    
    # Gaze statistics
    gaze_velocity = np.gradient(xy_gaze, t_gaze, axis=0)
    gaze_speed = np.linalg.norm(gaze_velocity, axis=1)
    
    # Handle NaN values in statistics
    valid_x = xy_gaze[:,0][~np.isnan(xy_gaze[:,0])]
    valid_y = xy_gaze[:,1][~np.isnan(xy_gaze[:,1])]
    valid_speed = gaze_speed[~np.isnan(gaze_speed)]
    
    print(f"Mean gaze speed: {np.mean(valid_speed):.2f} pixels/s")
    print(f"Max gaze speed: {np.max(valid_speed):.2f} pixels/s")
    print(f"Patient {pat}: X range [{np.min(valid_x):.1f}, {np.max(valid_x):.1f}]")
    print(f"Patient {pat}: Y range [{np.min(valid_y):.1f}, {np.max(valid_y):.1f}]")
    
    # Data quality metrics
    total_samples = len(xy_gaze)
    nan_x = np.sum(np.isnan(xy_gaze[:,0]))
    nan_y = np.sum(np.isnan(xy_gaze[:,1]))
    nan_both = np.sum(np.isnan(xy_gaze[:,0]) & np.isnan(xy_gaze[:,1]))
    
    print(f"Data quality:")
    print(f"  Total samples: {total_samples}")
    print(f"  NaN in X: {nan_x} ({nan_x/total_samples*100:.1f}%)")
    print(f"  NaN in Y: {nan_y} ({nan_y/total_samples*100:.1f}%)")
    print(f"  NaN in both: {nan_both} ({nan_both/total_samples*100:.1f}%)")
    print(f"  Valid samples: {total_samples - nan_both} ({(total_samples - nan_both)/total_samples*100:.1f}%)")
    
    # Check if coordinates are reasonable for screen dimensions
    if len(valid_x) > 0 and len(valid_y) > 0:
        if np.max(valid_x) > 2000 or np.max(valid_y) > 1200:
            print(f"  WARNING: Coordinates seem larger than expected screen size (1920x1080)")
        elif np.min(valid_x) < 0 or np.min(valid_y) < 0:
            print(f"  WARNING: Negative coordinates detected")
        else:
            print(f"  Coordinates appear to be within reasonable screen bounds")
    
    # Saccade statistics
    if len(saccade_pos) > 1:
        sacc_amplitudes = np.linalg.norm(np.diff(saccade_pos, axis=0), axis=1)
        print(f"Mean saccade amplitude: {np.mean(sacc_amplitudes):.2f} pixels")
        print(f"Max saccade amplitude: {np.max(sacc_amplitudes):.2f} pixels")
        print(f"Saccade rate: {len(saccade_onset_t) / (t_gaze[-1] - t_gaze[0]):.2f} saccades/s")
    else:
        print("Saccade statistics: Not available")
    
    # Pupil statistics
    if len(pupil_data) > 0:
        print(f"Pupil diameter mean: {np.mean(pupil_data):.3f}")
        print(f"Pupil diameter std: {np.std(pupil_data):.3f}")
        print(f"Pupil diameter range: [{np.min(pupil_data):.3f}, {np.max(pupil_data):.3f}]")
    else:
        print("Pupil statistics: Not available")
    
    print("=" * 50)

    # Append statistics to the list
    patient_stats = {
        'patient': pat,
        'gaze_speed_mean': np.mean(valid_speed) if len(valid_speed) > 0 else np.nan,
        'gaze_speed_max': np.max(valid_speed) if len(valid_speed) > 0 else np.nan,
        'gaze_dispersion_x': np.std(valid_x) if len(valid_x) > 0 else np.nan,
        'gaze_dispersion_y': np.std(valid_y) if len(valid_y) > 0 else np.nan,
        'x_range_min': np.min(valid_x) if len(valid_x) > 0 else np.nan,
        'x_range_max': np.max(valid_x) if len(valid_x) > 0 else np.nan,
        'y_range_min': np.min(valid_y) if len(valid_y) > 0 else np.nan,
        'y_range_max': np.max(valid_y) if len(valid_y) > 0 else np.nan,
        'data_quality_total_samples': total_samples,
        'data_quality_nan_x': nan_x,
        'data_quality_nan_y': nan_y,
        'data_quality_nan_both': nan_both,
        'data_quality_valid_samples': total_samples - nan_both,
        'data_quality_valid_percentage': (total_samples - nan_both) / total_samples * 100 if total_samples > 0 else 0,
        'fs_gaze': fs_gaze,
        'recording_duration': t_gaze[-1] - t_gaze[0] if len(t_gaze) > 1 else 0
    }
    
    # Add saccade statistics if available
    if len(saccade_onset_t) > 0:
        patient_stats.update({
            'saccade_rate': len(saccade_onset_t) / (t_gaze[-1] - t_gaze[0]) if (t_gaze[-1] - t_gaze[0]) > 0 else 0,
            'total_saccades': len(saccade_onset_t)
        })
        
        if len(saccade_pos) > 1:
            sacc_amplitudes = np.linalg.norm(np.diff(saccade_pos, axis=0), axis=1)
            patient_stats.update({
                'saccade_amplitude_mean': np.mean(sacc_amplitudes),
                'saccade_amplitude_max': np.max(sacc_amplitudes)
            })
        else:
            patient_stats.update({
                'saccade_amplitude_mean': np.nan,
                'saccade_amplitude_max': np.nan
            })
            
        if len(saccade_onset_t) > 1:
            fixation_durations = np.diff(saccade_onset_t)
            patient_stats.update({
                'mean_fixation_duration': np.mean(fixation_durations),
                'total_fixation_time': np.sum(fixation_durations)
            })
        else:
            patient_stats.update({
                'mean_fixation_duration': np.nan,
                'total_fixation_time': np.nan
            })
    else:
        patient_stats.update({
            'saccade_rate': np.nan,
            'total_saccades': 0,
            'saccade_amplitude_mean': np.nan,
            'saccade_amplitude_max': np.nan,
            'mean_fixation_duration': np.nan,
            'total_fixation_time': np.nan
        })
    
    # Add pupil statistics if available
    if len(pupil_data) > 0:
        patient_stats.update({
            'pupil_mean_diameter': np.mean(pupil_data),
            'pupil_std_diameter': np.std(pupil_data),
            'pupil_min_diameter': np.min(pupil_data),
            'pupil_max_diameter': np.max(pupil_data)
        })
    else:
        patient_stats.update({
            'pupil_mean_diameter': np.nan,
            'pupil_std_diameter': np.nan,
            'pupil_min_diameter': np.nan,
            'pupil_max_diameter': np.nan
        })
    
    stats_data.append(patient_stats)

#%% Eye Features Processing
print("\n=== Downstream Processing ===")

# Collect data for downstream processing
xy_data_list = []
t_data_list = []
measures_list = []
patient_names_list = []

for i, pat in enumerate(patients):
    if xy[i] is not None:  # Only process patients with valid data
        # Determine data source for this patient (same logic as above)
        if pat in old_data_patients:
            current_data_dir = data_dir_old
            data_source = 'old'
        elif pat in new_data_patients:
            current_data_dir = data_dir
            data_source = 'new'
        else:
            current_data_dir = data_dir
            data_source = 'default'
        
        # Get the original data from the loaded npz file
        pat_dir = '{:s}/{:s}'.format(current_data_dir, pat)
        
        # Check if patient directory exists in the specified data source
        if not os.path.exists(pat_dir):
            print(f"Patient directory {pat_dir} not found, skipping {pat}")
            continue
        
        # Check if Eye_prep directory exists
        eye_prep_dir = '{:s}/Eye_prep/'.format(pat_dir)
        if not os.path.exists(eye_prep_dir):
            print(f"Eye_prep directory not found for patient {pat} in {current_data_dir}, skipping...")
            continue
        
        et_files = os.listdir(eye_prep_dir)
        
        # Use appropriate file naming convention
        if data_source == 'old':
            et_file = [file for file in et_files if vid in file and 'et_prep.npz' in file and 'task-' in file]
        else:
            et_file = [file for file in et_files if vid in file and 'et_prep.npz' in file]
        
        if not et_file:
            print(f"No matching et_prep.npz files found for patient {pat} and video {vid}")
            continue
        
        et_file = et_file[0]
        et_data = np.load('{:s}/Eye_prep/{:s}'.format(pat_dir, et_file))
        
        # Extract data for downstream processing
        xy_gaze = et_data['xy']
        t_gaze = et_data['t_gaze']
        saccade_onset_t = et_data['saccade_onset_t']
        saccade_pos = et_data['saccade_pos']
        pupil_data = et_data['pupil']
        t_pupil = et_data['t_pupil']
        
        # Calculate gaze measures
        measures = calculate_gaze_measures(xy_gaze, t_gaze, saccade_onset_t, 
                                        saccade_pos, pupil_data, t_pupil)
        
        # Store for ISC calculation
        xy_data_list.append(xy_gaze)
        t_data_list.append(t_gaze)
        measures_list.append(measures)
        patient_names_list.append(pat)
        
        print(f"Processed {pat}: {measures}")
        
#%%

# Save statistics to CSV
stats_df = pd.DataFrame(stats_data)
stats_df.to_csv(f'{results_dir}/{vid}_gaze_statistics.csv', index=False)
print(f"\nGaze statistics saved to {results_dir}/{vid}_gaze_statistics.csv")

# Display summary statistics
print(f"\n=== SUMMARY STATISTICS FOR {vid.upper()} ===")
print(f"Total patients processed: {len(stats_df)}")
print(f"Patients with valid gaze data: {len(stats_df[stats_df['data_quality_valid_samples'] > 0])}")
print(f"Patients with saccade data: {len(stats_df[stats_df['total_saccades'] > 0])}")
print(f"Patients with pupil data: {len(stats_df[stats_df['pupil_mean_diameter'].notna()])}")

# Data quality summary
if len(stats_df) > 0:
    print(f"\nData Quality Summary:")
    print(f"Mean valid data percentage: {stats_df['data_quality_valid_percentage'].mean():.1f}%")
    print(f"Patients with >80% valid data: {len(stats_df[stats_df['data_quality_valid_percentage'] > 80])}")
    print(f"Patients with >50% valid data: {len(stats_df[stats_df['data_quality_valid_percentage'] > 50])}")
    
    # Gaze speed summary
    valid_speed_data = stats_df[stats_df['gaze_speed_mean'].notna()]
    if len(valid_speed_data) > 0:
        print(f"\nGaze Speed Summary:")
        print(f"Mean gaze speed: {valid_speed_data['gaze_speed_mean'].mean():.1f} pixels/s")
        print(f"Max gaze speed: {valid_speed_data['gaze_speed_max'].max():.1f} pixels/s")
    
    # Saccade summary
    valid_saccade_data = stats_df[stats_df['saccade_rate'].notna()]
    if len(valid_saccade_data) > 0:
        print(f"\nSaccade Summary:")
        print(f"Mean saccade rate: {valid_saccade_data['saccade_rate'].mean():.2f} saccades/s")
        print(f"Mean saccade amplitude: {valid_saccade_data['saccade_amplitude_mean'].mean():.1f} pixels")

# Display first few rows of the DataFrame
print(f"\nFirst 5 patients statistics:")
print(stats_df.head().to_string())

# Save a summary report
summary_report = f"""
GAZE DATA QUALITY SUMMARY REPORT
================================
Video: {vid}
Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

PATIENT SUMMARY:
- Total patients processed: {len(stats_df)}
- Patients with valid gaze data: {len(stats_df[stats_df['data_quality_valid_samples'] > 0])}
- Patients with saccade data: {len(stats_df[stats_df['total_saccades'] > 0])}
- Patients with pupil data: {len(stats_df[stats_df['pupil_mean_diameter'].notna()])}

DATA QUALITY:
- Mean valid data percentage: {stats_df['data_quality_valid_percentage'].mean():.1f}%
- Patients with >80% valid data: {len(stats_df[stats_df['data_quality_valid_percentage'] > 80])}
- Patients with >50% valid data: {len(stats_df[stats_df['data_quality_valid_percentage'] > 50])}

RECOMMENDATIONS:
- Patients with >80% valid data are recommended for ISC analysis
- Patients with <50% valid data should be excluded
- Check coordinate ranges for unusual values (should be within screen bounds)
"""

with open(f'{results_dir}/{vid}_quality_summary.txt', 'w') as f:
    f.write(summary_report)

print(f"\nSummary report saved to {results_dir}/{vid}_quality_summary.txt") 

#%% Plot average gaze trajectory across all subjects
print("\n=== Plotting Average Gaze Trajectory ===")

# Collect all valid gaze data
all_gaze_data = []
all_t_gaze = []
valid_patients = []

for i, pat in enumerate(patients):
    if xy[i] is not None:  # Only include patients with valid data
        # Determine data source for this patient
        if pat in old_data_patients:
            current_data_dir = data_dir_old
            data_source = 'old'
        elif pat in new_data_patients:
            current_data_dir = data_dir
            data_source = 'new'
        else:
            current_data_dir = data_dir
            data_source = 'default'
        
        pat_dir = '{:s}/{:s}'.format(current_data_dir, pat)
        
        # Check if patient directory exists
        if not os.path.exists(pat_dir):
            continue
        
        eye_prep_dir = '{:s}/Eye_prep/'.format(pat_dir)
        if not os.path.exists(eye_prep_dir):
            continue
        
        et_files = os.listdir(eye_prep_dir)
        
        # Use appropriate file naming convention
        if data_source == 'old':
            et_file = [file for file in et_files if vid in file and 'et_prep.npz' in file and 'task-' in file]
        else:
            et_file = [file for file in et_files if vid in file and 'et_prep.npz' in file]
        
        if not et_file:
            continue
        
        et_file = et_file[0]
        et_data = np.load('{:s}/Eye_prep/{:s}'.format(pat_dir, et_file))
        
        # Get original gaze data
        xy_gaze = et_data['xy']
        t_gaze = et_data['t_gaze']
        
        # Remove NaN values
        valid_mask = ~(np.isnan(xy_gaze[:,0]) | np.isnan(xy_gaze[:,1]))
        if np.sum(valid_mask) > 0:
            xy_clean = xy_gaze[valid_mask]
            t_clean = t_gaze[valid_mask]
            
            all_gaze_data.append(xy_clean)
            all_t_gaze.append(t_clean)
            valid_patients.append(pat)

print(f"Collected gaze data from {len(valid_patients)} patients: {valid_patients}")

if len(all_gaze_data) > 0:
    # Find common time range
    t_min = max([t[0] for t in all_t_gaze])
    t_max = min([t[-1] for t in all_t_gaze])
    
    print(f"Common time range: {t_min:.1f}s to {t_max:.1f}s")
    
    # Resample all data to common time grid
    fs_common = 30  # 30 Hz for averaging
    t_common = np.arange(t_min, t_max, 1/fs_common)
    
    xy_resampled = []
    for xy, t in zip(all_gaze_data, all_t_gaze):
        # Interpolate to common time grid
        f_x = interpolate.interp1d(t, xy[:,0], fill_value='extrapolate')
        f_y = interpolate.interp1d(t, xy[:,1], fill_value='extrapolate')
        
        xy_resampled.append(np.column_stack([f_x(t_common), f_y(t_common)]))
    
    # Calculate average gaze trajectory
    xy_avg = np.mean(xy_resampled, axis=0)
    xy_std = np.std(xy_resampled, axis=0)
    
    # Plot average gaze trajectory
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Average gaze trajectory
    axes[0,0].plot(xy_avg[:,0], xy_avg[:,1], 'b-', linewidth=2, label='Average trajectory')
    axes[0,0].fill_between(xy_avg[:,0], xy_avg[:,1] - xy_std[:,1], 
                           xy_avg[:,1] + xy_std[:,1], alpha=0.3, color='blue', label='±1 STD')
    axes[0,0].set_xlabel('X position (pixels)')
    axes[0,0].set_ylabel('Y position (pixels)')
    axes[0,0].set_title(f'Average Gaze Trajectory - {vid}')
    axes[0,0].legend()
    axes[0,0].invert_yaxis()
    
    # Plot 2: X position over time
    axes[0,1].plot(t_common, xy_avg[:,0], 'r-', linewidth=2, label='Average X')
    axes[0,1].fill_between(t_common, xy_avg[:,0] - xy_std[:,0], 
                           xy_avg[:,0] + xy_std[:,0], alpha=0.3, color='red')
    axes[0,1].set_xlabel('Time (s)')
    axes[0,1].set_ylabel('X position (pixels)')
    axes[0,1].set_title(f'Average X Position Over Time - {vid}')
    axes[0,1].legend()
    
    # Plot 3: Y position over time
    axes[1,0].plot(t_common, xy_avg[:,1], 'g-', linewidth=2, label='Average Y')
    axes[1,0].fill_between(t_common, xy_avg[:,1] - xy_std[:,1], 
                           xy_avg[:,1] + xy_std[:,1], alpha=0.3, color='green')
    axes[1,0].set_xlabel('Time (s)')
    axes[1,0].set_ylabel('Y position (pixels)')
    axes[1,0].set_title(f'Average Y Position Over Time - {vid}')
    axes[1,0].legend()
    
    # Plot 4: Individual trajectories (first 10 patients)
    for i, (xy, pat) in enumerate(zip(xy_resampled[:10], valid_patients[:10])):
        alpha = 0.3 if i < 5 else 0.1  # Make first 5 more visible
        axes[1,1].plot(xy[:,0], xy[:,1], alpha=alpha, linewidth=1, label=pat if i < 3 else "")
    
    #axes[1,1].plot(xy_avg[:,0], xy_avg[:,1], 'k-', linewidth=3, label='Average')
    axes[1,1].set_xlabel('X position (pixels)')
    axes[1,1].set_ylabel('Y position (pixels)')
    axes[1,1].set_title(f'Individual vs Average Trajectories - {vid}')
    axes[1,1].legend()
    axes[1,1].invert_yaxis()
    
    plt.tight_layout()
    plt.show()
    
    # Save the average trajectory data
    np.savez(f'{results_dir}/{vid}_average_gaze_trajectory.npz',
             xy_avg=xy_avg, xy_std=xy_std, t_common=t_common,
             valid_patients=valid_patients, n_patients=len(valid_patients))
    
    print(f"Average gaze trajectory saved to {results_dir}/{vid}_average_gaze_trajectory.npz")
    print(f"Average trajectory statistics:")
    print(f"  X range: [{np.min(xy_avg[:,0]):.1f}, {np.max(xy_avg[:,0]):.1f}]")
    print(f"  Y range: [{np.min(xy_avg[:,1]):.1f}, {np.max(xy_avg[:,1]):.1f}]")
    print(f"  X std: {np.mean(xy_std[:,0]):.1f}")
    print(f"  Y std: {np.mean(xy_std[:,1]):.1f}")
else:
    print("No valid gaze data found for averaging")

#%% Implement ISC using the approach from gaze_isc.py
if compute_ISC:
    
    print("\n=== Computing ISC using established approach ===")
    print(f"ISC will be computed for {len(patients)} patients:")
    print(f"Old data patients for ISC: {[p for p in patients if p in old_data_patients]}")
    print(f"New data patients for ISC: {[p for p in patients if p in new_data_patients]}")
    
    # Parameters from gaze_isc_CC_mods.py
    fs_eye = 300
    dsf = 3  # Downsampling factor
    n_perm = 100  # Number of permutations
    window = 1000
    overlap = 750
    
    # Collect all data using the same approach as gaze_isc_CC_mods.py
    xy = [None] * len(patients)
    gaze_var = [None] * len(patients)
    xy_perm = [None] * len(patients)
    gaze_var_perm = [None] * len(patients)
    
    for i, pat in enumerate(patients):
        # Determine data source for this patient (same logic as above)
        if pat in old_data_patients:
            current_data_dir = data_dir_old
            data_source = 'old'
        elif pat in new_data_patients:
            current_data_dir = data_dir
            data_source = 'new'
        else:
            current_data_dir = data_dir
            data_source = 'default'
        
        pat_dir = '{:s}/{:s}'.format(current_data_dir, pat)
        
        # Check if patient directory exists in the specified data source
        if not os.path.exists(pat_dir):
            print(f"Patient directory {pat_dir} not found, skipping {pat}")
            continue
        
        # Check if Eye_prep directory exists
        eye_prep_dir = '{:s}/Eye_prep/'.format(pat_dir)
        if not os.path.exists(eye_prep_dir):
            print(f"Eye_prep directory not found for patient {pat}, skipping...")
            continue
        
        et_files = os.listdir(eye_prep_dir)
        et_file = [file for file in et_files if vid in file and 'et_prep.npz' in file]
        
        if not et_file:
            print(f"No matching et_prep.npz files found for patient {pat} and video {vid}")
            continue
        
        et_file = et_file[0]
        
        print('Loading data for patient {:s} ...'.format(pat))
        
        et_data = np.load('{:s}/Eye_prep/{:s}'.format(pat_dir, et_file))
        t = et_data['t_gaze']
        
        # Interpolate missing data
        x = interp_nans(et_data['xy'][:,0])
        y = interp_nans(et_data['xy'][:,1])
           
        xy_pat = np.vstack((x,y)).T
        
        # Interpolate gaps in time axis
        t_res = np.arange(t[0], t[-1], 1/fs_eye)
        f = interpolate.interp1d(t, xy_pat.T)
        xy_pat = f(t_res).T
        t = t_res
    
        # Compute gaze variation
        xy_pat -= np.mean(xy_pat, axis=0)
        
        env = np.abs(signal.hilbert(xy_pat, axis=0))
        gaze_var_pat = np.sqrt(np.sum(env**2, axis=1))
        
        # Cut around movie    
        idx_mov = np.logical_and(t >= et_data['t_pupil'][0], t < et_data['t_pupil'][-1])
        
        xy[i] = xy_pat[idx_mov,:]
        gaze_var[i] = gaze_var_pat[idx_mov]
        
        # Downsample
        xy[i] = signal.resample(xy[i], int(len(xy[i])/dsf))
        gaze_var[i] = signal.resample(gaze_var[i], int(len(gaze_var[i])/dsf))
        
        # Permute the time series
        xy_perm[i] = np.empty((xy[i].shape[0], xy[i].shape[1], n_perm))
        gaze_var_perm[i] = np.empty((len(gaze_var[i]), n_perm))
        
        idx_perm = [np.random.randint(len(xy[i])) for j in range(n_perm)]
        
        for p in range(n_perm):
           xy_perm[i][:,:,p] = np.vstack((xy[i][idx_perm[p]:, :], xy[i][:idx_perm[p], :]))
           gaze_var_perm[i][:,p] = np.hstack((gaze_var[i][idx_perm[p]:], gaze_var[i][:idx_perm[p]]))
    
    #%% Reorganize data
    idx_not_none = [c is not None for c in xy]
    
    patients = list(compress(patients, idx_not_none))
    xy = list(compress(xy, idx_not_none))
    gaze_var = list(compress(gaze_var, idx_not_none))
    
    xy_perm = list(compress(xy_perm, idx_not_none))
    gaze_var_perm = list(compress(gaze_var_perm, idx_not_none))
    
    len_min = np.min([len(c) for c in xy])
    xy = [c[:len_min,:] for c in xy]
    gaze_var = [c[:len_min] for c in gaze_var]
    
    xy_perm = [c[:len_min,:,:] for c in xy_perm]
    gaze_var_perm = [c[:len_min,:] for c in gaze_var_perm]
    
    xy = np.concatenate([np.expand_dims(c, 2) for c in xy], axis=2)
    gaze_var = np.concatenate([np.expand_dims(c, 1) for c in gaze_var], axis=1)
    
    xy_perm = np.concatenate([np.expand_dims(c, 3) for c in xy_perm], axis=3)
    gaze_var_perm = np.concatenate([np.expand_dims(c, 2) for c in gaze_var_perm], axis=2)
    
    #%% Compute ISC
    isc = compute_isc(xy)
    
    isc_perm = np.empty((n_perm, len(isc)))
    
    for p in range(n_perm):
        isc_perm[p,:] = compute_isc(xy_perm[:,:,p,:])
    
    #%% Gaze variation ISC
    corr = np.corrcoef(gaze_var.T)
    corr[np.eye(corr.shape[0]) == 1] = np.nan
    isc_gaze_var = np.nanmean(corr, axis=0)
    
    isc_gaze_perm = np.empty((n_perm, len(isc)))
    
    for p in range(n_perm):
        corr = np.corrcoef(gaze_var.T)
        corr[np.eye(corr.shape[0]) == 1] = np.nan
        isc_gaze_var = np.nanmean(corr, axis=0)
        
    #%% Slow fluctuations 
    sos = signal.butter(5, [0.05, 0.15], btype='bandpass', fs=fs_eye, output='sos')
    xy_slow = signal.sosfiltfilt(sos, xy, axis=0)
    
    isc_slow = compute_isc(xy_slow)
    
    #%% Time resolved ISC 
    # Gaze position
    isc_time_x, time_isc = time_resolved_isc(xy[:,0,:], (fs_eye/dsf), window, overlap)
    isc_time_y, _ = time_resolved_isc(xy[:,1,:], (fs_eye/dsf), window, overlap)
    
    isc_time_x_perm = np.empty(np.hstack((isc_time_x.shape, n_perm)))
    isc_time_y_perm = np.empty(np.hstack((isc_time_x.shape, n_perm)))
    
    print('Computing surrogate distribution of time resolved ISC (Gaze position)')
    
    for ip in tqdm(range(n_perm)):
        isc_time_x_perm[:,:,ip], _ = time_resolved_isc(xy_perm[:,0,ip,:], (fs_eye/dsf), window, overlap)
        isc_time_y_perm[:,:,ip], _ = time_resolved_isc(xy_perm[:,1,ip,:], (fs_eye/dsf), window, overlap)
        
    isc_time_gaze = np.mean(np.concatenate((np.expand_dims(isc_time_x,2), 
                                            np.expand_dims(isc_time_y,2)), 
                                           axis=2), 
                            axis=2)
    
    isc_time_gaze_perm = np.mean(np.concatenate((np.expand_dims(isc_time_x_perm,3), 
                                                 np.expand_dims(isc_time_y_perm,3)), 
                                                axis=3), 
                                 axis=3)
    
    #%% Gaze variation time resolved ISC
    isc_time_var, _ = time_resolved_isc(gaze_var, (fs_eye/dsf), window, overlap)
    
    isc_time_var_perm = np.empty(np.hstack((isc_time_x.shape, n_perm)))
    
    print('Computing surrogate distribution of time resolved ISC (Gaze variation)')
    
    for ip in tqdm(range(n_perm)):
        isc_time_var_perm[:,:,ip], _ = time_resolved_isc(gaze_var_perm[:,ip,:], (fs_eye/dsf), window, overlap)
        
    #%% Plots
    plt.rcParams.update({'font.size': 14})
    
    #%% Gaze position ISC per patients
    plot_isc_pat(isc, 'Gaze position ISC')
    
    #%% Permutations statistics
    plt.figure()
    plt.hist(np.mean(isc_perm, axis=1), color='tab:Gray', ec='k')
    ylim = plt.ylim()
    
    plt.plot([np.mean(isc), np.mean(isc)], ylim, 'r', linewidth=2)
    plt.ylim(ylim)
    
    plt.xlabel('ISC')
    plt.ylabel('# of permutations')
    plt.legend(['Original Data', 'Permutations'])
    plt.title('Permutation test')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    #%% ISC of slow fluctuations
    plot_isc_pat(isc_slow, 'ISC of slow fluctuations')
        
    #%% ISC of gaze variation
    plot_isc_pat(isc_gaze_var, 'Gaze Variation ISC')
    
    #%% Time resolved ISC
    plt.figure()
    plt.plot(time_isc, isc_time_gaze.T, color='tab:Gray')
    plt.plot(time_isc, np.mean(isc_time_gaze, axis=0), 'k', linewidth=2)
    
    plt.xlim([time_isc[0], time_isc[-1]])
    
    plt.xlabel('Time [s]')
    plt.ylabel('ISC')
    plt.title('Time resolved ISC of Gaze Position')
    plt.tight_layout()
    
    plt.grid()
    plt.tight_layout()
    
    plt.figure()
    plt.plot(time_isc, isc_time_var.T, color='tab:Gray')
    plt.plot(time_isc, np.mean(isc_time_var, axis=0), 'k', linewidth=2)
    
    plt.xlim([time_isc[0], time_isc[-1]])
    
    plt.xlabel('Time [s]')
    plt.ylabel('ISC')
    plt.title('Time resolved ISC of Gaze Variance')
    plt.tight_layout()
    
    plt.grid()
    plt.tight_layout()
    
    #%% Stats of time resolved ISC
    
    # Gaze position
    plt.figure()
    plt.plot(time_isc, np.mean(isc_time_gaze_perm, axis=0), color='tab:Gray')
    plt.plot(time_isc, np.mean(isc_time_gaze, axis=0), color='k', linewidth=2)
    
    plt.xlim([time_isc[0], time_isc[-1]])
    ylim = plt.ylim()
    plt.ylim([0, ylim[1]])
    
    plt.xlabel('Time [s]')
    plt.ylabel('ISC')
    plt.title('Mean ISC (Gaze position) with surrogate distribution')
    
    plt.grid()
    plt.tight_layout()
    
    # Gaze variation 
    plt.figure()
    plt.plot(time_isc, np.mean(isc_time_var_perm, axis=0), color='tab:Gray')
    plt.plot(time_isc, np.mean(isc_time_var, axis=0), color='k', linewidth=2)
    
    plt.xlim([time_isc[0], time_isc[-1]])
    ylim = plt.ylim()
    plt.ylim([0, ylim[1]])
    
    plt.xlabel('Time [s]')
    plt.ylabel('ISC')
    plt.title('Mean ISC (Gaze variation) with surrogate distribution')
    
    plt.grid()
    plt.tight_layout()
    
    #%% Save data
    
    # Create results directory if it doesn't exist
    results_dir = f'/Volumes/Samsung/Movie_data/ISC_{vid}_newsubs'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # ISC for whole recording
    np.savez(f'{results_dir}/{vid}_isc_598_win.npz', isc_gaze_pos=isc, 
             isc_gaze_pos_slow=isc_slow,
             isc_gaze_var=isc_gaze_var,
             isc_gaze_pos_perm=isc_perm,
             patients=patients,
             n_perm=n_perm,
             fs=fs_eye/dsf)
    
    # Gaze position
    np.savez(f'{results_dir}/{vid}_isc_gaze_position_time_598_win.npz', isc_time_gaze=isc_time_gaze, 
             isc_time_gaze_perm=isc_time_gaze_perm,
             time_isc=time_isc,
             patients=patients,
             window=window,
             overlap=overlap,
             n_perm=n_perm,
             fs=fs_eye/dsf)
    
    # Gaze variation
    np.savez(f'{results_dir}/{vid}_isc_gaze_variation_time_598_win.npz', isc_time_var=isc_time_var, 
             isc_time_var_perm=isc_time_var_perm,
             time_isc=time_isc,
             patients=patients,
             window=window,
             overlap=overlap,
             n_perm=n_perm,
             fs=fs_eye/dsf)
    
    print(f"\nISC analysis complete! Results saved to {results_dir}")
    print(f"Processed {len(patients)} patients: {patients}")
    print(f"Mean ISC (gaze position): {np.mean(isc):.3f}")
    print(f"Mean ISC (slow fluctuations): {np.mean(isc_slow):.3f}")
    print(f"Mean ISC (gaze variation): {np.mean(isc_gaze_var):.3f}") 
