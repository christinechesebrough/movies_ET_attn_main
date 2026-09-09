#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 13:22:22 2024

Extract descriptive statistics for eye movement measures and EEG power for Movies data

@author: christinechesebrough
"""

from scipy.stats import pearsonr
import os
from itertools import compress
import numpy as np
import pandas as pd
from scipy import stats, signal, interpolate, correlate
import matplotlib.pyplot as plt
import mne
import seaborn as sns


drive = 'Samsung'
#vid ='inscapes'
vid = 'inscapes'
ref = 'avg'
freq_band = 'HFA'
region = 'all'
#range(8,13)

data_dir = f'/Volumes/{drive}/Movie_data/movies_prep_standard'
isc_dir = f'/Volumes/{drive}/Movie_data/data/isc'
mne_data_dir = f'/Volumes/{drive}/Movie_data/movies_prep_standard'
elec_dir = f'/Volumes/{drive}/Movie_data/data/electrode_localization'
movie_subs_table = pd.read_excel(f'/Volumes/{drive}/Movie_data/data/movie_subs_table.xlsx')

fig_dir = f'/Volumes/{drive}/Movie_data/ET_descriptors_{vid}_{freq_band}_{region}'

if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

if vid == 'despicable_me_english':
    good_ET_pats = ['NS127_02','NS135','NS136','NS137','NS138','NS140','NS153','NS154','NS164','NS166','NS174_02']
elif vid == 'inscapes':
    good_ET_pats = ['NS127_02','NS135','NS137','NS138','NS140','NS140_02','NS154','NS164']
    

patients = ['NS127_02','NS135','NS136','NS137','NS138','NS140','NS140_02', 'NS154','NS164']


#patients = good_ET_pats
patients.sort()

fs_eye = 300

#%% Load ISC data 
if vid =='despicable_me_english':
    data = np.load(isc_dir + '/isc_gaze_position_time.npz')  
if vid == 'inscapes':
    data = np.load(isc_dir+'/inscapes_isc_gaze_position_time.npz')

#is time_isc time or isc...
time_isc = data['time_isc']
patients_isc = data['patients']
isc_time = data['isc_time_gaze']

fs_isc = 1 / np.mean(np.diff(time_isc))   

patients_isc = data['patients']

#%%
# Modified Code for Appending Statistics for Each Patient to DataFrame

# Define a DataFrame to store all patient data

all_subs_data = pd.DataFrame(columns=['Patient', 'Total_Saccades', 'Mean_Saccades_Rolling', 'Std_Saccades_Rolling',
                                      'Mean_Vergence', 'Mean_Vergence_Rolling', 'Std_Vergence_Rolling',
                                      'Mean_ISC', 'Std_ISC'])

for pat in patients:
    pat_dir = os.path.join(data_dir, pat)
    fig_patient_dir = os.path.join(fig_dir, pat)
    if not os.path.exists(fig_patient_dir):
        os.makedirs(fig_patient_dir)
        
    idx_pat = np.in1d(patients_isc, pat)
    isc_pat = isc_time[idx_pat]

    pat_dir = '{:s}/{:s}'.format(data_dir, pat)
    verg_files = os.listdir('{:s}/Eye_prep/'.format(pat_dir))
    if vid == 'despicable_me_english':
        verg_file = [file for file in verg_files if f'{vid}_run-1_et_prep.csv' in file][0]
    elif vid == 'inscapes':
        verg_file = [file for file in verg_files if f'{vid}_et_prep.csv' in file][0]

    print('Loading data for patient {:s} ...'.format(pat))
    
    verg_dat = pd.read_csv('{:s}/Eye_prep/{:s}'.format(pat_dir, verg_file))
    vis_fd = verg_dat['vis_fd_interp'].values
    gaze_disp = verg_dat['dva_gaze_disp_x_interp']
    verg_t = verg_dat['time'].values
    
    t_verg = verg_t
    vergence = vis_fd

    t_start_res, t_end_res = t_verg[0], t_verg[-1]
    t_res = np.arange(t_start_res, t_end_res, 1/fs_eye)
    
    f = interpolate.interp1d(t_verg, vergence, bounds_error=False, fill_value="extrapolate")
    vergence = f(t_res)
    t_verg = t_res

    eye_pat_dir = '{:s}/{:s}/Eye_prep'.format(mne_data_dir, pat)
    eye_files = os.listdir(eye_pat_dir)
    eye_files = list(compress(eye_files, ['_et_prep' in f for f in eye_files]))
    eye_list = list(compress(eye_files, [vid in f for f in eye_files]))
    
    eye_file = eye_list[0]
    eye_data = np.load('{:s}/{:s}'.format(eye_pat_dir, eye_file))
    t_start = eye_data['t_pupil'][0]
    t_end = eye_data['t_pupil'][-1]
    print(t_start)
    print(t_end)
    
    idx_vid = np.logical_and(t_verg > t_start, t_verg < t_end)
    
    t_verg = t_verg[idx_vid]
    vergence = vergence[idx_vid]
    
    t_verg = t_verg - t_verg[0]
    
    window_length = 5
    step_size = 2.5
    
    total_time = t_verg[-1] - t_verg[0]
    num_steps = int(np.floor((total_time - window_length) / step_size) + 1)
    
    verg_sliding = np.zeros(num_steps)
    
    t_verg_sliding = np.linspace(window_length / 2, total_time - window_length / 2, num=num_steps)
    
    for i in range(num_steps):
        window_start = i * step_size
        window_end = window_start + window_length
        verg_sliding[i] = np.mean(vergence[(t_verg >= window_start) & (t_verg < window_end)])
    
    f_verg_sliding = interpolate.interp1d(t_verg_sliding, verg_sliding, kind='linear', fill_value="extrapolate")
    verg_interpolated_sliding = f_verg_sliding(time_isc)
    
    pat_dir = '{:s}/{:s}'.format(data_dir, pat)
    
    et_files = os.listdir('{:s}/Eye_prep/'.format(pat_dir))
    et_file = [file for file in et_files if f'task-{vid}_run-1_et_prep.npz' in file][0]

    print('Loading data for patient {:s} ...'.format(pat))

    et_data = np.load('{:s}/Eye_prep/{:s}'.format(pat_dir, et_file))
    
    t = et_data['t_gaze']

    idx_vid = np.logical_and(t > t_start, t < t_end)
    t = t[idx_vid]
    t -= t[0]
    
    saccade_onset_t = et_data['saccade_onset_t']
    
    is_saccade_onset = np.zeros(len(t), dtype=bool) 
    
    for onset_time in saccade_onset_t:
        if onset_time >= t_start and onset_time <= t_end:
            idx = np.argmin(np.abs(t - (onset_time - t_start)))
            is_saccade_onset[idx] = True
    
    saccade_onset_int = is_saccade_onset.astype(int)
    
    total_saccades = np.sum(saccade_onset_int)

    window_length = 5
    step_size = 2.5
    
    total_time = t[-1] - t[0]
    num_steps = int(np.floor((total_time - window_length) / step_size) + 1)
    
    saccade_rate_sliding = np.zeros(num_steps)
    
    t_sacc_rate_sliding = np.linspace(window_length / 2, total_time - window_length / 2, num=num_steps)
    
    for i in range(num_steps):
        window_start = i * step_size
        window_end = window_start + window_length
        saccade_rate_sliding[i] = np.sum(is_saccade_onset[(t >= window_start) & (t < window_end)])
    
    f_sacc_rate_sliding = interpolate.interp1d(t_sacc_rate_sliding, saccade_rate_sliding, kind='linear', fill_value="extrapolate")
    saccade_rate_interpolated_sliding = f_sacc_rate_sliding(time_isc)
        
    isc_pat_test = isc_pat.T
    isc_pat_test = np.squeeze(isc_pat_test)
    
    mean_saccades_rolling = np.mean(saccade_rate_interpolated_sliding)
    std_saccades_rolling = np.std(saccade_rate_interpolated_sliding)
    
    
    mean_vergence = np.mean(vergence)
    mean_vergence_rolling = np.mean(verg_interpolated_sliding)
    std_vergence_rolling = np.std(verg_interpolated_sliding)

    # Compute ISC metrics
    mean_isc = np.mean(isc_pat)
    std_isc = np.std(isc_pat)
    isc_pat = isc_pat.T

    # Append data for the current patient to DataFrame
    new_row = {        
        'Movie': vid,
        'Patient': pat,
        f'Total_Saccades_{vid}': total_saccades,
        f'Mean_Saccades_Rolling_{vid}': mean_saccades_rolling,
        f'Std_Saccades_Rolling_{vid}': std_saccades_rolling,
        f'Mean_Vergence_{vid}': mean_vergence,
        f'Mean_Vergence_Rolling_{vid}': mean_vergence_rolling,
        f'Std_Vergence_Rolling_{vid}': std_vergence_rolling,
        f'Mean_ISC_{vid}': mean_isc,
        f'Std_ISC_{vid}': std_isc
    }
    
    all_subs_data = pd.concat([all_subs_data, pd.DataFrame([new_row])], ignore_index=True)

    # --- Generate the figure for the patient ---
    plt.figure(figsize=(10, 6))

    # Plot Saccade Rate over Time
    plt.plot(t_sacc_rate_sliding, saccade_rate_sliding, label='Saccade Rate', color='blue')

    # Plot ISC over Time
    plt.plot(time_isc, isc_pat*10, label='ISC', color='green')

    # Plot Vergence over Time
    plt.plot(time_isc, verg_interpolated_sliding, label='Vergence', color='red')

    # Labels and title
    plt.title(f'Saccade Rate, ISC, and Vergence Over Time for {pat}')
    plt.xlabel('Time (s)')
    plt.ylabel('Rate / ISC / Vergence')
    plt.legend()

    # Save the figure to the patient-specific directory
    fig_filename = os.path.join(fig_patient_dir, f'{pat}_saccade_isc_vergence_plot.png')
    plt.savefig(fig_filename)
    plt.close()
    
    inscapes_all_subs_data = all_subs_data 

        
# Save the final DataFrame to a CSV file
all_subs_filename = os.path.join(fig_dir, f'all_subs_{vid}_{region}_{freq_band}_featureDat_wLabels.csv')
all_subs_data.to_csv(all_subs_filename, header=True, index=False)


#%%
import pandas as pd
from scipy import stats

# Assuming inscapes_all_subs_data and desp_me_all_subs_data are your two dataframes

# Step 1: Merge the dataframes on the index (assuming rows are in the same order and correspond to the same patients)
merged_data = pd.merge(inscapes_all_subs_data, desp_me_all_subs_data, left_index=True, right_index=True)
merged_data = merged_data.dropna(axis = 1)
merged_data_filename = os.path.join(fig_dir, f'all_subs_bothVids_{region}_ETdat_wLabels.csv')
merged_data.to_csv(merged_data_filename, header=True, index=False)
#%%
# Load data
merged_data = pd.read_csv('/Volumes/Samsung/Movie_data/ET_descriptors_inscapes_HFA_all/all_subs_bothVids_all_ETdat_wLabels.csv')

# Columns to compare
columns_to_compare = ['Total_Saccades', 'Mean_Vergence', 'Mean_ISC']

# Prepare to store the results of the paired t-tests and plot data
t_values = []
p_values = []
column_names = []
mean_inscapes = []
mean_desp_me = []
sem_inscapes = []
sem_desp_me = []

# Number of columns to compare (for subplots)
num_columns = len(columns_to_compare)

# Create a figure with subplots
fig, axes = plt.subplots(1, num_columns, figsize=(2 * num_columns, 3))

for idx, col in enumerate(columns_to_compare):
    # Perform paired t-test between the corresponding columns in the two data frames
    t_stat, p_val = stats.ttest_rel(merged_data[f'{col}_despicable_me_english'], merged_data[f'{col}_inscapes'])
    
    # Calculate means and standard errors
    mean_inscape = np.mean(merged_data[f'{col}_inscapes'])
    mean_desp = np.mean(merged_data[f'{col}_despicable_me_english'])
    sem_inscape = stats.sem(merged_data[f'{col}_inscapes'])
    sem_desp = stats.sem(merged_data[f'{col}_despicable_me_english'])
    
    # Store results
    t_values.append(t_stat)
    p_values.append(p_val)
    column_names.append(col)
    mean_inscapes.append(mean_inscape)
    mean_desp_me.append(mean_desp)
    sem_inscapes.append(sem_inscape)
    sem_desp_me.append(sem_desp)

    # Plotting the means with error bars (Standard Error) on the corresponding axis
    axes[idx].bar(['Narrative','Ambient'], [mean_desp, mean_inscape], 
                  yerr=[sem_desp,sem_inscape], capsize=3, 
                  color=['orange','purple'], alpha=.7)
    
    # Add titles and labels
    axes[idx].set_title(f'{col}\nT-stat: {t_stat:.2f}, P-val: {p_val:.3e}', fontsize=8)
    axes[idx].set_ylabel(f'Mean of {col}')
    
    # Adjust the x-axis label size
    axes[idx].tick_params(axis='x', labelsize=8)  # Adjust font size here

# Adjust layout
plt.tight_layout()

# Save the figure
fig_filename = os.path.join(fig_dir, 'summary_compare_ET_bw_movies.png')
plt.savefig(fig_filename, dpi=300)
plt.show()
