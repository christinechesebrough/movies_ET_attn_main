#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 18:55:15 2025

@author: christinechesebrough

"""

import os
from itertools import compress
import numpy as np
import pandas as pd
from scipy import stats, signal, interpolate, correlate
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA
from sklearn.decomposition import FactorAnalysis
from factor_analyzer import FactorAnalyzer


drive = 'Samsung'
vid = 'despicable_me_english'

    
#data_dir = f'/Volumes/{drive}/Movie_data/movies_prep_standard'
data_dir = '/Volumes/Samsung/Movie_data/movies_new_prep'
isc_dir = f'/Volumes/{drive}/Movie_data/data/isc'
elec_dir = f'/Volumes/{drive}/Movie_data/data/electrode_localization'
movie_subs_table = pd.read_excel(f'/Volumes/{drive}/Movie_data/data/movie_subs_table.xlsx')

fig_dir = f'/Volumes/{drive}/Movie_data/saccade_new_test_1Jul25'
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

patients =os.listdir(data_dir) 
patients.sort()

fs_eye = 300

# Load ISC data 
if vid =='despicable_me_english':
    data = np.load(isc_dir + '/isc_gaze_position_time.npz')  
if vid == 'inscapes':
    data = np.load('/Volumes/Samsung/Movie_data/ISC_inscapes_new_2/inscapes_isc_gaze_position_time.npz')

time_isc = data['time_isc']
patients_isc = data['patients']
isc_time = data['isc_time_gaze']
fs_isc = 1 / np.mean(np.diff(time_isc))   
patients_isc = data['patients']

compute_isc = False
compute_blinks = False
compute_verg = False
compute_pupil = False
compute_sacc = True
compute_rolling = False

#%% Loop to calculate and collect eye measures

# Load gaze, pupil, and oculometric data
for pat in patients:
    pat_dir = os.path.join(data_dir, pat)
    fig_patient_dir = os.path.join(fig_dir, pat)
    if not os.path.exists(fig_patient_dir):
        os.makedirs(fig_patient_dir)
        
    print('Loading data for patient {:s} ...'.format(pat))
    
    if compute_isc:
        #LOAD ISC
        idx_pat = np.in1d(patients_isc, pat)
        isc_pat = isc_time[idx_pat]
        # ISC is already calculated with gaze data cut around the movie times, derived from pupil data
        isc_pat_t = isc_pat.T #transpose to fit the dims of other data
        isc_pat_t = np.squeeze(isc_pat_t) 

    ### LOAD PUPIL DATA
    eye_pat_dir = '{:s}/{:s}/Eye_prep'.format(data_dir, pat)
    eye_files = os.listdir(eye_pat_dir)
    eye_files = list(compress(eye_files, ['_et_prep' in f for f in eye_files]))
    eye_list = list(compress(eye_files, [vid in f for f in eye_files]))
    eye_file = eye_list[0]
    eye_data = np.load('{:s}/{:s}'.format(eye_pat_dir, eye_file))
    eye_file = eye_list[0]
    eye_data = np.load('{:s}/{:s}'.format(eye_pat_dir, eye_file))
    
    # movie start and end times because t_pupil is cut around movie times
    # note that t_pupil is a different sampling rate as the other eye data
    t_start = eye_data['t_pupil'][0]
    t_end = eye_data['t_pupil'][-1]
    t_pupil = eye_data['t_pupil']
    pupil = eye_data['pupil']
    
    if compute_verg:
        ### PREPARE VERGENCE TIME SERIES ####
        pat_dir = '{:s}/{:s}'.format(data_dir, pat)
        verg_files = os.listdir('{:s}/Eye_prep/'.format(pat_dir))
        if vid == 'despicable_me_english':
            verg_file = [file for file in verg_files if f'{vid}_run-1_et_prep.csv' in file][0]
        elif vid == 'inscapes':
            verg_file = [file for file in verg_files if f'{vid}_et_prep.csv' in file][0]
    
        #extract patient vergence data
        verg_dat = pd.read_csv('{:s}/Eye_prep/{:s}'.format(pat_dir, verg_file))
        vis_fd = verg_dat['vis_fd_interp'].values
        gaze_disp = verg_dat['dva_gaze_disp_x_interp']
        verg_t = verg_dat['time'].values
        
        # vergence time
        t_verg = verg_t
        # vergence time series  
        vergence = vis_fd
        
        # Add debug prints after loading data
        print(f"\n=== Debug Info for Patient {pat} ===")
        print(f"Original vergence length: {len(vergence)}")
        print(f"Original t_verg length: {len(t_verg)}")
        print(f"t_start: {t_start:.3f}")
        print(f"t_end: {t_end:.3f}")
        
        # After cutting the data
        print(f"Vergence length after cut: {len(vergence)}")
        print(f"t_verg length after cut: {len(t_verg)}")
    
    
    if compute_blinks:
        ## EXTRACT BLINK RATE
        blink_files = os.listdir(eye_pat_dir)
        blink_list = list(compress(blink_files, [f'{vid}_blink_events.csv' in f for f in blink_files]))
        blink_file = blink_list[0]
        blink_data = pd.read_csv('{:s}/{:s}'.format(eye_pat_dir, blink_file))
    
        #the times for all blink data are on the same clock as gaze data
        blink_onset = blink_data['start_time'] 
        blink_end = blink_data['end_time']
        blink_duration = blink_data['duration']
            
        ### CUT ALL GAZE DATA AROUND MOVIE START AND END TIMES
        # boolean series to cut off beginning and end of gaze time series based on movie start and end times
        idx_vid = np.logical_and(t_verg > t_start, t_verg < t_end)
        
        #truncating vergence time accordingly
        t_verg = t_verg[idx_vid]
        
        #truncating vergence accordingly
        vergence = vergence[idx_vid]
        
        # this step makes the first value of t_verg (vergence timing) 0
        t_verg = t_verg - t_verg[0]
        
        # do the same for saccade timing and gaze data 
        idx_vid = np.logical_and(t > t_start, t < t_end)
    
        t = t[idx_vid]
        xy = xy[idx_vid]
        
        #zeroes out t_verg
        t = t-t[0]
        
        if np.allclose(t, t_verg, atol=1e-6, rtol=1e-5):
            print("gaze timings are identical within the specified tolerance")
        else:
            print("gaze timings are not identical, please check!")
            
        ## remove t_start from all of the timing values in saccade_onset and fixation onset timings
        saccade_onset_t = saccade_onset_t - t_start
        fixation_t = fixation_t - t_start
        
        # remove t_start from the blink onsets and ends 
        blink_onset = blink_onset - t_start
        blink_end = blink_end - t_start
        
        #and from t_pupil
        t_pupil = t_pupil - t_start
    
    #
    if compute_verg:
        ############ Calculate rolling vergence rate #############
        
        fs_eye = len(t)/(t[-1]-t[0])
        
        # Parameters for windowing (in samples)
        window_samples = 10*fs_eye  # 10 seconds
        overlap_samples = 7.5*fs_eye  # 7.5 seconds 
        step_size_samples = window_samples - overlap_samples
        
        # Compute the total number of windows
        num_steps = int(np.floor((len(vergence) - window_samples) / step_size_samples) + 1)
        
        # Initialize array for sliding averages and variation
        verg_sliding = np.zeros(num_steps)
        verg_sliding_std = np.zeros(num_steps)  # Rolling variation of vergence rate (standard deviation)
    
        # Time array for sliding window midpoints (aligned to time_isc)
        isc_aligned_time = np.arange(window_samples / 2, len(vergence) - window_samples / 2 + 1, step_size_samples) / 300  # Convert to seconds
        
        # Compute rolling averages
        for i in range(num_steps):
            window_start_idx = int(i * step_size_samples)  # Ensure integer index
            window_end_idx = int(window_start_idx + window_samples)  # Ensure integer index
        
            # Extract data within the current window
            window_data = vergence[window_start_idx:window_end_idx]
        
            # Calculate mean vergence within the window
            verg_sliding[i] = np.mean(window_data)
           
            # Calculate variation (standard deviation) within the window
            verg_sliding_std[i] = np.std(window_data)
    
        abs_sliding_vergence = np.abs(verg_sliding)
    
        # Verify alignment
        print(f"Number of windows: {num_steps}")
        print(f"Length of verg_sliding: {len(verg_sliding)}")
        print(f"Length of isc_aligned_time: {len(isc_aligned_time)}")
        
        # After calculating window parameters
        print(f"\nWindow calculations:")
        print(f"fs_eye: {fs_eye:.2f}")
        print(f"window_samples: {window_samples:.2f}")
        print(f"step_size_samples: {step_size_samples:.2f}")
        print(f"num_steps: {num_steps}")
        
        # After computing sliding windows
        print(f"\nSliding window results:")
        print(f"verg_sliding length: {len(verg_sliding)}")
        print(f"saccade_rate_sliding length: {len(saccade_rate_sliding)}")
        print(f"rolling_dispersion length: {len(rolling_dispersion)}")
        print(f"rolling_blink_rate length: {len(rolling_blink_rate)}")
        print("=" * 50)
    
    
    ########### Calculate saccades, fixations, and dispersions ###############
    
    if compute_sacc:
        #### PREPARE SACCADE TIME SERIES ####
        pat_dir = '{:s}/{:s}'.format(data_dir, pat)
        et_files = os.listdir('{:s}/Eye_prep/'.format(pat_dir))
        et_file = [file for file in et_files if f'et_prep.npz' in file][0]
        et_data_path = '{:s}/Eye_prep/{:s}'.format(pat_dir, et_file)
        et_data = np.load(et_data_path)

        et_data_dict = dict(et_data)

        # saccade (gaze)time (should be the same as t_verg)
        t = et_data['t_gaze']
        
        #load saccade onset timings
        saccade_onset_t = et_data['saccade_onset_t']
    
        #load fixation onset timings (should be the same length as saccade onset timings)
        fixation_t = et_data['fixation_t']
        
        #load gaze data
        xy = et_data['xy']
        
        # check that the gaze timings for saccade and vergence are identical
        if compute_verg and compute_sacc:
            if np.allclose(t, t_verg, atol=1e-6, rtol=1e-5):
                print("gaze timings are identical within the specified tolerance")
            else:
                print("gaze timings are not identical, please check!")

        # here, t is zeroed out, and saccade_onset_t has had t_start subtracted from it, so it should be aligned   
        is_saccade_onset = np.zeros(len(t), dtype=bool) 
        
        for onset_time in saccade_onset_t:
            idx = np.argmin(np.abs(t - (onset_time)))
            is_saccade_onset[idx] = True
        
        saccade_onset_int = is_saccade_onset.astype(int)
        
        is_fixation_onset = np.zeros(len(t), dtype=bool) 
        for onset_time in fixation_t:
            idx = np.argmin(np.abs(t - (onset_time)))
            is_fixation_onset[idx] = True
        
        fixation_int = is_fixation_onset.astype(int)

                
        if compute_rolling:
            saccade_rate_sliding = np.zeros(num_steps)
        # Compute rolling averages
            for i in range(num_steps):
                window_start_idx = int(i * step_size_samples)  # Ensure integer index
                window_end_idx = int(window_start_idx + window_samples)  # Ensure integer index
            
                # Calculate mean vergence within the window
                saccade_rate_sliding[i] = np.sum(saccade_onset_int[window_start_idx:window_end_idx])
        
            # Verify alignment
            print(f"Length of saccade_rate_sliding: {len(saccade_rate_sliding)}")    
        
        ############### Calculate saccade dispersions based on gaze and fixation timings  ################
        
        # Ensure saccade_onset_t and fixation_t have the same length
        assert len(saccade_onset_t) == len(fixation_t), "Mismatch in number of saccades and fixations"
        
        #Generate data frame of matched saccade and fixation onsets and the euclidean distance between the two
        # Initialize list for saccade dispersions
        saccade_dispersions = []
        
        # Loop through each saccade and its corresponding fixation
        for saccade_time, fixation_time in zip(saccade_onset_t, fixation_t):
            # Get gaze position at the saccade onset
            saccade_idx = np.argmin(np.abs(t - saccade_time))  # Closest index to saccade time
            saccade_pos = xy[saccade_idx]
        
            # Get gaze position at the fixation onset
            fixation_idx = np.argmin(np.abs(t - fixation_time))  # Closest index to fixation time
            fixation_pos = xy[fixation_idx]
        
            # Check for NaN values in either position
            if np.isnan(saccade_pos).any() or np.isnan(fixation_pos).any():
                dispersion = np.nan  # Assign NaN for invalid dispersions
            else:
                # Calculate Euclidean distance (dispersion)
                dispersion = np.linalg.norm(saccade_pos - fixation_pos)
        
            # Store the result
            saccade_dispersions.append({
                "saccade_onset_time": saccade_time,
                "fixation_onset_time": fixation_time,
                "dispersion": dispersion
            })
        
        # Convert results to a DataFrame
        saccade_dispersions_df = pd.DataFrame(saccade_dispersions)
        
        
        saccade_dispersions_dist = []
        
        saccade_positions = et_data['saccade_pos']

        for row in saccade_positions:
            # Extract first and second xy pairs from this row
            xy1 = row[0:2]
            xy2 = row[2:4]
            
            # Calculate Euclidean distance
            distance = np.linalg.norm(xy1 - xy2)
            
            # Append to list
            saccade_dispersions_dist.append(distance)
        
        if compute_rolling:
            
            ### Calculate  rolling average of saccade dispersions within the same windows ###
            
            # Initialize aligned array with NaNs for alignment with `t`
            saccade_dispersion_aligned = np.full_like(t, np.nan, dtype=float)
            
            # Assign dispersion values to their corresponding indices in `t`
            for _, row in saccade_dispersions_df.iterrows():
                saccade_time = row["saccade_onset_time"]
                dispersion = row["dispersion"]
                idx = np.argmin(np.abs(t - saccade_time))  # Find the closest index in `t`
                saccade_dispersion_aligned[idx] = dispersion
            
            # Initialize array for rolling dispersion averages
            rolling_dispersion = np.zeros(num_steps)
            
            # Initialize array for rolling dispersion std
            rolling_dispersion_std = np.zeros(num_steps)
            
            # Compute rolling averages of dispersions
            for i in range(num_steps):
                window_start_idx = int(i * step_size_samples)  # Start of the rolling window
                window_end_idx = int(window_start_idx + window_samples)  # End of the rolling window
            
                # Extract values within the window
                window_values = saccade_dispersion_aligned[window_start_idx:window_end_idx]
            
                # Compute the mean, ignoring NaNs
                rolling_dispersion[i] = np.nanmean(window_values)
            
            # Compute rolling std of dispersions
            for i in range(num_steps):
                window_start_idx = int(i * step_size_samples)  # Start of the rolling window
                window_end_idx = int(window_start_idx + window_samples)  # End of the rolling window
            
                # Extract values within the window
                window_values = saccade_dispersion_aligned[window_start_idx:window_end_idx]
            
                # Compute the mean, ignoring NaNs
                rolling_dispersion_std[i] = np.nanstd(window_values)
            
            # Check for consistency
            print(f"Rolling dispersion calculated over {num_steps} windows.")


    if compute_blinks:

        ######### CALCULATE BLINK RATE AND DURATION OVER SAME TIME PERIODS #############
        
        # Initialize a boolean array for blink onsets
        is_blink_onset = np.zeros(len(t), dtype=bool)
        
        # Map blink onsets to the time `t`
        for blink_time in blink_onset:
            idx = np.argmin(np.abs(t - blink_time))  # Find closest index in `t`
            is_blink_onset[idx] = True
        
        # Convert to integer array (1 for blink onset, 0 otherwise)
        blink_onset_int = is_blink_onset.astype(int)
        
        # Initialize array for rolling blink rate
        rolling_blink_rate = np.zeros(num_steps)
        
        # Compute rolling blink rate
        for i in range(num_steps):
            window_start_idx = int(i * step_size_samples)  # Start of the rolling window
            window_end_idx = int(window_start_idx + window_samples)  # End of the rolling window
        
            # Calculate the number of blink onsets within the window
            rolling_blink_rate[i] = np.sum(blink_onset_int[window_start_idx:window_end_idx])
        
        # Verify alignment
        print(f"Length of rolling blink rate: {len(rolling_blink_rate)}")
        
        # Initialize aligned array for blink durations
        blink_duration_aligned = np.full_like(t, np.nan, dtype=float)
        
        # Map blink durations to the corresponding indices in `t`
        for blink_time, duration in zip(blink_onset, blink_duration):
            idx = np.argmin(np.abs(t - blink_time))  # Find closest index in `t`
            blink_duration_aligned[idx] = duration  # Assign duration to the corresponding index
        
        # Initialize array for rolling blink duration
        rolling_blink_duration = np.zeros(num_steps)
        
        # Compute rolling average of blink durations
        for i in range(num_steps):
            window_start_idx = int(i * step_size_samples)  # Start of the rolling window
            window_end_idx = int(window_start_idx + window_samples)  # End of the rolling window
        
            # Extract blink durations within the window
            window_values = blink_duration_aligned[window_start_idx:window_end_idx]
        
            # Compute the mean, ignoring NaNs
            rolling_blink_duration[i] = np.nanmean(window_values)
        
        # Verify results
        print(f"Length of rolling blink duration: {len(rolling_blink_duration)}")
        #
    
    if compute_pupil:
       ############# Calculate average pupil size and pupil size variance over the same rolling windows ##########
       # time stamps of t_pupil are not the same clock or sampling rate as
       # t, but that the timeseries reflects the same time period as the
       # previous time series (e.g. vergence, saccade rate, etc.)
           
        # Sampling rate for pupil data
        fs_pupil = len(pupil) / (t_end - t_start)
        
        # Determine the overlapping time range
        t_common_start = max(t[0], t_pupil[0])
        t_common_end = min(t[-1], t_pupil[-1])
        
        # Target number of steps (matching isc_aligned_time)
        target_num_steps = len(isc_aligned_time)
        
        # Calculate total duration of the pupil data in samples
        total_samples = len(pupil)
        
        # Compute window and step sizes to achieve the target number of steps
        step_size_samples_pupil = total_samples / (target_num_steps + 1)  # Approximate step size
        window_samples_pupil = 2 * step_size_samples_pupil  # Ensure overlap of ~50%
        
        # Convert to integers
        step_size_samples_pupil = int(step_size_samples_pupil)
        window_samples_pupil = int(window_samples_pupil)
        
        # Recalculate the number of steps to ensure alignment
        num_steps_pupil = target_num_steps
        
        # Initialize arrays for rolling metrics
        rolling_pupil_avg = np.zeros(num_steps_pupil)
        rolling_pupil_std = np.zeros(num_steps_pupil)
        
        # Loop through rolling windows
        for i in range(num_steps_pupil):
            window_start_idx = int(i * step_size_samples_pupil)
            window_end_idx = int(window_start_idx + window_samples_pupil)
            
            # Ensure the window does not exceed the data length
            if window_end_idx > total_samples:
                window_end_idx = total_samples
            
            # Calculate mean pupil dilation within the window
            rolling_pupil_avg[i] = np.mean(pupil[window_start_idx:window_end_idx])
            
            # Calculate pupil variation during the window
            rolling_pupil_std[i] = np.std(pupil[window_start_idx:window_end_idx])
    

        
    et_file_new = et_file.replace('.npz', '_updated.npz')
    et_data_new_path = '{:s}/Eye_prep/{:s}'.format(pat_dir, et_file_new)
    
    et_data_dict = dict(et_data)
    et_data_dict['fixation_int'] = fixation_int
    et_data_dict['saccade_onset_int'] = saccade_onset_int
    et_data_dict['saccade_dispersion_dist'] = saccade_dispersions_dist

    np.savez(et_data_new_path, **et_data_dict)
    
    print(f"Saved updated et_data to {et_data_new_path}")

    

