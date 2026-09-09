#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 18:55:15 2025

@author: christinechesebrough

"""

import os
from itertools import compress
import glob

import numpy as np
import pandas as pd
from scipy import stats, signal, interpolate
from scipy.signal import correlate

from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA
from sklearn.decomposition import FactorAnalysis
import math
#from factor_analyzer import FactorAnalyzer


drive = 'Samsung'
vid = "inscapes" #"despicable_me_english" #'dme'

compute_measures = True
#'despicable_me_english'  # inscapes

# Check if final_eye_df exists in memory, if so delete
if 'final_eye_df' in locals():
    del final_eye_df
    
# Check if normed_eye_df exists in memory, if so delete
if 'normed_eye_df' in locals():
    del normed_eye_df

    
region = 'all'
data_dir = f'/Volumes/{drive}/Movie_data/movies_prep_standard'
isc_dir = f'/Volumes/{drive}/Movie_data/data/isc'
mne_data_dir = f'/Volumes/{drive}/Movie_data/movies_prep_standard'
elec_dir = f'/Volumes/{drive}/Movie_data/data/electrode_localization'
movie_subs_table = pd.read_csv('/Volumes/Samsung/anatomy/shared_correspondence/movie_subs_master_updated.csv')

fig_dir = f'/Volumes/{drive}/Movie_data/more_normed_gaze_features_11Dec25'
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

    
if vid == 'dme':
    good_ET_pats = ["NS191"]
    
if vid == 'despicable_me_english':
    good_ET_pats = ['NS127_02','NS135','NS136','NS137','NS138','NS140','NS153','NS154','NS164','NS166','NS174_02']
    #good_ET_pats = ["NS127_02"]
elif vid == 'inscapes':
    #good_ET_pats = ['NS127_02','NS135','NS136','NS137','NS138','NS140','NS140_02','NS164','NS166']
    #good_ET_pats = ['NS127_02','NS135','NS136','NS137','NS138','NS140','NS140_02','NS154','NS164','NS166','NS178', "NS205", 'NS210','NS211']
    good_ET_pats = ["NS137", "NS205", "NS210"]

patients = good_ET_pats
#patients = ['NS201_02']
patients.sort()

fs_eye = 300

# Load ISC data 
if vid =='despicable_me_english':
    data = np.load(isc_dir + '/isc_gaze_position_time.npz')  
if vid == "dme":
    data = np.load(isc_dir + '/isc_gaze_position_time.npz')  
if vid == 'inscapes':
    data = np.load('/Volumes/Samsung/Movie_data/ISC_inscapes_updated_new/inscapes_isc_gaze_position_time_updated.npz')

time_isc = data['time_isc']
patients_isc = data['patients']
isc_time = data['isc_time_gaze']
fs_isc = 1 / np.mean(np.diff(time_isc))   
patients_isc = data['patients']


#%% Loop to calculate and collect eye measures

if compute_measures:
    
    # Initialize an empty list to collect rows for the final DataFrame
    all_eye_data = []
    
    # Dictionary to store values for each patient
    all_subs_saccade_rates = {} 
    all_subs_sliding_vergence = {}  
    all_subs_vergence_std = {}
    all_subs_abs_vergence = {}
    all_subs_saccade_dispersion = {}
    all_subs_saccade_dispersion_std = {}
    all_subs_blink_rate = {}
    all_subs_blink_duration = {}
    all_subs_rolling_pupil = {}
    all_subs_rolling_pupil_std = {}
    all_subs_isc = {}
    #
    
    # Store standard deviations in new dictionaries
    all_subs_saccade_rates_std = {} 
    all_subs_vergence_rates_std = {} 
    
    # Load gaze, pupil, and oculometric data
    for pat in patients:
        pat_dir = os.path.join(data_dir, pat)
        fig_patient_dir = os.path.join(fig_dir, pat)
        if not os.path.exists(fig_patient_dir):
            os.makedirs(fig_patient_dir)
            
        print('Loading data for patient {:s} ...'.format(pat))
    
        #LOAD ISC
        if pat == 'NS201':
            pat_num = 'NS201_02'
        else:
            pat_num = pat
        idx_pat = np.in1d(patients_isc, pat_num)
        isc_pat = isc_time[idx_pat]
        # ISC is already calculated with gaze data cut around the movie times, derived from pupil data
        isc_pat_t = isc_pat.T #transpose to fit the dims of other data
        isc_pat_t = np.squeeze(isc_pat_t) 
    
        ### LOAD PUPIL DATA
        eye_pat_dir = '{:s}/{:s}/Eye_prep'.format(mne_data_dir, pat)
        eye_files = os.listdir(eye_pat_dir)
        eye_files = list(compress(eye_files, ['_et_prep' in f for f in eye_files]))
        if pat == "NS190":
            eye_file = '/Volumes/Samsung/Movie_data/movies_prep_standard/NS190/Eye_prep/NS190_ses-01_task-despicable_me_english_run-02_et_prep.npz'
            eye_data = np.load(eye_file)  
        elif pat == "NS191":
            eye_file = '/Volumes/Samsung/Movie_data/movies_prep_standard/NS191/Eye_prep/NS191_ses-01_task-despicable_me_english_run-01_et_prep_updated.npz'
            eye_data = np.load(eye_file)
        else:
            eye_list = list(compress(eye_files, [vid in f for f in eye_files]))
            eye_file = eye_list[0]
            eye_data = np.load('{:s}/{:s}'.format(eye_pat_dir, eye_file))

            
        # movie start and end times because t_pupil is cut around movie times
        # note that t_pupil is a different sampling rate as the other eye data
        t_start = eye_data['t_pupil'][0]
        t_end = eye_data['t_pupil'][-1]
        t_pupil = eye_data['t_pupil']
        pupil = eye_data['pupil']
        
        ### PREPARE VERGENCE TIME SERIES ####
        pat_dir = '{:s}/{:s}'.format(data_dir, pat)
        et_prep_dir = os.path.join(pat_dir, 'Eye_prep')
        verg_files = os.listdir(et_prep_dir)
        
        
        if vid == 'despicable_me_english':
            candidates = [
                f for f in verg_files
                if not f.startswith("._") and f"{vid}_run-1_et_prep.csv" in f
            ]
        elif vid == "dme":   
            if pat == 'NS190':
                candidates = '/Volumes/Samsung/Movie_data/movies_prep_standard/NS190/Eye_prep/NS190_ses-dme02_behavior+ecephys.csv'
            if pat == "NS191":
                candidates ="/Volumes/Samsung/Movie_data/movies_prep_standard/NS191/Eye_prep/NS191_ses-dme01_behavior+ecephys.csv"
        elif vid == 'inscapes':
            # Use the actual filenames you showed: they contain 'task-inscapes' and end with '_et_prep.csv'
            candidates = [
                f for f in verg_files
                if not f.startswith("._")
                and "inscapes" in f
                and f.endswith("_et_prep.csv")]
        else:
            candidates = []
        
        if not candidates:
            raise FileNotFoundError(f"No et_prep CSV found in {et_prep_dir} for movie {vid}")
        
        # This is just the filename (no directory)
        if pat == "NS190":
            verg_dat = pd.read_csv("/Volumes/Samsung/Movie_data/movies_prep_standard/NS190/Eye_prep/NS190_ses-dme02_behavior+ecephys.csv")
        elif pat == "NS191":
            verg_dat = pd.read_csv("/Volumes/Samsung/Movie_data/movies_prep_standard/NS191/Eye_prep/NS191_ses-dme01_behavior+ecephys.csv")
        else:
            verg_file = candidates[0]
            #extract patient vergence data
            verg_dat = pd.read_csv('{:s}/Eye_prep/{:s}'.format(pat_dir, verg_file))
       # vis_fd = verg_dat['vis_fd_interp'].values
        gaze_disp = verg_dat['dva_gaze_disp_x_interp']
        verg_t = verg_dat['time'].values
        
        # vergence time
        t_verg = verg_t
        # vergence time series  
        vergence = gaze_disp
        
        # Add debug prints after loading data
        print(f"\n=== Debug Info for Patient {pat} ===")
        print(f"Original vergence length: {len(vergence)}")
        print(f"Original t_verg length: {len(t_verg)}")
        print(f"t_start: {t_start:.3f}")
        print(f"t_end: {t_end:.3f}")
        
        # After cutting the data
        print(f"\nAfter cutting around movie times:")
        print(f"Vergence length after cut: {len(vergence)}")
        print(f"t_verg length after cut: {len(t_verg)}")
        
        #### PREPARE SACCADE TIME SERIES ####
        pat_dir = '{:s}/{:s}'.format(data_dir, pat)
        sub_et_pat_dir = '{:s}/Eye_prep'.format(pat_dir)
        #et_files = os.listdir('{:s}/Eye_prep/'.format(pat_dir))
        #et_file = [file for file in et_files if f'task-{vid}_run-1_et_prep.npz' in file][0]
        
        if vid == "inscapes":
            pattern = f"*{vid}*et_prep.npz"  # e.g. "*inscapes*et_prep.csv"
            candidates = [
            f for f in glob.glob(os.path.join(sub_et_pat_dir, pattern))
            if not os.path.basename(f).startswith("._")]
            et_file = candidates[0]
        elif vid == "despicable_me_english":
            pattern = f"*{vid}*et_prep.npz"  #
            candidates = [
            f for f in glob.glob(os.path.join(sub_et_pat_dir, pattern))
            if not os.path.basename(f).startswith("._")]
            et_file = candidates[0]
        elif vid == "dme":
            if pat == "NS190":
                et_file = '/Volumes/Samsung/Movie_data/movies_prep_standard/NS190/Eye_prep/NS190_ses-01_task-despicable_me_english_run-02_et_prep.npz'
            elif pat == "NS191":
                et_file = '/Volumes/Samsung/Movie_data/movies_prep_standard/NS191/Eye_prep/NS191_ses-01_task-despicable_me_english_run-01_et_prep_updated.npz'
        if not candidates:
            raise FileNotFoundError(f"No et_prep npz found in {sub_et_pat_dir} for movie {vid}")
 
        et_data = np.load(et_file)       # use it directly

        # saccade (gaze)time (should be the same as t_verg)
        t = et_data['t_gaze']
        
        #load saccade onset timings
        saccade_onset_t = et_data['saccade_onset_t']
    
        #load fixation onset timings (should be the same length as saccade onset timings)
        fixation_t = et_data['fixation_t']
        
        #load gaze data
        xy = et_data['xy']
        
        # check that the gaze timings for saccade and vergence are identical
        if np.allclose(t, t_verg, atol=1e-6, rtol=1e-5):
            print("gaze timings are identical within the specified tolerance")
        else:
            print("gaze timings are not identical, please check!")
        
        
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
   ##     print(f"saccade_rate_sliding length: {len(saccade_rate_sliding)}")
    #    print(f"rolling_dispersion length: {len(rolling_dispersion)}")
    #    print(f"rolling_blink_rate length: {len(rolling_blink_rate)}")
        print("=" * 50)
        
        ########### Calculate rolling saccade rate ###############
        
        # here, t is zeroed out, and saccade_onset_t has had t_start subtracted from it, so it should be aligned   
        is_saccade_onset = np.zeros(len(t), dtype=bool) 
        
        for onset_time in saccade_onset_t:
            idx = np.argmin(np.abs(t - (onset_time)))
            is_saccade_onset[idx] = True
        
        saccade_onset_int = is_saccade_onset.astype(int)
        
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
                dispersion = math.dist(saccade_pos, fixation_pos)
        
            # Store the result
            saccade_dispersions.append({
                "saccade_onset_time": saccade_time,
                "fixation_onset_time": fixation_time,
                "dispersion": dispersion
            })
        
        # Convert results to a DataFrame
        saccade_dispersions_df = pd.DataFrame(saccade_dispersions)
        
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
    
    
        #### COMBINE ALL DATA ###
        
        
        all_subs_saccade_rates[pat] = saccade_rate_sliding  # Store the saccade rates
        all_subs_sliding_vergence[pat] = verg_sliding  # Store vergence rates
        all_subs_vergence_std[pat] = verg_sliding_std
        all_subs_abs_vergence[pat] = abs_sliding_vergence
        all_subs_saccade_dispersion[pat] = rolling_dispersion
        all_subs_saccade_dispersion_std[pat] = rolling_dispersion_std
        all_subs_blink_rate[pat] = rolling_blink_rate
        all_subs_blink_duration[pat] = rolling_blink_duration
        all_subs_rolling_pupil[pat] = rolling_pupil_avg
        all_subs_rolling_pupil_std[pat] = rolling_pupil_std
        all_subs_isc[pat] = isc_pat_t
        
        #normalize saccade_rate_sliding, verg_slidingrolling_dispersion, rolling_dispersion_std, rolling_blink_rate, rolling blink duration (replacing nans with 0s)
    
        vectors = {
        'Patient': [pat] * len(all_subs_saccade_rates[pat]),
        'Saccade_Rate': all_subs_saccade_rates[pat],
        'Vergence': all_subs_sliding_vergence[pat],
        'Vergence_Std': all_subs_vergence_std[pat],
        'Abs_Vergence': all_subs_abs_vergence[pat],
        'Saccade_Dispersion': all_subs_saccade_dispersion[pat],
        'Saccade_Dispersion_Std': all_subs_saccade_dispersion_std[pat],
        'Blink_Rate': all_subs_blink_rate[pat],
        'Blink_Duration': all_subs_blink_duration[pat],
        'Pupil_Avg': all_subs_rolling_pupil[pat],
        'Pupil_Std': all_subs_rolling_pupil_std[pat],
        'ISC': all_subs_isc[pat],
        }
    
        for name, v in vectors.items():
            try:
                shape = v.shape
            except AttributeError:
                shape = (len(v),)
            print(f"{name:25s} length = {len(v):6d}, shape = {shape}")
    
        
    
    
        patient_data = pd.DataFrame({
            'Patient': [pat] * len(all_subs_saccade_rates[pat]),  # Repeat the patient ID for each time point
            'Saccade_Rate': all_subs_saccade_rates[pat],
            'Vergence': all_subs_sliding_vergence[pat],
            'Vergence_Std': all_subs_vergence_std[pat],
            'Abs_Vergence': all_subs_abs_vergence[pat],
            'Saccade_Dispersion': all_subs_saccade_dispersion[pat],
            'Saccade_Dispersion_Std': all_subs_saccade_dispersion_std[pat],
            'Blink_Rate': all_subs_blink_rate[pat],
            'Blink_Duration': all_subs_blink_duration[pat],
            'Pupil_Avg': all_subs_rolling_pupil[pat],
            'Pupil_Std': all_subs_rolling_pupil_std[pat],
            'ISC': all_subs_isc[pat],
        })
        
        # Save the Patient-Level DataFrame to a CSV file
        patient_eye_filename = os.path.join(fig_patient_dir, f'many_normed_et_features_{vid}_{pat}.csv')
        patient_data.to_csv(patient_eye_filename, index=False)
    
        print(f"Eye features saved to {patient_eye_filename}")
    
        
        # Columns to normalize
        columns_to_normalize = [
            'Saccade_Rate', 
            'Vergence', 
            'Vergence_Std',
            'Abs_Vergence',
            'Saccade_Dispersion', 
            'Saccade_Dispersion_Std', 
            'Blink_Rate', 
            'Blink_Duration', 
          #  'Pupil_Avg', 
          #  'Pupil_Std', 
          #  'ISC'
        ]
            
        # Normalize and replace NaNs with 0s
        for col in columns_to_normalize:
            if col in patient_data.columns:
                # Replace NaNs with 0 before normalization
                patient_data[col] = patient_data[col].fillna(0)
                
                # Normalize (z-score normalization)
                patient_data[col] = (patient_data[col] - patient_data[col].mean()) / patient_data[col].std()
                
    
        # Append this patient's data to the list
        all_eye_data.append(patient_data)
    
    # Concatenate all rows into a single DataFrame
    final_eye_df = pd.concat(all_eye_data, ignore_index=True)
    
        
    # Calculate the correlation matrix, excluding the first column
    correlation_matrix = final_eye_df.iloc[:, 1:].corr()  # Exclude the first column by slicing
    
    # Plot the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", cbar=True, square=True)
    
    # Add title and labels
    plt.title(f'Correlation Heatmap of Eye Movement Features {vid}', fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    
    # Show the plot
    plt.show()
    
    # Save the DataFrame to a CSV file
    eye_feature_filename = os.path.join(fig_dir, f'many_et_features_{vid}.csv')
    final_eye_df.to_csv(eye_feature_filename, index=False)
    
    print(f"Eye features saved to {eye_feature_filename}")
    
    
    #%% NORMALIZING THE ENTIRE DATA FRAME BEFORE PCA
    # Columns to normalize (excluding the 'Patient' column)
    columns_to_normalize = [
        'Saccade_Rate', 
        'Vergence', 
        'Vergence_Std',
        'Abs_Vergence',
        'Saccade_Dispersion', 
        'Saccade_Dispersion_Std', 
        'Blink_Rate', 
        'Blink_Duration', 
        'Pupil_Avg', 
        'Pupil_Std', 
        'ISC'
    ]
    
    normed_eye_df = final_eye_df.copy()
    
    # Normalize the entire dataset
    normed_eye_df[columns_to_normalize] = (final_eye_df[columns_to_normalize] - 
                                          final_eye_df[columns_to_normalize].mean()) / \
                                         final_eye_df[columns_to_normalize].std()
    
    # Save the DataFrame to a CSV file
    normed_feature_filename = os.path.join(fig_dir, f'many_normed_et_features_for_pca_{vid}.csv')
    normed_eye_df.to_csv(normed_feature_filename, index=False)
