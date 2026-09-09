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
from scipy import stats, signal, interpolate
from scipy.signal import correlate

from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA
from sklearn.decomposition import FactorAnalysis
#from factor_analyzer import FactorAnalyzer


drive = 'Samsung'
vid = 'despicable_me_english'

    
#data_dir = f'/Volumes/{drive}/Movie_data/movies_prep_standard'
data_dir = '/Volumes/Samsung/Movie_data/movies_prep_standard'
isc_dir = f'/Volumes/{drive}/Movie_data/data/isc'
elec_dir = f'/Volumes/{drive}/Movie_data/data/electrode_localization'
movie_subs_table = pd.read_csv('/Volumes/Samsung/anatomy/shared_correspondence/movie_subs_master_updated.csv')

fig_dir = f'/Volumes/{drive}/Movie_data/saccade_new_test_1Jul25'
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

patients =os.listdir(data_dir) 
patients.sort()

if vid == 'inscapes':
    patients =['NS127_02','NS135','NS136','NS137','NS138','NS140','NS140_02','NS153','NS154','NS164','NS166','NS178','NS174_02']
    key = 'inscapes'
    #patients = ['NS178']
    
if vid in ['dme', 'despicable_me_english']:
    patients = ["NS190"]
    keys = ['dme', 'dm', 'despicable_me']
    


#%%
fs_eye = 300

# Load ISC data 
if vid in ['dme', 'despicable_me_english']:
    data = np.load('/Volumes/Samsung/Movie_data/ISC_despicable_me_english_newsubs/despicable_me_english_isc_gaze_position_time_598_win.npz')  
if vid == 'inscapes':
    data = np.load('/Volumes/Samsung/Movie_data/ISC_inscapes_updated_new/inscapes_isc_gaze_position_time_updated.npz')

time_isc = data['time_isc']
patients_isc = data['patients']
isc_time = data['isc_time_gaze']
fs_isc = 1 / np.mean(np.diff(time_isc))   
patients_isc = data['patients']

compute_isc = True
compute_blinks = True
compute_verg = True
compute_pupil = False
compute_sacc = True
compute_rolling = True


#%%
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


def compare_timebases(t_a, t_b, *, atol=1e-6, rtol=1e-5, label_a="t", label_b="t_verg"):
    # Require 1D arrays
    t_a = np.asarray(t_a).ravel()
    t_b = np.asarray(t_b).ravel()

    if len(t_a) == 0 or len(t_b) == 0:
        print(f"{label_a}/{label_b}: one is empty; cannot compare.")
        return False

    # If both are already zeroed, offset will be ~0; otherwise this aligns clocks
    offset = t_a[0] - t_b[0]
    t_b_aligned = t_b + offset

    n = min(len(t_a), len(t_b_aligned))
    ok = np.allclose(t_a[:n], t_b_aligned[:n], atol=atol, rtol=rtol)

    if ok:
        print(f"{label_a}/{label_b}: match after offset alignment (checked first {n} samples).")
    else:
        max_err = np.max(np.abs(t_a[:n] - t_b_aligned[:n]))
        print(f"{label_a}/{label_b}: mismatch after alignment. max_abs_err={max_err:.6g}s, n={n}, "
              f"len({label_a})={len(t_a)}, len({label_b})={len(t_b)}")
    return ok



#%% Loop to calculate and collect eye measures

# Load gaze, pupil, and oculometric data
for pat in patients:
    
    # --- set keys/run defaults per patient ---
    if vid == "inscapes":
        keys = ["inscapes"]
    else:
        keys = ["dme", "dm", "despicable_me"]  # your aliases

    if vid in ["dme", "despicable_me_english"] and pat == "NS190":
        run = "run-02"
    else:
        run = "run-01"
    
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

#%%

    eye_pat_dir = '{:s}/{:s}/Eye_prep'.format(data_dir, pat)
    eye_files = os.listdir(eye_pat_dir)
    eye_files = list(compress(eye_files, ['_et_prep.npz' in f for f in eye_files]))
  
    eye_files_all = os.listdir(eye_pat_dir)
    
    # NPZ
    npz_candidates = pick_file(
        eye_files_all,
        contains_any=keys,
        endswith="_et_prep.npz",
        startswith_not="._"
    )
    # If multiple, filter by run; if only one, keep it.
    if len(npz_candidates) > 1:
        npz_candidates = [f for f in npz_candidates if run in f]
    if not npz_candidates:
        raise FileNotFoundError(f"No _et_prep.npz found for {pat=} {vid=} {run=} in {eye_pat_dir}")
    eye_file = npz_candidates[0]
    eye_data = np.load(os.path.join(eye_pat_dir, eye_file))
    
    # VERGENCE CSV
    verg_candidates = pick_file(
        eye_files_all,
        contains_any=keys,
        endswith="et_prep.csv",
        startswith_not="._"
    )
    if len(verg_candidates) > 1:
        verg_candidates = [f for f in verg_candidates if run in f]
    if not verg_candidates:
        raise FileNotFoundError(f"No et_prep.csv found for {pat=} {vid=} {run=} in {eye_pat_dir}")
    verg_file = verg_candidates[0]
    verg_dat = pd.read_csv(os.path.join(eye_pat_dir, verg_file))
    
    # BLINK CSV
    blink_candidates = pick_file(
        eye_files_all,
        contains_any=keys,
        endswith="blink_events.csv",
        startswith_not="._"
    )
    if len(blink_candidates) > 1:
        blink_candidates = [f for f in blink_candidates if run in f]
    if not blink_candidates:
        raise FileNotFoundError(f"No blink_events.csv found for {pat=} {vid=} {run=} in {eye_pat_dir}")
    blink_file = blink_candidates[0]
    blink_data = pd.read_csv(os.path.join(eye_pat_dir, blink_file))



#%%

    
    # movie start and end times because t_pupil is cut around movie times
    # note that t_pupil is a different sampling rate as the other eye data

     # --- movie bounds from pupil clock ---
    t_start = eye_data["t_pupil"][0]
    t_end   = eye_data["t_pupil"][-1]
    t_pupil = eye_data["t_pupil"]
    
    # --- gaze time from NPZ ---
    t_gaze = eye_data["t_gaze"]
    xy     = eye_data["xy"]
    
    # --- verg time from CSV ---
    t_verg = verg_dat["time"].values
    vergence = verg_dat["dva_gaze_disp_x_interp"].values
    
    # cut both to [t_start, t_end]
    idx_gaze = (t_gaze > t_start) & (t_gaze < t_end)
    idx_verg = (t_verg > t_start) & (t_verg < t_end)
    
    t_gaze = t_gaze[idx_gaze]
    xy     = xy[idx_gaze]
    
    t_verg = t_verg[idx_verg]
    vergence = vergence[idx_verg]
    
    # zero both
    t_gaze = t_gaze - t_gaze[0]
    t_verg = t_verg - t_verg[0]
    
    # compare robustly (won’t crash on length mismatch)
    if compute_verg and compute_sacc:
        compare_timebases(t_gaze, t_verg, label_a="t_gaze", label_b="t_verg")
    
    
    #load saccade onset timings
    saccade_onset_t = eye_data['saccade_onset_t']

    #load fixation onset timings (should be the same length as saccade onset timings)
    fixation_t = eye_data['fixation_t']
    saccade_onset_t = saccade_onset_t - t_start
    fixation_t = fixation_t - t_start
    
    blink_onset = blink_data['start_time'] 
    blink_end = blink_data['end_time']
    blink_duration = blink_data['duration']
    
    # remove t_start from the blink onsets and ends 
    blink_onset = blink_onset - t_start
    blink_end = blink_end - t_start
    
    #and from t_pupil
    t_pupil = t_pupil - t_start

     


### OLD CODE NOT SURE WHAT TO CUT       
    ### CUT ALL GAZE DATA AROUND MOVIE START AND END TIMES
    # boolean series to cut off beginning and end of gaze time series based on movie start and end times
    idx_vid = np.logical_and(t_verg > t_start, t_verg < t_end)
    
    #truncating vergence time accordingly
    t_verg = t_verg[idx_vid]
    
    #truncating vergence accordingly
    vergence = vergence[idx_vid]
    
    # this step makes the first value of t_verg (vergence timing) 0
    t_verg = t_verg - t_verg[0]
    
    t = eye_data['t_gaze']

    # do the same for saccade timing and gaze data 
    idx_vid = np.logical_and(t > t_start, t < t_end)

    t = t[idx_vid]
    
    xy = eye_data['xy']

    xy = xy[idx_vid]
    
    #zeroes out t_verg
    t = t-t[0]
    
    if np.allclose(t, t_verg, atol=1e-6, rtol=1e-5):
        print("gaze timings are identical within the specified tolerance")
    else:
        print("gaze timings are not identical, please check!")
        
    ## remove t_start from all of the timing values in saccade_onset and fixation onset timings
    

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
    
    ########### Calculate saccades, fixations, and dispersions ###############
    
    if compute_sacc:
        # saccade (gaze)time (should be the same as t_verg)
        t = eye_data['t_gaze']
        
        #load saccade onset timings
        saccade_onset_t = eye_data['saccade_onset_t']
    
        #load fixation onset timings (should be the same length as saccade onset timings)
        fixation_t = eye_data['fixation_t']
        
        #load gaze data
        xy = eye_data['xy']
        
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
        
        saccade_positions = eye_data['saccade_pos']

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
    
#%%
        
    et_file_new = et_file.replace('.npz', '_updated.npz')
    et_data_new_path = '{:s}/Eye_prep/{:s}'.format(pat_dir, et_file_new)
    
    et_data_dict = dict(et_data)
    et_data_dict['fixation_int'] = fixation_int
    et_data_dict['saccade_onset_int'] = saccade_onset_int
    et_data_dict['saccade_dispersion_dist'] = saccade_dispersions_dist

    np.savez(et_data_new_path, **et_data_dict)
    
    print(f"Saved updated et_data to {et_data_new_path}")

    

