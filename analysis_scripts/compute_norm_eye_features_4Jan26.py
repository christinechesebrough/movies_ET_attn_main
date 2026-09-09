#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 18:55:15 2025

@author: christinechesebrough

"""

import os, re
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

def extract_ses_label(fname: str) -> str:
    """
    Optional: extract 'ses-02' etc. Fallback: ''.
    """
    m = re.search(r'ses[-_]?(\d+)', fname, flags=re.IGNORECASE)
    if m:
        return f"ses-{int(m.group(1)):02d}"
    return ""

def sort_key(f):
    ses = extract_ses_label(f)
    run = extract_run_label(f)
    return (ses, run, f)

#%%

drive = 'Samsung'
vid = "despicable_me_hungarian" #'dme' "inscapes"

window_len = 10
overlap = 7.5

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

#fig_dir = f'/Volumes/{drive}/Movie_data/6Apr26_norm_eye_features_by_rec_{window_len}s'
fig_dir = f'/Volumes/{drive}/Movie_data/6Apr26_norm_eye_features_by_rec_{window_len}s'

if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

    
if vid == 'despicable_me_english':
   good_ET_pats = ['NS127_02','NS135','NS136','NS137','NS138','NS140','NS140_02','NS153','NS155','NS164','NS166','NS174_02','NS174_03',"NS178","NS190","NS191",'NS193',"NS194","NS201",'NS205']
   # good_ET_pats = ['NS127_02','NS135','NS136','NS137','NS138','NS140','NS153','NS164','NS166',
   # 'NS174_02',"NS178","NS190","NS191",'NS193',"NS194","NS201",'NS205']
     #good_ET_pats = ['NS190']
   good_ET_runs = ['NS127_02_ses-02_run-01', 'NS135_ses-01_run-01',
          'NS136_ses-01_run-01', 'NS137_ses-01_run-01',
          'NS138_ses-01_run-01', 'NS140_ses-01_run-01',
          'NS140_02_ses-02_run-01', 'NS153_ses-01_run-01',
          'NS155_02_ses-02_run-01', 'NS164_ses-01_run-01',
          'NS166_ses-01_run-01', 'NS174_02_ses-02_run-01',
          'NS174_03_ses-03_run-01', 'NS178_ses-01_run-01',
          'NS190_ses-01_run-01', 'NS190_ses-01_run-02',
          'NS191_ses-01_run-01', 'NS193_ses-01_run-01',
          'NS193_ses-01_run-02', 'NS194_ses-01_run-01',
          'NS205_ses-01_run-01']
  # good_ET_runs = ['NS174_03_ses-03_run-01']
  
  
if vid == 'despicable_me_hungarian':

   good_ET_runs = ['NS127_02_ses-02_run-01', 
                'NS135_ses-01_run-01',
        'NS136_ses-01_run-01', 'NS137_ses-01_run-01',
        'NS138_ses-01_run-01', 'NS140_ses-01_run-01',
        'NS140_02_ses-02_run-01',
        'NS145_ses-02_run-01', 'NS154_ses-01_run-01',
        'NS164_ses-01_run-01', 
        'NS174_02_ses-02_run-01',
        'NS174_03_ses-03_run-01', 'NS178_ses-01_run-01'
          ]
  # good_ET_runs = ['NS174_03_ses-03_run-01']
  
  
elif vid == 'inscapes':
    good_ET_pats = ['NS127_02','NS135','NS136','NS137','NS138','NS140','NS140_02','NS154','NS164','NS166','NS178', "NS205", 'NS210','NS211']
    good_ET_runs = ['NS127_02_ses-02_run-01', 'NS135_ses-01_run-01',
           'NS136_ses-01_run-01', 'NS137_ses-01_run-01',
           'NS138_ses-01_run-01', 'NS140_ses-01_run-01',
           'NS140_02_ses-02_run-01', 'NS144_ses-01_run-01',
           'NS151_ses-01_run-01', 'NS153_ses-01_run-01',
           'NS154_ses-01_run-01', 'NS155_ses-01_run-01',
           'NS155_02_ses-02_run-01', 'NS164_ses-01_run-01',
           'NS178_ses-01_run-01', 'NS205_ses-01_run-01',
           'NS210_ses-01_run-01', 'NS211_ses-01_run-01']
    
    
#patients = good_ET_pats
#patients = ['NS191']
#patients = ['NS201_02']
#patients.sort()

#good_ET_runs = ['NS190_ses-01_run-02'] #, 'NS190_ses-01_run-02','NS191_ses-01_run-01', 'NS193_ses-01_run-01','NS193_ses-01_run-02']

runs = good_ET_runs

fs_eye = 300

# Load ISC data 
if vid in ['dme', 'despicable_me_english']:
    if window_len == 5:
        data = np.load('/Volumes/Samsung/Movie_data/ISC_despicable_me_english_5s_windows/despicable_me_english_isc_gaze_position_time_updated.npz')
    elif window_len == 10:
        data = np.load('/Volumes/Samsung/Movie_data/ISC_despicable_me_english_30Dec25/ISC_despicable_me_english_30Dec25_despicable_me_english_isc_gaze_position.npz',allow_pickle = True)

if vid in ['dmh', 'despicable_me_hungarian']:
    if window_len == 5:
        data = np.load(f'/Volumes/Samsung/Movie_data/ISC_{vid}_5s_windows/{vid}_isc_gaze_position_time_updated.npz')
    elif window_len == 10:
        data = np.load(f'/Volumes/Samsung/Movie_data/ISC_{vid}_10s_windows/{vid}_isc_gaze_position_time_updated.npz',allow_pickle = True)
if vid == 'inscapes':
    if window_len == 5:
        data = np.load('/Volumes/Samsung/Movie_data/ISC_inscapes_5s_windows/inscapes_isc_gaze_position_time_updated.npz')
    elif window_len == 10:
        data = np.load("/Volumes/Samsung/Movie_data/ISC_inscapes_30Dec25/ISC_inscapes_30Dec25_inscapes_isc_gaze_position.npz",allow_pickle = True)


if window_len == 10:
    time_isc = data['time_isc']
   # isc_patients = data['entry_patients']
    entry_id_isc = data['entry_ids']
    isc_time = data['isc_time_gaze']
    fs_isc = 1 / np.mean(np.diff(time_isc))   
    isc_entries = data['patients']
elif window_len == 5:
    time_isc = data['time_isc']
    #isc_patients = data['patients']
    entry_id_isc = data['entry_ids']
    isc_time = data['isc_time_gaze']
    fs_isc = 1 / np.mean(np.diff(time_isc))   
    isc_entries = data['patients']
    

#%% Loop to calculate and collect eye measures

if compute_measures:
    
    # Initialize an empty list to collect rows for the final DataFrame
    all_eye_data = []
    
    # Dictionary to store values for each patient
    all_subs_saccade_rates = {}
    all_subs_saccade_dispersion = {}
    all_subs_saccade_dispersion_std = {}
    all_subs_sliding_vergence = {}  
    all_subs_vergence_std = {}
    all_subs_abs_vergence = {}
    all_subs_blink_rate = {}
    all_subs_blink_duration = {}
    all_subs_rolling_pupil = {}
    all_subs_rolling_pupil_std = {}
    all_subs_isc = {}  
        
    #
    all_subs_saccade_rates_std = {} 
    all_subs_vergence_rates_std = {} 

    # dict for counts/sums/means for each patient
    all_eye_counts = []
    
    all_subs_count_saccades = {}
    all_subs_count_blinks= {}
    all_subs_sum_vergence = {}
    all_subs_mean_vergence = {}
    all_subs_sum_abs_vergence = {}
    all_subs_mean_abs_vergence = {}
    all_subs_sum_sacc_distance = {}
    all_subs_mean_sacc_distance = {}
    all_subs_std_sacc_distance = {}
    all_subs_med_pupil_dilation = {}
    all_subs_pupil_variability = {}
    all_subs_isc_mean = {}
    

    # Load gaze, pupil, and oculometric data
    for rec in runs:   
        # --- set keys/run defaults per patient ---
        if vid == "inscapes":
            keys = ["inscapes"]
        elif vid == "despicable_me_hungarian":
            keys = ["despicable_me_hungarian"]
        else:
            keys = ["dme", "despicable_me_english"]  

        pat = extract_pat_id(rec)        
        ses, run, entry_id = sort_key(rec)

        rec_dir = os.path.join(data_dir, pat)
        fig_patient_dir = os.path.join(fig_dir, entry_id)
        if not os.path.exists(fig_patient_dir):
            os.makedirs(fig_patient_dir)
            
        print('Loading data for patient {:s} ...'.format(pat))
        if window_len == 5:
            run_id = '{:s}_{:s}'.format(pat, run)
        if window_len == 10:
            if vid == 'despicable_me_english' or 'inscapes':
                run_id = '{:s}_{:s}_{:s}'.format(pat,ses,run)
            if vid == 'despicable_me_hungarian':
                run_id = '{:s}_{:s}'.format(pat, run)

        idx_rec = np.in1d(entry_id_isc,run_id)
        match_idx = np.where(idx_rec)[0]
        isc_rec = isc_time[idx_rec]
        
        mean_isc = float(np.mean(isc_rec))
            
        # ISC is already calculated with gaze data cut around the movie times, derived from pupil data
        isc_rec_t = isc_rec.T #transpose to fit the dims of other data
        isc_rec_t = np.squeeze(isc_rec_t) 
        
        if isc_rec_t.ndim == 2:
            if isc_rec_t.shape[1] == 0:
                isc_rec_t = np.full((isc_rec_t.shape[0],), np.nan)
            else:
                isc_rec_t = np.nanmean(isc_rec_t, axis=1)
        elif isc_rec_t.ndim == 1:
            pass
        else:
            isc_rec_t = np.squeeze(isc_rec_t)
        
        eye_pat_dir = '{:s}/{:s}/Eye_prep'.format(data_dir, pat)
        eye_files = os.listdir(eye_pat_dir)
        eye_files = list(compress(eye_files, ['_et_prep.npz' in f for f in eye_files]))
     
        eye_files_all = os.listdir(eye_pat_dir)
        
        # NPZ
        # npz_candidates = pick_file(
        #     eye_files_all,
        #    # contains_any=keys,
        #     endswith="_et_prep.npz",
        #     startswith_not="._"
        # )
        
        npz_candidates = [f for f in eye_files_all 
                           if any(k in f for k in keys)
                           if f.endswith("et_prep.npz")
                           and not f.startswith("._")]

        
        
        if len(npz_candidates) > 1:
            # 1) Try run-* style filenames first
            npz_candidates = [f for f in npz_candidates if run in f]
        
        npz_candidates = sorted(npz_candidates)

        if not npz_candidates:
            raise FileNotFoundError(f"No _et_prep.npz found for {pat=} {vid=}in {eye_pat_dir}")
        
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
            # 1) Try run-* style filenames first
            verg_candidates = [f for f in verg_candidates if run in f]
        
        # Make deterministic
        verg_candidates = sorted(verg_candidates)

        if not verg_candidates:
            raise FileNotFoundError(f"No et_prep.csv found for {pat=} {vid=}in {eye_pat_dir}")
        verg_file = verg_candidates[0]
        verg_dat = pd.read_csv(os.path.join(eye_pat_dir, verg_file))
        
        # BLINK CSV
        blink_candidates = [
            f for f in eye_files_all
            if f.endswith('_blink_events.csv')
            and not f.startswith('._')
        ]

        blink_candidates = [
            f for f in blink_candidates
            if any(k in f for k in keys)
        ]
        if len(blink_candidates)>1:
            blink_candidates = [f for f in blink_candidates if run in f]
        
        if not blink_candidates:
            raise FileNotFoundError(f"No blink_events.csv found for {pat=} {vid=} in {eye_pat_dir}")
        blink_file = blink_candidates[0]
        blink_data = pd.read_csv(os.path.join(eye_pat_dir, blink_file))

       
      #     
        # movie start and end times because t_pupil is cut around movie times
        # note that t_pupil is a different sampling rate as the other eye data
        
        t_start = eye_data['t_pupil'][0]
        t_end = eye_data['t_pupil'][-1]
        t_pupil = eye_data['t_pupil']
        pupil = eye_data['pupil']
        
        med_pupil = float(np.nanmedian(pupil))
        std_pupil = float(np.nanstd(pupil))

        fs_pupil = len(t_pupil)/ ((t_end-t_start))
        
        # variability in pupil dilation (more informative than sd on already noralized pupil diameter)
        dp = np.diff(pupil) * fs_pupil
        var_dp = np.nanstd(dp)
        
        # smoothed signal in case it matters
        # import numpy as np
        # from scipy.signal import savgol_filter

        # # choose ~0.5–1.0 s window; must be odd
        # win_s = 0.7
        # win = int(win_s * fs_pupil)
        # win = win + 1 if win % 2 == 0 else win
        
        # p_smooth = savgol_filter(pupil, window_length=win, polyorder=2, mode="interp")
        
        # dp = np.diff(p_smooth) * fs_pupil
        # var_dp = np.nanstd(dp)  # or robust MAD version


        vergence = verg_dat['gaze_dist_x_interp']#['dva_gaze_disp_x_interp']
        t_verg = verg_dat['time'].values      
        
        mean_vergence = float(np.mean(vergence))
        mean_abs_vergence = float(np.mean(abs(vergence)))
        sum_vergence = float(np.sum(vergence))
        sum_abs_vergence = float(np.sum(abs(vergence)))

        # Add debug prints after loading data
        print(f"\n=== Debug Info for Patient {pat} ===")
        print(f"Original vergence length: {len(vergence)}")
        print(f"Original t_verg length: {len(t_verg)}")
        print(f"t_start: {t_start:.3f}")
        print(f"t_end: {t_end:.3f}")
        
        
        #### PREPARE SACCADE TIME SERIES ####
        # saccade (gaze)time (should be the same as t_verg)
        t = eye_data['t_gaze']
        print(f"Original t_gaze length: {len(t)}")

        #load saccade onset timings
        saccade_onset_t = eye_data['saccade_onset_t']
    
        #load fixation onset timings (should be the same length as saccade onset timings)
        fixation_t = eye_data['fixation_t']
        
        #load gaze data
        xy = eye_data['xy']
        
        # check that the gaze timings for saccade and vergence are identical
        if np.allclose(t, t_verg, atol=1e-6, rtol=1e-5):
            print("gaze timings are identical within the specified tolerance")
        else:
            print("gaze timings are not identical, please check!")
        
            
        ## EXTRACT BLINK RATE
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
        
        #if np.allclose(t, t_verg, atol=1e-6, rtol=1e-5):
        #    print("gaze timings are identical within the specified tolerance")
        #else:
        #    print("gaze timings are not identical, please check!")
            
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
        window_samples = window_len*fs_eye  
        overlap_samples = overlap*fs_eye 
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
        
        # --- Force window-level outputs to exactly 236 samples (for downstream alignment) ---
        if window_len == 10:
            target_steps = 236
        elif window_len == 5:
            target_steps = 238
        
        # if num_steps != target_steps:
        #     # Original window index grid (0..num_steps-1)
        #     x_old = np.linspace(0, 1, num_steps)
        #     x_new = np.linspace(0, 1, target_steps)
        
        #     # Interpolate window-level signals
        #     verg_sliding = np.interp(x_new, x_old, verg_sliding)
        #     verg_sliding_std = np.interp(x_new, x_old, verg_sliding_std)
        
        #     # Interpolate aligned time too (so everything stays consistent)
        #     isc_aligned_time = np.interp(x_new, x_old, isc_aligned_time)
        
        #     # Update derived series
        #     abs_sliding_vergence = np.abs(verg_sliding)
        
        #     # Update num_steps to reflect new length
        #     num_steps = target_steps

        if num_steps != target_steps:
        
            if num_steps == target_steps + 1:
                # Remove only the final extra window
                verg_sliding = verg_sliding[:-1]
                verg_sliding_std = verg_sliding_std[:-1]
                isc_aligned_time = isc_aligned_time[:-1]
        
            elif num_steps > target_steps:
                raise ValueError(
                    f"num_steps={num_steps} exceeds target_steps={target_steps} by more than 1. "
                    "Check whether this should be truncated or handled differently."
                )
        
            elif num_steps < target_steps:
                x_old = np.linspace(0, 1, num_steps)
                x_new = np.linspace(0, 1, target_steps)
        
                verg_sliding = np.interp(x_new, x_old, verg_sliding)
                verg_sliding_std = np.interp(x_new, x_old, verg_sliding_std)
                isc_aligned_time = np.interp(x_new, x_old, isc_aligned_time)
        
            abs_sliding_vergence = np.abs(verg_sliding)
            num_steps = target_steps
    
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
        
        count_saccades = float(np.sum(is_saccade_onset))
    
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
        
        mean_sacc_distance = float(np.mean(saccade_dispersions_df['dispersion']))
        sum_sacc_distance = float(np.sum(saccade_dispersions_df['dispersion']))
        std_sacc_distance = float(np.std(saccade_dispersions_df['dispersion']))
        
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

        # Ensure arrays
        t = np.asarray(t)
        blink_onset = np.asarray(blink_onset, dtype=float)
        blink_duration = np.asarray(blink_duration, dtype=float)
        
        # Sort by onset (important for searchsorted)
        order = np.argsort(blink_onset)
        blink_onset = blink_onset[order]
        blink_duration = blink_duration[order]
        
        # Time step (seconds/sample) inferred from t
        dt = np.median(np.diff(t))  # robust to tiny jitter
        window_len_sec = window_samples * dt
        
        count_blinks = len(blink_onset)
        
        # Outputs
        rolling_blink_count = np.zeros(num_steps, dtype=float)
        rolling_blink_rate = np.zeros(num_steps, dtype=float)         # blinks/sec
        rolling_blink_duration = np.full(num_steps, np.nan, dtype=float)
        
        # Choose how to handle windows with no blinks for duration:
        # "ffill" = carry forward last value; "zero" = 0.0; "nan" = keep NaN
        no_blink_policy = "ffill"
        
        for i in range(num_steps):
            window_start_idx = int(i * step_size_samples)
            window_end_idx = int(window_start_idx + window_samples)
        
            # Guard against running past the end of t
            if window_start_idx >= len(t):
                break
            window_end_idx = min(window_end_idx, len(t))
        
            w0 = t[window_start_idx]
            w1 = t[window_end_idx - 1] + dt  # make window end exclusive-ish
        
            # Find blink events whose onset falls within [w0, w1)
            j0 = np.searchsorted(blink_onset, w0, side="left")
            j1 = np.searchsorted(blink_onset, w1, side="left")
        
            n_blinks = j1 - j0
            rolling_blink_count[i] = n_blinks
            rolling_blink_rate[i] = n_blinks / ( (window_end_idx - window_start_idx) * dt )
        
            if n_blinks > 0:
                rolling_blink_duration[i] = np.mean(blink_duration[j0:j1])
            else:
                if no_blink_policy == "ffill":
                    rolling_blink_duration[i] = rolling_blink_duration[i-1] if i > 0 else 0.0
                elif no_blink_policy == "zero":
                    rolling_blink_duration[i] = 0.0
                elif no_blink_policy == "nan":
                    rolling_blink_duration[i] = np.nan
                else:
                    raise ValueError(f"Unknown no_blink_policy: {no_blink_policy}")
        
        print(f"Length of rolling blink rate: {len(rolling_blink_rate)}")
        print(f"Length of rolling blink duration: {len(rolling_blink_duration)}")
        
        
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
        target_num_steps = len(isc_rec_t)
        
        # Calculate total duration of the pupil data in samples
        total_samples = len(pupil)
        
        # Compute window and step sizes to achieve the target number of steps
        step_size_samples_pupil = total_samples / (target_num_steps + 1)  # Approximate step size
        window_samples_pupil = 2 * step_size_samples_pupil  # Ensure overlap of ~50%
        
        # Convert to integers
        step_size_samples_pupil = int(step_size_samples_pupil)
        window_samples_pupil = int(window_samples_pupil)
        
        # Recalculate the number of steps to ensure alignment
        num_steps_pupil = target_steps
        
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
        
        all_subs_saccade_rates[rec] = saccade_rate_sliding  # Store the saccade rates
        all_subs_sliding_vergence[rec] = verg_sliding  # Store vergence rates
        all_subs_vergence_std[rec] = verg_sliding_std
        all_subs_abs_vergence[rec] = abs_sliding_vergence
        all_subs_saccade_dispersion[rec] = rolling_dispersion
        all_subs_saccade_dispersion_std[rec] = rolling_dispersion_std
        all_subs_blink_rate[rec] = rolling_blink_rate
        all_subs_blink_duration[rec] = rolling_blink_duration
        all_subs_rolling_pupil[rec] = rolling_pupil_avg
        all_subs_rolling_pupil_std[rec] = rolling_pupil_std
        all_subs_isc[rec] = isc_rec_t
        

        all_subs_count_saccades[rec] = count_saccades
        all_subs_count_blinks[rec]= count_blinks
        all_subs_sum_vergence[rec] = sum_vergence
        all_subs_mean_vergence[rec] = mean_vergence
        all_subs_sum_abs_vergence[rec] = sum_abs_vergence
        all_subs_mean_abs_vergence[rec] = mean_abs_vergence
        all_subs_sum_sacc_distance[rec] = mean_sacc_distance
        all_subs_mean_sacc_distance[rec] = sum_sacc_distance
        all_subs_std_sacc_distance[rec] = std_sacc_distance
        all_subs_med_pupil_dilation[rec] = med_pupil
        all_subs_pupil_variability[rec]= var_dp
        all_subs_isc_mean[rec] = mean_isc
            

        vectors = {
        'Patient': [rec] * len(all_subs_saccade_rates[rec]),
        'Saccade_Rate': all_subs_saccade_rates[rec],
        'Vergence': all_subs_sliding_vergence[rec],
        'Vergence_Std': all_subs_vergence_std[rec],
        'Abs_Vergence': all_subs_abs_vergence[rec],
        'Saccade_Dispersion': all_subs_saccade_dispersion[rec],
        'Saccade_Dispersion_Std': all_subs_saccade_dispersion_std[rec],
        'Blink_Rate': all_subs_blink_rate[rec],
        'Blink_Duration': all_subs_blink_duration[rec],
        'Pupil_Avg': all_subs_rolling_pupil[rec],
        'Pupil_Std': all_subs_rolling_pupil_std[rec],
        'ISC': all_subs_isc[rec],
        }
    
        for name, v in vectors.items():
            try:
                shape = v.shape
            except AttributeError:
                shape = (len(v),)
            print(f"{name:25s} length = {len(v):6d}, shape = {shape}")
    
        
        #normalize saccade_rate_sliding, verg_slidingrolling_dispersion, rolling_dispersion_std, rolling_blink_rate, rolling blink duration (replacing nans with 0s)
    
        rec_data = pd.DataFrame({
            'patient': [rec] * len(all_subs_saccade_rates[rec]),  # Repeat the recording ID for each time point
            'Saccade_Rate': all_subs_saccade_rates[rec],
            'Vergence': all_subs_sliding_vergence[rec],
            'Vergence_Std': all_subs_vergence_std[rec],
            'Abs_Vergence': all_subs_abs_vergence[rec],
            'Saccade_Dispersion': all_subs_saccade_dispersion[rec],
            'Saccade_Dispersion_Std': all_subs_saccade_dispersion_std[rec],
            'Blink_Rate': all_subs_blink_rate[rec],
            'Blink_Duration': all_subs_blink_duration[rec],
            'Pupil_Avg': all_subs_rolling_pupil[rec],
            'Pupil_Std': all_subs_rolling_pupil_std[rec],
            'ISC': all_subs_isc[rec],
        })
        
        # Save the Patient-Level DataFrame to a CSV file
        recording_eye_filename = os.path.join(fig_patient_dir, f'many_normed_et_features_{vid}_{pat}.csv')
        rec_data.to_csv(recording_eye_filename, index=False)
        print(f"Eye features saved to {recording_eye_filename}")
    
        eye_counts_row = pd.DataFrame({
            'recording': [rec],
            'count_saccades': [all_subs_count_saccades[rec]],
            'sum_sacc_distance': [all_subs_sum_sacc_distance[rec]],
            'mean_sacc_distance': [all_subs_mean_sacc_distance[rec]],
            'std_sacc_distance': [all_subs_std_sacc_distance[rec]],
            'count_blinks': [all_subs_count_blinks[rec]],
            'sum_vergence': [all_subs_sum_vergence[rec]],
            'mean_vergence': [all_subs_mean_vergence[rec]],
            'sum_abs_vergence': [all_subs_sum_abs_vergence[rec]],
            'mean_abs_vergence': [all_subs_mean_abs_vergence[rec]],
            'med_pupil_dilation': [all_subs_med_pupil_dilation[rec]],
            'pupil_variability': [all_subs_pupil_variability[rec]],
            'isc': [all_subs_isc_mean[rec]],
        })

        eye_desc_filename = os.path.join(fig_patient_dir, f'eye_descriptives_{vid}_{pat}.csv')
        eye_counts_row.to_csv(eye_desc_filename, index=False)
        print(f"Eye descriptives saved to {eye_desc_filename}")
        all_eye_counts.append(eye_counts_row)

    
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


#%
        # Normalize and replace NaNs with 0s
        for col in columns_to_normalize:
            if col in rec_data.columns:
                # Replace NaNs with 0 before normalization
                rec_data[col] = rec_data[col].fillna(0)
                
                # Normalize (z-score normalization)
                rec_data[col] = (rec_data[col] - rec_data[col].mean())/ rec_data[col].std()
                
    
        # Append this patient's data to the list
        all_eye_data.append(rec_data)
    
    # Concatenate all rows into a single DataFrame
    final_eye_df = pd.concat(all_eye_data, ignore_index=True)
    
    final_eye_counts_df = pd.concat(all_eye_counts,ignore_index=True)
    
    # Calculate the correlation matrix, excluding the first column
    correlation_matrix = final_eye_df.iloc[:, 1:].corr()  # Exclude the first column by slicing
    
    # Plot the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", cbar=True, square=True)
    
    # Add title and labels
    plt.title(f'Correlation Heatmap of Eye Movement Features {vid} {window_len}', fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    
    # Show the plot
    plt.show()
    
    # Save the DataFrame to a CSV file
    eye_feature_filename = os.path.join(fig_dir, f'many_et_features_{vid}.csv')
    final_eye_df.to_csv(eye_feature_filename, index=False)
    
    print(f"Eye features saved to {eye_feature_filename}")
    
    # Save the DataFrame to a CSV file
    eye_counts_filename = os.path.join(fig_dir, f'eye_counts_{vid}.csv')
    final_eye_counts_df.to_csv(eye_counts_filename, index=False)
    
    print(f"Eye counts saved to {eye_feature_filename}")

    # NORMALIZING THE ENTIRE DATA FRAME BEFORE PCA
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
      #  'ISC'
    ]
    
    normed_eye_df = final_eye_df.copy()
    
    # Normalize the entire dataset
    normed_eye_df[columns_to_normalize] = (normed_eye_df[columns_to_normalize] - 
                                         normed_eye_df[columns_to_normalize].mean()) / \
                                        normed_eye_df[columns_to_normalize].std()
    
    # Save the DataFrame to a CSV file
    normed_feature_filename = os.path.join(fig_dir, f'many_normed_et_features_for_pca_{vid}.csv')
    normed_eye_df.to_csv(normed_feature_filename, index=False)
