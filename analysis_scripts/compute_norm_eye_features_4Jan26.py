#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 18:55:15 2025

@author: christinechesebrough

"""

import os, re, sys
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
    Extract patient ID like 'NS127', 'NS127_02' or 'LH010' from an entry string.

    The NS-only pattern this used to carry silently excluded every LH patient
    (LH010, LH012 are hungarian-only) with a ValueError.
    """
    m = re.match(r'((?:NS|LH)\d+(?:_\d+)?)', entry)
    if not m:
        raise ValueError(f"Could not extract patient ID from: {entry}")
    return m.group(1)


def run_keys(run_label: str):
    """
    Both spellings a run label appears under in Eye_prep filenames:
    'run-01' (english, inscapes) and 'run-1' (hungarian).
    """
    n = int(run_label.split('-')[-1])
    return [f'run-{n:02d}', f'run-{n}']


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

machine_path = 'media/christine'   # as in the other scripts; was hardcoded /Volumes
sys.path.append(f'/{machine_path}/Samsung/scripts/movies_ET_attn_main/src')   # Spyder-safe (no __file__)
from eye_normalise import standardise_recording   # robust per-recording scale, defined once
drive = 'Samsung'
vid = "despicable_me_hungarian" #'dme' "inscapes"

window_len = 10
overlap = 7.5

# NORM_SCHEME (2026-09-14) - how the windowed features are standardised.
#   'legacy'          the original two-stage scheme: per-recording z of 8
#                     features (pupil, ISC untouched), then across-movie z of
#                     10 (ISC untouched). Removes each viewer's range as well
#                     as their level, for 8 of 11 features. Output dir
#                     6Apr26_norm_eye_features_by_rec_{win}s (unchanged).
#   'centred_global'  stage 1 CENTRES only, all 11 features (level removed,
#                     range kept); stage 2 scales every feature by its SD
#                     across all recordings of the movie (one yardstick per
#                     feature per movie). Preserves individual differences in
#                     attentional range. Output dir ..._centred_global.
#   'robust'          (2026-09-14) legacy structure with three stage-1 changes:
#                     (a) the 8 gaze features are standardised per recording
#                     as (x - median) / (1.4826 * MAD), SD fallback when the
#                     MAD is 0, because the SD is tail-inflated in 11 (Vergence)
#                     and 21 (Vergence_Std) of 53 recordings (SD up to 12x MAD);
#                     (b) missing windows are filled AFTER standardisation, at
#                     the recording median (0), instead of with a raw 0 before
#                     it (which put no-saccade windows 1-4 SD below the mean in
#                     Saccade_Dispersion); (c) windows with fewer than
#                     GAZE_VALID_MIN valid gaze samples are treated as having
#                     NO gaze information: all 8 gaze features are set missing
#                     (-> 0) and the window is flagged by Gaze_Valid_Frac, so
#                     downstream can exclude them. A window with valid gaze
#                     but no saccades keeps a missing dispersion (-> 0);
#                     Saccade_Rate = 0 carries that information.
#                     Pupil is left alone in stage 1 (already rescaled per
#                     individual upstream); ISC follows ISC_MODE. Stage 2 is
#                     the legacy across-movie z. Output dir ..._robust.
NORM_SCHEME = 'legacy'
ISC_MODE = 'raw'        # 'raw'        ISC untouched (legacy; SD ~0.14 vs 1 for the
                        #              z-scored features, so it barely enters the PCA)
                        # 'within_rec' robust z per recording like the gaze features
                        # (only used by NORM_SCHEME = 'robust')
GAZE_VALID_MIN = 0.5    # 'robust' only: min fraction of finite gaze samples in a
                        # 10 s window for its gaze features to count as observed
# Vergence from MEASURED samples only (2026-09-14, 'robust' scheme). The vergence
# file's gaze_dist_x_interp fills every sample where one or both eyes were
# missing by linear interpolation; those fills produce runs of outlying
# disparity up to ~3 s long (NS164 Hungarian: 76 % of outlying samples are
# interpolated). Christine's rule: treat ONLY interpolated samples; a measured
# value stands even when extreme. So interpolated samples (NaN in the raw
# dva_gaze_disp_x column) are excluded from the window mean / SD, and a
# window with fewer than VERG_MIN_MEASURED of its samples measured gets NaN
# (-> recording median after standardisation). Verg_Valid_Frac records the
# measured fraction per window. Legacy scheme: unchanged (interpolated
# series, plain mean / SD).
VERG_MEASURED_ONLY = True   # applies under NORM_SCHEME == 'robust' only
VERG_MIN_MEASURED = 0.3     # min measured fraction of a window for a vergence value (~900 samples)
VERG_MIN_REC_MEDIAN = 0.5   # if a recording's MEDIAN measured fraction is below this, vergence is
                            # unmeasurable for that recording: all three vergence features are set
                            # missing (-> recording median = 0, no vergence information) instead of
                            # letting a mostly-median, bimodal series into the PCA. Measured
                            # 2026-09-14: 5 of 53 recordings (NS190 r1, NS194, NS174_03 English;
                            # NS155, NS210 Inscapes) at 0.5.
ALL_FEATURES = ['Saccade_Rate', 'Vergence', 'Vergence_Std', 'Abs_Vergence', 'Saccade_Dispersion',
                'Saccade_Dispersion_Std', 'Blink_Rate', 'Blink_Duration', 'Pupil_Avg', 'Pupil_Std', 'ISC']

compute_measures = True
#'despicable_me_english'  # inscapes

# Check if final_eye_df exists in memory, if so delete
if 'final_eye_df' in locals():
    del final_eye_df
    
# Check if normed_eye_df exists in memory, if so delete
if 'normed_eye_df' in locals():
    del normed_eye_df

region = 'all'
movie_data_dir = f'/{machine_path}/{drive}/Movie_data'
data_dir = f'{movie_data_dir}/movies_prep_standard'
isc_dir = f'{movie_data_dir}/data/isc'
mne_data_dir = f'{movie_data_dir}/movies_prep_standard'
elec_dir = f'{movie_data_dir}/data/electrode_localization'
# movie_subs_master_updated.csv was loaded here but never used; it lives on the
# Data drive (/{machine_path}/Data/anatomy/shared_correspondence/), not Samsung.

fig_dir = f'{movie_data_dir}/6Apr26_norm_eye_features_by_rec_{window_len}s' + ('' if NORM_SCHEME == 'legacy' else f'_{NORM_SCHEME}')

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

   good_ET_runs = ['LH010_ses-01_run-01',      # 92.9/92.8% gaze present; excluded before
                                              # only by the NS-only patient regex
                'NS127_02_ses-02_run-01', 
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
        data = np.load(f'{movie_data_dir}/ISC_despicable_me_english_5s_windows/despicable_me_english_isc_gaze_position_time_updated.npz')
    elif window_len == 10:
        data = np.load(f'{movie_data_dir}/ISC_despicable_me_english_30Dec25/ISC_despicable_me_english_30Dec25_despicable_me_english_isc_gaze_position.npz',allow_pickle = True)

if vid in ['dmh', 'despicable_me_hungarian']:
    if window_len == 5:
        data = np.load(f'{movie_data_dir}/ISC_{vid}_5s_windows/{vid}_isc_gaze_position_time_updated.npz')
    elif window_len == 10:
        data = np.load(f'{movie_data_dir}/ISC_{vid}_10s_windows/{vid}_isc_gaze_position_time_updated.npz',allow_pickle = True)
if vid == 'inscapes':
    if window_len == 5:
        data = np.load(f'{movie_data_dir}/ISC_inscapes_5s_windows/inscapes_isc_gaze_position_time_updated.npz')
    elif window_len == 10:
        data = np.load(f'{movie_data_dir}/ISC_inscapes_30Dec25/ISC_inscapes_30Dec25_inscapes_isc_gaze_position.npz',allow_pickle = True)


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
    all_subs_gaze_valid_frac = {}   # fraction of finite gaze samples per window
    all_subs_verg_valid_frac = {}   # fraction of MEASURED (both eyes) disparity samples per window
        
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
        # ISC entry_ids differ by video: english/inscapes carry the session
        # ('NS127_02_ses-02_run-01'), hungarian does not ('NS127_02_run-01').
        # The previous test `vid == 'despicable_me_english' or 'inscapes'` was
        # always True and only worked because the hungarian branch overrode it.
        if window_len == 5 or vid == 'despicable_me_hungarian':
            run_id = f'{pat}_{run}'
        else:
            run_id = f'{pat}_{ses}_{run}'

        idx_rec = np.in1d(entry_id_isc, run_id)
        match_idx = np.where(idx_rec)[0]
        if len(match_idx) == 0:
            raise KeyError(
                f"{run_id} not found in ISC entry_ids for {vid}; "
                f"examples: {list(entry_id_isc[:5])}"
            )
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
            npz_candidates = [f for f in npz_candidates if any(k in f for k in run_keys(run))]
        
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
            verg_candidates = [f for f in verg_candidates if any(k in f for k in run_keys(run))]
        
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
            blink_candidates = [f for f in blink_candidates if any(k in f for k in run_keys(run))]
        
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
        verg_measured = np.isfinite(verg_dat['dva_gaze_disp_x'].values)   # False where the sample was interpolated
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
        verg_measured = verg_measured[idx_vid]
        
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
        verg_valid_frac = np.full(num_steps, np.nan)   # measured (both-eyes-valid) fraction per window
    
        # Time array for sliding window midpoints (aligned to time_isc)
        isc_aligned_time = np.arange(window_samples / 2, len(vergence) - window_samples / 2 + 1, step_size_samples) / 300  # Convert to seconds
        
        # Compute rolling averages
        for i in range(num_steps):
            window_start_idx = int(i * step_size_samples)  # Ensure integer index
            window_end_idx = int(window_start_idx + window_samples)  # Ensure integer index
        
            # Extract data within the current window
            window_data = np.asarray(vergence[window_start_idx:window_end_idx], dtype=float)
            meas = verg_measured[window_start_idx:window_end_idx]
            verg_valid_frac[i] = meas.mean() if len(meas) else np.nan

            if NORM_SCHEME == 'robust' and VERG_MEASURED_ONLY:
                # measured samples only; too few measured -> NaN (filled at the
                # recording median after standardisation)
                if verg_valid_frac[i] >= VERG_MIN_MEASURED and meas.sum() > 1:
                    verg_sliding[i] = np.nanmean(window_data[meas])
                    verg_sliding_std[i] = np.nanstd(window_data[meas])
                else:
                    verg_sliding[i] = np.nan
                    verg_sliding_std[i] = np.nan
            else:
                # legacy: interpolated series, plain mean / SD
                verg_sliding[i] = np.mean(window_data)
                verg_sliding_std[i] = np.std(window_data)
    
        abs_sliding_vergence = np.abs(verg_sliding)
        if NORM_SCHEME == 'robust' and VERG_MEASURED_ONLY:
            print(f"Vergence: {int(np.sum(~np.isfinite(verg_sliding)))} windows below VERG_MIN_MEASURED "
                  f"(median measured fraction {np.nanmedian(verg_valid_frac):.3f})")
        
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
                verg_valid_frac = verg_valid_frac[:-1]
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
                verg_valid_frac = np.interp(x_new, x_old, verg_valid_frac)
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

        # Fraction of gaze samples in each window with a finite position: the
        # per-window record of whether the tracker saw the eye at all. Written
        # out as Gaze_Valid_Frac; NORM_SCHEME = 'robust' uses it to tell a
        # window with NO gaze data from one with gaze but no saccades.
        xy_arr = np.asarray(xy, dtype=float)
        gaze_ok = np.isfinite(xy_arr).all(axis=1) if xy_arr.ndim == 2 else np.isfinite(xy_arr)
        gaze_valid_frac = np.full(num_steps, np.nan)
        for i in range(num_steps):
            window_start_idx = int(i * step_size_samples)
            window_end_idx = int(window_start_idx + window_samples)
            seg = gaze_ok[window_start_idx:window_end_idx]
            if len(seg):
                gaze_valid_frac[i] = float(np.mean(seg))
        all_subs_gaze_valid_frac[rec] = gaze_valid_frac
        print(f"Gaze valid fraction: median {np.nanmedian(gaze_valid_frac):.3f}, "
              f"{int(np.sum(gaze_valid_frac < GAZE_VALID_MIN))} windows below {GAZE_VALID_MIN}")
    
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
        # t_pupil is on a different clock and sampling rate from t (gaze), but
        # both were zeroed at t_start above, so a window defined in SECONDS is
        # the same window on either series. Every other feature uses window i
        # = [i * step_sec, i * step_sec + window_len); do the same here.
        #
        # Previously pupil used its own grid (step = n_samples / 237, window =
        # 2 x step, ~50% overlap), so Pupil_Avg / Pupil_Std did not describe
        # the same 10 s the other columns in the row describe.

        step_sec = window_len - overlap

        rolling_pupil_avg = np.full(num_steps, np.nan)
        rolling_pupil_std = np.full(num_steps, np.nan)

        t_pupil = np.asarray(t_pupil, dtype=float)
        pupil = np.asarray(pupil, dtype=float)

        for i in range(num_steps):
            w0 = i * step_sec
            w1 = w0 + window_len
            j0 = np.searchsorted(t_pupil, w0, side='left')
            j1 = np.searchsorted(t_pupil, w1, side='left')
            if j1 > j0:
                seg = pupil[j0:j1]
                rolling_pupil_avg[i] = np.nanmean(seg)
                rolling_pupil_std[i] = np.nanstd(seg)

        n_empty = int(np.isnan(rolling_pupil_avg).sum())
        if n_empty:
            print(f"WARNING: {n_empty} pupil windows had no samples for {rec}")

        print(f"Length of rolling pupil avg: {len(rolling_pupil_avg)}")


        #### COMBINE ALL DATA ###
        
        all_subs_saccade_rates[rec] = saccade_rate_sliding  # Store the saccade rates
        all_subs_sliding_vergence[rec] = verg_sliding  # Store vergence rates
        all_subs_vergence_std[rec] = verg_sliding_std
        all_subs_abs_vergence[rec] = abs_sliding_vergence
        all_subs_verg_valid_frac[rec] = verg_valid_frac
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
        all_subs_sum_sacc_distance[rec] = sum_sacc_distance
        all_subs_mean_sacc_distance[rec] = mean_sacc_distance
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
        'Gaze_Valid_Frac': all_subs_gaze_valid_frac[rec],
        'Verg_Valid_Frac': all_subs_verg_valid_frac[rec],
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
            'Gaze_Valid_Frac': all_subs_gaze_valid_frac[rec],
            'Verg_Valid_Frac': all_subs_verg_valid_frac[rec],
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
        if NORM_SCHEME == 'centred_global':
            # stage 1: centre only, ALL features; a missing window becomes
            # "at this viewer's average" (0 after centring), not raw zero
            for col in ALL_FEATURES:
                rec_data[col] = rec_data[col] - rec_data[col].mean(skipna=True)
                rec_data[col] = rec_data[col].fillna(0)
        elif NORM_SCHEME == 'robust':
            # stage 1: robust z per recording; see the NORM_SCHEME comment.
            no_gaze = rec_data['Gaze_Valid_Frac'] < GAZE_VALID_MIN
            rec_data.loc[no_gaze, columns_to_normalize] = np.nan
            if VERG_MEASURED_ONLY and rec_data['Verg_Valid_Frac'].median() < VERG_MIN_REC_MEDIAN:
                rec_data[['Vergence', 'Vergence_Std', 'Abs_Vergence']] = np.nan
                print(f"  {rec}: median measured vergence fraction {rec_data['Verg_Valid_Frac'].median():.2f} "
                      f"< VERG_MIN_REC_MEDIAN -> vergence features set missing for the whole recording")
            robust_cols = columns_to_normalize + (['ISC'] if ISC_MODE == 'within_rec' else [])
            rec_data = standardise_recording(rec_data, robust_cols, scheme='robust', verbose_name=rec)
            print(f"  {rec}: {int(no_gaze.sum())} windows below GAZE_VALID_MIN set to missing; "
                  f"{int(rec_data[columns_to_normalize].eq(0).all(axis=1).sum())} windows at 0 on all gaze features")
        else:
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
    if NORM_SCHEME == 'centred_global':
        # stage 2: one scale per feature per movie, ALL features incl. ISC;
        # no re-centring (recordings are already centred), so each viewer's
        # range is preserved relative to the movie-typical fluctuation
        normed_eye_df[ALL_FEATURES] = normed_eye_df[ALL_FEATURES] / normed_eye_df[ALL_FEATURES].std()
    else:
        # legacy and robust: z across the movie for the 10 (ISC untouched).
        # After stage 1 this is near-identity for the standardised features
        # and only matters for pupil.
        normed_eye_df[columns_to_normalize] = (normed_eye_df[columns_to_normalize] - 
                                             normed_eye_df[columns_to_normalize].mean()) / \
                                            normed_eye_df[columns_to_normalize].std()
    
    # Save the DataFrame to a CSV file
    normed_feature_filename = os.path.join(fig_dir, f'many_normed_et_features_for_pca_{vid}.csv')
    normed_eye_df.to_csv(normed_feature_filename, index=False)
