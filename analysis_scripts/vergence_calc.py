#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 14:21:58 2024

@author: christinechesebrough
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 12:39:52 2024

@author: christinechesebrough
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 14:19:08 2024

@author: christinechesebrough
"""

# Calculate differences in gaze positions between right and left eye (gaze disparity, disparity of visual angle)

import os, sys, re
import numpy as np
from pynwb import NWBHDF5IO

import pandas as pd
import matplotlib.pyplot as plt
import scipy.interpolate as interp
import scipy.signal as signal

import importlib.util

# Define the path to helpers.py
helper_path = '/Volumes/Samsung/scripts/eyetracking_process/helpers.py'

# Load the module
spec = importlib.util.spec_from_file_location("helpers", helper_path)
helpers = importlib.util.module_from_spec(spec)
sys.modules["helpers"] = helpers
spec.loader.exec_module(helpers)

# Now you can use its functions:
sys.path.append('/Users/christinechesebrough/Documents/EPIPE-movie_nwb/Python')

 #%%

def interp_bad_data(data_gaze, data_val, k_small, k_large, k_delay, visualize):
   
    """
    Interpolate bad samples in gaze data
    
    Inputs:
        data_gaze       - data array [samples x dimension (xy)]
        data_val        - index of samples with invalid data [samples]
        k_small         - number of samples of small gaps that are filled
        k_large         - number of samples to add to large gaps (blinks, etc)
        k_delay         - number of samples to shift delay around large gaps
        visualize       - flag to plot gaze data before and after processing
        
    Returns:
        data_gaze       - processed data vector
        data_val        - upadted vector of invalid data
    """
    
   # data_val = copy.copy(data_val)
    data_val = data_val.copy()
    
    # Plot signal
    if visualize:
        plt.figure()
        plt.plot(data_gaze)
    
    # Create sample vector
    samples = np.arange(0, len(data_gaze))
    
    # Interpolate small gaps
    idx_gap = signal.convolve(data_val, np.ones(k_small), 'same')
    gap_val = signal.convolve(idx_gap==0, np.ones((k_small)), mode='same') == 0
    
    gaps = np.logical_and(gap_val, np.invert(data_val))
    data_val[gaps] = True
    
    f_gaze = interp.interp1d(samples[np.invert(gaps)],
                             data_gaze[np.invert(gaps)], 
                             kind='linear', fill_value="extrapolate")
    data_gaze[gaps] = f_gaze(samples[gaps])
    
    # Remove samples around longer gaps (mostly blinks)
    idx_bad = signal.convolve(np.invert(data_val), 
                              np.concatenate([np.zeros(k_delay),
                                              np.ones(k_large)]), 
                              'same') > 1
    
    data_gaze[idx_bad] = np.nan
    data_val[idx_bad] = False
    
    if visualize:
        plt.plot(data_gaze)
        
    return data_gaze, data_val

def detect_blinks(val_left, val_right, t_nwb, fs_eye, min_duration=0.09, max_duration=0.5):
    """
    Detect blinks based on validity values for left and right eyes.
    
    Inputs:
        val_left       - Validity array for left eye (True = valid, False = invalid).
        val_right      - Validity array for right eye (True = valid, False = invalid).
        t_nwb          - Timestamps for gaze data.
        fs_eye         - Sampling rate of eye-tracking data (Hz).
        min_duration   - Minimum duration of a blink (in seconds).
        max_duration   - Maximum duration of a blink (in seconds).
    
    Returns:
        blink_events   - List of blink events with start and end times and durations.
    """
    # Combine validity arrays to find invalid periods
    invalid_data = ~val_left & ~val_right  # Both eyes invalid
    
    # Identify start and end indices of invalid periods
    invalid_starts = np.where(np.diff(invalid_data.astype(int)) == 1)[0] + 1
    invalid_ends = np.where(np.diff(invalid_data.astype(int)) == -1)[0] + 1

    # Handle case where the data starts or ends with invalid periods
    if invalid_data[0]:
        invalid_starts = np.insert(invalid_starts, 0, 0)
    if invalid_data[-1]:
        invalid_ends = np.append(invalid_ends, len(invalid_data) - 1)

    # Calculate durations of invalid periods
    durations = (invalid_ends - invalid_starts) / fs_eye

    # Classify blinks based on duration thresholds
    blink_events = []
    for start, end, duration in zip(invalid_starts, invalid_ends, durations):
        if min_duration <= duration <= max_duration:
            blink_events.append({
                "start_time": t_nwb[start],
                "end_time": t_nwb[end],
                "duration": duration
            })
    
    return blink_events


def save_blink_events(blink_events, output_dir, pat, mov):
    df_blinks = pd.DataFrame(blink_events)
    os.makedirs(output_dir, exist_ok=True)

    # Hard guard: output_dir must contain the subject string
    if pat not in os.path.normpath(output_dir).split(os.sep):
        raise ValueError(f"Refusing to save: output_dir ({output_dir}) does not match patient ({pat}).")

    base, _ = os.path.splitext(mov)
    blink_output_filename = os.path.join(output_dir, base + '_blink_events.csv')
    #base = os.path.splitext(os.path.basename(movie))[0]
    #out = os.path.join(output_dir, f"{pat}_{base}_blink_events.csv")
    df_blinks.to_csv(blink_output_filename, index=False) 
    print(f"Blink events saved to {blink_output_filename}")
    
    
def interp_nans_maxgap(x, max_gap=30):
    """
    Linearly interpolate NaNs only when the NaN-run length <= max_gap.
    Otherwise leave as NaN.
    max_gap is in samples.
    """
    x = np.asarray(x, float)
    isn = ~np.isfinite(x)
    if not isn.any():
        return x

    # Identify contiguous NaN runs
    idx = np.arange(len(x))
    s = pd.Series(isn.astype(int))
    # run id increments at each change
    run_id = (s.diff().fillna(0).ne(0)).cumsum()
    run_lengths = s.groupby(run_id).sum()  # counts of NaNs in each run
    # map each position to its run length (0 for non-NaN positions)
    run_len_per_pos = run_id.map(run_lengths).to_numpy()
    short_gap = isn & (run_len_per_pos <= max_gap)

    # Interpolate globally, then re-mask long gaps
    y = x.copy()
    good = np.isfinite(y)
    y[~good] = np.interp(idx[~good], idx[good], y[good])
    y[isn & ~short_gap] = np.nan
    return y


def mad(x):
    x = x[np.isfinite(x)]
    med = np.median(x)
    return np.median(np.abs(x - med))

def mask_spikes_by_derivative(x, z=8.0):
    x = np.asarray(x, float)
    dx = np.diff(x)
    m = mad(dx)
    if m == 0 or not np.isfinite(m):
        return x
    thresh = z * 1.4826 * m  # robust sigma
    spike = np.zeros_like(x, dtype=bool)
    spike[1:] = np.abs(dx) > thresh
    y = x.copy()
    y[spike] = np.nan
    return y



def derive_vmax_amax_gap_safe(x, y, valid, fs, pad=10, q_v=0.999, q_a=0.999):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    valid = np.asarray(valid, bool)

    # Ensure we only use finite samples marked valid
    good = valid & np.isfinite(x) & np.isfinite(y)

    # Identify samples "too close" to invalid segments (dilate invalid mask)
    bad = ~good
    if pad > 0:
        kernel = np.ones(2*pad + 1, dtype=int)
        bad_padded = np.convolve(bad.astype(int), kernel, mode="same") > 0
        good2 = good & (~bad_padded)
    else:
        good2 = good

    # Finite differences, only where both samples are good2
    dx = np.diff(x)
    dy = np.diff(y)
    pair_good = good2[:-1] & good2[1:]

    speed = np.sqrt(dx[pair_good]**2 + dy[pair_good]**2) * fs  # units/s

    # Acceleration from speed diffs, need adjacent speeds that come from adjacent valid pairs
    # Build a speed array aligned to original "between-sample" indices to preserve adjacency
    speed_full = np.full(len(x)-1, np.nan)
    speed_full[pair_good] = np.sqrt(dx[pair_good]**2 + dy[pair_good]**2) * fs

    ds = np.diff(speed_full)
    accel = np.abs(ds) * fs
    accel = accel[np.isfinite(accel)]

    # Drop nans from speed
    speed = speed[np.isfinite(speed)]

    stats = {
        "speed_max": float(np.max(speed)) if speed.size else np.nan,
        "accel_max": float(np.max(accel)) if accel.size else np.nan,
        "speed_q": float(np.quantile(speed, q_v)) if speed.size else np.nan,
        "accel_q": float(np.quantile(accel, q_a)) if accel.size else np.nan,
        "n_speed": int(speed.size),
        "n_accel": int(accel.size),
        "pad": pad,
    }

    # Conservative multiplier
    vmax = stats["speed_q"] * 1.2
    amax = stats["accel_q"] * 1.2
    return vmax, amax, stats



def clean_xy(x, y, valid, lo=-0.25, hi=1.25):
    x = x.astype(float).copy()
    y = y.astype(float).copy()
    valid = valid.astype(bool).copy()

    # invalid -> NaN
    x[~valid] = np.nan
    y[~valid] = np.nan

    # out of bounds -> invalidate and NaN
    oob = (~np.isfinite(x)) | (~np.isfinite(y)) | (x < lo) | (x > hi) | (y < lo) | (y > hi)
    valid[oob] = False
    x[oob] = np.nan
    y[oob] = np.nan

    return x, y, valid

def dilate_invalid(valid, pad):
    valid = np.asarray(valid, bool)
    invalid = ~valid
    kernel = np.ones(2*pad + 1, dtype=int)
    invalid_dil = signal.convolve(invalid.astype(int), kernel, mode="same") > 0
    return ~invalid_dil  # new valid

#%% Eye tracking parameters

# Frequency range
freq_range = [70, 170]
n_freq_bins = 10
freq_space = 'log'      # 'log', 'lin'

resample_bha_fs = 100
t_diff = 0.1

dist = 60

# Screen dimentions [cm]
d = 23.8*2.54
ar = 16/9

k_small = 15 #15
k_large = 90 #90
k_delay = 20
vis_bads = False

screen_pix = np.array([1920, 1080])
screen_cm = np.array([50.92, 28.64])

calc_vis_displacement = True

edge_pads = [30,60]
    
#%%
drive = 'Samsung'
data_dir = '/Volumes/Samsung/Movie_data/movies_nwb_standard'
fs_dir = '/Volumes/Samsung/anatomy'
#prep_dir = '/Volumes/Samsung/Movie_data/movies_prep_standard'
prep_dir = '/Volumes/Samsung/Movie_data/movies_prep_standard'


et_prep_dir = 'Eye_prep'
et_qual_dir = 'quality_figs'

movie = 'despicable_me_hungarian' #'inscapes'#

if movie == 'despicable_me_english':
    movie_keys = ['despicable_me_english','dme']
if movie == 'despicable_me_hungarian':
    movie_keys = ['despicable_me_hungarian','dmh']
elif movie == 'inscapes':
    movie_keys = ['inscapes']

#%%  
broken_nwb = [] 

#patients = [patient for patient in os.listdir(data_dir) if not patient.startswith('.DS_Store')]

if movie == 'despicable_me_english':
    good_ET_pats = ['NS127','NS135','NS136','NS137','NS138','NS140','NS153','NS151','NS154','NS164','NS166','NS174','NS191','NS193','NS194','NS178','NS201','NS204','NS205']
elif movie == 'despicable_me_hungarian':
    good_ET_pats = [
    # 'LH010',
    # 'NS127',
    # 'NS128',
    # 'NS135',
    # 'NS136',
    # 'NS137',
    # 'NS138',
    # 'NS140',
    # 'NS140',
    # 'NS144',
    # 'NS145',
    # 'NS151',
    # 'NS153',
    # 'NS154',
    # 'NS164',
    # 'NS166',
    # 'NS167',
    'NS174',
    # 'NS174_02',
    # 'NS174_03',
    'NS178'
    ]
    
elif movie == 'inscapes':
     good_ET_pats = ['NS127','NS135','NS136','NS137','NS138','NS140','NS144',
                     'NS151','NS153','NS154','NS155','NS164','NS178','NS205','NS210','NS211']
     # 140 and 140_02 for both dm and inscapes
     #good_ET_pats = ['NS127','NS135','NS136','NS137','NS138',"NS153","NS205", 'NS210','NS211']
  #  good_ET_pats = ["NS153", "NS205", "NS210"]
     #good_ET_pats = ['NS205']


patients = good_ET_pats

patients.sort()

#%%
for pat in patients:
    directory_path = os.path.join(data_dir, pat)
    all_files = os.listdir(directory_path)
    
    implants = [filename for filename in all_files if os.path.isdir(os.path.join(directory_path, filename)) and not filename == '.DS_Store']
    
    for imp in implants:
    
        num_imp = int(re.findall(r'\d+', imp)[0])
        
        
        if num_imp == 1:
            pat_fs = pat.replace('sub-', '')
        elif num_imp >= 2:
            pat_fs = f'{pat.replace("sub-", "")}_{num_imp:02d}'
        
            # Pupil data - save to the same directory structure that compute_eye_measures.py expects
        sub_et_prep_dir = f'{prep_dir}/{pat_fs}/{et_prep_dir}'
        sub_et_qual_fig_dir = f'{sub_et_prep_dir}/{et_qual_dir}'

        if not os.path.exists(sub_et_prep_dir):
            os.makedirs(sub_et_prep_dir)
        
        if not os.path.exists(sub_et_qual_fig_dir):
            os.makedirs(sub_et_qual_fig_dir)

        movies = os.listdir('{:s}/{:s}/{:s}'.format(data_dir, pat, imp))    

        # Filter movies using any key
        filtered_movies = [mov for mov in movies 
                           if any(k in mov for k in movie_keys)
                           if mov.endswith(".nwb")
                           and not mov.startswith("._")
                           and not mov.startswith(".")]

        
        
        # Handle no matches
        if not filtered_movies:
            print(f"No movie file with keys {movie_keys} found in the directory.")
            continue
        #
        # Process the filtered movies
        for mov in filtered_movies:
                
            et_prep_filename = os.path.join(
                sub_et_prep_dir,
                mov.replace('_ieeg.nwb', '_et_prep.csv')
            )
        
            nwb_fname = os.path.join(data_dir, pat, imp, mov)

            #
             # NWB read       
            io = NWBHDF5IO(nwb_fname, mode='r', load_namespaces=True)
            nwb = io.read()
            print(f'Loading nwb data for patient {pat} recording name {mov}...')
             
            #  Preprocess and visualize eyetracking data
             
            # Get position and time
            # adcs is gaze position on screen 2D vector x, y between 0 and 1
            l_gaze = nwb.processing['eye_tracking']['eyes']['l_eye_adcs'].data[:]
            r_gaze = nwb.processing['eye_tracking']['eyes']['r_eye_adcs'].data[:]
              
            x_right = r_gaze[:,0]
            y_right = r_gaze[:,1]
              
            x_left = l_gaze[:,0]
            y_left = l_gaze[:,1]
            
            # a, b, c, points in space tracking eye location in three dimensions    
            eye_pos_left = nwb.processing['eye_tracking']['eyes']['l_eye_pos'].data[:]
            eye_pos_right = nwb.processing['eye_tracking']['eyes']['r_eye_pos'].data[:]
             
            # same point as gaze position on screen but measured from origin at the eye tracker
            gaze_pos_left = nwb.processing['eye_tracking']['eyes']['l_eye_gaze'].data[:]
            gaze_pos_right = nwb.processing['eye_tracking']['eyes']['r_eye_gaze'].data[:]
             
            # Calculate the approximate interpupillary distance 
             
            # Calculate the Euclidean distance between corresponding pairs of eye position points
            distances = np.sqrt(np.sum((eye_pos_left - eye_pos_right) ** 2, axis=1))
        
            # The IPD can be approximated as the median of the Euclidean distance between the two eye positions in space.
            IPD_cm = np.nanmedian(distances) / 10  # If distances are in mm, convert to cm.
             
            # time
            t_nwb = nwb.processing['eye_tracking']['eyes']['l_eye_adcs'].timestamps[:]
                        
            # # Sampling rate
            fs_eye = 1/np.median(np.diff(t_nwb))
            
            # # Validity
            val_left = nwb.processing['eye_tracking']['eyes']['l_eye_adcs'].control[:] 
            val_right = nwb.processing['eye_tracking']['eyes']['r_eye_adcs'].control[:]
            
            val_left = nwb.processing['eye_tracking']['eyes']['l_eye_adcs'].control[:] <= 1
            val_right = nwb.processing['eye_tracking']['eyes']['r_eye_adcs'].control[:] <= 1
            # comparison_array = np.where(val_left == val_right, 0, 1)
            
            #Mark samples that aren't valid on both eyes as NaN
            both_valid = val_left & val_right
#
            x_left_clean = x_left.astype(float).copy()
            x_right_clean = x_right.astype(float).copy()
            y_left_clean = y_left.astype(float).copy()
            y_right_clean = y_right.astype(float).copy()
            x_left_clean[~both_valid]  = np.nan
            x_right_clean[~both_valid] = np.nan
            y_left_clean[~both_valid]  = np.nan
            y_right_clean[~both_valid] = np.nan
            
            if pat == 'NS210':
                k_small = 100 #15
                k_large = 120 #90
                k_delay = 20

                x_left_clean_interp, _ = interp_bad_data(x_left_clean,val_left, k_small, k_large, k_delay, vis_bads)
                x_right_clean_interp, _ = interp_bad_data(x_right_clean,val_right, k_small, k_large, k_delay, vis_bads)
            else:
                x_left_clean_interp, _ = interp_bad_data(x_left_clean,val_left, k_small, k_large, k_delay, vis_bads)
                x_right_clean_interp, _ = interp_bad_data(x_right_clean,val_left, k_small, k_large, k_delay, vis_bads)

            # Detect blinks based on validity arrays
            blink_events = detect_blinks(val_left, val_right, t_nwb, fs_eye)
            save_blink_events(blink_events, sub_et_prep_dir, pat_fs, mov)


            # mark gaze values out of bounds as invalid
            lower_bound, upper_bound = -0.25, 1.25

            both_valid = (val_left & val_right).astype(bool)
            
            # Start from raw adcs (0–1) and invalidate oob in either eye
            # v0 is now "good baseline validity": both eyes valid AND within bounds
            
            xL, yL, valL = clean_xy(x_left,  y_left,  both_valid, lower_bound, upper_bound)
            xR, yR, valR = clean_xy(x_right, y_right, both_valid, lower_bound, upper_bound)
            
            # normalize to 0 min to allow for allowed negative gaze positions to be positive 
            xmin = np.nanmin([np.nanmin(xL), np.nanmin(xR)])
            xL -= xmin
            xR -= xmin
            
            ymin = np.nanmin([np.nanmin(yL), np.nanmin(yR)])
            yL -= ymin
            yR -= ymin

            bias = np.nanmedian((xL - xR))
            
            d_corr = (xL - xR) - bias
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(t_nwb,d_corr,color = 'red')
           # ax.plot(xL,yL,color = 'blue')
            #ax.plot(t_nwb,xR)
            ax.set_ylim(-.3, 1.5)   # or whatever fixed range you want
            plt.xlabel('x pos')
            plt.ylabel('y pos')
            plt.title(f'{pat} gaze bias over time')
            ax.legend()
            fig.tight_layout()

#%%
            vmax_left,  amax_left,  statsL = derive_vmax_amax_gap_safe(xL, yL, valL, fs=fs_eye, pad=30, q_v=0.999, q_a=0.999)
            vmax_right, amax_right, statsR = derive_vmax_amax_gap_safe(xR, yR, valR, fs=fs_eye, pad=30, q_v=0.999, q_a=0.999)
            
            vmax_left,  amax_left,  statsL   
            vmax_right, amax_right, statsR
            
            if pat in ['NS210','NS204']:
                edge_pad = 10
            elif pat == 'NS140':
                edge_pad = 50
            elif pat == 'NS164':
                edge_pad = 0
            else:    
                edge_pad = 30  # 50 ms at 600 Hz; consider 60 (100 ms) if needed
                
            valL2 = dilate_invalid(valL, edge_pad)
            xL2, yL2, _ = clean_xy(xL, yL, valL2, lower_bound, upper_bound)
            
            valR2 = dilate_invalid(valR, edge_pad)
            xR2, yR2, _ = clean_xy(xR, yR, valR2, lower_bound, upper_bound)
            
            ## tryign to smooth the raw signal
            
            from scipy.signal import medfilt

            x_left_med = medfilt(x_left_clean, kernel_size=7)
            x_right_med = medfilt(x_right_clean, kernel_size=7)
              #%%
            # def hampel(x, win, nsig):
            #     x = x.copy()
            #     for i in range(win, len(x)-win):
            #         window = x[i-win:i+win]
            #         med = np.nanmedian(window)
            #         mad = 2.5 * np.nanmedian(np.abs(window - med))
            #         if np.abs(x[i] - med) > nsig * mad:
            #             x[i] = med
            #     return x
            
            # x_left_hamp = hampel(x_left_clean, win=30, nsig=3)
            # x_right_hamp = hampel(x_right_clean, win=30, nsig=3)


            # win = int(0.3 * fs_eye) | 1  # ~300 ms
            # x_left_tonic = signal.savgol_filter(x_left_med, win, polyorder=2)
            # x_right_tonic = signal.savgol_filter(x_right_med, win, polyorder=2)

            #plot gaze position of right and left eye
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(xR[0:8000],yR[0:8000],color = 'red',linewidth = 5)
            ax.plot(xL[0:8000],yL[0:8000],color = 'yellow',linewidth = 5)
            #ax.plot(t_nwb,xR)
            ax.set_ylim(-.3, 1.5)   # or whatever fixed range you want
            plt.xlabel('x pos')
            plt.ylabel('y pos')
            plt.title(f'{pat}right eye gaze position')
            ax.legend()
            fig.tight_layout()
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(xR,yR,color = 'red')
            ax.plot(xL,yL,color = 'blue')
            #ax.plot(t_nwb,xR)
            ax.set_ylim(-.3, 1.5)   # or whatever fixed range you want
            plt.xlabel('x pos')
            plt.ylabel('y pos')
            plt.title(f'{pat}right eye gaze position')
            ax.legend()
            fig.tight_layout()

            # fig, ax = plt.subplots(figsize=(12, 6))
            # #ax.plot(t_nwb,xL)
            # #ax.plot(t_nwb,xR)
            # ax.plot(t_nwb,xL)
            # ax.plot(t_nwb,xR)
            # ax.set_ylim(-.3, 1.5)   # or whatever fixed range you want
            # plt.xlabel('Time')
            # #plt.ylabel('')
            # plt.title(f'{pat}right and left eye before gap removal')
            # ax.legend()
            # fig.tight_layout()
            
            # fig, ax = plt.subplots(figsize=(12, 6))
            # #ax.plot(t_nwb,xL)
            # #ax.plot(t_nwb,xR)
            # ax.plot(t_nwb,x_left_clean_interp)
            # ax.plot(t_nwb,x_right_clean_interp)
            # ax.set_ylim(-.3, 1.5)   # or whatever fixed range you want
            # plt.xlabel('Time')
            # #plt.ylabel('')
            # plt.title(f'{pat}right and left eye after original interpolation')
            # ax.legend()
            # fig.tight_layout()
            
           #  fig, ax = plt.subplots(figsize=(12, 6))
           # # ax.plot(t_nwb,x_left_copy)
           # # ax.plot(t_nwb,x_right_copy)
           #  ax.plot(t_nwb,xL2)
           #  ax.plot(t_nwb,xR2)
           #  plt.xlabel('Time')
           #  #plt.ylabel('')
           #  ax.set_ylim(-.3, 1.5)   # or whatever fixed range you want
           #  plt.title(f'{pat}right and left eye after gap removal')
           #  ax.legend()
           #  fig.tight_layout()
            
            # fig, ax = plt.subplots(figsize=(12, 6))
            # #ax.plot(t_nwb,xL)
            # #ax.plot(t_nwb,xR)
            # ax.plot(t_nwb,x_left_med)
            # ax.plot(t_nwb,x_right_med)
            # ax.set_ylim(-.3, 1.5)   # or whatever fixed range you want
            # plt.xlabel('Time')
            # #plt.ylabel('')
            # plt.title(f'{pat}right and left eye after light smoothing')
            # ax.legend()
            # fig.tight_layout()
        
            # fig, ax = plt.subplots(figsize=(12, 6))
            # #ax.plot(t_nwb,xL)
            # #ax.plot(t_nwb,xR)
            # ax.plot(t_nwb,x_left_hamp)
            # ax.plot(t_nwb,x_right_hamp)
            # ax.set_ylim(-.3, 1.5)   # or whatever fixed range you want
            # plt.xlabel('Time')
            # #plt.ylabel('')
            # plt.title(f'{pat}right and left eye after hampel smoothing')
            # ax.legend()
            # fig.tight_layout()  
        
#           
#%%
            # --- Compute disparity on initial signals ---            
            valid_lr = np.isfinite(x_left_clean) & np.isfinite(x_right_clean)
            
            x_diff = np.full_like(x_left_clean, np.nan, dtype=float)
            x_diff[valid_lr] = x_right_clean[valid_lr] - x_left_clean[valid_lr]

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(t_nwb, x_diff, label='uninterpolated raw displacement value')
            plt.xlabel('Time')
            #plt.ylabel('')
            plt.title(f'{pat}raw horizontal displacement')
            ax.legend()
            fig.tight_layout()
            
            # --- Interpolate only where x_diff is finite ---
            
            valid_x = np.isfinite(x_diff)
            eye_samples_x = np.arange(len(x_diff))
            
            f_x_diff = interp.interp1d(
                eye_samples_x[valid_x],
                x_diff[valid_x],
                kind="linear",
                bounds_error=False,
                fill_value=np.nan
            )
            x_diff_filled = f_x_diff(eye_samples_x)
            
            #x_diff_filled = interp_nans_maxgap(x_diff, max_gap=200)  # e.g., 100 ms at 300 Hz
        
            # Create a boolean array indicating where valid data is present in both eyes for x 
            valid_x = ~np.isnan(x_diff)
             
            # Create an array of sample indices
            eye_samples_x = np.arange(len(x_diff))
            
           #  fig, ax = plt.subplots(figsize=(12, 6))
           #  ax.plot(t_nwb, x_diff_filled, c="orange", label='x gaze distance interpolated')
           #  plt.xlabel('Time')
           #  plt.title(f'{pat} x gaze disparity (adcs) (interpolated)')
           #  ax.legend()
           # # ax.set_title(f'Patient {pat} x gaze disparity (DVA) (interpolated)')
           #  fig.tight_layout()
           #  fig_filename = os.path.join(sub_et_qual_fig_dir, f"{mov.replace('_ieeg.nwb', '')}_x_dva_diff_interp.png")
           #  plt.show()
           # fig.savefig(fig_filename)
            #plt.close(fig)  
        
            #### --- Compute disparity on gap-removed or smoothed signals ---  
            
            if pat == 'NS210':
                valid_filt = np.isfinite(x_left_med) & np.isfinite(x_right_med)
                x_diff2 = np.full_like(x_left_med, np.nan, dtype=float)
                x_diff2[valid_filt] = x_right_med[valid_filt] - x_left_med[valid_filt]
                
            else:
                valid_lr2 = np.isfinite(xL2) & np.isfinite(xR2)    
                x_diff2 = np.full_like(xL2, np.nan, dtype=float)
                x_diff2[valid_lr2] = xR2[valid_lr2] - xL2[valid_lr2]

            #### ON GAP-REMOVED SAMPLES --- Interpolate only where x_diff is finite ---
            valid_x2 = np.isfinite(x_diff2)
            eye_samples_x2 = np.arange(len(x_diff2))   
            
            f_x_diff2 = interp.interp1d(
                eye_samples_x2[valid_x2],
                x_diff2[valid_x2],
                kind="linear",
                bounds_error=False,
                fill_value=np.nan
            )
            x_diff_filled2 = f_x_diff2(eye_samples_x2)
            
            #x_diff_filled2 = interp_nans_maxgap(x_diff2, max_gap=200)

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(t_nwb, x_diff2, c="orange", label='x gaze distance raw')
            plt.xlabel('Time')
            plt.title(f'{pat} x gaze disparity (adcs) raw with gap-removed values')
            ax.legend()
            fig.tight_layout()
            plt.show()
            
            # fig, ax = plt.subplots(figsize=(12, 6))
            # ax.plot(t_nwb, x_diff_filled2, c="orange", label='x gaze distance interpolated')
            # plt.xlabel('Time')
            # plt.title(f'{pat} x gaze disparity (adcs) (interpolated) with gap-removed values')
            # ax.legend()
            # fig.tight_layout()
            # fig_filename = os.path.join(sub_et_qual_fig_dir, f"{mov.replace('_ieeg.nwb', '')}_x_dva_diff_interp.png")
            # plt.show()
                
            
            # Calculate the standard deviation of the entire time series
            std_x_diff2 = np.nanstd(x_diff2)
             
            # Cap the interpolated values to 3 times the standard deviation
            if pat == 'NS210':
                cap_value = 3 * std_x_diff2
            else:
                cap_value = 3 * std_x_diff2
             
            # Identify the interpolated indices (i.e., where the original data was NaN)
            interpolated_indices = np.isnan(x_diff2)
             
            center = np.nanmedian(x_diff2)
            
            # Create a copy of xdiff2 to apply the cap only on interpolated values
            x_diff_filled_capped = x_diff_filled2.copy()
            x_diff_filled_capped[interpolated_indices] = np.clip(
                x_diff_filled2[interpolated_indices],
                center - cap_value,
                center + cap_value)
          
        
            # fig, ax = plt.subplots(figsize=(12, 6))
            # ax.plot(t_nwb, x_diff_filled_capped, c="orange", label='x gaze distance interpolated')
            # plt.xlabel('Time')
            # plt.title(f'{pat} x gaze disparity (adcs) (interpolated) with gap-removed values')
            # ax.legend()
            # fig.tight_layout()
            # fig_filename = os.path.join(sub_et_qual_fig_dir, f"{mov.replace('_ieeg.nwb', '')}_x_dva_diff_interp.png")
            # plt.show()
            # fig.savefig(fig_filename)
            # plt.close(fig) 
    
            # # Fill any remaining NaNs (edges, long gaps) conservatively
            # fill_value = np.nanmedian(x_diff_filled_capped)
            # if np.isnan(fill_value):
            #     fill_value = 0.0
            # x_diff_filled_capped[np.isnan(x_diff_filled_capped)] = fill_value
            
            # --- smooth the final fully-populated series ---
            smooth_sec_final = .5   # 500 ms for visibly smoother output
            win2 = int(max(5, round(smooth_sec_final * fs_eye)))
            if win2 % 2 == 0:
                win2 += 1
            poly2 = 2
            if win2 <= poly2:
                win2 = poly2 + 3  # e.g., 5 when poly2=2
                if win2 % 2 == 0:
                    win2 += 1
            if win2 >= len(x_diff_filled_capped):
                win2 = max(5, (len(x_diff_filled_capped) // 100) * 2 + 1)
            
            x_diff_final = signal.savgol_filter(x_diff_filled_capped, window_length=win2, polyorder=poly2)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(t_nwb, x_diff_final, c="orange", label='x gaze distance smoothed and interpolated')
            plt.xlabel('Time')
            plt.title(f'{pat} x smoothed gaze disparity (adcs) (interpolated) with gap-removed values')
            ax.legend()
            fig.tight_layout()
            fig_filename = os.path.join(sub_et_qual_fig_dir, f"{mov.replace('_ieeg.nwb', '')}_x_diff_interp_capped.png")
            plt.show()
            plt.close(fig) 

            #
             
            # Calculate gaze disparity in terms of pixel distance
            eye_pos_left_interp, _ = interp_bad_data(eye_pos_left[:, 2], val_left, k_small, k_large, k_delay, vis_bads)
            eye_pos_right_interp, _ = interp_bad_data(eye_pos_right[:, 2], val_left, k_small, k_large, k_delay, vis_bads)
            gaze_pos_left_interp, _ = interp_bad_data(gaze_pos_left[:, 2], val_left, k_small, k_large, k_delay, vis_bads)
            gaze_pos_right_interp, _ = interp_bad_data(gaze_pos_right[:, 2], val_left, k_small, k_large, k_delay, vis_bads)
             
            dist_left = eye_pos_left_interp - gaze_pos_left_interp
            dist_right = eye_pos_right_interp - gaze_pos_right_interp
             
             # Distance in cm (converted from median distance in mm to cm for both eyes)
            dist_left_cm = dist_left / 10
            dist_right_cm = dist_right / 10
             
            #avg distance should be the median from both eyes averaged? Maybe this shouldn't be a constant
            avg_dist = (dist_left_cm + dist_right_cm)/2

            # 0–1 gaze to pixels
            xL_pix = xL2 * screen_pix[0]
            xR_pix = xR2 * screen_pix[0]
            
            # Center pixel coordinates so 0 is screen center (important for per-eye angle)
            x_center = (screen_pix[0] - 1) / 2
            xL_off = xL_pix - x_center
            xR_off = xR_pix - x_center
            
            # Instantaneous distance (cm) per eye (you already computed dist_left_cm, dist_right_cm)
            # Optional: clip to avoid insane px2deg spikes if dist briefly goes near 0 or negative
            min_dist_cm = 10.0   # adjust if needed (typical viewing distances often ~40–80 cm)
            max_dist_cm = 200.0
            dL = np.clip(dist_left_cm,  min_dist_cm, max_dist_cm)
            dR = np.clip(dist_right_cm, min_dist_cm, max_dist_cm)
            
            # Instantaneous px2deg per eye (deg per pixel)
            px2deg_L = np.rad2deg(2 * np.arctan(screen_cm[0] / (2 * dL))) / screen_pix[0]
            px2deg_R = np.rad2deg(2 * np.arctan(screen_cm[0] / (2 * dR))) / screen_pix[0]
            
            # Eye-specific gaze angles (deg) and disparity (deg)
            xL_deg = xL_off * px2deg_L
            xR_deg = xR_off * px2deg_R
            gaze_disparity_x = xR_deg - xL_deg
            
              # Plot results
            # plt.figure(figsize=(12, 6))
            # plt.plot(t_nwb, gaze_disparity_x, c="blue")
            # plt.title(f' {pat} Gaze Disparity without interpolation {movie}')
            # plt.xlabel('Time')
            # plt.ylabel('X Disparity')
            # #plt.legend()
            # plt.tight_layout()
            # plt.show()
            # plt.close() 

            #raw = x_diff_filled2.copy()
            raw = gaze_disparity_x.copy()
            x = np.arange(len(raw))
            idx_valid = ~np.isnan(raw)
            
            # --- smoothing parameters ---
            smooth_sec = 0.50        # 50 ms smoothing window (adjust if needed)
            win = int(max(3, round(smooth_sec * fs_eye)))
            if win % 2 == 0:
                win += 1
            poly = 2
            
            # Smooth only the valid samples
            x_valid = raw[idx_valid]
            if len(x_valid) > win:
                x_valid_smooth = signal.savgol_filter(
                    x_valid, window_length=win, polyorder=poly
                )
            else:
                x_valid_smooth = x_valid
                
            print("win (samples):", win, "fs_eye:", fs_eye, "smooth_sec:", win/fs_eye)

            # Interpolate using smoothed valid data (NO extrapolation)
            f_x_disp = interp.interp1d(
                x[idx_valid],
                x_valid_smooth,
                kind='linear',
                bounds_error=False,
                fill_value=np.nan
            )
            x_disp_filled = f_x_disp(x)
            
            # Fill any remaining NaNs (edges, long gaps) conservatively
            fill_value = np.nanmedian(raw)
            if np.isnan(fill_value):
                fill_value = 0.0
            x_disp_filled[np.isnan(x_disp_filled)] = fill_value
     
            #   # Plot results
            # plt.figure(figsize=(12, 6))
            # plt.plot(t_nwb, x_disp_filled, c="blue")
            # plt.title(f' {pat} Gaze Disparity with Interpolation {movie}')
            # plt.xlabel('Time')
            # plt.ylabel('X Disparity')
            # #plt.legend()
            # plt.tight_layout()
            # plt.show()
           # plt.close()  
         
            # Calculate the standard deviation of the entire time series
            std_gaze_disp_x = np.nanstd(gaze_disparity_x)
            
            interp_idx = np.isnan(gaze_disparity_x)
            
            # Define clipping bounds from the raw distribution (conservative)
            mu = np.nanmedian(gaze_disparity_x)          # center
            sd = np.nanstd(gaze_disparity_x)             # spread
            k = 4                                        # 3–5 typical; 4 is conservative
            lower, upper = mu - k * sd, mu + k * sd
            
            x_disp_filled_capped = x_disp_filled.copy()
          #  x_disp_filled_capped[interp_idx] = np.clip(x_disp_filled_capped[interp_idx], lower, upper)

            bad_interp = interp_idx & ((x_disp_filled_capped < lower) | (x_disp_filled_capped > upper))
            x_disp_filled_capped[bad_interp] = mu  # or np.nanmean(gaze_disparity_x)
            
            # --- smooth the final fully-populated series ---
            smooth_sec_final = 2   # 500 ms for visibly smoother output
            win2 = int(max(5, round(smooth_sec_final * fs_eye)))
            if win2 % 2 == 0:
                win2 += 1
            poly2 = 2
            if win2 <= poly2:
                win2 = poly2 + 3  # e.g., 5 when poly2=2
                if win2 % 2 == 0:
                    win2 += 1
            if win2 >= len(x_disp_filled_capped):
                win2 = max(5, (len(x_disp_filled_capped) // 100) * 2 + 1)
            
            x_disp_final = signal.savgol_filter(x_disp_filled_capped, window_length=win2, polyorder=poly2)
            
            # visualize the capped values
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(t_nwb, x_disp_final, label='Final (Smoothed) Gaze Disparity', alpha=0.8)
            ax.set_xlabel('Time')
            ax.set_ylabel('Gaze Disparity')
            ax.set_title(f'{pat} Gaze Disparity with Interpolation and Capping of Interpolated Values {movie}')
            ax.legend()
            ax.grid(True)
            fig_filename = os.path.join(sub_et_qual_fig_dir, f"{mov.replace('_ieeg.nwb', '')}_gaze_disp_interp_capped.png")
            fig.tight_layout()
            fig.savefig(fig_filename)
            plt.show()
           # plt.close(fig)
            
            if calc_vis_displacement:
                #Calculate visual focus displacement per Huang et al. (2019)    
                 
                beta = 0.283
                PD_cm = IPD_cm  # Use the approximated IPD from before
                 
                # Calculate E (gaze disparity in the world coordinate)
                G = np.linalg.norm(gaze_disparity_x)  # Euclidean norm of the gaze disparity vector
                E = beta * G
                 
                # Convert D from cm to mm for consistency with formula
                D_mm = ((dist_left_cm + dist_right_cm) / 2) * 10  # Average distance from eyes to screen converted to mm
                     
                # Initialize visual focus displacement array
                visual_focus_displacement = np.zeros(len(gaze_disparity_x))
                      
                # Use the actual distance in the visual focus displacement calculation
                for i in range(len(gaze_disparity_x)):
                    G = np.linalg.norm(gaze_disparity_x[i])  # G is euclidean norm (2) of gaze disparity
                    E = beta * G  # Calculate E using the current G
                 
                     # Apply conditions for convergence and divergence
                    if gaze_disparity_x[i] > 0:  # Divergence
                        visual_focus_displacement[i] = E * (dist_left_cm[i] + dist_right_cm[i]) * 5 / (PD_cm * 10 - E)
                    else:  # Convergence
                        visual_focus_displacement[i] = -E * (dist_left_cm[i] + dist_right_cm[i]) * 5 / (PD_cm * 10 + E)
                 
               # Generate figure of vis focus disparity before interpolation to save
                # fig, ax = plt.subplots(figsize=(12, 6))
                # ax.plot(t_nwb, visual_focus_displacement, c = 'pink',label='Visual Focus Displacement')
                # plt.xlabel('Time')
                # plt.ylabel('Displacement (mm)')
                # plt.title('Visual Focus Displacement Over Time')
                # ax.legend()
                # ax.set_title(f'Patient {pat} Vis Focus Disparity')
                # fig.tight_layout()
               #fig_filename = os.path.join(sub_et_qual_fig_dir, f"{mov.replace('_ieeg.nwb', '')}_vf_disp_plot.png")
               #fig.savefig(fig_filename)
               # plt.close(fig) 
                 
                if pat == 'NS140':
                    if num_imp == 2:
                        if movie == 'inscapes':
                            visual_focus_displacement[0:222] = np.nanmedian(visual_focus_displacement)
                
                if pat == 'NS191':
                    if movie == 'despicable_me_english':
                        visual_focus_displacement[0:190] = np.nanmedian(visual_focus_displacement)
               
                if pat == 'NS201':
                    if movie == 'despicable_me_english':
                        visual_focus_displacement[0:256] = np.nanmedian(visual_focus_displacement)
                
                 # Assuming visual_focus_displacement and interp_kind are defined
                valid_vis_fd = ~np.isnan(visual_focus_displacement)
                #if len(np.nonzero(valid_vis_fd)[0]) > 1:
                eye_samples_fd = np.arange(len(visual_focus_displacement))
                 # Interpolate missing values
                f_vis_fd = interp.interp1d(eye_samples_fd[valid_vis_fd], visual_focus_displacement[valid_vis_fd], kind='linear', fill_value='extrapolate')
                vis_fd_filled = f_vis_fd(eye_samples_fd)
                 
                # Calculate the standard deviation of the entire time series
                std_vis_fd = np.nanstd(visual_focus_displacement)
                 
                # Cap the interpolated values to 3 times the standard deviation
                center = np.nanmedian(visual_focus_displacement)
                cap_value = 3 * np.nanstd(visual_focus_displacement)
                #cap_value = 3 * std_vis_fd
                 
                # Identify the interpolated indices (i.e., where the original data was NaN)
                interpolated_indices = np.isnan(visual_focus_displacement)
                 
                # Create a copy of vis_fd_filled to apply the cap only on interpolated values
                vis_fd_filled_capped = vis_fd_filled.copy()
                vis_fd_filled_capped[interpolated_indices] = np.clip(
                    vis_fd_filled[interpolated_indices],
                    center - cap_value,
                    center + cap_value)
            
                # fig, ax = plt.subplots(figsize=(12, 6))
                # ax.plot(t_nwb, vis_fd_filled_capped, c = 'purple',label='Visual Focus Displacement')
                # plt.xlabel('Time')
                # plt.ylabel('Displacement (mm)')
                # plt.title('Visual Focus Displacement Over Time')
                # ax.legend()
                # ax.set_title(f'Patient {pat} Vis Focus Disparity')
                # fig.tight_layout()
            
                
                # --- smooth the final fully-populated series ---
                smooth_sec_final = 2   # 500 ms for visibly smoother output
                win2 = int(max(5, round(smooth_sec_final * fs_eye)))
                if win2 % 2 == 0:
                    win2 += 1
                poly2 = 2
                if win2 <= poly2:
                    win2 = poly2 + 3  # e.g., 5 when poly2=2
                    if win2 % 2 == 0:
                        win2 += 1
                if win2 >= len(vis_fd_filled_capped):
                    win2 = max(5, (len(vis_fd_filled_capped) // 100) * 2 + 1)
                
                vis_fd_final = signal.savgol_filter(vis_fd_filled_capped, window_length=win2, polyorder=poly2)
    
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(t_nwb, vis_fd_final, label='Visual Focus Displacement (Interpolated and Smoothed)')
                plt.xlabel('Time')
                plt.ylabel('Displacement (mm)')
                #plt.title('Visual Focus Displacement Over Time (Interpolated)')
                ax.legend()
                ax.set_title(f'Patient {pat} Vis Focus Disparity (interpolated)')
                fig.tight_layout()
                fig_filename = os.path.join(sub_et_qual_fig_dir, f"{mov.replace('_ieeg.nwb', '')}{movie}_vf_disp_interp_smooth.png")
                fig.savefig(fig_filename)
                #plt.close(fig) 

            if calc_vis_displacement:
                df = pd.DataFrame({
                'time': t_nwb,
                'gaze_dist_x_raw': x_diff,
                'gaze_dist_x_interp': x_diff_filled2,
                'dva_gaze_disp_x': gaze_disparity_x,
                'dva_gaze_disp_x_interp': x_disp_final,
                'vis_focus_disp': visual_focus_displacement,
                'vis_fd_interp': vis_fd_final,
                })
            else: 
                df = pd.DataFrame({
                'time': t_nwb,
                'gaze_dist_x_raw': x_diff,
                'gaze_dist_x_interp': x_diff_filled,
                'dva_gaze_disp_x': gaze_disparity_x,
               # 'dva_gaze_disp_x_interp': x_disp_filled_capped
                'dva_gaze_disp_x_interp': x_disp_final
                })
             
             # Save to csv
            # For despicable_me_english, add run-1 to match compute_eye_measures.py expectations
            base, _ = os.path.splitext(mov)
            et_prep_filename = os.path.join(sub_et_prep_dir, base + '_et_prep.csv')
             
            csv_filename = et_prep_filename 
            df.to_csv(csv_filename, index=False)  
         
            io.close()        
#%%
# Run the script if executed directly
if __name__ == "__main__":
    print("="*60)
    print("VERGENCE CALCULATION SCRIPT")
    print("="*60)
    print(f"Input directory: {data_dir}")
    print(f"Movie: {movie}")
    print(f"Patients: {patients}")
    print("="*60)
