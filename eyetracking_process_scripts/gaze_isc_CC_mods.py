#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 20:40:27 2024

@author: max

To do:   
    ISC over time
    Stats for time resolved ISC
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

#%%
data_dir = '/Volumes/Samsung/Movie_data/movies_prep_standard'
fs_dir = '/Volumes/Samsung/Movie_data/anatomy'
eloc_dir = '/Volumes/Samsung/Movie_data/data/electrode_localization'
#prep_dir = '/Volumes/Expansion/Movie_data/movies_prep_standard'

vid = 'despicable_me_english'

# Create results directory 
if not os.path.exists(f'/Volumes/Samsung/Movie_data/ISC_{vid}_updated_new'):
    os.makedirs(f'/Volumes/Samsung/Movie_data/ISC_{vid}_updated_new')

search_string = f'{vid}_run-1_et_prep.npz'


patients = os.listdir(data_dir)
patients.sort()

if vid == 'inscapes':
    patients = ['NS127_02','NS135','NS136','NS137','NS138','NS140','NS140_02','NS154','NS164','NS166','NS178', "NS205", 'NS210','NS211']
elif vid == 'despicable_me_english':
    patients = ['NS190']#['NS127','NS135','NS136','NS137','NS138','NS140','NS153','NS154','NS164','NS166','NS174','NS190','NS191','NS193','NS192','NS194','NS178','NS201','NS204','NS205']

patients.sort()

# Eyetracking sampling rate
fs_eye = 300

# Downsampling factor
dsf = 340

# Time resolved ISC
window = 1000 #1000
overlap = 750 #750

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
    #%%
import os

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

matching_directories = find_patient_directories_with_file(data_dir, search_string)

print(matching_directories)

#%% Collect all data
xy = [None] * len(patients)
gaze_var = [None] * len(patients)

xy_perm = [None] * len(patients)
gaze_var_perm = [None] * len(patients)


for i, pat in enumerate(patients):
    
    pat_dir = '{:s}/{:s}'.format(data_dir, pat)
    
    et_files = os.listdir('{:s}/Eye_prep/'.format(pat_dir))
   # et_file = [file for file in et_files if f'{search_string}' in file][0]
    pattern = f"_task-{vid}_run-"
    
    et_candidates = [
        f for f in et_files
        if pattern in f
        and f.endswith("_et_prep.npz")      # or "_et_prep_updated.npz"
        and not os.path.basename(f).startswith("._")  # ignore macOS junk
    ]
    
    if not et_candidates:
        raise FileNotFoundError(f"No ET prep file found for vid={vid!r} in {et_files}")
    
    et_file = et_candidates[0]
    print(et_file)

    #if not np.in1d(pat, aic_patients)[0]:
    #    continue
    
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

#%% Gaze variation
corr = np.corrcoef(gaze_var.T)
corr[np.eye(corr.shape[0]) == 1] = np.nan
isc_gaze_var = np.nanmean(corr, axis=0)

isc_gaze_perm = np.empty((n_perm, len(isc)))

for p in range(n_perm):
    corr = np.corrcoef(gaze_var.T)
    corr[np.eye(corr.shape[0]) == 1] = np.nan
    isc_gaze_var = np.nanmean(corr, axis=0)
    
#%% Slow fluctutations 
sos = signal.butter(5, [0.05, 0.15], btype='bandpass', fs=fs_eye, output='sos')
xy_slow = signal.sosfiltfilt(sos, xy, axis=0)

isc_slow = compute_isc(xy_slow)

#%% Time resolved ISC 
#%%

#%% Gaze position
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

#%% Gaze variation
isc_time_var, _ = time_resolved_isc(gaze_var, (fs_eye/dsf), window, overlap)

isc_time_var_perm = np.empty(np.hstack((isc_time_x.shape, n_perm)))

print('Computing surrogate distribution of time resolved ISC (Gaze variation)')

for ip in tqdm(range(n_perm)):
    isc_time_var_perm[:,:,ip], _ = time_resolved_isc(gaze_var_perm[:,ip,:], (fs_eye/dsf), window, overlap)
    
#%% Plots
#%%

#%% Gaze position ISC per patients
plt.rcParams.update({'font.size': 14})

plot_isc_pat(isc, 'Gaze position ISC')

#plt.savefig('../results/figures/gaze_position_isc_patients.png', dpi=300)

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

#plt.savefig('../results/figures/gaze_position_isc_patients_perm_stats.png', dpi=300)

#%% ISC of slow fluctuations
plot_isc_pat(isc_slow, 'ISC of slow fluctuations')
    
#plt.savefig('../results/figures/gaze_position_isc_patients_slow.png', dpi=300)

#%% ISC of gaze variation
plot_isc_pat(isc_gaze_var, 'Gaze Variation ISC')

plt.savefig('../results/figures/gaze_variation_isc_patients.png', dpi=300)

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

#plt.savefig('../results/figures/gaze_position_isc_time_patients.png', dpi=300)

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

#plt.savefig('../results/figures/gaze_variation_isc_time_patients.png', dpi=300)

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
plt.title('Mean ISC (Gaze variation) with surrogate distribution')

plt.grid()
plt.tight_layout()

#plt.savefig('../results/figures/gaze_position_isc_time_mean_permutations.png', dpi=300)

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
#plt.savefig('../results/figures/gaze_variation_isc_time_mean_permutations.png', dpi=300)

#%% Save data

# ISC for whole recording
np.savez(f'/Volumes/Samsung/Movie_data/ISC_{vid}_updated_new/{vid}_isc_updated.npz', isc_gaze_pos=isc, 
         isc_gaze_pos_slow=isc_slow,
         isc_gaze_var=isc_gaze_var,
         isc_gaze_pos_perm=isc_perm,
         patients=patients,
         n_perm=n_perm,
         fs=fs_eye/dsf)

# Gaze position
np.savez(f'/Volumes/Samsung/Movie_data/ISC_{vid}_updated_new/{vid}_isc_gaze_position_time_updated.npz', isc_time_gaze=isc_time_gaze, 
         isc_time_gaze_perm=isc_time_gaze_perm,
         time_isc=time_isc,
         patients=patients,
         window=window,
         overlap=overlap,
         n_perm=n_perm,
         fs=fs_eye/dsf)

# Gaze variation
np.savez(f'/Volumes/Samsung/Movie_data/ISC_{vid}_updated_new/{vid}_isc_gaze_variation_time_updated.npz', isc_time_var=isc_time_var, 
         isc_time_var_perm=isc_time_var_perm,
         time_isc=time_isc,
         patients=patients,
         window=window,
         overlap=overlap,
         n_perm=n_perm,
         fs=fs_eye/dsf)
