#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 14:26:51 2023

@author: max

"""



#%% To-Do
# Need to label spikes and other artifacts in data

import os, sys, re
import numpy as np
import mne
from itertools import compress
from pynwb import NWBHDF5IO
#from helpers import interp_bad_samples, combine_left_right, detect_saccades_remodnav
from tqdm import tqdm
from scipy import ndimage
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import scipy.interpolate as interp
import scipy.stats as stats
import scipy.signal as signal
import importlib.util


# Define path to your script
module_path = '/Users/christinechesebrough/Documents/data_standardization-main/src/temporal_response_function.py'

# Define module name (can be arbitrary)
module_name = 'temporal_response_function'

# Load spec and module
spec = importlib.util.spec_from_file_location(module_name, module_path)
trf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trf)


# Define the path to helpers.py
helper_path = '/Volumes/Samsung/scripts/eyetracking_process/helpers.py'

# Load the module
spec = importlib.util.spec_from_file_location("helpers", helper_path)
helpers = importlib.util.module_from_spec(spec)
sys.modules["helpers"] = helpers
spec.loader.exec_module(helpers)

# Now you can use its functions:
from helpers import interp_bad_samples, combine_left_right, detect_saccades_remodnav, plot_warning_indices_and_saccades_on_gaze, warnings_near_saccades


#sys.path.append('/Users/christinechesebrough/Documents/EPIPE-master/Python/')


sys.path.insert(0, '/Users/christinechesebrough/Documents/EPIPE-movie_nwb/Python')
sys.path.append('/Users/christinechesebrough/Documents/iEEG2NWB-main')

# 4. Now safe to import from other packages

from epipe import inspectNwb, nwb2mne, read_ielvis, reref_avg, reref_bipolar, filter_hfa_continuous
# After plotting raw channels and picking new ones to reject the kernel freezes
# One solution is disabling "Active support" for Matplotlib 
# (under tools -> preferences -> IPython console -> Graphics)
# And then manually setting the matplotlib backend, as described here:
# https://github.com/mne-tools/mne-python/issues/6528#issuecomment-892066104
matplotlib.use('Qt5Agg')


#%% Parameters
full_task_name = 'movies'
pipeline_name = 'preprocess_movies'
pipeline_version = 'v.0.0.0'

resample_fs = 600
notch_freqs = (60, 120, 180)

# Types of references to use in analyses
# Must be a list containing at least one of the options: "avg", "bip"
ref_types = ['avg', 'bip']

n_jobs = 16

convert_db = True

# Frequency range
freq_range = [70, 170]
n_freq_bins = 10
freq_space = 'log'      # 'log', 'lin'

resample_bha_fs = 100

# Eye tracking parameters
t_diff = 0.1

dist = 60

# Screen dimentions [cm]
d = 23.8*2.54
ar = 16/9

k_small = 15
k_large = 90
k_delay = 30
vis_bads = False

savgol_length = .029

screen_pix = np.array([1920, 1080])
screen_cm = np.array([50.92, 28.64])

# Pupil 
t_buffer_pre = 0.1
t_buffer_post = 0.2
t_smooth = 0.2
t_diff = 0.1
sigma_smooth = 0.5

frame_rate = 30

# Utility to save figures
def save_fig(fig, filename, sub_dir):
    save_path = os.path.join(sub_dir, filename)
    fig.savefig(save_path)
    plt.close(fig)


save_plots = True

#%% Define directories
#data_dir = '/Volumes/Samsung/Movie_data/movies_nwb_standard'
#data_dir = '/Volumes/Samsung/Movie_data/movies_Jun25_reprocess'
data_dir = '/Volumes/Samsung/Movie_data/movies_new_nwb'
fs_dir = '/Volumes/Samsung/anatomy'
prep_dir = '/Volumes/Samsung/Movie_data/movies_new_prep'
frame_dir = '/Volumes/Samsung/Movie_data/data/video_frames'
lum_dir = '/Volumes/Samsung/Movie_data/data/luminance'

et_prep_dir = 'Eye_prep'
audio_dir = 'Audio'
neural_prep_dir = 'Neural_prep'
hfa_dir = 'HFA'

patients = os.listdir(data_dir)
patients = ['NS210']
patients.sort()

vid = 'inscapes'

#%%
movie_table = pd.read_excel('/Volumes/Samsung/Movie_data/data/electrode_localization/movie_table.xlsx')

if not os.path.exists(lum_dir):
    os.makedirs(lum_dir)
    
#%%
broken_nwb = []

for pat in patients:
        
    implants = os.listdir('{:s}/{:s}'.format(data_dir, pat))
       
    for imp in implants:
        
        if data_dir == '/Volumes/Samsung/Movie_data/movies_nwb_standard':
            ieeg_dir = '{:s}/{:s}/{:s}'.format(data_dir, pat, imp)
        elif data_dir == '/Volumes/Samsung/Movie_data/movies_new_nwb':
            ieeg_dir = '{:s}/{:s}/{:s}/ieeg/'.format(data_dir, pat, imp)

        movies = [f for f in os.listdir(ieeg_dir) if f.endswith('.nwb')]
        
        num_imp = int(re.findall(r'\d+', imp)[0])
        
        if num_imp == 1:
            pat_fs = pat.replace('sub-', '')
        elif num_imp >= 2:
            pat_fs = '{:s}_{:02d}'.format(pat.replace('sub-', ''), num_imp)

        sub_fs_dir = '{:s}/{:s}'.format(fs_dir, pat_fs)
        
        sub_et_prep_dir = '{:s}/{:s}/{:s}'.format(prep_dir, pat_fs, et_prep_dir)
        
        # Assume `vid` is defined above, e.g. vid = 'despicable_me_english'
        filtered_movies = [mov for mov in movies if vid in mov]

        for mov in filtered_movies:
            # Extract the movie file name (strip path if needed)
            movie_name = mov.split('/')[-1]  # adjust if mov is already just the file name

            # Construct the NWB file path
            if data_dir == '/Volumes/Samsung/Movie_data/movies_nwb_standard':
                nwb_fname = '{:s}/{:s}/{:s}/{:s}'.format(data_dir, pat, imp, movie_name)
            elif data_dir == '/Volumes/Samsung/Movie_data/movies_new_nwb':
                nwb_fname = '{:s}/{:s}/{:s}/ieeg/{:s}'.format(data_dir, pat, imp, movie_name)

            # Now you can proceed to read or process nwb_fname
            print(f"Processing: {nwb_fname}")

            #%%
            # NWB read
           # try:
            io = NWBHDF5IO(nwb_fname, mode='r', load_namespaces=True)
            nwb = io.read()
           # except:
            #   broken_nwb.append(nwb_fname)
             #  continue
            
            # Get info on data in NWB file
            nwbInfo = inspectNwb(nwb)
            tsInfo = nwbInfo['timeseries']
            elecTable = nwbInfo['elecs']
            #%%
            # Get ieeg data
           # try: 
            
            if 'ieeg' in tsInfo['name'].to_list():
                ecogContainer = nwb.acquisition.get('ieeg')
                fs = ecogContainer.rate
                ecog = nwb2mne(ecogContainer,preload=False)
            
                # Get coordinates of each electrode that has
                try:
                    ielvis_df = read_ielvis(sub_fs_dir)
                    ch_coords = {}
                    nan_array = np.empty((3,)) * np.nan
                    for thisChn in ecog.ch_names:
                        idx = np.where(ielvis_df['label'] == thisChn)[0]
                        if len(idx) == 1:
                            xyz = np.array(ielvis_df.iloc[idx[0]]['LEPTO'])
                            ch_coords[thisChn] = xyz/1000
                        elif len(idx) == 0:
                            ch_coords[thisChn] = nan_array
                        else:
                            raise ValueError('More than 1 found!')
                except:
                    ch_coords = {}
                    for thisChn in ecog.ch_names:
                        ch_coords[thisChn] = np.empty((3,)) * np.nan
            
                # Create `montage` data structure as required by MNE
                montage = mne.channels.make_dig_montage(ch_pos=ch_coords, coord_frame='mri')
                montage.add_estimated_fiducials(pat_fs, fs_dir)
                ecog.set_montage(montage)
                
                # Load audio
                audioContainer = nwb.acquisition.get('audio')
                fs_audio = audioContainer.rate
                audio = audioContainer.data[:]
                t_audio = np.arange(0, audio.shape[0]) / fs_audio
                
                from scipy.io import wavfile
                import numpy as np
                
                # Assuming fs_audio and audio are already defined as above
                
                # Normalize audio to avoid clipping if necessary
                audio_norm = audio / np.max(np.abs(audio))
                
                # Convert to 16-bit PCM format if needed
                audio_int16 = np.int16(audio_norm * 32767)
                
                # Write to WAV file
                wavfile.write('exported_audio.wav', int(fs_audio), audio_int16)
                
                print("Audio exported to 'exported_audio.wav'")

           # except:
                
          #      broken_nwb.append(nwb_fname)
          #      continue
            
            # Get the current sampling rate. Important for later
            orig_fs = ecog.info['sfreq']
            
            # Get the TTL pulses. Specify the name of the container with the TTL pulses
            ttl_container_name = 'TTL'
            try:
                ttls = nwb.get_acquisition(ttl_container_name).timestamps[()]
            except:
                # An analog TTL channel, convert to discrete timestamps
                ana_ttls = nwb.get_acquisition(ttl_container_name).data[()].flatten()
                ttl_rate = nwb.get_acquisition(ttl_container_name).rate
                from epipe import ana2dig
                _, ttls = ana2dig(ana_ttls, fs=ttl_rate, min_diff=0.4, return_time=True)
            
            ttl_id = nwb.acquisition['TTL'].data[:]
            
            #%% Cut and save the audio data        
            if 'Fix' in mov:
            # not sure if this should work in most cases with new data??  (CC note Jun 26 2025)...
                idx_movie = np.logical_and(t_audio >= ttls[0], t_audio < ttls[1])
                
            else:
                
                # Frame time
                frame_time = nwb.trials['start_time'].data[:]
                frame_time_end = nwb.trials['stop_time'].data[:]
                
                idx_movie = np.logical_and(t_audio >= frame_time[0], t_audio < frame_time[-1])
                
            audio = audio[idx_movie, :]
            t_audio = t_audio[idx_movie]
            
            sub_audio_dir = '{:s}/{:s}/{:s}'.format(prep_dir, pat_fs, audio_dir)
            audio_filename = '{:s}/{:s}'.format(sub_audio_dir, mov.replace('_ieeg.nwb', '_audio.npz'))
            
            if not os.path.exists(sub_audio_dir):
                os.makedirs(sub_audio_dir)
            
            np.savez(audio_filename, audio=audio, t_audio=t_audio, fs_audio=fs_audio)   
            
            #%% Pupil data
            sub_et_prep_dir = '{:s}/{:s}/{:s}'.format(prep_dir, pat_fs, et_prep_dir)
            
            if not os.path.exists(sub_et_prep_dir):
                os.makedirs(sub_et_prep_dir)
                
            #region Notch filter, down sample, add/remove bad channels by inspecting raw trace, save
            et_prep_filename = '{:s}/{:s}'.format(sub_et_prep_dir, mov.replace('_ieeg.nwb', '_et_prep.npz'))
            
            if len(nwb.processing) != 0: #not os.path.exists(et_prep_filename) and
                    
                t_pupil = nwb.processing['eye_tracking']['pupils']['l_eye'].timestamps[:]
                l_pupil = nwb.processing['eye_tracking']['pupils']['l_eye'].data[:]
                r_pupil = nwb.processing['eye_tracking']['pupils']['r_eye'].data[:]
                
                # Validity
                l_pupil_val = nwb.processing['eye_tracking']['pupils']['l_eye'].control[:]
                r_pupil_val = nwb.processing['eye_tracking']['pupils']['r_eye'].control[:]
                
                pupil = np.stack((l_pupil, r_pupil))
                fs_pupil = 1 / np.mean(np.diff(t_pupil))
                
                #%% Eye data
                l_gaze = nwb.processing['eye_tracking']['eyes']['l_eye_adcs'].data[:]
                r_gaze = nwb.processing['eye_tracking']['eyes']['r_eye_adcs'].data[:]
                
                t_gaze = nwb.processing['eye_tracking']['eyes']['l_eye_adcs'].timestamps[:]
                
                fs_gaze = 1 / np.mean(np.diff(t_gaze))
                
                gaze_val = np.vstack((l_pupil_val <= 1, r_pupil_val <= 1)).T
                
                #%% Detect saccades
                
                #old versions
                #saccade_onset_vec, fixation_vec, saccade_pos, xy, t_eye, t_nwb, fs_eye = detect_saccades_remodnav(nwb_fname,k_small,k_large, k_delay, vis_bads, t_diff, screen_cm,screen_pix, savgol_length)
                #saccade_onset_vec, fixation_vec, saccade_pos, xy, t_eye, t_nwb, fs_eye, num_implausible, max_velocity= detect_saccades_remodnav(nwb_fname,k_small,k_large, k_delay, vis_bads, t_diff, screen_cm,screen_pix, savgol_length)
                #saccade_onset_t = t_nwb[saccade_onset_vec == 1]
                #fixation_t = t_nwb[fixation_vec == 1]
                
                #version with warning_indices
                saccade_onset_vec, fixation_vec, saccade_pos, xy, t_eye, t_nwb, fs_eye, num_implausible, max_velocity, warning_indices = detect_saccades_remodnav(nwb_fname, k_small, k_large, k_delay, vis_bads, t_diff, screen_cm, screen_pix,savgol_length)
                
                saccade_onset_t = t_nwb[saccade_onset_vec == 1]
                fixation_t = t_nwb[fixation_vec == 1]
               
                
                # Recalculate velocities for plotting (or return them from function if preferred)
                vel_x = np.gradient(xy[:,0], t_nwb)
                vel_y = np.gradient(xy[:,1], t_nwb)
                vel_total = np.sqrt(vel_x**2 + vel_y**2)
                
                print(f"number of warning segments: {len(warning_indices)}")
                
                #plot_warning_segments(xy, t_nwb, vel_total, warning_indices, window=50)
                title = f"Warning indices on gaze X ({pat}, {mov})"
                plot_warning_indices_and_saccades_on_gaze(t_nwb, xy, warning_indices, saccade_onset_vec,title=title)
                
                summary = warnings_near_saccades(warning_indices, saccade_onset_vec, fs_eye, window_ms=10)
                print(summary)
                
                #%%
                rest = False
                if rest:
                    # Calculate gaze velocity (pixels/sec)
                    
                    vel_x = np.gradient(xy[:,0], t_nwb)
                    vel_y = np.gradient(xy[:,1], t_nwb)
                    
                    # Compute overall velocity magnitude
                    vel_total = np.sqrt(vel_x**2 + vel_y**2)
                    
                    fig, ax = plt.subplots(figsize=(6, 4))
    
                    ax.plot(t_nwb, vel_x, label='X velocity')
                    ax.plot(t_nwb, vel_y, label='Y velocity')
                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel('Velocity (pix/s)')
                    ax.set_title(f'Gaze velocity over time ({pat}_{mov})')
                    ax.legend()
                    #save_fig(fig, f'{pat}_{mov}_velocity_time.png', sub_et_prep_dir)
    
                    fig, ax = plt.subplots(figsize=(6, 4))
    
                    ax.plot(t_nwb, xy[:,0], label='Gaze X')
                    ax.scatter(saccade_onset_t, saccade_pos[:,0], color='red', s=10, label='Saccades')
                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel('X position (pix)')
                    ax.set_title(f'Gaze X position with saccades ({pat}_{mov})')
                    ax.legend()
                   # save_fig(fig, f'{pat}_{mov}_gazeX_saccades.png', sub_et_prep_dir)
                    
                    fig, ax = plt.subplots(figsize=(6, 4))
    
                    ax.plot(t_nwb, xy[:,1], label='Gaze Y')
                    ax.scatter(saccade_onset_t, saccade_pos[:,1], color='black', s=10, label='Saccades')
                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel('Y position (pix)')
                    ax.set_title(f'Gaze Y position with saccades ({pat}_{mov})')
                    ax.legend()
                    #save_fig(fig, f'{pat}_{mov}_gazeY_saccades.png', sub_et_prep_dir)
                    
    
    
                    #%% Preprocess pupil data                
                    # Interpolate bad samples and average left and right
                    l_pupil_bad = l_pupil_val > 1
                    r_pupil_bad = r_pupil_val > 1
                    
                    l_pupil, l_pupil_bad = interp_bad_samples(l_pupil, l_pupil_bad, t_pupil, 
                                                              t_buffer_pre, t_buffer_post, fs_pupil, 
                                                              interp_kind='linear')
                    
                    r_pupil, r_pupil_bad = interp_bad_samples(r_pupil, r_pupil_bad, t_pupil, 
                                                              t_buffer_pre, t_buffer_post, fs_pupil, 
                                                              interp_kind='linear')
                    
                    pupil, pupil_bad = combine_left_right(l_pupil, 
                                                          r_pupil, 
                                                          l_pupil_bad, 
                                                          r_pupil_bad, 
                                                          t_diff, 
                                                          fs_pupil)
                    
                    # Smooth
                    s_sigma_smooth = int(np.round(fs_pupil * sigma_smooth))
                    
                    pupil_smooth = ndimage.gaussian_filter1d(pupil, s_sigma_smooth)
                                                          
                    # Pupil response window
                    resp_win = [-5, 5]
                    
                    if not 'Fix' in mov:
                        
                        # Compute luminance 
                        mov_file_parts = mov.split('_')
                        idx_task = ['task' in part for part in mov_file_parts]
                        task_str = list(compress(mov_file_parts, idx_task))
                        task = task_str[0].replace('task-', '')
                        
                        if task == 'despicable':
                            task = 'despicable_me_english'
                        vid_name = movie_table[movie_table.task == task].vid_file.values[0]
                        
                        lum_file = '{:s}/{:s}.npy'.format(lum_dir, vid_name)
                        
                        
                        if not os.path.exists(lum_file):
                        
                            vid_frame_dir = '{:s}/{:s}'.format(frame_dir, vid_name)
                            frame_files = os.listdir(vid_frame_dir)
                            frame_files.sort()
                            
                            luminance = np.empty(len(frame_files))
                            
                            # Initiate a progress bar to track what's going on and how long it will take
                            pbar = tqdm(total=len(frame_files), ncols=100, desc='Comupting luminance')
                            
                            for fr,file in enumerate(frame_files):
                                img = Image.open('{:s}/{:s}/{:s}'.format(frame_dir, vid_name, file))
                                img = ImageOps.grayscale(img) 
                                img = np.asarray(img)
                                luminance[fr] = np.sum(img[:])
                                pbar.update(1)
                                
                            pbar.close()
                            
                            np.save(lum_file, luminance)
                            
                        else:  
                            luminance = np.load(lum_file)
    
                    # Cut pupil data
                    if 'Fix' in mov:
                        idx_mov = np.logical_and(t_pupil >= ttls[0], t_pupil < ttls[1])
                    else:
                        idx_mov = np.logical_and(t_pupil >= frame_time[0], t_pupil < frame_time_end[-1])
                    
                    t_pupil = t_pupil[idx_mov]
                    pupil_smooth = pupil_smooth[idx_mov]
                    pupil_bad = pupil_bad[idx_mov]
                    
                    if not 'Fix' in mov:
                            
                        # Resample pupil data
                        f = interp.interp1d(t_pupil, pupil_smooth, fill_value='extrapolate')
                        pupil_rs = f(frame_time)
                        
                        f = interp.interp1d(t_pupil, pupil_bad, fill_value='extrapolate')
                        pupil_bad = f(frame_time)
                        
                        if 'Monkey' in mov:
                            
                            pupil_rs = signal.resample(pupil_rs, round(len(pupil_rs)/2))
                            t_rs = np.arange(0,len(pupil_rs)) * (np.mean(np.diff(frame_time))*2) + frame_time[0]
                            
                            f = interp.interp1d(frame_time, pupil_bad, fill_value='extrapolate')
                            pupil_bad = f(t_rs)
                            
                            frame_time = t_rs
                            luminance = luminance[:len(pupil_rs)]
                            
                        # Log
                        log_lum = np.log(luminance+1)
                        
                        # z-score
                        pupil_rs = stats.zscore(pupil_rs)
                        luminance = stats.zscore(luminance)
                        log_lum = stats.zscore(log_lum)
                        
                        pupil_rs = pupil_rs - pupil_rs[0]
                        luminance = luminance - luminance[0]
                        log_lum = log_lum - log_lum[0]
                        
                        # Cut the luminance array if the recording was shorter (shorter pupil diameter vector)
                        if len(pupil_rs) < len(luminance):
                            luminance = luminance[:len(pupil_rs)]
                            log_lum = log_lum[:len(pupil_rs)]
        
                        # luminance[pupil_bad == 1] = np.nan
                        # log_lum[pupil_bad == 1] = np.nan
                        
                        # Model response to luminance
                        stim_lum, t_resp = trf.toeplitz(luminance, resp_win, 1/np.mean(np.diff(frame_time)))
                        stim_lum_log, _ = trf.toeplitz(log_lum, resp_win, 1/np.mean(np.diff(frame_time)))
                        
                        stim = np.concatenate((np.ones((len(stim_lum),1)), stim_lum, stim_lum_log), axis=1)
                        
                        # idx_bad = np.sum(np.isnan(stim), 1) != 0
                        # stim_good = stim[np.invert(idx_bad), :]
                        # pupil_good = pupil_rs[np.invert(idx_bad)]
                        
                        # lum_resp_smooth = trf.trf_train(stim, pupil_smooth_rs, 0)
                        lum_resp = trf.trf_train(stim, pupil_rs, 0.2)
                        
                        # Plot the filters
                        plt.figure()
                        plt.plot(t_resp, lum_resp[1:len(t_resp)+1])
                        plt.plot(t_resp, lum_resp[len(t_resp)+1:])
                        
                        # Remove luminance resp
                        pupil_lum = stim @ lum_resp
                        
                        print(np.corrcoef(pupil_lum, pupil_rs)[0,1])
                        
                        pupil_rs -= pupil_lum
                        
                    else:
                        
                        frame_time = np.arange(t_pupil[0], t_pupil[-1], 1/frame_rate)
                        
                        # Resample pupil data
                        f = interp.interp1d(t_pupil, pupil_smooth, fill_value='extrapolate')
                        pupil_rs = f(frame_time)
                        
                        f = interp.interp1d(t_pupil, pupil_bad, fill_value='extrapolate')
                        pupil_bad = f(frame_time)
                        
                        # z-score
                        pupil_rs = stats.zscore(pupil_rs)
                                         
                    #%% Remove effect of saccades
                    idx_in = np.logical_and(saccade_onset_t > frame_time[0], 
                                            fixation_t < frame_time[-1])
                    
                    saccade_onset_t = saccade_onset_t[idx_in]
                    fixation_t = fixation_t[idx_in]
                    saccade_pos = saccade_pos[idx_in, :]
                    
                    f = interp.interp1d(frame_time, np.arange(0, len(frame_time)))
                    saccade_frame = np.round(f(saccade_onset_t)).astype(int)
                    
                    saccade_vec = np.zeros(frame_time.shape)
                    saccade_vec[saccade_frame] = 1
                    
                    # Model response to luminance
                    stim_sacc, t_resp = trf.toeplitz(saccade_vec, resp_win, 1/np.mean(np.diff(frame_time)))
                    
                    # lum_resp_smooth = trf.trf_train(stim, pupil_smooth_rs, 0)
                    sacc_resp = trf.trf_train(stim_sacc, pupil_rs, 0)
                    
                    # Remove saccade resp
                    pupil_sacc = stim_sacc @ sacc_resp
                    pupil_rs -= pupil_sacc
                    
                    # Save pupil data
                    np.savez(et_prep_filename, saccade_onset_t=saccade_onset_t, 
                             fixation_t=fixation_t, saccade_pos=saccade_pos,
                             xy=xy, pupil=pupil_rs, t_pupil=frame_time, t_gaze=t_nwb,
                             fs_gaze=fs_gaze)    
                
               #%%
                #Write some figures to examine the gaze, saccade, and pupil data to do sanity checks
                # The task is human patients watching a short movie, so the eye measures should not deviate from expected normal eye movements
               
                    vel_x = np.gradient(xy[:,0], t_nwb)
                    vel_y = np.gradient(xy[:,1], t_nwb)
                    vel_total = np.sqrt(vel_x**2 + vel_y**2)
                    
                    # Plot raw gaze time series
                    fig, ax = plt.subplots(figsize=(6, 4))
    
                    ax.plot(t_nwb, xy[:,0], label='Gaze X')
                    ax.plot(t_nwb, xy[:,1], label='Gaze Y')
                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel('Position (pix)')
                    ax.set_title(f'Raw Gaze Position Time Series {pat}')
                    ax.legend()
                    plt.show()
                    if save_plots:
                        save_fig(fig, f'{pat}_{mov}_gaze_position_timeseries.png', sub_et_prep_dir)
                    
                    # Plot gaze trajectory with saccades
                    fig, ax = plt.subplots(figsize=(6, 4))
    
                    ax.plot(xy[:,0], xy[:,1], label='Gaze trajectory', alpha=0.5)
                    ax.scatter(saccade_pos[:,0], saccade_pos[:,1], color='red', s=10, label='Saccade onsets')
                    ax.set_xlabel('X position (pix)')
                    ax.set_ylabel('Y position (pix)')
                    ax.set_title(f'Gaze trajectory with saccade onsets {pat}')
                    ax.legend()
                    ax.invert_yaxis()  # if y=0 is top of screen
                    plt.show()
                    if save_plots:
                        save_fig(fig, f'{pat}_{mov}_gaze_trajectory_saccades.png', sub_et_prep_dir)
                    
                    # Histogram of saccade amplitudes
                    sacc_amplitudes = np.linalg.norm(np.diff(saccade_pos, axis=0), axis=1)
                    fig, ax = plt.subplots(figsize=(6, 4))
    
                    ax.hist(sacc_amplitudes, bins=50)
                    ax.set_xlabel('Saccade amplitude (pix)')
                    ax.set_ylabel('Count')
                    ax.set_title(f'Histogram of saccade amplitudes {pat}')
                    plt.show()
                    if save_plots:
                        save_fig(fig, f'{pat}_{mov}_saccade_amplitude_histogram.png', sub_et_prep_dir)
                    
                    # Pupil diameter time series
                    fig, ax = plt.subplots(figsize=(6, 4))
    
                    ax.plot(frame_time, pupil_rs)
                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel('Pupil diameter (z-score)')
                    ax.set_title(f'Pupil diameter time series (after preprocessing) {pat}')
                    plt.show()
                    if save_plots:
                        save_fig(fig, f'{pat}_{mov}_pupil_timeseries.png', sub_et_prep_dir)
                        
                    
                    # Heatmap of gaze positions
                    valid_idx = np.isfinite(xy[:,0]) & np.isfinite(xy[:,1])
                    xy_valid = xy[valid_idx, :]
                    
                    fig, ax = plt.subplots(figsize=(6, 4))
    
                    h = ax.hist2d(xy_valid[:,0], xy_valid[:,1], bins=[50, 50])
                    ax.set_xlabel('X position (pix)')
                    ax.set_ylabel('Y position (pix)')
                    ax.set_title(f'{pat}_{mov}Heatmap of gaze positions')
                    ax.invert_yaxis()
                    plt.colorbar(h[3], label='Count')
                    plt.show()
                    if save_plots:
                        save_fig(fig, f'{pat}_{mov}_gaze_heatmap.png', sub_et_prep_dir)
                    
                    # Velocity time series (X, Y, total)
                    fig, ax = plt.subplots(figsize=(6, 4))
    
                    ax.plot(t_nwb, vel_x, label='X velocity')
                    ax.plot(t_nwb, vel_y, label='Y velocity')
                    ax.plot(t_nwb, vel_total, label='Total velocity', alpha=0.5)
                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel('Velocity (pix/s)')
                    ax.set_title(f'Gaze velocity over time ({pat})')
                    ax.legend()
                    plt.show()
                    if save_plots:
                        save_fig(fig, f'{pat}_{mov}_gaze_velocity.png', sub_et_prep_dir)
                    
                    
    
               
                    #%%
                     # Close NWB file
                    io.close()
