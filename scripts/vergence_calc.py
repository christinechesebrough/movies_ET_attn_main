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
import mne
from itertools import compress
from pynwb import NWBHDF5IO

from tqdm import tqdm
from scipy import ndimage
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import scipy.interpolate as interp
import copy, math
import scipy.interpolate as interp
import scipy.signal as signal
import scipy.stats as stats
import remodnav

import importlib.util

# Define the path to helpers.py
helper_path = '/Volumes/Samsung/scripts/eyetracking_process/helpers.py'

# Load the module
spec = importlib.util.spec_from_file_location("helpers", helper_path)
helpers = importlib.util.module_from_spec(spec)
sys.modules["helpers"] = helpers
spec.loader.exec_module(helpers)

# Now you can use its functions:
from helpers import interp_bad_samples, combine_left_right, detect_saccades_remodnav


sys.path.append('/Users/christinechesebrough/Documents/EPIPE-master/Python')
from epipe import inspectNwb, nwb2mne, read_ielvis, reref_avg, reref_bipolar, filter_hfa_continuous

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
    
    data_val = copy.copy(data_val)
    
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

def save_blink_events(blink_events, output_dir, subject,movie):
    """
    Save blink events to a CSV file.
    
    Inputs:
        blink_events   - List of blink event dictionaries.
        output_dir     - Directory to save the CSV file.
        subject        - Subject ID for naming the file.
    """
    # Convert blink events to a DataFrame
    df_blinks = pd.DataFrame(blink_events)

    # Construct file path
    output_file = os.path.join(output_dir, f"{subject}__{movie}_blink_events.csv")

    # Save to CSV
    df_blinks.to_csv(output_file, index=False)
    print(f"Blink events saved to {output_file}")
    


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
    
#%%
drive = 'Samsung'
data_dir = '/Volumes/Samsung/Movie_data/movies_new_nwb'
fs_dir = '/Volumes/Samsung/anatomy'
prep_dir = '/Volumes/Samsung/Movie_data/movies_new_prep'

et_prep_dir = 'Eye_prep'
movie = 'despicable_me_english'

#%%  
broken_nwb = [] 

#patients = [patient for patient in os.listdir(data_dir) if not patient.startswith('.DS_Store')]

#if movie == 'despicable_me_english':
    #good_ET_pats = ['NS127','NS135','NS136','NS137','NS138','NS140','NS153','NS154','NS164','NS166','NS174']
#elif movie == 'inscapes':
 #   good_ET_pats =  [] #['NS127','NS135','NS136','NS137','NS138','NS140','NS154','NS164','NS166']

good_ET_pats = ['NS154']

patients = good_ET_pats

patients.sort()


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
        sub_et_prep_dir = f'{data_dir}/{pat}/{et_prep_dir}'

        if not os.path.exists(sub_et_prep_dir):
            os.makedirs(sub_et_prep_dir)

        movies = os.listdir('{:s}/{:s}/{:s}'.format(data_dir, pat, imp))    
                
        # Filter movies to include only files with the string "task-despicable_me_english"
        filtered_movies = [mov for mov in movies if (f"{movie}") in mov]
        
        # If no such movie is found, you may want to handle it (e.g., raise an error or continue)
        if filtered_movies == []:
            print(f"No movie file with {movie} found in the directory.")
            continue
  
        # Process the filtered movies
        for mov in filtered_movies:
            # Construct the file path in a safe manner
            # For despicable_me_english, add run-1 to match compute_eye_measures.py expectations
            if movie == 'despicable_me_english':
                et_prep_filename = '{:s}/{:s}'.format(sub_et_prep_dir, mov.replace('_ieeg.nwb', '_run-1_et_prep.csv'))
            else:
                et_prep_filename = '{:s}/{:s}'.format(sub_et_prep_dir, mov.replace('_ieeg.nwb', '_et_prep.csv'))
             
            nwb_fname = os.path.join(data_dir, pat, imp, mov)  
             
             # NWB read       
            io = NWBHDF5IO(nwb_fname, mode='r', load_namespaces=True)
            nwb = io.read()
            print('Loading nwb data for patient {:s} ...'.format(pat))
             
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
            IPD_cm = np.median(distances) / 10  # If distances are in mm, convert to cm.
             
            # time
            t_nwb = nwb.processing['eye_tracking']['eyes']['l_eye_adcs'].timestamps[:]
             
            # Validity
            val_left = nwb.processing['eye_tracking']['eyes']['l_eye_adcs'].control[:] <= 1
            val_right = nwb.processing['eye_tracking']['eyes']['r_eye_adcs'].control[:] <= 1
            comparison_array = np.where(val_left == val_right, 0, 1)
             
            # Sampling rate
            fs_eye = 1/np.median(np.diff(t_nwb))
             
            # Integration in your script
            # Detect blinks based on validity arrays
            blink_events = detect_blinks(val_left, val_right, t_nwb, fs_eye)
            
            save_blink_events(blink_events, sub_et_prep_dir, pat_fs,movie)

            
            #
             
            # Interpolate over disparity values
            interp_kind='linear'
             
            # Mark bad data as NaN for x and y coordinates
            x_left[~val_left] = np.nan
            x_right[~val_right] = np.nan
             
            # Calculate visual disparity before interpolation
            x_diff = x_right - x_left
             
            # Create a boolean array indicating where valid data is present in both eyes for x 
            valid_x = ~np.isnan(x_diff)
             
            # Create an array of sample indices
            eye_samples_x = np.arange(len(x_diff))
             
            # Interpolate over gaps in x_diff and y_diff
            f_x_diff = interp.interp1d(eye_samples_x[valid_x], x_diff[valid_x], kind=interp_kind, fill_value='extrapolate')
            x_diff_filled = f_x_diff(eye_samples_x)
             
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(t_nwb, x_diff_filled, c="orange", label='x gaze distance interpolated')
            plt.xlabel('Time')
            plt.ylabel('degrees of visual angle')
            plt.title('x gaze disparity (DVA) (interpolated)')
            ax.legend()
            ax.set_title(f'Patient {pat} x gaze disparity (DVA) (interpolated)')
            fig.tight_layout()
            fig_filename = os.path.join(sub_et_prep_dir, f"{mov.replace('_ieeg.nwb', '')}_x_dva_diff_interp.png")
            plt.show()
            fig.savefig(fig_filename)
            plt.close(fig)  
             
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
              
            #converting gaze position to pixel dimensions 
             
            #left eye
            x_left = x_left* screen_pix[0]  # Apply x dimension
            #x_left = x_left_filled * screen_pix[0]  # Apply x dimension
            n_left = len(x_left)
             
            data_left = np.core.records.fromarrays([x_left],names=['x'])
             
            # Right eye
            x_right = x_right * screen_pix[0]  # Apply x dimension
            #x_right = x_right_filled * screen_pix[0]  # Apply x dimension
            n_right = len(x_right)
             
            data_right = np.core.records.fromarrays([x_right],names=['x'])
             
            # Calculate px2deg for each dimension using the specific width and height in cm
            px2deg_x_left = np.rad2deg(2 * np.arctan(screen_cm[0] / (2 * dist_left_cm))) / screen_pix[0]
            px2deg_x_right = np.rad2deg(2 * np.arctan(screen_cm[0] / (2 * dist_right_cm))) / screen_pix[0]
             
            # Apply dva calculation to x and y coordinates for both eyes
            data_left['x'] *= px2deg_x_left
            data_right['x'] *= px2deg_x_right
             
            n = len(data_left)  # or len(data_right), since they should be the same
              
            # Calculate disparity in angle of eye vergence for
            gaze_disparity_x = data_right['x'] - data_left['x']
             
            #interpolate gaze disparity in DVA
            # Create a boolean array indicating where valid data is present in both eyes for x and y
            valid_disp_x = ~np.isnan(gaze_disparity_x)
             
            # Create an array of sample indices
            eye_samples_x = np.arange(len(gaze_disparity_x))
             
            # Interpolate over gaps in x_diff 
            f_x_disp = interp.interp1d(eye_samples_x[valid_disp_x], gaze_disparity_x[valid_disp_x], kind=interp_kind, fill_value='extrapolate')
            x_disp_filled = f_x_disp(eye_samples_x)
             
              # Plot results
            plt.figure(figsize=(14, 6))
            plt.plot(t_nwb, x_disp_filled, c="blue")
            plt.title(f' {pat} Gaze Disparity with Interpolation {movie}')
            plt.xlabel('Time')
            plt.ylabel('X Disparity')
            #plt.legend()
            plt.tight_layout()
            plt.show()
            plt.close()  
            
            # Calculate the standard deviation of the entire time series
            std_gaze_disp_x = np.nanstd(gaze_disparity_x)
            
            # Cap the interpolated values to 2 times the standard deviation
            cap_value = 2 * std_gaze_disp_x
            
            # Identify the interpolated indices (i.e., where the original data was NaN)
            interp_idx = np.isnan(gaze_disparity_x)
            
            # Create a copy of vis_fd_filled to apply the cap only on interpolated values
            x_disp_filled_capped = x_disp_filled.copy()
            x_disp_filled_capped[interp_idx] = np.clip(x_disp_filled[interp_idx], -cap_value, cap_value)
            
            # Example plot to visualize the capped values (optional)
            plt.figure(figsize=(12, 6))
            plt.plot(t_nwb, x_disp_filled_capped, label='Interpolated and Capped Gaze Disparity', alpha=0.5)
            plt.xlabel('Samples')
            plt.ylabel('Gaze Disparity')
            plt.title(f' {pat} Gaze Disparity with Interpolation and Capping of Interpolated Values {movie}')
            plt.legend()
            plt.grid(True)
            plt.show()
            fig_filename = os.path.join(sub_et_prep_dir, f"{mov.replace('_ieeg.nwb', '')}_gaze_disp_interp_plot.png")
            fig.savefig(fig_filename)
            plt.close()
                        
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
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(t_nwb, visual_focus_displacement, c = 'pink',label='Visual Focus Displacement')
            plt.xlabel('Time')
            plt.ylabel('Displacement (mm)')
            plt.title('Visual Focus Displacement Over Time')
            ax.legend()
            ax.set_title(f'Patient {pat} Vis Focus Disparity')
            fig.tight_layout()
            fig_filename = os.path.join(sub_et_prep_dir, f"{mov.replace('_ieeg.nwb', '')}_vf_disp_plot.png")
            fig.savefig(fig_filename)
            plt.close(fig) 
             
             # Assuming visual_focus_displacement and interp_kind are defined
            # valid_vis_fd = ~np.isnan(visual_focus_displacement)
            # eye_samples_fd = np.arange(len(visual_focus_displacement))
             # Interpolate missing values
            # f_vis_fd = interp.interp1d(eye_samples_fd[valid_vis_fd], visual_focus_displacement[valid_vis_fd], kind=interp_kind, fill_value='extrapolate')
             
             # Assuming visual_focus_displacement and interp_kind are defined
            valid_vis_fd = ~np.isnan(visual_focus_displacement)
            eye_samples_fd = np.arange(len(visual_focus_displacement))
             
             # Interpolate missing values
            f_vis_fd = interp.interp1d(eye_samples_fd[valid_vis_fd], visual_focus_displacement[valid_vis_fd], kind=interp_kind, fill_value='extrapolate')
            vis_fd_filled = f_vis_fd(eye_samples_fd)
             
             # Calculate the standard deviation of the entire time series
            std_vis_fd = np.nanstd(visual_focus_displacement)
             
             # Cap the interpolated values to 3 times the standard deviation
            cap_value = 3 * std_vis_fd
             
             # Identify the interpolated indices (i.e., where the original data was NaN)
            interpolated_indices = np.isnan(visual_focus_displacement)
             
             # Create a copy of vis_fd_filled to apply the cap only on interpolated values
            vis_fd_filled_capped = vis_fd_filled.copy()
            vis_fd_filled_capped[interpolated_indices] = np.clip(vis_fd_filled[interpolated_indices], -cap_value, cap_value)
             
             # Example plot to visualize the capped values (optional)
            plt.figure(figsize=(12, 6))
            plt.plot(eye_samples_fd, visual_focus_displacement, label='Original Visual Focus Displacement', alpha=0.5)
            plt.plot(eye_samples_fd, vis_fd_filled, label='Interpolated Visual Focus Displacement', alpha=0.5)
            plt.plot(eye_samples_fd, vis_fd_filled_capped, label='Capped Visual Focus Displacement', alpha=0.8)
            plt.xlabel('Samples')
            plt.ylabel('Visual Focus Displacement')
            plt.title(f' {pat} Visual Focus Displacement with Interpolation and Capping of Interpolated Values {movie}')
            plt.legend()
            plt.grid(True)
            plt.close()
             
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(t_nwb, vis_fd_filled_capped, label='Visual Focus Displacement (Interpolated)')
            plt.xlabel('Time')
            plt.ylabel('Displacement (mm)')
            plt.title('Visual Focus Displacement Over Time (Interpolated)')
            ax.legend()
            ax.set_title(f'Patient {pat} Vis Focus Disparity (interpolated)')
            fig.tight_layout()
            fig_filename = os.path.join(sub_et_prep_dir, f"{mov.replace('_ieeg.nwb', '')}{movie}_vf_disp_plot_interp.png")
            fig.savefig(fig_filename)
            plt.close(fig)  
               
            df = pd.DataFrame({
            'time': t_nwb,
            'gaze_dist_x_raw': x_diff,
            'gaze_dist_x_interp': x_diff_filled,
            'dva_gaze_disp_x': gaze_disparity_x,
            'dva_gaze_disp_x_interp': x_disp_filled_capped,
            'vis_focus_disp': visual_focus_displacement,
             'vis_fd_interp': vis_fd_filled_capped,
            })
             
             # Save to csv
            # For despicable_me_english, add run-1 to match compute_eye_measures.py expectations
            if movie == 'despicable_me_english':
                et_prep_filename = ('{:s}/{:s}'.format(sub_et_prep_dir, mov.replace('_ieeg.nwb', '_run-1_et_prep.csv')))
            else:
                et_prep_filename = ('{:s}/{:s}'.format(sub_et_prep_dir, mov.replace('_ieeg.nwb', '_et_prep.csv')))
             
            csv_filename = et_prep_filename 
            df.to_csv(csv_filename, index=False)  
         
            io.close()        

# Run the script if executed directly
if __name__ == "__main__":
    print("="*60)
    print("VERGENCE CALCULATION SCRIPT")
    print("="*60)
    print(f"Input directory: {data_dir}")
    print(f"Movie: {movie}")
    print(f"Patients: {patients}")
    print("="*60)
