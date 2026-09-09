# Calculate differences in gaze positions between right and left eye (gaze disparity, disparity of visual angle)

import os, sys, re
import numpy as np
import mne
from itertools import compress
from pynwb import NWBHDF5IO
from helpers import interp_bad_samples, combine_left_right, detect_saccades_remodnav
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

sys.path.append('/Users/christinechesebrough/Documents/HBML_all/EPIPE-master/Python/')
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


#%%
def interp_bad_data(data_gaze, data_val, combined_val, k_small, k_large, k_delay, visualize):
    """
    Interpolate bad samples in gaze data, considering the combined validity of both eyes.
    
    Inputs:
        data_gaze       - data array [samples x dimension (xy)]
        data_val        - index of samples with invalid data [samples]
        combined_val    - combined validity from both eyes [samples]
        k_small         - number of samples of small gaps that are filled
        k_large         - number of samples to add to large gaps (blinks, etc)
        k_delay         - number of samples to shift delay around large gaps
        visualize       - flag to plot gaze data before and after processing
        
    Returns:
        data_gaze       - processed data vector
        data_val        - updated vector of invalid data
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import signal, interpolate
    
    data_val = np.copy(data_val)
    
    # Ensure data_val reflects combined validity
    data_val = np.logical_or(data_val, np.logical_not(combined_val))
    
    # Plot signal
    if visualize:
        plt.figure()
        plt.plot(data_gaze, label='Original')
    
    # Create sample vector
    samples = np.arange(0, len(data_gaze))
    
    # Interpolate small gaps
    idx_gap = signal.convolve(data_val, np.ones(k_small), 'same')
    gap_val = signal.convolve(idx_gap==0, np.ones((k_small)), mode='same') == 0
    
    gaps = np.logical_and(gap_val, np.invert(data_val))
    data_val[gaps] = True
    
    f_gaze = interpolate.interp1d(samples[np.invert(gaps)],
                             data_gaze[np.invert(gaps)], 
                             kind='linear', fill_value="extrapolate")
    data_gaze[gaps] = f_gaze(samples[gaps])
    
    # Handle samples around longer gaps (mostly blinks)
    idx_bad = signal.convolve(np.invert(data_val), 
                              np.concatenate([np.zeros(k_delay),
                                              np.ones(k_large)]), 
                              'same') > 0
    
    data_gaze[idx_bad] = np.nan
    data_val[idx_bad] = False
    
    if visualize:
        plt.plot(data_gaze, label='Interpolated')
        plt.legend()
        
    return data_gaze, data_val

#%%

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
k_delay = 20
vis_bads = False

screen_pix = np.array([1920, 1080])
screen_cm = np.array([50.92, 28.64])

#%% Eye tracking parameters
t_diff = 0.1

dist = 60

# Screen dimentions [cm]
d = 23.8*2.54
ar = 16/9

k_small = 15 #15
k_large = 90 #90
k_delay = 20
vis_bads = True

screen_pix = np.array([1920, 1080])
screen_cm = np.array([50.92, 28.64])
    
#%%

#nwb_fname = '/Users/christinechesebrough/Documents/HBML_all/data/movies_nwb/sub-NS135/ses-implant01/sub-NS135_ses-implant01_task-DespMeEng_ieeg.nwb'
    
#nwb_fname = '/Users/christinechesebrough/Documents/HBML_all/data/movies_nwb/sub-NS153/ses-implant01/sub-NS153_ses-implant01_task-DespMeEng_ieeg.nwb'

nwb_fname = '/Volumes/Expansion/movies_nwb/sub-NS137/ses-implant01/sub-NS137_ses-implant01_task-DespMeEng_ieeg.nwb'

io = NWBHDF5IO(nwb_fname, mode='r', load_namespaces=True)

nwb = io.read()

 #%%  Preprocess and visualize eyetracking data

 # Get position and time
 #adcs is gaze position on screen 2D vector x, y between 0 and 1
l_gaze = nwb.processing['eye_tracking']['eyes']['l_eye_adcs'].data[:]
r_gaze = nwb.processing['eye_tracking']['eyes']['r_eye_adcs'].data[:]
 
x_right = r_gaze[:,0]
y_right = r_gaze[:,1]
 
x_left = l_gaze[:,0]
y_left = l_gaze[:,1]

x_right_orig = x_right
y_right_orig = y_right
x_left_orig = x_left
y_left_orig = y_left

#plot original gaze positions
plt.figure(figsize=(10, 5))
plt.scatter(x_right, y_right, c='blue', alpha=0.5, label='Right Eye Original')
plt.scatter(x_left, y_left, c='red', alpha=0.5, label='Left Eye Original')
plt.title('Original Gaze Positions')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.legend()
plt.grid(True)
plt.show()

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


val_left_bool_orig = val_left
val_right_bool_orig = val_right


#plt.plot(t_nwb,val_left,c='red',label = 'left_control')
#plt.plot(t_nwb,val_right,c='blue',label = 'right_control')
#plt.plot(t_nwb,comparison_array)

plt.plot(t_nwb,x_right,c='red',label = 'horz_right')
plt.plot(t_nwb,x_left,c='blue',label = 'horz_left')


val_left_orig = nwb.processing['eye_tracking']['eyes']['l_eye_adcs'].control[:]
val_right_orig = nwb.processing['eye_tracking']['eyes']['r_eye_adcs'].control[:]


# Sampling rate
fs_eye = 1/np.median(np.diff(t_nwb))

validity_df = pd.DataFrame()
validity_df['val_left_orig'] = val_left_orig
validity_df['val_right_orig'] = val_right_orig
validity_df['val_left_bool_orig'] = val_left_bool_orig
validity_df['val_right_bool_orig'] = val_right_bool_orig

validity_df['x_left_orig'] = x_left_orig
validity_df['y_left_orig'] = y_left_orig
validity_df['x_right_orig'] = x_right_orig
validity_df['y_right_orig'] = y_right_orig


#%%
# Exclude data outside the screen 

val_left[np.logical_or(y_left > 1, y_left < 0)] = False   
val_left[np.logical_or(x_left > 1, x_left < 0)] = False
val_right[np.logical_or(y_right > 1, y_right < 0)] = False
val_right[np.logical_or(x_right > 1, x_right < 0)] = False

valid_for_interpolation = np.logical_and(val_left, val_right)

validity_df['val_left_exclude_screen'] = val_left
validity_df['val_right_exclude_screen'] = val_right

#combined value showing if either the val_left and val_right before interpolation are 
combined_val = np.logical_and(val_left, val_right)

# Interpolate bad data for both eyes
x_left, val_left = interp_bad_data(x_left, val_left, #combined_val, 
                                   k_small, k_large, k_delay, vis_bads)
y_left, val_left = interp_bad_data(y_left, val_left, #combined_val,
                                   k_small, k_large, k_delay, vis_bads)
x_right, val_right = interp_bad_data(x_right, val_right,#combined_val, 
                                     k_small, k_large, k_delay, vis_bads)
y_right, val_right = interp_bad_data(y_right, val_right,#combined_val, 
                                     k_small, k_large, k_delay, vis_bads)

#dist_x = np.where(combined_val, x_right - x_left, np.nan)  # Replace invalid differences with np.nan or another placeholder
#dist_y = np.where(combined_val, y_right - y_left, np.nan)

dist_x = x_right - x_left
dist_y = y_right - y_left

#plot horizontal and vertical gaze distance
#plt.figure(figsize=(10, 5))
#plt.plot(t_nwb,dist_x,c = 'blue')
#plt.plot(t_nwb,dist_y,c = 'red')
#plt.title('Gaze Distance')
#plt.xlabel('Time')
#plt.ylabel('')
#plt.legend()
#plt.grid(True)
#plt.show()

#plot horizontal and vertical gaze distance
plt.figure(figsize=(10, 5))
plt.plot(t_nwb,dist_x,c = 'orange')
plt.plot(t_nwb,dist_y,c = 'purple')
plt.title('Gaze Distance')
plt.xlabel('Time')
#plt.ylabel('')
plt.legend()
plt.grid(True)
plt.show()

validity_df['x_left_interp'] = x_left
validity_df['y_left_interp'] = y_left
validity_df['x_right_interp'] = x_right
validity_df['y_right_interp'] = y_right

validity_df['val_left_interp'] = val_left
validity_df['val_right_interp'] = val_right
validity_df['combined_val']=combined_val

#if x_left_interp and/or y_left_interp == TRUE but combined_val == FALSE


 # Calculate gaze disparity as the difference between right and left gaze positions
dist_x = x_right - x_left
dist_y = y_right - y_left

validity_df['x_dist'] = dist_x
validity_df['y_dist'] = dist_y

validity_df.to_csv('vergence_vals_in_process_132_2.csv')


#%% Interpolate all x values
    
interp_kind='linear'
# Vector of samples
eye_samples = np.arange(len(x_left))

# Invert the boolean arrays to mark 'False' (bad data) as NaN for interpolation
x_left[~val_left] = np.nan
x_right[~val_right] = np.nan

# Interpolate missing values in x_left
idx_nan_left = np.isnan(x_left)
if np.any(idx_nan_left):
    f_left = interp.interp1d(eye_samples[~idx_nan_left], x_left[~idx_nan_left], 
                                  kind=interp_kind, fill_value='extrapolate')
    x_left_filled = np.copy(x_left)
    x_left_filled[idx_nan_left] = f_left(eye_samples[idx_nan_left])
else:
    x_left_filled = x_left

# Interpolate missing values in x_right
idx_nan_right = np.isnan(x_right)
if np.any(idx_nan_right):
    f_right = interp.interp1d(eye_samples[~idx_nan_right], x_right[~idx_nan_right], 
                                   kind=interp_kind, fill_value='extrapolate')
    x_right_filled = np.copy(x_right)
    x_right_filled[idx_nan_right] = f_right(eye_samples[idx_nan_right])
else:
    x_right_filled = x_right
    
    
plt.plot(t_nwb,x_right_filled,c='red',label = 'right_x_filled')
plt.plot(t_nwb,x_left_filled,c='blue',label = 'left x_filled')

# Calculate the difference between the interpolated left and right positions
x_diff_filled = x_left_filled - x_right_filled
plt.plot(t_nwb,x_diff_filled)   

#%% Interpolate all y values
    
interp_kind='linear'
# Vector of samples
eye_samples = np.arange(len(y_left))

# Invert the boolean arrays to mark 'False' (bad data) as NaN for interpolation
y_left[~val_left] = np.nan
y_right[~val_right] = np.nan

# Interpolate missing values in x_left
idx_nan_left = np.isnan(y_left)
if np.any(idx_nan_left):
    f_left = interp.interp1d(eye_samples[~idx_nan_left], y_left[~idx_nan_left], 
                                  kind=interp_kind, fill_value='extrapolate')
    y_left_filled = np.copy(y_left)
    y_left_filled[idx_nan_left] = f_left(eye_samples[idx_nan_left])
else:
    y_left_filled = y_left

# Interpolate missing values in x_right
idx_nan_right = np.isnan(y_right)
if np.any(idx_nan_right):
    f_right = interp.interp1d(eye_samples[~idx_nan_right], y_right[~idx_nan_right], 
                                   kind=interp_kind, fill_value='extrapolate')
    y_right_filled = np.copy(y_right)
    y_right_filled[idx_nan_right] = f_right(eye_samples[idx_nan_right])
else:
    y_right_filled = y_right
    
    
plt.plot(t_nwb,y_right_filled,c='red',label = 'right_y_filled')
plt.plot(t_nwb,y_left_filled,c='blue',label = 'left y_filled')

# Calculate the difference between the interpolated left and right positions
y_diff_filled = y_left_filled - y_right_filled
plt.plot(t_nwb,y_diff_filled,c="yellow")    
    
#%% Gaze disparity (interpolated gaps)

 # Calculate gaze disparity as the difference between right and left gaze positions
dist_x = x_diff_filled
dist_y = y_diff_filled
 
#Extract distance from eyes to screen and eyes to eye tracker
eye_pos_left, _ = interp_bad_data(eye_pos_left[:,2], val_left, k_small, k_large, k_delay, vis_bads)
eye_pos_right, _ = interp_bad_data(eye_pos_right[:,2], val_left, k_small, k_large, k_delay, vis_bads)
gaze_pos_left, _ = interp_bad_data(gaze_pos_left[:,2], val_left, k_small, k_large, k_delay, vis_bads)
gaze_pos_right, _ = interp_bad_data(gaze_pos_right[:,2], val_left, k_small, k_large, k_delay, vis_bads)

dist_left = np.nanmedian(eye_pos_left - gaze_pos_left)
dist_right = np.nanmedian(eye_pos_right - gaze_pos_right)

# Distance in cm (converted from median distance in mm to cm for both eyes)
dist_left_cm = dist_left / 10
dist_right_cm = dist_right / 10
#avg distance should be the median from both eyes averaged? Maybe this shouldn't be a constant
avg_dist = (dist_left_cm + dist_right_cm)/2
 
#converting gaze position to pixel dimensions
# Left eye
x_left = x_left_filled * screen_pix[0]  # Apply x dimension
y_left = y_left_filled * screen_pix[1]  # Apply y dimension
n_left = len(x_left)

data_left = np.core.records.fromarrays([
  x_left,
  y_left],
  names=['x', 'y'])

# Right eye
x_right = x_right_filled * screen_pix[0]  # Apply x dimension
y_right = y_right_filled * screen_pix[1]  # Apply y dimension
n_right = len(x_right)

data_right = np.core.records.fromarrays([
    x_right,
    y_right],
    names=['x', 'y'])

# Calculate px2deg for each dimension using the specific width and height in cm
px2deg_x_left = np.rad2deg(2 * np.arctan(screen_cm[0] / (2 * dist_left_cm))) / screen_pix[0]
px2deg_y_left = np.rad2deg(2 * np.arctan(screen_cm[1] / (2 * dist_left_cm))) / screen_pix[1]

px2deg_x_right = np.rad2deg(2 * np.arctan(screen_cm[0] / (2 * dist_right_cm))) / screen_pix[0]
px2deg_y_right = np.rad2deg(2 * np.arctan(screen_cm[1] / (2 * dist_right_cm))) / screen_pix[1]

# Apply dva calculation to x and y coordinates for both eyes
data_left['x'] *= px2deg_x_left
data_left['y'] *= px2deg_y_left

data_right['x'] *= px2deg_x_right
data_right['y'] *= px2deg_y_right

# plot DVA
plt.figure(figsize=(10, 5))
plt.scatter(data_left['x'], data_left['y'], c='blue', alpha=0.5,  label='Left Eye DVA')
plt.scatter(data_right['x'], data_right['y'], c='red', alpha=0.5, label='Right Eye DVA')
plt.title('Gaze Positions in Degrees of Visual Angle')
plt.xlabel('Degrees of Visual Angle (X)')
plt.ylabel('Degrees of Visual Angle (Y)')
plt.legend()
plt.grid(True)
plt.show()
 
n = len(data_left)  # or len(data_right), since they should be the same
 
# Calculate disparity in angle of eye vergence for x and y separately
gaze_disparity_x = data_right['x'] - data_left['x']
gaze_disparity_y = data_right['y'] - data_left['y']

# Alternatively, calculate gaze disparity and angle in terms of the centroids of both x and y for each time point
gaze_disparity = np.sqrt((gaze_disparity_x)**2 + (gaze_disparity_y)**2)
gaze_disparity_angle = np.arctan2((data_right['y']-data_left['y']), (data_right['x']-data_left['x']))  # Angle in radians

#%% Gaze disparity (non-interpolated gaps)


 # Calculate gaze disparity as the difference between right and left gaze positions
#dist_x = x_diff_filled
#dist_y = y_diff_filled
 
#Extract distance from eyes to screen and eyes to eye tracker
eye_pos_left, _ = interp_bad_data(eye_pos_left[:,2], val_left, k_small, k_large, k_delay, vis_bads)
eye_pos_right, _ = interp_bad_data(eye_pos_right[:,2], val_left, k_small, k_large, k_delay, vis_bads)
gaze_pos_left, _ = interp_bad_data(gaze_pos_left[:,2], val_left, k_small, k_large, k_delay, vis_bads)
gaze_pos_right, _ = interp_bad_data(gaze_pos_right[:,2], val_left, k_small, k_large, k_delay, vis_bads)

dist_left = np.nanmedian(eye_pos_left - gaze_pos_left)
dist_right = np.nanmedian(eye_pos_right - gaze_pos_right)

# Distance in cm (converted from median distance in mm to cm for both eyes)
dist_left_cm = dist_left / 10
dist_right_cm = dist_right / 10
#avg distance should be the median from both eyes averaged? Maybe this shouldn't be a constant
avg_dist = (dist_left_cm + dist_right_cm)/2
 
#converting gaze position to pixel dimensions
# Left eye
x_left = x_left * screen_pix[0]  # Apply x dimension
y_left = y_left * screen_pix[1]  # Apply y dimension
n_left = len(x_left)

data_left = np.core.records.fromarrays([
  x_left,
  y_left],
  names=['x', 'y'])

# Right eye
x_right = x_right * screen_pix[0]  # Apply x dimension
y_right = y_right * screen_pix[1]  # Apply y dimension
n_right = len(x_right)

data_right = np.core.records.fromarrays([
    x_right,
    y_right],
    names=['x', 'y'])

# Calculate px2deg for each dimension using the specific width and height in cm
px2deg_x_left = np.rad2deg(2 * np.arctan(screen_cm[0] / (2 * dist_left_cm))) / screen_pix[0]
px2deg_y_left = np.rad2deg(2 * np.arctan(screen_cm[1] / (2 * dist_left_cm))) / screen_pix[1]

px2deg_x_right = np.rad2deg(2 * np.arctan(screen_cm[0] / (2 * dist_right_cm))) / screen_pix[0]
px2deg_y_right = np.rad2deg(2 * np.arctan(screen_cm[1] / (2 * dist_right_cm))) / screen_pix[1]

# Apply dva calculation to x and y coordinates for both eyes
data_left['x'] *= px2deg_x_left
data_left['y'] *= px2deg_y_left

data_right['x'] *= px2deg_x_right
data_right['y'] *= px2deg_y_right

#data_left['x'] = np.nan_to_num(data_left['x'], nan=0.0)
#data_left['y'] = np.nan_to_num(data_left['y'], nan=0.0)

#data_right['x'] = np.nan_to_num(data_right['x'], nan=0.0)
#data_right['y'] = np.nan_to_num(data_right['y'], nan=0.0)

# plot DVA
plt.figure(figsize=(10, 5))
plt.scatter(data_left['x'], data_left['y'], c='blue', alpha=0.5,  label='Left Eye DVA')
plt.scatter(data_right['x'], data_right['y'], c='red', alpha=0.5, label='Right Eye DVA')
plt.title('Gaze Positions in Degrees of Visual Angle')
plt.xlabel('Degrees of Visual Angle (X)')
plt.ylabel('Degrees of Visual Angle (Y)')
plt.legend()
plt.grid(True)
plt.show()
 
n = len(data_left)  # or len(data_right), since they should be the same
 
# Calculate disparity in angle of eye vergence for x and y separately
gaze_disparity_x = data_right['x'] - data_left['x']
gaze_disparity_y = data_right['y'] - data_left['y']

# Apply np.nan_to_num to gaze disparity arrays
#gaze_disparity_x = np.nan_to_num(gaze_disparity_x, nan=0.0)
#gaze_disparity_y = np.nan_to_num(gaze_disparity_y, nan=0.0)

# Alternatively, calculate gaze disparity and angle in terms of the centroids of both x and y for each time point
gaze_disparity = np.sqrt((gaze_disparity_x)**2 + (gaze_disparity_y)**2)
gaze_disparity_angle = np.arctan2((data_right['y']-data_left['y']), (data_right['x']-data_left['x']))  # Angle in radians

#%% Plots of processed time series

# Plot horizontal gaze disparity 
plt.figure(figsize=(10, 5))
plt.plot(t_nwb, gaze_disparity_x, c='green', alpha=0.5, label='Horizontal Gaze Disparity (NS135)')
plt.title('Gaze Disparity')
plt.xlabel('Time')
#plt.ylabel('')
plt.legend()
plt.grid(True)
plt.show()
  
# Plot vertical gaze disparity 
plt.figure(figsize=(10, 5))
plt.plot(t_nwb, gaze_disparity_y, c='green', alpha=0.5, label='Vertical Gaze Disparity')
plt.title('Gaze Disparity')
plt.xlabel('Time')
plt.ylabel('Disparity (Degrees of Visual Angle)')
plt.legend()
plt.grid(True)
plt.show()

# Plot centroid gaze disparity 
plt.figure(figsize=(10, 5))
plt.plot(t_nwb, gaze_disparity, c='green', alpha=0.5, label='Gaze Disparity (centroid)')
plt.title('Gaze Disparity')
plt.xlabel('Time')
plt.ylabel('distance (pixels))')
plt.legend()
plt.grid(True)
plt.show()
 
plt.figure(figsize=(10, 5))
#plt.plot(t_nwb, gaze_disparity, c='black', alpha=0.5, label='Gaze Disparity (centroid)')
#plt.plot(t_nwb, gaze_disparity_y, c='green', alpha=0.5, label='Vertical Gaze Disparity')
plt.plot(t_nwb, gaze_disparity_x, c='red', alpha=0.5, label='Horizontal Gaze Disparity')
#plt.plot(t_nwb, dist_x*100, label='r - l gaze distance')
plt.title('Gaze Disparity')
plt.xlabel('Time')
plt.ylabel('Disparity (Degrees of Visual Angle)')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(t_nwb, gaze_disparity, c='black', label='Gaze Disparity (centroid)')
plt.plot(t_nwb, gaze_disparity_y, c='green',  label='Vertical Gaze Disparity')
plt.plot(t_nwb, gaze_disparity_x, c='red', label='Horizontal Gaze Disparity')
#plt.plot(t_nwb, dist_x*100,label='r - l gaze distance')
plt.title('Gaze Disparity')
plt.xlabel('Time')
plt.ylabel('Disparity (Degrees of Visual Angle)')
plt.legend()
plt.grid(True)
plt.show()
 

 #%%

#Calculate visual focus displacement per Huang et al. (2019)    

beta = 0.283
PD_cm = IPD_cm  # Use the approximated IPD from before

# Calculate E (gaze disparity in the world coordinate)
G = np.linalg.norm(gaze_disparity)  # Euclidean norm of the gaze disparity vector
E = beta * G

# Convert D from cm to mm for consistency with formula
D_mm = ((dist_left_cm + dist_right_cm) / 2) * 10  # Average distance from eyes to screen converted to mm
    
# Initialize visual focus displacement array
visual_focus_displacement = np.zeros(len(gaze_disparity_x))

for i in range(len(gaze_disparity_x)):
    G = np.linalg.norm(gaze_disparity_x[i])  #G is euclidean norm (2) of gaze disparity
    E = beta * G  # Calculate E using the current G
    
    # Apply conditions for convergence and divergence
    if gaze_disparity_x[i] > 0:  # Divergence
        visual_focus_displacement[i] = E * D_mm / (PD_cm * 10 - E)
    else:  # Convergence
        visual_focus_displacement[i] = -E * D_mm / (PD_cm * 10 + E)

# Visualize Visual Focus Displacement over Time
plt.figure(figsize=(12, 6))
plt.plot(t_nwb, visual_focus_displacement, label='Visual Focus Displacement (NS135)')
plt.xlabel('Time')
plt.ylabel('Displacement (mm)')
plt.title('Visual Focus Displacement Over Time')
plt.legend()
plt.grid(True)
plt.show()

#%% Create array with all disparity measures
data_combined = np.core.records.fromarrays([
    t_nwb,
    dist_x,
    dist_y,
    gaze_disparity_x,
    gaze_disparity_y,
    gaze_disparity,
    gaze_disparity_angle,
    visual_focus_displacement],
    names=['time',
           'gaze_dist_x',
           'gaze_dist_y',
           'dva_gaze_disparity_x', 
           'dva_gaze_disparity_y',
           'gaze_disparity_centroids',
           'gaze_disparity_angle',
           'vis_displacement'])

data_array = np.array(data_combined.tolist())

csv_file_path = '/Users/christinechesebrough/Documents/HBML_all/data/results/NS135/Eye_prep/eye_vergence.csv'

# Save the output array to a CSV file
np.savetxt(csv_file_path, data_array, delimiter=',', header=','.join(data_combined.dtype.names), comments='')

     
