#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 12:15:12 2024

@author: christinechesebrough
"""
#%%
import os, sys, re
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import numpy as np
import mne
from pynwb import NWBHDF5IO
import pandas as pd
import matplotlib

import matplotlib.pyplot as plt
#import cv2
import scipy.interpolate as interp
import scipy.stats as stats

import scipy.signal as signal
#import temporal_response_function as trf
from mne.time_frequency import psd_array_welch
from mne.filter import filter_data
from scipy.io.wavfile import write
import importlib.util
import glob
from scipy.signal import hilbert
from collections import defaultdict

mne.viz.set_browser_backend('qt')

#matplotlib.use("Qt5Agg")


sys.path.insert(0, '/Users/christinechesebrough/Documents/EPIPE-movie_nwb/Python')
sys.path.append('/Users/christinechesebrough/Documents/iEEG2NWB-main')

import pycircstat2
from pycircstat2 import hypothesis
from epipe import inspectNwb, nwb2mne, read_ielvis, reref_avg, reref_bipolar,filter_hfa_continuous, reref_white_matter, reref_white_matter_strict

wd = '/Volumes/Samsung/scripts/movies_ET_attn_main'
src_dir = os.path.join(wd, 'src')
src_dir = os.path.abspath(src_dir)

# Add src to path if not already there
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Peek inside src to confirm the file is really there and spelled exactly
print('src contents (first 20):')
print([os.path.basename(p) for p in glob.glob(os.path.join(src_dir, '*'))][:20])

# 1) Put src on sys.path (at the front)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
print('on sys.path?', src_dir in sys.path)

import eeg_preproc_helpers

# Import EEG preprocessing helper functions
from eeg_preproc_helpers import (
    plot_power_spectra,  plot_psd_batched, plot_psd_with_scales,
    create_file_paths, apply_highpass_filter,
    save_bad_channels, load_bad_channels, check_bad_channels_integrity, load_preprocessed_data, validate_mne_structure, preserve_mne_structure, restore_mne_structure, load_data_with_bad_channels, summarize_bad_channels, 
    interpolate_spikes, make_groups_from_prefix, regress_out_noise_by_group,
    detect_spikes_all_channels, reref_avg_by_group,
    ProcessingLogger, detect_spikes_ref1
)

# path to temporal response function
module_path = '/Users/christinechesebrough/Documents/data_standardization-main/src/temporal_response_function.py'
module_name = 'temporal_response_function'

spec = importlib.util.spec_from_file_location(module_name, module_path)
trf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trf)

def electrode_group(ch: str) -> str:
    return re.sub(r'\d+$', '', ch)

def pick_ptd_column(elecs_subs: pd.DataFrame) -> str | None:
    for col in elecs_subs.columns:
        if 'ptd' in col.lower():
            return col
    return None

def compute_line_noise_ratio(psd, freqs, mains=60.0, tol=1.0):
    # ratio: power around mains+harmonics divided by broadband power
    broadband = psd[(freqs >= 1) & (freqs <= 150)].mean()
    ln_bins = []
    for h in [mains, 2*mains, 3*mains]:
        ln_bins.append(psd[(freqs >= h - tol) & (freqs <= h + tol)].mean())
    line = np.nanmean(ln_bins)
    return float(line / (broadband + 1e-12))

def make_wm_metrics_df(ecogPreproc, wm_contacts, elecs_subs, fs_data, fmin=1, fmax=150):
    # Clean candidates
    wm_clean = [ch for ch in wm_contacts
                if ch in ecogPreproc.ch_names and ch not in ecogPreproc.info['bads']]
    if len(wm_clean) == 0:
        raise RuntimeError("No WM contacts remain after excluding bads / missing channels.")

    # Extract PTD values
    ptd_col = pick_ptd_column(elecs_subs)
    if ptd_col is None:
        raise RuntimeError("No PTD column found in elecs_subs.")

    ptd_map = dict(zip(elecs_subs['Label'], elecs_subs[ptd_col]))

    # Data
    wm_data = ecogPreproc.get_data(picks=wm_clean)
    wm_psd, freqs = psd_array_welch(wm_data, sfreq=fs_data, fmin=fmin, fmax=fmax, n_fft=int(fs_data * 2))

    # Metrics per channel
    mean_psd = wm_psd.mean(axis=1)
    variance = wm_data.var(axis=1)

    line_ratio = []
    for i in range(wm_psd.shape[0]):
        line_ratio.append(compute_line_noise_ratio(wm_psd[i], freqs))

    df = pd.DataFrame({
        'Contact': wm_clean,
        'Group': [electrode_group(ch) for ch in wm_clean],
        'Mean_PSD': mean_psd,
        'Variance': variance,
        'LineNoiseRatio': line_ratio,
        'PTD': [ptd_map.get(ch, np.nan) for ch in wm_clean],
    })

    return df


def robust_z(x):
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) + 1e-12
    return (x - med) / (1.4826 * mad)

def auto_select_wm_refs(wm_df, *,
                        ptd_thresh=-0.8,
                        max_per_group=1,
                        max_total=12,
                        var_floor_quantile=0.02):
    df = wm_df.copy()

    # Filter to WM-like contacts and those with PTD present
    df = df[df['PTD'].notna()]
    df = df[df['PTD'] < ptd_thresh].copy()

    if len(df) == 0:
        raise RuntimeError("No WM candidates pass PTD threshold.")

    # Avoid 'dead' contacts: extremely low variance can be suspicious
   # var_floor = df['Variance'].quantile(var_floor_quantile)
   # df = df[df['Variance'] > var_floor].copy()

    # Robust z-scoring: lower is better for PSD, variance, line noise
    z_psd = robust_z(df['Mean_PSD'].values)
    z_var = robust_z(df['Variance'].values)
    z_ln  = robust_z(df['LineNoiseRatio'].values)

    # For PTD, "more negative is better" (more WM). Convert so lower is better:
    # e.g., distance to -1
    ptd_dist = np.abs(df['PTD'].values - (-1.0))
    z_ptd = robust_z(ptd_dist)

    # Composite score (tweak weights as you like)
    df['Score'] = (0.20*z_psd + 0.20*z_var +  0.40*z_ptd)

    # Select top N per group
    selected = (
        df.sort_values('Score')
          .groupby('Group', group_keys=False)
          .head(max_per_group)
          .sort_values('Score')
    )

    # Cap total
    selected = selected.head(max_total)

    return selected


def plot_stft_spectrogram_raw(
    raw,
    ch_names,
    fmin=80,
    fmax=130,
    tmin=None,
    tmax=None,
    win_sec=1.0,
    overlap=0.75,
    db=True
):
    """
    STFT spectrogram for Raw data (no averaging), plotted for each channel.
    """

    # Crop if desired (recommended for very long recordings)
    raw_use = raw.copy()
    if (tmin is not None) or (tmax is not None):
        raw_use.crop(tmin=tmin, tmax=tmax)

    sfreq = raw_use.info["sfreq"]
    data, times = raw_use.get_data(picks=ch_names, return_times=True)

    nperseg = int(round(win_sec * sfreq))
    noverlap = int(round(overlap * nperseg))

    for i, ch in enumerate(ch_names):
        x = data[i]
        f, t, Sxx = spectrogram(
            x,
            fs=sfreq,
            nperseg=nperseg,
            noverlap=noverlap,
            scaling="density",
            mode="psd"
        )

        keep = (f >= fmin) & (f <= fmax)
        f2 = f[keep]
        S2 = Sxx[keep, :]

        # Convert to dB for visibility
        if db:
            Splot = 10 * np.log10(S2 + 1e-20)
        else:
            Splot = S2

        plt.figure(figsize=(10, 4))
        plt.pcolormesh(t + (tmin or 0), f2, Splot, shading="auto")
        plt.ylim(fmin, fmax)
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.title(f"Spectrogram (STFT, no averaging) — {ch}")
        plt.colorbar(label="PSD (dB)" if db else "PSD")
        plt.tight_layout()
        plt.show()


#%% Parameters
full_task_name = 'movies'
pipeline_name = 'preprocess_movies'
pipeline_version = 'v.1.0.725'

resample_fs = 600

# Types of references to use in analyses
# Must be a list containing at least one of the options: "avg", "bip"
ref_types = ['wm_bip']

n_jobs = 16

convert_db = True

# Frequency range
freq_range = [70, 170]
n_freq_bins = 10
freq_space = 'log'      # 'log', 'lin'

resample_bha_fs = 100

freq_bands = ['low','middle','high']

movie_table = pd.read_excel('/Volumes/Samsung/Movie_data/data/electrode_localization/movie_table.xlsx')

#%% Define directories

data_dir = '/Volumes/Samsung/Movie_data/movies_nwb_standard'
#data_dir = '/Volumes/Samsung/Movie_data/movies_new_nwb'
#data_dir = '/Volumes/Samsung/AV40_data/new_converted'
fs_dir = '/Volumes/Samsung/anatomy'
#prep_dir = '/Volumes/Samsung/movie_data_new/movies_new_prep'
prep_dir = '/Volumes/Samsung/Movie_data/movies_prep_standard'

frame_dir = '/Volumes/Samsung/Movie_data/data/video_frames'
lum_dir = '/Volumes/Samsung/Movie_data/data/luminance'

if not os.path.exists(lum_dir):
    os.makedirs(lum_dir)

compute_luminance = False

et_prep_dir = 'Eye_prep'
audio_dir = 'Audio'
neural_prep_dir = 'Neural_prep'
hfa_dir = 'HFA'


# Single File Processing Mode
# Set to True to process only one specific file
single_file_mode = True 

# When single_file_mode is True, specify the file to process
# Format: 'patient_id' or 'patient_id_implant_number'
target_patient = 'NS178'#'NS205'  # e.g., 'NS189' or 'NS189_01'
#target_movie = 'sub-NS127_ses-02_task-the_present_run-1_ieeg.nwb'
#target_movie = 'sub-NS190_ses-01_task-present_run-1_ieeg.nwb'
target_movie = 'sub-NS178_ses-01_task-present_run-01_ieeg.nwb'
#target_movie = 'sub-NS178_ses-01_task-present_run-01_ieeg.nwb'


vid = 'the_present'#["dme", "despicable_me_english"]

if vid == 'inscapes':
    vid_keys = ['inscapes','Inscapes','inkscapes']
if vid == 'despicable_me_english':
    vid_keys = ['despicable_me_english','dme']
if vid == 'the_present':
    vid_keys = ['the_present','Present','present']

patients = ["NS208"]#,"NS190","NS191","NS193","NS194","NS201_02","NS204","NS205"]


#%% Initialize loop and open NWB

broken_nwb = []

if single_file_mode:
    # Single file processing mode
    print(f"\n{'='*60}")
    print(f"SINGLE FILE PROCESSING MODE")
    print(f"Target Patient: {target_patient}")
    print(f"Target Movie: {target_movie}")
    print(f"{'='*60}\n")
    
    # Extract session number from the target movie filename
    # Expected format: NS189_ses-02_task-despicable_me_english_run-01_ieeg.nwb
    if 'ses-0' in target_movie:
        # Extract session number from filename
        ses_match = re.search(r'ses-(\d+)', target_movie)
        if ses_match:
            ses_num = ses_match.group(1)
            imp = f'ses-{ses_num}'
        else:
            print(f"❌ ERROR: Could not extract session number from filename: {target_movie}")
            #sys.exit(1)
    else:
        # Fallback to ses-01 if no session info in filename
        imp = 'ses-01'
    
    # Set patient path
    pat = f'{target_patient}'

    # Validate that the target file exists
    if data_dir == '/Volumes/Samsung/Movie_data/movies_nwb_standard':
        target_nwb_path = '{:s}/{:s}/{:s}/{:s}'.format(data_dir, pat, imp, target_movie)
    elif data_dir == '/Volumes/Samsung/movie_data_new/movies_new_nwb_fall25':
        target_nwb_path = '{:s}/{:s}/{:s}'.format(data_dir, pat, target_movie)
    elif data_dir == '/Volumes/Samsung/Movie_data/movies_new_nwb':
        target_nwb_path = '{:s}/{:s}/{:s}/ieeg/{:s}'.format(data_dir, pat, imp, target_movie)
    elif data_dir =='/Volumes/Samsung/AV40_data/new_converted':
        target_nwb_path = '{:s}/{:s}/{:s}/ieeg/{:s}'.format(data_dir, pat, imp, target_movie)
    
    if not os.path.exists(target_nwb_path):
        print(f"❌ ERROR: Target file not found: {target_nwb_path}")
        print("Please check the target_patient and target_movie parameters.")
        print(f"Expected path: {target_nwb_path}")
        #sys.exit(1)
    
    print(f"✅ Target file found: {target_nwb_path}")
    
    # Process single file
    patients_to_process = [pat]
    implants_to_process = [imp]
    movies_to_process = [target_movie]
    
else:
    # Batch processing mode
    print(f"\n{'='*60}")
    print(f"BATCH PROCESSING MODE")
    print(f"Patients to process: {patients}")
    print(f"{'='*60}\n")
    
    patients_to_process = [f'{p}' for p in patients]
    implants_to_process = []
    movies_to_process = []

# Process files
for pat in patients_to_process:
        
    implants = os.listdir('{:s}/{:s}'.format(data_dir, pat))
       
    if single_file_mode:
        # Single file mode - use predefined implants and movies
        implants_to_process = implants_to_process
        movies_to_process = movies_to_process
    else:
        # Batch mode - get all implants and movies
        implants_to_process = implants
        movies_to_process = []
    
    for imp in implants_to_process:
        
        if data_dir == '/Volumes/Samsung/Movie_data/movies_nwb_standard':
            ieeg_dir = '{:s}/{:s}/{:s}'.format(data_dir, pat, imp)
        elif data_dir == '/Volumes/Samsung/movie_data_new/movies_new_nwb_fall25':
            ieeg_dir = '{:s}/{:s}/'.format(data_dir, pat)
        elif data_dir == '/Volumes/Samsung/Movie_data/movies_new_nwb':
            ieeg_dir = '{:s}/{:s}/{:s}'.format(data_dir, pat, imp)
        elif data_dir =='/Volumes/Samsung/AV40_data/new_converted':
            ieeg_dir = '{:s}/{:s}/{:s}'.format(data_dir, pat, imp)

        if single_file_mode:
            # Single file mode - use predefined movie
            movies = movies_to_process
        else:
            # Batch mode - get all movies
            movies = [f for f in os.listdir(ieeg_dir) if f.endswith('.nwb')]
            # Filter for specific video if vid is defined
            if 'vid' in locals():
                vid_keys = [vid] if isinstance(vid, str) else list(vid)
                movies = [
                    mov for mov in movies
                    if any(k in mov for k in vid_keys)
                ]
        
        # Extract session number from implant name (e.g., 'ses-01' -> 1)
        ses_match = re.search(r'ses-(\d+)', imp)
        if ses_match:
            ses_num = int(ses_match.group(1))
        else:
            ses_num = 1  # Default to session 1 if no match
        
        if ses_num == 1:
            pat_fs = pat.replace('sub-', '')
        else:
            pat_fs = '{:s}_{:02d}'.format(pat.replace('sub-', ''), ses_num)

        sub_fs_dir = '{:s}/{:s}'.format(fs_dir, pat_fs)
        
        sub_et_prep_dir = '{:s}/{:s}/{:s}'.format(prep_dir, pat_fs, et_prep_dir)

        for mov in movies:
            # Create standardized file paths for this movie
            file_paths = create_file_paths(
                patient_id=pat_fs,
                implant_id=imp,
                movie_filename=mov,
                prep_dir=prep_dir,
                neural_prep_dir=neural_prep_dir,
                hfa_dir=hfa_dir
            )
            
            # Initialize logger for this movie
            movie_name = file_paths['movie_base']
            logger = ProcessingLogger(pat_fs, movie_name, file_paths['sub_prep_dir'])
            
            # Construct the NWB file path
            if data_dir == '/Volumes/Samsung/Movie_data/movies_nwb_standard':
                nwb_fname = '{:s}/{:s}/{:s}/{:s}'.format(data_dir, pat, imp, mov)
            elif data_dir == '/Volumes/Samsung/movie_data_new/movies_new_nwb_fall25':
                nwb_fname = '{:s}/{:s}/{:s}'.format(data_dir, pat, mov)
            elif data_dir == '/Volumes/Samsung/Movie_data/movies_new_nwb':
                nwb_fname = '{:s}/{:s}/{:s}/ieeg/{:s}'.format(data_dir, pat, imp, mov)
            elif data_dir =='/Volumes/Samsung/AV40_data/new_converted':
                nwb_fname = '{:s}/{:s}/{:s}/ieeg/{:s}'.format(data_dir, pat, imp,mov)

            # proceed
            print(f"\n{'='*60}")
            print(f"PROCESSING FILE:")
            print(f"Patient: {pat_fs}")
            print(f"Implant: {imp}")
            print(f"Movie: {mov}")
            print(f"File: {nwb_fname}")
            print(f"{'='*60}\n")
            
            if single_file_mode:
                print("⚠️ SINGLE FILE MODE: Processing only this file")
                print("⚠️ Make sure this is the correct file before proceeding!\n")
                
                # Ask for confirmation in single file mode
                confirm = input("Continue with this file? (y/n): ")
                if confirm.lower() != 'y':
                    print("Processing cancelled.")
                    sys.exit(0)
                print("✅ Proceeding with processing...\n")
            
            # Log file information
            logger.log_decision("FILE", f"Processing: {nwb_fname}")
            logger.log_decision("PATIENT", f"Patient: {pat_fs}, Implant: {imp}")

#%%
            
            # Define the subject-specific directory
            subid = os.path.basename(sub_fs_dir)
            elecReconDir = os.path.join(sub_fs_dir, 'elec_recon')
            
            # List all files in the directory and filter for the one containing 'correspondence'
            excel_files = [f for f in os.listdir(elecReconDir)
                           if 'correspondence' in f and
                           f.endswith('.xlsx') and
                           not f.startswith('.')]
            
            if not excel_files:
                print("No correspondence Excel file found.")
            
            elif len(excel_files) == 1:
                excel_file = os.path.join(elecReconDir, excel_files[0])
                print(f"Found electrode correspondence file: {excel_file}")
            
            else:
                print("Multiple correspondence Excel files found:")
                for i, f in enumerate(excel_files):
                    print(f"  [{i}] {f}")
            
                while True:
                    try:
                        idx = int(input("Select the correct file by index: "))
                        if 0 <= idx < len(excel_files):
                            excel_file = os.path.join(elecReconDir, excel_files[idx])
                            print(f"Selected electrode correspondence file: {excel_file}")
                            break
                        else:
                            print("Index out of range. Try again.")
                    except ValueError:
                        print("Please enter a valid integer.")

        
            # Load into Pandas DataFrame
            elec_ref_table = pd.read_excel(excel_file)
            print(elec_ref_table.head())  # Preview the data
            #ADD ERROR HANDLING HERE
            #print("No correspondence Excel file found in", elecReconDir)
#%%
            # NWB read
            io = NWBHDF5IO(nwb_fname, mode='r', load_namespaces=True)

            nwb = io.read()
            # Log NWB file loading
         #   logger.log_nwb_load(nwb_fname)
            
            # Get info on data in NWB file
            nwbInfo = inspectNwb(nwb)
            tsInfo = nwbInfo['timeseries']
            elecTable = nwbInfo['elecs']
            
            # Get ieeg data
                
            if 'ieeg' in tsInfo['name'].to_list():
                ecogContainer = nwb.acquisition.get('ieeg')
                fs = ecogContainer.rate
                ecog = nwb2mne(ecogContainer,preload=False)
                
            
                # Get coordinates of each electrode
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
            
                # Create `montage` data structure as required by MNE1
                montage = mne.channels.make_dig_montage(ch_pos=ch_coords, coord_frame='mri')
                montage.add_estimated_fiducials(pat_fs, fs_dir)
                ecog.set_montage(montage)
                
                #mont = mne.channels.make_dig_montage( ch_pos={ch: np.array([np.nan, np.nan, np.nan]) for ch in raw_ekg.ch_names}, coord_frame="mri", )
               # raw_ekg = nwb2mne(nwb.acquisition["ekg"], preload=False, create_montage=False)
               # raw_ekg.set_channel_types({ch: "ecg" for ch in raw_ekg.ch_names})
                
                # ecog.add_channels([raw_ekg], force_update_info=True)
                
                # single montage call, but allow missing (EKG)
               # ecog.set_montage(montage, on_missing="ignore")
                
                # Validate MNE structure after montage setup
                validate_mne_structure(ecog, "After Montage Setup")
                
                # Load audio
                audioContainer = nwb.acquisition.get('audio')
                fs_audio = audioContainer.rate
                audio = audioContainer.data[:]
                t_audio = np.arange(0, audio.shape[0]) / fs_audio
                
            #except:
                
                #broken_nwb.append(nwb_fname)
           # continue
            
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
           #%
            # Cut and save the audio data        
            if 'Fix' in mov:
                idx_movie = np.logical_and(t_audio >= ttls[0], t_audio < ttls[1])
                
            else:
                
                # Frame time
                frame_time = nwb.trials['start_time'].data[:]
                frame_time_end = nwb.trials['stop_time'].data[:]
                  
                idx_movie = np.logical_and(t_audio >= frame_time[0], t_audio < frame_time[-1])
                
 #         
            # Preprocess the EEG data
            # Create preprocessing directory using standardized paths
            if not os.path.exists(file_paths['sub_prep_dir']):
                os.makedirs(file_paths['sub_prep_dir'])
            
            sub_prep_dir = file_paths['sub_prep_dir']

            # ecog.plot(
            #     scalings=dict(seeg=200e-6),  
            #     n_channels=32,               
            #     remove_dc=False,              # remove mean per channel
            #     show_scrollbars=True,        # allow vertical scrolling
            #     duration=12.0,
            #     block = True# time window in seconds
            # )
            
            # raw_ekg.plot(
            #     scalings=dict(seeg=200e-6),  
            #     n_channels=32,               
            #     remove_dc=False,              # remove mean per channel
            #     show_scrollbars=True,        # allow vertical scrolling
            #     duration=12.0,
            #     block = True# time window in seconds
            # )

            
            #%% #Notch filter, bandpass filter, then downsample

            print('--->Applying notch filters, bandpass filter, and downsampling to %2.fHz' % resample_fs)
            
            # Log filtering section
           # logger.log_section("FILTERING STEPS")
            
            # Apply notch filter first to remove line noise
            if pat == "NS135":
                notch_freqs = (60,104,120,180)
           # elif pat == "NS166":
                #notch_freqs = (57,60,67,114,120,134,180)
                #notch_freqs = (57, 60, 67,68, 84, 111,112,120, 128, 135, 167, 180)
            # elif pat == "NS211":
            #     notch_freqs = (57,60,67,114,120,134,180)
            elif pat == "NS174":
                if nwb_fname == '/Volumes/Samsung/Movie_data/movies_nwb_standard/NS174/ses-03/sub-NS174_ses-03_task-despicable_me_english_run-01_ieeg.nwb':
                    notch_freqs = (60,120,143,180)
            else:    
                notch_freqs = (60, 120, 180)
                
            ecog.notch_filter(freqs=notch_freqs, notch_widths=2)
          #  logger.log_filtering("NOTCH", f"Frequencies: {notch_freqs} Hz, Widths: 2 Hz")
            
            #
            # Apply bandpass filter 
            ecog.filter(l_freq=0.1, h_freq=170.0)
           # logger.log_filtering("BANDPASS", f"Low: 0.1 Hz, High: 170.0 Hz")
            
            # Then downsample to target sampling rate
            ecogResampled = ecog.resample(resample_fs)
           # logger.log_filtering("DOWNSAMPLING", f"From: {orig_fs} Hz, To: {resample_fs} Hz")
            
            # Validate structure after resampling
            validate_mne_structure(ecogResampled, "After Resampling")
        
            ecogResampled.crop(tmin=frame_time[0], tmax=frame_time[-1])
            
            # Validate structure after cropping
            validate_mne_structure(ecogResampled, "After Cropping")
            
         #  # Plot'
         
        #    raw_ekg.load_data()
        #    raw_ekg_notched = raw_ekg.copy().notch_filter([60, 120, 180], picks=raw_ekg.ch_names)
            
            # fig = ecogResampled.plot(
            #     scalings=dict(seeg=200e-6),
            #     n_channels=16,
            #     remove_dc=False,
            #     show_scrollbars=False,
            #     decim=5,  # or 10 for very dense data
            #     duration=12.0
            # )
                      
            # fig = raw_ekg_notched.plot( 
            # scalings=dict(seeg=200e-6),
            # n_channels=32,
            # remove_dc=False,
            # show_scrollbars=True,
            # decim=5,  # or 10 for very dense data
            # duration=12.0
            # )

#%% Remove spikes if necessary
            remove_spikes = False
            if remove_spikes:          
                if pat == 'NS178':
                    spike_samples = detect_spikes_ref1(ecogResampled, ref_name='Ref1',
                                                       thresh_z=8.0,
                                                       max_width_ms=20.0,
                                                       min_separation_ms=5.0)
                    
                    ecogResampled = interpolate_spikes(ecogResampled, spike_samples, window_ms=10.0)
                    
                    fig = ecogResampled.plot(
    
                        scalings=dict(seeg=200e-6),
                        n_channels=32,
                        remove_dc=True,
                        show_scrollbars=True,
                        duration=12.0
                    )
                    
                    ecogResampled.notch_filter(freqs=notch_freqs, notch_widths=2)
                    
                    
                else: 
                    spikes_by_chan = detect_spikes_all_channels(ecogResampled,
                                                               picks = None,
                                                               thresh_z=8.0,
                                                               max_width_ms=10.0,
                                                               min_separation_ms=5.0)
                    
                    spikes_by_chan = detect_spikes_all_channels(ecogPreproc,
                                                               picks = None,
                                                               thresh_z=8.0,
                                                               max_width_ms=10.0,
                                                               min_separation_ms=5.0)
                    
                            
                pad_ms = 5.0        # extra padding around spike center on each side
                half_width_ms = 10.0  # half of expected spike width (so total ~20 ms window)
                
                sfreq = ecogPreproc.info['sfreq']
               # sfreq = ecogResampled.info['sfreq']
                pad_samp = int(pad_ms * sfreq / 1000.0)
                half_width_samp = int(half_width_ms * sfreq / 1000.0)
                
                # Work on a copy so we don’t destroy the original
               # raw_clean = ecogResampled.copy()
                raw_clean = ecogPreproc.copy()
                data = raw_clean.get_data()   # shape: (n_channels, n_samples)
                
                n_ch, n_samp = data.shape
                
                for ch_idx, ch_name in enumerate(raw_clean.ch_names):
                    if ch_name not in spikes_by_chan:
                        continue
                
                    spike_samples = spikes_by_chan[ch_name]
                    if spike_samples is None or len(spike_samples) == 0:
                        continue
                
                    print(f"Interpolating {len(spike_samples)} spikes in channel {ch_name}")
                
                    for s in spike_samples:
                        # Define the "spike" segment (centered on s)
                        seg_start = max(0, s - half_width_samp - pad_samp)
                        seg_end   = min(n_samp - 1, s + half_width_samp + pad_samp)
                
                        # Define the interpolation anchors just outside the segment
                        left = seg_start - 1
                        right = seg_end + 1
                
                        # Handle edge cases at the extremes of the recording
                        if left < 0 or right >= n_samp:
                            # If we can’t get both anchors, skip interpolation for this spike
                            continue
                
                        # x positions and y values for interpolation endpoints
                        x = np.array([left, right])
                        y = data[ch_idx, x]
                
                        # Indices to be replaced (the spike segment itself)
                        seg_idx = np.arange(seg_start, seg_end + 1)
                
                        # Linear interpolation across the segment
                        data[ch_idx, seg_idx] = np.interp(seg_idx, x, y)
                
                # Put the modified data back (raw_clean already points to data)
                ecogResampled._data = data
                #ecogPreproc._data = data
                        
                fig = raw_clean.plot(
        
                    scalings=dict(seeg=200e-6),
                    n_channels=32,
                    remove_dc=True,
                    show_scrollbars=True,
                    duration=12.0
                )
                
                notch_freqs = (60, 120, 180)
        
                ecogResampled.notch_filter(freqs=notch_freqs, notch_widths=2)
                
            

#%%         # Display the raw traces and mark bad channels
            print("\n--- Manual Bad Channel Marking ---")
            print("Mark bad channels in the interactive plot, then close the plot to continue.")
            
            fig = ecogResampled.plot(
                scalings=dict(seeg=200e-6),
                n_channels=16,
                remove_dc=True,
                show_scrollbars=False,
                duration=10.0,
                block=True,  # <-- key
            )
            
            # Now you are guaranteed the plot is closed and bads are updated (if your marking took effect).
            summarize_bad_channels(ecogResampled, "After Manual Marking (post-close)")
            save_success = save_bad_channels(
                ecogResampled,
                file_paths['bad_channels_file'],
                pat_fs,
                file_paths['movie_base']
            )
                        
            # Check bad channel integrity after manual marking
            integrity_check = check_bad_channels_integrity(ecogResampled, "After Manual Marking")
            

            
            # Print bad channel status for debugging
            summarize_bad_channels(ecogResampled, "After Manual Marking")
            print(f"Bad channels saved to: {file_paths['bad_channels_file']}")
            
            # Log bad channel selection
            logger.log_section("BAD CHANNEL SELECTION")
            logger.log_bad_channels(ecogResampled.info['bads'])
            logger.log_file_save("BAD_CHANNELS", file_paths['bad_channels_file'])
            
            # Preprocessed file name after downsampling and bad channel removal
            preprocessed_filename = file_paths['preprocessed_file']
            
            # Save the current state of the data in the MNE format                                
            ecogPreproc = ecogResampled

            # Save preprocessed data after downsampling and bad channel removal
            ecogPreproc.save(preprocessed_filename, fmt='single', overwrite=True)
            print(f"Preprocessed data saved to: {preprocessed_filename}")
            logger.log_fif_save(preprocessed_filename, file_description="Preprocessed FIF")
            
            # Final integrity check before saving
            final_integrity = check_bad_channels_integrity(ecogPreproc, "Final Preprocessed Data")
            
            # Final MNE structure validation
            final_structure = validate_mne_structure(ecogPreproc, "Final Preprocessed Data")


#%% spectrogram of particular channels
            plot_spect = False
            if plot_spect:
                 peak_chs  = ["RFi10","RFi9","RFi8"]#"LFx6","LFx10","LFx11","LFx12"]#["LPs6","LPs7","LPs8","LPs9", "LPs10"]      # replace with your channels that show the 105 Hz spike
                 clean_chs = []#["RFa6", "RFm10"]      # replace with channels that look clean
                 chs_to_plot = peak_chs + clean_chs
                   
                # --- parameters ---
                fmin, fmax = 50, 130
                freqs = np.arange(fmin, fmax + 1, 1)   # 1 Hz steps
                n_cycles = freqs / 2.0                 # ~2 cycles at 80 Hz, ~65 ms windows; adjust if you want
                t_start, t_end = 0, 120                # seconds to visualize (change as needed)
                
                # --- crop for speed (optional but recommended) ---
                raw = ecogResampled.copy().crop(tmin=t_start, tmax=t_end)
                
                # --- pick channels ---
                raw_pick = raw.copy().pick_channels(chs_to_plot)
                
                # --- create fixed-length epochs (recommended for TFR speed) ---
                # 2-second epochs with 1-second overlap gives decent time resolution
                epochs = mne.make_fixed_length_epochs(
                    raw_pick, duration=2.0, overlap=1.0, preload=True
                )
                
                # --- compute TFR ---
                tfr = mne.time_frequency.tfr_morlet(
                    epochs,
                    freqs=freqs,
                    n_cycles=n_cycles,
                    use_fft=True,
                    return_itc=False,
                    average=True,     # average across epochs -> cleaner view of persistent line noise
                    decim=1,
                    n_jobs=-1
                )
                
                # Optional: if you want relative change instead of absolute power
                # tfr.apply_baseline(baseline=(None, None), mode="logratio")  # use cautiously for artifact inspection
                
                # --- plot: one figure per channel ---
                for ch in chs_to_plot:
                    tfr_ch = tfr.copy().pick(ch)
                    fig = tfr_ch.plot(
                        picks=0,
                        baseline=None,
                        mode=None,
                        title=f"TFR (Morlet) {ch}: {fmin}-{fmax} Hz",
                        show=False
                    )
                    plt.show()
                
                
                #peak_chs  = ["RTx9","RFx10","LDa12"]#"LFx6","LFx10","LFx11","LFx12"]#["LPs6","LPs7","LPs8","LPs9", "LPs10"]      # replace with your channels that show the 105 Hz spike
                peak_chs = ["EKG1",'EKG2','EKG3','EKG4']
                clean_chs = []#["LTs10", "LTs2"]   # replace
                chs = peak_chs + clean_chs
                
                plot_stft_spectrogram_raw(
                    ecog,
                    chs,
                    fmin=40, fmax=170,
                   # tmin=0, tmax=598,     # e.g., first 10 minutes; set None/None for full recording
                    win_sec=1.0,
                    overlap=0.75
                )
    
#%% reload if necessary
            reload = True
            
            if reload:
                
                preprocessed_filename = file_paths['preprocessed_file']
                fif_file = file_paths['preprocessed_file']
                
                if not fif_file or not os.path.exists(fif_file):
                    print("No new preprocessed .fif file found.")
                
                    fif_files = [
                        f for f in os.listdir(sub_prep_dir)
                        if f.endswith(".fif")
                        #and ref in f
                        and ('prep_ieeg.fif' in f or 'preprocessed' in f)
                        and 'aic' not in f
                        and any(k.lower() in f.lower() for k in vid_keys)
                        and not f.startswith("._")
                    ]
                    
                    print(f"Found {len(fif_files)} matching runs for {pat}:")
                    for f in fif_files:
                        print("  ", f)  
                    
                    while True:
                        try:
                            idx = int(input("Select the correct file by index: "))
                            if 0 <= idx < len(fif_files):
                                fif_file = os.path.join(sub_prep_dir, fif_files[idx])
                                print(f"Selected preprocessed file: {fif_file}")
                                break
                            else:
                                print("Index out of range. Try again.")
                        except ValueError:
                            print("Please enter a valid integer.")
                    
                print(f"reloading cut and filtered data:{preprocessed_filename}")
                
                ecogPreproc = mne.io.read_raw_fif(fif_file, preload=True)
                
                # Print basic info summary
                print(ecogPreproc)
                
                # Print channel names
                print("\nChannel names:")
                print(ecogPreproc.ch_names)
                
                # Plot to visually inspect data traces
                ecogPreproc.plot(
                    scalings=dict(seeg=200e-6),
                    n_channels=32,
                    remove_dc=True,
                    decim=10,
                    show_scrollbars=False,   # test
                    duration=10.0,           # test
                )

            ecogPreproc.save(preprocessed_filename, fmt='single', overwrite=True)

#%%  # Visualize PSD 
            data = ecogPreproc.get_data()  # Shape: (n_channels, n_samples)

            fs_data = resample_fs
            
            # Get good channel indices
            labels = list(ecogPreproc.ch_names)
            good_ch_indices = [i for i, ch in enumerate(labels) if ch not in ecogPreproc.info['bads']]
            
            # Filter the data to include only good channels
            data_good = data[good_ch_indices, :]
            labels_good = [labels[i] for i in good_ch_indices]
            
            # Inspect power spectra at each frequency band
            for freq_band in freq_bands:
                if freq_band == 'low':
                    freq_range = (0.5, 7)
                elif freq_band == 'middle':
                    freq_range = (8, 50)
                elif freq_band == 'high':
                    freq_range = (50, 170)
                elif freq_band == "all":
                    freq_range = (0,170)
                
                # Bandpass filter only the good channels
                sos = signal.butter(5, list(freq_range), btype='bandpass', output='sos', fs=fs_data)
                band_dat = signal.sosfiltfilt(sos, data_good, axis=1)
            
                # Plot power spectra excluding bad channels
                plot_power_spectra(
                    band_dat, labels_good, fs_data, freq_range, pat, 
                    plot_title=f"Power Spectral Density (Welch) for {freq_band}" #for {pat}
                )
                  

#%%  #      #Visualize PSD in smaller batches for readability
                
            print("\n" + "="*60)
            print("NEW INTERACTIVE PSD VISUALIZATION")
            print("="*60)
            
            #Subset visualization for manageable overview
            print("\n--- Option 1: PSD Subset Visualization ---")
            print("Creating manageable subset plots (20 channels)...")
            
            # Create batched PSD plots using the original plotting style (32 channels = 2 electrodes)
            batched_figures = plot_psd_with_scales(
                ecogPreproc, 
                scale_type='log_y',  # Log x-axis, linear y-axis
                batch_size=32,
                freq_bands=['high'],
                patient_id=pat)
            
#%%    

            fs_data = resample_fs
    
            if 'wm_bip' in ref_types:
                
                elecs_subs = elec_ref_table  # or corr_aligned if that's the aligned one you want
                
                # -----------------------------
                # 1) Identify PTD column once
                # -----------------------------
                ptd_col = None
                for col in elecs_subs.columns:
                    if col.lower() == "ptd" or "ptd" in col.lower():
                        ptd_col = col
                        break
                if ptd_col is None:
                    raise ValueError("No PTD column found in elec_ref_table/corr_aligned (expected something containing 'ptd').")
                
                # -----------------------------
                # Choose WHICH table is the source of truth for labels/coords/ptd
                # -----------------------------
                src = elecs_subs  # <-- recommended if this is your "aligned to raw" table
                
                # sanity: must have Label + lepto coords + ptd col
                required = {"Label", "lepto_coords_1", "lepto_coords_2", "lepto_coords_3"}
                missing = required - set(src.columns)
                if missing:
                    raise ValueError(f"corr_aligned missing columns: {missing}")
                
                # If corr_aligned PTD column name differs, use that
                if "ptd" in src.columns:
                    ptd_col_src = "ptd"
                else:
                    # fall back to the ptd_col we found in elecs_subs
                    if ptd_col not in src.columns:
                        raise ValueError(f"PTD column '{ptd_col}' not found in corr_aligned.")
                    ptd_col_src = ptd_col
                
                # -----------------------------
                # Reorder to raw channel order + drop bads
                # -----------------------------
                raw_labels = [ch for ch in ecogPreproc.ch_names if ch not in ecogPreproc.info.get("bads", [])]
                
                src_idx = src.set_index("Label", drop=False)
                # keep only channels present in both
                keep = [ch for ch in raw_labels if ch in src_idx.index]
                
                if len(keep) == 0:
                    raise ValueError("No overlap between ecogPreproc.ch_names (minus bads) and corr_aligned['Label'].")
                
                src_ordered = src_idx.loc[keep].reset_index(drop=True)
                
                labels = src_ordered["Label"].astype(str).tolist()
                
                coords = np.c_[
                    src_ordered["lepto_coords_1"].to_numpy(float),
                    src_ordered["lepto_coords_2"].to_numpy(float),
                    src_ordered["lepto_coords_3"].to_numpy(float),
                ]
                
                ptd = src_ordered[ptd_col_src].to_numpy(float)
                
                # -----------------------------
                # 4) (Optional) Diagnostic WM contacts from atlas (NOT used for ptd/coords vectors)
                # -----------------------------
                wm_group = ['Right-Cerebral-White-Matter', 'Left-Cerebral-White-Matter']
                wm_contacts = []
                
                if 'FS_vol' in elecs_subs.columns:
                    wm_contacts = elecs_subs.loc[elecs_subs['FS_vol'].isin(wm_group), 'Label'].astype(str).tolist()
                elif 'AparcAseg_Atlas' in elecs_subs.columns:
                    # watch out: you used 'Contact' here earlier; be consistent with your naming
                    wm_contacts = elecs_subs.loc[elecs_subs['AparcAseg_Atlas'].isin(wm_group), 'Label'].astype(str).tolist()
                elif 'aparc_aseg' in elecs_subs.columns:
                    wm_contacts = elecs_subs.loc[elecs_subs['aparc_aseg'].isin(wm_group), 'Label'].astype(str).tolist()
                else:
                    print("⚠️ No appropriate atlas column found for white matter extraction.")
                
                wm_contacts_cleaned = [ch for ch in wm_contacts if ch not in ecogPreproc.info.get("bads", [])]
                print(f"Atlas WM contacts (cleaned): {wm_contacts_cleaned}")
                
                print(f"PTD-based vectors ready: N={len(labels)} (coords={coords.shape}, ptd={ptd.shape})")

#%%


            # Apply referencing:
            logger.log_section("REFERENCING STEPS")
            
            # Create reference-specific bad channel files
            ref_bad_channels_files = {}
            
            for ref in ref_types:
                referenced_filename = file_paths['referenced_files'][ref]
                
                # Create reference-specific bad channels file
                ref_bad_channels_file = file_paths['bad_channels_file'].replace('_bad_channels.txt', f'_bad_channels_{ref}.txt')
                ref_bad_channels_files[ref] = ref_bad_channels_file
                
                # Copy original bad channels to reference-specific file
                if os.path.exists(file_paths['bad_channels_file']):
                    import shutil
                    shutil.copy2(file_paths['bad_channels_file'], ref_bad_channels_file)
                    print(f"Copied original bad channels to: {ref_bad_channels_file}")
                else:
                    # Create empty bad channels file for this reference
                    with open(ref_bad_channels_file, 'w') as f:
                        f.write(f"# Bad channels for {pat_fs} - {file_paths['movie_base']} - {ref} reference\n")
                        f.write(f"# Total channels: {len(ecogPreproc.ch_names)}\n")
                        f.write("# Bad channels: 0\n")
                        f.write("# Channel names:\n")
                    print(f"Created new bad channels file: {ref_bad_channels_file}")
            
                
                if ref == 'wm_bip':
                    assert len(labels) == len(ptd) == coords.shape[0]
                    assert set(labels).issubset(set(ecogPreproc.ch_names))
                    assert all(ch not in ecogPreproc.info.get("bads", []) for ch in labels)
                    
                    raw_for_reref = ecogPreproc.copy().drop_channels(ecogPreproc.info["bads"])
                    raw_labels = raw_for_reref.ch_names          
                    
                    # Run WM bipolar referencing
                    ecog_reref, refs = reref_white_matter(
                        raw_for_reref,
                        ptd=ptd,
                        coords=coords,
                        ptd_thresh=-0.8,
                        copy=True
                    )
                                        
                    ecog_reref.save(referenced_filename, overwrite=True)
                    print(f"WM-BIP_referenced data saved to: {referenced_filename}")
                                
                elif ref == 'avg':
                    # Apply average referencing
                    if pat in ['NS211','NS144','NS128']:
                        elec_groups = make_groups_from_prefix(ecogPreproc)
                        ecog_reref = reref_avg_by_group(ecogPreproc, elec_groups)
                    else:
                        ecog_reref = reref_avg(ecogPreproc)
                    ecog_reref.save(referenced_filename, overwrite=True)
                    print(f"Average-referenced data saved to: {referenced_filename}")
                    logger.log_referencing("AVERAGE", "Using all good channels")
                    logger.log_file_save("AVG_REFERENCED", referenced_filename)
                    logger.log_fif_save(referenced_filename, file_description="Average Referenced FIF")
        
                elif ref == 'bip':
                    # Apply bipolar referencing (implement custom logic if needed)
                    ecog_reref = reref_bipolar(ecogPreproc)  # Example logic
                    ecog_reref.save(referenced_filename, overwrite=True)
                    print(f"Bipolar-referenced data saved to: {referenced_filename}")
                    logger.log_referencing("BIPOLAR", "Bipolar referencing applied")
                    logger.log_file_save("BIP_REFERENCED", referenced_filename)
                    logger.log_fif_save(referenced_filename, file_description="Bipolar Referenced FIF")
            
            #%% Visual inspection and bad channel marking for each reference type
            print("\n" + "="*60)
            print("REFERENCE-SPECIFIC BAD CHANNEL MARKING")
            print("="*60)
            print("Each reference type may have different bad channels.")
            print("Mark bad channels specific to each reference type.")
            print("="*60)
            
            for ref in ref_types:
                referenced_filename = file_paths['referenced_files'][ref]
                ref_bad_channels_file = ref_bad_channels_files[ref]
                
                print(f"\n--- Processing {ref.upper()} Reference ---")
                print(f"Referenced file: {referenced_filename}")
                print(f"Bad channels file: {ref_bad_channels_file}")
                
                # Load data with reference-specific bad channels
                ecog_reref = load_data_with_bad_channels(
                    referenced_filename,
                    ref_bad_channels_file, 
                    pat_fs, 
                    file_paths['movie_base']
                )
                
                # Display current bad channel status
                total_channels = len(ecog_reref.ch_names)
                current_bad = len(ecog_reref.info['bads'])
                good_channels = total_channels - current_bad
                
                print(f" Current status for {ref} reference:")
                print(f"  Total channels: {total_channels}")
                print(f"  Current bad channels: {current_bad}")
                print(f"  Good channels: {good_channels}")
                print(f"  Good channel percentage: {(good_channels/total_channels)*100:.1f}%")
                
                # Interactive plot for bad channel marking
                print(f"\n Interactive bad channel marking for {ref} reference")
                print("Mark bad channels in the plot, then close to continue...")
                
                fig = ecog_reref.plot(
                    title=f"{pat_fs} - {ref.upper()} Referenced Data (Mark Bad Channels)",
                    scalings=dict(seeg=200e-6),
                    n_channels=32,
                    remove_dc=True,
                    show_scrollbars=True,
                    duration=12.0,
                    show=True,
                    block=True
                )
                
                # Check bad channel integrity after marking
                integrity_check = check_bad_channels_integrity(ecog_reref, f"After {ref} Reference Marking")
                
                # Save reference-specific bad channels
                save_success = save_bad_channels(
                    ecog_reref, 
                    ref_bad_channels_file, 
                    pat_fs, 
                    f"{file_paths['movie_base']}_{ref}"
                )
                
                # Print updated bad channel status
                summarize_bad_channels(ecog_reref, f"After {ref} Reference Marking")
                print(f"Bad channels saved to: {ref_bad_channels_file}")
                
                # Log reference-specific bad channel selection
                logger.log_section(f"{ref.upper()} REFERENCE BAD CHANNELS")
                logger.log_bad_channels(ecog_reref.info['bads'], f"{ref} reference")
                logger.log_file_save(f"{ref.upper()}_BAD_CHANNELS", ref_bad_channels_file)
                #%%
                # Optional: PSD visualization for this reference
                print(f"\n📊 PSD visualization for {ref} reference")
                psd_figures = plot_psd_with_scales(
                    ecog_reref, 
                    scale_type='log_y',
                    batch_size=32,
                    freq_bands=['high'],
                    patient_id=f"{pat_fs}_{ref}"
                )
                
                print(f"✅ Completed bad channel marking for {ref} reference")
                print(f"  - Bad channels: {len(ecog_reref.info['bads'])}")
                print(f"  - Good channels: {len(ecog_reref.ch_names) - len(ecog_reref.info['bads'])}")
                print(f"  - Saved to: {ref_bad_channels_file}")

# Optional            
            #%%
                # Plot PSD for each frequency band
            for ref in ref_types:
                referenced_filename = file_paths['referenced_files'][ref]
                ref_bad_channels_file = ref_bad_channels_files[ref]
                
                print(f"\n--- Processing {ref.upper()} Reference ---")
                print(f"Referenced file: {referenced_filename}")
                print(f"Bad channels file: {ref_bad_channels_file}")
                
                # Load data with reference-specific bad channels
                ecog_reref = load_data_with_bad_channels(
                    referenced_filename,
                    ref_bad_channels_file, 
                    pat_fs, 
                    file_paths['movie_base']
                )
                
                # Display current bad channel status
                total_channels = len(ecog_reref.ch_names)
                current_bad = len(ecog_reref.info['bads'])
                good_channels = total_channels - current_bad
                
                print(f"📊 Current status for {ref} reference:")
                print(f"  Total channels: {total_channels}")
                print(f"  Current bad channels: {current_bad}")
                print(f"  Good channels: {good_channels}")
                print(f"  Good channel percentage: {(good_channels/total_channels)*100:.1f}%")
                
                print("\n" + "="*60)
                print("NEW INTERACTIVE PSD VISUALIZATION")
                print("="*60)
                
                #Subset visualization for manageable overview
                print("\n--- Option 1: PSD Subset Visualization ---")
                print("Creating manageable subset plots (20 channels)...")
                
                # Create batched PSD plots using the original plotting style (32 channels = 2 electrodes)
                batched_figures = plot_psd_with_scales(
                    ecog_reref, 
                    scale_type='log_y',  # Log x-axis, linear y-axis
                    batch_size=16,
                    #freq_bands = ['all'],
                    freq_bands=['high'],
                    patient_id=pat
)
    
        #%%
            hfa_refs = ['avg']
            
            compute_HFA = False
            if compute_HFA:
            
                # Filter for HFA
                # Create HFA directory using standardized paths
                if not os.path.exists(file_paths['sub_hfa_dir']):
                    os.makedirs(file_paths['sub_hfa_dir'])
                     
                for ref in hfa_refs:
                                
                    hfa_fname = file_paths['hfa_files'][ref]
                    referenced_filename = file_paths['referenced_files'][ref]
                    ref_bad_channels_file = ref_bad_channels_files[ref]
                    
                    print(f"\n--- Processing HFA for {ref.upper()} Reference ---")
                    print(f"HFA file: {hfa_fname}")
                    print(f"Referenced file: {referenced_filename}")
                    print(f"Bad channels file: {ref_bad_channels_file}")
                    
                    # Check if reference-specific bad channels file exists
                    if not os.path.exists(ref_bad_channels_file):
                        print(f"⚠️ Warning: Reference-specific bad channels file not found at {ref_bad_channels_file}")
                        print("Loading data without bad channel restoration...")
                        ecog_reref = mne.io.read_raw_fif(referenced_filename, preload=True)
                    else:
                        # Load data with reference-specific bad channels restored
                        ecog_reref = load_data_with_bad_channels(
                            referenced_filename,
                            ref_bad_channels_file, 
                            pat_fs, 
                            f"{file_paths['movie_base']}_{ref}"
                        )
                    
                    if ecog_reref is not None and not os.path.exists(hfa_fname):
    
                        # Compute HFA
                        hfa_mne = filter_hfa_continuous(ecog_reref, hfa_fname, freq_range, freq_space, n_freq_bins,
                                                        convert_db, n_jobs, resample_bha_fs)
                    
                                       
                        # Save the cut HFA data
                        hfa_mne.save(hfa_fname,fmt='single',overwrite=True)
                        logger.log_file_save("HFA", hfa_fname)
                        logger.log_fif_save(hfa_fname, file_description="HFA FIF")
            
            # Finish logging
            logger.finish_log()
            
            if single_file_mode:
                print(f"\n{'='*60}")
                print(f"✅ SINGLE FILE PROCESSING COMPLETE")
                print(f"Patient: {pat_fs}")
                print(f"Movie: {mov}")
                print(f"Log file: {logger.log_file}")
                print(f"{'='*60}\n")
                # Exit after processing single file
                sys.exit(0)
