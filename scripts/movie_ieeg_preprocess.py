#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 12:15:12 2024

@author: christinechesebrough
"""

#%% To-Do
# Need to label spikes and other artifacts in data

import os, sys, re
import numpy as np
import mne
from pynwb import NWBHDF5IO
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import cv2
import scipy.interpolate as interp
import scipy.stats as stats
import scipy.signal as signal
#import temporal_response_function as trf
from mne.time_frequency import psd_array_welch
from mne.filter import filter_data
import sounddevice as sd
from scipy.io.wavfile import write
import importlib.util

# Add current directory to Python path to allow importing local modules
current_dir = '/Volumes/Samsung/scripts/nwb_convert_preprocess'

if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# path to temporal response function
module_path = '/Users/christinechesebrough/Documents/data_standardization-main/src/temporal_response_function.py'

module_name = 'temporal_response_function'

spec = importlib.util.spec_from_file_location(module_name, module_path)
trf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trf)

# Import EEG preprocessing helper functions
from eeg_preproc_helpers import (
    plot_power_spectra, plot_psd_interactive_mne, plot_psd_subset_mne, plot_psd_batched,
    create_file_paths, apply_highpass_filter,
    save_bad_channels, load_bad_channels, check_bad_channels_integrity,
    load_preprocessed_data, validate_mne_structure, preserve_mne_structure, restore_mne_structure, load_data_with_bad_channels, summarize_bad_channels, ProcessingLogger
)

sys.path.insert(0, '/Users/christinechesebrough/Documents/EPIPE-movie_nwb/Python')
sys.path.append('/Users/christinechesebrough/Documents/iEEG2NWB-main')

from epipe import inspectNwb, nwb2mne, read_ielvis, reref_avg, reref_bipolar, filter_hfa_continuous

matplotlib.use('Qt5Agg')

# Parameters
full_task_name = 'movies'
pipeline_name = 'preprocess_movies'
pipeline_version = 'v.1.0.725'

# Single File Processing Mode
# Set to True to process only one specific file
single_file_mode = True

# When single_file_mode is True, specify the file to process
# Format: 'patient_id' or 'patient_id_implant_number'
target_patient = 'NS193'  # e.g., 'NS189' or 'NS189_01'
target_movie = 'NS193_ses-01_task-despicable_me_english_run-01_ieeg.nwb'  # Full movie filename with .nwb extension

# When single_file_mode is False, these parameters control batch processing
patients = ['NS193']  # List of patients to process in batch mode

# Example configurations:
# Single file mode:
#   target_patient = 'NS189'           # Processes sub-NS189/ses-01/ieeg/NS189_ses-01_task-*.nwb
#   target_movie = 'NS189_ses-02_task-despicable_me_english_run-01_ieeg.nwb'  # Session extracted from filename
#
# Batch mode:
#   single_file_mode = False
#   patients = ['NS189', 'NS190']      # Processes all matching files for these patients

resample_fs = 600
notch_freqs = (60, 120, 180)

# Types of references to use in analyses
# Must be a list containing at least one of the options: "avg", "bip"
ref_types = ['wm','avg','bip']

n_jobs = 16

convert_db = True

# Frequency range
freq_range = [70, 170]
n_freq_bins = 10
freq_space = 'log'      # 'log', 'lin'

resample_bha_fs = 100


freq_bands = ['low','middle','high']

#
movie_table = pd.read_excel('/Volumes/Samsung/Movie_data/data/electrode_localization/movie_table.xlsx')


#%% Define directories

#data_dir = '/Volumes/Samsung/Movie_data/movies_nwb_standard'
data_dir = '/Volumes/Samsung/Movie_data/movies_new_nwb'
fs_dir = '/Volumes/Samsung/anatomy'
prep_dir = '/Volumes/Samsung/Movie_data/movies_new_prep'
frame_dir = '/Volumes/Samsung/Movie_data/data/video_frames'
lum_dir = '/Volumes/Samsung/Movie_data/data/luminance'

if not os.path.exists(lum_dir):
    os.makedirs(lum_dir)

compute_luminance = False

et_prep_dir = 'Eye_prep'
audio_dir = 'Audio'
neural_prep_dir = 'Neural_prep'
hfa_dir = 'HFA'

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
    if 'ses-' in target_movie:
        # Extract session number from filename
        ses_match = re.search(r'ses-(\d+)', target_movie)
        if ses_match:
            ses_num = ses_match.group(1)
            imp = f'ses-{ses_num}'
        else:
            print(f"❌ ERROR: Could not extract session number from filename: {target_movie}")
            sys.exit(1)
    else:
        # Fallback to ses-01 if no session info in filename
        imp = 'ses-01'
    
    # Set patient path
    pat = f'{target_patient}'
    
    # Validate that the target file exists
    if data_dir == '/Volumes/Samsung/Movie_data/movies_nwb_standard':
        target_nwb_path = '{:s}/{:s}/{:s}/{:s}'.format(data_dir, pat, imp, target_movie)
    elif data_dir == '/Volumes/Samsung/Movie_data/movies_new_nwb':
        target_nwb_path = '{:s}/{:s}/{:s}/ieeg/{:s}'.format(data_dir, pat, imp, target_movie)
    
    if not os.path.exists(target_nwb_path):
        print(f"❌ ERROR: Target file not found: {target_nwb_path}")
        print("Please check the target_patient and target_movie parameters.")
        print(f"Expected path: {target_nwb_path}")
        sys.exit(1)
    
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
        elif data_dir == '/Volumes/Samsung/Movie_data/movies_new_nwb':
            ieeg_dir = '{:s}/{:s}/{:s}/ieeg/'.format(data_dir, pat, imp)

        if single_file_mode:
            # Single file mode - use predefined movie
            movies = movies_to_process
        else:
            # Batch mode - get all movies
            movies = [f for f in os.listdir(ieeg_dir) if f.endswith('.nwb')]
            # Filter for specific video if vid is defined
            if 'vid' in locals():
                movies = [mov for mov in movies if vid in mov]
        
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
            elif data_dir == '/Volumes/Samsung/Movie_data/movies_new_nwb':
                nwb_fname = '{:s}/{:s}/{:s}/ieeg/{:s}'.format(data_dir, pat, imp, mov)

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
            # NWB read
            io = NWBHDF5IO(nwb_fname, mode='r', load_namespaces=True)
            nwb = io.read()
            
            # Get info on data in NWB file
            nwbInfo = inspectNwb(nwb)
            tsInfo = nwbInfo['timeseries']
            elecTable = nwbInfo['elecs']
            
            # Define the subject-specific directory
            subid = os.path.basename(sub_fs_dir)
            elecReconDir = os.path.join(sub_fs_dir, 'elec_recon')
            
            # List all files in the directory and filter for the one containing 'correspondence'
            excel_files = [f for f in os.listdir(elecReconDir)
                           if 'correspondence' in f and
                           f.endswith('.xlsx') and
                           not f.startswith('.')]
            
            if excel_files:
                excel_file = os.path.join(elecReconDir, excel_files[0])  
                print(f"Found electrode correspondence file: {excel_file}")
            else:
                print("No correspondence Excel file found.")
                
            # Load into Pandas DataFrame
            elec_ref_table = pd.read_excel(excel_file)
            print(elec_ref_table.head())  # Preview the data
            #ADD ERROR HANDLING HERE
            #print("No correspondence Excel file found in", elecReconDir)

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
            
                # Create `montage` data structure as required by MNE
                montage = mne.channels.make_dig_montage(ch_pos=ch_coords, coord_frame='mri')
                montage.add_estimated_fiducials(pat_fs, fs_dir)
                ecog.set_montage(montage)
                
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
           #%% 
            # Cut and save the audio data        
            if 'Fix' in mov:
                idx_movie = np.logical_and(t_audio >= ttls[0], t_audio < ttls[1])
                
            else:
                
                # Frame time
                frame_time = nwb.trials['start_time'].data[:]
                frame_time_end = nwb.trials['stop_time'].data[:]
                  
                idx_movie = np.logical_and(t_audio >= frame_time[0], t_audio < frame_time[-1])
                
 #%%           
            # Preprocess the EEG data
            # Create preprocessing directory using standardized paths
            if not os.path.exists(file_paths['sub_prep_dir']):
                os.makedirs(file_paths['sub_prep_dir'])

            ecog.plot(
                scalings=dict(seeg=200e-6),  
                n_channels=32,               
                remove_dc=True,              # remove mean per channel
                show_scrollbars=True,        # allow vertical scrolling
                duration=12.0                # time window in seconds
            )
            
            
 #%%       THIS IS OPTIONAL BUT SOMETIMES NECESSARY; IT'S APPARENT WHETHER IT IS NECESSARY BASED ON THE NEXT STEP OF NOTCH FILTERING, BANDPASS, AND DOWNSAMPLING
 ## If bad initial reference, option to rereference before notch filtering and downsample
            ref_channels = [ch for ch in ecog.ch_names if ch.lower().startswith('ref')]
            print("Candidate reference channels:", ref_channels)
            
            for ref in ref_channels:
                data = ecog.copy().pick(ref).get_data()
                std = np.std(data)
                print(f"{ref}: std={std}")
                            
            # Ask for user input by channel name
            selected_ref = input("Enter the NAME of the best reference channel to use (case-sensitive): ")
            
            if selected_ref in ref_channels:
                print(f"Selected reference channel: {selected_ref}")
                
                # Apply referencing
                ecog.set_eeg_reference(ref_channels=[selected_ref], projection=False)
                print("Re-referencing applied with selected channel.")
            else:
                print("Invalid channel name. No referencing applied.")
#%% #Notch filter, bandpass filter, then downsample

            print('--->Applying notch filters, bandpass filter, and downsampling to %2.fHz' % resample_fs)
            
            # Log filtering section
            logger.log_section("FILTERING STEPS")
            
            # Apply notch filter first to remove line noise
            ecog.notch_filter(freqs=notch_freqs, notch_widths=2)
            logger.log_filtering("NOTCH", f"Frequencies: {notch_freqs} Hz, Widths: 2 Hz")
            
            # Apply bandpass filter (e.g. 0.1 Hz to 200 Hz)
            ecog.filter(l_freq=0.1, h_freq=200.0)
            logger.log_filtering("BANDPASS", f"Low: 0.1 Hz, High: 200.0 Hz")
            
            # Then downsample to target sampling rate
            ecogResampled = ecog.resample(resample_fs)
            logger.log_filtering("DOWNSAMPLING", f"From: {orig_fs} Hz, To: {resample_fs} Hz")
            
            # Validate structure after resampling
            validate_mne_structure(ecogResampled, "After Resampling")
        
            ecogResampled.crop(tmin=frame_time[0], tmax=frame_time[-1])
            
            # Validate structure after cropping
            validate_mne_structure(ecogResampled, "After Cropping")
            
            # Plot
            fig = ecogResampled.plot(
                scalings=dict(seeg=200e-6),
                n_channels=32,
                remove_dc=True,
                show_scrollbars=True,
                duration=12.0
            )
            
#%%
            ## OPTIONAL: Apply high-pass filter to remove drift
            apply_highpass = False  # Set to True if drift removal is needed
            
            if apply_highpass:
                ecogFiltered = apply_highpass_filter(ecogResampled, resample_fs, l_freq=0.5)
            else:
                ecogFiltered = ecogResampled
            
#%%            
            # Display the raw traces and mark bad channels
            print("\n--- Manual Bad Channel Marking ---")
            print("Mark bad channels in the interactive plot, then close the plot to continue.")
            
            fig = ecogFiltered.plot(
                scalings=dict(seeg=200e-6),  # adjust per your data amplitudes
                n_channels=32,               # number of channels displayed at once
                remove_dc=True,              # remove mean per channel
                show_scrollbars=True,        # allow vertical scrolling
                duration=15.0                # time window in seconds
            )
            
            # Check bad channel integrity after manual marking
            integrity_check = check_bad_channels_integrity(ecogFiltered, "After Manual Marking")
            
            # Save bad channels to file with proper error handling
            save_success = save_bad_channels(
                ecogFiltered, 
                file_paths['bad_channels_file'], 
                pat_fs, 
                file_paths['movie_base']
            )
            
            # Print bad channel status for debugging
            summarize_bad_channels(ecogFiltered, "After Manual Marking")
            print(f"Bad channels saved to: {file_paths['bad_channels_file']}")
            
            # Log bad channel selection
            logger.log_section("BAD CHANNEL SELECTION")
            logger.log_bad_channels(ecogFiltered.info['bads'])
            logger.log_file_save("BAD_CHANNELS", file_paths['bad_channels_file'])
            
            # Preprocessed file name after downsampling and bad channel removal
            preprocessed_filename = file_paths['preprocessed_file']
            
            # Save the current state of the data in the MNE format                                
            ecogPreproc = ecogFiltered

            # Save preprocessed data after downsampling and bad channel removal
            ecogFiltered.save(preprocessed_filename, fmt='single', overwrite=True)
            print(f"Preprocessed data saved to: {preprocessed_filename}")
            
            # Final integrity check before saving
            final_integrity = check_bad_channels_integrity(ecogPreproc, "Final Preprocessed Data")
            
            # Final MNE structure validation
            final_structure = validate_mne_structure(ecogPreproc, "Final Preprocessed Data")


#%%
            # Reload ecogPreproc
            reload = True
            
            if reload:
                
                preprocessed_filename = file_paths['preprocessed_file']
                fif_file = preprocessed_filename
                print(f"reloading cut and filtered data:{preprocessed_filename}")
                
                ecogPreproc = mne.io.read_raw_fif(fif_file, preload=True)
                
                # Print basic info summary
                print(ecogPreproc)
                
                # Print channel names
                print("\nChannel names:")
                print(ecogPreproc.ch_names)
                
                # Plot to visually inspect data traces
                ecogPreproc.plot(
                    scalings=dict(seeg=200e-6),  # Adjust scaling for your typical amplitudes
                    n_channels=32,
                    remove_dc=True,
                    show_scrollbars=True,
                    duration=15.0
                )


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
                    freq_range = (8, 30)
                elif freq_band == 'high':
                    freq_range = (30, 170)
                
                # Bandpass filter only the good channels
                sos = signal.butter(5, list(freq_range), btype='bandpass', output='sos', fs=fs_data)
                band_dat = signal.sosfiltfilt(sos, data_good, axis=1)
            
                # Plot power spectra excluding bad channels
                plot_power_spectra(
                    band_dat, labels_good, fs_data, freq_range, pat, 
                    plot_title=f"Power Spectral Density (Welch) for {freq_band} for {pat}"
                )
                  
            #%% OPTIONAL
                            
                # Compute PSD using Welch's method
                psds, freqs = psd_array_welch(
                    data_good, sfreq=fs_data, fmin=0.5, fmax=170, 
                    n_fft=int(fs_data * 2), n_overlap=int(fs_data), average='mean'
                )
                
                # Convert PSD to dB scale
                psds_db = 10 * np.log10(psds)
                
                # Compute median and standard deviation of PSD across channels
                median_psd = np.median(psds_db, axis=0)
                z_scores = (psds_db - median_psd) / np.std(psds_db, axis=0)
                
                # Identify bad channels based on z-score threshold
                bad_threshold = 5  # Adjust threshold as needed
                bad_channel_indices = np.where(np.abs(z_scores).max(axis=1) > bad_threshold)[0]
                bad_channels = [labels_good[i] for i in bad_channel_indices]
                
                print(f"Identified {len(bad_channels)} bad channels: {bad_channels}")

#%%  #      #Visualize PSD in smaller batches for readability

            print("\n" + "="*60)
            print("NEW INTERACTIVE PSD VISUALIZATION")
            print("="*60)
            
            #Subset visualization for manageable overview
            print("\n--- Option 1: PSD Subset Visualization ---")
            print("Creating manageable subset plots (20 channels)...")
            
            # Create batched PSD plots using the original plotting style (32 channels = 2 electrodes)
            batched_figures = plot_psd_batched(
                ecogPreproc, 
                batch_size=32,
                freq_bands=['low', 'middle', 'high'],
                patient_id=pat
            )
    

            #%%   # Find white matter electrodes to use as reference montage
            
            # Check if a white matter contact file exists for this patient and movie
            wm_contact_file = file_paths['wm_contacts_file']
            
            elecs_subs = elec_ref_table
                        
            if os.path.exists(wm_contact_file):
                # Load the previously saved WM contacts
                with open(wm_contact_file, 'r') as f:
                    wm_references = [line.strip() for line in f.readlines()]
                print(f"Loaded white matter references from file: {wm_references}")
            
            else:
                # Use the electrode reference table to find white matter electrodes to use as reference montage
            
                # Check if a white matter contact file exists for this patient and movie
                wm_contact_file = file_paths['wm_contacts_file']
                elecs_subs = elec_ref_table
                

                # Define white matter group labels
                wm_group = ['Right-Cerebral-White-Matter', 'Left-Cerebral-White-Matter']
            
                # Determine WM contacts based on available atlas columns
                if 'FS_vol' in elecs_subs.columns:
                    wm_mask = elecs_subs['FS_vol'].isin(wm_group)
                    wm_contacts = elecs_subs.loc[wm_mask, 'Label'].values
                elif 'AparcAseg_Atlas' in elecs_subs.columns:
                    wm_mask = elecs_subs['AparcAseg_Atlas'].isin(wm_group)
                    wm_contacts = elecs_subs.loc[wm_mask, 'Contact'].values
                elif 'aparc_aseg' in elecs_subs.columns:
                    wm_mask = elecs_subs['aparc_aseg'].isin(wm_group)
                    wm_contacts = elecs_subs.loc[wm_mask, 'Label'].values
                else:
                    wm_contacts = []
                    print("⚠️ No appropriate atlas column found for white matter extraction.")
            
                print(f"Identified white matter contacts: {wm_contacts}")
            
                # Exclude bad contacts
                wm_contacts_cleaned = [ch for ch in wm_contacts if ch not in ecogPreproc.info['bads']]
                print(f"White matter contacts (cleaned): {wm_contacts_cleaned}")
            
                # Extract data for cleaned WM contacts
                wm_data = ecogPreproc.get_data(picks=wm_contacts_cleaned)
            
                # Compute PSD
                wm_psd, freqs = psd_array_welch(
                    wm_data, sfreq=fs_data, fmin=1, fmax=150, n_fft=int(fs_data * 2)
                )
            
                # Calculate mean PSD and variance per contact
                mean_psd = wm_psd.mean(axis=1)
                variance = wm_data.var(axis=1)
            
                # Extract PTD_index values and DK Atlas labels for each contact
                ptd_values, dk_labels = [], []
                for ch in wm_contacts_cleaned:
                    ptd_row = elecs_subs.loc[elecs_subs['Label'] == ch, 'PTD_index']
                    ptd_values.append(ptd_row.values[0] if not ptd_row.empty else np.nan)
                
                    if 'Desikan_Killiany' in elecs_subs.columns:
                        atlas_label = elecs_subs.loc[elecs_subs['Label'] == ch, 'Desikan_Killiany']
                    elif 'DK_Atlas' in elecs_subs.columns:
                        atlas_label = elecs_subs.loc[elecs_subs['Label'] == ch, 'DK_Atlas']
                    else:
                        atlas_label = pd.Series(['Unknown'])
                
                    dk_labels.append(atlas_label.values[0] if not atlas_label.empty else 'Unknown')
                
                ptd_values = np.array(ptd_values)


                # Extract PTD_index values and DK Atlas labels for each contact
                ptd_values= []
                
                for col in elecs_subs.columns:
                    if 'ptd' in col.lower():
                        ptd_col = col
                        break
                
                if ptd_col is None:
                    print("⚠️ No PTD column found.")
                    ptd_values = [np.nan] * len(wm_contacts_cleaned)
                else:
                    # Extract PTD_index values using the identified column
                    ptd_values = []
                    for ch in wm_contacts_cleaned:
                        ptd_row = elecs_subs.loc[elecs_subs['Label'] == ch, ptd_col]
                        ptd_values.append(ptd_row.values[0] if not ptd_row.empty else np.nan)
                        
                dk_labels = []
                for ch in wm_contacts_cleaned:
                    if 'Desikan_Killiany' in elecs_subs.columns:
                        atlas_label = elecs_subs.loc[elecs_subs['Label'] == ch, 'Desikan_Killiany']
                    elif 'DK_Atlas' in elecs_subs.columns:
                        atlas_label = elecs_subs.loc[elecs_subs['Label'] == ch, 'DK_Atlas']
                    else:
                        atlas_label = pd.Series(['Unknown'])
            
                    dk_labels.append(atlas_label.values[0] if not atlas_label.empty else 'Unknown')
            
            
                # Construct DataFrame
                wm_df = pd.DataFrame({
                    'Contact': wm_contacts_cleaned,
                    'Mean_PSD': mean_psd,
                    'Variance': variance,
                    'PTD_index': ptd_values,
                    'DK_Atlas_Label': dk_labels
                })
                
                # Define output CSV filename in the same directory as wm_contact_file
                wm_df_csv_file = wm_contact_file.replace('_wm_contacts.txt', '_wm_contacts_metrics.csv')
                
                # Save wm_df to CSV
                wm_df.to_csv(wm_df_csv_file, index=False)
                print(f"WM contact metrics saved to: {wm_df_csv_file}")

 
                # Top 10 by lowest variance
                wm_df_sorted_var = wm_df.sort_values(by='Variance')
                print("\n📊 Top 10 contacts by lowest variance:")
                for i, (_, row) in enumerate(wm_df_sorted_var.head(10).iterrows()):
                    print(f"  {i+1:2d}. {row['Contact']:8s} - PSD: {row['Mean_PSD']:8.2f}, Var: {row['Variance']:8.2e}, PTD: {row['PTD_index']:6.3f}")
                
                # Top 10 by PTD closest to -1
                wm_df_sorted_ptd = wm_df.sort_values(by='PTD_index')
                print("\n📊 Top 10 contacts by PTD closest to -1:")
                for i, (_, row) in enumerate(wm_df_sorted_ptd.head(10).iterrows()):
                    print(f"  {i+1:2d}. {row['Contact']:8s} - PSD: {row['Mean_PSD']:8.2f}, Var: {row['Variance']:8.2e}, PTD: {row['PTD_index']:6.3f}")
                
                # Create composite score for best contacts
                # Normalize each metric to 0-1 scale (lower is better)
                wm_df['PSD_rank'] = wm_df['Mean_PSD'].rank()
                wm_df['Var_rank'] = wm_df['Variance'].rank()
                wm_df['PTD_rank'] = wm_df['PTD_index'].rank()
                
                # Composite score (lower is better)
                wm_df['Composite_score'] = (wm_df['PSD_rank'] + wm_df['Var_rank'] + wm_df['PTD_rank']) / 3
                
                # Top 15 by composite score
                wm_df_sorted_composite = wm_df.sort_values(by='Composite_score')
                print("\n" + "="*60)
                print("LOW VARIANCE /PSD WM CONTACTS (COMPOSITE SCORE)")
                print("="*60)
                print("Rank | Contact | PSD Rank | Var Rank | PTD Rank | Composite | PSD    | Variance | PTD")
                print("-" * 80)
                
                for i, (_, row) in enumerate(wm_df_sorted_composite.head(15).iterrows()):
                    print(f" {i+1:2d}  | {row['Contact']:7s} | {row['PSD_rank']:8.1f} | {row['Var_rank']:8.1f} | {row['PTD_rank']:8.1f} | {row['Composite_score']:9.1f} | {row['Mean_PSD']:6.2f} | {row['Variance']:8.2e} | {row['PTD_index']:5.3f}")
                
                print("\n")
                print("   These contacts have the best combination of low PSD, low variance, and PTD close to -1.")

                # Plot WM contacts for visual inspection BEFORE user prompt
                print("\n📈 Plotting WM contacts for visual inspection...")
                fig = ecogPreproc.plot(
                    scalings=dict(seeg=200e-6),
                    n_channels=32,
                    remove_dc=True,
                    show_scrollbars=True,
                    duration=12.0
                )
                
                # Wait for user to close the plot
                plt.show(block=True)
                
                #%%
            
                # Ask for user input to define good WM contacts by label
                print("\n Select a subset of wm contacts from different electrodes/spatial locations")

                selected_labels = input(
                    "\nEnter the LABELS of good WM contacts (comma-separated, e.g., RIp12,LFp4,LDh8): "
                )
                selected_wm_references = [label.strip() for label in selected_labels.split(',')]
            
                # Verify entered labels are valid
                valid_selected_wm_references = [ch for ch in selected_wm_references if ch in wm_df['Contact'].values]
                invalid_labels = [ch for ch in selected_wm_references if ch not in wm_df['Contact'].values]
                if invalid_labels:
                    print(f"⚠️ These entered labels were not found and will be skipped: {invalid_labels}")
            
                print(f"Final selected WM references: {valid_selected_wm_references}")
            
                # Save to file
                with open(wm_contact_file, 'w') as f:
                    f.writelines("\n".join(valid_selected_wm_references))
                print(f"Selected WM references saved to: {wm_contact_file}")

#%%
            # Use the selected WM contacts for referencing
            wm_references = selected_wm_references
            print(f"Selected white matter references: {wm_references}")

        
        # Apply referencing:
            logger.log_section("REFERENCING STEPS")
            
            for ref in ref_types:
                referenced_filename = file_paths['referenced_files'][ref]
            
                if ref == 'wm':
                    # Apply WM referencing
                    ecog_reref = ecogPreproc.set_eeg_reference(ref_channels=wm_references)
                    ecog_reref.save(referenced_filename, overwrite=True)
                    print(f"WM-referenced data saved to: {referenced_filename}")
                    logger.log_referencing("WM", f"Reference channels: {wm_references}")
                    logger.log_file_save("WM_REFERENCED", referenced_filename)
            
                elif ref == 'avg':
                    # Apply average referencing
                    ecog_reref = reref_avg(ecogPreproc)
                    ecog_reref.save(referenced_filename, overwrite=True)
                    print(f"Average-referenced data saved to: {referenced_filename}")
                    logger.log_referencing("AVERAGE", "Using all good channels")
                    logger.log_file_save("AVG_REFERENCED", referenced_filename)
        
                elif ref == 'bip':
                    # Apply bipolar referencing (implement custom logic if needed)
                    ecog_reref = reref_bipolar(ecogPreproc)  # Example logic
                    ecog_reref.save(referenced_filename, overwrite=True)
                    print(f"Bipolar-referenced data saved to: {referenced_filename}")
                    logger.log_referencing("BIPOLAR", "Bipolar referencing applied")
                    logger.log_file_save("BIP_REFERENCED", referenced_filename)
            
            # Visualize trace and PSD again after rereferencing
            for ref in ref_types:
                referenced_filename = file_paths['referenced_files'][ref]
                print(referenced_filename)
                # Load data with bad channels restored
                ecog_reref = load_data_with_bad_channels(
                referenced_filename,
                file_paths['bad_channels_file'], 
                pat_fs, 
                file_paths['movie_base'])
                ecog_reref.plot(title=f"{pat} - {ref} Referenced Data", show=True, block=True, duration=12.0, n_channels=32)

                # Check if bad channels file exists
               # if not os.path.exists(file_paths['bad_channels_file']):
               ##     print(f"⚠️ Warning: Bad channels file not found at {file_paths['bad_channels_file']}")
                #    print("Loading data without bad channel restoration...")
                #    ecog_reref = mne.io.read_raw_fif(referenced_filename, preload=True)
               # else:

#%% Optional            
            
                # Plot PSD for each frequency band
                for freq_band in freq_bands: 
                    
                    if freq_band == 'low':
                        freq_range = (.5, 7)
                    if freq_band == 'middle':
                        freq_range = (8, 30)
                    if freq_band == 'high':
                        freq_range = (30, 170)
         
                    sos = signal.butter(5, list(freq_range), btype='bandpass', output='sos', fs=fs_data)
                    band_dat = signal.sosfiltfilt(sos, ecog_reref.get_data(), axis=1)
                    
                    plot_power_spectra(
                        band_dat, ecog_reref.ch_names, fs_data, freq_range, pat, 
                        plot_title=f"Power Spectral Density (Welch) for {freq_band} for {pat} - {ref} referenced"
                    )

    
        #%%
            hfa_refs = ['avg','wm']
            
            # Filter for HFA
            # Create HFA directory using standardized paths
            if not os.path.exists(file_paths['sub_hfa_dir']):
                os.makedirs(file_paths['sub_hfa_dir'])
                 
            for ref in hfa_refs:
                            
                hfa_fname = file_paths['hfa_files'][ref]

                referenced_filename = file_paths['referenced_files'][ref]
                
                # Check if bad channels file exists
                if not os.path.exists(file_paths['bad_channels_file']):
                    print(f"⚠️ Warning: Bad channels file not found at {file_paths['bad_channels_file']}")
                    print("Loading data without bad channel restoration...")
                    ecog_reref = mne.io.read_raw_fif(referenced_filename, preload=True)
                else:
                    # Load data with bad channels restored
                    ecog_reref = load_data_with_bad_channels(
                        referenced_filename,
                        file_paths['bad_channels_file'], 
                        pat_fs, 
                        file_paths['movie_base']
                    )
                
                if ecog_reref is not None and not os.path.exists(hfa_fname):

                    # Compute HFA
                    hfa_mne = filter_hfa_continuous(ecog_reref, hfa_fname, freq_range, freq_space, n_freq_bins,
                                                    convert_db, n_jobs, resample_bha_fs)
                
                                   
                    # Save the cut HFA data
                    hfa_mne.save(hfa_fname,fmt='single',overwrite=True)
                    logger.log_file_save("HFA", hfa_fname)
            
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
