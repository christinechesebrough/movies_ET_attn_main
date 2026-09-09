#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 07:30:31 2025

@author: christinechesebrough
"""

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
#import cv2
import scipy.interpolate as interp
import scipy.stats as stats
import scipy.signal as signal
#import temporal_response_function as trf
from mne.time_frequency import psd_array_welch
from mne.filter import filter_data
#import sounddevice as sd
from scipy.io.wavfile import write
import importlib.util

# Add the src directory to the path for importing helper functions
src_dir = os.path.join(os.path.dirname(__file__), '..', 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# path to temporal response function
module_path = '/Users/christinechesebrough/Documents/data_standardization-main/src/temporal_response_function.py'

module_name = 'temporal_response_function'

spec = importlib.util.spec_from_file_location(module_name, module_path)
trf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trf)

# Import EEG preprocessing helper functions
from eeg_preproc_helpers import (
    plot_power_spectra,  plot_psd_batched, plot_psd_with_scales,
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
single_file_mode = False

#%%

# When single_file_mode is True, specify the file to process
# Format: 'patient_id' or 'patient_id_implant_number'
target_patient = 'NS205'  # e.g., 'NS189' or 'NS189_01'
target_movie = 'NS194_ses-01_task-despicable_me_english_run-01_ieeg.nwb'  # Full movie filename with .nwb extension

# When single_file_mode is False, these parameters control batch processing
#patients = ['NS155','NS178','NS190','NS191', 'NS192','NS194', 'NS201', 'NS204', 'NS205']  # List of patients to process in batch mode
patients = ['NS155','NS178'] 

#patients = ['NS190']

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
ref_types = ['avg']

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
#data_dir = '/Volumes/Samsung/movie_data_new/movies_raw_fall25'

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

vid = 'inscapes'

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
    print(f"Video filter: {vid}")
    print(f"Reference types: {ref_types}")
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
            # Check for final .fif files for each reference type
            print(f"\n{'='*60}")
            print(f"CHECKING FINAL .FIF FILES")
            print(f"{'='*60}")
            
            for ref in ref_types:
                print(f"\n--- Checking {ref.upper()} reference ---")
                
                # Get the referenced file path
                referenced_filename = file_paths['referenced_files'][ref]
                
                # Get the reference-specific bad channels file
                ref_bad_channels_file = file_paths['bad_channels_file'].replace('_bad_channels.txt', f'_bad_channels_{ref}.txt')
                
                print(f"Referenced file: {referenced_filename}")
                print(f"Bad channels file: {ref_bad_channels_file}")
                
                # Check if files exist
                if os.path.exists(referenced_filename):
                    print(f"✅ Found referenced file: {referenced_filename}")
                    
                    # Load data with reference-specific bad channels
                    try:
                        ecog_reref = load_data_with_bad_channels(
                            referenced_filename,
                            ref_bad_channels_file, 
                            pat_fs, 
                            f"{file_paths['movie_base']}_{ref}"
                        )
                        
                        if ecog_reref is not None:
                            print(f"✅ Successfully loaded {ref} referenced data")
                            
                            # Display current bad channel status
                            total_channels = len(ecog_reref.ch_names)
                            current_bad = len(ecog_reref.info['bads'])
                            good_channels = total_channels - current_bad
                            
                            print(f"📊 Data summary for {ref} reference:")
                            print(f"  Total channels: {total_channels}")
                            print(f"  Current bad channels: {current_bad}")
                            print(f"  Good channels: {good_channels}")
                            print(f"  Good channel percentage: {(good_channels/total_channels)*100:.1f}%")
                            print(f"  Sampling rate: {ecog_reref.info['sfreq']:.0f} Hz")
                            print(f"  Duration: {ecog_reref.times[-1]:.1f} seconds")
                            
                            # Log data summary
                            logger.log_decision(f"{ref.upper()}_SUMMARY", 
                                f"Channels: {total_channels}, Good: {good_channels}, Bad: {current_bad}, "
                                f"SR: {ecog_reref.info['sfreq']:.0f}Hz, Duration: {ecog_reref.times[-1]:.1f}s")
                            
                            #%% Interactive Plotting for Manual Review
                            
                            print(f"\n{'='*60}")
                            print(f"INTERACTIVE PLOTTING - {ref.upper()} REFERENCE")
                            print(f"{'='*60}")
                            print(f"Patient: {pat_fs}")
                            print(f"Movie: {mov}")
                            print(f"Reference: {ref}")
                            print(f"File: {referenced_filename}")
                            print(f"Bad channels: {ecog_reref.info['bads']}")
                            print(f"{'='*60}")
                            
                            # Plot the data for manual review
                            fig = ecog_reref.plot(
                                title=f"{pat_fs} - {ref.upper()} Referenced Data (Manual Review)",
                                scalings=dict(seeg=200e-6),
                                n_channels=32,
                                remove_dc=True,
                                show_scrollbars=True,
                                duration=15.0,
                                show=True,
                                block=True
                            )
                            
                            #%% Data Quality Check and Visual Inspection
                            
                            print(f"\n{'='*60}")
                            print(f"DATA QUALITY CHECK - {ref.upper()} REFERENCE")
                            print(f"{'='*60}")
                            
                            # Count total and good contacts
                            total_contacts = len(ecog_reref.ch_names)
                            good_contacts = len([ch for ch in ecog_reref.ch_names if ch not in ecog_reref.info['bads']])
                            bad_contacts = total_contacts - good_contacts
                            
                            print(f"📊 Contact Summary:")
                            print(f"  Total contacts: {total_contacts}")
                            print(f"  Good contacts: {good_contacts}")
                            print(f"  Bad contacts: {bad_contacts}")
                            print(f"  Good contact percentage: {(good_contacts/total_contacts)*100:.1f}%")
                            
                            # Log contact information
                            logger.log_decision(f"{ref.upper()}_CONTACTS", f"Total: {total_contacts}, Good: {good_contacts}, Bad: {bad_contacts}")
                            logger.log_decision(f"{ref.upper()}_GOOD_CONTACTS_PCT", f"{(good_contacts/total_contacts)*100:.1f}%")
                            
                            # Check montage quality
                            montage_info = ecog_reref.info['dig']
                            if montage_info is not None:
                                valid_positions = sum(1 for pos in montage_info if not np.any(np.isnan(pos['r'])))
                                print(f"📍 Montage Quality:")
                                print(f"  Total electrode positions: {len(montage_info)}")
                                print(f"  Valid positions: {valid_positions}")
                                print(f"  Position validity: {(valid_positions/len(montage_info))*100:.1f}%")
                                
                                logger.log_decision(f"{ref.upper()}_MONTAGE", f"Valid positions: {valid_positions}/{len(montage_info)}")
                            else:
                                print("⚠️ No montage information available")
                                logger.log_decision(f"{ref.upper()}_MONTAGE", "No montage information")
                            
                            # Data quality metrics
                            data = ecog_reref.get_data()
                            good_ch_indices = [i for i, ch in enumerate(ecog_reref.ch_names) 
                                             if ch not in ecog_reref.info['bads']]
                            
                            if good_ch_indices:
                                good_data = data[good_ch_indices, :]
                                
                                # Calculate quality metrics
                                mean_amplitude = np.mean(np.abs(good_data))
                                std_amplitude = np.std(good_data)
                                signal_range = np.max(good_data) - np.min(good_data)
                                
                                print(f"📈 Signal Quality Metrics:")
                                print(f"  Mean amplitude: {mean_amplitude:.2e}")
                                print(f"  Std amplitude: {std_amplitude:.2e}")
                                print(f"  Signal range: {signal_range:.2e}")
                                
                                logger.log_decision(f"{ref.upper()}_SIGNAL_QUALITY", f"Mean: {mean_amplitude:.2e}, Std: {std_amplitude:.2e}")
                                
                                # Check for flat channels (potential issues)
                                channel_stds = np.std(good_data, axis=1)
                                flat_threshold = np.median(channel_stds) * 0.1
                                flat_channels = np.sum(channel_stds < flat_threshold)
                                
                                if flat_channels > 0:
                                    print(f"⚠️ Warning: {flat_channels} channels appear unusually flat")
                                    logger.log_decision(f"{ref.upper()}_FLAT_CHANNELS", f"Count: {flat_channels}")
                                else:
                                    print("✅ No unusually flat channels detected")
                                    logger.log_decision(f"{ref.upper()}_FLAT_CHANNELS", "None detected")
                            
                            #%% PSD Visualization for Quality Assessment
                            
                            print(f"\n{'='*60}")
                            print(f"PSD VISUALIZATION - {ref.upper()} REFERENCE")
                            print(f"{'='*60}")
                            
                            # Create PSD plots for quality assessment
                            print("   Creating PSD plots for quality assessment...")
                            psd_figures = plot_psd_with_scales(
                                ecog_reref,
                                scale_type='log_y',  # Log y-axis for better power visualization
                                batch_size=32,
                                freq_bands=['all'],
                                patient_id=f"{pat_fs}_{ref}"
                            )
                            
                            print("   PSD plots created. Review for:")
                            print("   - Line noise artifacts (peaks at 60Hz, 120Hz, etc.)")
                            print("   - Unusual frequency distributions")
                            print("   - Channels with very high or very low power")
                            
                            #%% Final Summary for this reference
                            
                            print(f"\n{'='*60}")
                            print(f"DATA CHECK SUMMARY - {ref.upper()} REFERENCE")
                            print(f"{'='*60}")
                            
                            print(f"✅ Data Quality Assessment Complete")
                            print(f"  Patient: {pat_fs}")
                            print(f"  Movie: {mov}")
                            print(f"  Reference: {ref}")
                            print(f"  Sampling rate: {ecog_reref.info['sfreq']:.0f} Hz")
                            print(f"  Duration: {ecog_reref.times[-1]:.1f} seconds")
                            print(f"  Good contacts: {good_contacts}/{total_contacts} ({(good_contacts/total_contacts)*100:.1f}%)")
                            
                            # Quality assessment
                            quality_score = (good_contacts/total_contacts) * 100
                            if quality_score >= 90:
                                quality_status = "EXCELLENT"
                            elif quality_score >= 75:
                                quality_status = "GOOD"
                            elif quality_score >= 50:
                                quality_status = "FAIR"
                            else:
                                quality_status = "POOR"
                            
                            print(f"  Overall quality: {quality_status}")
                            logger.log_decision(f"{ref.upper()}_QUALITY_ASSESSMENT", f"Score: {quality_score:.1f}%, Status: {quality_status}")
                            
                        else:
                            print(f"❌ Failed to load {ref} referenced data")
                            logger.log_error(f"Failed to load {ref} referenced data")
                            
                    except Exception as e:
                        print(f"❌ Error loading {ref} referenced data: {e}")
                        logger.log_error(f"Error loading {ref} referenced data: {e}")
                        
                else:
                    print(f"❌ Referenced file not found: {referenced_filename}")
                    logger.log_error(f"Referenced file not found: {referenced_filename}")
                    
                # Check if bad channels file exists
                if os.path.exists(ref_bad_channels_file):
                    print(f"✅ Found bad channels file: {ref_bad_channels_file}")
                else:
                    print(f"⚠️ Bad channels file not found: {ref_bad_channels_file}")
                    logger.log_error(f"Bad channels file not found: {ref_bad_channels_file}")
            
            # Finish logging
            logger.finish_log()
            
            if single_file_mode:
                print(f"\n{'='*60}")
                print("✅ SINGLE FILE PROCESSING COMPLETE")
                print(f"Patient: {pat_fs}")
                print(f"Movie: {mov}")
                print(f"Log file: {logger.log_file}")
                print(f"{'='*60}\n")
                # Exit after processing single file
                sys.exit(0)
