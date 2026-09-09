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

machine_path = 'media/christine'#'media/christine'#'Volumes' #'media/christine'

sys.path.insert(0, f'/{machine_path}/Samsung/EPIPE-movie_nwb/Python')
sys.path.append(f'/{machine_path}/Samsung/iEEG2NWB-main')

# sys.path.insert(0, '/Users/christinechesebrough/Documents/EPIPE-movie_nwb/Python')
# sys.path.append('/Users/christinechesebrough/Documents/iEEG2NWB-main')

import pycircstat2
from pycircstat2 import hypothesis
from epipe import inspectNwb, nwb2mne, read_ielvis, reref_avg, reref_bipolar,filter_hfa_continuous, reref_white_matter

wd = f'/{machine_path}/Samsung/scripts/movies_ET_attn_main'
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
    ProcessingLogger, detect_spikes_ref1, interpolate_spikes_psd
)

# path to temporal response function
module_path = '/Users/christinechesebrough/Documents/data_standardization-main/src/temporal_response_function.py'
module_name = 'temporal_response_function'

# spec = importlib.util.spec_from_file_location(module_name, module_path)
# trf = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(trf)

def electrode_group(ch: str) -> str:
    return re.sub(r'\d+$', '', ch)

#%%

# Parameters
full_task_name = 'movies'
pipeline_name = 'preprocess_movies'
pipeline_version = 'v.1.0.725'

resample_fs = 600

vid = 'inscapes'#["dme", "despicable_me_english"]

# Types of references to use in analyses
# Must be a list containing at least one of the options: "avg", "bip"
ref_types = ['avg']

n_jobs = 16

convert_db = True

# Frequency range
freq_range = [70, 150]
n_freq_bins = 10
freq_space = 'log'      # 'log', 'lin'

resample_bha_fs = 100

freq_bands = ['low','middle','high']

movie_table = pd.read_excel(f'/{machine_path}/Samsung/Movie_data/data/electrode_localization/movie_table.xlsx')

# Define directories
data_dir = f'/{machine_path}/Samsung/Movie_data/movies_nwb_standard'
#data_dir = f'/{machine_path}/Samsung/exp_sampling/es_nwb_standard'
#data_dir = f'/{machine_path}/Samsung/Movie_data/movies_new_nwb'
#data_dir = f'/{machine_path}/Samsung/AV40_data/new_converted'
#data_dir = f'/{machine_path}/Samsung/exp_samp_data/converted'
fs_dir = f'/{machine_path}/Data/anatomy'
#prep_dir = f'/{machine_path}/Samsung/AV40_data/new_converted'
prep_dir = f'/{machine_path}/Samsung/Movie_data/movies_prep_standard'
#prep_dir = f'/{machine_path}/Samsung/Movie_data/exp_samp_prep_standard'
corr_dir = f'/{machine_path}/Samsung/Movie_data/data/movie_elec_corr_sheets'

frame_dir = f'/{machine_path}/Samsung/Movie_data/data/video_frames'
lum_dir = f'/{machine_path}/Samsung/Movie_data/data/luminance'

if not os.path.exists(lum_dir):
    os.makedirs(lum_dir)

compute_luminance = False

et_prep_dir = 'Eye_prep'
audio_dir = 'Audio'
neural_prep_dir = 'Neural_prep'
hfa_dir = 'HFA'

# Single File Processing Mode
# Set to True to process only one specific file
single_file_mode = False 

# When single_file_mode is True, specify the file to process
# Format: 'patient_id' or 'patient_id_implant_number'
target_patient = 'LH019'#'NS205'  # e.g., 'NS189' or 'NS189_01'
#target_movie = 'sub-NS127_ses-02_task-the_present_run-1_ieeg.nwb'
target_movie = 'LH019_ses-Exp_Samp_Inscapes01_behavior+ecephys.nwb'

if vid == 'despicable_me_hungarian':
    patients= [
    # 'LH010',
    # 'NS127',
    # 'NS135',
    # 'NS136',
    # 'NS137',
    # 'NS138',
    # 'NS140',
    # 'NS144',
    # 'NS145',
    # 'NS149',
    # 'NS151',
    # 'NS153',
    # 'NS154',
     # 'NS155',
    #  'NS164',
      'NS174',
    #'NS178',
    # 'NS211'
     ]
    
  # dmh_bad_ET_pats = 
    #    ' NS128'
    # 'NS166'
    # 'NS167'
    # 'NS153'
    # 'NS155'
    #' NS149'
    #' LH012'
    # 'NS142'
    #'NS170'


if vid == 'despicable_me_english':
     patients = [
     'NS127',
    # 'NS135',
    # 'NS136',
    # 'NS137',
    # 'NS138',
   #  'NS140',
    # 'NS154',
    # 'NS155',
   # 'NS164',
   #  'NS174',
  #   'NS178',
    # 'NS190',
    # 'NS191',
    # 'NS193',
    # 'NS194',
    # 'NS201',
    # 'NS204',
    # 'NS205'
    ]
     
    #  dme_bad_ET_pats = 
    # ['NS153'
    # 'NS151'
    # 'NS128'
    # 'NS155'
    # 'NS167'
    # 'LH012'
    # 'NS149'
    # 'NS142'
    # 'NS144'
    # 'NS170',
    #'NS192',
    # 'NS189',
    # 'LH012',
    # 'NS145',
    # 'NS170']


if vid == 'inscapes':
    patients = [
  #   'NS127',
    # 'NS135',
    # 'NS136',
    # 'NS137',
    # 'NS138',
  #   'NS140',
   # 'NS174',
    # "NS155,
    # 'NS164',
    # 'NS178',
    # 'NS205',
    # 'NS210',
  #  'NS211',
  'NS166'
    ]     
    
   # inscapes_bad_ET_pats = 
   # ['NS174_02',
   #  'NS151',
   #  'NS155',
   #  "NS153",
   #  "NS166",
   #  "NS142",
   #  "NS149",
   #  'NS167',
   #144
   # 'NS154' # bad EEG,
   #'NS211' #bad EEG]
   
   
### PREPROC NOTES ###
# 178 spike interpolation introduces large artifacts at notch freqs
# 166 high fs, change filter, 
    
vid_keys = {
    'betta': ['betta', 'Betta'],
    'inscapes': ['inscapes', 'Inscapes'],
    'despicable_me_english': ['despicable_me_english','dme'],
    'despicable_me_hungarian': ['despicable_me_hungarian','dmh'],    
}

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
    if data_dir == f'/{machine_path}/Samsung/Movie_data/movies_nwb_standard':
        target_nwb_path = '{:s}/{:s}/{:s}/{:s}'.format(data_dir, pat, imp, target_movie)
    elif data_dir == f'/{machine_path}/Samsung/movie_data_new/movies_new_nwb_fall25':
        target_nwb_path = '{:s}/{:s}/{:s}'.format(data_dir, pat, target_movie)
    elif data_dir == f'/{machine_path}/Samsung/Movie_data/movies_new_nwb':
        target_nwb_path = '{:s}/{:s}/{:s}/ieeg/{:s}'.format(data_dir, pat, imp, target_movie)
    elif data_dir ==f'/{machine_path}/Samsung/AV40_data/new_converted':
        target_nwb_path = '{:s}/{:s}/{:s}/ieeg/{:s}'.format(data_dir, pat, imp, target_movie)
    elif data_dir == f'/{machine_path}/Samsung/exp_samp_data/converted':
        target_nwb_path = '{:s}/{:s}/{:s}/ieeg/{:s}'.format(data_dir, pat, imp, target_movie)
    elif data_dir == f'/{machine_path}/Samsung/exp_sampling/es_nwb_standard':
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
for pat in patients:
        
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
        
        if data_dir == f'/{machine_path}/Samsung/Movie_data/movies_nwb_standard':
            ieeg_dir = '{:s}/{:s}/{:s}'.format(data_dir, pat, imp)
        elif data_dir == f'/{machine_path}/Samsung/movie_data_new/movies_new_nwb_fall25':
            ieeg_dir = '{:s}/{:s}/'.format(data_dir, pat)
        elif data_dir == f'/{machine_path}/Samsung/Movie_data/movies_new_nwb':
            ieeg_dir = '{:s}/{:s}/{:s}'.format(data_dir, pat, imp)
        elif data_dir ==f'/{machine_path}/Samsung/AV40_data/new_converted':
            ieeg_dir = '{:s}/{:s}/{:s}'.format(data_dir, pat, imp)
        elif data_dir == f'/{machine_path}/Samsung/exp_samp_data/converted':
            ieeg_dir = '{:s}/{:s}/{:s}'.format(data_dir, pat, imp)
        elif data_dir == f'/{machine_path}/Samsung/exp_sampling/es_nwb_standard':
            ieeg_dir = '{:s}/{:s}/{:s}'.format(data_dir, pat, imp)
    
        if single_file_mode:
            movies = movies_to_process
        else:
            movies = [f for f in os.listdir(ieeg_dir) if f.endswith('.nwb')]
        
            if 'vid' in locals():
                keys = vid_keys.get(vid, [vid])
                movies = [mov for mov in movies if any(k.lower() in mov.lower() for k in keys)]
        
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
        
        sub_neural_prep_dir = '{:s}/{:s}/{:s}'.format(prep_dir, pat_fs, neural_prep_dir)
        sub_et_prep_dir = '{:s}/{:s}/{:s}'.format(prep_dir, pat_fs, et_prep_dir)
        
        if not os.path.exists(sub_neural_prep_dir):
            os.makedirs(sub_neural_prep_dir)
        if not os.path.exists(sub_et_prep_dir):
            os.makedirs(sub_et_prep_dir)
    
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
            if data_dir == f'/{machine_path}/Samsung/Movie_data/movies_nwb_standard':
                nwb_fname = '{:s}/{:s}/{:s}/{:s}'.format(data_dir, pat, imp, mov)
            elif data_dir == f'/{machine_path}/Samsung/movie_data_new/movies_new_nwb_fall25':
                nwb_fname = '{:s}/{:s}/{:s}'.format(data_dir, pat, mov)
            elif data_dir == f'/{machine_path}/Samsung/Movie_data/movies_new_nwb':
                nwb_fname = '{:s}/{:s}/{:s}/ieeg/{:s}'.format(data_dir, pat, imp, mov)
            elif data_dir ==f'/{machine_path}/Samsung/AV40_data/new_converted':
                nwb_fname = '{:s}/{:s}/{:s}/ieeg/{:s}'.format(data_dir, pat, imp,mov)
            elif data_dir == f'/{machine_path}/Samsung/exp_samp_data/converted':
                nwb_fname = '{:s}/{:s}/{:s}/ieeg/{:s}'.format(data_dir, pat, imp,mov)
            elif data_dir ==f'/{machine_path}/Samsung/exp_sampling/es_nwb_standard':
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
            # Log NWB file loading
         #   logger.log_nwb_load(nwb_fname)
            
            # Get info on data in NWB file
            nwbInfo = inspectNwb(nwb)
            tsInfo = nwbInfo['timeseries']
            elecTable = nwbInfo['elecs']
            
            # Define the subject-specific directory
            subid = os.path.basename(sub_fs_dir)
            elecReconDir = os.path.join(sub_fs_dir, 'elec_recon')
            
            elec_recon_dir = corr_dir    
    
            excel_files = sorted([
                f for f in os.listdir(elec_recon_dir)
                if pat in f
                and f.endswith('.xlsx')
                and not f.startswith('.')
            ])
            if not excel_files:
                print(f"  [SKIP] No correspondence .xlsx for {pat} found in {elec_recon_dir}")
                #continue
    
            # Use most recently modified if multiple exist (matches Python HFO script)
            excel_path = max(
                [os.path.join(elec_recon_dir, f) for f in excel_files],
                key=os.path.getmtime
            )
            print(f"  Using: {os.path.basename(excel_path)}")
            elec_ref_table = pd.read_excel(excel_path)
            
            
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
            
                # Create `montage` data structure as required by MNE1
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
            
            if "TTL" in nwb.acquisition:
                ttl_container_name = 'TTL'
                ttls = nwb.acquisition["TTL"]
                
            elif "ttl" in nwb.acquisition:
                ttls = nwb.acquisition["ttl"]
                ttl_container_name = 'ttl'
            else:
                raise KeyError("No TTL acquisition found")

            try:
                ttls = nwb.get_acquisition(ttl_container_name).timestamps[()]
            except:
                # An analog TTL channel, convert to discrete timestamps
                ana_ttls = nwb.get_acquisition(ttl_container_name).data[()].flatten()
                ttl_rate = nwb.get_acquisition(ttl_container_name).rate
                from epipe import ana2dig
                _, ttls = ana2dig(ana_ttls, fs=ttl_rate, min_diff=0.4, return_time=True)
            
            ttl_id = nwb.acquisition['TTL'].data[:]
            
                   
            # Cut and save the audio data        
            if 'Fix' in mov:
                idx_movie = np.logical_and(t_audio >= ttls[0], t_audio < ttls[1])
                
            else:
                
                # Frame time
                frame_time = nwb.trials['start_time'].data[:]
                frame_time_end = nwb.trials['stop_time'].data[:]
                idx_movie = np.logical_and(t_audio >= frame_time[0], t_audio < frame_time[-1])
                
               
            # Preprocess the EEG data
            # Create preprocessing directory using standardized paths
            if not os.path.exists(file_paths['sub_prep_dir']):
                os.makedirs(file_paths['sub_prep_dir'])
    
            # ecog.plot(
            #     scalings=dict(seeg=200e-6),  
            #     n_channels=32,               
            #     remove_dc=False,              # remove mean per channel
            #     show_scrollbars=True,        # allow vertical scrolling
            #     duration=12.0,
            #     block = True# time window in seconds
            # )
            
            # Load event timing alignment dfs and metadata
      
                        
        #%%    
     #       THIS IS OPTIONAL BUT SOMETIMES NECESSARY; IT'S APPARENT WHETHER IT IS NECESSARY BASED ON THE NEXT STEP OF NOTCH FILTERING, BANDPASS, AND DOWNSAMPLING
             ## If bad initial reference, option to rereference before notch filtering and downsample
     
            # reref_to_ref = False
            # if reref_to_ref:
            #     ref_channels = [ch for ch in ecog.ch_names if ch.lower().startswith('ref')]
            #     print("Candidate reference channels:", ref_channels)
                
            #     for ref in ref_channels:
            #         data = ecog.copy().pick(ref).get_data()
            #         std = np.std(data)
            #         print(f"{ref}: std={std}")
                                
            #     # Ask for user input by channel name
            #     selected_ref = input("Enter the NAME of the best reference channel to use (case-sensitive): ")
                
            #     if selected_ref in ref_channels:
            #         print(f"Selected reference channel: {selected_ref}")
                    
            #         # Apply referencing
            # if pat == 'NS178':
            #     selected_ref = 'RDa15'
            #     ecog.set_eeg_reference(ref_channels=[selected_ref], projection=False)
                
            #     ecog.plot(
            #         scalings=dict(seeg=200e-6),  
            #         n_channels=32,               
            #         remove_dc=False,              # remove mean per channel
            #         show_scrollbars=True,        # allow vertical scrolling
            #         duration=12.0,
            #         block = True# time window in seconds
            #     )
            #         print("Re-referencing applied with selected channel.")
            #     else:
            #         print("Invalid channel name. No referencing applied.")
                

                #%%        
            reload =False
            
            if reload:    
                
                preprocessed_filename = file_paths["preprocessed_file"]
                preprocessed_dir = os.path.dirname(preprocessed_filename)

                fif_files = sorted([
                    f for f in os.listdir(preprocessed_dir)
                    if vid in f
                    and f.endswith(".fif")
                    and not f.startswith(".")
                    #and ("prep_ieeg" in f or "preprocessed_ieeg" in f)
                ])
                
                
                preprocessed_matches = [f for f in fif_files if "ieeg_preprocessed" in f]
                prep_matches = [f for f in fif_files if "ieeg_preprocessed" in f]
                
                if preprocessed_matches:
                    fif_file = os.path.join(preprocessed_dir, preprocessed_matches[0])        
                    print(f"Reloading cut and filtered data: {fif_file}")
                    ecogResampled = mne.io.read_raw_fif(fif_file, preload=True)
                
                    print(ecogResampled)
                    print("\nChannel names:")
                    print(ecogResampled.ch_names)
                    filt_and_resample = False
                # elif prep_matches:
                #     #fif_file = os.path.join(preprocessed_dir, prep_matches[0])
                #     filt_and_resample = True       
                else:
                    filt_and_resample = True #raise FileNotFoundError(f"No matching preprocessed FIF file found for {vid} in {preprocessed_dir}")
                
            else: 
                filt_and_resample = True
                # if len(fif_files) > 1:
                #     raise RuntimeError(f"Multiple possible preprocessed FIF files found: {fif_files}")
            
                #fif_file = fif_files[0]

                # # Plot to visually inspect data tracesf
                # ecogPreproc.plot(
                #     scalings=dict(seeg=200e-6),
                #     n_channels=32,
                #     remove_dc=True,
                #     decim=10,
                #     show_scrollbars=True,   # test
                #     duration=20.0,           # test
                # )

    #%% Remove spikes if necessary
    
            if pat == 'NS178':
                selected_ref = 'RDa15'
                ecog.set_eeg_reference(ref_channels=[selected_ref], projection=False)
                
                ecog.plot(
                    scalings=dict(seeg=200e-6),  
                    n_channels=32,               
                    remove_dc=False,              # remove mean per channel
                    show_scrollbars=True,        # allow vertical scrolling
                    duration=12.0,
                    block = True# time window in seconds
                )
    
            if pat in ['NS178','NS191','NS211']:
                remove_spikes =True
                if remove_spikes:          
                    spikes_by_chan = detect_spikes_all_channels(
                        ecog,
                        picks=None,
                        thresh_z=5.5,
                        max_width_ms=30.0,
                        min_separation_ms=5.0
                        )
                
                    interp_half_window_ms = 15
                    sfreq = ecog.info['sfreq']
                    interp_half_window_samp = int(interp_half_window_ms * sfreq / 1000.0)
                    
                    raw_clean = ecog.copy()
                    data = raw_clean._data
                    n_ch, n_samp = data.shape
                    
                    for ch_idx, ch_name in enumerate(raw_clean.ch_names):
                        spike_samples = spikes_by_chan.get(ch_name, np.array([], dtype=int))
                        if len(spike_samples) == 0:
                            continue
                    
                        print(f"Interpolating {len(spike_samples)} spikes in channel {ch_name}")
                    
                        for s in spike_samples:
                            seg_start = max(1, s - interp_half_window_samp)
                            seg_end   = min(n_samp - 2, s + interp_half_window_samp)
                    
                            left = seg_start - 1
                            right = seg_end + 1
                    
                            seg_idx = np.arange(seg_start, seg_end + 1)
                            data[ch_idx, seg_idx] = np.interp(
                                seg_idx,
                                [left, right],
                                data[ch_idx, [left, right]]
                            )
                    
                    ecog = raw_clean
                    
                    # spikes_by_chan = detect_spikes_all_channels(
                    #     ecog,
                    #     picks=None,
                    #     thresh_z=5.5,
                    #     max_width_ms=20.0,
                    #     min_separation_ms=5.0
                    # )
                
                    # raw_clean = ecog.copy()
                
                    # for ch_idx, ch_name in enumerate(raw_clean.ch_names):
                    #     spike_samples = spikes_by_chan.get(ch_name, np.array([], dtype=int))
                
                    #     if len(spike_samples) == 0:
                    #         continue
                
                    #     print(f"Replacing {len(spike_samples)} spikes in channel {ch_name}")
                
                    #     raw_clean = interpolate_spikes_psd(
                    #         raw_clean,
                    #         spike_samples,
                    #         ch_idx=ch_idx,
                    #         window_ms=6.5,
                    #         context_ms=250.0
                    #     )
                
                    # ecog = raw_clean

                                    
                    #%%
            #Notch filter, bandpass filter, then downsample
            if filt_and_resample:
                print('--->Applying notch filters, bandpass filter, and downsampling to %2.fHz' % resample_fs)
                
                # Log filtering section
               # logger.log_section("FILTERING STEPS")
                
                # Apply notch filter first to remove line noise
                if pat == "NS135":
                    notch_freqs = (60,104,120,180)
                # if pat == 'NS178':
                #     notch_freqs = (58,60,62,118,120,122,178,180,182)
                if pat == "NS151": 
                    notch_freqs = (60,104,117,180)
                # if pat == "NS178":
                #     notch_freqs = (60,120,133,180)
                # elif pat == "NS166":
                    #notch_freqs = (57,60,67,114,120,134,180)
                    # notch_freqs = (57, 60, 67,68, 84, 111,112,120, 128, 135, 167, 180)
                # elif pat == "NS211":
                #     notch_freqs = (57,60,67,114,120,134,180)
                elif pat == "NS174":
                    if ses_num == 3:
                        notch_freqs = (97,142)
                        ecog.notch_filter(freqs=notch_freqs, notch_widths=4)
                        notch_freqs = (60,120,180)
                        ecog.notch_filter(freqs=notch_freqs, notch_widths=2)

                    else: 
                        notch_freqs = (60,120,180)
                else:    
                    notch_freqs = (60, 120, 180)
                     
                    ecog.notch_filter(freqs=notch_freqs, notch_widths=2)
              #  logger.log_filtering("NOTCH", f"Frequencies: {notch_freqs} Hz, Widths: 2 Hz")
                
                #
                # Apply bandpass filter 
                ecog.filter(l_freq=0.1, h_freq=170.0)
               # logger.log_filtering("BANDPASS", f"Low: 0.1 Hz, High: 170.0 Hz")
                
                # if pat == 'NS178':
                #     selected_ref = 'RDa15'
                
                #     # Make sure the reference channel is not marked bad
                #     if selected_ref in ecog.info["bads"]:
                #         ecog.info["bads"].remove(selected_ref)
                #         print(f"Removed {selected_ref} from bad channels before rereferencing.")
                
                #     ecog.set_eeg_reference(ref_channels=[selected_ref], projection=False)
                
                #     ecog.plot(
                #         scalings=dict(seeg=100e-6),
                #         n_channels=32,
                #         remove_dc=False,
                #         show_scrollbars=True,
                #         duration=12.0,
                #         block=True
                #     )
               
                # Then downsample to target sampling rate
                ecogResampled = ecog.resample(resample_fs)
               # logger.log_filtering("DOWNSAMPLING", f"From: {orig_fs} Hz, To: {resample_fs} Hz")
        
                # crop by movie start and end times
#                ecogResampled.crop(tmin=start_time, tmax=stop_time)
                ecogResampled.crop(tmin=frame_time[0], tmax=frame_time[-1])

                
                # Rebuild as an independent Raw object so time starts at sample 0
                ecogResampled = mne.io.RawArray(
                    ecogResampled.get_data(),
                    ecogResampled.info.copy()
                )
                
                print("first_samp:", ecogResampled.first_samp)
                print("first_time:", ecogResampled.first_time)
                print("times start:", ecogResampled.times[0])
                                


                
             
                    #%%
                    
                   ### ECOGRESAMPLED??
                        
            # if pat in ['NS178','NS191','NS211']:
            #     remove_spikes =True
            #     if remove_spikes:          
            #         spikes_by_chan = detect_spikes_all_channels(
            #             ecog,
            #             picks=None,
            #             thresh_z=5.5,
            #             max_width_ms=20.0,
            #             min_separation_ms=5.0
            #             )
                
            #         interp_half_window_ms = 15
            #         sfreq = ecog.info['sfreq']
            #         interp_half_window_samp = int(interp_half_window_ms * sfreq / 1000.0)
                    
            #         raw_clean = ecog.copy()
            #         data = raw_clean._data
            #         n_ch, n_samp = data.shape
                    
            #         for ch_idx, ch_name in enumerate(raw_clean.ch_names):
            #             spike_samples = spikes_by_chan.get(ch_name, np.array([], dtype=int))
            #             if len(spike_samples) == 0:
            #                 continue
                    
            #             print(f"Interpolating {len(spike_samples)} spikes in channel {ch_name}")
                    
            #             for s in spike_samples:
            #                 seg_start = max(1, s - interp_half_window_samp)
            #                 seg_end   = min(n_samp - 2, s + interp_half_window_samp)
                    
            #                 left = seg_start - 1
            #                 right = seg_end + 1
                    
            #                 seg_idx = np.arange(seg_start, seg_end + 1)
            #                 data[ch_idx, seg_idx] = np.interp(
            #                     seg_idx,
            #                     [left, right],
            #                     data[ch_idx, [left, right]]
            #                 )
                    
            #         ecog = raw_clean
                    
                    # spikes_by_chan = detect_spikes_all_channels(
                    #     ecogResampled,
                    #     picks=None,
                    #     thresh_z=5.5,
                    #     max_width_ms=20.0,
                    #     min_separation_ms=5.0
                    # )
                
                    # raw_clean = ecogResampled.copy()
                
                    # for ch_idx, ch_name in enumerate(raw_clean.ch_names):
                    #     spike_samples = spikes_by_chan.get(ch_name, np.array([], dtype=int))
                
                    #     if len(spike_samples) == 0:
                    #         continue
                
                    #     print(f"Replacing {len(spike_samples)} spikes in channel {ch_name}")
                
                    #     raw_clean = interpolate_spikes_psd(
                    #         raw_clean,
                    #         spike_samples,
                    #         ch_idx=ch_idx,
                    #         window_ms=6.5,
                    #         context_ms=150.0
                    #     )
                
                    # ecogResampled = raw_clean
        

    #%%         # Display the raw traces and mark bad channels
    
            load_existing_bads = True
            
            if load_existing_bads:
                if os.path.exists(file_paths['bad_channels_file']):
                    with open(file_paths['bad_channels_file'], 'r') as f:
                        bad_channels = [
                            line.strip()
                            for line in f
                            if line.strip() and not line.startswith('#')
                        ]
                    
                    # Keep only channels that exist in the current MNE object
                    bad_channels = [
                        ch for ch in bad_channels
                        if ch in ecogResampled.ch_names
                    ]
                    
                    ecogResampled.info['bads'] = bad_channels
                    
                    print(f"Loaded {len(bad_channels)} bad channels:")
                    print(ecogResampled.info['bads'])
                    
                
                        
            print("\n--- Manual Bad Channel Marking ---")
            print("Mark bad channels in the interactive plot, then close the plot to continue.")
            fig = ecogResampled.plot(
                scalings=dict(seeg=100e-6),
                n_channels=64,
                remove_dc=True,
                show_scrollbars=True,
                duration=10.0,
                block=True,  
                )
            
            # fig = ecog.plot(
            #     scalings=dict(seeg=100e-6),
            #     n_channels=64,
            #     remove_dc=True,
            #     show_scrollbars=True,
            #     duration=10.0,
            #     block=True,  
            #     )
            
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
            ecogResampled.save(preprocessed_filename, fmt='single', overwrite=True)
            print(f"Preprocessed data saved to: {preprocessed_filename}")
            logger.log_fif_save(preprocessed_filename, file_description="Preprocessed FIF")
            
            # Final integrity check before saving
            final_integrity = check_bad_channels_integrity(ecogPreproc, "Final Preprocessed Data")
            
            # Final MNE structure validation
            final_structure = validate_mne_structure(ecogPreproc, "Final Preprocessed Data")
    
    
    
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
                    freq_range = (70, 150)
                elif freq_band == "all":
                    freq_range = (1,170)
                
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
                scale_type='log_y',  # Log x-axis, linear y-axis #log_y #log_x
                batch_size=64,
                freq_bands=['all'],
                patient_id=pat)
            
            print("\nClose all PSD plot windows to continue...")
            plt.show(block=True)
    
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
            
                if ref == 'wm_legacy':
                    # Apply WM referencing
                    ecog_reref = ecogPreproc.set_eeg_reference(ref_channels=wm_references)
                    ecog_reref.save(referenced_filename, overwrite=True)
                    print(f"WM-referenced data saved to: {referenced_filename}")
                    logger.log_referencing("WM", f"Reference channels: {wm_references}")
                    logger.log_file_save("WM_REFERENCED", referenced_filename)
                    logger.log_fif_save(referenced_filename, file_description="WM Referenced FIF")
            
                if ref == 'wm':
                    # Apply WM referencing
                    #ecog_reref = ecogPreproc.set_eeg_reference(ref_channels=wm_references_auto)
     
                    raw_for_reref = ecogPreproc.copy().drop_channels(ecogPreproc.info["bads"])
    
                    ecog_reref, ref_data = mne.set_eeg_reference(
                        raw_for_reref,
                        ref_channels=wm_references_auto,
                        ch_type="seeg",
                    )
    
                    #ecogWMavg, _ = set_eeg_reference(ecogPreproc, ref_channels=wm_references_auto, copy=True)
                    ecog_reref.save(referenced_filename, overwrite=True)
                    print(f"WM-referenced data saved to: {referenced_filename}")
                    logger.log_referencing("WM", f"Reference channels: {wm_references_auto}")
                    logger.log_file_save("WM_REFERENCED", referenced_filename)
                    logger.log_fif_save(referenced_filename, file_description="WM Referenced FIF")
            
                elif ref == 'avg':
                    # Apply average referencing
                    if pat in []:#'NS140','NS144','NS128','NS217','NS224','NS154','NS166','NS167','NS174']:
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
                        scalings=dict(seeg=100e-6),
                        n_channels=64,
                        remove_dc=True,
                        show_scrollbars=True,
                        duration=15.0,
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
                    
                    ecog_reref.save(referenced_filename, overwrite=True)

                    #%%
                    # Optional: PSD visualization for this reference
                    print(f"\n📊 PSD visualization for {ref} reference")
                    psd_figures = plot_psd_with_scales(
                        ecog_reref, 
                        scale_type='log_y',
                        batch_size=64,
                        freq_bands=['high'],
                        patient_id=f"{pat_fs}_{ref}"
                    )
                    print("\nClose all PSD plot windows to continue...")
                    plt.show(block=True)
                    
                    print(f"✅ Completed bad channel marking for {ref} reference")
                    print(f"  - Bad channels: {len(ecog_reref.info['bads'])}")
                    print(f"  - Good channels: {len(ecog_reref.ch_names) - len(ecog_reref.info['bads'])}")
                    print(f"  - Saved to: {ref_bad_channels_file}")
    
        
                    
                    
                    #%%
                    del nwb, ecog, ecogResampled, ecog_reref, ecogPreproc, file_paths
                    import gc
                    gc.collect()
                    