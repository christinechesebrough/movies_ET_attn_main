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

machine_path = 'Volumes'#'Volumes' #'media/christine'

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


# Import EEG preprocessing helper functions
from eeg_preproc_helpers import (
    plot_power_spectra,  plot_psd_batched, plot_psd_with_scales,
    create_file_paths, apply_highpass_filter,
    save_bad_channels, load_bad_channels, check_bad_channels_integrity, load_preprocessed_data, validate_mne_structure, preserve_mne_structure, restore_mne_structure, load_data_with_bad_channels, summarize_bad_channels, 
    interpolate_spikes, make_groups_from_prefix, regress_out_noise_by_group,
    detect_spikes_all_channels, reref_avg_by_group,
    ProcessingLogger, detect_spikes_ref1
)

# Parameters
full_task_name = 'movies'
pipeline_name = 'preprocess_movies'
pipeline_version = 'v.1.0.725'

resample_fs = 600

vid = ['']#["dme", "despicable_me_english"]

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

base_dir = f'/{machine_path}/Samsung/AV40_data'
data_dir = '{:s}/{:s}'.format(base_dir, 'new_converted') 
fs_dir = f'/{machine_path}/Samsung/anatomy'
prep_dir =  '{:s}/{:s}'.format(base_dir,'data_prep')
corr_dir = f'/{machine_path}/Samsung/Movie_data/data/movie_elec_corr_sheets'

log_dir =  '{:s}/{:s}'.format(base_dir,'logs')


et_prep_dir = 'Eye_prep'
audio_dir = 'Audio'
neural_prep_dir = 'Neural_prep'
hfa_dir = 'HFA'

# Set to True to process only one specific file
single_file_mode = True
if single_file_mode:
    target_patient = 'NS193'#'NS205'  # e.g., 'NS189' or 'NS189_01'
    #target_run = 'sub-NS127_ses-02_task-the_present_run-1_ieeg.nwb'
    target_run = 'NS193_ses-AV40_AV01_behavior+ecephys.nwb'

# if single_file_mode is false, list patients
patients = ["NS224"]#,"NS190","NS191","NS193","NS194","NS201_02","NS204","NS205"]

#%% helper functions

def parse_stimulus_start_times(log_file, out_csv=None):
    """
    Parse a PsychoPy .log file and save stimulus start times to CSV.

    Extracts only stimulus START events:
        - visual_standard: autoDraw = True
        - visual_deviant: autoDraw = True
        - Sound audio_standard started
        - Sound audio_deviant started

    Ignores stimulus END events:
        - visual_*: autoDraw = False
        - Sound audio_* reached end of file

    Parameters
    ----------
    log_file : str
        Full path to the PsychoPy .log file.

    out_csv : str or None
        Full path for output CSV.
        If None, saves next to the log file with suffix '_stimulus_start_times.csv'.

    Returns
    -------
    stim_df : pandas.DataFrame
        DataFrame with stimulus start times and stimulus labels.
    """

    if out_csv is None:
        base, _ = os.path.splitext(log_file)
        out_csv = base + "_stimulus_start_times.csv"

    # Match visual starts, e.g.:
    # 39.0600     EXP     visual_standard: autoDraw = True
    visual_start_re = re.compile(
        r"^\s*(?P<time>\d+\.\d+)\s+EXP\s+"
        r"(?P<stimulus>visual_(?P<condition>standard|deviant)):\s+autoDraw\s+=\s+True"
    )

    # Match audio starts, e.g.:
    # 39.1103     EXP     Sound audio_standard started
    audio_start_re = re.compile(
        r"^\s*(?P<time>\d+\.\d+)\s+EXP\s+"
        r"Sound\s+(?P<stimulus>audio_(?P<condition>standard|deviant))\s+started"
    )

    rows = []

    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            visual_match = visual_start_re.search(line)
            audio_match = audio_start_re.search(line)

            if visual_match:
                rows.append({
                    "time_s": float(visual_match.group("time")),
                    "modality": "visual",
                    "condition": visual_match.group("condition"),
                    "stimulus": visual_match.group("stimulus"),
                    "event": "start",
                    "raw_line": line.strip()
                })

            elif audio_match:
                rows.append({
                    "time_s": float(audio_match.group("time")),
                    "modality": "auditory",
                    "condition": audio_match.group("condition"),
                    "stimulus": audio_match.group("stimulus"),
                    "event": "start",
                    "raw_line": line.strip()
                })

    stim_df = pd.DataFrame(rows)

    if not stim_df.empty:
        stim_df = stim_df.sort_values("time_s").reset_index(drop=True)

    stim_df.to_csv(out_csv, index=False)

    print(f"Saved {len(stim_df)} stimulus start events to:")
    print(out_csv)

    if not stim_df.empty:
        print("\nCounts by stimulus:")
        print(stim_df.groupby(["modality", "condition", "stimulus"]).size())

    return stim_df


def add_log_cooccurrence_labels(log_df, tolerance_s=0.030):
    """
    Mark log events that occur close to an event from the other modality.

    Parameters
    ----------
    log_df : pandas.DataFrame
        Must contain:
            - time_s
            - modality

    tolerance_s : float
        Time window for calling visual/audio events co-occurring.

    Returns
    -------
    log_df : pandas.DataFrame
        Same dataframe with:
            - cooccurs_with_other_modality
            - nearest_other_modality_time_s
            - nearest_other_modality_dt_s
    """

    log_df = log_df.copy()

    log_df["cooccurs_with_other_modality"] = False
    log_df["nearest_other_modality_time_s"] = np.nan
    log_df["nearest_other_modality_dt_s"] = np.nan

    visual_times = log_df.loc[log_df["modality"] == "visual", "time_s"].to_numpy()
    audio_times = log_df.loc[log_df["modality"] == "auditory", "time_s"].to_numpy()

    for idx, row in log_df.iterrows():
        this_time = row["time_s"]

        if row["modality"] == "visual":
            other_times = audio_times
        elif row["modality"] == "auditory":
            other_times = visual_times
        else:
            continue

        if len(other_times) == 0:
            continue

        diffs = other_times - this_time
        best_idx = np.argmin(np.abs(diffs))
        best_dt = diffs[best_idx]

        log_df.loc[idx, "nearest_other_modality_time_s"] = other_times[best_idx]
        log_df.loc[idx, "nearest_other_modality_dt_s"] = best_dt

        if abs(best_dt) <= tolerance_s:
            log_df.loc[idx, "cooccurs_with_other_modality"] = True

    return log_df

def match_ttls_to_log(ttl_df, log_df, offset_s, max_error_s=0.050):
    """
    Match TTL pulses to nearest log-derived stimulus events.

    Parameters
    ----------
    ttl_df : pandas.DataFrame
        Must contain:
            - ttl_time_s
            - ttl_id

    log_df : pandas.DataFrame
        Must contain:
            - time_s
            - modality
            - condition
            - stimulus

    offset_s : float
        Offset to add to log times to put them into neural/NWB time.

    max_error_s : float
        Maximum allowed absolute difference, in seconds, for accepting a match.

    Returns
    -------
    matched_df : pandas.DataFrame
        TTL dataframe with matched log event information.
    """

    ttl_df = ttl_df.copy().sort_values("ttl_time_s").reset_index(drop=True)
    log_df = log_df.copy().sort_values("time_s").reset_index(drop=True)

    log_df["aligned_time_s"] = log_df["time_s"] + offset_s
    log_df["log_event_type"] = log_df["modality"] + "_" + log_df["condition"]

    matched_rows = []

    aligned_log_times = log_df["aligned_time_s"].to_numpy()

    for _, ttl_row in ttl_df.iterrows():
        ttl_time = ttl_row["ttl_time_s"]

        diffs = ttl_time - aligned_log_times
        abs_diffs = np.abs(diffs)

        best_idx = np.argmin(abs_diffs)
        best_abs_error = abs_diffs[best_idx]
        best_signed_error = diffs[best_idx]

        out = ttl_row.to_dict()

        if best_abs_error <= max_error_s:
            log_row = log_df.iloc[best_idx]

            out["matched"] = True
            out["matched_log_index"] = best_idx
            out["matched_log_time_s"] = log_row["time_s"]
            out["matched_aligned_time_s"] = log_row["aligned_time_s"]
            out["matched_modality"] = log_row["modality"]
            out["matched_condition"] = log_row["condition"]
            out["matched_stimulus"] = log_row["stimulus"]
            out["matched_log_event_type"] = log_row["log_event_type"]
            out["match_error_s"] = best_signed_error

        else:
            out["matched"] = False
            out["matched_log_index"] = np.nan
            out["matched_log_time_s"] = np.nan
            out["matched_aligned_time_s"] = np.nan
            out["matched_modality"] = None
            out["matched_condition"] = None
            out["matched_stimulus"] = None
            out["matched_log_event_type"] = None
            out["match_error_s"] = np.nan

        matched_rows.append(out)

    matched_df = pd.DataFrame(matched_rows)

    return matched_df



#%% Initialize loop and open NWB

broken_nwb = []

if single_file_mode:
    # Single file processing mode
    print(f"\n{'='*60}")
    print(f"SINGLE FILE PROCESSING MODE")
    print(f"Target Patient: {target_patient}")
    print(f"Target Movie: {target_run}")
    print(f"{'='*60}\n")
    
    # Extract session number from the target movie filename
    # Expected format: NS189_ses-02_task-despicable_me_english_run-01_ieeg.nwb
    if 'ses-0' in target_run:
        # Extract session number from filename
        ses_match = re.search(r'ses-(\d+)', target_run)
        if ses_match:
            ses_num = ses_match.group(1)
            imp = f'ses-{ses_num}'
        else:
            print(f"❌ ERROR: Could not extract session number from filename: {target_run}")
            #sys.exit(1)
    else:
        # Fallback to ses-01 if no session info in filename
        imp = 'ses-01'
    
    # Set patient path
    pat = f'{target_patient}'

    # Validate that the target file exists
    if data_dir == f'/{machine_path}/Samsung/Movie_data/movies_nwb_standard':
        target_nwb_path = '{:s}/{:s}/{:s}/{:s}'.format(data_dir, pat, imp, target_run)
    elif data_dir == f'/{machine_path}/Samsung/movie_data_new/movies_new_nwb_fall25':
        target_nwb_path = '{:s}/{:s}/{:s}'.format(data_dir, pat, target_run)
    elif data_dir == f'/{machine_path}/Samsung/Movie_data/movies_new_nwb':
        target_nwb_path = '{:s}/{:s}/{:s}/ieeg/{:s}'.format(data_dir, pat, imp, target_run)
    elif data_dir ==f'/{machine_path}/Samsung/AV40_data/new_converted':
        target_nwb_path = '{:s}/{:s}/{:s}/ieeg/{:s}'.format(data_dir, pat, imp, target_run)
    elif data_dir == f'/{machine_path}/Samsung/exp_samp_data/converted':
        target_nwb_path = '{:s}/{:s}/{:s}/ieeg/{:s}'.format(data_dir, pat, imp, target_run)
    elif data_dir == f'/{machine_path}/Samsung/exp_sampling/es_nwb_standard':
        target_nwb_path = '{:s}/{:s}/{:s}/ieeg/{:s}'.format(data_dir, pat, imp, target_run)

    if not os.path.exists(target_nwb_path):
        print(f"❌ ERROR: Target file not found: {target_nwb_path}")
        print("Please check the target_patient and target_run parameters.")
        print(f"Expected path: {target_nwb_path}")
        #sys.exit(1)
    
    print(f"✅ Target file found: {target_nwb_path}")
    
    # Process single file
    patients_to_process = [pat]
    implants_to_process = [imp]
    movies_to_process = [target_run]
    
else:
    # Batch processing mode
    print(f"\n{'='*60}")
    print(f"BATCH PROCESSING MODE")
    print(f"Patients to process: {patients}")
    print(f"{'='*60}\n")
    
    patients_to_process = [f'{pat}' for pat in patients]
    pat = patients_to_process[0]
    implants_to_process = []
    runs_to_process = []

# Process files
    
implants = os.listdir('{:s}/{:s}'.format(data_dir, pat))
implants =  sorted([
            f for f in implants
            if 'ses'in f
            and not f.startswith('.')
        ])
   
if single_file_mode:
    # Single file mode - use predefined implants and movies
    implants_to_process = implants_to_process
    runs = movies_to_process

else:
    # Batch mode - get all implants and movies
    implants_to_process = implants
    movies_to_process = []

for imp in implants_to_process:
    
    if data_dir ==f'/{machine_path}/Samsung/AV40_data/new_converted':
        ieeg_dir = '{:s}/{:s}/{:s}/{:s}'.format(data_dir, pat, imp,'ieeg')


    if single_file_mode:
        # Single file mode - use predefined movie
        recordings = movies_to_process
    else:
        # Batch mode - get all movies
        runs = sorted([
            f for f in os.listdir(ieeg_dir)
            if f.endswith(".nwb")
            #if 'AB' in f
            and not f.startswith("._")
        ])
        print(f"\nFound {len(runs)} NWB runs in {ieeg_dir}:")
        for r in runs:
            print("  ", r)

    
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
#%%
    for run in runs:
        # Create standardized file paths for this movie
        file_paths = create_file_paths(
            patient_id=pat_fs,
            implant_id=imp,
            movie_filename=run,
            prep_dir=prep_dir,
            neural_prep_dir=neural_prep_dir,
            hfa_dir=hfa_dir
        )
        

        if single_file_mode:
            target_run = target_run
        else:
            target_run = run
            
        log_base = target_run.replace("_behavior+ecephys.nwb", "")
        pat_log_dir = '{:s}/{:s}/{:s}'.format(log_dir, pat,log_base) 
        
        pat_timing_dir = os.path.join(prep_dir, pat, "timing")
        os.makedirs(pat_timing_dir, exist_ok=True)
        
        # Initialize logger for this movie
        movie_name = file_paths['movie_base']
        logger = ProcessingLogger(pat_fs, movie_name, file_paths['sub_prep_dir'])
        
        # Construct the NWB file path
        if data_dir ==f'/{machine_path}/Samsung/AV40_data/new_converted':
            nwb_fname = '{:s}/{:s}/{:s}/ieeg/{:s}'.format(data_dir, pat, imp,run)

        # proceed
        print(f"\n{'='*60}")
        print(f"PROCESSING FILE:")
        print(f"Patient: {pat_fs}")
        print(f"Implant: {imp}")
        print(f"Run: {run}")
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
        print(elec_ref_table.head()) 

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

        #%%
        
        def zscore_for_plot(x):
            x = np.asarray(x, dtype=float)
            x = x - np.nanmean(x)
            sd = np.nanstd(x)
        
            if sd > 0:
                x = x / sd
        
            return x
        
        # Get the current sampling rate. 
        orig_fs = ecog.info['sfreq']
        
        # -----------------------------
        # Get TTL acquisition
        # -----------------------------
        
        ttlContainer = None
        ttl_container_name = None
        
        for name in ["TTL", "ttl_trigger",'TTL pulses']:
            if name in nwb.acquisition:
                ttlContainer = nwb.acquisition[name]
                ttl_container_name = name
                break
        
        if ttlContainer is None:
            raise KeyError(
                "No TTL acquisition found. "
                f"Available acquisitions: {list(nwb.acquisition.keys())}"
            )
        
        print(f"Using TTL acquisition: {ttl_container_name}")
        
        
        # -----------------------------
        # Get TTL event times
        # -----------------------------
        
        if ttlContainer.timestamps is not None:
        
            # Discrete TTL events with explicit timestamps
            ttls = np.asarray(ttlContainer.timestamps[:]).squeeze()
        
            # Data may contain the TTL/event IDs associated with each timestamp
            ttl_id = np.asarray(ttlContainer.data[:]).squeeze()
        
        else:
        
            # Analog TTL channel sampled continuously
            ana_ttls = np.asarray(ttlContainer.data[:]).squeeze()
        
            ttl_rate = ttlContainer.rate
        
            if ttl_rate is None:
                raise ValueError(
                    f"{ttl_container_name} has neither explicit timestamps "
                    "nor a sampling rate."
                )
        
            from epipe import ana2dig
        
            ttl_id, ttls = ana2dig(
                ana_ttls,
                fs=ttl_rate,
                min_diff=0.4,
                return_time=True
            )
        
        ttls = np.asarray(ttls).squeeze()
        ttl_id = np.asarray(ttl_id).squeeze()
        
        print(f"Found {len(ttls)} TTL events")
        
        # TTL IDs/codes
        
        print("TTL times shape:", ttls.shape)
        print("TTL IDs shape:", ttl_id.shape)
        print("Unique TTL IDs:", np.unique(ttl_id))
        
        # Load audio
        audioContainer = nwb.acquisition.get('audio')
        fs_audio = audioContainer.rate
        audio = audioContainer.data[:]
        t_audio = np.arange(0, audio.shape[0]) / fs_audio

        # Load photo
        photoContainer = None
        
        for photo_name in ["photo", "photodiode"]:
            if photo_name in nwb.acquisition:
                photoContainer = nwb.acquisition.get(photo_name)
                photo_container_name = photo_name
                break
        
        if photoContainer is None:
            print(
                f"No photo/photodiode acquisition found. "
                f"Available acquisitions: {list(nwb.acquisition.keys())}"
            )
        elif photoContainer is not None:
            fs_photo = photoContainer.rate
            photo = photoContainer.data[:]
            t_photo = np.arange(0, photo.shape[0]) / fs_photo
            
            print(f"Loaded photodiode signal from acquisition: {photo_container_name}")
            
        # plt.figure()
        # plt.plot(t_photo,photo, color="blue")
        # plt.title(f"{target_run} photo")
        # plt.show()
        # plt.close()
    
            photo_plot = zscore_for_plot(photo)
            photo_offset = -2
            photo_plot = photo_plot + photo_offset


        audio_plot = zscore_for_plot(audio)
        
        # Vertical offsets so traces do not overlap
        audio_offset = 2
        
        audio_plot = audio_plot + audio_offset
        #%%
                
        # -----------------------------
        # Plot audio + photo + TTLs
        # -----------------------------
        timing_fig_dir = os.path.join(prep_dir, pat, "timing", "figures")
        os.makedirs(timing_fig_dir, exist_ok=True)
        
        fig_path = os.path.join(
            timing_fig_dir,
            f"{log_base}_audio_photo_ttl_timings.png"
        )
        
        plt.figure(figsize=(16, 6))
        
        plt.plot(
            t_audio,
            audio_plot,
            color="blue",
            linewidth=0.8,
            label="audio, z-scored + offset"
        )
        
        if photoContainer is not None:
            plt.plot(
                t_photo,
                photo_plot,
                color="black",
                linewidth=0.8,
                label="photo, z-scored + offset"
            )
        
        # TTL colors by ID
        unique_ttl_ids = np.unique(ttl_id)
        cmap = plt.get_cmap("tab10")
        
        for i, this_id in enumerate(unique_ttl_ids):
            idx = ttl_id == this_id
            ttl_times_this_id = ttls[idx]
        
            plt.vlines(
                ttl_times_this_id,
                ymin=np.nanmin(photo_plot) - 1,
                ymax=np.nanmax(audio_plot) + 1,
                colors=[cmap(i % 10)],
                linewidth=1.0,
                alpha=0.8,
                label=f"TTL id {this_id}"
            )
        
        plt.title(f"{target_run}: audio, photo, and TTL timings")
        plt.xlabel("Time (s)")
        plt.ylabel("Signal amplitude, normalized/offset")
        plt.legend(loc="upper right")
        plt.tight_layout()
        
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        print(f"Saved timing plot: {fig_path}")
        
        plt.show()
        #plt.close()
        
        
        #%%
                # -----------------------------
        # Plot audio + photo + TTLs
        # -----------------------------
        timing_fig_dir = os.path.join(prep_dir, pat, "timing", "figures")
        os.makedirs(timing_fig_dir, exist_ok=True)
        
        fig_path = os.path.join(
            timing_fig_dir,
            f"{log_base}_audio_photo_ttl_timings.png"
        )
        
        plt.figure(figsize=(16, 6))
        
        plt.plot(
            t_audio,
            audio_plot,
            color="blue",
            linewidth=0.8,
            label="audio, z-scored + offset"
        )
        
        if photoContainer is not None:
            plt.plot(
                t_photo,
                photo_plot,
                color="black",
                linewidth=0.8,
                label="photo, z-scored + offset"
            )
        
        # Set plot-wide vertical range
        signals_to_plot = [audio_plot]
        
        if photoContainer is not None:
            signals_to_plot.append(photo_plot)
        
        plot_min = min(np.nanmin(signal) for signal in signals_to_plot) - 1
        plot_max = max(np.nanmax(signal) for signal in signals_to_plot) + 1
        
        # Plot each detected TTL event
        plt.vlines(
            ttls,
            ymin=plot_min,
            ymax=plot_max,
            color="red",
            linewidth=0.8,
            alpha=0.7,
            label="TTL events"
        )
        
        plt.scatter(
            ttls,
            np.full(len(ttls), plot_max),
            marker="v",
            color="red",
            s=15,
            label="TTL events",
            zorder=5
        )
        
        plt.title(f"{target_run}: audio, photo, and TTL timings")
        plt.xlabel("Time (s)")
        plt.ylabel("Signal amplitude, normalized/offset")
        plt.legend(loc="upper right")
        plt.tight_layout()
        
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        print(f"Saved timing plot: {fig_path}")
        
        plt.show()
        #%%
        
        # Load button press times
        
        for button_name in ["button", "button press"]:
            if button_name in nwb.acquisition:
                buttonContainer = nwb.acquisition.get(button_name)
                button_container_name = button_name
                break
        
        
        if buttonContainer is None:
            raise KeyError(
                f"No button press acquisition found. "
                f"Available acquisitions: {list(nwb.acquisition.keys())}"
            )
            
        fs_button = buttonContainer.rate
        button = buttonContainer.data[:]
        t_button = np.arange(0, button.shape[0]) / fs_button
        
        # plt.figure()
        # plt.plot(t_button,button, color="blue")
        # plt.title(f"{target_run} button")
        # plt.show()

        button = np.asarray(button).squeeze()

        threshold = np.mean(button) + 2 * np.std(button)

        # Initial onset detection
        button_binary = button > threshold
        button_onsets = np.where(np.diff(button_binary.astype(int)) == 1)[0] + 1
        
        button_onset_times = t_button[button_onsets]
        button_onset_values = button[button_onsets]
        
        # -----------------------------
        # Debounce button presses
        # -----------------------------
        min_button_interval_s = 0.5  # ignore onsets within 500 ms of previous accepted press
        
        keep_idx = []
        
        last_kept_time = -np.inf
        
        for i, onset_time in enumerate(button_onset_times):
            if onset_time - last_kept_time >= min_button_interval_s:
                keep_idx.append(i)
                last_kept_time = onset_time
        
        keep_idx = np.asarray(keep_idx)
        
        button_onsets_clean = button_onsets[keep_idx]
        button_onset_times_clean = button_onset_times[keep_idx]
        button_onset_values_clean = button_onset_values[keep_idx]
        
        button_onsets_array = np.column_stack([
            button_onsets_clean,
            button_onset_times_clean,
            button_onset_values_clean
        ])
        
        print(f"Raw detected button onsets: {len(button_onset_times)}")
        print(f"Cleaned button presses: {len(button_onset_times_clean)}")
                        
        plt.figure(figsize=(14, 4))
        
        plt.plot(t_button, button, color="blue", label="button signal")
        
        for onset_time in button_onset_times_clean:
            plt.axvline(onset_time, color="red", alpha=0.4)
        
        # plt.title(f"{target_run} button presses")
        # plt.xlabel("Time (s)")
        # plt.ylabel("Button signal")
        # plt.legend()
        # plt.show()
        # plt.close()
                
        button_onsets_df = pd.DataFrame({
            "button_sample": button_onsets_clean,
            "button_time_s": button_onset_times_clean,
            "button_value": button_onset_values_clean
        })
        
        button_onset_csv = os.path.join(
            pat_timing_dir,
            f"{log_base}_button_onsets_clean.csv"
        )
        
        button_onsets_df.to_csv(button_onset_csv, index=False)
        
        print(f"Saved clean button onsets: {button_onset_csv}")
            
   #%%

        log_files = [filename for filename in os.listdir(pat_log_dir) 
            if filename.endswith('.log')
            and not filename.startswith("._")
            ]
        
        log_file = log_files[0]
        log_path= '{:s}/{:s}'.format(pat_log_dir,log_file)
        
        log_df = parse_stimulus_start_times(log_path)
        
        log_df = add_log_cooccurrence_labels(log_df, tolerance_s=0.030)
        
        ttl_df = pd.DataFrame({
            "ttl_time_s": ttls,
            "ttl_id": ttl_id
        })
        
        ttl_df = ttl_df.sort_values("ttl_time_s").reset_index(drop=True)
        
        rough_offset_s = ttl_df["ttl_time_s"].iloc[0] - log_df["time_s"].iloc[0]
        
        print("Rough offset:", rough_offset_s)
        
        matched_df = match_ttls_to_log(
            ttl_df=ttl_df,
            log_df=log_df,
            offset_s=rough_offset_s,
            max_error_s=0.050
        )
        
        print(matched_df.head(20))
        
        print("\nMatched counts:")
        print(matched_df["matched"].value_counts())
        
        print("\nTTL id by matched log event:")
        print(pd.crosstab(
            matched_df["ttl_id"],
            matched_df["matched_log_event_type"]
        ))
        
        print("\nMatch error summary, in ms:")
        print((matched_df["match_error_s"].dropna() * 1000).describe())
        
        plt.figure(figsize=(14, 4))
        
        plt.scatter(
            matched_df["matched_aligned_time_s"],
            matched_df["ttl_time_s"],
            c=matched_df["ttl_id"],
            s=12
        )
        
        plt.plot(
            matched_df["matched_aligned_time_s"],
            matched_df["matched_aligned_time_s"],
            color="black",
            linewidth=1,
            label="perfect alignment"
        )
        
        plt.xlabel("Aligned log event time, s")
        plt.ylabel("TTL time, s")
        plt.title(f"TTL times vs aligned experiment-log event times {target_run}")
        plt.legend()
        plt.tight_layout()
        plt.show()
        #plt.close()
        
        aud_dev_times = matched_df.loc[
            matched_df["matched_stimulus"].eq("audio_deviant"),
            "matched_aligned_time_s"
        ].dropna().values
        
        vis_dev_times = matched_df.loc[
            matched_df["matched_stimulus"].eq("visual_deviant"),
            "matched_aligned_time_s"
        ].dropna().values
        
                
        matched_time_csv = os.path.join(
            pat_timing_dir,
            f"{log_base}_ttl_time_match_summary.csv"
        )
        
        matched_df.to_csv(matched_time_csv, index=False)
        
        #%%
        deviant_fig_dir = os.path.join(prep_dir, pat, "timing", "figures")
        os.makedirs(deviant_fig_dir, exist_ok=True)
        
        deviant_fig_path = os.path.join(
            deviant_fig_dir,
            f"{log_base}_audio_photo_deviants_buttons.png"
        )
        
        plt.savefig(deviant_fig_path, dpi=300, bbox_inches="tight")
        print(f"Saved deviant/button timing plot: {deviant_fig_path}")
        
        plt.figure(figsize=(16, 6))
        
        plt.plot(
            t_audio,
            audio_plot,
            color="blue",
            linewidth=0.8,
            label="audio, z-scored + offset"
        )
        
        # plt.plot(
        #     t_photo,
        #     photo_plot,
        #     color="black",
        #     linewidth=0.8,
        #     label="photo, z-scored + offset"
        # )
        
        # # Auditory deviants: vertical red lines
        for tt in aud_dev_times:
            plt.axvline(
                tt,
                color="red",
                linestyle="--",
                linewidth=1,
                alpha=0.7
            )
        
        # Visual deviants: vertical green lines
        # for tt in vis_dev_times:
        #     plt.axvline(
        #         tt,
        #         color="green",
        #         linestyle="--",
        #         linewidth=1,
        #         alpha=0.7
        #     )
        
        # Visual deviants: vertical green lines
        # Button presses: vertical yellow lines
        for tt in button_onset_times_clean:
            plt.axvline(
                tt,
                color="yellow",
                linestyle="-",
                linewidth=2,
                alpha=0.7
            )
        # # Legend handles
        plt.axvline(
            np.nan,
            color="red",
            linestyle="--",
            linewidth=1,
            label="audio deviant"
        )
        
        # plt.axvline(
        #     np.nan,
        #     color="green",
        #     linestyle="--",
        #     linewidth=1,
        #     label="visual deviant"
        # )
        plt.axvline(
            np.nan,
            color="yellow",
            linestyle="-",
            linewidth=2,
            label="button press deviant"
        )
        
        plt.xlabel("Time (s)")
        plt.ylabel("Signal, z-scored + offset")
        plt.title("Audio and photodiode traces with matched deviants and button presses")
        plt.legend()
        plt.tight_layout()
        
        deviant_fig_dir = os.path.join(prep_dir, pat, "timing", "figures")
        os.makedirs(deviant_fig_dir, exist_ok=True)
        
        deviant_fig_path = os.path.join(
            deviant_fig_dir,
            f"{log_base}_audio_photo_deviants_buttons.png"
        )
        plt.savefig(deviant_fig_path, dpi=300, bbox_inches="tight")
        print(f"Saved deviant/button timing plot: {deviant_fig_path}")
        plt.show()
      #  plt.close()
        
        #%% Compare button presses to auditory deviants to get accuracy
        
        
        # -----------------------------
        # Core timing arrays saved for all conditions
        # -----------------------------
        aud_dev_times_np = np.asarray(aud_dev_times, dtype=float)
        vis_dev_times_np = np.asarray(vis_dev_times, dtype=float)
        
        button_onset_times_clean_np = np.asarray(button_onset_times_clean, dtype=float)
        button_onsets_clean_np = np.asarray(button_onsets_clean, dtype=int)
        button_onset_values_clean_np = np.asarray(button_onset_values_clean, dtype=float)
        
        ttls_np = np.asarray(ttls, dtype=float)
        ttl_id_np = np.asarray(ttl_id)
        
        # Choose output directory
        timing_out_dir = os.path.join(prep_dir, pat, "timing")
        os.makedirs(timing_out_dir, exist_ok=True)
                
        log_base_upper = log_base.upper()
        
        if "AV40_AA" in log_base_upper:
            cond = "AA"
        elif "AV40_AV" in log_base_upper:
            cond = "AV"
        elif "AV40_AB" in log_base_upper:
            cond = "AB"
        else:
            cond = None
            
        if cond == "AA":
            dev_times = np.asarray(aud_dev_times, dtype=float)
            cond_name = "auditory"
        elif cond == "AV":
            dev_times = np.asarray(vis_dev_times, dtype=float)
            cond_name = "visual"
        elif cond == "AB":
            timing_npz = os.path.join(
                timing_out_dir,
                f"{log_base}_final_timing_info_AB_no_behavior_classification.npz"
            )
            
            np.savez_compressed(
                timing_npz,
            
                # Metadata-like scalar values
                log_base=log_base,
                condition=cond,
                target_modality="auditory_and_visual",
                fs_button=fs_button,
                fs_audio=fs_audio,
                fs_photo=fs_photo,
            
                # TTL/log timing
                ttl_times_s=ttls_np,
                ttl_ids=ttl_id_np,
                aud_dev_times_s=aud_dev_times_np,
                vis_dev_times_s=vis_dev_times_np,
            
                # Button timing
                button_onsets_clean_samples=button_onsets_clean_np,
                button_onset_times_clean_s=button_onset_times_clean_np,
                button_onset_values_clean=button_onset_values_clean_np
            )
            
        #     print(f"Saved AB timing info without behavioral classification: {timing_npz}")
                        
        #     print(f"[SKIP] AB condition has no single target modality for button accuracy: {log_base}")
        #     continue
        # else:
        #     print(f"[SKIP] Could not determine condition from {log_base}")
        #     continue
        
        
        button_onset_times = np.asarray(button_onset_times_clean, dtype=float)
        
        # Remove NaNs and sort
        dev_times = np.sort(dev_times[~np.isnan(dev_times)])
        button_onset_times = np.sort(button_onset_times[~np.isnan(button_onset_times)])
        
        response_window_s = 1.5
        
        matched_buttons = []
        used_button_indices = set()
        
        for dev_i, dev_time in enumerate(dev_times):
            
            candidate_idx = np.where(
                (button_onset_times >= dev_time) &
                (button_onset_times <= dev_time + response_window_s)
            )[0]
            
            # Only allow unused buttons
            candidate_idx = [idx for idx in candidate_idx if idx not in used_button_indices]
            
            if len(candidate_idx) > 0:
                button_idx = candidate_idx[0]
                button_time = button_onset_times[button_idx]
                used_button_indices.add(button_idx)
                
                matched_buttons.append({
                    "condition": cond,
                    "target_modality": cond_name,
                    f"{cond_name}_dev_time": dev_time,
                    "dev_time": dev_time,
                    "button_time": button_time,
                    "response_latency_s": button_time - dev_time,
                    "has_button_response": True,
                    "button_index": button_idx
                    })        

                
            else:
                matched_buttons.append({
                    "condition": cond,
                    "target_modality": cond_name,
                    f"{cond_name}_dev_time": dev_time,
                    "dev_time": dev_time,
                    "button_time": np.nan,
                    "response_latency_s": np.nan,
                    "has_button_response": False,
                    "button_index": np.nan
                })
        
        button_match_df = pd.DataFrame(matched_buttons)
        
        button_match_csv = os.path.join(
            pat_timing_dir,
            f"{log_base}_button_match_summary.csv"
        )
        
        button_match_df.to_csv(button_match_csv, index=False)
        
        # Accuracy / hit rate
        n_deviants = len(dev_times)
        n_hits = int(button_match_df["has_button_response"].sum())
        hit_rate = n_hits / n_deviants if n_deviants > 0 else np.nan
        
        # False alarms: buttons that were not assigned as hits
        button_is_hit = np.zeros(len(button_onset_times), dtype=bool)
        button_is_hit[list(used_button_indices)] = True
        
        false_alarm_times = button_onset_times[~button_is_hit]
        
        false_alarm_df = pd.DataFrame({
            "button_time": false_alarm_times
        })
        
        n_buttons = len(button_onset_times)
        n_false_alarms = len(false_alarm_times)
        
        print(button_match_df)
        print(f"{cond_name} deviants: {n_deviants}")
        print(f"Hits within {response_window_s} s: {n_hits}")
        print(f"Hit rate: {hit_rate:.3f}")
        print(f"Total button presses: {n_buttons}")
        print(f"False alarms: {n_false_alarms}")
        print(false_alarm_df)
        
        
        false_alarm_csv = os.path.join(
            pat_timing_dir,
            f"{log_base}_false_alarm_summary.csv"
        )
        
        false_alarm_df.to_csv(false_alarm_csv, index=False)

        accuracy_log = pd.DataFrame([{
            "log_base": log_base,
            "condition": cond,
            "target_modality": cond_name,
            "n_deviants": n_deviants,
            "response_window_s": response_window_s,
            "n_hits": n_hits,
            "hit_rate": hit_rate,
            "n_button_presses": len(button_onset_times),
            "n_false_alarms": len(false_alarm_times),
            "mean_rt_s": button_match_df["response_latency_s"].mean(),
            "median_rt_s": button_match_df["response_latency_s"].median()
        }])
        
        out_csv = os.path.join(
            pat_timing_dir,
            f"{log_base}_button_accuracy_summary.csv"
        )
        
        accuracy_log.to_csv(out_csv, index=False)

        print(f"Saved accuracy log: {out_csv}")
        print(accuracy_log)

        #%% Save final timing info as compressed NumPy archive
        
        # Hit and miss deviant times
        hit_idx = button_match_df["has_button_response"] == True
        miss_idx = button_match_df["has_button_response"] == False
        
        hit_dev_times = button_match_df.loc[hit_idx, "dev_time"].to_numpy(dtype=float)
        miss_dev_times = button_match_df.loc[miss_idx, "dev_time"].to_numpy(dtype=float)
        
        hit_button_times = button_match_df.loc[hit_idx, "button_time"].to_numpy(dtype=float)
        hit_response_latencies = button_match_df.loc[hit_idx, "response_latency_s"].to_numpy(dtype=float)
        
        target_dev_times_np = np.asarray(dev_times, dtype=float)
        false_alarm_times_np = np.asarray(false_alarm_times, dtype=float)
        
        
        # Choose output directory
        timing_out_dir = os.path.join(prep_dir, pat, "timing")
        os.makedirs(timing_out_dir, exist_ok=True)
        
        timing_npz = os.path.join(
            timing_out_dir,
            f"{log_base}_final_timing_info.npz"
        )
        
        np.savez_compressed(
            timing_npz,
        
            # Metadata-like scalar values
            log_base=log_base,
            condition=cond,
            target_modality=cond_name,
            response_window_s=response_window_s,
            fs_button=fs_button,
            fs_audio=fs_audio,
            fs_photo=fs_photo,
        
            # TTL/log timing
            ttl_times_s=ttls_np,
            ttl_ids=ttl_id_np,
            aud_dev_times_s=aud_dev_times_np,
            vis_dev_times_s=vis_dev_times_np,
            target_dev_times_s=target_dev_times_np,
        
            # Button timing
            button_onsets_clean_samples=button_onsets_clean_np,
            button_onset_times_clean_s=button_onset_times_clean_np,
            button_onset_values_clean=button_onset_values_clean_np,
        
            # Behavioral classification
            hit_dev_times_s=hit_dev_times,
            miss_dev_times_s=miss_dev_times,
            hit_button_times_s=hit_button_times,
            hit_response_latencies_s=hit_response_latencies,
            false_alarm_times_s=false_alarm_times_np,
        
            # Accuracy summary
            n_deviants=n_deviants,
            n_hits=n_hits,
            hit_rate=hit_rate,
            n_button_presses=n_buttons,
            n_false_alarms=n_false_alarms
        )
        
        print(f"Saved final timing info: {timing_npz}")
        
