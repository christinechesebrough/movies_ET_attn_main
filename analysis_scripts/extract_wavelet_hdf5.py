#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 20:37:57 2026

@author: christine
"""
 
import os, re, sys
from scipy.stats import pearsonr
from itertools import compress
import numpy as np
import pandas as pd
from scipy import stats, signal, interpolate
import matplotlib.pyplot as plt
import mne
import seaborn as sns
from mne.time_frequency import psd_array_welch
from mne.time_frequency import tfr_array_morlet, tfr_array_multitaper
import multiprocessing


#from antropy import sample_entropy, spectral_entropy, perm_entropy, lziv_complexity

# Add Linux library paths BEFORE importing epipe
# sys.path.insert(0, '/media/christine/Samsung/EPIPE/Python')
# sys.path.insert(0, '/media/christine/Samsung/iEEG2NWB-main')
machine_path = 'media/christine'

sys.path.insert(0, f'/{machine_path}/Samsung/EPIPE-movie_nwb/Python')
sys.path.insert(0, f'/{machine_path}/Samsung/iEEG2NWB-main')

#vids = ['inscapes','despicable_me_english']#,'despicable_me_english']
vids = ['inscapes']#,'despicable_me_english']#,'despicable_me_english']

drive = 'Samsung'
ref = 'avg'
#freq_band = 'HFA'
region = 'all'

data_dir = f'/{machine_path}/Samsung/Movie_data/movies_prep_standard'#f'/{machine_path}/Samsung/Movie_data/movies_prep_standard'
isc_dir = f'/{machine_path}/Samsung/Movie_data/data/isc'
mne_data_dir = f'/{machine_path}/Samsung/Movie_data/movies_prep_standard' #f'/{machine_path}/Samsung/Movie_data/movies_prep_standard'
elec_dir = f'/{machine_path}/Samsung/Movie_data/data/electrode_localization'
fs_dir = f'/{machine_path}/Data/anatomy'

corr_dir = f'/{machine_path}/Samsung/Movie_data/data/movie_elec_corr_sheets'


output = 'wavelet'
pow_type = 'raw'

visualize_mne_steps = False

tf_freq_range = (1, 150)   # broad spectrum to preserve
freq_step = 2             # or 1 if you want denser sampling
decim_tf = 4               # 2 is fine, 4 may be more practical
n_cycles_mode = 'fixed'   # 'scaled' or 'fixed'
fixed_n_cycles = 5

save_continuous_tf = False

# window_length_sec = 10
# overlap_sec = 7.5
# target_num_steps = 236

wd = '/Volumes/Samsung/scripts/movies_ET_attn_main'
src_dir = os.path.join(wd, 'src')
src_dir = os.path.abspath(src_dir)

# Add src to path if not already there
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)


# 1) Put src on sys.path (at the front)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
print('on sys.path?', src_dir in sys.path)

from epipe import nwb2mne, inspectNwb
from epipe import inspectNwb, nwb2mne, read_ielvis
import sys

sys.path.insert(0, '/media/christine/Samsung/scripts/movies_ET_attn_main/src')
# Import EEG preprocessing helper functions
from eeg_preproc_helpers import (
    plot_power_spectra,  plot_psd_batched, 
    create_file_paths, apply_highpass_filter,
    save_bad_channels, load_bad_channels, check_bad_channels_integrity, load_preprocessed_data, validate_mne_structure, preserve_mne_structure, restore_mne_structure, load_data_with_bad_channels, summarize_bad_channels, 
    interpolate_spikes, make_groups_from_prefix, regress_out_noise_by_group,
    detect_spikes_all_channels, reref_avg_by_group,
    ProcessingLogger, detect_spikes_ref1)

###### EXTRACT NORMALIZED AND NON-NORMALIZED VALUES FOR EACH EYE MOVEMENT FOR EACH WINDOW?????
#### AND TRY TO FIGURE OUT WHICH EYE MOVEMENT FEATURES ARE THE BEST PREDICTORS / FIT

#%%
network_mapping = {
     "17Networks_1": "Visual Central (Visual A)",
     "17Networks_2": "Visual Peripheral (Visual B)",
     "17Networks_3": "Somatomotor A",
     "17Networks_4": "Somatomotor B",
     "17Networks_5": "Dorsal Attention A",
     "17Networks_6": "Dorsal Attention B",
     "17Networks_7": "Salience / Ventral Attention A",
     "17Networks_8": "Salience / Ventral Attention B",
     "17Networks_9": "Limbic A",
     "17Networks_10": "Limbic B",
     "17Networks_11": "Control C",
     "17Networks_12": "Control A",
     "17Networks_13": "Control B",
     "17Networks_14": "Temporal Parietal",
     "17Networks_15": "Default C",
     "17Networks_16": "Default A",
     "17Networks_17": "Default B"
 }

def _extract_run_label(fname: str) -> str:
    """
    Try to extract a run label like 'run-01' or 'run-1' from a filename.
    Fallback: 'run-01'.
    """
    m = re.search(r'run[-_]?(\d+)', fname, flags=re.IGNORECASE)
    if m:
        return f"run-{int(m.group(1)):02d}"
    return "run-01"

def _extract_ses_label(fname: str) -> str:
    """
    Optional: extract 'ses-02' etc. Fallback: ''.
    """
    m = re.search(r'ses[-_]?(\d+)', fname, flags=re.IGNORECASE)
    if m:
        return f"ses-{int(m.group(1)):02d}"
    return ""

def _sort_key(f):
    ses = _extract_ses_label(f)
    run = _extract_run_label(f)
    return (ses, run, f)



#%% Main script
vids.sort()

processed_lfp_files = []  # initialize once

for vid in vids:
    
    if vid == 'despicable_me_english':
        #good_ET_pats = ["NS190"]#['NS140_02','NS153','NS164','NS166','NS174_02']#['NS127_02','NS135','NS136','NS137','NS138','NS140','NS140_02'] #'NS153','NS164','NS166','NS174_02',"NS178","NS190","NS191"]#,'NS193',"NS194","NS201",'NS205']
       # good_ET_pats = ['NS193',"NS194","NS201",'NS205']
       # good_ET_pats = ['NS127_02','NS135','NS136','NS137','NS138','NS140','NS151','NS153','NS154','NS155_02','NS164','NS174_02','NS174_03','NS190','NS191','NS193','NS194','NS178','NS201_02','NS204','NS205']
      # good_ET_pats = ['NS193'] 
       patients = [
         'NS127_02',
         'NS135',
         'NS136',
         'NS137',
         'NS138',
         'NS140',
         'NS140_02',
         'NS153',
         'NS154',
         'NS155_02',
         'NS164',
         'NS174_02',
         'NS174_03',
         'NS178',
         'NS191',
         'NS193',
         'NS194',
         'NS201_02',
         'NS205']
       


    elif vid == 'inscapes':
        patients = [
         'NS127_02',
         'NS135',
         'NS136',
         'NS137',
         'NS138',
         'NS140',
         'NS140_02',
         'NS153',
         'NS155_02',
         'NS164',
         'NS178',
         'NS205',
         'NS210']
            
    elif vid == 'despicable_me_hungarian':
          patients= [
      #     'LH010',
      #     'NS127_02',
      # #    'NS135',
      #     'NS136',
      #     'NS137',
      #     'NS138',
      #     'NS140',
      #     'NS140_02',
      #      'NS142', 
      #      'NS144',
      #      'NS145',
      #      'NS154',
      #      'NS155_02',
      #      'NS164',
      #     'NS174_02',
          'NS174_03',
           'NS178',
           'NS211'
            ]
        
    patients.sort()
        
    if vid == 'despicable_me_english':
        keys = ['despicable_me_english','dme']
    if vid == 'despicable_me_hungarian':
        keys = ['despicable_me_hungarian','dmh']
    if vid == 'inscapes':
        keys = ['inscapes']
    if vid == 'the_present':
        keys = ['present','the_present']

    print(f"Processing data for {vid}")

    fig_dir = f'/{machine_path}/Data/Movie_data/{output}_{vid}_all_cortContacts_tf_10s_{tf_freq_range}_26Aug26'
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
      #
            #%
    for pat in patients:
        pat_dir = os.path.join(data_dir, pat)
        fig_patient_dir = os.path.join(fig_dir, pat)
        if not os.path.exists(fig_patient_dir):
            os.makedirs(fig_patient_dir)
                    
        lfp_pat_dir = '{:s}/{:s}/Neural_prep'.format(mne_data_dir, pat)
        
        lfp_files = os.listdir(lfp_pat_dir)
      
        lfp_files = [
            f for f in os.listdir(lfp_pat_dir)
            if f.endswith(".fif")
            and ref in f
            and ('referenced' in f)
            and 'aic' not in f
            and any(k.lower() in f.lower() for k in keys)
            and not f.startswith("._")
        ]
  
          # Deterministic sort: by session then run then filename
        
        lfp_files = sorted(lfp_files, key=_sort_key)
        
        print(f"Found {len(lfp_files)} matching runs for {pat}:")
        for f in lfp_files:
            print("  ", f)
    
            processed_lfp_files.append(f)

        # # Iterate over each file/run as an independent entry
        for lfp_file in lfp_files:
            # Build an entry label that will propagate to outputs
            ses_label = _extract_ses_label(lfp_file)
            run_label = _extract_run_label(lfp_file)
            if ses_label:
                entry_id = f"{pat}_{ses_label}_{run_label}"
            else:
                entry_id = f"{pat}_{run_label}"

            print(f"Loading data for entry {entry_id} from {lfp_file} ...")
            
            if run_label == 'run-01':
                run_keys = ['run-01','run-1']
            if run_label == 'run-02':
                run_keys = ['run-02','run-2']

            mne_data = mne.io.read_raw(os.path.join(lfp_pat_dir, lfp_file), preload=False)
            
            sub_fs_dir = '{:s}/{:s}'.format(fs_dir, pat)
              
            subid = os.path.basename(sub_fs_dir)

            elec_recon_dir = corr_dir    

            excel_files = sorted([
                f for f in os.listdir(elec_recon_dir)
                if pat in f
                and f.endswith('.xlsx')
                and not f.startswith('.')
            ])
            if not excel_files:
                print(f"  [SKIP] No correspondence .xlsx for {pat} found in {elec_recon_dir}")
                continue

            # Use most recently modified if multiple exist (matches Python HFO script)
            excel_path = max(
                [os.path.join(elec_recon_dir, f) for f in excel_files],
                key=os.path.getmtime
            )
            print(f"  Using: {os.path.basename(excel_path)}")
            elec_ref_table = pd.read_excel(excel_path)
            
            elecs_subs =pd.read_excel(excel_path)
            required_cols = {"label"}
            col_map = {col.lower(): col for col in elecs_subs.columns}
            elecs_subs.rename(columns={col_map[req]: req for req in required_cols}, inplace=True)

            
            # Find candidate files already in the patient dir
            bad_channel_files = [
                f for f in os.listdir(lfp_pat_dir)
                if (
                    ('bad_channels' in f)
                    and f.endswith('.txt')
                    and any(k.lower() in f.lower() for k in keys)
                    and not f.startswith('._')
                )
            ]
            
            # Narrow to candidates matching ref + run
            ref_bad_channel_files = [
                f for f in bad_channel_files
                if (
                    (ref in f)
                    and any(r in f for r in run_keys)
                )
            ]
            
            # Helper: pick most recent if multiple
            def pick_most_recent(files):
                if not files:
                    return None
                return max(files, key=lambda fn: os.path.getmtime(os.path.join(lfp_pat_dir, fn)))
            
            picked = pick_most_recent(ref_bad_channel_files)
            
            # If none found, define a NEW file name we will create
            # Make sure this naming is unique enough for your workflow
            if picked is None:
                # You can include ses_label/run_label if you want; run_keys might be list like ["run-01", "run-1"]
                # Use run_label if you have a single normalized run label available.
                picked = f'{pat}_{ses_label}_{vid}_{run_label}_{ref}_bad_channels.txt'

            bad_channel_path = os.path.join(lfp_pat_dir, picked)
            
            # Read existing file (if any)
            #    Preserve header/comments, parse channel lines robustly.
                           # Read existing file (if any)
            header_lines = []
            existing_bads = []
            
            if os.path.exists(bad_channel_path):
                with open(bad_channel_path, "r") as f:
                    for ln in f:
                        s = ln.strip()
                        if not s:
                            continue
                        if s.startswith("#"):
                            header_lines.append(ln.rstrip("\n"))
                            continue
                        existing_bads.append(s)
            
            # Safely get FIF bads (empty list if none)
            fif_bads = list(mne_data.info.get("bads", []))
            
            # --- NEW: filter to only channels that exist in this Raw ---
            ch_set = set(mne_data.ch_names)  # or mne_data.info["ch_names"]
            
            existing_bads_valid = [ch for ch in existing_bads if ch in ch_set]
            existing_bads_missing = [ch for ch in existing_bads if ch not in ch_set]
            
            # Optional: log missing bads once (useful for debugging)
            if existing_bads_missing:
                print(
                    f"Warning: {len(existing_bads_missing)} bad channels from file are not in this recording and will be ignored. "
                    f"Examples: {existing_bads_missing[:10]}"
                )
            
            # Merge (dedupe, preserve order) using only valid names
            merged_bads = list(dict.fromkeys(fif_bads + existing_bads_valid))
            
            # Assign back
            mne_data.info["bads"] = merged_bads

            # Interactive marking

            if visualize_mne_steps:
                mne_data.plot(
                    scalings=dict(seeg=200e-6),
                    n_channels=32,
                    remove_dc=True,
                    show_scrollbars=True,
                    duration=12.0,
                    block=True
                )
            
            # Save bads after closing plot
            final_bads = list(dict.fromkeys(mne_data.info.get("bads", [])))  # dedupe again
            
            os.makedirs(lfp_pat_dir, exist_ok=True)
            
            tmp_path = bad_channel_path + ".tmp"
            with open(tmp_path, "w") as f:
                # If file had no header, write a minimal one
                if not header_lines:
                    f.write(f"# Bad channels for {pat} {vid} {run_label} {ref}\n")
                else:
                    f.write("\n".join(header_lines) + "\n")
            
                # Write channel names, one per line
                if final_bads:
                    f.write("\n".join(final_bads) + "\n")
            
            os.replace(tmp_path, bad_channel_path)
            print(f"Bad channels saved to: {bad_channel_path}")
            
            # Drop them for downstream processing

            if final_bads:
                mne_data.drop_channels(final_bads)
            
                            
            labels = mne_data.ch_names
                                    
            exclude_strings = ['bankssts']
        
            #ip_contacts = elecs_subs.Contact.values
            ip_contacts = elecs_subs.label.values

                     
           # ip_contacts = elecs_subs.Contact.values[(elecs_subs.AparcAseg_Atlas != 'Right-Cerebral-White-Matter') & (elecs_subs.AparcAseg_Atlas != 'Left-Cerebral-White-Matter')]
            # ip_contacts = elecs_subs.loc[elecs_subs['DK_Lobe'].isin(['Right-Hippocampus', 'Left-Hippocampus']),
            #     'Contact'
            # ].values
            if len(ip_contacts) == 0:
                pass
            else:
                if ref == 'wm_bip':
                    labels_split = [l.split('-') for l in labels]
                    idx_ip = np.unique(np.concatenate([np.where([np.sum([ld == ls for ls in lf]) for lf in labels_split])[0] for ld in ip_contacts]))
                    idx_ip = np.in1d(np.arange(len(labels)), idx_ip)
                if ref in ['avg','wm']:
                    idx_ip = np.array([label in ip_contacts for label in labels])
        
                lfp = mne_data.get_data()
                fs_lfp = mne_data.info['sfreq']
            #    time_lfp = mne_data.times
                    
                lfp_ip = lfp[idx_ip, :]
                    
                labels_ip = list(compress(labels, idx_ip))


                f_start, f_end = tf_freq_range
                freqs_tf = np.arange(f_start, f_end + freq_step, freq_step)
                
                # Remove flat channels before processing (std < threshold)
                flat_std_thresh = 1e-6
                stds = np.std(lfp_ip, axis=1)
                nonflat_idx = stds > flat_std_thresh
                lfp_ip = lfp_ip[nonflat_idx]
                labels_ip = [label for i, label in enumerate(labels_ip) if nonflat_idx[i]]
                
                if lfp_ip.shape[0] == 0:
                    print(f"[SKIP] No non-flat intracranial channels remain for {entry_id}")
                    continue

                             
                import h5py
                
                # ----------------------------------------------------------
                # Wavelet settings
                # ----------------------------------------------------------
                
                wavelet_batch_size = 16
                n_channels = lfp_ip.shape[0]
                
                if n_cycles_mode == 'scaled':
                    n_cycles = freqs_tf / 2.0
                else:
                    n_cycles = fixed_n_cycles
                
                # No downsampling
                fs_tf = fs_lfp
                
                # Number of samples in the original recording
                n_times = lfp_ip.shape[1]
                
                # ----------------------------------------------------------
                # Output file
                # ----------------------------------------------------------
                
                tf_save_path = os.path.join(
                    fig_patient_dir,
                    f'{entry_id}_{vid}_{output}_raw_tf.h5'
                )
                
                print(f"Creating HDF5 file: {tf_save_path}")
                
                # ----------------------------------------------------------
                # Create file and datasets
                # ----------------------------------------------------------
                
                with h5py.File(tf_save_path, "w") as h5f:
                
                    # -----------------------------
                    # Metadata
                    # -----------------------------
                
                    h5f.attrs["pat"] = pat
                    h5f.attrs["run_label"] = run_label
                    h5f.attrs["vid"] = vid
                    h5f.attrs["output"] = output
                    h5f.attrs["pow_type"] = "raw"
                    h5f.attrs["fs_lfp"] = fs_lfp
                    h5f.attrs["fs_tf"] = fs_tf
                    h5f.attrs["n_cycles_mode"] = n_cycles_mode
                
                    if np.isscalar(n_cycles):
                        h5f.attrs["n_cycles"] = float(n_cycles)
                    else:
                        h5f.create_dataset(
                            "n_cycles",
                            data=np.asarray(n_cycles, dtype=np.float32)
                        )
                
                    # -----------------------------
                    # Channel and frequency info
                    # -----------------------------
                
                    h5f.create_dataset("labels_ip",
                        data=np.asarray(labels_ip, dtype="S")
                    )
                
                    h5f.create_dataset(
                        "freqs_tf",
                        data=np.asarray(freqs_tf, dtype=np.float32)
                    )
                
                    # -----------------------------
                    # Continuous time vector
                    # -----------------------------
                
                    t_tf = np.arange(
                        n_times,
                        dtype=np.float32
                    ) / fs_tf
                
                    h5f.create_dataset(
                        "t_tf",
                        data=t_tf
                    )
                
                    # ------------------------------------------------------
                    # Main TF dataset ON DISK
                    #
                    # shape:
                    # channels × frequencies × time
                    # ------------------------------------------------------
                
                    tf_dataset = h5f.create_dataset(
                        "pow_tf_dat",
                        shape=(
                            n_channels,
                            len(freqs_tf),
                            n_times
                        ),
                        dtype=np.float32,
                
                        # Chunked storage
                        chunks=(
                            1,
                            len(freqs_tf),
                            min(int(fs_tf * 10), n_times)
                        ),
                
                        compression="gzip",
                        compression_opts=4,
                        shuffle=True
                    )
                
                    print(
                        "Dataset shape:",
                        tf_dataset.shape
                    )
                
                    # ------------------------------------------------------
                    # Calculate + write each channel batch
                    # ------------------------------------------------------
                
                    for batch_start in range(
                        0,
                        n_channels,
                        wavelet_batch_size
                    ):
                
                        batch_end = min(
                            batch_start + wavelet_batch_size,
                            n_channels
                        )
                
                        print(
                            f"Processing channels "
                            f"{batch_start + 1}-{batch_end} "
                            f"of {n_channels}"
                        )
                
                        data_tf = lfp_ip[
                            batch_start:batch_end,
                            :
                        ][np.newaxis, :, :]
                
                        if output == 'wavelet':
                
                            power_batch = tfr_array_morlet(
                                data_tf,
                                sfreq=fs_lfp,
                                freqs=freqs_tf,
                                n_cycles=n_cycles,
                                output='power',
                                n_jobs=1
                            )
                
                        elif output == 'multitaper':
                
                            power_batch = tfr_array_multitaper(
                                data_tf,
                                sfreq=fs_lfp,
                                freqs=freqs_tf,
                                n_cycles=n_cycles,
                                output='power',
                                time_bandwidth=4.0,
                                n_jobs=1
                            )
                
                        else:
                
                            raise ValueError(
                                f"Unknown output type: {output}"
                            )
                
                        # Remove epoch dimension
                        power_batch = power_batch[0]
                
                        # Preserve LINEAR power
                        power_batch = power_batch.astype(
                            np.float32,
                            copy=False
                        )
                
                        # --------------------------------------------------
                        # Write directly into appropriate disk location
                        # --------------------------------------------------
                
                        tf_dataset[
                            batch_start:batch_end,
                            :,
                            :
                        ] = power_batch
                
                        # Make sure data has been written
                        h5f.flush()
                
                        print(
                            f"  Saved channels "
                            f"{batch_start + 1}-{batch_end}"
                        )
                
                        del data_tf, power_batch
                
                print(f"Finished TF file: {tf_save_path}")
    
                #
                # tf_save_path = os.path.join(
                #     fig_patient_dir,
                #     f'{entry_id}_{vid}_{output}_{pow_type}_tf.npz'
                # )
                                
                # save_dict = {
                #     'labels_ip': np.array(labels_ip, dtype=object),
                #     'pat': pat,
                #     'run_label': run_label,
                #     'vid': vid,
                #     'fs_lfp': fs_lfp,
                #     'fs_tf': fs_tf,
                #     'freqs_tf': freqs_tf.astype(np.float32),
                #     'n_cycles': np.asarray(n_cycles),
                #     'pow_type': pow_type,
                #     'pow_tf_dat': pow_tf_dat,
                #     't_tf': t_tf
                # }
                # np.savez_compressed(tf_save_path, **save_dict)
                # print(f"Saved TF data to: {tf_save_path}")
                
      
                    #%%

                # Initialize lists for DK_Atlas, Y7_Atlas, and AparcAseg_Atlas regions
                dk_regions = []
                y7_regions = []
                y17_regions = []
                aparc_aseg_regions = []
                
                for label in labels_ip:
                    match_row = elecs_subs[elecs_subs['label'] == label]
                
                    if not match_row.empty:
                        dk_col = 'desikan_killiany' if 'desikan_killiany' in elecs_subs.columns else 'Desikan_Killiany'
                        y7_col = 'y7_atlas' if 'y7_atlas' in elecs_subs.columns else 'Yeo7'
                        y17_col = 'y17_atlas' if 'y17_atlas' in elecs_subs.columns else 'Yeo17'
                        aparc_col = 'aparc_aseg' if 'aparc_aseg' in elecs_subs.columns else 'aparc_aseg'
                
                        dk_regions.append(match_row[dk_col].values[0])
                        y7_regions.append(match_row[y7_col].values[0])
                        y17_regions.append(match_row[y17_col].values[0])
                        aparc_aseg_regions.append(match_row[aparc_col].values[0])
                    else:
                        dk_regions.append('Unknown')
                        y7_regions.append('Unknown')
                        y17_regions.append('Unknown')
                        aparc_aseg_regions.append('Unknown')
                
                channel_meta = pd.DataFrame({
                    'label': labels_ip,
                    'DK_Atlas': dk_regions,
                    'Y7_Atlas': y7_regions,
                    'Y17_Atlas': y17_regions,
                    'AparcAseg_Atlas': aparc_aseg_regions
                })
                
                meta_csv = os.path.join(fig_patient_dir, f'{entry_id}_{vid}_channel_metadata.csv')
                channel_meta.to_csv(meta_csv, index=False)
                
                
            