#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 21:31:45 2026

@author: christine
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
## Simplified script for loading epoched attention state data, extracting analytic power using hilbert transform, and saving
"""
import os, re, sys    

from itertools import compress
import numpy as np
import pandas as pd
from scipy import stats, signal, interpolate
import matplotlib.pyplot as plt
import mne
import seaborn as sns
from mne.time_frequency import psd_array_welch

from fooof.plts.spectra import plot_spectrum



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


vid = 'despicable_me_english' #'inscapes'
ref = 'avg'
pow_type = 'log'

if vid == 'despicable_me_hungarian':
    patients= [
  #   'LH010',
   #  'NS127_02',
    # 'NS135',
    # 'NS136',
    # 'NS137',
    # 'NS138',
     #  'NS140',
     #  'NS140_02',
     # 'NS142', 
     # 'NS144',
     # 'NS145',
     # 'NS154',
     # 'NS155_02',
     # 'NS164',
    # 'NS174_02',
    # 'NS174_03',
     # 'NS178',
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
      #151

if vid == 'despicable_me_english':

     patients = [
     # 'NS127_02',
    # 'NS135',
    # 'NS136',
    # 'NS137',
    # 'NS138',
     # 'NS140',
   #  'NS140_02'
     # 'NS154',
     # 'NS155_02',
     # 'NS164',
     # 'NS174_02',
     # 'NS174_03',
     # 'NS178',
    # 'NS191',
    # 'NS193',
    # 'NS194',
    # 'NS201_02',
    # 'NS205'
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
       #'NS170',
       # 'NS190'
       #NS204

if vid == 'inscapes':
    patients = [
   # 'NS127_02',
   # 'NS135',
   #  'NS136',
   #  'NS137',
   #  'NS138',
   #  'NS140',
   #  'NS140_02',
   # 'NS153',
   # 'NS155_02',
   #  'NS164',
   #  'NS178',
   #  'NS205',
   #  'NS210'
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

vid_keys = {
    'betta': ['betta', 'Betta'],
    'inscapes': ['inscapes', 'Inscapes'],
    'despicable_me_english': ['despicable_me_english','dme'],
    'despicable_me_hungarian': ['despicable_me_hungarian','dmh'],    
}

machine_path = '/media/christine'

for pat in patients:
    pat_prep_dir = f'{machine_path}/Samsung/Movie_data/movies_prep_standard/{pat}/Neural_prep' #path to directory where the recordings are
    # epoch_dat_path = os.path.join(
    #     mne_data_dir,
    #     f"{pat}_ses-Exp_Samp_Betta01_behavior+ecephys_referenced_avg_probe_pre_onset-epo.fif"
    # )
    
    #epoch_dat_path =
    #'/media/christine/Samsung/Movie_data/exp_samp_prep_standard/LH019/Neural_prep/LH019_ses-Exp_Samp_Betta_combined_behavior+ecephys_referenced_avg_probe_pre_onset-epo.fif'
    preprocessed_files = os.listdir(pat_prep_dir)
    
    if 'vid' in locals():
        keys = vid_keys.get(vid, [vid])
        
                
    # filtered_movies = [
    #     f for f in movies_list
    #     if f.endswith(".nwb")
    #     and any(k.lower() in f.lower() for k in keys)
    #     and not f.startswith("._")
    # ]
    
    preprocessed_files = sorted([
        f for f in preprocessed_files
        if any(k.lower() in f.lower() for k in keys)
        and f.endswith(".fif")
        and "referenced" in f
        and ref in f
        and not f.startswith(".")
        #and ("prep_ieeg" in f or "preprocessed_ieeg" in f)
    ])
    
    if len(preprocessed_files) > 1:
        print(f"Multiple possible preprocessed FIF files found: {preprocessed_files}")
    else:
        preprocessed_file = preprocessed_files[0]
    
    for preprocessed_file in preprocessed_files:
        
        preprocessed_path = os.path.join(pat_prep_dir, preprocessed_file)
        
        out_dir = f'{machine_path}/Samsung/Movie_data/movies_bad_windows/{pat}' # path to your output directory
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        
        # Load preprocessed recording
        lfp_dat = mne.io.read_raw_fif(preprocessed_path, preload=True)
        labels = lfp_dat.ch_names
        
        ses_label = _extract_ses_label(preprocessed_file)
        run_label = _extract_run_label(preprocessed_file)
        if ses_label:
            entry_id = f"{pat}_{ses_label}_{run_label}"
        else:
            entry_id = f"{pat}_{run_label}"
    
        print(f"Loading data for entry {entry_id} from {preprocessed_file} ...")
        
        if run_label == 'run-01':
            run_keys = ['run-01','run-1']
        if run_label == 'run-02':
            run_keys = ['run-02','run-2']
        
        # Find candidate files already in the patient dir
        bad_channel_files = [
            f for f in os.listdir(pat_prep_dir)
            if (
                ('bad_channels' in f)
                and f.endswith('.txt')
                and vid in f
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
            return max(files, key=lambda fn: os.path.getmtime(os.path.join(pat_prep_dir, fn)))
        
        picked = pick_most_recent(ref_bad_channel_files)
        
        # If none found, define a NEW file name we will create
        # Make sure this naming is unique enough for your workflow
        if picked is None:
            # You can include ses_label/run_label if you want; run_keys might be list like ["run-01", "run-1"]
            # Use run_label if you have a single normalized run label available.
            picked = f'{pat}_{ses_label}_{vid}_{run_label}_{ref}_bad_channels.txt'
    
        bad_channel_path = os.path.join(pat_prep_dir, picked)
        
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
        fif_bads = list(lfp_dat.info.get("bads", []))
        
        # --- NEW: filter to only channels that exist in this Raw ---
        ch_set = set(lfp_dat.ch_names)  # or mne_data.info["ch_names"]
        
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
        lfp_dat.info["bads"] = merged_bads
        
        window_duration = 1.0
        sfreq = lfp_dat.info["sfreq"]
        
        qc_epochs = mne.make_fixed_length_epochs(
            lfp_dat,
            duration=window_duration,
            overlap=0.0,
            preload=True,
            reject_by_annotation=True
        )
        
        n_epochs = len(qc_epochs)
        
        epoch_start_samples = qc_epochs.events[:, 0]
        
        epoch_start_times = (epoch_start_samples - lfp_dat.first_samp) / sfreq
        
        epoch_end_times = epoch_start_times + window_duration
        
        qc_table = pd.DataFrame({
            "window_id": np.arange(n_epochs),
            "start_sample": epoch_start_samples,
            "start_time_sec": epoch_start_times,
            "end_time_sec": epoch_end_times,
        })
        
        # create metadata
        qc_epochs.metadata = qc_table.copy()
        qc_table = qc_epochs.metadata.copy()
        
        qc_csv_path = os.path.join(
            out_dir,
            f"{pat}_{vid}_{ref}_continuous_1s_window_qc.csv"
        )
        
        # ------------------------------------------------------------
        # Load previously rejected windows, if QC file already exists
        # ------------------------------------------------------------
        
        if os.path.exists(qc_csv_path):
            old_qc = pd.read_csv(qc_csv_path)
        
            previous_bad_window_ids = old_qc.loc[
                # 'LH010',
          # 'NS127_02',
#          'NS135',
          # 'NS136',
          # 'NS137',
          # 'NS138',
          # 'NS140',
          # 'NS140_02',
          #  'NS142', 
          #  'NS144',
          #  'NS145',
          #  'NS154',
          #  'NS155_02',
          #  'NS164',
          # 'NS174_02',
          # 'NS174_03',
          #  'NS178',
          #  'NS211'old_qc["bad"] == True,
                "window_id"
            ].astype(int).tolist()
        
            print(f"Previously rejected windows: {previous_bad_window_ids}")
        
            # qc_epochs.metadata still contains the original window IDs
            epochs_to_drop = np.where(
                qc_epochs.metadata["window_id"].isin(previous_bad_window_ids)
            )[0]
        
            if len(epochs_to_drop) > 0:
                qc_epochs.drop(
                    epochs_to_drop,
                    reason="PREVIOUS_USER"
                )
        
                print(f"Loaded {len(epochs_to_drop)} previously rejected windows.")
        else:
            print("No existing QC file found; starting with no rejected windows.")
        
        qc_epochs.plot(
            picks=qc_epochs.ch_names,
            title=f"{pat} - {vid} - continuous recording - 1-second QC windows",
            scalings=dict(seeg=100e-6, ecog=100e-6),
            n_channels=64,
            n_epochs=20,
            show_scrollbars=True,
            block=True
        )
        
        # Final bad-channel state after manual inspection
        final_bads = list(qc_epochs.info.get("bads", []))
                
        print(f"Final bad channels: {final_bads}")
        
        lfp_dat.info["bads"] = final_bads
        
        # Save updated bad-channel state into FIF
        lfp_dat.save(preprocessed_path, overwrite=True)
        
        # Replace contents of bad-channel text file
        with open(bad_channel_path, "w") as f:
            for line in header_lines:
                f.write(line + "\n")
            for ch in final_bads:
                f.write(ch + "\n")
        
        print(f"Updated FIF: {preprocessed_path}")
        print(f"Updated bad-channel file: {bad_channel_path}")

        kept_window_ids = set(qc_epochs.metadata["window_id"].astype(int))
        
        qc_table["bad"] = ~qc_table["window_id"].isin(kept_window_ids)
        
        qc_table["rejection_reason"] = np.where(qc_table["bad"],"USER","")
        
        # ------------------------------------------------------------
        # Save QC table
        # ------------------------------------------------------------
        
        qc_csv_path = os.path.join(out_dir,f"{pat}_{vid}_{ref}_continuous_1s_window_qc.csv")
        
        qc_table.to_csv(qc_csv_path,index=False)
        
        print(f"Rejected windows: {qc_table['bad'].sum()}")
        print(f"Retained windows: {(~qc_table['bad']).sum()}")
        print(f"QC table saved to: {qc_csv_path}")
