#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 05:52:17 2026

Calculate aperiodic and oscillatory activity over rolling windows

@author: christinechesebrough
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
import multiprocessing

from scipy.signal import decimate
from scipy.signal import correlate

from joblib import Parallel, delayed
from fooof import FOOOF
from fooof.plts.spectra import plot_spectrum

machine_path = 'Volumes'#'media/christine'#'Volumes' #'media/christine'


#from antropy import sample_entropy, spectral_entropy, perm_entropy, lziv_complexity

# Add Linux library paths BEFORE importing epipe
sys.path.insert(0, f'/{machine_path}/Samsung/EPIPE/Python')
sys.path.insert(0, f'/{machine_path}/Samsung/iEEG2NWB-main')

#vids = ['inscapes','despicable_me_english']#,'despicable_me_english']
vids = ['despicable_me_hungarian','despicable_me_english']#,'inscapes']#,'despicable_me_english']#,'despicable_me_english']
#freq_bands = ['theta_alpha','all_gamma']

ref = 'avg'
#freq_band = 'HFA'
region = 'all'


elec_dir = f'/{machine_path}/Samsung/Movie_data/data/electrode_localization'
fs_dir = f'/{machine_path}/Samsung/anatomy'
elec_recon_dir = f'/{machine_path}/Samsung/Movie_data/data/movie_elec_corr_sheets'


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

sys.path.insert(0, f'/{machine_path}/Samsung/scripts/movies_ET_attn_main/src')
# Import EEG preprocessing helper functions
from eeg_preproc_helpers import (
    plot_power_spectra,  plot_psd_batched, 
    create_file_paths, apply_highpass_filter,
    save_bad_channels, load_bad_channels, check_bad_channels_integrity, load_preprocessed_data, validate_mne_structure, preserve_mne_structure, restore_mne_structure, load_data_with_bad_channels, summarize_bad_channels, 
    interpolate_spikes, make_groups_from_prefix, regress_out_noise_by_group,
    detect_spikes_all_channels, reref_avg_by_group,
    ProcessingLogger, detect_spikes_ref1)

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

processed_files = []  # initialize once


for vid in vids:
    
    data_dir = f'/{machine_path}/Samsung/Movie_data/rolling_fooof_low_mid_{vid}_26Jun26'
    
    if vid == 'despicable_me_english':
       good_ET_pats = [
         'NS127_02',
         'NS135',
         'NS136',
         'NS137',
         'NS138',
         'NS140',
         'NS151',
         'NS153',
         'NS154',
         'NS155_02',
         'NS164',
         'NS174_02',
         'NS174_03',
         'NS178',
         'NS190',
         'NS191',
         'NS193',
         'NS194',
         'NS201_02',
         'NS204',
         'NS205'
         ]
         
    elif vid == 'inscapes':
        good_ET_pats = [
         'NS127_02',
         'NS135',
         'NS136',
         'NS137',
         'NS138',
         'NS140',
         'NS140_02',
         'NS151',
         'NS153',
         'NS155',
         'NS155_02',
         'NS164',
         'NS178',
         'NS205',
         'NS210']
    
    elif vid == 'despicable_me_hungarian':
        good_ET_pats = [
        'LH010',
        'NS127_02',
        'NS128_02',
        'NS135',
        'NS136',
        'NS137',
        'NS138',
        'NS140',
        'NS140_02',
        'NS145',
        'NS154',
        'NS164',
        'NS167',
        'NS174_02',
        'NS174_03',
        'NS178'
        ]
        
    patients = good_ET_pats
    
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
    fig_dir = f'/{machine_path}/Samsung/Movie_data/'       ## ...something descriptive
    
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
            
        #%
    for pat in patients:
        pat_dir = os.path.join(data_dir, pat)
        if not os.path.isdir(pat_dir):
            print(f"[SKIP] Missing directory: {pat_dir}")
            continue
        fig_patient_dir = os.path.join(fig_dir, pat)
        if not os.path.exists(fig_patient_dir):
            os.makedirs(fig_patient_dir)
            
        pat_dir = '{:s}/{:s}'.format(data_dir, pat)
        
        files = os.listdir(pat_dir)
                    
        
        files = [
            f for f in os.listdir(pat_dir)
            if f.endswith(".csv")
            and ('peaks_long' in f)
            and any(k.lower() in f.lower() for k in keys)
            and not f.startswith("._")
        ]
        
        # Deterministic sort: by session then run then filename

        files = sorted(files, key=_sort_key)

        print(f"Found {len(files)} matching runs for {pat}:")
        for f in files:
            print("  ", f)
            processed_files.append(f)

        # # Iterate over each file/run as an independent entry
        for file in files:
            # Build an entry label that will propagate to outputs
            ses_label = _extract_ses_label(file)
            run_label = _extract_run_label(file)
            if ses_label:
                entry_id = f"{pat}_{ses_label}_{run_label}"
            else:
                entry_id = f"{pat}_{run_label}"

            print(f"Loading data for entry {entry_id} from {file} ...")
            
            if run_label == 'run-01':
                run_keys = ['run-01','run-1']
            if run_label == 'run-02':
                run_keys = ['run-02','run-2']

            fooof_file = os.path.join(pat_dir, file)

            # find patient-specific electrode label /correspondence sheet. This is how we will fill in the missing "region" information for each electrode. 

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
            
            
            
            # add electrode anatomy
            # for each contact ("label"), find corresponding atlas values for four . The atlas column names are "desikan_killiany", "Yeo17", "Yeo7", 
            # and "aparc_aseg" and variations of those (case insensitive), or including _Atlas e.g. "AparcAseg_Atlas"). 
            # e.g. elecs_subs.Contact.values[(elecs_subs.AparcAseg_Atlas)]
            
            # aggregate for the whole movie

#