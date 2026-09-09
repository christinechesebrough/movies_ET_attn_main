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
vids = ['despicable_me_hungarian']#,'despicable_me_english']#,'inscapes']#,'despicable_me_english']#,'despicable_me_english']
freq_bands = ['low_mid']#['delta','theta','alpha','gamma','HFA']#'beta','gamma','HFA'] #'delta','theta','alpha','beta','gamma'
#freq_bands = ['theta_alpha','all_gamma']

ref = 'avg'
#freq_band = 'HFA'
region = 'all'


data_dir = f'/{machine_path}/Samsung/Movie_data/new_dmh_prep_standard'
isc_dir = f'/{machine_path}/SamsungMovie_data/data/isc'
mne_data_dir = f'/{machine_path}/Samsung/Movie_data/new_dmh_prep_standard'
elec_dir = f'/{machine_path}/Samsung/Movie_data/data/electrode_localization'
fs_dir = f'/{machine_path}/Samsung/anatomy'


corr_dir = f'/{machine_path}/Samsung/Movie_data/data/movie_elec_corr_sheets'

fs_eye = 300

visualize_mne_steps = False
plot_power = False


output = 'power'
pow_type = 'log'

entropy_type = 'mssd'


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

sys.path.insert(0, f'/{machine_path}/Samsung/scripts/movies_ET_attn_main/src')
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


def smooth_and_decimate(x, fs, smooth_sec=0.2, decim=10):
    """
    x: (n_channels, n_times)
    smooth_sec: moving average window length in seconds
    decim: downsample factor for plotting
    """
    if smooth_sec is not None and smooth_sec > 0:
        win = int(round(smooth_sec * fs))
        win = max(win, 1)
        kernel = np.ones(win) / win
        x = signal.filtfilt(kernel, [1.0], x, axis=1)  # zero-phase moving average

    if decim is not None and decim > 1:
        x = x[:, ::decim]
        fs = fs / decim

    return x, fs

def plot_overlay(t, pow_dat, labels, max_ch=10, title="Power over time"):
    fig, ax = plt.subplots(figsize=(12, 5))
    for i in range(min(max_ch, pow_dat.shape[0])):
        ax.plot(t, pow_dat[i], label=labels[i], linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Power (env / log / z)")
    ax.grid(True)
    if min(max_ch, pow_dat.shape[0]) <= 12:
        ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    return fig

def plot_stacked(t, pow_dat, labels, title="Power over time (stacked)", spacing=5.0):
    fig, ax = plt.subplots(figsize=(12, 8))
    for i in range(pow_dat.shape[0]):
        ax.plot(t, pow_dat[i] + i * spacing, linewidth=0.6)
    ax.set_yticks([i * spacing for i in range(pow_dat.shape[0])])
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channels (offset)")
    ax.grid(True, axis="x")
    plt.tight_layout()
    return fig

def plot_heatmap(t, pow_dat, labels, title="Power heatmap"):
    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(
        pow_dat,
        aspect='auto',
        origin='lower',
        extent=[t[0], t[-1], 0, pow_dat.shape[0]],
        interpolation='nearest'
    )
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channel index")
    # Optional: label every Nth channel to avoid clutter
    step = max(1, pow_dat.shape[0] // 20)
    ax.set_yticks(np.arange(0, pow_dat.shape[0], step))
    ax.set_yticklabels([labels[i] for i in range(0, pow_dat.shape[0], step)], fontsize=8)
    fig.colorbar(im, ax=ax, label="Power (env / log / z)")
    plt.tight_layout()
    return fig

def atlas_labels_for_channels(corr, pat, labels_ip, atlas_col="DK_Atlas"):
    atlas_df = corr.rename(columns={"SubID": "pat_base", "Contact": "electrode"}).copy()
    atlas_df["pat_base"] = atlas_df["pat_base"].astype(str).str.strip()
    atlas_df["electrode"] = atlas_df["electrode"].astype(str).str.strip()

    lab = pd.DataFrame({"electrode": [str(x).strip() for x in labels_ip]})
    lab["pat_base"] = str(pat).strip()

    lab = lab.merge(
        atlas_df[["pat_base", "electrode", "Hem", "DK_Atlas", "DK_Lobe", "Y7_Atlas", "Y17_Atlas", "AparcAseg_Atlas"]],
        on=["pat_base", "electrode"],
        how="left"
    )

    if atlas_col not in lab.columns:
        raise ValueError(f"atlas_col='{atlas_col}' not found. Available: {list(lab.columns)}")

    lab["atlas_label"] = lab[atlas_col].astype("string").fillna("Unknown").str.strip()

    # optional hemi standardization
    lab["hem"] = lab["Hem"].astype(str).str.upper().str.strip()
    lab.loc[lab["hem"].isin(["LEFT","LH"]), "hem"] = "L"
    lab.loc[lab["hem"].isin(["RIGHT","RH"]), "hem"] = "R"
    lab["hem"] = lab["hem"].where(lab["hem"].isin(["L","R"]), "")

    return lab

def reorder_pow_by_atlas(pow_plot_win, labels_ip, lab_df, *, sort_cols=("atlas_label", "hem", "electrode")):
    lab_df = lab_df.sort_values(list(sort_cols)).reset_index(drop=True)
    idx = [labels_ip.index(e) for e in lab_df["electrode"].tolist()]
    pow_re = pow_plot_win[idx, :]
    ylabels = (lab_df["atlas_label"] + " | " + lab_df["electrode"]).tolist()
    return pow_re, ylabels, lab_df


#%% Main script

vids.sort()
freq_bands.sort()

processed_lfp_files = []  # initialize once


for vid in vids:
    
    if vid == 'despicable_me_english':
        #good_ET_pats = ["NS190"]#['NS140_02','NS153','NS164','NS166','NS174_02']#['NS127_02','NS135','NS136','NS137','NS138','NS140','NS140_02'] #'NS153','NS164','NS166','NS174_02',"NS178","NS190","NS191"]#,'NS193',"NS194","NS201",'NS205']
       # good_ET_pats = ['NS193',"NS194","NS201",'NS205']
       # good_ET_pats = ['NS127_02','NS135','NS136','NS137','NS138','NS140','NS151','NS153','NS154','NS155_02','NS164','NS174_02','NS174_03','NS190','NS191','NS193','NS194','NS178','NS201_02','NS204','NS205']
      # good_ET_pats = ['NS193'] 
       good_ET_pats = [
        'NS127_02',
        'NS135',
        'NS136',
        'NS137',
        'NS138',
        'NS140',
        # 'NS151',
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
       
    if vid == 'the_present':
        patients = [
         # 'NS135',
         'NS140',
         'NS137',
         'NS144_02',
         'NS149',
         'NS153',
         'NS154',
         'NS155',
         'NS155_02',
         'NS164',
         'NS174_03',
         'NS178',
         'NS190',
         'NS192',
         'NS193',
         'NS208',
         'NS210']
         #['NS174_03','NS210']

         
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
        # 'LH010',
        # 'NS127_02',
        'NS128_02',
        'NS135',
        # 'NS136',
        # 'NS137',
        'NS138',
        'NS140',
        'NS140_02',
        'NS145',
        'NS154',
        'NS164',
        'NS167',
        'NS174_02',
        'NS174_03',
        # 'NS178'
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

    
    time_isc = ([0., 2.5, 5., 7.5, 10., 12.5, 15., 17.5, 20., 22.5, 25.,
            27.5,  30. ,  32.5,  35. ,  37.5,  40. ,  42.5,  45. ,  47.5,
            50. ,  52.5,  55. ,  57.5,  60. ,  62.5,  65. ,  67.5,  70. ,
            72.5,  75. ,  77.5,  80. ,  82.5,  85. ,  87.5,  90. ,  92.5,
            95. ,  97.5, 100. , 102.5, 105. , 107.5, 110. , 112.5, 115. ,
           117.5, 120. , 122.5, 125. , 127.5, 130. , 132.5, 135. , 137.5,
           140. , 142.5, 145. , 147.5, 150. , 152.5, 155. , 157.5, 160. ,
           162.5, 165. , 167.5, 170. , 172.5, 175. , 177.5, 180. , 182.5,
           185. , 187.5, 190. , 192.5, 195. , 197.5, 200. , 202.5, 205. ,
           207.5, 210. , 212.5, 215. , 217.5, 220. , 222.5, 225. , 227.5,
           230. , 232.5, 235. , 237.5, 240. , 242.5, 245. , 247.5, 250. ,
           252.5, 255. , 257.5, 260. , 262.5, 265. , 267.5, 270. , 272.5,
           275. , 277.5, 280. , 282.5, 285. , 287.5, 290. , 292.5, 295. ,
           297.5, 300. , 302.5, 305. , 307.5, 310. , 312.5, 315. , 317.5,
           320. , 322.5, 325. , 327.5, 330. , 332.5, 335. , 337.5, 340. ,
           342.5, 345. , 347.5, 350. , 352.5, 355. , 357.5, 360. , 362.5,
           365. , 367.5, 370. , 372.5, 375. , 377.5, 380. , 382.5, 385. ,
           387.5, 390. , 392.5, 395. , 397.5, 400. , 402.5, 405. , 407.5,
           410. , 412.5, 415. , 417.5, 420. , 422.5, 425. , 427.5, 430. ,
           432.5, 435. , 437.5, 440. , 442.5, 445. , 447.5, 450. , 452.5,
           455. , 457.5, 460. , 462.5, 465. , 467.5, 470. , 472.5, 475. ,
           477.5, 480. , 482.5, 485. , 487.5, 490. , 492.5, 495. , 497.5,
           500. , 502.5, 505. , 507.5, 510. , 512.5, 515. , 517.5, 520. ,
           522.5, 525. , 527.5, 530. , 532.5, 535. , 537.5, 540. , 542.5,
           545. , 547.5, 550. , 552.5, 555. , 557.5, 560. , 562.5, 565. ,
           567.5, 570. , 572.5, 575. , 577.5, 580. , 582.5, 585. , 587.5,
           590])
    
    
    freq_band_count = 0
    
    for freq_band in freq_bands:        
        if freq_band == 'low_mid':
            freq_range = (2, 57)
        elif freq_band == 'high':
            freq_range = (62,118)
        
        else:
            raise Exception("no range assigned")
         
        if freq_band == 'alpha':
            bin_width = 2
        elif freq_band == 'beta':
            bin_width = 4
        elif freq_band in ['gamma', 'all_gamma','mid_gamma']:
            bin_width = 8
        elif freq_band == 'HFA':
            bin_width = 10
        else:
            bin_width = 2  # default

    
        print(f"Processing data for {vid} in {freq_band} with range: {freq_range} Hz")
        if output == 'entropy':
            fig_dir = f'/{machine_path}/Samsung/Movie_data/{freq_band}_{region}_{vid}_all_cortContacts_entropy_29Mar25/entropy_extracted'
        else:
            fig_dir = f'/{machine_path}/Samsung/Movie_data/{output}_{pow_type}_{freq_band}_{vid}_26Mar26'
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
            
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
                    time_lfp = mne_data.times
                        
                    lfp_ip = lfp[idx_ip, :]
                        
                    labels_ip = list(compress(labels, idx_ip))
    
                    info = mne.create_info(ch_names=labels_ip,sfreq = fs_lfp)
    
                    f_start, f_end = freq_range
                    freq_bins = [(f, min(f + bin_width, f_end)) for f in range(f_start, f_end, bin_width)]
                    
                    # Remove flat channels before processing (std < threshold)
                    flat_std_thresh = 1e-6
                    stds = np.std(lfp_ip, axis=1)
                    nonflat_idx = stds > flat_std_thresh
                    lfp_ip = lfp_ip[nonflat_idx]
                    labels_ip = [label for i, label in enumerate(labels_ip) if nonflat_idx[i]]




#%%
                    
                    # FOOOF fit across rolling windows:
                    # 10-second windows, with start times defined by time_isc.
                    # Since time_isc is spaced every 2.5 s, this creates 7.5 s overlap.
                    #
                    # Outputs:
                    #   1) one row per channel × window with aperiodic offset/exponent
                    #   2) one row per detected peak within each channel × window
                    
                    fooof_fig_dir = f'/{machine_path}/Samsung/Movie_data/rolling_fooof_{freq_band}_{vid}_26Jun26'
                    if not os.path.exists(fooof_fig_dir):
                        os.makedirs(fooof_fig_dir)
                    
                    fooof_fig_patient_dir = os.path.join(fooof_fig_dir, pat)
                    if not os.path.exists(fooof_fig_patient_dir):
                        os.makedirs(fooof_fig_patient_dir)
                    
                    
                    # -------------------------
                    # Rolling-window settings
                    # -------------------------
                    
                    window_len_sec = 10.0
                    window_overlap_sec = 7.5
                    
                    # You said time_isc should be treated as the window start times.
                    window_starts = np.asarray(time_isc, dtype=float)
                    
                    # Use the actual sampling rate from the file, but warn if it is not 600 Hz.
                    fs_lfp = float(fs_lfp)
                    
                    if not np.isclose(fs_lfp, 600.0):
                        print(f"Warning: expected fs_lfp ≈ 600 Hz, but got {fs_lfp}")
                    
                    window_len_samp = int(round(window_len_sec * fs_lfp))
                    
                    
                    # -------------------------
                    # FOOOF frequency settings
                    # -------------------------
                    
                    fooof_fit_ranges = {
                        'low_mid': (2, 57),
                        'high': (62, 118),
                    }
                    
                    if freq_band not in fooof_fit_ranges:
                        raise ValueError(f"No FOOOF fit range defined for freq_band={freq_band}")
                    
                    fooof_fmin, fooof_fmax = fooof_fit_ranges[freq_band]
                    
                    # For this script, the peak search range is the full fit range.
                    # That means:
                    #   low_mid: save all peaks between 1 and 57 Hz
                    #   high:    save all peaks between 62 and 118 Hz
                    peak_search_range = (fooof_fmin, fooof_fmax)
                    
                    if freq_band == 'high':
                        peak_width_limits = [2, 20]
                        max_n_peaks = 6
                    else:
                        peak_width_limits = [1, 10]
                        max_n_peaks = 12
                    
                    # -------------------------
                    # Storage
                    # -------------------------
                    
                    fooof_window_results = []
                    fooof_peak_results = []
                    
                    # -------------------------
                    # Optional: atlas lookup map
                    # -------------------------
                    
                    region_map = {}
                    
                    for label in labels_ip:
                        fooof_region = None
                        match_row = elecs_subs[elecs_subs['label'] == label]
                    
                        if not match_row.empty:
                            if 'Y17_Atlas' in match_row.columns:
                                fooof_region = match_row['Y17_Atlas'].values[0]
                            elif 'Y7_Atlas' in match_row.columns:
                                fooof_region = match_row['Y7_Atlas'].values[0]
                    
                        region_map[label] = fooof_region
                    
                    # -------------------------
                    # Main rolling FOOOF loop
                    # -------------------------
                    
                    for win_idx, win_start_sec in enumerate(window_starts):
                    
                        win_end_sec = win_start_sec + window_len_sec
                        win_center_sec = win_start_sec + (window_len_sec / 2)
                    
                        start_samp = int(round(win_start_sec * fs_lfp))
                        end_samp = start_samp + window_len_samp
                    
                        # Skip windows that extend past the available data.
                        if start_samp < 0:
                            print(f"Skipping window {win_idx}: start before 0 s")
                            continue
                    
                        if end_samp > lfp_ip.shape[1]:
                            print(
                                f"Skipping window {win_idx}: "
                                f"{win_start_sec:.2f}-{win_end_sec:.2f} s exceeds data length "
                                f"({lfp_ip.shape[1] / fs_lfp:.2f} s)"
                            )
                            continue
                    
                        lfp_win = lfp_ip[:, start_samp:end_samp]
                    
                        # Compute PSD for this 10 s window.
                        # This computes one PSD per channel for this local time window.
                        try:
                            psd_fooof, freqs_fooof = psd_array_welch(
                                lfp_win,
                                sfreq=fs_lfp,
                                fmin=fooof_fmin,
                                fmax=fooof_fmax,
                                n_fft=int(fs_lfp * 2),
                                n_overlap=int(fs_lfp),
                                average='mean'
                            )
                        except Exception as e:
                            print(
                                f"PSD failed for {pat} {run_label} "
                                f"window {win_idx} {win_start_sec:.2f}-{win_end_sec:.2f} s: {e}"
                            )
                            continue
                    
                        for ch_idx, label in enumerate(labels_ip):
                    
                            fm = FOOOF(
                                peak_width_limits=peak_width_limits,
                                max_n_peaks=max_n_peaks,
                                min_peak_height=0.1,
                                aperiodic_mode='fixed',
                                peak_threshold = 2.0,
                                verbose=False
                            )
                    
                            found_peak = False
                            n_peaks_total = 0
                            n_peaks_in_band = 0
                            offset = np.nan
                            exponent = np.nan
                            r_squared = np.nan
                            fit_error = np.nan
                    
                            try:
                                fm.fit(freqs_fooof, psd_fooof[ch_idx, :])
                    
                                # In fixed mode, aperiodic_params_ = [offset, exponent]
                                offset = fm.aperiodic_params_[0]
                                exponent = fm.aperiodic_params_[1]
                    
                                all_peaks = fm.peak_params_
                    
                                if all_peaks is not None and all_peaks.shape[0] > 0:
                                    n_peaks_total = all_peaks.shape[0]
                    
                                    in_band = (
                                        (all_peaks[:, 0] >= peak_search_range[0]) &
                                        (all_peaks[:, 0] <= peak_search_range[1])
                                    )
                    
                                    band_peaks = all_peaks[in_band]
                                    n_peaks_in_band = band_peaks.shape[0]
                                    found_peak = n_peaks_in_band > 0
                    
                                    # Save one row per peak.
                                    for peak_idx, peak in enumerate(band_peaks):
                                        peak_cf = peak[0]      # center frequency
                                        peak_amp = peak[1]     # amplitude above aperiodic background
                                        peak_bw = peak[2]      # bandwidth
                    
                                        fooof_peak_results.append({
                                            'Patient': pat,
                                            'Run': run_label,
                                            'Video': vid,
                                            'FreqBand': freq_band,
                                            'Channel': label,
                                            'Region': region_map.get(label, None),
                    
                                            'Window_Index': win_idx,
                                            'Window_Start_Sec': win_start_sec,
                                            'Window_End_Sec': win_end_sec,
                                            'Window_Center_Sec': win_center_sec,
                                            'Start_Sample': start_samp,
                                            'End_Sample': end_samp,
                    
                                            'FOOOF_Fit_Range_Low': fooof_fmin,
                                            'FOOOF_Fit_Range_High': fooof_fmax,
                    
                                            'Peak_Index_In_Window': peak_idx,
                                            'Peak_CF_Hz': peak_cf,
                                            'Peak_Amplitude': peak_amp,
                                            'Peak_Bandwidth_Hz': peak_bw,
                    
                                            'Aperiodic_Offset': offset,
                                            'Aperiodic_Exponent': exponent,
                                            'FOOOF_R2': fm.r_squared_,
                                            'FOOOF_Error': fm.error_
                                        })
                    
                                r_squared = fm.r_squared_
                                fit_error = fm.error_
                    
                            except Exception as e:
                                print(
                                    f"FOOOF failed for {pat} {run_label} {label} "
                                    f"window {win_idx} {win_start_sec:.2f}-{win_end_sec:.2f} s: {e}"
                                )
                    
                            # Save one row per channel × window, regardless of whether peaks were found.
                            fooof_window_results.append({
                                'Patient': pat,
                                'Run': run_label,
                                'Video': vid,
                                'FreqBand': freq_band,
                                'Channel': label,
                                'Region': region_map.get(label, None),
                    
                                'Window_Index': win_idx,
                                'Window_Start_Sec': win_start_sec,
                                'Window_End_Sec': win_end_sec,
                                'Window_Center_Sec': win_center_sec,
                                'Start_Sample': start_samp,
                                'End_Sample': end_samp,
                    
                                'FOOOF_Fit_Range_Low': fooof_fmin,
                                'FOOOF_Fit_Range_High': fooof_fmax,
                                'Peak_Search_Range_Low': peak_search_range[0],
                                'Peak_Search_Range_High': peak_search_range[1],
                    
                                'Aperiodic_Offset': offset,
                                'Aperiodic_Exponent': exponent,
                    
                                'Found_Peak': found_peak,
                                'N_Peaks_Total': n_peaks_total,
                                'N_Peaks_In_Band': n_peaks_in_band,
                    
                                'FOOOF_R2': r_squared,
                                'FOOOF_Error': fit_error
                            })
                    
                    
                    # -------------------------
                    # Save outputs
                    # -------------------------
                    
                    fooof_window_df = pd.DataFrame(fooof_window_results)
                    fooof_peak_df = pd.DataFrame(fooof_peak_results)
                    
                    fooof_window_csv = os.path.join(
                        fooof_fig_patient_dir,
                        f'{pat}_{vid}_{run_label}_{region}_{freq_band}_rolling_fooof_aperiodic_windows_{max_n_peaks}maxPeaks.csv'
                    )
                    
                    fooof_peak_csv = os.path.join(
                        fooof_fig_patient_dir,
                        f'{pat}_{vid}_{run_label}_{region}_{freq_band}_rolling_fooof_peaks_long_{max_n_peaks}maxPeaks.csv'
                    )
                    
                    fooof_window_df.to_csv(fooof_window_csv, index=False)
                    fooof_peak_df.to_csv(fooof_peak_csv, index=False)
                    
                    print(f'Saved rolling FOOOF aperiodic window results to {fooof_window_csv}')
                    print(f'Saved rolling FOOOF peak results to {fooof_peak_csv}')