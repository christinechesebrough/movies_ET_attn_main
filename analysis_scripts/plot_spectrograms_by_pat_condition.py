#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Load saved TF outputs, recover channel metadata, and prepare a unified
run-level structure for flexible spectrogram analyses.

Intended downstream uses:
- region x frequency static spectra
- time-resolved region spectrograms
- internal vs external window comparisons

- subject-level and group-level summaries
"""

import os
import re
from itertools import compress
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import median_abs_deviation



# =========================
# USER SETTINGS
# =========================

machine_path = 'Volumes'


plt.rcParams['font.family'] = 'sans-serif'   # or 'serif'
plt.rcParams['font.sans-serif'] = ['Arial']  # or 'Helvetica', 'DejaVu Sans'

vids = ['despicable_me_english','inscapes']#,


atlas_col = 'Y17_Atlas'   # options: 'DK_Atlas', 'Y7_Atlas', 'Y17_Atlas', 'AparcAseg_Atlas'
output = 'wavelet'
pow_type = 'log'
title_style = 'analysis'

tf_root = f'/{machine_path}/Samsung/Movie_data'
corr_dir = f'/{machine_path}/Samsung/Movie_data/data/movie_elec_corr_sheets'


# optional: restrict to certain patients or leave None
patients_to_use = None

# optional: skip unknown atlas labels
drop_unknown = False

plot_indiv = False

scheme = 'within_sub'

win_len = 10

atlas_col = 'Y17_Atlas'
#    region_name = 'Default A' #'Dorsal Attention A'

region_list = ['Default A','Dorsal Attention A','Visual Central (Visual A)']#, 'Dorsal Attention A']
   # region_list = ["Dorsal Attention Network (DAN)","Default Mode Network (DMN)"]
   # ['Visual Network (VN)',"Dorsal Attention Network (DAN)","Ventral Attention Network (VAN)","Limbic Network (LN)","Frontoparietal Network (FPN)","Default Mode Network (DMN)"]
PC_type = 'shared_PC1'
threshold = (0.5, 0.5)
thr_str = f"[{threshold[0]:.1f}, {threshold[1]:.1f}]"

save_figs = False
# fig_out_dir = f'/{machine_path}/Samsung/Movie_data/region_spectrograms_{region_name.replace(" ", "_")}'
# os.makedirs(fig_out_dir, exist_ok=True)

# =========================
# HELPERS
# =========================

def _extract_run_label(fname: str) -> str:
    m = re.search(r'run[-_]?(\d+)', fname, flags=re.IGNORECASE)
    if m:
        return f"run-{int(m.group(1)):02d}"
    return "run-01"


def _extract_ses_label(fname: str) -> str:
    m = re.search(r'ses[-_]?(\d+)', fname, flags=re.IGNORECASE)
    if m:
        return f"ses-{int(m.group(1)):02d}"
    return ""


def _extract_patient_from_entry(entry_id: str) -> str:
    m = re.match(r'(NS\d+(?:_\d+)?)', entry_id)
    if not m:
        raise ValueError(f"Could not parse patient ID from entry_id: {entry_id}")
    return m.group(1)


def _find_latest_corr_sheet(pat: str, corr_dir: str) -> str:
    excel_files = sorted([
        f for f in os.listdir(corr_dir)
        if pat in f and f.endswith('.xlsx') and not f.startswith('.')
    ])
    if not excel_files:
        raise FileNotFoundError(f"No correspondence sheet found for {pat} in {corr_dir}")

    return max(
        [os.path.join(corr_dir, f) for f in excel_files],
        key=os.path.getmtime
    )


def _load_corr_table(pat: str, corr_dir: str) -> pd.DataFrame:
    excel_path = _find_latest_corr_sheet(pat, corr_dir)
    corr = pd.read_excel(excel_path)

    # standardize label column
    col_map = {c.lower(): c for c in corr.columns}
    if 'label' not in col_map:
        raise ValueError(f"'label' column not found in correspondence sheet for {pat}")

    corr = corr.rename(columns={col_map['label']: 'label'})
    corr['label'] = corr['label'].astype(str).str.strip()
    return corr


def build_channel_metadata(labels_ip, corr_df):
    """
    Return a channel-level metadata table aligned to labels_ip order.
    """
    rows = []

    for label in labels_ip:
        match_row = corr_df[corr_df['label'] == label]

        if not match_row.empty:
            row = match_row.iloc[0]

            dk_col = 'DK_Atlas' if 'DK_Atlas' in corr_df.columns else 'Desikan_Killiany'
            y7_col = 'Y7_Atlas' if 'Y7_Atlas' in corr_df.columns else 'Yeo7'
            y17_col = 'Y17_Atlas' if 'Y17_Atlas' in corr_df.columns else 'Yeo17'
            aparc_col = 'AparcAseg_Atlas' if 'AparcAseg_Atlas' in corr_df.columns else None

            rows.append({
                'label': label,
                'DK_Atlas': row[dk_col] if dk_col in corr_df.columns else 'Unknown',
                'Y7_Atlas': row[y7_col] if y7_col in corr_df.columns else 'Unknown',
                'Y17_Atlas': row[y17_col] if y17_col in corr_df.columns else 'Unknown',
                'AparcAseg_Atlas': row[aparc_col] if aparc_col and aparc_col in corr_df.columns else 'Unknown'
            })
        else:
            rows.append({
                'label': label,
                'DK_Atlas': 'Unknown',
                'Y7_Atlas': 'Unknown',
                'Y17_Atlas': 'Unknown',
                'AparcAseg_Atlas': 'Unknown'
            })

    meta = pd.DataFrame(rows)
    return meta


def load_tf_npz(tf_path: str) -> dict:
    """
    Load one saved TF file.
    Expected keys:
    - windowed_tf: (n_windows, n_channels, n_freqs)
    - freqs_tf
    - window_centers
    - labels_ip
    - pat
    - run_label
    - vid
    - optionally pow_tf_dat, t_tf
    """
    data = np.load(tf_path, allow_pickle=True)

    out = {
        'windowed_tf': data['windowed_tf'],
        'freqs_tf': data['freqs_tf'],
        'window_centers': data['window_centers'],
        'labels_ip': [str(x) for x in data['labels_ip']],
        'pat': str(data['pat']),
        'run_label': str(data['run_label']),
        'vid': str(data['vid']),
        'fs_tf': float(data['fs_tf']),
        'decim_tf': int(data['decim_tf']),
    }

    if 'pow_tf_dat' in data.files:
        out['pow_tf_dat'] = data['pow_tf_dat']
    if 't_tf' in data.files:
        out['t_tf'] = data['t_tf']

    return out

def load_bad_window_indices(bad_windows_dir, pat, vid, run_label):
    """
    Load bad win_len s window indices for this patient/run/video.
    Returns a sorted unique integer numpy array.
    """
    pat_dir = os.path.join(bad_windows_dir, pat)
    if not os.path.isdir(pat_dir):
        return np.array([], dtype=int)

    files = [
        f for f in os.listdir(pat_dir)
        if f.endswith('.csv')
        and f'{win_len}s_bad_window_indices' in f
        and vid in f
        and run_label in f
        and not f.startswith('.')
    ]


    if len(files) == 0:
        print(f"[INFO] No bad-window file found for {pat} {vid} {run_label}")
        return np.array([], dtype=int)

    if len(files) > 1:
        print(f"[WARN] Multiple bad-window files found for {pat} {vid} {run_label}: {files}")
    
    bad_path = os.path.join(pat_dir, files[0])
    bad_df = pd.read_csv(bad_path)

    if f'window_idx_{win_len}s' not in bad_df.columns:
        raise ValueError(f"'window_idx_{win_len}s' column missing in {bad_path}")

    bad_idx = np.sort(bad_df[f'window_idx_{win_len}s'].dropna().astype(int).unique())
    return bad_idx


    #%%
    # =========================
    # LOAD TF Files + ATTACH METADATA
    # =========================
for vid in vids:
    
    bad_windows_dir = f'/{machine_path}/Samsung/Movie_data/1secEpochs_for_review_power_log_{vid}_HFA_1Apr26'
     
    run_data = []
    
    if win_len == 10:
        tf_dir = f'/Volumes/Samsung/Movie_data/wavelet_power_{win_len}s/wavelet_{vid}_all_cortContacts_tf'
    elif win_len == 5:
        tf_dir = f'/Volumes/Samsung/Movie_data/wavelet_power_5s/wavelet_{vid}_all_cortContacts_tf_5s'
        
    if vid == 'despicable_me_english':
        entry_ids = ['NS127_02_ses-02_run-01', 
                     'NS135_ses-01_run-01',
                       'NS136_ses-01_run-01', 
                       'NS137_ses-01_run-01',
                       'NS138_ses-01_run-01', 
                       'NS140_ses-01_run-01',
                       'NS140_02_ses-02_run-01', 
                       'NS153_ses-01_run-01',
                       'NS155_02_ses-02_run-01', 
                       'NS164_ses-01_run-01',
                       #'NS166_ses-01_run-01', 
                       'NS174_02_ses-02_run-01',
                       'NS174_03_ses-03_run-01', 
                       'NS178_ses-01_run-01',
                       'NS190_ses-01_run-01',
                       'NS190_ses-01_run-02',
                       'NS191_ses-01_run-01', 
                       'NS193_ses-01_run-01',
                       'NS193_ses-01_run-02',
                     #  'NS194_ses-01_run-01',
                       'NS205_ses-01_run-01']
        vid_name = 'Narrative Movie'
    
        
    if vid == 'inscapes':
        entry_ids = [
                     'NS127_02_ses-02_run-01', 
                     'NS135_ses-01_run-01',
                     'NS136_ses-01_run-01', 
                    'NS137_ses-01_run-01',
                    'NS138_ses-01_run-01', 
                    'NS140_ses-01_run-01',
                    'NS140_02_ses-02_run-01',
                    'NS151_ses-01_run-01', 
                    'NS153_ses-01_run-01',
                   # 'NS155_ses-01_run-01',
                    'NS155_02_ses-02_run-01', 
                     'NS164_ses-01_run-01',
                  #  'NS178_ses-01_run-01', 
                    'NS205_ses-01_run-01',
                    'NS210_ses-01_run-01'
                    ] 
        vid_name = 'Ambient Movie'

    tf_files = []
    
    
    entry_map = {}
    for entry in entry_ids:
        pat = _extract_patient_from_entry(entry)
        if pat:
            entry_map.setdefault(pat, []).append(entry)
    
    
    # Loop through subdirectories
    for subdir in os.listdir(tf_dir):
        subdir_path = os.path.join(tf_dir, subdir)
        
        if not os.path.isdir(subdir_path):
            continue
        
        if subdir not in entry_map:
            continue  # skip irrelevant patients
        
        # Look for matching files
        for fname in os.listdir(subdir_path):
            if not fname.endswith('.npz'):
                continue
            
            if 'wavelet_log_tf.npz' not in fname:
                continue
            
            # Check if filename matches any entry_id
            if any(entry in fname for entry in entry_map[subdir]):
                tf_files.append(os.path.join(subdir_path, fname))
    
    for tf_path in tf_files:
        try:
            tf_dat = load_tf_npz(tf_path)
    
            pat = tf_dat['pat']
            labels_ip = tf_dat['labels_ip']
    
            corr_df = _load_corr_table(pat, corr_dir)
            channel_meta = build_channel_metadata(labels_ip, corr_df)
    
            if drop_unknown:
                keep_idx = channel_meta[atlas_col].fillna('Unknown') != 'Unknown'
                keep_idx = keep_idx.to_numpy()
    
                channel_meta = channel_meta.loc[keep_idx].reset_index(drop=True)
                tf_dat['windowed_tf'] = tf_dat['windowed_tf'][:, keep_idx, :]
                tf_dat['labels_ip'] = list(channel_meta['label'].values)
    
                if 'pow_tf_dat' in tf_dat:
                    tf_dat['pow_tf_dat'] = tf_dat['pow_tf_dat'][keep_idx, :, :]
    
            tf_dat['channel_meta'] = channel_meta
            tf_dat['tf_path'] = tf_path
    
            run_data.append(tf_dat)
    
            print(
                f"Loaded {pat} | {tf_dat['vid']} | {tf_dat['run_label']} | "
                f"windows={tf_dat['windowed_tf'].shape[0]} "
                f"channels={tf_dat['windowed_tf'].shape[1]} "
                f"freqs={tf_dat['windowed_tf'].shape[2]}"
            )
    
        except Exception as e:
            print(f"[SKIP] {tf_path}: {e}")


#%%
    # -------------------------
    # LOOP THROUGH RUNS
    # -------------------------
    
    vmin_global = -3
    vmax_global = 3
    
    for region_name in region_list:
        fig_out_dir = f'/{machine_path}/Samsung/Movie_data/region_spectrograms_{win_len}s_windows/{vid}/{region_name.replace(" ", "_")}'
        os.makedirs(fig_out_dir, exist_ok=True)
        
        all_region_tf_mean = []
        all_region_tf_z = []
        
        all_internal_mean_z = []
        all_external_mean_z = []
        all_diff_mean_z = []
        
        all_internal_spectrum = []
        all_external_spectrum = []
        all_diff_spectrum = []
        
        all_ids = []
    
        
        for run_item in run_data:
            
            freqs_tf = run_item['freqs_tf']
            window_centers = run_item['window_centers']
            channel_meta = run_item['channel_meta']
        
            pat = run_item['pat']
            vid = run_item['vid']
            run = run_item['run_label']
            tf_path = run_item['tf_path']
            
            windowed_tf = run_item['windowed_tf']  
              
            print("\n" + "="*80)
            print(f"patient={pat} | video={vid} | run={run}")
            print("windowed_tf shape:", windowed_tf.shape)
        
            bad_idx = load_bad_window_indices(bad_windows_dir, pat, vid, run)
            print("Bad windows:", len(bad_idx))
            
        
            # -------------------------
            # REGION INDEXING
            # -------------------------
            
            ## Z-SCORE EACH CHANNEL, THEN COMBINE ACROSS REGION PER PATIENT
            region_idx = (channel_meta[atlas_col].astype(str) == str(region_name)).to_numpy()
            n_region_ch = int(np.sum(region_idx))
            
            print(f"Region: {region_name}")
            print("Channels in region:", n_region_ch)
            
            if n_region_ch == 0:
                print(f"[SKIP] No channels found in {region_name}")
                continue
            
            print("Channel labels:", channel_meta.loc[region_idx, 'label'].tolist())
            
            region_tf = windowed_tf[:, region_idx, :]   # (windows, region_channels, freqs)
            
        
            n_windows_tf = windowed_tf.shape[0]
            all_idx = np.arange(n_windows_tf)
            good_idx = np.setdiff1d(all_idx, bad_idx)
            
            if len(good_idx) == 0:
                print(f"[SKIP] No clean windows left after masking for {pat} | {vid} | {run}")
                continue
            
            region_tf_good = region_tf[good_idx, :, :]   # (good_windows, channels, freqs)
            
            region_med_ch = np.nanmedian(region_tf_good, axis=0, keepdims=True)  # (1, channels, freqs)
            
            region_mad_ch = np.empty((1, region_tf.shape[1], region_tf.shape[2]))
            for ch in range(region_tf.shape[1]):
                for f in range(region_tf.shape[2]):
                    region_mad_ch[0, ch, f] = median_abs_deviation(
                        region_tf_good[:, ch, f],
                        scale='normal',
                        nan_policy='omit'
                    )
            
            region_mad_ch[region_mad_ch < 1e-8] = np.nan
            
            region_tf_ch_z = (region_tf - region_med_ch) / region_mad_ch

            # keep BOTH
            region_tf_mean = np.nanmean(region_tf, axis=1)      # raw/log power
            region_tf_z = np.nanmean(region_tf_ch_z, axis=1)    # z-scored
            
            region_tf_mean_masked = region_tf_mean.copy()
            region_tf_z_masked = region_tf_z.copy()
            
            region_tf_mean_masked[bad_idx, :] = np.nan
            region_tf_z_masked[bad_idx, :] = np.nan
            
            region_tf_mean = region_tf_mean_masked
            region_tf_z = region_tf_z_masked
                        
            # -------------------------
            # LOAD WINDOW LABELS
            # -------------------------
            if PC_type == 'shared_PC1':
               # et_file = f'/{machine_path}/Samsung/Movie_data/shared_PC_features_9Jan26/{vid}_features_df_{thr_str}.csv' 
                et_file = f'/{machine_path}/Samsung/Movie_data/shared_PC_features_{win_len}s_20Apr26/{vid}_features_df_{thr_str}.csv'
            elif PC_type == "sep_PCs":
               if win_len == 5:
                  et_file = f'/{machine_path}/Samsung/Movie_data/separate_PC_features_5s_8Apr26/{vid}_features_df_{thr_str}.csv'
               elif win_len == 10:
                   et_file = f'/{machine_path}/Samsung/Movie_data/separate_PC_features_7Jan26/{vid}_features_df_{thr_str}.csv'

            else:
                raise ValueError(f"Unknown PC_type: {PC_type}")
        
            et_df = pd.read_csv(et_file)
            patient_col = next(c for c in et_df.columns if c.lower() == 'patient')
        
            ses = _extract_ses_label(tf_path)
            if ses:
                rec = f"{pat}_{ses}_{run}"
            else:
                rec = f"{pat}_{run}"
        
            print("rec:", rec)
        
            et_patient_rows = et_df[et_df[patient_col] == rec].reset_index(drop=True)
        
            if len(et_patient_rows) == 0:
                print(f"[SKIP] No ET rows matched rec={rec}")
                continue
        
            n_windows_tf = windowed_tf.shape[0]
            n_windows_et = len(et_patient_rows)
        
            print("TF windows:", n_windows_tf)
            print("ET windows:", n_windows_et)
        
            if n_windows_tf != n_windows_et:
                print(f"[SKIP] Window mismatch: TF={n_windows_tf}, ET={n_windows_et}")
                continue
        
            if scheme == 'within_sub':
                int_idx = et_patient_rows[et_patient_rows[f'Attention_Label_within_subject_dev_{thr_str}'] == 'Internal_HighConfidence'].index.to_numpy()
                ext_idx = et_patient_rows[et_patient_rows[f'Attention_Label_within_subject_dev_{thr_str}'] == 'External_HighConfidence'].index.to_numpy()
            elif scheme == 'within_timepoint':
                int_idx = et_patient_rows[et_patient_rows[f'Attention_Label_within_timepoint_dev_{thr_str}'] == 'Internal_HighConfidence'].index.to_numpy()
                ext_idx = et_patient_rows[et_patient_rows[f'Attention_Label_within_timepoint_dev_{thr_str}'] == 'External_HighConfidence'].index.to_numpy()
         
         
            int_idx_clean = np.setdiff1d(int_idx, bad_idx)
            ext_idx_clean = np.setdiff1d(ext_idx, bad_idx)
            
            print("Original int:", len(int_idx), "→ Clean:", len(int_idx_clean))
            print("Original ext:", len(ext_idx), "→ Clean:", len(ext_idx_clean))
            
            int_idx = int_idx_clean
            ext_idx = ext_idx_clean
         
            # int_idx = et_patient_rows[
            #     et_patient_rows[f'Attention_Label_{thr_str}'] == 'Internal_HighConfidence'
            # ].index.to_numpy()
        
            # ext_idx = et_patient_rows[
            #     et_patient_rows[f'Attention_Label_{thr_str}'] == 'External_HighConfidence'
            # ].index.to_numpy()
        
            window_labels = np.full(n_windows_tf, 'none', dtype=object)
            window_labels[int_idx] = 'internal'
            window_labels[ext_idx] = 'external'
        
            print("Internal windows:", np.sum(window_labels == 'internal'))
            print("External windows:", np.sum(window_labels == 'external'))
        
            if len(int_idx) == 0 or len(ext_idx) == 0:
                print("[SKIP] Missing internal or external windows")
                continue
            
            # -------------------------
            # TIME-RESOLVED REGION SPECTROGRAMS
            # -------------------------
            if plot_indiv:
                # fig1, ax = plt.subplots(figsize=(8, 5))
                # im = ax.imshow(
                #     region_tf_mean.T,
                #     aspect='auto',
                #     origin='lower',
                #     extent=[window_centers[0], window_centers[-1], freqs_tf[0], freqs_tf[-1]],
                #     vmin=np.nanpercentile(region_tf_mean, 5),
                #     vmax=np.nanpercentile(region_tf_mean, 95)
                # )
                # plt.colorbar(im, ax=ax, label='Mean log power')
                # ax.set_xlabel('Time (s)')
                # ax.set_ylabel('Frequency (Hz)')
                # ax.set_title(f"{pat} | {vid} | {run} | {win_len}s | {region_name} (raw)")
                # plt.show()
            
                zlim = np.nanpercentile(np.abs(region_tf_z), 95)
                
                
                
            if plot_indiv:
                
                log_freqs = np.log10(freqs_tf)
                
                internal_tf = region_tf_z[int_idx, :]   # (internal_windows, freqs)
                external_tf = region_tf_z[ext_idx, :]   # (external_windows, freqs)
                
                # -------------------------
                # GLOBAL COLOR SCALE (important)
                # -------------------------
                all_vals = np.concatenate([
                    region_tf_z.flatten(),
                    internal_tf.flatten(),
                    external_tf.flatten()
                ])
                vmax = np.nanpercentile(np.abs(all_vals), 99)
                vmin = -vmax
            
                # -------------------------
                # CREATE 3-PANEL FIGURE
                # -------------------------
                fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True)
            
                # -------- FULL (time-based) --------
                im0 = axes[0].imshow(
                    region_tf_z.T,
                    aspect='auto',
                    origin='lower',
                    # extent=[window_centers[0], window_centers[-1],
                    #         freqs_tf[0], freqs_tf[-1]],
                    extent=[window_centers[0], window_centers[-1],
                            log_freqs[0], log_freqs[-1]],
                    vmin=vmin,
                    vmax=vmax,
                    cmap='RdBu_r'
                )
                axes[0].set_title('Full')
                axes[0].set_xlabel('Time (s)')
                axes[0].set_ylabel('Frequency (Hz)',fontsize = 12)
            
                # -------- INTERNAL --------
                im1 = axes[1].imshow(
                    internal_tf.T,
                    aspect='auto',
                    origin='lower',
                    # extent=[0, internal_tf.shape[0],
                    #         freqs_tf[0], freqs_tf[-1]],
                    extent=[0, internal_tf.shape[0],
                            log_freqs[0], log_freqs[-1]],
                    vmin=vmin,
                    vmax=vmax,
                    cmap='RdBu_r'
                )
                axes[1].set_title('Internal')
                axes[1].set_xlabel('Window #')
            
                # -------- EXTERNAL --------
                im2 = axes[2].imshow(
                    external_tf.T,
                    aspect='auto',
                    origin='lower',
                    # extent=[0, external_tf.shape[0],
                    #         freqs_tf[0], freqs_tf[-1]],
                    extent=[0, external_tf.shape[0],
                            log_freqs[0], log_freqs[-1]],
                    vmin=vmin,
                    vmax=vmax,
                    cmap='RdBu_r'
                )
                axes[2].set_title('External')
                axes[2].set_xlabel('Window #')
            
                # -------------------------
                # SHARED COLORBAR
                # -------------------------
                
                # leave space on the right for colorbar
                fig.subplots_adjust(right=0.88)
                
                # create a dedicated colorbar axis: [left, bottom, width, height]
                cax = fig.add_axes([0.999, 0.15, 0.02, 0.7])
                cbar = fig.colorbar(im2, cax=cax)
                cbar.set_label('Z-scored log power')
            
                # -------------------------
                # TITLE + LAYOUT
                # -------------------------
                fig.suptitle(f"{pat} | {vid_name} | {run} | {win_len}s | {region_name}",fontsize = 14)
            
                yticks = [2, 4, 8, 13, 30, 70, 150]

                axes[0].set_yticks(np.log10(yticks))
                axes[0].set_yticklabels([str(y) for y in yticks])
                axes[0].set_ylabel('Frequency (Hz, log scale)', fontsize=12)
            
                plt.tight_layout()
                plt.show()
 
                    
            # -------------------------
            # CONDITION DIFFERENCE SPECTRUM
            # -------------------------
            internal_spectrum = np.nanmean(region_tf_mean[int_idx, :], axis=0)
            external_spectrum = np.nanmean(region_tf_mean[ext_idx, :], axis=0)
            diff_spectrum = internal_spectrum - external_spectrum
        

            # -------------------------
            # CONDITION-AVERAGED HEATMAPS
            # -------------------------
            internal_mean_z = np.nanmean(region_tf_z[int_idx, :], axis=0)[None, :]
            external_mean_z = np.nanmean(region_tf_z[ext_idx, :], axis=0)[None, :]
            diff_mean_z = internal_mean_z - external_mean_z

            if plot_indiv:
            
                # consistent color scale across all three
                all_vals = np.concatenate([
                    internal_mean_z.flatten(),
                    external_mean_z.flatten(),
                    diff_mean_z.flatten()
                ])
                vmax = np.nanpercentile(np.abs(all_vals), 99)
                vmin = -vmax
            
                # ---- LOG TRANSFORM FREQUENCIES ----
                log_freqs = np.log10(freqs_tf)
            
                fig, axes = plt.subplots(1, 3, figsize=(8, 6), sharey=True)
            
                data_list = [internal_mean_z, external_mean_z, diff_mean_z]
                titles = ['Internal', 'External', 'Difference']
            
                for ax, data, title in zip(axes, data_list, titles):
            
                    data_col = np.atleast_2d(data).T
            
                    im = ax.imshow(
                        data_col,
                        aspect='auto',
                        origin='lower',
                        extent=[0, 1, log_freqs[0], log_freqs[-1]],  # <-- LOG SPACE
                        vmin=vmin,
                        vmax=vmax,
                        cmap='RdBu_r'
                    )
            
                    ax.set_title(title)
                    ax.set_xticks([])
            
                # ---- LOG TICKS (IMPORTANT) ----
                yticks = [2, 4, 8, 13, 30, 70, 150]
                axes[0].set_yticks(np.log10(yticks))
                axes[0].set_yticklabels([str(y) for y in yticks])
                axes[0].set_ylabel('Frequency (Hz, log scale)',fontsize = 12)
            
                # ---- COLORBAR ----
                fig.subplots_adjust(right=0.88)
            
                cax = fig.add_axes([0.9, 0.15, 0.02, 0.7])  # <-- fixed position
                cbar = fig.colorbar(im, cax=cax)
                cbar.set_label('Z-scored log power')
            
                plt.suptitle(f"{pat} | {vid_name} | {run} | {region_name}")
            
                plt.tight_layout(rect=[0, 0, 0.88, 1])  # leave space for colorbar
            
                # ---- SAVE ----
                fname = f"{pat}_{vid_name}_{run}_{region_name.replace(' ', '_')}_logfreq.png"
                save_path = os.path.join(fig_out_dir, fname)
            
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Saved: {save_path}")
            
                plt.show()
                plt.close(fig)
            
                        
            all_region_tf_mean.append(region_tf_mean)
            all_region_tf_z.append(region_tf_z)
            
            all_internal_mean_z.append(internal_mean_z.squeeze())
            all_external_mean_z.append(external_mean_z.squeeze())
            all_diff_mean_z.append(diff_mean_z.squeeze())
            
            all_internal_spectrum.append(internal_spectrum)
            all_external_spectrum.append(external_spectrum)
            all_diff_spectrum.append(diff_spectrum)
            
            all_ids.append({
                'pat': pat,
                'run': run,
                'vid': vid
            })
        
        if len(all_region_tf_mean) == 0:
            print(f"[SKIP] No valid runs for {vid} in region {region_name}")
            continue
        
        all_region_tf_mean = np.stack(all_region_tf_mean, axis=0)   # (runs, windows, freqs)
        all_region_tf_z = np.stack(all_region_tf_z, axis=0)
        
        all_internal_mean_z = np.vstack(all_internal_mean_z)        # (runs, freqs)
        all_external_mean_z = np.vstack(all_external_mean_z)
        all_diff_mean_z = np.vstack(all_diff_mean_z)
        
        all_internal_spectrum = np.vstack(all_internal_spectrum)    # (runs, freqs)
        all_external_spectrum = np.vstack(all_external_spectrum)
        all_diff_spectrum = np.vstack(all_diff_spectrum)
        
        
        avg_region_tf_mean = np.nanmean(all_region_tf_mean, axis=0)   # (windows, freqs)
        avg_region_tf_z = np.nanmean(all_region_tf_z, axis=0)
        
        avg_internal_mean_z = np.nanmean(all_internal_mean_z, axis=0)[None, :]
        avg_external_mean_z = np.nanmean(all_external_mean_z, axis=0)[None, :]
        avg_diff_mean_z = np.nanmean(all_diff_mean_z, axis=0)[None, :]
        
        avg_internal_spectrum = np.nanmean(all_internal_spectrum, axis=0)
        avg_external_spectrum = np.nanmean(all_external_spectrum, axis=0)
        avg_diff_spectrum = np.nanmean(all_diff_spectrum, axis=0)
        
        # -------------------------
        # AVG TIME-RESOLVED SPECTROGRAM
        # -------------------------
        fig1, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(
            avg_region_tf_z.T,
            aspect='auto',
            origin='lower',
            extent=[window_centers[0], window_centers[-1], freqs_tf[0], freqs_tf[-1]]
        )
        plt.colorbar(im, ax=ax, label='Mean z-scored log power')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)',fontsize = 12)
        ax.set_title(f"{vid} | {region_name} |  {win_len}s | Average time-resolved spectrogram")
        
        fname_base = f"{vid}_{atlas_col}_{win_len}s_region_name.replace(' ', '_')" 
        
        fig1.savefig(
            os.path.join(fig_out_dir, f"{fname_base}_{win_len}s_avg_time_resolved.png"),
            dpi=300,
            bbox_inches='tight'
        )
        
        #plt.show()
        plt.close(fig1)
        
        
        # -------------------------
        # CONDITION HEATMAPS
        # -------------------------
        # -------------------------

        vmin_heatmap = -0.75
        vmax_heatmap = 0.75
                
        fig, axes = plt.subplots(1, 3, figsize=(9, 6), sharey=True)
        
        data_list = [
            avg_internal_mean_z,
            avg_external_mean_z,
            avg_diff_mean_z
        ]
        titles = ['Internal', 'External', 'Int - Ext']
        
        # log-spaced y coordinates
        log_freqs = np.log10(freqs_tf)
        
        # make frequency bin edges for pcolormesh
        freq_edges = np.empty(len(log_freqs) + 1)
        freq_edges[1:-1] = (log_freqs[:-1] + log_freqs[1:]) / 2
        freq_edges[0] = log_freqs[0] - (log_freqs[1] - log_freqs[0]) / 2
        freq_edges[-1] = log_freqs[-1] + (log_freqs[-1] - log_freqs[-2]) / 2
        
        x_edges = [0, 1]
        
        for ax, data, title in zip(axes, data_list, titles):
            im = ax.pcolormesh(
                x_edges,
                freq_edges,
                data.T,
                vmin=vmin_heatmap,
                vmax=vmax_heatmap,
                cmap='RdBu_r',
                shading='flat'
            )
            ax.set_title(title)
            ax.set_xticks([])
        
        yticks = [2, 2, 4, 8, 13, 30, 70, 150]
        axes[0].set_yticks(np.log10(yticks))
        axes[0].set_yticklabels([str(y) for y in yticks])
        axes[0].set_ylabel('Frequency (Hz, log scale)')
        
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.9)
        cbar.set_label('Z-scored log power')
        
        if title_style == 'analysis':
            plt.suptitle(f"{vid_name} |{PC_type} {thr_str} {win_len}s |{scheme} | {region_name}")
        elif title_style == 'figure':
            plt.suptitle(f"{vid_name} | {win_len}s | {region_name}")

        plt.show()
        
        # save
        fname_base = f"{vid}_{atlas_col}_{region_name.replace(' ', '_')}"
        fig.savefig(
            os.path.join(fig_out_dir, f"{fname_base}_{win_len}s_condition_heatmaps_combined.png"),
            dpi=300,
            bbox_inches='tight'
        )
        
        plt.show()
        plt.close(fig)
 