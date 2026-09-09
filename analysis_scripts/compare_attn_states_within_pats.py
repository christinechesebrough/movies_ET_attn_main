#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 13:12:53 2026

@author: christinechesebrough
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  7 12:07:03 2026

@author: christinechesebrough
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 13:12:56 2024

@author: christinechesebrough
"""



import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import seaborn as sns  
from statsmodels.stats.multitest import fdrcorrection
import matplotlib.patches as patches
from scipy.spatial.distance import norm
from scipy.stats import ttest_1samp
from scipy.stats import median_abs_deviation

plt.rcParams['font.family'] = 'sans-serif'   # or 'serif'
plt.rcParams['font.sans-serif'] = ['Arial']  # or 'Helvetica', 'DejaVu Sans'


Y7_REGION_MAP = {
    "7Networks_1": "Visual",
    "7Networks_2": "Somatomotor",
    "7Networks_3": "Dorsal Attention",
    "7Networks_4": "Ventral Attention",
    "7Networks_5": "Limbic",
    "7Networks_6": "Frontoparietal",
    "7Networks_7": "Default",
}

def permute_split(vals, n_int, rng):
    """
    Randomly assign n_int samples to 'int' and the rest to 'ext' from pooled vals.
    """
    idx = rng.permutation(len(vals))
    int_idx = idx[:n_int]
    ext_idx = idx[n_int:]
    return vals[int_idx], vals[ext_idx]

def empirical_p_from_null(t_obs, t_null):
    """
    Two-sided empirical p-value with +1 smoothing.
    """
    t_obs = float(t_obs)
    t_null = np.asarray(t_null, dtype=float)
    return (np.sum(np.abs(t_null) >= abs(t_obs)) + 1) / (len(t_null) + 1)

def extract_pat_id(entry: str) -> str:
    """
    Extract patient ID like 'NS127' or 'NS127_02' from an entry string.
    """
    m = re.match(r'(NS\d+(?:_\d+)?)', entry)
    if not m:
        raise ValueError(f"Could not extract patient ID from: {entry}")
    return m.group(1)


def extract_run_label(fname: str) -> str:
    """
    Try to extract a run label like 'run-01' or 'run-1' from a filename.
    Fallback: 'run-01'.
    """
    m = re.search(r'run[-_]?(\d+)', fname, flags=re.IGNORECASE)
    if m:
        return f"run-{int(m.group(1)):02d}"
    return "run-01"

def pick_file(files, *, contains_any=None, contains_all=None, endswith=None, startswith_not=None):
    out = files
    if startswith_not is not None:
        out = [f for f in out if not f.startswith(startswith_not)]
    if endswith is not None:
        out = [f for f in out if f.endswith(endswith)]
    if contains_any is not None:
        out = [f for f in out if any(k in f for k in contains_any)]
    if contains_all is not None:
        out = [f for f in out if all(k in f for k in contains_all)]
    return out

# Flag to control normalization
normalize = True

# Lists of frequency bands, movies, and atlases
freq_bands = ['theta','alpha','gamma','HFA']#,'HFA']  # Example: alpha and HFA bands
#freq_bands = ['alpha','HFA']

lfp_type = 'power'

entropy_type = 'mssd'

movies = ['despicable_me_hungarian','despicable_me_english']
PC_type = 'shared_PC1'#'shared_PC1'#'sep_PCs'#'shared_PC1'
atlases = ['Y17_Atlas_Region']#,'Y7_Atlas_Region','Y17_Atlas_Region']

method = 'power_log'

# Define threshold label - this should match the one used in examining_shared_PC_features.py
#threshold_label = "PC1z_high0.75_low0.5_dev_high.7_low.3"  # Update this to match your actual thresholds

#threshold = (.6,.6)

threshold_grid = [(t, t) for t in np.arange(0.6,0.7)]

#thr_str = f"[{threshold[0]:.1f}, {threshold[1]:.1f}]"

scheme = 'within_timepoint'

machine_path = 'Volumes'

win_len = 10

title_style = 'analysis'

#if PC_type == 'shared_PC1':
#    z_threshold = z_threshold_both
    
#if PC_type == 'sep_PCs':
#    z_threshold = z_threshold_both

correction_method = 'fdr'  # Options: 'fdr', 'bonferroni', or 'none'

# Output directory
if lfp_type == 'entropy': 
    fig_dir = os.path.join(f'/{machine_path}/Samsung/Movie_data/PC_derived_freq_x_atlas_matrices_ENTROPY/')
    
if lfp_type == 'power':
    if PC_type == 'sep_PCs':
        if win_len == 10:
            fig_dir = os.path.join(f'/{machine_path}/Samsung/Movie_data/PC_derived_freq_x_atlas_matrices_7Jan26')
        elif win_len ==5:
            fig_dir = os.path.join(f'/{machine_path}/Samsung/Movie_data/PC_derived_freq_x_atlas_matrices_5s_6Apr26')

    elif PC_type == 'shared_PC1':
       # fig_dir = os.path.join('f/{machine_path}/Samsung/Movie_data/PC_derived_freq_x_atlas_matrices_shared_PC1_9Jan26')
        if win_len == 10:
            fig_dir = os.path.join(f'/{machine_path}/Samsung/Movie_data/PC_derived_freq_x_atlas_matrices_shared_PC1_10s_20Apr26')
        if win_len == 5:
            fig_dir = os.path.join(f'/{machine_path}/Samsung/Movie_data/PC_derived_freq_x_atlas_matrices_shared_PC1_5s_20Apr26')

    
    os.makedirs(fig_dir, exist_ok=True)

def run_analysis_for_threshold(threshold):
    thr_str = f"[{threshold[0]:.1f}, {threshold[1]:.1f}]"

    patient_delta_rows = []
    electrode_rows = []

    # your existing loops go here
    # movie -> atlas -> freq_band -> rec
    # use thr_str exactly as you already do when loading et_file
    # and selecting Attention_Label columns
    
    
    # BEFORE loops: create collector
    patient_delta_rows = []   # put near top of script (outside loops)
    electrode_rows = []
    patient_region_mean_rows = []
    
    for movie in movies:
        
        bad_windows_dir = f'/{machine_path}/Samsung/Movie_data/1secEpochs_for_review_power_log_{movie}_HFA_1Apr26'
    
        if movie == 'despicable_me_english':
            vidname = 'Narrative'
            #patients = ['NS127_02','NS135','NS137','NS136','NS138','NS140','NS153','NS164','NS166','NS174_02']
           # entry_ids = ['NS140_ses-01_run-01']
            entry_ids = ['NS127_02_ses-02_run-01', 
                        'NS135_ses-01_run-01',
                        'NS136_ses-01_run-01', 
                        'NS137_ses-01_run-01',
                        'NS138_ses-01_run-01', 
                        'NS140_ses-01_run-01',
                        'NS153_ses-01_run-01',
                        'NS155_02_ses-02_run-01', 
                        'NS164_ses-01_run-01',
                       # 'NS166_ses-01_run-01', 
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
        
            # Load the new combined features file
            if PC_type == 'shared_PC1':
              #  et_file = f'/{machine_path}/Samsung/Movie_data/shared_PC_features_9Jan26/{movie}_features_df_{thr_str}.csv'
                et_file = f'/{machine_path}/Samsung/Movie_data/shared_PC_features_{win_len}s_20Apr26/{movie}_features_df_{thr_str}.csv'

            elif PC_type == "sep_PCs":
               #et_file = f'/Volumes/Samsung/Movie_data/combined_PC_features_output_21Apr25/despicable_me_english_features_df_{z_threshold}.csv'
               if win_len == 5:
                  et_file = f'/{machine_path}/Samsung/Movie_data/separate_PC_features_5s_8Apr26/{movie}_features_df_{thr_str}.csv'
               elif win_len == 10:
                   et_file = f'/{machine_path}/Samsung/Movie_data/separate_PC_features_7Jan26/{movie}_features_df_{thr_str}.csv'
            # Read et file as df
            et_df = pd.read_csv(et_file)
            factor = 'PC1'
            
        elif movie == 'despicable_me_hungarian':
            vidname = 'Narrative - Uncomprehended Language'
            entry_ids =  [
                #'LH010_ses-01_run-01',
                 'NS127_02_ses-02_run-01', 
                 'NS135_ses-01_run-01',
                 'NS136_ses-01_run-01', 
                 'NS137_ses-01_run-01',
                 'NS138_ses-01_run-01', 
                 'NS140_ses-01_run-01',
                 'NS140_02_ses-02_run-01',
                # 'NS145_ses-02_run-01', 
               #  'NS154_ses-01_run-01',
                 'NS164_ses-01_run-01', 
                 'NS174_02_ses-02_run-01',
                 'NS174_03_ses-03_run-01', 
                 'NS178_ses-01_run-01'
                   ]
            # Load the new combined features file
            if PC_type == 'shared_PC1':
                #et_file = '/Volumes/Samsung/Movie_data/combined_PC_features_output_21Apr25/combined_features_despicable_me_english.csv'
                  # et_file = f'/{machine_path}/Samsung/Movie_data/shared_PC_features_9Jan26/{movie}_features_df_{thr_str}.csv'
                   et_file = f'/{machine_path}/Samsung/Movie_data/shared_PC_features_{win_len}s_20Apr26/{movie}_features_df_{thr_str}.csv'
            elif PC_type == "sep_PCs":
               #et_file = f'/Volumes/Samsung/Movie_data/combined_PC_features_output_21Apr25/despicable_me_english_features_df_{z_threshold}.csv'
               if win_len == 5:
                  et_file = f'/{machine_path}/Samsung/Movie_data/separate_PC_features_5s_8Apr26/{movie}_features_df_{thr_str}.csv'
               elif win_len == 10:
                   et_file = f'/{machine_path}/Samsung/Movie_data/separate_PC_features_7Jan26/{movie}_features_df_{thr_str}.csv'
            # Read et file as df
            et_df = pd.read_csv(et_file)
            factor = 'PC1'

        elif movie == 'inscapes':
            vidname = 'Ambient'
            #patients = ['NS127_02','NS135','NS136','NS137','NS138','NS140','NS140_02','NS164','NS166']
            #entry_ids = ['NS140_ses-01_run-01']
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
                      #  'NS155_ses-01_run-01',
                        'NS155_02_ses-02_run-01', 
                        'NS164_ses-01_run-01',
                       # 'NS178_ses-01_run-01', 
                        'NS205_ses-01_run-01',
                        'NS210_ses-01_run-01'
                       #'NS211_ses-01_run-01'
                       ]
            
            
            # Load the new combined features file
            if PC_type == 'shared_PC1':
              #  et_file = f'/{machine_path}/Samsung/Movie_data/shared_PC_features_9Jan26/{movie}_features_df_{thr_str}.csv'
               et_file = f'/{machine_path}/Samsung/Movie_data/shared_PC_features_{win_len}s_20Apr26/{movie}_features_df_{thr_str}.csv'

            elif PC_type == "sep_PCs":
               if win_len == 5:
                  et_file = f'/{machine_path}/Samsung/Movie_data/separate_PC_features_5s_8Apr26/{movie}_features_df_{thr_str}.csv'
               elif win_len == 10:
                   et_file = f'/{machine_path}/Samsung/Movie_data/separate_PC_features_7Jan26/{movie}_features_df_{thr_str}.csv'
            # Read et file as df
            et_df = pd.read_csv(et_file)
            factor = 'PC1'
        
        for atlas in atlases:
            if atlas == 'DK_Atlas_Region':
                atlas_name = 'DK Atlas Regions'
            elif atlas == 'Y7_Atlas_Region':
                atlas_name = 'Yeo7 Networks'
            elif atlas == 'Y17_Atlas_Region':
                atlas_name = 'Yeo17 Atlas Region'
            elif atlas == 'AparcAseg_Atlas_Region':
                atlas_name = 'AparcAsec Atlas'
            
            # Initialize matrices with atlas groups as index and frequency bands as columns
            atlas_groups = []  # Will store unique atlas groups
            
            t_matrix = pd.DataFrame()
            p_matrix = pd.DataFrame()
            mean_delta_matrix = pd.DataFrame()
            n_pat_matrix = pd.DataFrame()
    
            
            # Loop through all frequency bands, movies, and atlases
            for freq_band in freq_bands:
                if freq_band == 'HFA':
                    freq_range = '70 - 150 Hz'
                    freq_name = 'HFA (70-150 Hz)'
                elif freq_band == 'alpha':
                    freq_range = '8 - 13 Hz'
                    freq_name = 'Alpha (8-13 Hz)'
                elif freq_band == 'theta':
                    freq_range = '4-7 Hz'
                    freq_name = 'Theta (4-7 Hz)'
                elif freq_band == 'beta':
                    freq_range = '14-40 Hz'
                    freq_name = 'Beta (14-30 Hz)'
                elif freq_band == 'gamma':
                    freq_range = 'gamma'
                    freq_name = 'Gamma (31-50 Hz)'
                elif freq_band == 'delta':
                    freq_range = '1-3 Hz'
                    freq_name = "Delta (1-3 Hz)"
                    
                # define the data directory inside the frequency band loop
                if lfp_type == 'power':
                    #data_dir = f'/Volumes/Samsung/Movie_data/{freq_band}_all_{movie}_all_cortContacts_wLabels_y17_4Mar25/'
                   # data_dir = f'/{machine_path}/Samsung/Movie_data/{freq_band}_{movie}_4Jan26'
                    if win_len == 10:              
                        data_dir = f'/Volumes/Samsung/Movie_data/windowed_power_10s/windowed_unnormed_{method}_{movie}_{freq_band}_1Apr26'
                    elif win_len == 5:
                        data_dir = f'/Volumes/Samsung/Movie_data/windowed_power_5s/windowed_unnormed_{method}_{movie}_{freq_band}_5sec_5Apr26'
    
                elif lfp_type == 'entropy':
                    data_dir = f'/{machine_path}/Samsung/Movie_data/entropy_extracted/{freq_band}_all_{movie}_all_cortContacts_entropy_29Mar25'
    
                # Initialize dictionary to store data for each atlas group across all conditions
               # atlas_group_data = {}
                patient_region_diffs = {}  # atlas_group -> list of patient-level diffs
                
    
                # Loop through patients
                for rec in entry_ids:
                    pat = extract_pat_id(rec)
                    run = extract_run_label(rec)
                    
                    pat_dir = os.path.join(data_dir, pat)
                                                   
                    bad_windows_pat_dir = os.path.join(bad_windows_dir, pat)
                   
                    bad_windows_files = [filename for filename in os.listdir(bad_windows_pat_dir) 
                     if f'{method}' in filename 
                     and filename.endswith('.csv')
                     and f'{win_len}s_bad_window_indices' in filename
                     ]
                    
                    bad_windows_file = bad_windows_files[0]  # Select the first matching file
    
                    bad_windows_path = os.path.join(bad_windows_pat_dir,bad_windows_file)
                    bad_windows = pd.read_csv(bad_windows_path)
                    bad_idx = np.array(bad_windows[f'window_idx_{win_len}s'].tolist())
                    
                    
                    # Find the CSV files containing atlas info and lfp data
                    if lfp_type == 'power':
                        lfp_files = [filename for filename in os.listdir(pat_dir) if f'{method}' in filename and filename.endswith('.csv')]
                    elif lfp_type == 'entropy':
                        lfp_files = [filename for filename in os.listdir(pat_dir) if f'{entropy_type}' in filename and filename.endswith('.csv')]
    
                    if not lfp_files:
                        print(f"No LFP files found for patient {pat} in {pat_dir}")
                        continue
    
                    if pat in ['NS190','NS193']:
                        lfp_file = pick_file( lfp_files, contains_any=run, startswith_not="._" )
                        lfp_file = lfp_file[0] if isinstance(lfp_file, list) else lfp_file
                    else:
                        lfp_file = lfp_files[0]
                        
                    full_lfp_path = os.path.join(pat_dir, lfp_file)
                    
    
                    try:
                        lfp_values = pd.read_csv(full_lfp_path)
                        print(f"Processed LFP data for: {full_lfp_path}")
                    except Exception as e:
                        print(f"Error reading {full_lfp_path}: {e}")
                        continue
    
                    atlas_rows = lfp_values.head(4)
                    data_rows = lfp_values.iloc[4:].reset_index(drop=True)
    
                    # Extract relevant data 
                    
                    # Find and extract the rows corresponding to the patient 'pat' in et_file (index by 'Patient' column)                
                    et_patient_rows = et_df[et_df['patient'] == rec].reset_index(drop=True)
                    
                    n_lfp = len(data_rows)
                    n_et  = len(et_patient_rows)
                    if n_lfp != n_et:
                        print(f"[WARN] Length mismatch {rec} {movie} {freq_band}: LFP={n_lfp}, ET={n_et}.")
                       # continue
            
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
                    
                    n_windows = len(data_rows)
                    all_idx = np.arange(n_windows)
                    good_idx = np.setdiff1d(all_idx, bad_idx)
                    
                    # Normalize electrode columns using only good windows
                    electrode_cols = [col for col in lfp_values.columns if col[0] in ['L', 'R'] and col[1:].isalnum()]
                    
                    if normalize:
                        data_rows[electrode_cols] = data_rows[electrode_cols].apply(pd.to_numeric, errors='coerce')
                    
                        for col in electrode_cols:
                            x = data_rows[col].astype(float).values
                    
                            # compute robust center/scale using only clean windows
                            mu = np.nanmedian(x[good_idx])
                            sd = median_abs_deviation(x[good_idx], scale='normal', nan_policy='omit')
                    
                            if np.isnan(sd) or sd < 1e-8:
                                sd = 1.0
                    
                            # apply normalization to the full column
                            data_rows[col] = (x - mu) / sd
      
                    atlas_row = atlas_rows[atlas_rows['Atlas'] == atlas].iloc[0]
                    grouped_cols = {}
                    
                    for col in electrode_cols:
                        atlas_value = atlas_row[col]
                    
                        # Normalize types / missing
                        if pd.isna(atlas_value):
                            continue
                        atlas_value = str(atlas_value)
                    
                        # Fix Y7 mislabeled regions
                        if atlas == "Y7_Atlas_Region":
                            atlas_value = Y7_REGION_MAP.get(atlas_value, atlas_value)
                    
                        if atlas_value not in grouped_cols:
                            grouped_cols[atlas_value] = []
                        grouped_cols[atlas_value].append(col)
    
                    # Exclude regions based on the atlas
                    if atlas == 'Y7_Atlas_Region':
                        excluded_regions = ['FreeSurfer_Defined_Medial_Wall']
                    elif atlas == 'Y17_Atlas_Region':
                        excluded_regions = ['FreeSurfer_Defined_Medial_Wall']
                    elif atlas == 'DK_Atlas_Region':
                        excluded_regions = ['bankssts', 'unknown']
                    elif atlas == 'AparcAseg_Atlas_Region':
                        excluded_regions = ['Unknown']

    
                    # Process each atlas group
                    for atlas_group, cols in grouped_cols.items():
                        if atlas_group in excluded_regions:
                           # print(f"Skipping excluded region: {atlas_group}")
                            continue
    
                        aggregated_int_values = []
                        aggregated_ext_values = []
                        
                        int_vals_patient = []
                        ext_vals_patient = []
                        
                        elec_deltas = []
                        elec_int_means = []
                        elec_ext_means = []
                        
                        # for elec in cols:
                        #     if elec not in data_rows.columns:
                        #         continue
                        
                        #     col_data = data_rows[elec].astype(float).values
                        #     iv = col_data[int_idx]
                        #     ev = col_data[ext_idx]
                        
                        #     if np.sum(~np.isnan(iv)) < 10 or np.sum(~np.isnan(ev)) < 10:
                        #         continue
                        
                        #     elec_int_means.append(np.nanmean(iv))
                        #     elec_ext_means.append(np.nanmean(ev))
                        
                        for elec in cols:
                            if elec not in data_rows.columns:
                                continue
                        
                            col_data = data_rows[elec].astype(float).values
                            iv = col_data[int_idx]
                            ev = col_data[ext_idx]
                        
                            if np.sum(~np.isnan(iv)) < 10 or np.sum(~np.isnan(ev)) <10:
                                continue
                        
                            mean_int_e = float(np.nanmean(iv))
                            mean_ext_e = float(np.nanmean(ev))
                            delta_e = mean_int_e - mean_ext_e
                        
                            # store electrode-level row
                            electrode_rows.append({
                                "movie": movie,
                                "atlas": atlas,
                                "freq_band": freq_band,
                                "region": atlas_group,
                                "patient": rec,
                                "pat_base": pat,
                                "run": run,
                                "electrode": elec,
                                "mean_int": mean_int_e,
                                "mean_ext": mean_ext_e,
                                "delta": delta_e,
                                "n_int": int(np.sum(~np.isnan(iv))),
                                "n_ext": int(np.sum(~np.isnan(ev))),
                            })
                        
                            # keep for region aggregation
                            elec_int_means.append(mean_int_e)
                            elec_ext_means.append(mean_ext_e)
    
                        
                        if len(elec_int_means) > 0:
                            mean_int = float(np.nanmean(elec_int_means))
                            mean_ext = float(np.nanmean(elec_ext_means))
                            delta = mean_int - mean_ext
                        
                            # for inferential stats
                            patient_region_diffs.setdefault(atlas_group, []).append(delta)
                        
                            # for plotting/export
                            patient_delta_rows.append({
                                "movie": movie,
                                "atlas": atlas,
                                "freq_band": freq_band,
                                "region": atlas_group,
                                "patient": rec,
                                "pat_base": pat,
                                "run": run,
                                "mean_int": mean_int,
                                "mean_ext": mean_ext,
                                "delta": delta,
                                "n_elec": int(len(elec_int_means)),
                                "n_int": int(len(int_idx)),
                                "n_ext": int(len(ext_idx)),
                            })
    
                        # if len(elec_deltas) > 0:
                        #     delta = float(np.nanmean(elec_deltas))  # patient-level region delta
                        
                        #     # for the inferential stats (across patients)
                        #     patient_region_diffs.setdefault(atlas_group, []).append(delta)
                        
                        #     # for visualization / tidy export
                        #     patient_delta_rows.append({
                        #         "movie": movie,
                        #         "atlas": atlas,
                        #         "freq_band": freq_band,
                        #         "region": atlas_group,
                        #         "patient": rec,       # full rec ID
                        #         "pat_base": pat,      # NS### or NS###_##
                        #         "run": run,
                        #         "delta": delta,
                        #         "n_elec": len(elec_deltas),
                        #         "n_int": int(len(int_idx)),
                        #         "n_ext": int(len(ext_idx)),
                        #     })
                        
                        electrode_df = pd.DataFrame(electrode_rows)
                        electrode_df.to_csv(os.path.join(fig_dir,f"{movie}_electrode_stats_{threshold}.csv"))
    
    
                        electrode_df.assign(abs_delta=lambda d: d.delta.abs()) \
                            .sort_values("abs_delta", ascending=True)
                        
                # --- Across-patient stats per region: one-sample test of delta vs 0 ---
    
                n_lfp = len(data_rows)
                n_et  = len(et_patient_rows)
                if n_lfp != n_et:
                    print(f"[WARN] Length mismatch {rec} {movie} {freq_band}: LFP={n_lfp}, ET={n_et}")
                n_int, n_ext = len(int_idx), len(ext_idx)
                if n_int < 10 or n_ext < 10:
                    print(f"[WARN] Few bins {rec} {movie} {freq_band}: n_int={n_int}, n_ext={n_ext}")
                
                atlas_groups = []
                t_values = []
                p_values = []
                mean_delta_values = []
                n_patient_values = []
                
                for atlas_group, deltas in patient_region_diffs.items():
                    deltas = np.asarray(deltas, dtype=float)
                    deltas = deltas[~np.isnan(deltas)]
                
                    if len(deltas) < 3:
                        continue  # minimum patients
                
                    t_stat, p_val = ttest_1samp(deltas, popmean=0.0, nan_policy="omit")
                
                    atlas_groups.append(atlas_group)
                    t_values.append(float(t_stat))
                    p_values.append(float(p_val))
                    mean_delta_values.append(float(np.mean(deltas)))
                    n_patient_values.append(int(len(deltas)))
    
                # Initialize matrices on first band
                if mean_delta_matrix.empty:
                    mean_delta_matrix = pd.DataFrame(index=atlas_groups, columns=freq_bands, dtype=float)
                    p_matrix = pd.DataFrame(index=atlas_groups, columns=freq_bands, dtype=float)
                    t_matrix = pd.DataFrame(index=atlas_groups, columns=freq_bands, dtype=float)
                    n_pat_matrix = pd.DataFrame(index=atlas_groups, columns=freq_bands, dtype=float)
                
                # Assign using .loc so missing regions in some bands don't misalign
                mean_delta_matrix.loc[atlas_groups, freq_band] = mean_delta_values
                p_matrix.loc[atlas_groups, freq_band] = p_values
                t_matrix.loc[atlas_groups, freq_band] = t_values
                n_pat_matrix.loc[atlas_groups, freq_band] = n_patient_values
                
                
            # Apply specific ordering to t_matrix
            if atlas == 'DK_Atlas_Region':
                network_order = {
                    'Default Mode Network (DMN)': ['medialorbitofrontal', 'parsopercularis', 'rostralanteriorcingulate', 'rostralmiddlefrontal', 'parahippocampal', 'temporalpole', 'posteriorcingulate', 'precuneus','inferiorparietal'],
                    'Somatomotor Network (SMN)': ['postcentral', 'precentral', 'insula', 'isthmuscingulate'],
                    'Dorsal Attention Network (DAN)': ['superiorfrontal', 'caudalmiddlefrontal', 'superiorparietal'],
                    'Ventral Attention Network (VAN)': ['lateralorbitofrontal', 'parsorbitalis', 'parstriangularis', 'supramarginal', 'transversetemporal'],
                    'Limbic Network (LN)': ['entorhinal'],
                    'Frontoparietal Network (FPN)': ['superiorfrontal', 'caudalmiddlefrontal'],
                    'Visual Network (VN)': ['fusiform', 'inferiortemporal', 'middletemporal', 'lateraloccipital', 'lingual', 'pericalcarine', 'cuneus', 'paracentral']
                }
                ordered_atlas_groups = [region for network, regions in network_order.items() for region in regions]
            elif atlas == 'Y7_Atlas_Region':
                ordered_atlas_groups = ['Default', 'Somatomotor', 'Dorsal Attention', 'Ventral Attention', 'Limbic', 'Frontoparietal', 'Visual']
            elif atlas == 'Y17_Atlas_Region':
               # ordered_atlas_groups = ['Default C', 'Default B', 'Default A', 'Somatomotor A','Somatomotor B', 'Dorsal Attention A', 'Dorsal Attention B', 'Salience / Ventral Attention A','Salience / Ventral Attention B', 'Limbic A','Limbic B', 'Control C', 'Control A', 'Control B', 'Visual Central (Visual A)','Visual Peripheral (Visual B)', 'Temporal Parietal']
               ordered_atlas_groups = ['Default A', 'Dorsal Attention A', 'Visual Central (Visual A)']

            elif atlas == 'AparcAseg_Atlas_Region':
                ordered_atlas_groups = ['Right-Hippocampus','Left-Hippocampus','ctx-rh-entorhinal','ctx-lh-entorhinal','ctx-rh-entorhinal','ctx-rh-precuneus','ctx-lh-precuneus']
            
            patient_delta_df = pd.DataFrame(patient_delta_rows)
            patient_delta_df.to_csv(os.path.join(fig_dir, "patient_level_region_deltas.csv"), index=False)
            print(patient_delta_df.shape)
                  
                # Blown up plots of key regions
                  
              # --- Define regions to plot ---
            # Option A: supply explicitly
            # regions = ["Default A", "Salience / Ventral Attention A", "Visual Central (Visual A)"]
            
            # Option B: keep your atlas-dependent default, but allow multiple
            if atlas == "Y17_Atlas_Region":
                regions = ["Default A","Dorsal Attention A","Visual Central (Visual A)"]  # replace/extend with more regions as desired
            elif atlas == "Y7_Atlas_Region":
                regions = ["Default Mode Network (DMN)",'Dorsal Attention Network (DAN)','Visual Network (VN)']
            elif atlas == "DK_Atlas_Region":
                regions = ["posteriorcingulate"]
            elif atlas == 'AparcAseg_Atlas_Region':
                regions = ['Right-Hippocampus','Left-Hippocampus','ctx-rh-precuneus','ctx-lh-precuneus']
            
            
            # --- Build a combined DF for all requested regions ---
            df_all = patient_delta_df[
                (patient_delta_df["movie"] == movie) &
                (patient_delta_df["atlas"] == atlas) &
                (patient_delta_df["freq_band"] == freq_band) &
                (patient_delta_df["region"].isin(regions))
            ].copy()
            
            # Collapse duplicates per patient *within each region*
            df_all = (
                df_all.groupby(["region", "patient"], as_index=False)[["mean_int", "mean_ext"]]
                .mean()
            )
            
            # Drop patients missing either state (within region)
            df_all = df_all.dropna(subset=["mean_int", "mean_ext"])
         #  
            # --- Compute per-region group means and SEMs ---
            summary = (
                df_all.groupby("region", as_index=False)
                .agg(
                    m_int=("mean_int", "mean"),
                    m_ext=("mean_ext", "mean"),
                    sd_int=("mean_int", "std"),
                    sd_ext=("mean_ext", "std"),
                    n=("patient", "nunique"),
                )
            )
            
            summary["sem_int"] = summary["sd_int"] / np.sqrt(summary["n"])
            summary["sem_ext"] = summary["sd_ext"] / np.sqrt(summary["n"])
            
            # Ensure plotting order matches your input list
            summary["region"] = pd.Categorical(summary["region"], categories=regions, ordered=True)
            summary = summary.sort_values("region")
            
            # --- Plot ---
    
            
            region_labels = summary["region"].astype(str).tolist()
            x_base = np.arange(len(region_labels))
            offset = 0.20
            
            x_int = x_base - offset
            x_ext = x_base + offset
            
            plt.figure(figsize=(max(5, 1.2 * len(region_labels)), 5))
            
            bar_width = .4
            alpha_val = .6
            
            bar = False
            box = True 
            
            if bar: 
                plt.bar(x_int, summary["m_int"], width=bar_width,
                        yerr=summary["sem_int"], capsize=6, alpha = alpha_val,color = 'blue', label="Internal (mean ± SEM)")
                plt.bar(x_ext, summary["m_ext"],width=bar_width,                
                        yerr=summary["sem_ext"], capsize=6, alpha= alpha_val, color = 'grey',label="External (mean ± SEM)")
            if box:
                alpha_val = 0.3
                width_val = .35
                for i, reg in enumerate(region_labels):
                    dfr = df_all[df_all["region"] == reg]
                
                    # Internal box
                    plt.boxplot(
                        dfr["mean_int"].values,
                        positions=[x_int[i]],
                        widths=width_val,
                        patch_artist=True,
                        showfliers=False,
                        boxprops=dict(facecolor="blue", alpha=alpha_val),
                        medianprops=dict(color="black"),
                        whiskerprops=dict(color="black"),
                        capprops=dict(color="black")
                    )
                
                    # External box
                    plt.boxplot(
                        dfr["mean_ext"].values,
                        positions=[x_ext[i]],
                        widths=width_val,
                        patch_artist=True,
                        showfliers=False,
                        boxprops=dict(facecolor="grey", alpha=alpha_val),
                        medianprops=dict(color="black"),
                        whiskerprops=dict(color="black"),
                        capprops=dict(color="black")
                    )
                              
    
            # Per-patient paired lines (draw within each region)
            for i, reg in enumerate(region_labels):
                dfr = df_all[df_all["region"] == reg]
                for _, r in dfr.iterrows():
                    plt.plot([x_int[i], x_ext[i]], [r["mean_int"], r["mean_ext"]], linewidth=1.5, alpha=0.7)
            
            from scipy.stats import ttest_rel
            from matplotlib.patches import Patch
            
            def p_to_stars(p):
                if p < 0.001:
                    return "***"
                elif p < 0.01:
                    return "**"
                elif p < 0.05:
                    return "*"
                else:
                    return None
            
            def add_sig_bracket(ax, x1, x2, y, h, text, fontsize=10):
                # bracket
                ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c="k")
                # text
                ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom", fontsize=fontsize)
            
            ax = plt.gca()
            
            # --- compute paired stats per region + draw markers ---
            ymin, ymax = ax.get_ylim()
            yr = ymax - ymin
            
            # Put brackets above the higher of the two boxes for each region
            pad = 0.03 * yr      # vertical padding above max datapoint
            h   = 0.02 * yr      # bracket height
            step = 0.06 * yr     # in case you ever add multiple brackets per region later
            
            for i, reg in enumerate(region_labels):
                dfr = df_all[df_all["region"] == reg].dropna(subset=["mean_int", "mean_ext"])
                if len(dfr) < 3:
                    continue  # too few pairs to be meaningful
            
                # paired test
                t_stat, p_val = ttest_rel(dfr["mean_int"].values, dfr["mean_ext"].values, nan_policy="omit")
                stars = p_to_stars(p_val)
                
                if stars is None:
                    continue
                
                # bracket placement: above the maximum observed value in either condition for that region
                local_max = np.nanmax(np.r_[dfr["mean_int"].values, dfr["mean_ext"].values])
                y = local_max + pad
                
                add_sig_bracket(ax, x_int[i], x_ext[i], y=y, h=h, text=stars, fontsize=14)
                            
                            # Expand y-limits so brackets are not clipped
            ax.set_ylim(ymin, ymax + 0.15 * yr)
            
            # --- legend for box colors + paired lines (boxplot doesn't auto-legend) ---
            internal_patch = Patch(facecolor="blue", edgecolor="black", alpha=alpha_val, label="Internal")
            external_patch = Patch(facecolor="grey", edgecolor="black", alpha=alpha_val, label="External")
            line_handle = plt.Line2D([0], [0], color="k", lw=1.5, alpha=0.7, label="Paired patient")
            
            ax.legend(handles=[internal_patch, external_patch, line_handle], frameon=False, loc="best")
            
                    # Cosmetics
            
            region_labels_nice = ['Default - A','Dorsal Attention - A','Visual - Central']

            plt.xticks(x_base, region_labels_nice, rotation=0, ha="center",size = 9.5)
          # plt.ylim(-0.12, 0.15)
            plt.ylabel("Mean log power (norm) (per patient, region-aggregated)",size = 10)
            if title_style == 'analysis':
                plt.title(f"{vidname} | {freq_name} | {scheme} {thr_str} {win_len}| {atlas_name}")
            elif title_style == 'figure':
                plt.title(f"{vidname} Movie \n {freq_name}")

            plt.legend(frameon=True)
            plt.tight_layout()
            
            # Save
            regions_tag = "-".join([str(r).replace(" ", "") for r in regions])
            out = os.path.join(fig_dir, f"{movie}_{atlas}_{freq_band}_{regions_tag}_{win_len}_pairedbars_lines.png")
            plt.savefig(out, dpi=300)
            plt.show()
            
            
            
            mean_delta_matrix_filtered = mean_delta_matrix.loc[
                mean_delta_matrix.index.intersection(ordered_atlas_groups)
            ].reindex(ordered_atlas_groups)
    
        
            # Force numeric
            # --- Align p-values to the matrix we plot ---
            p_plot = p_matrix.reindex(
                index=mean_delta_matrix_filtered.index,
                columns=mean_delta_matrix_filtered.columns
            ).astype(float)
            
            flat = p_plot.to_numpy().ravel()
            mask = ~np.isnan(flat)
            pvals_raw = flat[mask]
            
            # Initialize corrected matrices with NaNs (same shape as p_plot)
            p_plot_fdr = p_plot.copy()
            p_plot_bonf = p_plot.copy()
            
            print(f"\n--- {movie} | {atlas} ---")
            for region in ordered_atlas_groups:
                if region not in patient_region_diffs:
                    continue
                deltas = np.asarray(patient_region_diffs[region], float)
                deltas = deltas[~np.isnan(deltas)]
                if len(deltas) == 0:
                    continue
                print(region, "n=", len(deltas),
                      "mean=", np.mean(deltas),
                      "median=", np.median(deltas),
                      "std=", np.std(deltas),
                      "sign flips=", np.sum(deltas>0), "/", len(deltas))
                p_unc = p_matrix.loc[region, freq_band] if region in p_matrix.index else np.nan
            
                print(
                    region,
                    f"n={len(deltas):2d}",
                    f"mean={np.mean(deltas): .4f}",
                    f"median={np.median(deltas): .4f}",
                    f"std={np.std(deltas): .4f}",
                    f"p_unc={p_unc: .4f}",
                    f"sign+={np.sum(deltas > 0)}/{len(deltas)}"
                )
    
            
            if pvals_raw.size > 0:
                # FDR
                _, pvals_fdr = fdrcorrection(pvals_raw, alpha=0.05)
                flat_fdr = p_plot_fdr.to_numpy().ravel()
                flat_fdr[mask] = pvals_fdr
                p_plot_fdr.iloc[:, :] = flat_fdr.reshape(p_plot_fdr.shape)
            
                # Bonferroni
                pvals_bonf = np.minimum(pvals_raw * len(pvals_raw), 1.0)
                flat_bonf = p_plot_bonf.to_numpy().ravel()
                flat_bonf[mask] = pvals_bonf
                p_plot_bonf.iloc[:, :] = flat_bonf.reshape(p_plot_bonf.shape)
            
            # --- Annotations from corrected p-values ---
            sig_thresh = 0.05  # or 0.05
            annotations_filtered = (p_plot_fdr < sig_thresh).map(lambda x: "*" if x else "")
            annotations_unc = (p_plot < 0.05).map(lambda x: "*" if x else "")
    
            # --- Save for inspection (aligned to plotted rows/cols) ---
            p_plot_fdr.to_csv(os.path.join(fig_dir, f"{movie}_{atlas}_pvals_fdr_ALIGNED.csv"))
            p_plot_bonf.to_csv(os.path.join(fig_dir, f"{movie}_{atlas}_pvals_bonferroni_ALIGNED.csv"))
            
            #fig, ax = plt.subplots(figsize=(5, 7))
            fig, ax = plt.subplots(figsize=(4, 3))

            sns.heatmap(
                mean_delta_matrix_filtered.astype(float),
                annot=annotations_unc,
                fmt='',
                cmap='coolwarm',
                center=0,
                cbar_kws={'label': 'Mean Δ across patients (Int − Ext)'},
                linewidths=0.5,
                linecolor='gray',
                ax=ax
            )
            
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10, va='center')
            
            ax.set_title(
                f"{freq_name} | within-patient int-ext in {atlas_name}\n"
                f"{vidname} Movie | {PC_type} | {scheme} | threshold:{threshold}",
                fontsize=10
            )
            ax.set_xlabel("Frequency Band")
            ax.set_ylabel("Brain Region / Network")
            
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, f"{movie}_{atlas}_{PC_type}_{freq_range}_{win_len}_meanDelta_matrix.png"), dpi=300)
            plt.show()
    
    
            # Save matrices to CSV files
            if lfp_type == 'power':
                mean_delta_matrix.to_csv(os.path.join(
                    fig_dir, f'{movie}_{atlas}_{PC_type}_{freq_range}_{thr_str}_meanDelta_acrossPatients.csv'
                ))
                n_pat_matrix.to_csv(os.path.join(
                    fig_dir, f'{movie}_{atlas}_{PC_type}_{freq_range}_{thr_str}_nPatients_acrossPatients.csv'
                ))
                p_matrix.to_csv(os.path.join(
                    fig_dir, f'{movie}_{atlas}_{PC_type}_{freq_range}_{thr_str}_pvals_onesamp.csv'
                ))
                t_matrix.to_csv(os.path.join(
                    fig_dir, f'{movie}_{atlas}_{PC_type}_{freq_range}_{thr_str}_tvals_onesamp.csv'
                ))
    
    

        
    patient_delta_df = pd.DataFrame(patient_delta_rows)
    electrode_df = pd.DataFrame(electrode_rows)

    return patient_delta_df, electrode_df

all_threshold_results = []

for threshold in threshold_grid:
    print("\n" + "="*100)
    print(f"RUNNING THRESHOLD {threshold}")
    print("="*100)

    patient_delta_df, electrode_df = run_analysis_for_threshold(threshold)

    patient_delta_df["threshold_low"] = threshold[0]
    patient_delta_df["threshold_high"] = threshold[1]
    patient_delta_df["thr_str"] = f"[{threshold[0]:.1f}, {threshold[1]:.1f}]"

    all_threshold_results.append(patient_delta_df)

all_threshold_df = pd.concat(all_threshold_results, ignore_index=True)

all_threshold_df.to_csv(
    os.path.join(fig_dir, "patient_level_region_deltas_all_thresholds.csv"),
    index=False
)

