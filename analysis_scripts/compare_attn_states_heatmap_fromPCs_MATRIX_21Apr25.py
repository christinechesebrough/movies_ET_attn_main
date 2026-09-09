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
from scipy.stats import median_abs_deviation

plt.rcParams['font.family'] = 'sans-serif'   # or 'serif'
plt.rcParams['font.sans-serif'] = ['Arial']  # or 'Helvetica', 'DejaVu Sans'


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
freq_bands = ['theta','alpha','beta','gamma','HFA']  # Example: alpha and HFA bands
#freq_bands = ['alpha','HFA']

lfp_type = 'power'

entropy_type = 'mssd'

movies = ['despicable_me_hungarian']#,'despicable_me_english'] #inscapes

PC_type = 'sep_PCs'#'shared_PC1'#'shared_PC1'#'sep_PCs'#'shared_PC1'

scheme = 'within_timepoint'#'within_timepoint'

atlases = ['Y17_Atlas_Region']#,'Y7_Atlas_Region','Y17_Atlas_Region']'AparcAseg_Atlas_Region'
#atlases = ['AparcAseg_Atlas_Region']
method = 'power_log'

title_type = 'figure' #analysis


# Choose which z-scoring approach to use: 'individual' or 'group'
z_scoring_approach = 'individual'  # Options: 'individual' or 'group'

# Note: These thresholds are used for documentation only as the actual thresholds
# are now determined by examining_shared_PC_features.py

# Define threshold label - this should match the one used in examining_shared_PC_features.py
#threshold_label = "PC1z_high0.75_low0.5_dev_high.7_low.3"  # Update this to match your actual thresholds

win_len = 10
threshold = (.6,.6)

thr_str = f"[{threshold[0]:.1f}, {threshold[1]:.1f}]"

#if PC_type == 'shared_PC1':
#    z_threshold = z_threshold_both
    
#if PC_type == 'sep_PCs':
#    z_threshold = z_threshold_both

correction_method = 'fdr'  # Options: 'fdr', 'bonferroni', or 'none'

machine_path = 'media/christine'


# Output directory
if lfp_type == 'entropy': 
    fig_dir = os.path.join(f'/{machine_path}/Samsung/Movie_data/PC_derived_freq_x_atlas_matrices_ENTROPY/')
    
if lfp_type == 'power':
    if PC_type == 'sep_PCs':
        if win_len == 10:
            fig_dir = os.path.join(f'/{machine_path}/Samsung/Movie_data/PC_derived_freq_x_atlas_matrices_7Jan26')
        elif win_len ==5:
            fig_dir = os.path.join(f'/{machine_path}/Samsung/Movie_data/PC_derived_freq_x_atlas_matrices_5s_6Apr26')

        os.makedirs(fig_dir, exist_ok=True)
    elif PC_type == 'shared_PC1':
        if win_len == 10:
            fig_dir = os.path.join(f'/{machine_path}/Samsung/Movie_data/PC_derived_freq_x_atlas_matrices_shared_PC1_10s_20Apr26')
        if win_len == 5:
            fig_dir = os.path.join(f'/{machine_path}/Samsung/Movie_data/PC_derived_freq_x_atlas_matrices_shared_PC1_5s_20Apr26')

        os.makedirs(fig_dir, exist_ok=True)

for movie in movies:
        
    bad_windows_dir = f'/{machine_path}/Samsung/Movie_data/1secEpochs_for_review_power_log_{movie}_HFA_1Apr26'
   # bad_windows_dir = '/{machine_path}/Samsung/Movie_data/1secEpochs_for_review_power_log_despicable_me_hungarian_HFA_1Apr26
    
   
    if movie == 'despicable_me_english':
        vidname = 'Narrative'
        #patients = ['NS127_02','NS135','NS137','NS136','NS138','NS140','NS153','NS164','NS166','NS174_02']
        entry_ids = ['NS127_02_ses-02_run-01', 
                     'NS135_ses-01_run-01',
                       'NS136_ses-01_run-01', 
                       'NS137_ses-01_run-01',
                       'NS138_ses-01_run-01', 
                       'NS140_ses-01_run-01',
                       #'NS140_02_ses-02_run-01', 
                       'NS153_ses-01_run-01',
                       'NS155_02_ses-02_run-01', 
                       'NS164_ses-01_run-01',
                       #'NS166_ses-01_run-01', 
                       'NS174_02_ses-02_run-01',
                       'NS174_03_ses-03_run-01', 
                       'NS178_ses-01_run-01',
                      # 'NS190_ses-01_run-01',
                       'NS190_ses-01_run-02',
                       'NS191_ses-01_run-01', 
                     #  'NS193_ses-01_run-01',
                       'NS193_ses-01_run-02',
                       'NS194_ses-01_run-01',
                       'NS205_ses-01_run-01']
    
        
        # Load the new combined features file
        if PC_type == 'shared_PC1':
            #et_file = '/{machine_path}/Samsung/Movie_data/combined_PC_features_output_21Apr25/combined_features_despicable_me_english.csv'
              # et_file = f'/{machine_path}/Samsung/Movie_data/shared_PC_features_9Jan26/{movie}_features_df_{thr_str}.csv'
               et_file = f'/{machine_path}/Samsung/Movie_data/shared_PC_features_{win_len}s_20Apr26/{movie}_features_df_{thr_str}.csv'
        elif PC_type == "sep_PCs":
           #et_file = f'/{machine_path}/Samsung/Movie_data/combined_PC_features_output_21Apr25/despicable_me_english_features_df_{z_threshold}.csv'
           if win_len == 5:
              et_file = f'/{machine_path}/Samsung/Movie_data/separate_PC_features_5s_8Apr26/{movie}_features_df_{thr_str}.csv'
           elif win_len == 10:
               et_file = f'/{machine_path}/Samsung/Movie_data/separate_PC_features_7Jan26/{movie}_features_df_{thr_str}.csv'
        # Read et file as df
        et_df = pd.read_csv(et_file)
        factor = 'PC1'
        
        
    if movie == 'despicable_me_hungarian':
        vidname = 'Narrative - Uncomprehended Language'
        entry_ids =  [
         'LH010_ses-01_run-01',
          'NS127_02_ses-02_run-01', 
          'NS135_ses-01_run-01',
          'NS136_ses-01_run-01', 
          'NS137_ses-01_run-01',
          'NS138_ses-01_run-01', 
          'NS140_ses-01_run-01',
          'NS140_02_ses-02_run-01',
        #'NS145_ses-02_run-01', 
         'NS154_ses-01_run-01',
          'NS164_ses-01_run-01', 
          'NS174_02_ses-02_run-01',
          'NS174_03_ses-03_run-01', 
          'NS178_ses-01_run-01'
               ]
        # Load the new combined features file
        if PC_type == 'shared_PC1':
            #et_file = '/{machine_path}/Samsung/Movie_data/combined_PC_features_output_21Apr25/combined_features_despicable_me_english.csv'
              # et_file = f'/{machine_path}/Samsung/Movie_data/shared_PC_features_9Jan26/{movie}_features_df_{thr_str}.csv'
               et_file = f'/{machine_path}/Samsung/Movie_data/shared_PC_features_{win_len}s_20Apr26/{movie}_features_df_{thr_str}.csv'
        elif PC_type == "sep_PCs":
           #et_file = f'/{machine_path}/Samsung/Movie_data/combined_PC_features_output_21Apr25/despicable_me_english_features_df_{z_threshold}.csv'
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
        entry_ids = ['NS127_02_ses-02_run-01', 
                    'NS135_ses-01_run-01',
                    'NS136_ses-01_run-01', 
                    'NS137_ses-01_run-01',
                    'NS138_ses-01_run-01', 
                    'NS140_ses-01_run-01',
                    'NS140_02_ses-02_run-01',
                    'NS151_ses-01_run-01', 
                    'NS153_ses-01_run-01',
                    'NS155_ses-01_run-01',
                    'NS155_02_ses-02_run-01', 
                    'NS164_ses-01_run-01',
                   # 'NS178_ses-01_run-01', 
                    'NS205_ses-01_run-01',
                    'NS210_ses-01_run-01'
                    ] 
        
        # Load the new combined features file
        if PC_type == 'shared_PC1':
            #et_file = '/{machine_path}/Samsung/Movie_data/combined_PC_features_output_21Apr25/combined_features_inscapes.csv'
           # et_file = f'/{machine_path}/Samsung/Movie_data/shared_PC_features_9Jan26/{movie}_features_df_{thr_str}.csv'
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
            atlas_name = 'AparcAseg Atlas'
        
        # To be filled during analysis
        # Initialize matrices with atlas groups as index and frequency bands as columns
        atlas_groups = []  # Will store unique atlas groups
        t_matrix = pd.DataFrame()
        p_matrix = pd.DataFrame()
        mean_int_matrix = pd.DataFrame()
        mean_ext_matrix = pd.DataFrame()
        
        # Loop through all frequency bands, movies, and atlases
        for freq_band in freq_bands:
            if freq_band == 'HFA':
                freq_range = '70 - 150 Hz'
                freq_name = 'HFA (50-150 Hz)'
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
                #data_dir = f'/{machine_path}/Samsung/Movie_data/{freq_band}_all_{movie}_all_cortContacts_wLabels_y17_4Mar25/'
               # data_dir = f'/{machine_path}/Samsung/Movie_data/{freq_band}_{movie}_4Jan26'
               # data_dir = f'/{machine_path}/Samsung/Movie_data/windowed_power_log_{movie}_{freq_band}_26Mar26'
#               data_dir = f'/{machine_path}/Samsung/Movie_data/windowed_unnormed_{method}_{movie}_{freq_band}_1Apr26'
                if win_len == 10:              
                    data_dir = f'/{machine_path}/Samsung/Movie_data/windowed_power_10s/windowed_unnormed_{method}_{movie}_{freq_band}_1Apr26'
                elif win_len == 5:
                    data_dir = f'/{machine_path}/Samsung/Movie_data/windowed_power_5s/windowed_unnormed_{method}_{movie}_{freq_band}_5sec_5Apr26'
            elif lfp_type == 'entropy':
                data_dir = f'/{machine_path}/Samsung/Movie_data/entropy_extracted/{freq_band}_all_{movie}_all_cortContacts_entropy_29Mar25'

            # Initialize dictionary to store data for each atlas group across all conditions
            atlas_group_data = {}
    
          #  available_patients = [entry for entry in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, entry))]
           # filtered_patients = [entry for entry_ids in available_patients if patient in patients]
           
            # Loop through patients
            for rec in entry_ids:
                pat = extract_pat_id(rec)
                run = extract_run_label(rec)
                
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

                pat_dir = os.path.join(data_dir, pat)
                # Find the CSV files containing atlas info and lfp data
                if lfp_type == 'power':
                    lfp_files = [filename for filename in os.listdir(pat_dir) if f'{method}' in filename and filename.endswith('.csv')]
                elif lfp_type == 'entropy':
                    lfp_files = [filename for filename in os.listdir(pat_dir) if f'{entropy_type}' in filename and filename.endswith('.csv')]

                if not lfp_files:
                    print(f"No LFP files found for patient {pat} in {pat_dir}")
                    continue
                                
                if pat in ['NS190', 'NS193']:
                    matches = pick_file(
                        lfp_files,
                        contains_any=[run],
                        startswith_not="._"
                    )
                    if len(matches) == 0:
                        print(f"No matching file found for {pat} {run}")
                        continue
                    lfp_file = matches[0]
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
                # Find the patient column regardless of case
                patient_col = next(
                    c for c in et_df.columns if c.lower() == 'patient'
                )
                
                et_patient_rows = et_df[et_df[patient_col] == rec].reset_index(drop=True)

                # Extract indices based on the chosen z-scoring approach
                # if z_scoring_approach == 'individual':
                #     int_idx = et_patient_rows[et_patient_rows['Attention_Label_Individual'] == 'Internal_HighDeviation'].index.to_numpy()
                #     ext_idx = et_patient_rows[et_patient_rows['Attention_Label_Individual'] == 'External_LowDeviation'].index.to_numpy()
                # else:  # group z-scoring
                #     int_idx = et_patient_rows[et_patient_rows['Attention_Label_Group'] == 'Internal_HighDeviation'].index.to_numpy()
                #     ext_idx = et_patient_rows[et_patient_rows['Attention_Label_Group'] == 'External_LowDeviation'].index.to_numpy()
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
        
                # Group electrodes by atlas and process each group
                atlas_row = atlas_rows[atlas_rows['Atlas'] == atlas].iloc[0]
                grouped_cols = {}
                for col in electrode_cols:
                    atlas_value = atlas_row[col]
                    if atlas_value not in grouped_cols:
                        grouped_cols[atlas_value] = []
                    grouped_cols[atlas_value].append(col)

                # Exclude regions based on the atlas
                if atlas == 'Y7_Atlas_Region':
                    excluded_regions = ['FreeSurfer_Defined_Medial_Wall']
                elif atlas == 'Y17_Atlas_Region':
                    excluded_regions = ['FreeSurfer_Defined_Medial_Wall']
                elif atlas == 'DK_Atlas_Region':
                    excluded_regions = ['bankssts']
                elif atlas == 'AparcAseg_Atlas_Region':
                    excluded_regions = []

                # Process each atlas group
                for atlas_group, cols in grouped_cols.items():
                    if atlas_group in excluded_regions:
                        print(f"Skipping excluded region: {atlas_group}")
                        continue

                    aggregated_int_values = []
                    aggregated_ext_values = []

                    for elec in cols:
                        if elec in data_rows.columns:
                            col_data = data_rows[elec].astype(float).values
                            int_vals = col_data[int_idx]
                            ext_vals = col_data[ext_idx]

                            aggregated_int_values.extend(int_vals)
                            aggregated_ext_values.extend(ext_vals)

                    if atlas_group not in atlas_group_data:
                        atlas_group_data[atlas_group] = {}
                    
                    if 'int' not in atlas_group_data[atlas_group]:
                        atlas_group_data[atlas_group]['int'] = []
                    if 'ext' not in atlas_group_data[atlas_group]:
                        atlas_group_data[atlas_group]['ext'] = []

                    atlas_group_data[atlas_group]['int'].extend(aggregated_int_values)
                    atlas_group_data[atlas_group]['ext'].extend(aggregated_ext_values)

                    
            # Initialize variables to store t-values and significance
            atlas_groups = []
            t_values = []
            significance = []
            mean_int_values = []
            mean_ext_values = []
            p_values = []  # Add p-values list
        
            # Loop through the atlas groups to calculate t-values
            for atlas_group, conditions_data in atlas_group_data.items():
                int_values = conditions_data['int']  # Removed conditions[0] reference
                ext_values = conditions_data['ext']  # Removed conditions[0] reference
                mean_int = np.mean(int_values)
                mean_ext = np.mean(ext_values)
                
                mean_int_values.append(mean_int)
                mean_ext_values.append(mean_ext)
        
                # Perform t-test on int vs ext values
                t_stat, p_val = ttest_ind(int_values, ext_values, nan_policy='omit')
        
                atlas_groups.append(atlas_group)
                t_values.append(t_stat)
                p_values.append(p_val)  # Store p-value
        
                # Add significance marker
                if p_val < 0.001:
                    significance.append('*')
                else:
                    significance.append('')

            # Initialize matrices if they're empty
            if t_matrix.empty:
                t_matrix = pd.DataFrame(index=atlas_groups, columns=freq_bands)
                p_matrix = pd.DataFrame(index=atlas_groups, columns=freq_bands)
                mean_int_matrix = pd.DataFrame(index=atlas_groups, columns=freq_bands)
                mean_ext_matrix = pd.DataFrame(index=atlas_groups, columns=freq_bands)

            # Add values to matrices
            t_matrix[freq_band] = t_values
            p_matrix[freq_band] = p_values
            mean_int_matrix[freq_band] = mean_int_values
            mean_ext_matrix[freq_band] = mean_ext_values

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
          # ordered_atlas_groups = [ 'Default A','Default B', 'Default C',  'Somatomotor A', 'Somatomotor B', 'Dorsal Attention A', 'Dorsal Attention B', 'Salience / Ventral Attention A', 'Salience / Ventral Attention B', 'Limbic A', 'Limbic B', 'Control C', 'Control A', 'Control B', 'Visual Central (Visual A)', 'Visual Peripheral (Visual B)', 'Temporal Parietal']
            ordered_atlas_groups = ['Default A', 'Dorsal Attention A', 'Visual Central (Visual A)']
        elif atlas == 'AparcAseg_Atlas_Region':
            ordered_atlas_groups = t_matrix.index.tolist()
            ordered_atlas_groups = ['Right-Hippocampus','Left-Hippocampus','ctx-rh-entorhinal','ctx-lh-entorhinal','ctx-rh-entorhinal','ctx-rh-precuneus','ctx-lh-precuneus']
            

                    
             # Reorder/filter t matrix
        t_matrix_filtered = t_matrix.loc[t_matrix.index.intersection(ordered_atlas_groups)]
        t_matrix_filtered = t_matrix_filtered.reindex(ordered_atlas_groups)
       # t_matrix_filtered = t_matrix_filtered.iloc[::-1]
        
        # Apply the exact same filtering/reordering to p matrix
        p_matrix_filtered = p_matrix.loc[p_matrix.index.intersection(ordered_atlas_groups)]
        p_matrix_filtered = p_matrix_filtered.reindex(ordered_atlas_groups)
      #  p_matrix_filtered = p_matrix_filtered.iloc[::-1]
        
        # Force numeric
        p_matrix_filtered = p_matrix_filtered.astype(float)
        
        # # Flatten filtered p-values for correction
        # flat_pvals = p_matrix_filtered.values.flatten()
        # mask = ~np.isnan(flat_pvals)
        # pvals_raw = flat_pvals[mask]
        
        # # FDR correction
        # _, pvals_fdr = fdrcorrection(pvals_raw, alpha=0.05)
        
        # # Put corrected p-values back into filtered matrix
        # p_corrected_matrix_filtered = p_matrix_filtered.copy()
        # p_corrected_matrix_filtered.values.flat[mask] = pvals_fdr
        
        # # Annotation matrix
        # significance_mask = p_corrected_matrix_filtered < 0.001
        # annotations_filtered = significance_mask.applymap(lambda x: '*' if x else '')

        # # FDR correction
        # _, pvals_fdr = fdrcorrection(pvals_raw, alpha=0.05)
        
        # # Bonferroni correction
        # pvals_bonf = np.minimum(pvals_raw * len(pvals_raw), 1.0)
        
        # # Create corrected matrices
        # p_matrix_fdr = p_matrix.copy()
        # p_matrix_bonf = p_matrix.copy()
        
        # p_matrix_fdr.values.flat[mask] = pvals_fdr
        # p_matrix_bonf.values.flat[mask] = pvals_bonf
        
        # # Save for inspection
        # p_matrix_fdr.to_csv(os.path.join(fig_dir, f"{movie}_{atlas}_{win_len}_pvals_fdr.csv"))
        # p_matrix_bonf.to_csv(os.path.join(fig_dir, f"{movie}_{atlas}_{win_len}_pvals_bonferroni.csv"))
        
        
        # -------------------------------------------------
        # Multiple-comparisons correction on FILTERED matrix
        # -------------------------------------------------
        flat = p_matrix_filtered.to_numpy().ravel()
        mask = ~np.isnan(flat)
        pvals_raw = flat[mask]
        
        # FDR correction
        _, pvals_fdr = fdrcorrection(pvals_raw, alpha=0.05)
        
        # Bonferroni correction
        pvals_bonf = np.minimum(pvals_raw * len(pvals_raw), 1.0)
        
        # Put corrected p-values back into FILTERED matrices
        p_plot_fdr = p_matrix_filtered.copy()
        p_plot_bonf = p_matrix_filtered.copy()
        
        p_plot_fdr_vals = p_plot_fdr.to_numpy().copy()
        p_plot_bonf_vals = p_plot_bonf.to_numpy().copy()
        
        p_plot_fdr_vals.ravel()[mask] = pvals_fdr
        p_plot_bonf_vals.ravel()[mask] = pvals_bonf
        
        p_plot_fdr.iloc[:, :] = p_plot_fdr_vals
        p_plot_bonf.iloc[:, :] = p_plot_bonf_vals
        
        # Save for inspection
        p_plot_fdr.to_csv(os.path.join(fig_dir, f"{movie}_{atlas}_{win_len}_pvals_fdr.csv"))
        p_plot_bonf.to_csv(os.path.join(fig_dir, f"{movie}_{atlas}_{win_len}_pvals_bonferroni.csv"))
        
        # Choose which corrected matrix to annotate on the heatmap
        # p_annot = p_plot_bonf
        p_annot = p_plot_fdr
        
        def p_to_stars(p):
            if pd.isna(p):
                return ''
            elif p < 0.001:
                return '***'
            elif p < 0.01:
                return '**'
            elif p < 0.05:
                return '*'
            else:
                return ''
        
        annotations_filtered = p_annot.applymap(p_to_stars)


        fig, ax = plt.subplots(figsize=(4, 3))
        #fig,ax = plt.subplots(figsize=(6, 6))
        sns.heatmap(
            t_matrix_filtered.astype(float),
            annot=annotations_filtered,
            fmt='',
            cmap='coolwarm',
            center=0,
            vmin=-15,
            vmax=15,
            cbar_kws={'label': 't-value'},
            linewidths=0.5,
            linecolor='gray',
            ax=ax
        )
        
        # Fix label alignment
        ax.set_yticklabels(
            ax.get_yticklabels(),
            rotation=0,
            fontsize=8,
            va='center'
        )
        if lfp_type == 'power':
            if title_type == 'analysis':
                ax.set_title(f"Power Int-Ext States (t) {freq_name} in {atlas_name} \n{vidname} Movie | {PC_type} {scheme} {win_len}s windows| int-ext threshold:{threshold}", fontsize=6)
            elif title_type == 'figure':
                ax.set_title(f"{vidname} Movie \nLFP Power Int-Ext States \n{freq_name} ", fontsize=10)

            #  ax.set_title(f"{vidname}", fontsize=12)
        elif lfp_type == 'entropy':
            ax.set_title(f"Entropy Differences Int-Ext Attention States (t_vals) in {atlas_name} {scheme} \{vidname} Movie | {entropy_type} Entropy | {z_scoring_approach} z-scoring", fontsize=12)
        ax.set_xlabel("Frequency Band")
       # ax.set_ylabel("Functional Network")
        
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"{movie}_{atlas}_{PC_type}_{freq_range}_{threshold}_{scheme}_{win_len}_matrix.png"), dpi=300)
        plt.show()

        #p_corrected_matrix_filtered = p_corrected_matrix.reindex(t_matrix_filtered.index)
        #p_corrected_matrix_filtered.to_csv(os.path.join(fig_dir, f"{movie}_{atlas}_pvals_corrected.csv"))

        # # Save matrices to CSV files
        # if lfp_type == 'power':
        #     t_matrix.to_csv(os.path.join(fig_dir, f'{movie}_{atlas}_{PC_type}_{freq_range}_{threshold}_zscoring_t_matrix.csv'))
        #     p_matrix.to_csv(os.path.join(fig_dir, f'{movie}_{atlas}_{PC_type}_{threshold}_zscoring_p_matrix.csv'))
        #     mean_int_matrix.to_csv(os.path.join(fig_dir, f'{movie}_{atlas}_{PC_type}_{z_scoring_approach}_{threshold}_int_matrix.csv'))
        #     mean_ext_matrix.to_csv(os.path.join(fig_dir, f'{movie}_{atlas}_{PC_type}_{z_scoring_approach}_{threshold}_ext_matrix.csv'))
        # elif lfp_type == 'entropy':
        #     t_matrix.to_csv(os.path.join(fig_dir, f'{movie}_{atlas}_{PC_type}_{entropy_type}_{z_scoring_approach}_zscoring_entropy_t_matrix.csv'))
        #     p_matrix.to_csv(os.path.join(fig_dir, f'{movie}_{atlas}_{PC_type}_{entropy_type}_{z_scoring_approach}_zscoring_entropy_p_matrix.csv'))
        #     mean_int_matrix.to_csv(os.path.join(fig_dir, f'{movie}_{atlas}_{PC_type}_{entropy_type}_{z_scoring_approach}_zscoring_entropy_mean_int_matrix.csv'))
        #     mean_ext_matrix.to_csv(os.path.join(fig_dir, f'{movie}_{atlas}_{PC_type}_{entropy_type}_{z_scoring_approach}_zscoring_entropy_mean_ext_matrix.csv'))


