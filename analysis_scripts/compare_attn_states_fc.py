
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 20:51:45 2024
Modified to loop through all frequency bands, movies, atlases, and conditions.
@author: christinechesebrough
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import seaborn as sns  
from statsmodels.stats.multitest import fdrcorrection
import pickle  # For saving FC matrices
from collections import Counter
import glob

# Flag to control normalization
normalize =True

# Lists of frequency bands, movies, and atlases
freq_bands = ['alpha','HFA']  # Example: alpha and HFA bands
#freq_bands = ['theta_alpha','all_gamma']
#freq_bands = ['alpha','HFA']

lfp_type = 'power'
entropy_type = 'perm'

movies = ['despicable_me_english','inscapes']  #['inscapes']
PC_type = 'shared_PC1'#'shared_PC1'#'sep_PCs'#'shared_PC1'
atlases = ['Y17_Atlas_Region']#['DK_Atlas_Region', 'Y7_Atlas_Region']

method = 'rolling_avg'

z_threshold = [.5,1.25]
z_threshold_high = z_threshold[0]
z_threshold_low = z_threshold[1]
z_threshold_both = .5

z_thresh = '0.5'
z_thresh_int = 0.5
z_thresh_ext = 0.5

#if PC_type == 'shared_PC1':
#    z_threshold = z_threshold_both
    
#if PC_type == 'sep_PCs':
#    z_threshold = z_threshold_both

correction_method = 'bonferroni'  # Options: 'fdr', 'bonferroni', or 'none'


# Output directory
if lfp_type == 'entropy':
    fig_dir = os.path.join('/Volumes/Samsung/Movie_data/PC_derived_freq_x_atlas_matrices_ENTROPY_robustPCA/')
elif lfp_type == 'power':
    fig_dir = os.path.join('/Volumes/Samsung/Movie_data/FC_test_22Jul25')
    
os.makedirs(fig_dir, exist_ok=True)

        
for movie in movies:
    if movie == 'despicable_me_english':
        vidname = 'Narrative'
        patients = ['NS127_02','NS135','NS137','NS136','NS138','NS140','NS153','NS164','NS166','NS174_02']
        #et_file = f'/Volumes/Samsung/Movie_data/more_normed_gaze_features_23Dec24/despicable_me_english_PC2_HiLo_count_{z_threshold_high}_high_{z_threshold_low}_low.csv'
        if PC_type == 'shared_PC1':
            et_file = '/Volumes/Samsung/Movie_data/combined_PC_features_output_12May25_robust_pca/combined_features_despicable_me_english.csv'
        elif PC_type == "sep_PCs":
            et_file = f'/Volumes/Samsung/Movie_data/combined_PC_features_27Mar25/despicable_me_english_features_df_{z_threshold}.csv'
        # Read et file as df
        et_df = pd.read_csv(et_file)
        factor = 'PC2'

    elif movie == 'inscapes':
        vidname = 'Ambient'
        patients = ['NS127_02','NS135','NS136','NS137','NS138','NS140','NS140_02','NS164','NS166']

        #et_file = f'/Volumes/Samsung/Movie_data/more_normed_gaze_features_23Dec24/inscapes_PC1_HiLo_count_{z_threshold_high}_high_{z_threshold_low}_low.csv'
        #et_file = '/Volumes/Samsung/Movie_data/more_normed_gaze_features_12Mar25/inscapes_PC1_IntExt_count_1_0.5.csv'
        if PC_type == 'shared_PC1':
            et_file = '/Volumes/Samsung/Movie_data/combined_PC_features_output_12May25_robust_pca/combined_features_inscapes.csv'
        elif PC_type == "sep_PCs":
            et_file = f'/Volumes/Samsung/Movie_data/combined_PC_features_27Mar25/inscapes_features_df_{z_threshold}.csv'
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
                freq_range = '31-69 Hz'
                freq_name = 'Gamma (31-50 Hz)'
            elif freq_band == 'delta':
                freq_range = '1-3 Hz'
                freq_name = "Delta (1-3 Hz)"
            elif freq_band == 'all_gamma':
                freq_range = '31-150 Hz'
                freq_name = 'Gamma (31-150 Hz)'
            elif freq_band == 'theta_alpha':
                freq_range = '4-13 Hz'
                freq_name = 'theta - alpha (3 - 14 Hz)'
                
            # define the data directory inside the frequency band loop
            if lfp_type == 'power':
                #mne_data_dir = f'/Volumes/Samsung/Movie_data/{freq_band}_newbands_{movie}_all_cortContacts_20May25/'
                mne_data_dir = f'/Volumes/Samsung/Movie_data/{freq_band}_newbands_normFirst_{movie}_all_cortContacts_fc_test_22Jul25'

            elif lfp_type == 'entropy':
                mne_data_dir = f'/Volumes/Samsung/Movie_data/{freq_band}_all_{movie}_all_cortContacts_entropy_29Mar25/entropy_extracted'
            # Initialize dictionary to store data for each atlas group across all conditions
            atlas_group_data = {}
    
            available_patients = [entry for entry in os.listdir(mne_data_dir) if os.path.isdir(os.path.join(mne_data_dir, entry))]
            filtered_patients = [patient for patient in available_patients if patient in patients]
                
            all_fc_int = []  # List of (n_channels, n_channels) FC matrices for internal state
            all_fc_ext = []  # List of (n_channels, n_channels) FC matrices for external state
            channel_labels_list = []  # To keep track of channel order for each subject
            subject_ids = []
            # Loop through patients
            for pat in filtered_patients:
                # Construct the correct path to the extracted power/entropy CSVs
                region = 'all'
                if lfp_type == 'power':
                    lfp_dir = f"/Volumes/Samsung/Movie_data/{freq_band}_newbands_normFirst_{movie}_all_cortContacts_fc_test_22Jul25/{pat}"
                    lfp_file = f"{pat}_{movie}_{region}_{freq_band}_cortical_power_wLabels.csv"
                elif lfp_type == 'entropy':
                    lfp_dir = f"/Volumes/Samsung/Movie_data/{freq_band}_all_{movie}_all_cortContacts_entropy_29Mar25/entropy_extracted/{pat}"
                    lfp_file = f"{pat}_{movie}_{region}_{freq_band}_cortical_{entropy_type}_wLabels.csv"
                full_lfp_path = os.path.join(lfp_dir, lfp_file)

                try:
                    lfp_values = pd.read_csv(full_lfp_path)
                    print(f"Processed LFP data for: {full_lfp_path}")
                except Exception as e:
                    print(f"Error reading {full_lfp_path}: {e}")
                    continue

                # Define electrode_cols immediately after loading lfp_values
                electrode_cols = [col for col in lfp_values.columns if col[0] in ['L', 'R'] and col[1:].isalnum()]

                # Define data_rows for this patient
                data_rows = lfp_values.iloc[4:].reset_index(drop=True)

                # Define et_patient_rows for this patient
                et_patient_rows = et_df[et_df['Patient'] == pat].reset_index(drop=True)

                # --- Dynamically select atlas row and region list based on 'atlas' variable ---
                atlas_row = lfp_values.head(4)
                region_list = []
                if atlas == 'Y7_Atlas_Region':
                    atlas_row_selected = atlas_row[atlas_row['Atlas'] == 'Y7_Atlas_Region']
                    region_list = ['Default', 'Somatomotor', 'Dorsal Attention', 'Ventral Attention', 'Limbic', 'Frontoparietal', 'Visual']
                elif atlas == 'Y17_Atlas_Region':
                    atlas_row_selected = atlas_row[atlas_row['Atlas'] == 'Y17_Atlas_Region']
                    region_list = [
                        'Default C', 'Default B', 'Default A', 'Somatomotor A', 'Somatomotor B',
                        'Dorsal Attention A', 'Dorsal Attention B', 'Salience / Ventral Attention A',
                        'Salience / Ventral Attention B', 'Limbic A', 'Limbic B', 'Control C',
                        'Control A', 'Control B', 'Visual Central (Visual A)', 'Visual Peripheral (Visual B)', 'Temporal Parietal'
                    ]
                elif atlas == 'DK_Atlas_Region':
                    atlas_row_selected = atlas_row[atlas_row['Atlas'] == 'DK_Atlas_Region']
                    # Example DK region list (replace with your actual DK regions as needed)
                    region_list = [
                        'medialorbitofrontal', 'parsopercularis', 'rostralanteriorcingulate', 'rostralmiddlefrontal',
                        'parahippocampal', 'temporalpole', 'posteriorcingulate', 'precuneus', 'inferiorparietal',
                        'postcentral', 'precentral', 'insula', 'isthmuscingulate', 'superiorfrontal',
                        'caudalmiddlefrontal', 'superiorparietal', 'lateralorbitofrontal', 'parsorbitalis',
                        'parstriangularis', 'supramarginal', 'transversetemporal', 'entorhinal', 'fusiform',
                        'inferiortemporal', 'middletemporal', 'lateraloccipital', 'lingual', 'pericalcarine',
                        'cuneus', 'paracentral'
                    ]
                else:
                    atlas_row_selected = pd.DataFrame()
                    region_list = []

                if not atlas_row_selected.empty:
                    channel_to_region = [atlas_row_selected[col].values[0] if col in atlas_row_selected.columns else 'Unknown' for col in electrode_cols]
                else:
                    channel_to_region = ['Unknown'] * len(electrode_cols)

                n_regions = len(region_list)

                # --- Diagnostic: Count contacts per region and correlations per region pair ---
                region_counts = Counter(channel_to_region)
                print(f"\nSubject {pat} ({movie}, {freq_band}, {atlas}): Contacts per region:")
                for reg in region_list:
                    print(f"  {reg}: {region_counts[reg]}")
                # Prepare to save as CSV
                contacts_df = pd.DataFrame({
                    'Region': region_list,
                    'Num_Contacts': [region_counts[reg] for reg in region_list]
                })
                # Correlations per region pair
                corr_counts = []
                for i, reg_i in enumerate(region_list):
                    n_i = region_counts[reg_i]
                    for j, reg_j in enumerate(region_list):
                        n_j = region_counts[reg_j]
                        if i == j:
                            n_corr = n_i * (n_i - 1) // 2
                        else:
                            n_corr = n_i * n_j
                        corr_counts.append({'Region1': reg_i, 'Region2': reg_j, 'Num_Correlations': n_corr})
                        print(f"  Region pair ({reg_i}, {reg_j}): {n_corr} correlations")
                corr_counts_df = pd.DataFrame(corr_counts)
                # Save diagnostics to CSV
                diag_dir = os.path.join(fig_dir, 'fc_results', f'{movie}_{atlas}_{freq_band}', 'diagnostics')
                os.makedirs(diag_dir, exist_ok=True)
                contacts_df.to_csv(os.path.join(diag_dir, f'{pat}_contacts_per_region.csv'), index=False)
                corr_counts_df.to_csv(os.path.join(diag_dir, f'{pat}_correlations_per_region_pair.csv'), index=False)

                # --- Expand attention labels to match short windows ---
                # Assume attention labels are in order for 236 long windows
                # and we have 4x as many short windows (e.g., 944)
                if PC_type == 'old_way':
                    if f'{factor}_Int_GroupDemeaned_Indiv' in et_patient_rows.columns and f'{factor}_Ext_GroupWindow_Indiv' in et_patient_rows.columns:
                        # Get boolean masks for internal and external indices
                        int_labels = et_patient_rows[f'{factor}_Int_GroupDemeaned_Indiv'].values
                        ext_labels = et_patient_rows[f'{factor}_Ext_GroupWindow_Indiv'].values
                        # Expand each label 4x
                        int_labels_expanded = np.repeat(int_labels, 4)
                        ext_labels_expanded = np.repeat(ext_labels, 4)
                        int_idx = np.where(int_labels_expanded == True)[0]
                        ext_idx = np.where(ext_labels_expanded == True)[0]
                elif PC_type == 'shared_PC1':
                    # find the column titled "Attention_Label_Individual"
                    attn_labels = et_patient_rows['Attention_Label_Individual'].values  # length 236
                    attn_labels_expanded = np.repeat(attn_labels, 4)  # length 944
                    int_idx = np.where(attn_labels_expanded == 'Internal_HighDeviation')[0]
                    ext_idx = np.where(attn_labels_expanded == 'External_LowDeviation')[0]
                elif PC_type == 'sep_PCs':
                    attn_labels = et_patient_rows[f'Attention_Label_{z_threshold}'].values
                    attn_labels_expanded = np.repeat(attn_labels, 4)
                    int_idx = np.where(attn_labels_expanded == 'Internal_HighConfidence')[0]
                    ext_idx = np.where(attn_labels_expanded == 'External_HighConfidence')[0]
                else:
                    print(f"Warning: Required columns not found in {et_file} for patient {pat}")
                    int_idx, ext_idx = [], []

                # --- FC ANALYSIS SECTION ---
                # Get electrode columns and data matrix
                if normalize:
                    data_rows[electrode_cols] = data_rows[electrode_cols].apply(pd.to_numeric, errors='coerce')
                    data_rows[electrode_cols] = (data_rows[electrode_cols] - data_rows[electrode_cols].mean()) / data_rows[electrode_cols].std()
                data_matrix = data_rows[electrode_cols].astype(float).values  # shape: (n_windows, n_channels)

                # Compute FC for internal and external windows (within subject)
                if len(int_idx) > 1 and len(ext_idx) > 1:
                    data_int = data_matrix[int_idx, :]
                    data_ext = data_matrix[ext_idx, :]
                    # Compute FC matrices (channels x channels)
                    fc_int = np.corrcoef(data_int, rowvar=False)
                    fc_ext = np.corrcoef(data_ext, rowvar=False)

                    # --- Aggregate to region-by-region FC (dynamic atlas) ---
                    region_fc_int = np.full((n_regions, n_regions), np.nan)
                    region_fc_ext = np.full((n_regions, n_regions), np.nan)
                    for i, reg_i in enumerate(region_list):
                        idx_i = [k for k, r in enumerate(channel_to_region) if r == reg_i]
                        for j, reg_j in enumerate(region_list):
                            idx_j = [k for k, r in enumerate(channel_to_region) if r == reg_j]
                            if idx_i and idx_j:
                                # Internal
                                submat_int = fc_int[np.ix_(idx_i, idx_j)]
                                region_fc_int[i, j] = np.nanmean(submat_int)
                                # External
                                submat_ext = fc_ext[np.ix_(idx_i, idx_j)]
                                region_fc_ext[i, j] = np.nanmean(submat_ext)
                    all_fc_int.append(region_fc_int)
                    all_fc_ext.append(region_fc_ext)
                    channel_labels_list.append(electrode_cols)
                    subject_ids.append(pat)
                else:
                    print(f"Not enough windows for FC computation for subject {pat} in {freq_band} {atlas}")

            # --- GROUP-LEVEL FC AGGREGATION ---
            # Only aggregate if at least one subject has FC data
            if all_fc_int and all_fc_ext:
                # Stack and compute mean region-level FC matrices
                mean_fc_int = np.nanmean(np.stack(all_fc_int, axis=0), axis=0)
                mean_fc_ext = np.nanmean(np.stack(all_fc_ext, axis=0), axis=0)
                diff_fc = mean_fc_int - mean_fc_ext
                # Save mean FC matrices and difference
                out_dir = os.path.join(fig_dir, 'fc_results', f'{movie}_{atlas}_{freq_band}')
                os.makedirs(out_dir, exist_ok=True)
                np.save(os.path.join(out_dir, 'mean_fc_int.npy'), mean_fc_int)
                np.save(os.path.join(out_dir, 'mean_fc_ext.npy'), mean_fc_ext)
                np.save(os.path.join(out_dir, 'diff_fc.npy'), diff_fc)
                # Save channel labels and subject IDs
                with open(os.path.join(out_dir, 'channel_labels.pkl'), 'wb') as f:
                    pickle.dump(channel_labels_list, f)
                with open(os.path.join(out_dir, 'subject_ids.pkl'), 'wb') as f:
                    pickle.dump(subject_ids, f)
                # Optional: visualize mean and difference FC matrices
                import matplotlib.pyplot as plt
                import seaborn as sns
                # Use the correct region list for plotting
                plt.figure(figsize=(8, 6))
                sns.heatmap(mean_fc_int, xticklabels=region_list, yticklabels=region_list, cmap='viridis', vmin=-1, vmax=1, cbar_kws={'label': 'FC (Internal)'})
                plt.title(f'Mean FC (Internal)\n{movie} | {atlas} | {freq_band}')
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, 'mean_fc_int_matrix.png'), dpi=200)
                plt.close()
                plt.figure(figsize=(8, 6))
                sns.heatmap(mean_fc_ext, xticklabels=region_list, yticklabels=region_list, cmap='viridis', vmin=-1, vmax=1, cbar_kws={'label': 'FC (External)'})
                plt.title(f'Mean FC (External)\n{movie} | {atlas} | {freq_band}')
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, 'mean_fc_ext_matrix.png'), dpi=200)
                plt.close()
                plt.figure(figsize=(8, 6))
                sns.heatmap(diff_fc, xticklabels=region_list, yticklabels=region_list, cmap='bwr', vmin=-1, vmax=1, cbar_kws={'label': 'FC (Internal - External)'})
                plt.title(f'Difference FC (Internal - External)\n{movie} | {atlas} | {freq_band}')
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, 'diff_fc_matrix.png'), dpi=200)
                plt.close()
            # Reset for next freq_band
            all_fc_int = []
            all_fc_ext = []
            channel_labels_list = []
            subject_ids = []

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
        ordered_atlas_groups = ['Default C', 'Default B', 'Default A', 'Somatomotor A','Somatomotor B', 'Dorsal Attention A', 'Dorsal Attention B', 'Salience / Ventral Attention A','Salience / Ventral Attention B', 'Limbic A','Limbic B', 'Control C', 'Control A', 'Control B', 'Visual Central (Visual A)','Visual Peripheral (Visual B)', 'Temporal Parietal']


                
        # Reorder both t_matrix and annotations
        t_matrix_filtered = t_matrix.loc[t_matrix.index.intersection(ordered_atlas_groups)]
        t_matrix_filtered = t_matrix_filtered.reindex(ordered_atlas_groups)
        
        # Force numeric matrix
        p_matrix = p_matrix.astype(float)
        
        # Flatten for corrections
        flat_pvals = p_matrix.values.flatten()
        mask = ~np.isnan(flat_pvals)
        pvals_raw = flat_pvals[mask]
        
        # FDR correction (main one for plotting)
        _, pvals_fdr = fdrcorrection(pvals_raw, alpha=0.05)
        
        # Create corrected p-value matrix (used for annotations)
        p_corrected_matrix = p_matrix.copy()
        p_corrected_matrix.values.flat[mask] = pvals_fdr

        # Reindex corrected matrix to match t_matrix_filtered
        p_corrected_matrix_filtered = p_corrected_matrix.reindex(t_matrix_filtered.index)
            
        # Create annotation matrix from significance threshold
        significance_mask = p_corrected_matrix_filtered < 0.001
        annotations_filtered = significance_mask.applymap(lambda x: '*' if x else '')
        
        # FDR correction
        _, pvals_fdr = fdrcorrection(pvals_raw, alpha=0.05)
        
        # Bonferroni correction
        pvals_bonf = np.minimum(pvals_raw * len(pvals_raw), 1.0)
        
        # Create corrected matrices
        p_matrix_fdr = p_matrix.copy()
        p_matrix_bonf = p_matrix.copy()
        
        p_matrix_fdr.values.flat[mask] = pvals_fdr
        p_matrix_bonf.values.flat[mask] = pvals_bonf
        
        # Save for inspection
        p_matrix_fdr.to_csv(os.path.join(fig_dir, f"{movie}_{atlas}_pvals_fdr.csv"))
        p_matrix_bonf.to_csv(os.path.join(fig_dir, f"{movie}_{atlas}_pvals_bonferroni.csv"))

        fig, ax = plt.subplots(figsize=(10, 14))
        sns.heatmap(
            t_matrix_filtered.astype(float),
            annot=annotations_filtered,
            fmt='',
            cmap='coolwarm',
            center=0,
            vmin=-.5,
            vmax=.5,
            cbar_kws={'label': 't-value'},
            linewidths=0.5,
            linecolor='gray',
            ax=ax
        )
        
        # Fix label alignment
        ax.set_yticklabels(
            ax.get_yticklabels(),
            rotation=0,
            fontsize=12,
            va='center'
        )
        if lfp_type == 'power':
            ax.set_title(f"Power Differences Int-Ext Attention States (t_vals) in {atlas_name} \n{vidname} Movie | {PC_type}| z_threshold = {z_thresh}",fontsize = 12)
        elif lfp_type == 'entropy':
            ax.set_title(f"Entropy Differences Int-Ext Attention States (t_vals) in {atlas_name} \n{vidname} Movie | {entropy_type} Entropy | z_threshold = {z_thresh}",fontsize = 12)
        ax.set_xlabel("Frequency Band")
        ax.set_ylabel("Brain Region / Network")
        
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"{movie}_{atlas}_{PC_type}_{entropy_type}_entropy_freq_x_region_matrix.png"), dpi=300)
        plt.show()

        
        #p_corrected_matrix_filtered = p_corrected_matrix.reindex(t_matrix_filtered.index)
        #p_corrected_matrix_filtered.to_csv(os.path.join(fig_dir, f"{movie}_{atlas}_pvals_corrected.csv"))

        # Save matrices to CSV files
        #if lfp_type == 'power':
        #    t_matrix.to_csv(os.path.join(fig_dir, f'{movie}_{atlas}_{PC_type}_t_matrix.csv'))
        #    p_matrix.to_csv(os.path.join(fig_dir, f'{movie}_{atlas}_{PC_type}_p_matrix.csv'))
        #    mean_int_matrix.to_csv(os.path.join(fig_dir, f'{movie}_{atlas}_{PC_type}_mean_int_matrix.csv'))
        #    mean_ext_matrix.to_csv(os.path.join(fig_dir, f'{movie}_{atlas}_{PC_type}_mean_ext_matrix.csv'))
        #elif lfp_type == 'entropy':
        #    t_matrix.to_csv(os.path.join(fig_dir, f'{movie}_{atlas}_{PC_type}_{entropy_type}_entropy_t_matrix.csv'))
        #    p_matrix.to_csv(os.path.join(fig_dir, f'{movie}_{atlas}_{PC_type}_{entropy_type}_entropy_p_matrix.csv'))
        #    mean_int_matrix.to_csv(os.path.join(fig_dir, f'{movie}_{atlas}_{PC_type}_{entropy_type}_entropy_mean_int_matrix.csv'))
        #    mean_ext_matrix.to_csv(os.path.join(fig_dir, f'{movie}_{atlas}_{PC_type}_{entropy_type}_entropy_mean_ext_matrix.csv'))

        # --- Aggregate diagnostics across all subjects for this movie/atlas/freq_band ---
        # Collect all per-subject contacts/correlations CSVs in the diagnostics dir
        diag_dir = os.path.join(fig_dir, 'fc_results', f'{movie}_{atlas}_{freq_band}', 'diagnostics')
        contact_files = glob.glob(os.path.join(diag_dir, '*_contacts_per_region.csv'))
        corr_files = glob.glob(os.path.join(diag_dir, '*_correlations_per_region_pair.csv'))
        # Aggregate contacts per region
        if contact_files:
            contacts_all = pd.concat([pd.read_csv(f) for f in contact_files], keys=contact_files, names=['SubjectFile'])
            contacts_summary = contacts_all.groupby('Region')['Num_Contacts'].agg(['mean', 'min', 'max', 'std', 'count']).reset_index()
            contacts_summary.to_csv(os.path.join(diag_dir, 'summary_contacts_per_region.csv'), index=False)
        # Aggregate correlations per region pair
        if corr_files:
            corrs_all = pd.concat([pd.read_csv(f) for f in corr_files], keys=corr_files, names=['SubjectFile'])
            corrs_summary = corrs_all.groupby(['Region1', 'Region2'])['Num_Correlations'].agg(['mean', 'min', 'max', 'std', 'count']).reset_index()
            corrs_summary.to_csv(os.path.join(diag_dir, 'summary_correlations_per_region_pair.csv'), index=False)


