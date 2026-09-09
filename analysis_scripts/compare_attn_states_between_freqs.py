#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 09:44:43 2025

@author: christinechesebrough
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 20:51:45 2024
Modified to loop through all frequency bands, movies, atlases, and conditions.
@author: christinechesebrough
"""

import os, re, sys
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
normalize =False

# Lists of frequency bands, movies, and atlases
freq_bands = ['alpha','HFA']  # Example: alpha and HFA bands
#freq_bands = ['theta_alpha','all_gamma']
#freq_bands = ['alpha','HFA']

lfp_type = 'power_z'
entropy_type = 'perm'

movies = ['inscapes']  #['inscapes']
PC_type = 'sep_PCs'#'shared_PC1'#'sep_PCs'#'shared_PC1'
atlases = ['Y7_Atlas_Region']#['DK_Atlas_Region', 'Y7_Atlas_Region']

method = 'rolling_avg'

z_threshold = [.75,.75]
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
    fig_dir = os.path.join('/Volumes/Samsung/Movie_data/FC_test_7Jan26_betweenFreqs')
    
os.makedirs(fig_dir, exist_ok=True)

        
for movie in movies:
    if movie == 'despicable_me_english':
        vidname = 'Narrative'
        patients = ['NS127_02','NS135','NS137','NS136','NS138','NS140','NS153','NS164','NS166','NS174_02']
        #et_file = f'/Volumes/Samsung/Movie_data/more_normed_gaze_features_23Dec24/despicable_me_english_PC2_HiLo_count_{z_threshold_high}_high_{z_threshold_low}_low.csv'
        if PC_type == 'shared_PC1':
            et_file = '/Volumes/Samsung/Movie_data/combined_PC_features_output_12May25_robust_pca/combined_features_despicable_me_english.csv'
        elif PC_type == "sep_PCs":
            et_file = f'/Volumes/Samsung/Movie_data/separate_PC_features_7Jan26/despicable_me_english_features_df_[0.5, 0.5].csv'
        # Read et file as df
        et_df = pd.read_csv(et_file)
        factor = 'PC1'

    elif movie == 'inscapes':
        vidname = 'Ambient'
        #patients = ['NS127_02','NS135','NS136','NS137','NS138','NS140','NS140_02','NS164','NS166']
        patients = ['NS127_02','NS135','NS136','NS137','NS138','NS140','NS140_02','NS164','NS178', "NS205", 'NS210','NS211'] #'NS166',

        #et_file = f'/Volumes/Samsung/Movie_data/more_normed_gaze_features_23Dec24/inscapes_PC1_HiLo_count_{z_threshold_high}_high_{z_threshold_low}_low.csv'
        #et_file = '/Volumes/Samsung/Movie_data/more_normed_gaze_features_12Mar25/inscapes_PC1_IntExt_count_1_0.5.csv'
        if PC_type == 'shared_PC1':
            et_file = '/Volumes/Samsung/Movie_data/combined_PC_features_output_12May25_robust_pca/combined_features_inscapes.csv'
        elif PC_type == "sep_PCs":
            et_file = '/Volumes/Samsung/Movie_data/separate_PC_features_7Jan26/inscapes_features_df_[0.5, 0.5].csv'
        # Read et file as df
        et_df = pd.read_csv(et_file)
        factor = 'PC1'


#%%
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
        # Instead of looping over freq_bands, we want to load both HFA and alpha for each subject
        # and correlate them within each electrode
        # We'll do this for each subject, then group by region
        if set(freq_bands) >= set(['alpha', 'HFA']):
            for pat in patients:
                region = 'all'
                # Load alpha data
                alpha_dir = f"/Volumes/Samsung/Movie_data/{freq_band}_inscapes_10Dec25/{pat}"/Volumes/Samsung/Movie_data/alpha_despicable_me_english_4Jan26
                alpha_file = f"{pat}_{movie}_{region}_alpha_cortical_power_wLabels.csv"
                alpha_path = os.path.join(alpha_dir, alpha_file)
                # Load HFA data
                hfa_dir = f"/Volumes/Samsung/Movie_data/HFA_inscapes_10Dec25/{pat}"                
                hfa_file = f"{pat}_{movie}_{region}_HFA_cortical_power_wLabels.csv"
                hfa_path = os.path.join(hfa_dir, hfa_file)
                try:
                    alpha_values = pd.read_csv(alpha_path)
                    hfa_values = pd.read_csv(hfa_path)
                    print(f"Processed alpha and HFA data for: {pat} {movie}")
                except Exception as e:
                    print(f"Error reading alpha/HFA for {pat}: {e}")
                    continue
                # Get electrode columns
                electrode_cols = [col for col in alpha_values.columns if col[0] in ['L', 'R'] and col[1:].isalnum()]
                # Get atlas row for region mapping
                atlas_row = alpha_values.head(4)
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
                # Get data rows
                alpha_data = alpha_values.iloc[4:][electrode_cols].astype(float).values  # shape: (n_windows, n_electrodes)
                hfa_data = hfa_values.iloc[4:][electrode_cols].astype(float).values
                # Compute per-electrode correlation between HFA and alpha
                per_electrode_corrs = []
                for i, col in enumerate(electrode_cols):
                    if np.all(np.isnan(alpha_data[:, i])) or np.all(np.isnan(hfa_data[:, i])):
                        corr = np.nan
                    else:
                        corr = np.corrcoef(alpha_data[:, i], hfa_data[:, i])[0, 1]
                    per_electrode_corrs.append(corr)
                # Save per-electrode correlations
                per_electrode_df = pd.DataFrame({
                    'Electrode': electrode_cols,
                    'Region': channel_to_region,
                    'Corr_HFA_Alpha': per_electrode_corrs
                })
                out_dir = os.path.join(fig_dir, 'hfa_alpha_corr', f'{movie}_{atlas}')
                os.makedirs(out_dir, exist_ok=True)
                per_electrode_df.to_csv(os.path.join(out_dir, f'{pat}_corr_hfa_alpha_per_electrode.csv'), index=False)
                # Group by region
                region_summary = per_electrode_df.groupby('Region')['Corr_HFA_Alpha'].agg(['mean', 'std', 'count']).reset_index()
                region_summary.to_csv(os.path.join(out_dir, f'{pat}_corr_hfa_alpha_by_region.csv'), index=False)
                print(f"Saved per-electrode and per-region HFA/alpha correlations for {pat} ({movie})")

            # --- GROUP-LEVEL FC AGGREGATION ---
            # Remove old FC aggregation code (all_fc_int/all_fc_ext) as it is not used in this analysis
            # (No code needed here for per-electrode HFA/alpha correlation analysis)

        # Initialize variables to store t-values and significance
        # Remove old region-by-region t-test/statistics code that used atlas_group_data
        # (No code needed here for per-electrode HFA/alpha correlation analysis)

        # Loop through the atlas groups to calculate t-values
        # Remove old region-by-region t-test/statistics code that used atlas_group_data
        # (No code needed here for per-electrode HFA/alpha correlation analysis)

        # Initialize matrices if they're empty
        if t_matrix.empty:
            t_matrix = pd.DataFrame(index=atlas_groups, columns=freq_bands)
            p_matrix = pd.DataFrame(index=atlas_groups, columns=freq_bands)
            mean_int_matrix = pd.DataFrame(index=atlas_groups, columns=freq_bands)
            mean_ext_matrix = pd.DataFrame(index=atlas_groups, columns=freq_bands)

        # Add values to matrices
        # Remove old region-by-region t-test/statistics matrix code
        # (No code needed here for per-electrode HFA/alpha correlation analysis)

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
        diag_dir = os.path.join(fig_dir, 'fc_results', f'{movie}_{atlas}', 'diagnostics')
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

        # After all subjects for a movie/atlas are processed, aggregate region × subject matrix
        out_dir = os.path.join(fig_dir, 'hfa_alpha_corr', f'{movie}_{atlas}')
        import glob
        region_files = glob.glob(os.path.join(out_dir, '*_corr_hfa_alpha_by_region.csv'))
        all_region_corrs = []
        subject_ids = []
        for region_file in region_files:
            subject_id = os.path.basename(region_file).split('_')[0]
            subject_ids.append(subject_id)
            df = pd.read_csv(region_file)
            df = df.set_index('Region')
            all_region_corrs.append(df['mean'])
        if all_region_corrs:
            region_matrix = pd.concat(all_region_corrs, axis=1)
            region_matrix.columns = subject_ids
            region_matrix.to_csv(os.path.join(out_dir, f'region_subject_hfa_alpha_corr_matrix.csv'))
            print(f"Saved region × subject HFA/alpha correlation matrix to {os.path.join(out_dir, f'region_subject_hfa_alpha_corr_matrix.csv')}")
            # --- Group-level heatmap visualization ---
            import matplotlib.pyplot as plt
            import seaborn as sns
            plt.figure(figsize=(max(8, 0.5*region_matrix.shape[1]), 8))
            sns.heatmap(region_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, cbar_kws={'label': 'HFA/Alpha Correlation'})
            plt.title(f'HFA/Alpha Correlation by Region and Subject\n{movie} | {atlas}')
            plt.xlabel('Subject')
            plt.ylabel('Region')
            plt.tight_layout()
            heatmap_path = os.path.join(out_dir, f'region_subject_hfa_alpha_corr_matrix_heatmap.png')
            plt.savefig(heatmap_path, dpi=200)
           # plt.close()
            print(f"Saved group-level region × subject heatmap to {heatmap_path}")


