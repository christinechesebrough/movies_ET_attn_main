#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 2024

@author: christinechesebrough
"""

import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats




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

# Define y-axis limits 
y_axis_limits = {
    'HFA': {'ymin': -0.4, 'ymax': 0.5},  # Fixed limits for HFA
    'alpha': {'ymin': -0.4, 'ymax': 0.5},
    'theta_alpha': {'ymin': -0.5, 'ymax': 0.5},
    'all_gamma': {'ymin': -0.5, 'ymax': 0.5}# Fixed limits for alpha
}

# Lists of frequency bands, movies, and atlases
freq_bands = ['alpha','HFA']  # Example: alpha and HFA bands
movies = ['despicable_me_english','inscapes']  # ['inscapes']
atlases = ['Y17_Atlas_Region']#, 'DK_Atlas_Region']  # ['DK_Atlas_Region', 'Y7_Atlas_Region']
PC_type = 'sep_PCs'

# Add entropy-related variables
lfp_type = 'power'  # Options: 'power' or 'entropy'
entropy_type = 'ziv'  # Options: 'perm' or 'samp'

method = 'power_z'

threshold = (0.5,0.5)

thr_str = f"[{threshold[0]:.1f}, {threshold[1]:.1f}]"

machine_path = 'media/christine'

#z_threshold_high = z_threshold[0]
#z_threshold_low = z_threshold[1]
#z_threshold_both = .5

# if PC_type == 'shared_PC1':
#     z_threshold = z_threshold_both
    
# if PC_type == 'sep_PCs':
#     z_threshold = z_threshold_both

# Output directory
if lfp_type == 'entropy': 
    fig_dir = os.path.join(f'/{machine_path}/Samsung/Movie_data/PC_derived_freq_x_atlas_matrices_ENTROPY/')
elif lfp_type == 'power':
    fig_dir = os.path.join(f'/{machine_path}/Samsung/Movie_data/PC_derived_freq_x_atlas_matrices_7Jan26')
os.makedirs(fig_dir, exist_ok=True)


z_scoring_approach = 'individual'

# Loop through all frequency bands, movies, and atlases
for freq_band in freq_bands:
    if freq_band == 'HFA':
        freq_range = '70 - 150 Hz'
        freq_name = 'HFA (50-150 Hz)'
    elif freq_band == 'alpha':
        freq_range = '8 - 13 Hz'
        freq_name = 'Alpha (8-13 Hz)'
    elif freq_band == 'all_gamma':
        freq_range = '31-150 Hz'
        freq_name = 'Gamma (31-150 Hz)'
    elif freq_band == 'theta_alpha':
        freq_range = '4-13 Hz'
        freq_name = 'theta - alpha (3 - 14 Hz)'
        
    for movie in movies:
        if movie == 'despicable_me_english':
            vidname = 'Narrative Movie'
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
                         #  'NS166_ses-01_run-01', 
                           'NS174_02_ses-02_run-01',
                           'NS174_03_ses-03_run-01', 
                           'NS178_ses-01_run-01',
                           'NS190_ses-01_run-01',
                           'NS190_ses-01_run-02',
                           'NS191_ses-01_run-01', 
                           'NS193_ses-01_run-01',
                           'NS193_ses-01_run-02',
                           'NS194_ses-01_run-01',
                           'NS205_ses-01_run-01']
        
            # Load the new combined features file
            if PC_type == 'shared_PC1':
                #et_file = '/Volumes/Samsung/Movie_data/combined_PC_features_output_21Apr25/combined_features_despicable_me_english.csv'
                et_file = f'/{machine_path}/Samsung/Movie_data/shared_PC_features_9Jan26/{movie}_features_df_{thr_str}.csv'
            elif PC_type == "sep_PCs":
                et_file = f'/{machine_path}/Samsung/Movie_data/separate_PC_features_7Jan26/{movie}_features_df_{thr_str}.csv'
            # Read et file as df
            et_df = pd.read_csv(et_file)
            factor = 'PC1'
    
        elif movie == 'inscapes':
            vidname = 'Ambient Movie'
            #patients = ['NS127_02','NS135','NS136','NS137','NS138','NS140','NS140_02','NS164','NS166']
            entry_ids = ['NS127_02_ses-02_run-01', 
                         'NS135_ses-01_run-01',
                         'NS136_ses-01_run-01', 
                         'NS137_ses-01_run-01',
                         'NS138_ses-01_run-01', 
                         'NS140_ses-01_run-01',
                         'NS140_02_ses-02_run-01',
                         #'NS144_ses-01_run-01',
                         'NS151_ses-01_run-01', 
                           'NS153_ses-01_run-01',
                           #'NS154_ses-01_run-01', 
                           'NS155_ses-01_run-01',
                           'NS155_02_ses-02_run-01', 
                           'NS164_ses-01_run-01',
                           #'NS178_ses-01_run-01', 
                           'NS205_ses-01_run-01',
                           'NS210_ses-01_run-01'] #'NS211_ses-01_run-01']
            
               # Load the new combined features file
            if PC_type == 'shared_PC1':
                #et_file = '/Volumes/Samsung/Movie_data/combined_PC_features_output_21Apr25/combined_features_inscapes.csv'
                et_file = f'/{machine_path}/Samsung/Movie_data/shared_PC_features_9Jan26/{movie}_features_df_{thr_str}.csv'
            elif PC_type == "sep_PCs":
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
    
            # Directory paths
            # define the data directory inside the frequency band loop
            if lfp_type == 'power':
                #data_dir = f'/Volumes/Samsung/Movie_data/{freq_band}_all_{movie}_all_cortContacts_wLabels_y17_4Mar25/'
                data_dir = f'/{machine_path}/Samsung/Movie_data/{freq_band}_{movie}_4Jan26'
            elif lfp_type == 'entropy':
                data_dir = f'/{machine_path}/Samsung/Movie_data/entropy_extracted/{freq_band}_all_{movie}_all_cortContacts_entropy_29Mar25'

            
            conditions = ['PCAs']  # ['sacc_verg']#['isc_verg_sacc']
            
            # Initialize dictionary to store data for each atlas group
            atlas_group_data = {}
    
            # Loop through conditions
            for condition in conditions:
                # Get the list of patients in the directory and filter based on the predefined patients list
               # available_patients = [entry for entry in os.listdir(mne_data_dir) if os.path.isdir(os.path.join(mne_data_dir, entry))]
               # filtered_patients = [patient for patient in available_patients if patient in patients]
                for rec in entry_ids:
                    pat = extract_pat_id(rec)
                    run = extract_run_label(rec)
                    
                    pat_dir = os.path.join(data_dir, pat)
                    # Find the CSV files containing atlas info and lfp data
                    if lfp_type == 'power':
                        lfp_files = [filename for filename in os.listdir(pat_dir) if f'{method}' in filename and filename.endswith('.csv')]
                    elif lfp_type == 'entropy':
                        lfp_files = [filename for filename in os.listdir(pat_dir) if f'{entropy_type}' in filename and filename.endswith('.csv')]

                    if not lfp_files:
                        print(f"No LFP files found for patient {pat} in {pat_dir}")
                        continue

                    if pat in ['NS190','NS193']:
                        
                        lfp_file = pick_file(
                            lfp_files,
                            contains_any=run,
                            startswith_not="._"
                        )
                    lfp_file = lfp_files[0]  # Select the first matching file
                    full_lfp_path = os.path.join(pat_dir, lfp_file)

                    try:
                        lfp_values = pd.read_csv(full_lfp_path)
                        print(f"Processed LFP data for: {full_lfp_path}")
                    except Exception as e:
                        print(f"Error reading {full_lfp_path}: {e}")
                        continue

                # Loop through patients
                # for pat in filtered_patients:
                #     pat_dir = os.path.join(mne_data_dir, pat)
                    
                #     # Find the CSV files containing atlas info and lfp data
                #     if lfp_type == 'power':
                #         lfp_files = [filename for filename in os.listdir(pat_dir) if filename.endswith('.csv')]
                #     elif lfp_type == 'entropy':
                #         lfp_files = [filename for filename in os.listdir(pat_dir) if f'{entropy_type}' in filename and filename.endswith('.csv')]
                    
                #     if not lfp_files:
                #         print(f"No LFP files found for patient {pat} in {pat_dir}")
                #         continue
    
                #     lfp_file = lfp_files[0]  # Select the first matching file
                #     full_lfp_path = os.path.join(pat_dir, lfp_file)
    
                #     try:
                #         lfp_values = pd.read_csv(full_lfp_path)
                #         print(f"Processed LFP data for: {full_lfp_path}")
                #     except Exception as e:
                #         print(f"Error reading {full_lfp_path}: {e}")
                #         continue
    
                    atlas_rows = lfp_values.head(4)
                    data_rows = lfp_values.iloc[4:].reset_index(drop=True)
                    
                    # Find and extract the rows corresponding to the patient 'pat' in et_file
                    et_patient_rows = et_df[et_df['patient'] == rec].reset_index(drop=True)
    
                        
                    # Find the columns titled PC1_High_GroupDemeaned_Indiv and PC1_Low_GroupDemeaned_Indiv
                    # if PC_type == 'old_way':
                    #      # Find the columns titled PC1_High_GroupDemeaned_Indiv and PC1_Low_GroupDemeaned_Indiv
                    #     if f'{factor}_Int_GroupDemeaned_Indiv' in et_patient_rows.columns and f'{factor}_Ext_GroupWindow_Indiv' in et_patient_rows.columns:
                    #         # Get boolean masks for internal and external indices
                    #         #int_idx = et_patient_rows[et_patient_rows[f'{factor}_High_GroupDemeaned_Indiv'] == True].index.to_numpy()
                    #         #ext_idx = et_patient_rows[et_patient_rows[f'{factor}_Low_GroupDemeaned_Indiv'] == True].index.to_numpy()
                            
                    #         int_idx = et_patient_rows[et_patient_rows[f'{factor}_Int_GroupDemeaned_Indiv'] == True].index.to_numpy()
                    #         ext_idx = et_patient_rows[et_patient_rows[f'{factor}_Ext_GroupWindow_Indiv'] == True].index.to_numpy()
                    
                    # # Extract indices based on the chosen z-scoring approach
                    # if z_scoring_approach == 'individual':
                    #     int_idx = et_patient_rows[et_patient_rows['Attention_Label_Individual'] == 'Internal_HighDeviation'].index.to_numpy()
                    #     ext_idx = et_patient_rows[et_patient_rows['Attention_Label_Individual'] == 'External_LowDeviation'].index.to_numpy()
                    # else:  # group z-scoring
                    #     int_idx = et_patient_rows[et_patient_rows['Attention_Label_Group'] == 'Internal_HighDeviation'].index.to_numpy()
                    #     ext_idx = et_patient_rows[et_patient_rows['Attention_Label_Group'] == 'External_LowDeviation'].index.to_numpy()
        
                    int_idx = et_patient_rows[et_patient_rows['Attention_Label_[0.5, 0.5]'] == 'Internal_HighConfidence'].index.to_numpy()
                    ext_idx = et_patient_rows[et_patient_rows['Attention_Label_[0.5, 0.5]'] == 'External_HighConfidence'].index.to_numpy()
                       
    
                   ## else:
                    #    print(f"Warning: Required columns not found in {et_file} for patient {pat}")
                    #    int_idx, ext_idx = [], []
                    

                    # Normalize electrode columns if required
                    electrode_cols = [col for col in lfp_values.columns if col[0] in ['L', 'R'] and col[1:].isalnum()]
                    if normalize:
                        data_rows[electrode_cols] = data_rows[electrode_cols].apply(pd.to_numeric, errors='coerce')
                        data_rows[electrode_cols] = (data_rows[electrode_cols] - data_rows[electrode_cols].mean()) / data_rows[electrode_cols].std()

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
                    
                        if condition not in atlas_group_data[atlas_group]:
                            atlas_group_data[atlas_group][condition] = {
                                'int': [], 'ext': [], 'num_contacts': 0  # <<< ADDED
                            }
                    
                        atlas_group_data[atlas_group][condition]['int'].extend(aggregated_int_values)
                        atlas_group_data[atlas_group][condition]['ext'].extend(aggregated_ext_values)
                        atlas_group_data[atlas_group][condition]['num_contacts'] += len(cols)  # <<< ADDED

            # Initialize variables to store means and standard errors
            atlas_groups = []
            int_means = []
            ext_means = []
            int_sems = []
            ext_sems = []
            significance = []
            contact_counts = []  # <<< ADDED

            
            # Loop through the atlas groups to calculate means and standard errors
            for atlas_group, conditions_data in atlas_group_data.items():
                int_values = conditions_data[conditions[0]]['int']
                ext_values = conditions_data[conditions[0]]['ext']
                num_contacts = conditions_data[conditions[0]]['num_contacts']  # <<< ADDED
                
                int_mean = np.mean(int_values)
                ext_mean = np.mean(ext_values)
                int_sem = stats.sem(int_values)
                ext_sem = stats.sem(ext_values)
                
                atlas_groups.append(atlas_group)
                int_means.append(int_mean)
                ext_means.append(ext_mean)
                int_sems.append(int_sem)
                ext_sems.append(ext_sem)
                
                # Perform t-test
                t_stat, p_val = stats.ttest_ind(int_values, ext_values, nan_policy='omit')
                
                # Add significance marker
                if p_val < 0.001:
                    significance.append('*')
                else:
                    significance.append('')
                
                contact_counts.append(num_contacts)  # <<< ADDED

            
            # Create a DataFrame for plotting
            plot_data = pd.DataFrame({
                'Atlas Group': atlas_groups,
                'Internal Mean': int_means,
                'External Mean': ext_means,
                'Internal SEM': int_sems,
                'External SEM': ext_sems,
                'Significance': significance,
                'Contacts': contact_counts  # <<< ADDED
            })

            # Apply specific ordering based on atlas
            if atlas == 'DK_Atlas_Region':
                network_order = {
                    'Default Mode Network (DMN)': ['medialorbitofrontal', 'parsopercularis', 'rostralanteriorcingulate', 'rostralmiddlefrontal', 'parahippocampal', 'temporalpole', 'posteriorcingulate', 'precuneus', 'inferiorparietal'],
                    'Somatomotor Network (SMN)': ['postcentral', 'precentral', 'insula', 'isthmuscingulate'],
                    'Dorsal Attention Network (DAN)': ['superiorfrontal', 'caudalmiddlefrontal', 'superiorparietal'],
                    'Ventral Attention Network (VAN)': ['lateralorbitofrontal', 'parsorbitalis', 'parstriangularis', 'supramarginal', 'transversetemporal'],
                    'Limbic Network (LN)': ['entorhinal'],
                    'Frontoparietal Network (FPN)': ['superiorfrontal', 'caudalmiddlefrontal'],
                    'Visual Network (VN)': ['fusiform', 'inferiortemporal', 'middletemporal', 'lateraloccipital', 'lingual', 'pericalcarine', 'cuneus', 'paracentral']
                }
                
                # Flatten the network order into a list
                ordered_atlas_groups = [region for network, regions in network_order.items() for region in regions]
                
                # Filter and reorder the plot data
                plot_data_filtered = plot_data.loc[plot_data['Atlas Group'].isin(ordered_atlas_groups)]
                plot_data_filtered = plot_data_filtered.set_index('Atlas Group').reindex(ordered_atlas_groups).reset_index()
            
            elif atlas == 'Y7_Atlas_Region':
                network_order = ['Default', 'Somatomotor', 'Dorsal Attention', 'Ventral Attention', 'Limbic', 'Frontoparietal', 'Visual']
                
                # Filter and reorder the plot data
                plot_data_filtered = plot_data.loc[plot_data['Atlas Group'].isin(network_order)]
                plot_data_filtered = plot_data_filtered.set_index('Atlas Group').reindex(network_order).reset_index()
                
            elif atlas == 'Y17_Atlas_Region':
                network_order = [ 'Default A','Default B', 'Default C',  'Somatomotor A', 'Somatomotor B', 'Dorsal Attention A', 'Dorsal Attention B', 'Salience / Ventral Attention A', 'Salience / Ventral Attention B', 'Limbic A', 'Limbic B', 'Control C', 'Control A', 'Control B', 'Visual Central (Visual A)', 'Visual Peripheral (Visual B)', 'Temporal Parietal']
                #network_order = ['Default A', 'Dorsal Attention A', 'Visual Central (Visual A)']
                # Filter and reorder the plot data
                plot_data_filtered = plot_data.loc[plot_data['Atlas Group'].isin(network_order)]
                plot_data_filtered = plot_data_filtered.set_index('Atlas Group').reindex(network_order).reset_index()
                
            
            plot_data_filtered = plot_data_filtered.iloc[::-1].reset_index(drop=True)  
            
            
            # Create horizontal grouped bar plot
            plt.figure(figsize=(7, max(6, len(plot_data_filtered) * 0.5)))
            
            y = np.arange(len(plot_data_filtered))
            height = 0.5
            
            # Internal bars
            plt.barh(
                y - height/2,
                plot_data_filtered['Internal Mean'],
                height,
                xerr=plot_data_filtered['Internal SEM'],
                capsize=6,
                label='Internal',
                color='red'
            )
            
            # External bars
            plt.barh(
                y + height/2,
                plot_data_filtered['External Mean'],
                height,
                xerr=plot_data_filtered['External SEM'],
                capsize=6,
                label='External'
            )
            
            # Add significance markers
            for i, sig in enumerate(plot_data_filtered['Significance']):
                if sig == '*':
                    internal_max = (
                        plot_data_filtered['Internal Mean'].iloc[i]
                        + plot_data_filtered['Internal SEM'].iloc[i]
                    )
                    external_max = (
                        plot_data_filtered['External Mean'].iloc[i]
                        + plot_data_filtered['External SEM'].iloc[i]
                    )
                    x_max = max(internal_max, external_max)
            
                    plt.text(
                        x_max + 0.03,
                        i,
                        '*',
                        va='center',
                        ha='left',
                        color='red',
                        fontsize=20
                    )
            
            # Add contact count to labels
            plot_data_filtered['Label'] = plot_data_filtered.apply(
                lambda row: f"{row['Atlas Group']} (n={row['Contacts']})",
                axis=1
            )
            
            plt.yticks(y, plot_data_filtered['Label'], fontsize=10)
            
            # Axis labels
            if lfp_type == 'power':
                plt.xlabel(f'Normalized {freq_name} Power')
                plt.title(f"{vidname} - {freq_name}")
            elif lfp_type == 'entropy':
                plt.xlabel(f'{freq_name} {entropy_type} Entropy')
                plt.title(f"{vidname} - {freq_name} ({entropy_type} Entropy)")
            
            plt.legend()
            plt.grid(False)
            
            # Consistent x-axis limits
            plt.xlim(
                y_axis_limits[freq_band]['ymin'],
                y_axis_limits[freq_band]['ymax'] + 0.2
            )
            
            plt.tight_layout()

            # Save
            if lfp_type == 'power':
                plt.savefig(
                    os.path.join(fig_dir, f'{atlas}_{movie}_{freq_range}_{threshold}_{method}_bargraph_horizontal.png'),
                    dpi=300,
                    bbox_inches='tight'
                )
            elif lfp_type == 'entropy':
                plt.savefig(
                    os.path.join(fig_dir, f'{atlas}_{movie}_{freq_range}_{threshold}_{entropy_type}_entropy_bargraph_horizontal.png'),
                    dpi=300,
                    bbox_inches='tight'
                )
            
            plt.show()

            # # Create vertical grouped bar plot
            # plt.figure(figsize=(max(8, len(plot_data_filtered) * 0.6), 6))
            
            # # Set up x positions
            # x = np.arange(len(plot_data_filtered))
            # width = 0.38
            
            # # Internal bars
            # plt.bar(
            #     x - width/2,
            #     plot_data_filtered['Internal Mean'],
            #     width=width,
            #     yerr=plot_data_filtered['Internal SEM'],
            #     capsize=6,
            #     label='Internal',
            #     color='red'
            # )
            
            # # External bars
            # plt.bar(
            #     x + width/2,
            #     plot_data_filtered['External Mean'],
            #     width=width,
            #     yerr=plot_data_filtered['External SEM'],
            #     capsize=6,
            #     label='External'
            # )
            
            # # Add significance markers
            # for i, sig in enumerate(plot_data_filtered['Significance']):
            #     if sig == '*':
            #         internal_top = (
            #             plot_data_filtered['Internal Mean'].iloc[i]
            #             + plot_data_filtered['Internal SEM'].iloc[i]
            #         )
            #         external_top = (
            #             plot_data_filtered['External Mean'].iloc[i]
            #             + plot_data_filtered['External SEM'].iloc[i]
            #         )
            #         y_max = max(internal_top, external_top)
            #         plt.text(i, y_max + 0.03, '*', ha='center', va='bottom', color='red', fontsize=20)
            
            # # Add contact count to x-axis labels
            # plot_data_filtered['Label'] = plot_data_filtered.apply(
            #     lambda row: f"{row['Atlas Group']}\n(n={row['Contacts']})", axis=1
            # )
            
            # # Labels and title
            # if lfp_type == 'power':
            #     plt.ylabel(f'Normalized {freq_name} Power')
            #     plt.title(f"{vidname} - {freq_name}")
            # elif lfp_type == 'entropy':
            #     plt.ylabel(f'{freq_name} {entropy_type} Entropy')
            #     plt.title(f"{vidname} - {freq_name} ({entropy_type} Entropy)")
            
            # plt.xticks(x, plot_data_filtered['Label'], rotation=45, ha='right', fontsize=9)
            # plt.legend()
            # plt.grid(False)
            
            # # Y-axis limits
            # plt.ylim(
            #     y_axis_limits[freq_band]['ymin'],
            #     y_axis_limits[freq_band]['ymax'] + 0.2
            # )
            
            # plt.tight_layout()
            
            # # Save
            # if lfp_type == 'power':
            #     plt.savefig(
            #         os.path.join(
            #             fig_dir,
            #             f'{atlas}_{movie}_{freq_range}_{threshold}_{method}_vertical_bargraph.png'
            #         ),
            #         dpi=300,
            #         bbox_inches='tight'
            #     )
            # elif lfp_type == 'entropy':
            #     plt.savefig(
            #         os.path.join(
            #             fig_dir,
            #             f'{atlas}_{movie}_{freq_range}_{threshold}_{entropy_type}_entropy_vertical_bargraph.png'
            #         ),
            #         dpi=300,
            #         bbox_inches='tight'
            #     )
            
            # plt.show()

#%%

import matplotlib.colors as mcolors

agg = plot_data_filtered

agg['delta'] =agg['Internal Mean']-agg['External Mean']

vals = agg['delta'].to_numpy(float)

# symmetric scaling (robust)
v = np.nanpercentile(np.abs(vals), 95)
v = float(v) if np.isfinite(v) and v > 0 else float(np.nanmax(np.abs(vals)))
norm = mcolors.TwoSlopeNorm(vmin=-v, vcenter=0.0, vmax=v)

cmap = plt.get_cmap("coolwarm")

agg["parc_id"] = agg["Atlas Group"].map(name_to_17)


# plot_brain_surf expects actual colors per parcel (strings or rgba)
# safest: give RGBA tuples
agg["rgba"] = [cmap(norm(x)) for x in vals]

parcs_to_show = agg["parc_id"].tolist()
parc_colors   = agg["rgba"].tolist()

elec_names, coords, elec_hem, elec_vals = vis_df_to_plot_brain_inputs(
    vis_df, movie=movie, freq_band=freq_band, atlas=atlas, value_col=value_col, abs_val=False
)

fig = plot_brain_surf(
    elec_names=elec_names,
    coords=coords,
    elec_hem=elec_hem.tolist(),
    elec_colors="k",
    elec_size=0.01,
    cmap="coolwarm",
    cbar=False,
   # cbar_title=value_col,
    #cbar_minmax=(-np.nanpercentile(np.abs(elec_vals),95), np.nanpercentile(np.abs(elec_vals),95)),
    subject="fsaverage",
    subjects_dir=recon_dir,
    surf="inflated",
    views=[["l_lateral", "l_medial"], ["r_lateral", "r_medial"]],
    parc="y17",
    parcs_to_show=parcs_to_show,
    parc_colors=parc_colors,
    parc_alpha=1,        # translucent parcels so electrodes remain visible
    parc_borders=False,
    clear_overlay=True,
    title=f"{movie} {freq_band} {atlas}: parcels + electrodes",
)

out_path = os.path.join(
    img_dir,
    f"{movie}_{freq_band}_{atlas}_{value_col}_parcels_plus_electrodes.png"
)

fig.savefig(out_path, dpi=300, bbox_inches="tight")
print("Saved:", out_path)

