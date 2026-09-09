#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 2024

@author: christinechesebrough
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Flag to control normalization
normalize = True

# Define y-axis limits 
y_axis_limits = {
    'HFA': {'ymin': -0.6, 'ymax': 0.5},  # Fixed limits for HFA
    'alpha': {'ymin': -0.6, 'ymax': 0.5},
    'theta_alpha': {'ymin': -0.6, 'ymax': 0.5},
    'all_gamma':{'ymin': -0.6, 'ymax': 0.6} # Fixed limits for alpha
}

# Lists of frequency bands, movies, and atlases
freq_bands = ['HFA']  # Example: alpha and HFA bands
movies = ['inscapes']  # ['inscapes']
atlases = ['Y17_Atlas_Region']  # ['DK_Atlas_Region', 'Y7_Atlas_Region']
PC_type = 'shared_PC1'

# Add entropy-related variables
lfp_type = 'power_z'  # Options: 'power' or 'entropy'
entropy_type = 'ziv'  # Options: 'perm' or 'samp'

method = 'rolling_avg'

# Output directory
if lfp_type == 'entropy': 
    fig_dir = os.path.join('/Volumes/Samsung/Movie_data/PC_derived_freq_x_atlas_matrices_ENTROPY/')
elif lfp_type == 'power':
    fig_dir = os.path.join('/Volumes/Samsung/Movie_data/PC_derived_freq_x_atlas_matrices_7Jan26')
os.makedirs(fig_dir, exist_ok=True)


#%%
def sanitize_filename(name):
    """Convert a string to a valid filename by replacing spaces and special characters"""
    # Replace spaces and special characters with underscores
    sanitized = name.replace(' ', '_').replace('/', '_').replace('\\', '_')
    # Remove any other potentially problematic characters
    sanitized = ''.join(c for c in sanitized if c.isalnum() or c in '._-')
    return sanitized

def analyze_power_differences(et_df, lfp_values, atlas_group, cols, threshold_col):
    """Analyze power differences for a specific threshold combination"""
    int_idx = et_df[et_df[threshold_col] == 'Internal_HighDeviation'].index.to_numpy()
    ext_idx = et_df[et_df[threshold_col] == 'External_LowDeviation'].index.to_numpy()
    
    # Add debug printing
    print(f"\nAnalyzing {atlas_group} for {threshold_col}")
    print(f"Number of internal indices: {len(int_idx)}")
    print(f"Number of external indices: {len(ext_idx)}")
    print(f"Number of electrodes: {len(cols)}")
    
    aggregated_int_values = []
    aggregated_ext_values = []
    
    for elec in cols:
        if elec in lfp_values.columns:
            col_data = lfp_values[elec].astype(float).values
            int_vals = col_data[int_idx]
            ext_vals = col_data[ext_idx]
            
            aggregated_int_values.extend(int_vals)
            aggregated_ext_values.extend(ext_vals)
    
    if not aggregated_int_values or not aggregated_ext_values:
        print(f"No data found for {atlas_group} in {threshold_col}")
        return None, None, None, None, None
    
    int_mean = np.mean(aggregated_int_values)
    ext_mean = np.mean(aggregated_ext_values)
    int_sem = stats.sem(aggregated_int_values)
    ext_sem = stats.sem(aggregated_ext_values)
    
    # Perform t-test
    t_stat, p_val = stats.ttest_ind(aggregated_int_values, aggregated_ext_values, nan_policy='omit')
    
    print(f"Results for {atlas_group}:")
    print(f"Internal mean: {int_mean:.3f}, External mean: {ext_mean:.3f}")
    print(f"p-value: {p_val:.3f}")
    
    return int_mean, ext_mean, int_sem, ext_sem, p_val

def create_threshold_heatmap(threshold_results, atlas_group, freq_name, vidname, fig_dir):
    """Create a heatmap showing how power differences vary across threshold combinations"""
    # Sanitize the atlas group name for the filename
    safe_atlas_name = sanitize_filename(atlas_group)
    
    # Extract all threshold combinations
    threshold_cols = list(threshold_results.keys())
    z_highs = []
    z_lows = []
    dev_percentiles = []
    
    for col in threshold_cols:
        parts = col.split('_')
        z_highs.append(float(parts[2].replace('z', '')))
        z_lows.append(float(parts[3]))
        dev_percentiles.append(float(parts[4].replace('dev', '')))
    
    z_highs = sorted(list(set(z_highs)))
    z_lows = sorted(list(set(z_lows)))
    dev_percentiles = sorted(list(set(dev_percentiles)))
    
    # Create matrices for internal and external means
    int_means = np.zeros((len(z_highs), len(z_lows), len(dev_percentiles)))
    ext_means = np.zeros((len(z_highs), len(z_lows), len(dev_percentiles)))
    p_values = np.zeros((len(z_highs), len(z_lows), len(dev_percentiles)))
    
    for i, z_high in enumerate(z_highs):
        for j, z_low in enumerate(z_lows):
            for k, dev in enumerate(dev_percentiles):
                col_name = f'Attention_State_z{z_high}_{z_low}_dev{dev}'
                if col_name in threshold_results and atlas_group in threshold_results[col_name]:
                    results = threshold_results[col_name][atlas_group]
                    if results['int_means']:
                        int_means[i,j,k] = np.mean(results['int_means'])
                        ext_means[i,j,k] = np.mean(results['ext_means'])
                        p_values[i,j,k] = np.mean(results['p_values'])
    
    # Create figure with subplots for each deviation percentile
    fig, axes = plt.subplots(1, len(dev_percentiles), figsize=(6*len(dev_percentiles), 5))
    if len(dev_percentiles) == 1:
        axes = [axes]
    
    # Set fixed color scale limits
    vmin, vmax = -0.5, 0.5
    
    for k, dev in enumerate(dev_percentiles):
        # Calculate power differences
        power_diff = int_means[:,:,k] - ext_means[:,:,k]
        
        # Create heatmap with fixed scale
        sns.heatmap(power_diff, 
                   ax=axes[k],
                   cmap='RdBu_r',
                   center=0,
                   vmin=vmin,
                   vmax=vmax,
                   xticklabels=[f'z_low={z}' for z in z_lows],
                   yticklabels=[f'z_high={z}' for z in z_highs],
                   annot=True,
                   fmt='.2f',
                   cbar_kws={'label': 'Power Difference (Internal - External)'})
        
        # Add significance markers
        for i in range(len(z_highs)):
            for j in range(len(z_lows)):
                if p_values[i,j,k] < 0.001:
                    axes[k].text(j + 0.5, i + 0.5, '*', 
                               ha='center', va='center', 
                               color='black', fontsize=12)
        
        axes[k].set_title(f'Deviation Percentile: {dev}')
        axes[k].set_xlabel('Z Low Threshold')
        axes[k].set_ylabel('Z High Threshold')
        
        # Add threshold information to the plot
        info_text = f'Internal: PC1 > z_high AND dev > {dev}\nExternal: PC1 < -z_low AND dev < {1-dev}'
        axes[k].text(0.02, 0.98, info_text,
                    transform=axes[k].transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle(f'{vidname} - {freq_name}\nPower Differences for {atlas_group}', y=1.05)
    plt.tight_layout()
    
    # Save the plot with sanitized filename
    plt.savefig(os.path.join(fig_dir, 
        f'power_diff_heatmap_{safe_atlas_name}_{freq_name}_{vidname}.png'),
        dpi=300, bbox_inches='tight')
    plt.close()

def create_threshold_violin_plots(threshold_results, atlas_group, freq_name, vidname, fig_dir):
    """Create violin plots showing the distribution of power values across threshold combinations"""
    # Sanitize the atlas group name for the filename
    safe_atlas_name = sanitize_filename(atlas_group)
    
    # Extract all threshold combinations
    threshold_cols = list(threshold_results.keys())
    
    # Prepare data for plotting
    plot_data = []
    for col in threshold_cols:
        if atlas_group in threshold_results[col]:
            results = threshold_results[col][atlas_group]
            if results['int_means']:
                parts = col.split('_')
                z_high = float(parts[2].replace('z', ''))
                z_low = float(parts[3])
                dev = float(parts[4].replace('dev', ''))
                
                # Add internal values
                for val in results['int_means']:
                    plot_data.append({
                        'Value': val,
                        'State': 'Internal',
                        'Z High': z_high,
                        'Z Low': z_low,
                        'Dev Percentile': dev,
                        'Threshold Label': f'z_high={z_high}\nz_low={z_low}'
                    })
                
                # Add external values
                for val in results['ext_means']:
                    plot_data.append({
                        'Value': val,
                        'State': 'External',
                        'Z High': z_high,
                        'Z Low': z_low,
                        'Dev Percentile': dev,
                        'Threshold Label': f'z_high={z_high}\nz_low={z_low}'
                    })
    
    if not plot_data:
        return
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create figure with subplots for each deviation percentile
    dev_percentiles = sorted(plot_df['Dev Percentile'].unique())
    fig, axes = plt.subplots(1, len(dev_percentiles), figsize=(12*len(dev_percentiles), 6))
    if len(dev_percentiles) == 1:
        axes = [axes]
    
    for k, dev in enumerate(dev_percentiles):
        dev_data = plot_df[plot_df['Dev Percentile'] == dev]
        
        # Create violin plot
        sns.violinplot(data=dev_data,
                      x='Threshold Label',
                      y='Value',
                      hue='State',
                      ax=axes[k],
                      split=True,
                      inner='quartile')
        
        # Add threshold information
        info_text = f'Internal: PC1 > z_high AND dev > {dev}\nExternal: PC1 < -z_low AND dev < {1-dev}'
        axes[k].text(0.02, 0.98, info_text,
                    transform=axes[k].transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        axes[k].set_title(f'Deviation Percentile: {dev}')
        axes[k].set_xlabel('Threshold Combinations')
        axes[k].set_ylabel('Power')
        axes[k].legend(title='State')
        
        # Rotate x-axis labels for better readability
        plt.setp(axes[k].get_xticklabels(), rotation=45, ha='right')
    
    plt.suptitle(f'{vidname} - {freq_name}\nPower Distributions for {atlas_group}', y=1.05)
    plt.tight_layout()
    
    # Save the plot with sanitized filename
    plt.savefig(os.path.join(fig_dir, 
        f'power_dist_violin_{safe_atlas_name}_{freq_name}_{vidname}.png'),
        dpi=300, bbox_inches='tight')
    plt.close()


#%%
# Loop through all frequency bands, movies, and atlases

# Loop through all frequency bands, movies, and atlases
for freq_band in freq_bands:
    if freq_band == 'HFA':
        freq_range = '70 - 150 Hz'
        freq_name = 'HFA (70-150 Hz)'
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
                           #'NS190_ses-01_run-01',
                           'NS190_ses-01_run-02',
                           'NS191_ses-01_run-01', 
                          # 'NS193_ses-01_run-01',
                           'NS193_ses-01_run-02',
                           'NS194_ses-01_run-01',
                           'NS205_ses-01_run-01']
        
            # Load the new combined features file
            if PC_type == 'shared_PC1':
                #et_file = '/Volumes/Samsung/Movie_data/combined_PC_features_output_21Apr25/combined_features_despicable_me_english.csv'
                et_file = '/Volumes/Samsung/Movie_data/combined_PC_features_output_12May25_robust_pca/combined_features_despicable_me_english.csv'
            elif PC_type == "sep_PCs":
               #et_file = f'/Volumes/Samsung/Movie_data/combined_PC_features_output_21Apr25/despicable_me_english_features_df_{z_threshold}.csv'
               et_file = f'/Volumes/Samsung/Movie_data/separate_PC_features_7Jan26/{movie}_features_df_[0.7, 0.7].csv'
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
                         #'NS144_ses-01_run-01',
                         'NS151_ses-01_run-01', 
                           'NS153_ses-01_run-01',
                           #'NS154_ses-01_run-01', 
                           'NS155_ses-01_run-01',
                           'NS155_02_ses-02_run-01', 
                           'NS164_ses-01_run-01',
                           'NS178_ses-01_run-01', 
                           'NS205_ses-01_run-01',
                           'NS210_ses-01_run-01'] #'NS211_ses-01_run-01']
            
            # Load the new combined features file
            if PC_type == 'shared_PC1':
                #et_file = '/Volumes/Samsung/Movie_data/combined_PC_features_output_21Apr25/combined_features_inscapes.csv'
                et_file = '/Volumes/Samsung/Movie_data/combined_PC_features_output_12May25_robust_pca/combined_features_inscapes.csv'
            elif PC_type == "sep_PCs":
               et_file = f'/Volumes/Samsung/Movie_data/separate_PC_features_7Jan26/{movie}_features_df_[0.7, 0.7].csv'
            # Read et file as df
            et_df = pd.read_csv(et_file)
            factor = 'PC1'
    
            # Read et file as df
        et_df = pd.read_csv(et_file)
        
        # Get all threshold columns
        threshold_cols = [col for col in et_df.columns if col.startswith('Attention_State_z')]
        
        for atlas in atlases:
            if atlas == 'DK_Atlas_Region':
                atlas_name = 'DK Atlas Regions'
            elif atlas == 'Y7_Atlas_Region':
                atlas_name = 'Yeo7 Networks'
            elif atlas == 'Y17_Atlas_Region':
                atlas_name = 'Yeo17 Atlas Region'
    
            # Directory paths
            if lfp_type == 'power':
                mne_data_dir = f'/Volumes/Samsung/Movie_data/{freq_band}_newbands_normFirst_{movie}_all_cortContacts_20May25'
            elif lfp_type == 'entropy':
                mne_data_dir = f'/Volumes/Samsung/Movie_data/entropy_extracted/{freq_band}_all_{movie}_all_cortContacts_entropy_29Mar25'

            # Initialize dictionary to store results for each threshold combination
            threshold_results = {col: {} for col in threshold_cols}
            
            # Process each patient
            for pat in patients:
                pat_dir = os.path.join(mne_data_dir, pat)
                    
                # Find the CSV files containing atlas info and lfp data
                if lfp_type == 'power':
                    #lfp_files = [filename for filename in os.listdir(pat_dir) if f'{method}' in filename and filename.endswith('.csv')]
                    lfp_files = [filename for filename in os.listdir(pat_dir) if filename.endswith('.csv')]

                elif lfp_type == 'entropy':
                    lfp_files = [filename for filename in os.listdir(pat_dir) if f'{entropy_type}' in filename and filename.endswith('.csv')]
                
                if not lfp_files:
                    print(f"No LFP files found for patient {pat} in {pat_dir}")
                    continue

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
                
                # Find and extract the rows corresponding to the patient
                et_patient_rows = et_df[et_df['Patient'] == pat].reset_index(drop=True)

                # Normalize electrode columns if required
                electrode_cols = [col for col in lfp_values.columns if col[0] in ['L', 'R'] and col[1:].isalnum()]
                if normalize:
                    data_rows[electrode_cols] = data_rows[electrode_cols].apply(pd.to_numeric, errors='coerce')
                    data_rows[electrode_cols] = (data_rows[electrode_cols] - data_rows[electrode_cols].mean()) / data_rows[electrode_cols].std()

                # Group electrodes by atlas
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
                    excluded_regions = ['bankssts', 'unknown']

                # Process each atlas group for each threshold combination
                for atlas_group, cols in grouped_cols.items():
                    if atlas_group in excluded_regions:
                        print(f"Skipping excluded region: {atlas_group}")
                        continue
                
                    print(f"\nProcessing atlas group: {atlas_group}")
                    print(f"Number of electrodes: {len(cols)}")
                    
                    for threshold_col in threshold_cols:
                        if atlas_group not in threshold_results[threshold_col]:
                            threshold_results[threshold_col][atlas_group] = {
                                'int_means': [], 'ext_means': [],
                                'int_sems': [], 'ext_sems': [],
                                'p_values': [], 'num_contacts': len(cols)
                            }
                        
                        int_mean, ext_mean, int_sem, ext_sem, p_val = analyze_power_differences(
                            et_patient_rows, data_rows, atlas_group, cols, threshold_col)
                        
                        if int_mean is not None:
                            threshold_results[threshold_col][atlas_group]['int_means'].append(int_mean)
                            threshold_results[threshold_col][atlas_group]['ext_means'].append(ext_mean)
                            threshold_results[threshold_col][atlas_group]['int_sems'].append(int_sem)
                            threshold_results[threshold_col][atlas_group]['ext_sems'].append(ext_sem)
                            threshold_results[threshold_col][atlas_group]['p_values'].append(p_val)
                        else:
                            print(f"No results for {atlas_group} in {threshold_col}")

            # Create visualizations for each threshold combination
            for threshold_col in threshold_cols:
                # Extract threshold values from column name
                # Format: Attention_State_z{high}_{low}_dev{dev}
                parts = threshold_col.split('_')
                # Remove 'z' prefix and convert to float
                z_high = float(parts[2].replace('z', ''))
                z_low = float(parts[3])
                dev_percentile = float(parts[4].replace('dev', ''))
                
                # Prepare data for plotting
                plot_data = []
                for atlas_group, results in threshold_results[threshold_col].items():
                    if results['int_means']:  # Only include if we have data
                        int_mean = np.mean(results['int_means'])
                        ext_mean = np.mean(results['ext_means'])
                        int_sem = np.mean(results['int_sems'])
                        ext_sem = np.mean(results['ext_sems'])
                        p_val = np.mean(results['p_values'])
                        
                        plot_data.append({
                            'Atlas Group': atlas_group,
                            'Internal Mean': int_mean,
                            'External Mean': ext_mean,
                            'Internal SEM': int_sem,
                            'External SEM': ext_sem,
                            'Significance': '*' if p_val < 0.001 else '',
                            'Contacts': results['num_contacts']
                        })
                
                plot_df = pd.DataFrame(plot_data)
                
                # Apply network ordering
                if atlas == 'Y7_Atlas_Region':
                    network_order = ['Default', 'Somatomotor', 'Dorsal Attention', 
                                   'Ventral Attention', 'Limbic', 'Frontoparietal', 'Visual']
                    plot_df = plot_df[plot_df['Atlas Group'].isin(network_order)]
                    plot_df = plot_df.set_index('Atlas Group').reindex(network_order).reset_index()
                
                # Create the plot
                plt.figure(figsize=(12, 6))
                x = np.arange(len(plot_df))
                width = 0.4
                
                plt.bar(x - width/2, plot_df['Internal Mean'], width, 
                       label='Internal', yerr=plot_df['Internal SEM'], capsize=5)
                plt.bar(x + width/2, plot_df['External Mean'], width, 
                       label='External', yerr=plot_df['External SEM'], capsize=5)
            
                # Add significance markers
                for i, sig in enumerate(plot_df['Significance']):
                    if sig == '*':
                        y_max = max(plot_df['Internal Mean'].iloc[i] + plot_df['Internal SEM'].iloc[i],
                                  plot_df['External Mean'].iloc[i] + plot_df['External SEM'].iloc[i])
                        plt.text(i, y_max + 0.05, '*', ha='center', va='bottom', color='red', size=16)
            
                # Customize plot
                plt.ylabel(f'{freq_name} Power')
                plt.title(f"{vidname} - {freq_name}\nThresholds: z_high={z_high}, z_low={z_low}, dev={dev_percentile}")
                plt.legend()
                plt.grid(True, axis='y', alpha=0.8)
                
                # Set y-axis limits
                plt.ylim(y_axis_limits[freq_band]['ymin'], y_axis_limits[freq_band]['ymax'])
                
                # Add contact counts to labels
                plot_df['Label'] = plot_df.apply(
                    lambda row: f"{row['Atlas Group']} (n={row['Contacts']})", axis=1)
                plt.xticks(x, plot_df['Label'], rotation=75, ha='right', fontsize=10)
            
                plt.tight_layout()
                
                # Save the plot
                #plt.savefig(os.path.join(fig_dir, 
                #    f'{atlas}_{movie}_{freq_range}_z{z_high}_{z_low}_dev{dev_percentile}_{method}_bargraph.png'),
                #           dpi=300, bbox_inches='tight')
                #plt.close()
                
                # Save the numerical results
                #plot_df.to_csv(os.path.join(fig_dir,
                #    f'{atlas}_{movie}_{freq_range}_z{z_high}_{z_low}_dev{dev_percentile}_{method}_bargraph_results.csv'),
                #                        index=False)
            
            # Add new visualizations for each atlas group
            for atlas_group in grouped_cols.keys():
                if atlas_group not in excluded_regions:
                    # Create heatmap showing power differences across thresholds
                    create_threshold_heatmap(threshold_results, atlas_group, freq_name, vidname, fig_dir)
                    
                    # Create violin plots showing power distributions
                    create_threshold_violin_plots(threshold_results, atlas_group, freq_name, vidname, fig_dir)