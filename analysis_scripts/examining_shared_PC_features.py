#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 16:17:37 2025

@author: christinechesebrough
"""

import seaborn as sns
from numpy.linalg import norm
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np
import os
import statistics
import matplotlib.patches as patches

# Define paths
drive = 'Samsung'
fig_dir = f'/Volumes/{drive}/Movie_data/combined_PC_features_12Mar25'
output_fig_dir = f'/Volumes/{drive}/Movie_data/combined_PC_features_output_12May25_robust_pca'
os.makedirs(fig_dir, exist_ok=True)
os.makedirs(output_fig_dir, exist_ok=True)

plot_density = False

# Load data
#combined_features_df = pd.read_csv("/Volumes/Samsung/Movie_data/more_normed_gaze_features_12Mar25/both_movies_unnormed_features_w_pcs.csv")
combined_features_df = pd.read_csv("/Volumes/Samsung/Movie_data/more_normed_gaze_features_12Mar25/both_movies_unnormed_features_w_pcs_Robust_PCA.csv")

# Define feature columns for group deviation calculation
feature_cols = [
    'Saccade_Rate', 'Vergence_Std', 'Abs_Vergence', 'Saccade_Dispersion',
    'Saccade_Dispersion_Std', 'Blink_Rate', 'Blink_Duration', 'Pupil_Avg',
    'Pupil_Std', 'ISC'
]

# Compute group means and deviation
group_means = combined_features_df.groupby(['Video', 'Time'])[feature_cols].mean().reset_index()
combined_features_df = combined_features_df.merge(group_means, on=['Video', 'Time'], suffixes=('', '_groupmean'))
combined_features_df['group_deviation'] = combined_features_df.apply(
    lambda row: norm([row[feat] - row[f"{feat}_groupmean"] for feat in feature_cols]),
    axis=1
)

# Recalculate group means and deviation
combined_features_df = combined_features_df.merge(group_means, on=['Video', 'Time'], suffixes=('', '_groupmean'))
combined_features_df['group_deviation'] = combined_features_df.apply(
    lambda row: norm([row[feat] - row[f"{feat}_groupmean"] for feat in feature_cols]),
    axis=1
)

# Modify individual PC1 z-scoring to compute separately for each movie
combined_features_df['PC1_z'] = combined_features_df.groupby(['Patient', 'Video'])['PC1'].transform(
    lambda x: (x - x.mean()) / x.std()
)

# Z-score group deviation
combined_features_df['group_dev_z'] = combined_features_df.groupby('Patient')['group_deviation'].transform(
    lambda x: (x - x.mean()) / x.std()
)

# Correlate group deviation with PC1
x_min, x_max = combined_features_df['PC1'].min(), combined_features_df['PC1'].max()

# Plot scatterplots for each video
for video in combined_features_df['Video'].unique():
    video_data = combined_features_df[combined_features_df['Video'] == video]
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=video_data, x='PC1', y='group_deviation', alpha=0.5)
    z = np.polyfit(video_data['PC1'], video_data['group_deviation'], 1)
    p = np.poly1d(z)
    plt.plot(video_data['PC1'], p(video_data['PC1']), "r--", alpha=0.8)
    corr = pearsonr(video_data['PC1'], video_data['group_deviation'])[0]
    plt.title(f'PC1 vs Group Deviation - {video}\nCorrelation: {corr:.3f}')
    plt.xlabel('PC1')
    plt.ylabel('Group Deviation')
    plt.xlim(x_min, x_max)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.close()

# Group-level correlation
group_corr = combined_features_df.groupby(['Video', 'Time']).apply(
    lambda x: pearsonr(x['PC1'], x['group_deviation'])[0]
).mean()
print(f"\nGroup-level correlation between PC1 and group deviation: {group_corr:.3f}")

# Individual-level correlations
individual_corrs = combined_features_df.groupby(['Video', 'Patient']).apply(
    lambda x: pearsonr(x['PC1'], x['group_deviation'])[0]
).reset_index(name='correlation')
print("\nIndividual-level correlations:")
print(individual_corrs.groupby('Video')['correlation'].describe())

# Plot individual correlations
plt.figure(figsize=(10, 6))
sns.boxplot(data=individual_corrs, x='Video', y='correlation')
plt.title('Individual Correlations between PC1 and Group Deviation')
plt.ylabel('Correlation Coefficient')
plt.tight_layout()
plt.close()

#%%

# good for power .75,.5,.7.,4
#good for entropy, 1,.5

# Define z thresholds
high_z_thresh = .75  # Threshold for high PC1 values
low_z_thresh = 0.5   # Threshold for low PC1 values

print("\nPC1 Z-score Thresholds:")
print(f"High threshold: {high_z_thresh}")
print(f"Low threshold: {low_z_thresh}")

# Calculate descriptive statistics of group deviation
deviation_stats = combined_features_df.groupby('Video')['group_deviation'].describe()
print("\nGroup Deviation Statistics by Video:")
print(deviation_stats)

# Define deviation thresholds based on percentiles or absolute values
# Compute thresholds separately for each movie
deviation_thresholds = {}
for video in combined_features_df['Video'].unique():
    video_data = combined_features_df[combined_features_df['Video'] == video]
    deviation_thresholds[video] = {
        'high': video_data['group_deviation'].quantile(0.7),
        'low': video_data['group_deviation'].quantile(0.4)
    }
    print(f"\nGroup Deviation Thresholds for {video}:")
    print(f"High threshold (75th percentile): {deviation_thresholds[video]['high']:.3f}")
    print(f"Low threshold (25th percentile): {deviation_thresholds[video]['low']:.3f}")

# Define function to convert time index to movie time
def index_to_time(idx, seconds_per_point):
    """
    Convert time index to movie time in MM:SS format
    
    Parameters:
    - idx: Time index
    - seconds_per_point: Number of seconds per timepoint
    
    Returns:
    - String in MM:SS format
    """
    total_seconds = idx * seconds_per_point
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

# Define attention state labeling functions
def label_attention_state_raw(pc_z, raw_dev, video, pc_thresh_high=high_z_thresh, pc_thresh_low=low_z_thresh, dev_thresh_high=None, dev_thresh_low=None):
    if dev_thresh_high is None:
        dev_thresh_high = deviation_thresholds[video]['high']
    if dev_thresh_low is None:
        dev_thresh_low = deviation_thresholds[video]['low']
        
    if pc_z > pc_thresh_high and raw_dev > dev_thresh_high:
        return 'Internal_HighDeviation'
    elif pc_z < -pc_thresh_low and raw_dev < dev_thresh_low:
        return 'External_LowDeviation'
    elif pc_z > pc_thresh_high and raw_dev < dev_thresh_low:
        return 'Internal_LowDeviation'
    elif pc_z < -pc_thresh_low and raw_dev > dev_thresh_high:
        return 'External_HighDeviation'
    else:
        return 'Neutral'

def label_attention_state_group(pc_z_group, raw_dev, video, pc_thresh_high=high_z_thresh, pc_thresh_low=low_z_thresh, dev_thresh_high=None, dev_thresh_low=None):
    if dev_thresh_high is None:
        dev_thresh_high = deviation_thresholds[video]['high']
    if dev_thresh_low is None:
        dev_thresh_low = deviation_thresholds[video]['low']
        
    if pc_z_group > pc_thresh_high and raw_dev > dev_thresh_high:
        return 'Internal_HighDeviation'
    elif pc_z_group < -pc_thresh_low and raw_dev < dev_thresh_low:
        return 'External_LowDeviation'
    elif pc_z_group > pc_thresh_high and raw_dev < dev_thresh_low:
        return 'Internal_LowDeviation'
    elif pc_z_group < -pc_thresh_low and raw_dev > dev_thresh_high:
        return 'External_HighDeviation'
    else:
        return 'Neutral'

# Create attention labels first
combined_features_df['Attention_Label_Individual'] = combined_features_df.apply(
    lambda row: label_attention_state_raw(
        row['PC1_z'], 
        row['group_deviation'],
        row['Video'],
        high_z_thresh,
        low_z_thresh,
        deviation_thresholds[row['Video']]['high'],
        deviation_thresholds[row['Video']]['low']
    ), 
    axis=1
)
if plot_density:
    # Now proceed with the density plots
    for video in combined_features_df['Video'].unique():
        video_data = combined_features_df[combined_features_df['Video'] == video]
        timepoints = sorted(video_data['Time'].unique())
        total_timepoints = len(timepoints)
        seconds_per_point = 600 / total_timepoints
        total_patients = video_data['Patient'].nunique()
        
        attention_types = ['Internal_HighDeviation', 'External_LowDeviation', 
                          'Internal_LowDeviation', 'External_HighDeviation']
        
        density_df = pd.DataFrame({
            'Time_Index': timepoints,
            'Movie_Time': [index_to_time(idx, seconds_per_point) for idx in timepoints]
        })
        
        for category in attention_types:
            category_counts = video_data[video_data['Attention_Label_Individual'] == category].groupby('Time').size()
            for idx in timepoints:
                density_df.loc[density_df['Time_Index'] == idx, category] = category_counts.get(idx, 0) / total_patients
        
        plt.figure(figsize=(8, 4))
        plt.stackplot(density_df['Time_Index'], 
                     [density_df['Internal_HighDeviation'], 
                      density_df['External_LowDeviation']],
                     labels=['Internal High Deviation', 
                            'External Low Deviation'],
                     colors=['red', 'blue'],
                     alpha=0.4)
                     
        
        plt.title(f'Density of Attention States Across Patients Over Time\n{video}', fontsize=14)
        plt.xlabel('Time Index', fontsize=12)
        plt.ylabel('Proportion of Patients', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
        plt.ylim(0, 1)
        
        minute_markers = [int(minute * 60 / seconds_per_point) for minute in range(11) if int(minute * 60 / seconds_per_point) < total_timepoints]
        minute_labels = [f"{minute:02d}:00" for minute in range(11) if int(minute * 60 / seconds_per_point) < total_timepoints]
        
        plt.xticks(minute_markers, minute_labels)
        plt.tight_layout()
        plt.savefig(os.path.join(output_fig_dir, f'attention_state_density_{video}.png'), dpi=300, bbox_inches='tight')
        
        density_df.to_csv(os.path.join(output_fig_dir, f'attention_state_density_{video}.csv'), index=False)
        
        # Create a heatmap visualization
        heatmap_data = density_df.drop(['Movie_Time'], axis=1).set_index('Time_Index')
        plt.figure(figsize=(15, 6))
        sns.heatmap(heatmap_data.T, cmap='viridis', vmin=0, vmax=1, xticklabels=50, cbar_kws={'label': 'Proportion of Patients'})
        plt.title(f'Heatmap of Attention States Across Patients Over Time\n{video}', fontsize=14)
        plt.xlabel('Time Index', fontsize=12)
        plt.ylabel('Attention State', fontsize=12)
        ax = plt.gca()
        ax.set_xticks(minute_markers)
        ax.set_xticklabels(minute_labels)
        plt.tight_layout()
        plt.savefig(os.path.join(output_fig_dir, f'attention_state_heatmap_{video}.png'), dpi=300, bbox_inches='tight')
    
        print(f"\nAttention state density data for {video} saved to:")
        print(f"  {os.path.join(output_fig_dir, f'attention_state_density_{video}.csv')}")
        print(f"  {os.path.join(output_fig_dir, f'attention_state_density_{video}.png')}")
        print(f"  {os.path.join(output_fig_dir, f'attention_state_heatmap_{video}.png')}")

# Count and normalize attention labels using raw deviations
attention_counts_raw = combined_features_df.groupby(['Video', 'Attention_Label_Individual']).size().reset_index(name='Count')
subject_counts = combined_features_df.groupby('Video')['Patient'].nunique().reset_index(name='Num_Subjects')
attention_counts_raw = attention_counts_raw.merge(subject_counts, on='Video')
attention_counts_raw['Timepoints_per_Subject'] = attention_counts_raw['Count'] / attention_counts_raw['Num_Subjects']

# Plot barplot of attention labels using raw deviations
plt.figure(figsize=(12, 6))
sns.barplot(data=attention_counts_raw, x='Attention_Label_Individual', y='Timepoints_per_Subject', hue='Video')
plt.title('Avg Timepoints per Subject by Attention State (Raw Deviation)')
plt.ylabel('Avg Timepoints per Subject')
plt.xlabel('Attention Label')
plt.xticks(rotation=45)
plt.tight_layout()
plt.close()

# Add group-level PC1 z-scoring
combined_features_df['PC1_z_group'] = combined_features_df.groupby(['Video', 'Time'])['PC1'].transform(
    lambda x: (x - x.mean()) / x.std()
)

# Apply group-level labeling
combined_features_df['Attention_Label_Group'] = combined_features_df.apply(
    lambda row: label_attention_state_group(
        row['PC1_z_group'], 
        row['group_deviation'],
        row['Video'],
        high_z_thresh,
        low_z_thresh,
        deviation_thresholds[row['Video']]['high'],
        deviation_thresholds[row['Video']]['low']
    ), 
    axis=1
)

# Count and normalize attention labels for group-level z-scoring
attention_counts = combined_features_df.groupby(['Video', 'Attention_Label_Group']).size().reset_index(name='Count')
attention_counts = attention_counts.merge(subject_counts, on='Video')
attention_counts['Timepoints_per_Subject'] = attention_counts['Count'] / attention_counts['Num_Subjects']

# Compare with original z-scored version
print("\nComparison of Attention State Distributions:")
print("\nUsing Raw Deviations:")
print(attention_counts_raw.groupby('Video')['Timepoints_per_Subject'].describe())
print("\nUsing Z-scored Deviations:")
print(attention_counts.groupby('Video')['Timepoints_per_Subject'].describe())

# Compare attention state distributions
attention_counts_individual = combined_features_df.groupby(['Video', 'Attention_Label_Individual']).size().reset_index(name='Count')
attention_counts_group = combined_features_df.groupby(['Video', 'Attention_Label_Group']).size().reset_index(name='Count')

subject_counts = combined_features_df.groupby('Video')['Patient'].nunique().reset_index(name='Num_Subjects')

attention_counts_individual = attention_counts_individual.merge(subject_counts, on='Video')
attention_counts_group = attention_counts_group.merge(subject_counts, on='Video')

attention_counts_individual['Timepoints_per_Subject'] = attention_counts_individual['Count'] / attention_counts_individual['Num_Subjects']
attention_counts_group['Timepoints_per_Subject'] = attention_counts_group['Count'] / attention_counts_group['Num_Subjects']

# Plot comparison of attention state distributions
plt.figure(figsize=(6, 4))

plt.subplot(1, 2, 1)
sns.barplot(data=attention_counts_individual, x='Attention_Label_Individual', y='Timepoints_per_Subject', hue='Video')
plt.title('Attention States (Individual Z-scoring)')
plt.ylabel('Avg Timepoints per Subject')
plt.xlabel('Attention Label')
plt.xticks(rotation=45)
plt.tight_layout()

plt.subplot(1, 2, 2)
sns.barplot(data=attention_counts_group, x='Attention_Label_Group', y='Timepoints_per_Subject', hue='Video')
plt.title('Attention States (Group Z-scoring)')
plt.ylabel('Avg Timepoints per Subject')
plt.xlabel('Attention Label')
plt.xticks(rotation=45)
plt.tight_layout()
plt.close()

# Print comparison statistics
print("\nComparison of Attention State Distributions:")
print("\nUsing Individual Z-scoring:")
print(attention_counts_individual.groupby('Video')['Timepoints_per_Subject'].describe())
print("\nUsing Group Z-scoring:")
print(attention_counts_group.groupby('Video')['Timepoints_per_Subject'].describe())

# Analyze agreement between the two approaches
combined_features_df['Label_Agreement'] = combined_features_df['Attention_Label_Individual'] == combined_features_df['Attention_Label_Group']
agreement_stats = combined_features_df.groupby('Video')['Label_Agreement'].mean()
print("\nProportion of timepoints where both approaches agree:")
print(agreement_stats)

# Save combined features DataFrame
for video in combined_features_df['Video'].unique():
    video_df = combined_features_df[combined_features_df['Video'] == video]
    video_df.to_csv(os.path.join(output_fig_dir, f'combined_features_{video}.csv'), index=False)
    print(f"\nSaved combined features DataFrame for {video} to {os.path.join(output_fig_dir, f'combined_features_{video}.csv')}")

combined_features_df.to_csv(os.path.join(output_fig_dir, 'combined_features_df.csv'), index=False)
print(f"\nSaved full combined features DataFrame to {os.path.join(output_fig_dir, 'combined_features_df.csv')}")


# Plot individual and mean values over time for both PC1 and group deviation
for video in combined_features_df['Video'].unique():
    video_data = combined_features_df[combined_features_df['Video'] == video]
    total_timepoints = len(video_data['Time'].unique())
    seconds_per_point = 600 / total_timepoints
    movie_times = [idx * seconds_per_point for idx in video_data['Time'].unique()]
    
    # Plot 1: Time series of PC1 and group deviation
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    # Plot PC1
    for patient in video_data['Patient'].unique():
        patient_data = video_data[video_data['Patient'] == patient]
        ax1.plot(patient_data['Time'] * seconds_per_point, patient_data['PC1_z'], alpha=0.3, linewidth=1)
    mean_pc1 = video_data.groupby('Time')['PC1_z'].mean()
    ax1.plot(mean_pc1.index * seconds_per_point, mean_pc1.values, color='black', linewidth=3, label='Group Mean')
    ax1.axhline(high_z_thresh, color='gray', linestyle=':', alpha=0.8)
    ax1.axhline(-low_z_thresh, color='gray', linestyle=':', alpha=0.8)
    ax1.text(0, high_z_thresh + 0.1, f'+{high_z_thresh}', color='gray')
    ax1.text(0, -low_z_thresh - 0.2, f'-{low_z_thresh}', color='gray')
    ax1.set_title(f'Individual and Mean PC1 Values\n{video}', fontsize=14)
    ax1.set_ylabel('PC1 (z-scored)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(['Individual Patients', 'Group Mean'])
    
    # Plot group deviation
    for patient in video_data['Patient'].unique():
        patient_data = video_data[video_data['Patient'] == patient]
        ax2.plot(patient_data['Time'] * seconds_per_point, patient_data['group_dev_z'], alpha=0.3, linewidth=1)
    mean_dev = video_data.groupby('Time')['group_dev_z'].mean()
    ax2.plot(mean_dev.index * seconds_per_point, mean_dev.values, color='black', linewidth=3, label='Group Mean')
    ax2.axhline(high_z_thresh, color='gray', linestyle=':', alpha=0.8)
    ax2.axhline(-low_z_thresh, color='gray', linestyle=':', alpha=0.8)
    ax2.text(0, high_z_thresh + 0.1, f'+{high_z_thresh}', color='gray')
    ax2.text(0, -low_z_thresh - 0.2, f'-{low_z_thresh}', color='gray')
    ax2.set_title('Individual and Mean Group Deviation Values\n{video}', fontsize=14)
    ax2.set_ylabel('Group Deviation (z-scored)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(['Individual Patients', 'Group Mean'])
    
    # Add x-axis time markers
    minute_markers = np.arange(0, 601, 60)
    minute_labels = [f"{int(x/60):02d}:{int(x%60):02d}" for x in minute_markers]
    ax2.set_xticks(minute_markers)
    ax2.set_xticklabels(minute_labels, rotation=45)
    ax2.set_xlabel('Movie Time (MM:SS)', fontsize=12)
    ax1.set_xlim(0, 600)
    ax2.set_xlim(0, 600)
    plt.tight_layout()
    plt.savefig(os.path.join(output_fig_dir, f'PC1_and_GroupDev_individual_and_mean_{video}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: PC1 distributions for each subject
    plt.figure(figsize=(6, 4))
    for patient in video_data['Patient'].unique():
        patient_data = video_data[video_data['Patient'] == patient]
        sns.kdeplot(patient_data['PC1'], alpha=0.3, label=f'Patient {patient}')
    
    # Add mean distribution
    sns.kdeplot(video_data['PC1'], color='black', linewidth=2, label='Group Mean')
    
    # Add threshold lines
    plt.axvline(high_z_thresh, color='red', linestyle='--', label=f'High Threshold ({high_z_thresh})')
    plt.axvline(-low_z_thresh, color='blue', linestyle='--', label=f'Low Threshold ({low_z_thresh})')
    
    plt.title(f'PC1 Distributions by Subject - {video}', fontsize=14)
    plt.xlabel('PC1 Value', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_fig_dir, f'PC1_distributions_{video}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 3: Group deviation distributions for each subject
    plt.figure(figsize=(6,4))
    for patient in video_data['Patient'].unique():
        patient_data = video_data[video_data['Patient'] == patient]
        sns.kdeplot(patient_data['group_deviation'], alpha=0.3, label=f'Patient {patient}')
    
    # Add mean distribution
    sns.kdeplot(video_data['group_deviation'], color='black', linewidth=2, label='Group Mean')
    
    # Add threshold lines for this movie
    high_thresh = deviation_thresholds[video]['high']
    low_thresh = deviation_thresholds[video]['low']
    plt.axvline(high_thresh, color='red', linestyle='--', label=f'High Threshold ({high_thresh:.3f})')
    plt.axvline(low_thresh, color='blue', linestyle='--', label=f'Low Threshold ({low_thresh:.3f})')
    
    plt.title(f'Group Deviation Distributions by Subject - {video}', fontsize=14)
    plt.xlabel('Group Deviation Value', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_fig_dir, f'group_deviation_distributions_{video}.png'), dpi=300, bbox_inches='tight')
    #plt.close()

    # Plot 4: Scatter plot with attention state quadrants
    plt.figure(figsize=(10, 7))
    ax = sns.scatterplot(
        data=video_data, x='PC1_z', y='group_deviation',
        hue='Attention_Label_Individual',
        alpha=0.4, edgecolor=None
    )
    
    # Add regression line
    sns.regplot(
        data=video_data, x='PC1_z', y='group_deviation',
        scatter=False, color='black'
    )

    # Add threshold lines
    plt.axvline(high_z_thresh, color='red', linestyle=':', alpha=0.5, label=f'PC1_z + {high_z_thresh}')
    plt.axvline(-low_z_thresh, color='red', linestyle=':', alpha=0.5, label=f'PC1_z - {low_z_thresh}')
    plt.axhline(high_thresh, color='blue', linestyle=':', alpha=0.5, label=f'Deviation + {high_thresh:.3f}')
    plt.axhline(low_thresh, color='blue', linestyle=':', alpha=0.5, label=f'Deviation - {low_thresh:.3f}')

    # Add quadrant labels
    ax.add_patch(patches.Rectangle((high_z_thresh, high_thresh), 5, 5, linewidth=0, facecolor='green', alpha=0.1))
    ax.text(high_z_thresh + 0.5, high_thresh + 0.5, 'Internal\nHigh Deviation', fontsize=12, color='black', fontweight='bold')
    
    ax.add_patch(patches.Rectangle((-low_z_thresh - 5, low_thresh - 5), 5, 5, linewidth=0, facecolor='purple', alpha=0.1))
    ax.text(-low_z_thresh - 2, low_thresh - 1, 'External\nLow Deviation', fontsize=12, color='black', fontweight='bold')
    
    ax.add_patch(patches.Rectangle((high_z_thresh, low_thresh - 5), 5, 5, linewidth=0, facecolor='yellow', alpha=0.1))
    ax.text(high_z_thresh + 0.5, low_thresh - 1, 'Internal\nLow Deviation', fontsize=12, color='black', fontweight='bold')
    
    ax.add_patch(patches.Rectangle((-low_z_thresh - 5, high_thresh), 5, 5, linewidth=0, facecolor='orange', alpha=0.1))
    ax.text(-low_z_thresh - 2, high_thresh + 0.5, 'External\nHigh Deviation', fontsize=12, color='black', fontweight='bold')

    plt.title(f'Scatterplot of PC1_z vs Group Deviation\nwith Attention State Quadrants - {video}')
    plt.xlabel('PC1 (z-scored per subject)')
    plt.ylabel('Group Deviation')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_fig_dir, f'attention_state_quadrants_{video}.png'), dpi=300, bbox_inches='tight')
    #plt.close()

    # Analyze distribution characteristics
    print(f"\nPC1 Distribution Analysis for {video}:")
    for patient in video_data['Patient'].unique():
        patient_data = video_data[video_data['Patient'] == patient]
        mean_pc1 = patient_data['PC1'].mean()
        std_pc1 = patient_data['PC1'].std()
        skewness = patient_data['PC1'].skew()
        kurtosis = patient_data['PC1'].kurtosis()
        
        # Calculate proportion of timepoints in each attention state
        states = patient_data['Attention_Label_Individual'].value_counts(normalize=True)
        
        print(f"\nPatient {patient}:")
        print(f"Mean PC1: {mean_pc1:.3f}")
        print(f"Std PC1: {std_pc1:.3f}")
        print(f"Skewness: {skewness:.3f}")
        print(f"Kurtosis: {kurtosis:.3f}")
        print("\nAttention State Proportions:")
        for state, prop in states.items():
            print(f"{state}: {prop:.3f}")

    # Analyze group deviation distribution characteristics
    print(f"\nGroup Deviation Distribution Analysis for {video}:")
    for patient in video_data['Patient'].unique():
        patient_data = video_data[video_data['Patient'] == patient]
        mean_dev = patient_data['group_deviation'].mean()
        std_dev = patient_data['group_deviation'].std()
        skewness = patient_data['group_deviation'].skew()
        kurtosis = patient_data['group_deviation'].kurtosis()
        
        # Calculate proportion of timepoints above/below thresholds
        prop_high_dev = (patient_data['group_deviation'] > high_thresh).mean()
        prop_low_dev = (patient_data['group_deviation'] < low_thresh).mean()
        
        print(f"\nPatient {patient}:")
        print(f"Mean Deviation: {mean_dev:.3f}")
        print(f"Std Deviation: {std_dev:.3f}")
        print(f"Skewness: {skewness:.3f}")
        print(f"Kurtosis: {kurtosis:.3f}")
        print(f"Proportion above high threshold: {prop_high_dev:.3f}")
        print(f"Proportion below low threshold: {prop_low_dev:.3f}")

# Find timepoints where group means cross thresholds and convert to movie time
for video in combined_features_df['Video'].unique():
    video_data = combined_features_df[combined_features_df['Video'] == video]
    total_timepoints = len(video_data['Time'].unique())
    seconds_per_point = 600 / total_timepoints
    mean_pc1 = video_data.groupby('Time')['PC1_z'].mean()
    mean_dev = video_data.groupby('Time')['group_dev_z'].mean()
    pc1_above = mean_pc1[mean_pc1 > high_z_thresh].index.tolist()
    pc1_below = mean_pc1[mean_pc1 < -low_z_thresh].index.tolist()
    dev_above = mean_dev[mean_dev > high_z_thresh].index.tolist()
    dev_below = mean_dev[mean_dev < -low_z_thresh].index.tolist()
    
    data = {'Measure': [], 'Threshold_Type': [], 'Index': [], 'Movie_Time': [], 'Value': []}
    
    for idx in pc1_above:
        data['Measure'].append('PC1')
        data['Threshold_Type'].append(f'Above {high_z_thresh}')
        data['Index'].append(idx)
        data['Movie_Time'].append(index_to_time(idx, seconds_per_point))
        data['Value'].append(mean_pc1[idx])
        
    for idx in pc1_below:
        data['Measure'].append('PC1')
        data['Threshold_Type'].append(f'Below -{low_z_thresh}')
        data['Index'].append(idx)
        data['Movie_Time'].append(index_to_time(idx, seconds_per_point))
        data['Value'].append(mean_pc1[idx])
    
    for idx in dev_above:
        data['Measure'].append('Group_Deviation')
        data['Threshold_Type'].append(f'Above {high_z_thresh}')
        data['Index'].append(idx)
        data['Movie_Time'].append(index_to_time(idx, seconds_per_point))
        data['Value'].append(mean_dev[idx])
        
    for idx in dev_below:
        data['Measure'].append('Group_Deviation')
        data['Threshold_Type'].append(f'Below -{low_z_thresh}')
        data['Index'].append(idx)
        data['Movie_Time'].append(index_to_time(idx, seconds_per_point))
        data['Value'].append(mean_dev[idx])
    
    df = pd.DataFrame(data).sort_values(['Measure', 'Index'])
    csv_filename = os.path.join(output_fig_dir, f'threshold_crossings_{video}.csv')
    df.to_csv(csv_filename, index=False)
    
    print(f"\n{video}:")
    print(f"Total timepoints: {total_timepoints}")
    print(f"Seconds per timepoint: {seconds_per_point:.2f}")
    print(f"Results saved to: {csv_filename}")
    print("\nThreshold Crossings:")
    print(df)
    print(f"\nPercentages:")
    print(f"PC1 above threshold: {len(pc1_above)/total_timepoints*100:.1f}%")
    print(f"PC1 below threshold: {len(pc1_below)/total_timepoints*100:.1f}%")
    print(f"Group deviation above threshold: {len(dev_above)/total_timepoints*100:.1f}%")
    print(f"Group deviation below threshold: {len(dev_below)/total_timepoints*100:.1f}%")

        

# Save attention state comparison plots
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.barplot(data=attention_counts_individual, x='Attention_Label_Individual', y='Timepoints_per_Subject', hue='Video')
plt.title('Attention States (Individual Z-scoring)')
plt.ylabel('Avg Timepoints per Subject')
plt.xlabel('Attention Label')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_fig_dir, 'attention_states_individual.png'), dpi=300, bbox_inches='tight')

plt.subplot(1, 2, 2)
sns.barplot(data=attention_counts_group, x='Attention_Label_Group', y='Timepoints_per_Subject', hue='Video')
plt.title('Attention States (Group Z-scoring)')
plt.ylabel('Avg Timepoints per Subject')
plt.xlabel('Attention Label')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_fig_dir, 'attention_states_group.png'), dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# Save all DataFrames
combined_features_df.to_csv(os.path.join(output_fig_dir, 'combined_features_df.csv'), index=False)
attention_counts_individual.to_csv(os.path.join(output_fig_dir, 'attention_counts_individual.csv'), index=False)
attention_counts_group.to_csv(os.path.join(output_fig_dir, 'attention_counts_group.csv'), index=False)

print(f"\nAll output files have been saved to: {output_fig_dir}")


# Calculate states per patient for individual z-scoring
states_per_patient_individual = combined_features_df.groupby(['Video', 'Patient', 'Attention_Label_Individual']).size().reset_index(name='Count')
total_timepoints = combined_features_df.groupby(['Video', 'Patient']).size().reset_index(name='Total_Timepoints')
states_per_patient_individual = states_per_patient_individual.merge(total_timepoints, on=['Video', 'Patient'])
states_per_patient_individual['Proportion'] = states_per_patient_individual['Count'] / states_per_patient_individual['Total_Timepoints']
states_per_patient_individual['Approach'] = 'Individual Z-scoring'

# Calculate states per patient for group z-scoring
states_per_patient_group = combined_features_df.groupby(['Video', 'Patient', 'Attention_Label_Group']).size().reset_index(name='Count')
states_per_patient_group = states_per_patient_group.merge(total_timepoints, on=['Video', 'Patient'])
states_per_patient_group['Proportion'] = states_per_patient_group['Count'] / states_per_patient_group['Total_Timepoints']
states_per_patient_group['Approach'] = 'Group Z-scoring'

# Combine both approaches
states_per_patient = pd.concat([
    states_per_patient_individual.rename(columns={'Attention_Label_Individual': 'Attention_Label'}),
    states_per_patient_group.rename(columns={'Attention_Label_Group': 'Attention_Label'})
])

# Filter for only External Low Deviation and Internal High Deviation states
states_per_patient = states_per_patient[states_per_patient['Attention_Label'].isin(['External_LowDeviation', 'Internal_HighDeviation'])]

# Create separate plots for each movie
for video in states_per_patient['Video'].unique():
    # Filter data for current video
    video_data = states_per_patient[states_per_patient['Video'] == video]
    
    # Create figure for this video
    plt.figure(figsize=(10, 4))
    
    # Plot 1: Individual Z-scoring
    plt.subplot(1, 2, 1)
    sns.barplot(data=video_data[video_data['Approach'] == 'Individual Z-scoring'], 
                x='Patient', y='Count', hue='Attention_Label')
    plt.title(f'Individual Z-scoring - {video}')
    plt.ylabel('Number of Timepoints')
    plt.xlabel('Patient')
    plt.xticks(rotation=90)
    plt.legend(title='Attention State')
    
    # Plot 2: Group Z-scoring
    plt.subplot(1, 2, 2)
    sns.barplot(data=video_data[video_data['Approach'] == 'Group Z-scoring'], 
                x='Patient', y='Count', hue='Attention_Label')
    plt.title(f'Group Z-scoring - {video}')
    plt.ylabel('Number of Timepoints')
    plt.xlabel('Patient')
    plt.xticks(rotation=90)
    plt.legend(title='Attention State')
    
    plt.suptitle(f'Number of External Low Deviation and Internal High Deviation States per Patient - {video}', y=1.02)
    plt.tight_layout()
    plt.show()

    # Save the plot
    plt.savefig(os.path.join(output_fig_dir, f'attention_states_per_patient_comparison_{video}.png'), dpi=300, bbox_inches='tight')

# Print summary statistics
print("\nSummary of External Low Deviation and Internal High Deviation States per Patient:")
for video in states_per_patient['Video'].unique():
    print(f"\n{video}:")
    print("\nIndividual Z-scoring:")
    print(states_per_patient[(states_per_patient['Approach'] == 'Individual Z-scoring') & 
                            (states_per_patient['Video'] == video)].groupby(['Patient', 'Attention_Label'])['Count'].describe())
    print("\nGroup Z-scoring:")
    print(states_per_patient[(states_per_patient['Approach'] == 'Group Z-scoring') & 
                            (states_per_patient['Video'] == video)].groupby(['Patient', 'Attention_Label'])['Count'].describe())

# Save the filtered patient-level data
states_per_patient.to_csv(os.path.join(output_fig_dir, 'attention_states_per_patient_comparison.csv'), index=False)

def analyze_threshold_sensitivity(combined_features_df, z_thresholds_high, z_thresholds_low, dev_percentiles, output_dir):
    """
    Analyze how different threshold values affect attention state distributions.
    
    Parameters:
    - combined_features_df: DataFrame containing the features
    - z_thresholds_high: List of high z-score thresholds to test
    - z_thresholds_low: List of low z-score thresholds to test
    - dev_percentiles: List of percentiles to use for deviation thresholds
    - output_dir: Directory to save results
    """
    results = []
    
    for z_thresh_high in z_thresholds_high:
        for z_thresh_low in z_thresholds_low:
            for dev_percentile in dev_percentiles:
                # Calculate deviation thresholds for each video
                deviation_thresholds = {}
                for video in combined_features_df['Video'].unique():
                    video_data = combined_features_df[combined_features_df['Video'] == video]
                    deviation_thresholds[video] = {
                        'high': video_data['group_deviation'].quantile(dev_percentile),
                        'low': video_data['group_deviation'].quantile(1 - dev_percentile)
                    }
                
                # Apply attention state labeling
                combined_features_df['Attention_Label'] = combined_features_df.apply(
                    lambda row: label_attention_state_raw(
                        row['PC1_z'], 
                        row['group_deviation'],
                        row['Video'],
                        z_thresh_high,
                        z_thresh_low,
                        deviation_thresholds[row['Video']]['high'],
                        deviation_thresholds[row['Video']]['low']
                    ), 
                    axis=1
                )
                
                # Calculate state distributions
                for video in combined_features_df['Video'].unique():
                    video_data = combined_features_df[combined_features_df['Video'] == video]
                    state_counts = video_data['Attention_Label'].value_counts(normalize=True)
                    
                    for state, proportion in state_counts.items():
                        results.append({
                            'Video': video,
                            'Z_Threshold_High': z_thresh_high,
                            'Z_Threshold_Low': z_thresh_low,
                            'Dev_Percentile': dev_percentile,
                            'State': state,
                            'Proportion': proportion,
                            'High_Dev_Threshold': deviation_thresholds[video]['high'],
                            'Low_Dev_Threshold': deviation_thresholds[video]['low']
                        })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Plot for each video
    for i, video in enumerate(combined_features_df['Video'].unique(), 1):
        plt.subplot(2, 1, i)
        video_results = results_df[results_df['Video'] == video]
        
        # Create pivot table for heatmap
        pivot_data = video_results.pivot_table(
            values='Proportion',
            index=['Z_Threshold_High', 'Z_Threshold_Low'],
            columns='Dev_Percentile',
            aggfunc='mean'
        )
        
        sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='YlOrRd')
        plt.title(f'Average Proportion of States - {video}')
        plt.xlabel('Deviation Percentile')
        plt.ylabel('Z-Score Thresholds (High, Low)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'threshold_sensitivity_heatmap.png'), dpi=300, bbox_inches='tight')
    
    # Create stacked bar plots for each video and deviation percentile
    for video in combined_features_df['Video'].unique():
        # Increase figure width and adjust height for better proportions
        plt.figure(figsize=(20, 6))
        video_results = results_df[results_df['Video'] == video]
        
        # Create subplots for each deviation percentile with reduced spacing
        for i, dev_percentile in enumerate(dev_percentiles, 1):
            plt.subplot(1, len(dev_percentiles), i)
            dev_data = video_results[video_results['Dev_Percentile'] == dev_percentile]
            
            # Create pivot table for stacked bars
            pivot_data = dev_data.pivot_table(
                values='Proportion',
                index=['Z_Threshold_High', 'Z_Threshold_Low'],
                columns='State',
                aggfunc='mean'
            )
            
            # Plot stacked bars with increased width
            ax = pivot_data.plot(kind='bar', stacked=True, ax=plt.gca(), width=0.8)
            plt.title(f'Dev Percentile: {dev_percentile}', pad=20)
            plt.xlabel('Z-Score Thresholds (High, Low)', labelpad=10)
            plt.ylabel('Proportion', labelpad=10)
            
            # Adjust legend position and size
            plt.legend(title='Attention State', 
                      bbox_to_anchor=(1.02, 1), 
                      loc='upper left',
                      fontsize='small',
                      title_fontsize='small')
            
            # Add percentage labels with adjusted font size
            for c in ax.containers:
                ax.bar_label(c, fmt='%.2f', label_type='center', fontsize=8)
            
            # Adjust grid appearance
            plt.grid(True, alpha=0.2)
            
            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Adjust y-axis limits to ensure labels are visible
            plt.ylim(0, 1.1)
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45, ha='right')
        
        # Adjust subplot spacing
        plt.subplots_adjust(wspace=0.1)  # Reduce space between subplots
        
        # Add main title with adjusted position
        plt.suptitle(f'Attention State Proportions by Z-Thresholds and Deviation Percentile - {video}', 
                    y=1.05, 
                    fontsize=12)
        
        # Save with tight layout and high DPI
        plt.savefig(os.path.join(output_dir, f'threshold_sensitivity_stacked_bars_{video}.png'), 
                   dpi=300, 
                   bbox_inches='tight',
                   pad_inches=0.5)
        plt.close()
    
    # Save results to CSV
    results_df.to_csv(os.path.join(output_dir, 'threshold_sensitivity_results.csv'), index=False)
    
    return results_df

# Define threshold ranges to test
z_thresholds_high = [0.75, 1.0, 1.25]
z_thresholds_low = [.25,0.5, 0.75]
dev_percentiles = [0.6, 0.8]

# Run threshold sensitivity analysis
threshold_results = analyze_threshold_sensitivity(
    combined_features_df,
    z_thresholds_high,
    z_thresholds_low,
    dev_percentiles,
    output_fig_dir
)

print("\nThreshold sensitivity analysis complete. Results saved to:")
print(f"- {os.path.join(output_fig_dir, 'threshold_sensitivity_results.csv')}")
print(f"- {os.path.join(output_fig_dir, 'threshold_sensitivity_heatmap.png')}")
print(f"- {os.path.join(output_fig_dir, 'threshold_sensitivity_stacked_bars.png')}")

# After the threshold sensitivity analysis, add columns for each threshold combination
print("\nAdding attention state columns for each threshold combination...")

# Process each video separately
for video in combined_features_df['Video'].unique():
    print(f"\nProcessing {video}...")
    
    # Create a copy of the video data
    video_data = combined_features_df[combined_features_df['Video'] == video].copy()
    
    # Add columns for each threshold combination
    for z_high in z_thresholds_high:
        for z_low in z_thresholds_low:
            for dev_percentile in dev_percentiles:
                # Calculate deviation thresholds for this video
                deviation_thresholds = {
                    'high': video_data['group_deviation'].quantile(dev_percentile),
                    'low': video_data['group_deviation'].quantile(1 - dev_percentile)
                }
                # Create column name
                col_name = f'Attention_State_z{z_high}_{z_low}_dev{dev_percentile}'
                # Apply attention state labeling
                video_data[col_name] = video_data.apply(
                    lambda row: label_attention_state_raw(
                        row['PC1_z'], 
                        row['group_deviation'],
                        row['Video'],
                        z_high,
                        z_low,
                        deviation_thresholds['high'],
                        deviation_thresholds['low']
                    ), 
                    axis=1
                )
                # Save the video-specific DataFrame for this threshold combination
                video_filename = f'combined_features_with_threshold_states_{video}_z{z_high}_{z_low}_dev{dev_percentile}.csv'
                video_data[[col for col in video_data.columns if col.startswith('Attention_State_z') or col in combined_features_df.columns]].to_csv(os.path.join(output_fig_dir, video_filename), index=False)
                print(f"Saved {video} features with threshold states to: {os.path.join(output_fig_dir, video_filename)}")
                # Create a summary of state distributions for this video and threshold combination
                print(f"Creating summary of state distributions for {video} (z_high={z_high}, z_low={z_low}, dev={dev_percentile})...")
                state_counts = video_data[col_name].value_counts(normalize=True)
                video_state_summaries = []
                for state, proportion in state_counts.items():
                    video_state_summaries.append({
                        'Z_Threshold_High': z_high,
                        'Z_Threshold_Low': z_low,
                        'Dev_Percentile': dev_percentile,
                        'State': state,
                        'Proportion': proportion
                    })
                # Convert to DataFrame and save
                video_state_summaries_df = pd.DataFrame(video_state_summaries)
                summary_filename = f'threshold_state_distributions_summary_{video}_z{z_high}_{z_low}_dev{dev_percentile}.csv'
                video_state_summaries_df.to_csv(os.path.join(output_fig_dir, summary_filename), index=False)
                print(f"Saved {video} state distribution summary to: {os.path.join(output_fig_dir, summary_filename)}")
                # Create a pivot table for easy comparison of threshold combinations
                pivot_data = video_state_summaries_df.pivot_table(
                    values='Proportion',
                    index=['Z_Threshold_High', 'Z_Threshold_Low'],
                    columns=['State'],
                    aggfunc='mean'
                )
                # Save the pivot table
                pivot_filename = f'threshold_state_pivot_{video}_z{z_high}_{z_low}_dev{dev_percentile}.csv'
                pivot_data.to_csv(os.path.join(output_fig_dir, pivot_filename))
                print(f"Saved {video} threshold state pivot table to: {os.path.join(output_fig_dir, pivot_filename)}")

print("\nAll video-specific files have been created successfully!")

