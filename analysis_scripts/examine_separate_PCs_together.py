#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 00:12:36 2025

@author: christinechesebrough
"""

import seaborn as sns
import numpy as np
from numpy.linalg import norm
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np
import os
import matplotlib.patches as patches
from numpy.linalg import inv
from sklearn.covariance import LedoitWolf  # robust shrinkage covariance
#%%
# -----------------------------
# Helper functions
# -----------------------------
def safe_zscore(series):
    """Z-score a pandas Series; return NaNs if variance is zero."""
    sd = series.std(ddof=0)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.nan, index=series.index)
    return (series - series.mean()) / sd


def label_attention_state_strict(pc_z, dev_z, z_thresh_int, z_thresh_ext):
    """
    Label windows using PC1_z and a chosen deviation metric.
    """
    if pd.isna(pc_z) or pd.isna(dev_z):
        return np.nan
    if (pc_z > z_thresh_int) and (dev_z > z_thresh_int):
        return 'Internal_HighConfidence'
    elif (pc_z < -z_thresh_ext) and (dev_z < -z_thresh_ext):
        return 'External_HighConfidence'
    else:
        return 'Neutral'


def summarize_attention_scheme(df, patient_col, label_col, video_name):
    """
    Return:
      1) counts per patient x label
      2) average counts per subject by label
    """
    counts_per_subject = (
        df.groupby([patient_col, label_col])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    counts_per_subject['Video'] = video_name

    label_totals = (
        df.groupby(label_col)
        .size()
        .reset_index(name='Count')
    )
    n_subjects = df[patient_col].nunique()
    label_totals['Timepoints_per_Subject'] = label_totals['Count'] / n_subjects

    return counts_per_subject, label_totals


def plot_counts_per_subject(counts_df, label_col, scheme_name, vid, fig_dir, z_thresh):
    """
    Stacked bar plot of counts per subject for one deviation scheme.
    """
    plot_df = counts_df.copy().set_index(['patient', 'Video'])
    plot_df.plot(kind='bar', stacked=True, figsize=(12, 6))
    plt.title(f'{vid} - Attention State Counts per Subject\n{scheme_name} (z_thresh={z_thresh})')
    plt.ylabel('Number of Timepoints')
    plt.xlabel('Patient')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(
        os.path.join(fig_dir, f'{vid}_{scheme_name}_Attention_States_Per_Subject_z{z_thresh}.png'),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()


def plot_average_counts(label_totals, label_col, scheme_name, vid, fig_dir, z_thresh):
    """
    Bar plot of average timepoints per subject by label for one scheme.
    """
    plt.figure(figsize=(6, 8))
    sns.barplot(
        data=label_totals,
        x=label_col,
        y='Timepoints_per_Subject'
    )
    plt.title(f'{vid} - Avg Timepoints per Subject by Attention State\n{scheme_name} (z_thresh={z_thresh})')
    plt.ylabel('Avg Timepoints per Subject')
    plt.xlabel('Attention State')
    plt.tight_layout()
    plt.savefig(
        os.path.join(fig_dir, f'{vid}_{scheme_name}_Attention_State_Barplot_z{z_thresh}.png'),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()


def plot_scatter_with_thresholds(df, x_col, y_col, scheme_name, factor, vid, fig_dir, z_thresh_int, z_thresh_ext,x_label=None,
    y_label=None):
    """
    Scatter plot of PC1_z vs a selected deviation metric.
    """
    plt.figure(figsize=(8, 6))
    ax = sns.scatterplot(
        data=df,
        x=x_col,
        y=y_col,
        alpha=0.3,
        edgecolor=None
    )

    # Since these are z-scored variables, thresholds are directly at +/- z_thresh
    plt.axvline(0, color='gray', linestyle='--', label=f'{x_col} mean')
    plt.axhline(0, color='gray', linestyle='--', label=f'{y_col} mean')
    plt.axvline(z_thresh_int, color='red', linestyle=':', alpha=0.9, label=f'+{z_thresh_int}')
    plt.axvline(-z_thresh_ext, color='red', linestyle=':', alpha=0.9, label=f'-{z_thresh_ext}')
    plt.axhline(z_thresh_int, color='blue', linestyle=':', alpha=0.9)
    plt.axhline(-z_thresh_ext, color='blue', linestyle=':', alpha=0.9)

    # highlight upper-right / lower-left quadrants
    ax.add_patch(patches.Rectangle((z_thresh_int, z_thresh_int), 5, 5,
                                   linewidth=0, facecolor='green', alpha=0.08))
    ax.text(z_thresh_int + 0.2, z_thresh_int + 0.5,
            'Internal\nHigh Confidence',
            fontsize=11, color='black', fontweight='bold')

    ax.add_patch(patches.Rectangle((-5, -5), 5 - z_thresh_ext, 5 - z_thresh_ext,
                                   linewidth=0, facecolor='purple', alpha=0.08))
    ax.text(-z_thresh_ext - 1.7, -z_thresh_ext - 1.2,
            'External\nHigh Confidence',
            fontsize=11, color='black', fontweight='bold')

    plt.title(f'{vid} - PC1_z vs {scheme_name}')
    
    # Pretty labels
    x_label = f"{scheme_name} deviation\n({x_col})"
    y_label = f"{factor} (z-scored)"
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f'{vid} - {factor} vs {scheme_name} deviation')    
    plt.tight_layout()
    plt.savefig(
        os.path.join(fig_dir, f'{vid}_{scheme_name}_scatter_z{[z_thresh_int, z_thresh_ext]}.png'),
        dpi=300,
        bbox_inches='tight'
    )


#%%

# Define paths
machine_path = 'Volumes' #'media/christine'

pc_type = 'shared'
win_len = 10

z_thresh = [0.7, 0.7]


# Load video-specific data
videos = ['despicable_me_hungarian','despicable_me_english']

pca_dir = f"/{machine_path}/Samsung/Movie_data/6Apr26_norm_eye_features_by_rec_{win_len}s/"

if pc_type == 'separate':
    desp_me_features_df = pd.read_csv(os.path.join(pca_dir,"despicable_me_english_unnormed_features_w_pcs_Robust_PCA.csv"))
    desp_me_hung_features_df = pd.read_csv(os.path.join(pca_dir,"despicable_me_hungarian_unnormed_features_w_pcs_Robust_PCA.csv"))
    inscapes_features_df = pd.read_csv(os.path.join(pca_dir,"inscapes_unnormed_features_w_pcs_Robust_PCA.csv"))
   # desp_me_features_df = pd.read_csv(f"/{machine_path}/Samsung/Movie_data/6Apr26_norm_eye_features_by_rec_5{win_len}s/")
   # desp_me_features_df = pd.read_csv(f"/{machine_path}/Samsung/Movie_data/4Jan26_compute_norm_eye_features_by_rec/despicable_me_english_unnormed_features_w_pcs_Robust_PCA.csv")
   # inscapes_features_df = pd.read_csv(f"/{machine_path}/Samsung/Movie_data/4Jan26_compute_norm_eye_features_by_rec/inscapes_unnormed_features_w_pcs_Robust_PCA.csv")
     
    if np.corrcoef(inscapes_features_df['PC1'], inscapes_features_df['ISC'])[0,1] > 0:
        print("PC1 is likely flipped for inscapes, multiplying by *1")
        inscapes_features_df['PC1'] *= -1

    if np.corrcoef(desp_me_features_df['PC1'], desp_me_features_df['ISC'])[0,1] > 0:
        print("PC1 is likely flipped for despicable me, multiplying by *1")
        desp_me_features_df['PC1'] *= -1

    if win_len == 5:
        fig_dir = f'/{machine_path}/Samsung/Movie_data/separate_PC_features_5s_8Apr26'
    elif win_len == 10:
        fig_dir = f'/{machine_path}/Samsung/Movie_data/separate_PC_features_7Jan26'
    os.makedirs(fig_dir, exist_ok=True)

if pc_type == 'shared':
   features_df = pd.read_csv(os.path.join(pca_dir,"both_movies_unnormed_features_w_pcs_Robust_PCA.csv"))
   
   if np.corrcoef(features_df['PC1'], features_df['ISC'])[0,1] > 0:
       print("PC1 is likely flipped, multiplying by *1")
       features_df['PC1'] *= -1

   desp_me_features_df = features_df[
       features_df['video'] == 'despicable_me_english'
   ].copy()

   inscapes_features_df = features_df[
       features_df['video'] == 'inscapes'
   ].copy()
   
   desp_me_hung_features_df = features_df[
       features_df['video'] == 'despicable_me_hungarian'
   ].copy()
    
   fig_dir = f'/{machine_path}/Samsung/Movie_data/shared_PC_features_{win_len}s_20Apr26'
   os.makedirs(fig_dir, exist_ok=True)


#%%


# -----------------------------
# Main loop
# -----------------------------
# Initialize master list for all attention counts across videos

all_attention_counts = []


# Features to compute deviation
feature_cols = [
    'Saccade_Rate', 'Vergence_Std', 'Abs_Vergence', 'Saccade_Dispersion',
    'Blink_Rate', 'Pupil_Avg', 'Pupil_Std', 'ISC'
]


for vid in videos:
    if vid == 'despicable_me_english':
        feature_df = desp_me_features_df.copy()
        factor = 'PC1'
        vid_name = 'Narrative'
    elif vid == 'inscapes':
        feature_df = inscapes_features_df.copy()
        factor = 'PC1'
        vid_name = 'Ambient'
    elif vid == 'despicable_me_hungarian':
        feature_df = desp_me_hung_features_df.copy()
        factor = 'PC1'
        vid_name = 'Narrative Non-Comprehended Language'

    # -----------------------------
    # 1) Group means per timepoint
    # -----------------------------
    group_means = (
        feature_df
        .groupby('Time')[feature_cols]
        .mean()
        .rename(columns=lambda c: f"{c}_groupmean")
        .reset_index()
    )

    df = feature_df.merge(group_means, on='Time', how='left')

    # -----------------------------
    # 2) Residual vectors: x_i(t) - group_mean(t)
    # -----------------------------
    X = df[feature_cols].to_numpy(dtype=float)
    Mu_t = df[[f"{c}_groupmean" for c in feature_cols]].to_numpy(dtype=float)
    R = X - Mu_t

    valid = np.isfinite(R).all(axis=1)
    df_valid = df.loc[valid].copy()
    R_valid = R[valid]

    # -----------------------------
    # 3) Shrinkage covariance across all valid rows
    # -----------------------------
    lw = LedoitWolf().fit(R_valid)
    Sigma_inv = inv(lw.covariance_)

    # -----------------------------
    # 4) Mahalanobis distance from group norm
    # -----------------------------
    d2 = np.einsum('ij,jk,ik->i', R_valid, Sigma_inv, R_valid)
    df_valid['group_deviation_mahal'] = np.sqrt(np.maximum(d2, 0))

    df['group_deviation_mahal'] = np.nan
    df.loc[valid, 'group_deviation_mahal'] = df_valid['group_deviation_mahal'].to_numpy()

    # -----------------------------
    # 5) Standardizations
    # -----------------------------
    # Within-subject z-score of Mahalanobis
    df['group_dev_mahal_z'] = (
        df.groupby('patient')['group_deviation_mahal']
        .transform(safe_zscore)
    )

    # Within-timepoint z-score of Mahalanobis
    df['mahal_time_z'] = (
        df.groupby('Time')['group_deviation_mahal']
        .transform(safe_zscore)
    )

    # Within-subject z-score of PC1
    df[f'{factor}_z'] = (
        df.groupby('patient')[factor]
        .transform(safe_zscore)
    )

    # For convenience / backward compatibility
    df['group_dev_z'] = df['group_dev_mahal_z']
    df['group_dev_time_z'] = df['mahal_time_z']

    feature_df = df.copy()

    # -----------------------------
    # 6) Compare two deviation schemes
    # -----------------------------
    z_thresh_int, z_thresh_ext = z_thresh

    deviation_schemes = {
        'within_subject_dev': 'group_dev_mahal_z',
        'within_timepoint_dev': 'mahal_time_z'
    }

    for scheme_name, dev_col in deviation_schemes.items():
        label_col = f'Attention_Label_{scheme_name}_{z_thresh}'
        internal_col = f'Internal_HighConf_{scheme_name}_{z_thresh_int}'
        external_col = f'External_HighConf_{scheme_name}_{z_thresh_ext}'

        feature_df[label_col] = feature_df.apply(
            lambda row: label_attention_state_strict(
                row[f'{factor}_z'],
                row[dev_col],
                z_thresh_int,
                z_thresh_ext
            ),
            axis=1
        )

        feature_df[internal_col] = (
            (feature_df[f'{factor}_z'] > z_thresh_int) &
            (feature_df[dev_col] > z_thresh_int)
        ).astype(int)

        feature_df[external_col] = (
            (feature_df[f'{factor}_z'] < -z_thresh_ext) &
            (feature_df[dev_col] < -z_thresh_ext)
        ).astype(int)

        # Print diagnostics
        corr_val = feature_df[[f'{factor}_z', dev_col]].corr().iloc[0, 1]
        print(f'\n{vid_name} | {scheme_name}')
        print(f'  corr({factor}_z, {dev_col}) = {corr_val:.3f}')
        print(feature_df[label_col].value_counts(normalize=True, dropna=False))

        # Summaries
        counts_per_subject, label_totals = summarize_attention_scheme(
            feature_df,
            patient_col='patient',
            label_col=label_col,
            video_name=vid
        )

        counts_per_subject['Scheme'] = scheme_name
        all_attention_counts.append(counts_per_subject.copy())

        # Plots
        plot_counts_per_subject(
            counts_df=counts_per_subject.drop(columns='Scheme'),
            label_col=label_col,
            scheme_name=scheme_name,
            vid=vid,
            fig_dir=fig_dir,
            z_thresh=z_thresh
        )

        plot_average_counts(
            label_totals=label_totals,
            label_col=label_col,
            scheme_name=scheme_name,
            vid=vid,
            fig_dir=fig_dir,
            z_thresh=z_thresh
        )

        plot_scatter_with_thresholds(
            df=feature_df,
            x_col=dev_col,
            y_col=f'{factor}_z',
            scheme_name=scheme_name,
            factor=factor,
            vid=vid,
            fig_dir=fig_dir,
            z_thresh_int=z_thresh_int,
            z_thresh_ext=z_thresh_ext,
            x_label=f"{scheme_name}: {dev_col}",
            y_label=f"{factor} (z-scored)"
        )

    # Save per-video dataframe with both schemes
    feature_df.to_csv(
        os.path.join(fig_dir, f'{vid}_features_df_{z_thresh}.csv'),
        index=False
    )
    print(f"Saved updated feature DataFrame for {vid} to CSV")


# -----------------------------
# Save combined counts across both videos and both schemes
# -----------------------------
attention_counts_df = pd.concat(all_attention_counts, ignore_index=True)
attention_counts_df.to_csv(
    os.path.join(fig_dir, f'all_subject_attention_counts_both_schemes_{z_thresh}.csv'),
    index=False
)

# Long-form plotting table
attention_counts_melted = attention_counts_df.melt(
    id_vars=['patient', 'Video', 'Scheme'],
    var_name='Attention_State',
    value_name='Count'
)

# Keep only the state columns
attention_counts_melted = attention_counts_melted[
    attention_counts_melted['Attention_State'].isin([
        'Internal_HighConfidence',
        'External_HighConfidence',
        'Neutral'
    ])
]

# Plot per video, per scheme
for vid in attention_counts_melted['Video'].unique():
    for scheme_name in attention_counts_melted['Scheme'].unique():
        sub = attention_counts_melted[
            (attention_counts_melted['Video'] == vid) &
            (attention_counts_melted['Scheme'] == scheme_name)
        ]

        plt.figure(figsize=(10, 5))
        sns.barplot(
            data=sub,
            x='patient',
            y='Count',
            hue='Attention_State'
        )
        plt.title(f'{vid} - Attention State Counts per Subject\n{scheme_name} (z={z_thresh})')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(
            os.path.join(fig_dir, f'{vid}_{scheme_name}_attention_counts_barplot_{z_thresh}.png'),
            dpi=300,
            bbox_inches='tight'
        )

# Plot average counts across videos by scheme
mean_counts = (
    attention_counts_melted
    .groupby(['Video', 'Scheme', 'Attention_State'])['Count']
    .mean()
    .reset_index()
)

for scheme_name in mean_counts['Scheme'].unique():
    sub = mean_counts[mean_counts['Scheme'] == scheme_name]

    plt.figure(figsize=(8, 6))
    sns.barplot(
        data=sub,
        x='Attention_State',
        y='Count',
        hue='Video'
    )
    plt.title(f'Average Attention State Counts per Subject\n{scheme_name} (z={z_thresh})')
    plt.ylabel('Avg Count per Subject')
    plt.xlabel('Attention State')
    plt.tight_layout()
    plt.savefig(
        os.path.join(fig_dir, f'average_attention_counts_per_subject_{scheme_name}_{z_thresh}.png'),
        dpi=300,
        bbox_inches='tight'
    )



#%%
# Diagnostics to understand what PC1 and group_deviation_mahal are measuring

from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

inscapes_df = pd.read_csv(os.path.join(fig_dir, f'inscapes_features_df_{z_thresh}.csv'))
dme_df = pd.read_csv(os.path.join(fig_dir, f'despicable_me_english_features_df_{z_thresh}.csv'))

def icc_oneway(df, group_col, value_col):
    """
    One-way random effects ICC (ICC1).

    Question answered:
    Across the full movie, how much of the variance in this variable
    reflects stable between-subject differences, versus within-subject
    fluctuations over time?

    Important:
    This collapses across time, so it is a coarse 'trait-like vs state-like'
    diagnostic, not a measure of momentary divergence.
    """
    d = df[[group_col, value_col]].dropna()

    model = ols(f'{value_col} ~ C({group_col})', data=d).fit()
    aov = anova_lm(model, typ=1)

    ms_between = aov.loc[f'C({group_col})', 'mean_sq']
    ms_within  = aov.loc['Residual', 'mean_sq']

    k = d.groupby(group_col).size().mean()

    icc = (ms_between - ms_within) / (ms_between + (k - 1) * ms_within)

    return {
        'ICC': icc,
        'MS_between': ms_between,
        'MS_within': ms_within,
        'k_avg': k,
        'n_groups': d[group_col].nunique(),
        'n_obs': len(d)
    }

def mean_subject_timeseries_corr(df, time_col, subject_col, value_col):
    """
    Average pairwise correlation across subjects of a derived variable over time.

    Question answered:
    Do subjects tend to fluctuate together over time on this variable?
    """
    mat = df.pivot(index=time_col, columns=subject_col, values=value_col).dropna()
    corr = mat.corr()
    n = len(corr)
    avg_r = (corr.values.sum() - n) / (n * (n - 1))
    return avg_r, corr


#%%
# 1) Window-level association between PC1 and Mahalanobis deviation
# Question: across all subject-windows, are these variables related?

for label, df in [('Narrative', dme_df), ('Ambient', inscapes_df)]:
    r = df[['PC1', 'group_deviation_mahal']].corr().iloc[0,1]
    print(f"{label} | window-level corr(PC1, mahal): {r:.3f}")


#%%
# 2) ICC for raw PC1 and raw Mahalanobis deviation
# Question: are these variables mostly state-like or trait-like across the movie?

for label, df in [('Narrative', dme_df), ('Narrative', dme_df), ('Ambient', inscapes_df)]:
    print(f"\n--- {label} ---")
    for col in ['PC1', 'group_deviation_mahal']:
        result = icc_oneway(df, group_col='patient', value_col=col)
        print(f"{col}: ICC = {result['ICC']:.3f}  "
              f"(MS_between={result['MS_between']:.3f}, MS_within={result['MS_within']:.3f})")


#%%
# 3) ICC for PC1 after removing the group mean at each timepoint
# Question: after removing the shared group timecourse, do subjects still show
# stable offsets relative to the group?

for label, df in [('Narrative', dme_df), ('Ambient', inscapes_df)]:
    df = df.copy()
    df['PC1_resid'] = df['PC1'] - df.groupby('Time')['PC1'].transform('mean')
    result = icc_oneway(df, group_col='patient', value_col='PC1_resid')
    print(f"{label} | PC1_resid ICC = {result['ICC']:.3f}")


for label, df in [('Narrative', dme_df), ('Ambient', inscapes_df)]:
    df = df.copy()
    
    df['mahal_resid'] = (
        df['group_deviation_mahal'] 
        - df.groupby('Time')['group_deviation_mahal'].transform('mean')
    )
    
    result = icc_oneway(df, group_col='patient', value_col='mahal_resid')
    print(f"{label} | mahal_resid ICC = {result['ICC']:.3f}")

#%%
# 4) Average intersubject correlation of derived variables over time
# Question: do subjects rise and fall together over time on PC1 or Mahalanobis?

for label, df in [('Narrative', dme_df), ('Ambient', inscapes_df)]:
    print(f"\n--- {label} ---")
    for col in ['PC1', 'group_deviation_mahal']:
        r, _ = mean_subject_timeseries_corr(df, 'Time', 'patient', col)
        print(f"{col}: mean intersubject time-series corr = {r:.3f}")


#%%
# 5) Correlation of group-mean PC1 and group-mean Mahalanobis over time
# Question: when the group's average PC1 is high, is the group's average
# deviation from the multivariate norm also high?

for label, df in [('Narrative', dme_df), ('Ambient', inscapes_df)]:
    group_means = df.groupby('Time')[['PC1', 'group_deviation_mahal']].mean()
    r = group_means.corr().iloc[0,1]
    print(f"{label} | corr(group mean PC1, group mean mahal) = {r:.3f}")
    
    #%%

for label, df in [('Narrative', dme_df), ('Ambient', inscapes_df)]:
    r1 = df[['PC1_z', 'group_dev_mahal_z']].corr().iloc[0,1]
    r2 = df[['PC1_z', 'mahal_time_z']].corr().iloc[0,1]
    
    print(f"{label}:")
    print(f"  corr(PC1_z, mahal_z)      = {r1:.3f}")
    print(f"  corr(PC1_z, mahal_time_z) = {r2:.3f}")
    
for label, df in [('Narrative', dme_df), ('Ambient', inscapes_df)]:
    group_means = df.groupby('Time')[['PC1_z', 'group_dev_mahal_z', 'mahal_time_z']].mean()
    
    r1 = group_means[['PC1_z', 'group_dev_mahal_z']].corr().iloc[0,1]
    r2 = group_means[['PC1_z', 'mahal_time_z']].corr().iloc[0,1]
    
    print(f"{label}:")
    print(f"  group corr(PC1_z, mahal_z)      = {r1:.3f}")
    print(f"  group corr(PC1_z, mahal_time_z) = {r2:.3f}")
    
def label_from_z(pc, dev, z_thresh):
    z_int, z_ext = z_thresh
    
    labels = []
    for p, d in zip(pc, dev):
        if (p < -z_int) and (d > z_ext):
            labels.append('Internal')
        elif (p > z_int) and (d < -z_ext):
            labels.append('External')
        else:
            labels.append('Neutral')
    return labels


for label, df in [('Narrative', dme_df), ('Ambient', inscapes_df)]:
    df = df.copy()
    
    df['label_mahal_z'] = label_from_z(df['PC1_z'], df['group_dev_mahal_z'], z_thresh)
    df['label_time_z']  = label_from_z(df['PC1_z'], df['mahal_time_z'], z_thresh)
    
    agreement = (df['label_mahal_z'] == df['label_time_z']).mean()
    
    print(f"{label}: label agreement = {agreement:.3f}")
    
    
    print(df['group_dev_mahal_z'].describe())
    print(df['mahal_time_z'].describe())

    plt.hist(df['group_dev_mahal_z'], bins=50, alpha=0.5, label='mahal_z')
    plt.hist(df['mahal_time_z'], bins=50, alpha=0.5, label='mahal_time_z')
    plt.legend()
    plt.show()