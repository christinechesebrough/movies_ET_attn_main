#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 12 12:32:31 2025

@author: christinechesebrough
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 14:11:45 2024

@author: christinechesebrough
"""

import os, re
from itertools import compress
import numpy as np
import pandas as pd
from scipy import stats, signal, interpolate
from scipy.signal import correlate

from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA
from sklearn.decomposition import FactorAnalysis
#from factor_analyzer import FactorAnalyzer
from sklearn.metrics.pairwise import cosine_similarity
import sys
sys.path.append('/Volumes/Samsung/scripts')  # Add this at top of your script

from r_pca import R_pca

drive = 'Samsung'
vid = 'both_movies'
pca_type = 'robust'

cond = 'both_movies'
num_features = 6

window_len = 10

if window_len == 5:
    num_timepoints = 238
elif window_len == 10:
    num_timepoints = 236

compute_measures = False
#'despicable_me_english'  # inscapes

# Check if final_eye_df exists in memory, if so delete
if 'final_eye_df' in locals():
    del final_eye_df
    
# Check if normed_eye_df exists in memory, if so delete
if 'normed_eye_df' in locals():
    del normed_eye_df
    
data_dir = f'/Volumes/{drive}/Movie_data/movies_prep_standard'
isc_dir = f'/Volumes/{drive}/Movie_data/data/isc'
mne_data_dir = f'/Volumes/{drive}/Movie_data/movies_prep_standard'
elec_dir = f'/Volumes/{drive}/Movie_data/data/electrode_localization'
movie_subs_table = pd.read_csv('/Volumes/Samsung/anatomy/shared_correspondence/movie_subs_master_updated.csv')

#fig_dir = '/Volumes/Samsung/Movie_data/4Jan26_compute_norm_eye_features_by_rec'
fig_dir = f'/Volumes/Samsung/Movie_data/6Apr26_norm_eye_features_by_rec_{window_len}s'
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

entry_type = 'entry_ids'

if cond == 'old_pats':
    if vid == 'despicable_me_english':
        good_entries = [
         'NS127_02_ses-02_run-01',
         'NS135_ses-01_run-01',
         'NS136_ses-01_run-01',
         'NS137_ses-01_run-01',
         'NS138_ses-01_run-01',
         'NS140_ses-01_run-01',
         'NS140_02_ses-02_run-01',
         'NS153_ses-01_run-01',
         'NS154_ses-01_run-01',
         'NS164_ses-01_run-01',
         'NS166_ses-01_run-01',
         'NS174_02_ses-02_run-01',
         'NS174_03_ses-03_run-01'
         ]
    if vid == 'inscapes':
        good_entries = [
         'NS127_02_ses-02_run-01',
         'NS135_ses-01_run-01',
         'NS136_ses-01_run-01',
         'NS137_ses-01_run-01',
         'NS138_ses-01_run-01',
         'NS140_ses-01_run-01',
         'NS140_02_ses-02_run-01',
         'NS154_ses-01_run-01',
         'NS164_ses-01_run-01',
         ]
    if vid == 'both_movies':
        good_entries = [
        'NS127_02_ses-02_run-01',
        'NS135_ses-01_run-01',
        'NS136_ses-01_run-01',
        'NS137_ses-01_run-01',
        'NS138_ses-01_run-01',
        'NS140_ses-01_run-01',
        'NS140_02_ses-02_run-01',
        'NS153_ses-01_run-01',
        'NS155_02_ses-02_run-01',
        'NS164_ses-01_run-01'
    ]

else:
    if vid == 'despicable_me_english':
        # good_entries = [
        #  'NS127_02_ses-02_run-01',
        #  'NS135_ses-01_run-01',
        #  'NS136_ses-01_run-01',
        #  'NS137_ses-01_run-01',
        #  'NS138_ses-01_run-01',
        #  'NS140_ses-01_run-01',
        #  'NS140_02_ses-02_run-01',
        #  'NS153_ses-01_run-01',
        #  'NS155_02_ses-02_run-01',
        #  'NS164_ses-01_run-01',
        #  'NS166_ses-01_run-01',
        #  'NS174_02_ses-02_run-01',
        #  'NS174_03_ses-03_run-01',
        #  'NS178_ses-01_run-01',
        #  #'NS190_ses-01_run-01',
        #  'NS190_ses-01_run-02',
        #  'NS191_ses-01_run-01',
        #  #'NS193_ses-01_run-01',
        #  'NS193_ses-01_run-02',
        #  'NS194_ses-01_run-01',
        #  'NS205_ses-01_run-01'
        #  ]
        good_entries= ['NS127_02_ses-02_run-01', 'NS135_ses-01_run-01',
               'NS136_ses-01_run-01', 'NS137_ses-01_run-01',
               'NS138_ses-01_run-01', 'NS140_ses-01_run-01',
              # 'NS140_02_ses-02_run-01',
               'NS153_ses-01_run-01',
               'NS155_02_ses-02_run-01', 'NS164_ses-01_run-01',
              # 'NS166_ses-01_run-01', 
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
    if vid == 'inscapes':
        # good_entries = [
        #  'NS127_02_ses-02_run-01'
        #  'NS135_ses-01_run-01',
        #  'NS136_ses-01_run-01',
        #  'NS137_ses-01_run-01',
        #  'NS138_ses-01_run-01',
        #  'NS140_ses-01_run-01',
        #  'NS140_02_ses-02_run-01',
        #  'NS154_ses-01_run-01',
        #  'NS155_02_ses-02_run-01',
        #  'NS164_ses-01_run-01',
        #  'NS178_ses-01_run-01',
        #  'NS205_ses-01_run-01',
        #  'NS210_ses-01_run-01',
        #  'NS211_ses-01_run-01'
        #  ]
        good_entries = ['NS127_02_ses-02_run-01', 'NS135_ses-01_run-01',
               'NS136_ses-01_run-01', 'NS137_ses-01_run-01',
               'NS138_ses-01_run-01', 'NS140_ses-01_run-01',
               'NS140_02_ses-02_run-01', 'NS144_ses-01_run-01',
               'NS151_ses-01_run-01', 'NS153_ses-01_run-01',
               'NS154_ses-01_run-01', 'NS155_ses-01_run-01',
               'NS155_02_ses-02_run-01', 'NS164_ses-01_run-01',
               'NS178_ses-01_run-01', 'NS205_ses-01_run-01',
               'NS210_ses-01_run-01', 'NS211_ses-01_run-01']
    
    if vid == 'despicable_me_hungarian':
        good_entries = [
            'LH010_ses-01_run-01',
             'NS127_02_ses-02_run-01', 
             'NS135_ses-01_run-01',
             'NS136_ses-01_run-01',
             'NS137_ses-01_run-01',
             'NS138_ses-01_run-01',
             'NS140_ses-01_run-01',
             'NS140_02_ses-02_run-01',
             'NS145_ses-02_run-01',
             'NS154_ses-01_run-01',
             'NS164_ses-01_run-01', 
             'NS174_02_ses-02_run-01',
             'NS174_03_ses-03_run-01',
             'NS178_ses-01_run-01'
               ]
    
    
    if vid == 'both_movies':
    #     good_entries = [
    #     'NS127_02_ses-02_run-01',
    #     'NS135_ses-01_run-01',
    #     'NS136_ses-01_run-01',
    #     'NS137_ses-01_run-01',
    #     'NS138_ses-01_run-01',
    #     'NS140_ses-01_run-01',
    #     'NS140_02_ses-02_run-01',
    #     'NS153_ses-01_run-01',
    #     'NS155_02_ses-02_run-01',
    #     'NS164_ses-01_run-01',
    #     'NS178_ses-01_run-01',
    #     'NS205_ses-01_run-01'
    # ]
        good_entries = [
            'LH010_ses-01_run-01',
            'NS127_02_ses-02_run-01',
            'NS135_ses-01_run-01',
            'NS136_ses-01_run-01',
            'NS137_ses-01_run-01',
            'NS138_ses-01_run-01',
            'NS140_ses-01_run-01',
            'NS140_02_ses-02_run-01',
            'NS144_ses-01_run-01',
             'NS145_ses-02_run-01',
            'NS151_ses-01_run-01',
            'NS153_ses-01_run-01',
            'NS154_ses-01_run-01',
            'NS155_ses-01_run-01',
            'NS155_02_ses-02_run-01',
            'NS164_ses-01_run-01',
            #'NS166_ses-01_run-01',
            'NS174_02_ses-02_run-01',
            'NS174_03_ses-03_run-01',
            'NS178_ses-01_run-01',
            'NS190_ses-01_run-02',
            'NS191_ses-01_run-01',
            'NS193_ses-01_run-02',
            'NS194_ses-01_run-01',
            'NS205_ses-01_run-01',
            'NS210_ses-01_run-01',
            'NS211_ses-01_run-01'
        ]


patients = good_entries
patients.sort()



#%%# Define a generalized function for computing inter-subject variability and ISC for any dataframe and factor

def parse_entry(entry_id: str):
    m = re.match(r"^(?P<pat>.+?)(?:_(?P<ses>ses-\d+))?(?:_(?P<run>run-\w+))?$", entry_id, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Could not parse entry_id: {entry_id}")

    pat = m.group("pat")
    ses = m.group("ses").lower() if m.group("ses") else None
    run = m.group("run").lower() if m.group("run") else None
    return pat, ses, run

def compute_factor_variability_general(df, factor_name, vid_name,time_col="Time", patient_col="Patient",plot=True):
    """
    Compute intersubject variability, ISC, and PCA for a given factor in a dataframe.

    Parameters:
    - df: DataFrame containing patient-wise time series data
    - factor_name: Column name of the factor to analyze
    - time_col: Name of the time column (default: "Time")
    - patient_col: Name of the patient ID column (default: "Patient")

    Returns:
    - Tuple containing (missing data count, mean ISC)
    """

    # Compute the group mean of the factor at each time point
    df[f"Group_Mean_{factor_name}"] = df.groupby(time_col)[factor_name].transform("mean")
    df[f"Group_Std_{factor_name}"] = df.groupby(time_col)[factor_name].transform("std")

    # Demean the factor across all patients at each time point
    df[f"{factor_name}_demeaned_group"] = df[factor_name] - df[f"Group_Mean_{factor_name}"]

    # Compute intersubject variability over time
    std_factor_over_time = df.groupby(time_col)[f"{factor_name}_demeaned_group"].std()
    mad_factor_over_time = df.groupby(time_col)[f"{factor_name}_demeaned_group"].apply(lambda x: np.mean(np.abs(x - x.mean())))

    # Compute Intersubject Correlation (ISC) Matrix
    factor_wide = df.pivot(index=time_col, columns=patient_col, values=f"{factor_name}_demeaned_group")
    missing_data_check = factor_wide.isna().sum().sum()  # Count missing values

    if missing_data_check == 0:
        isc_matrix = factor_wide.corr()
        mean_isc = isc_matrix.where(np.triu(np.ones(isc_matrix.shape), k=1).astype(bool)).mean().mean()
    else:
        isc_matrix = None
        mean_isc = None

    # Apply PCA on the factor time series across patients
    pca = PCA()
    factor_pca = pca.fit_transform(factor_wide.dropna().T)  # Transpose: rows=patients, cols=time
    explained_variance = pca.explained_variance_ratio_

    # ---- Plot Results ----
    if plot:
        # 1. Standard Deviation Over Time
        plt.figure(figsize=(10, 5))
        plt.plot(std_factor_over_time, marker="o", linestyle="-", label="Standard Deviation")
        plt.title(f"Inter-Subject Variability in {vid_name} {factor_name} (Standard Deviation)")
        plt.xlabel("Time Point")
        plt.ylabel(f"Standard Deviation of {factor_name}")
        plt.grid(True)
        plt.legend()
        plt.show()
    
        # 2. Mean Absolute Deviation (MAD) Over Time
        plt.figure(figsize=(10, 5))
        plt.plot(mad_factor_over_time, marker="o", linestyle="-", label="Mean Absolute Deviation", color="orange")
        plt.title(f"Inter-Subject Variability in {vid_name} {factor_name} (MAD)")
        plt.xlabel("Time Point")
        plt.ylabel(f"MAD of {factor_name}")
        plt.grid(True)
        plt.legend()
        plt.show()
    
        # 3. Heatmap of Intersubject Correlation (ISC) - Only if data is available
        if isc_matrix is not None:
            plt.figure(figsize=(10, 6))
            sns.heatmap(isc_matrix, cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.5)
            plt.title(f"Intersubject Correlation (ISC) of {vid_name} {factor_name}")
            plt.xlabel("Patient")
            plt.ylabel("Patient")
            plt.show()
        else:
            print(f"ISC computation for {factor_name} skipped due to missing data.")
    
        # 4. Scree Plot for PCA Variance Explained
        plt.figure(figsize=(8, 5))
        plt.plot(np.cumsum(explained_variance) * 100, marker="o", linestyle="-", label="Cumulative Variance Explained")
        plt.axhline(y=explained_variance[0] * 100, color="r", linestyle="--", label=f"1st PC: {explained_variance[0]*100:.2f}%")
        plt.title(f"PCA of {factor_name} {vid_name} Across Patients")
        plt.xlabel("Principal Component")
        plt.ylabel("Variance Explained (%)")
        plt.legend()
        plt.grid(True)
        plt.show()

    return missing_data_check, mean_isc

# Function to compute Euclidean distance
def euclidean_distance(p1, p2):
    """Calculate Euclidean distance between two two points."""
    return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)


def compute_patient_pca_influence(df, factor_col, time_col='Time', patient_col='patient'):
    """
    Computes influence of each patient on the 1st principal component (PC1)
    by performing Leave-One-Patient-Out PCA.

    Parameters:
    - df: DataFrame with long-form patient × time data
    - factor_col: name of the column containing the factor data

    Returns:
    - influence_df: DataFrame with similarity scores and influence flags
    - interpretation: summary dict with statistics and any flagged patients
    """

    patients = df[patient_col].unique()
    full_data = df.pivot(index=time_col, columns=patient_col, values=factor_col)

    # Drop any patients with all missing data
    full_data = full_data.dropna(axis=1, how='all')

    # Fit PCA on full data
    full_pca = PCA().fit(full_data.dropna(axis=0).T)
    full_pc1 = full_pca.components_[0]

    similarities = []

    for patient in full_data.columns:
        reduced_data = full_data.drop(columns=patient).dropna(axis=0)

        if reduced_data.shape[1] < 2:
            similarities.append((patient, np.nan))
            continue

        pca = PCA().fit(reduced_data.T)
        pc1 = pca.components_[0]
        sim = cosine_similarity(full_pc1.reshape(1, -1), pc1.reshape(1, -1))[0, 0]
        similarities.append((patient, sim))

    # Build result DataFrame
    influence_df = pd.DataFrame(similarities, columns=['Patient', 'PC1_similarity'])
    influence_df = influence_df.sort_values(by='PC1_similarity', ascending=True)

    # Compute stats
    valid_sims = influence_df['PC1_similarity'].dropna()
    mean_sim = valid_sims.mean()
    std_sim = valid_sims.std()
    min_sim = valid_sims.min()

    # Flag patients more than 2 SD below the mean
    threshold = mean_sim - 2 * std_sim
    influence_df['Influential'] = influence_df['PC1_similarity'] < threshold

    interpretation = {
        'mean_similarity': mean_sim,
        'std_similarity': std_sim,
        'min_similarity': min_sim,
        'influence_threshold': threshold,
        'num_influential_patients': influence_df['Influential'].sum(),
        'influential_patients': influence_df.loc[influence_df['Influential'], 'Patient'].tolist()
    }

    return influence_df, interpretation



#%%

# if entry == entry_id, the rest of the script will index by entry_id

# Define the filename
normed_feature_filename = os.path.join(fig_dir, f'many_normed_et_features_for_pca_{vid}.csv')

# Load if needed
if 'normed_eye_df' not in locals():
    if os.path.exists(normed_feature_filename):
        normed_eye_df = pd.read_csv(normed_feature_filename)
        print(f"Loaded {normed_feature_filename} into normed_eye_df.")
    else:
        raise FileNotFoundError(f"File {normed_feature_filename} not found. Ensure it exists before proceeding.")

entry_ids = normed_eye_df["patient"].astype(str).unique().tolist()

patients_set = set(patients)  # assumes you have `patients = [...]`

if entry_type == 'patients':
    # ---- FILTER TO PATIENTS LIST (do this once, before PCA prep) ----
    normed_eye_df = normed_eye_df[normed_eye_df["Patient"].isin(patients_set)].copy()
    
    # Exclude the 'Patient' column (first column) for PCA
    if vid == 'both_movies':
        data_for_pca = normed_eye_df.drop(columns=['Patient', 'Video'])
    else:
        data_for_pca = normed_eye_df.drop(columns=['Patient'])

if entry_type == 'entry_ids':
    normed_eye_df = normed_eye_df[normed_eye_df["patient"].isin(patients_set)].copy()
    # Count unique patients AFTER filtering
    n_pats = normed_eye_df['patient'].nunique()
    print("Patients kept:", sorted(normed_eye_df["patient"].unique()))

    # Exclude the 'Patient' column (first column) for PCA
    if vid == 'both_movies':
        drop_cols = ['patient','video','Saccade_Dispersion_Std', 'Abs_Vergence']
        data_for_pca = normed_eye_df.drop(columns=drop_cols)
    else:
        drop_cols = ['patient','Saccade_Dispersion_Std', 'Abs_Vergence']
        data_for_pca = normed_eye_df.drop(columns=drop_cols)


eye_feature_filename = os.path.join(fig_dir, f'many_et_features_{vid}.csv')

# Load if needed
if 'final_eye_df' not in locals():
    if os.path.exists(eye_feature_filename):
        final_eye_df = pd.read_csv(eye_feature_filename)
        print(f"Loaded {eye_feature_filename} into final_eye_df.")
    else:
        raise FileNotFoundError(f"File {eye_feature_filename} not found. Ensure it exists before proceeding.")

# ---- FILTER THIS DF TOO (keeps them aligned) ----

if entry_type == 'patients':
    # ---- FILTER TO PATIENTS LIST (do this once, before PCA prep) ----
    final_eye_df = final_eye_df[final_eye_df["Patient"].isin(patients_set)].copy()
    

if entry_type == 'entry_ids':
    final_eye_df = final_eye_df[final_eye_df["patient"].isin(patients_set)].copy()
    print("Patients kept:", sorted(final_eye_df["patient"].unique()))

#%% correlation and VIF of features to be included in PCA to examine collinearity

# Define the file path to save the figure
correlation_plot_path = os.path.join(fig_dir, f"{vid}_feature_correlation_matrix.png")

# Create the heatmap and save it
plt.figure(figsize=(12, 8))
sns.heatmap(data_for_pca.corr(), annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
plt.title(f"{vid} Feature Correlation Matrix (N = {n_pats})")
plt.tight_layout()

# Save the plot as a PNG file
plt.savefig(correlation_plot_path, dpi=300, bbox_inches='tight')
#plt.show()

# Display the saved file path
correlation_plot_path

vif_data = pd.DataFrame()
vif_data["Feature"] = data_for_pca.columns
vif_data["VIF"] = [variance_inflation_factor(data_for_pca.values, i) for i in range(data_for_pca.shape[1])]

print(vif_data)

#%%

# Define feature order and custom names (same as used in earlier plots)
feature_order = [
    'Saccade_Rate', 
    'Saccade_Dispersion', 
   # 'Saccade_Dispersion_Std',
    'Vergence',
    'Vergence_Std', 
   # 'Abs_Vergence',
    'Blink_Rate', 
    'Blink_Duration',
    'Pupil_Avg', 
    'Pupil_Std',
    'ISC'
]

feature_renames = {
    'Saccade_Rate': 'Saccade Rate',
    'Saccade_Dispersion': 'Saccade Dispersion',
   # 'Saccade_Dispersion_Std': 'Saccade Dispersion SD',
    'Vergence': 'Gaze Disparity',
    'Vergence_Std': 'Gaze Disparity SD',
   # 'Abs_Vergence': 'Abs Vergence',
    'Blink_Rate': 'Blink Rate',
    'Blink_Duration': 'Blink Duration',
    'Pupil_Avg': 'Pupil Size Avg',
    'Pupil_Std': 'Pupil Size SD',
    'ISC': 'Gaze Position \n Inter-subject correlation'
}


if pca_type == 'regular':
    # Perform PCA
    pca = PCA()
    pca_result = pca.fit_transform(data_for_pca)
    
    # Explained variance ratio 
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = explained_variance_ratio.cumsum()
    
    # Plot explained variance ratio
    plt.figure(figsize=(8, 5))
    plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.7, align='center',
            label='Individual Explained Variance')
    plt.step(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, where='mid',
             label='Cumulative Explained Variance')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title(f'{vid} PCA Explained Variance')
    plt.legend(loc='best')
    plt.tight_layout()
    #plt.show()
    
    eye_df_w_pcs = normed_eye_df.copy()
    
    # Save PCA results back into the DataFrame
    for i in range(pca_result.shape[1]):
        eye_df_w_pcs[f'PC{i+1}'] = pca_result[:, i]
    
    # Create a time variable (1 to num_timepoints repeating for each patient)
    num_timepoints = num_timepoints  # Number of timepoints per patient
    eye_df_w_pcs["Time"] = np.tile(np.arange(1, num_timepoints + 1), len(eye_df_w_pcs) // num_timepoints)
    
    features_w_pcs = os.path.join(fig_dir, f"{vid}_normed_features_w_pcs.csv")
    
    # Save the eye features w pcs to csv file
    eye_df_w_pcs.to_csv(features_w_pcs, index=False)
    
    unnormed_eye_df_w_pcs = final_eye_df.copy()
    
    # Save PCA results into not-normed eye df 
    for i in range(pca_result.shape[1]):
        unnormed_eye_df_w_pcs[f'PC{i+1}'] = pca_result[:, i]
    
    # Create a time variable (1 to num_timepoints repeating for each patient)
    num_timepoints = num_timepoints  # Number of timepoints per patient
    unnormed_eye_df_w_pcs["Time"] = np.tile(np.arange(1, num_timepoints + 1), len(unnormed_eye_df_w_pcs) // num_timepoints)
    
    unnormed_features_w_pcs = os.path.join(fig_dir, f"{vid}_unnormed_features_w_pcs.csv")
    
    # Save the eye features w pcs to csv file
    unnormed_eye_df_w_pcs.to_csv(unnormed_features_w_pcs, index=False)


if pca_type == 'robust':
    
    # Run Robust PCA
    data_for_pca = data_for_pca[feature_order]  # Enforces correct order
    print("Feature order going into robust PCA:", list(data_for_pca.columns))

    rpca = R_pca(data_for_pca)
    L, S = rpca.fit(max_iter=10000, iter_print=1000)
    
    # Now run standard PCA on the low-rank signal (L)
    from sklearn.decomposition import PCA
    
   # pca = PCA()
   # pca_result = pca.fit_transform(L)
    
    k = num_features
    pca = PCA(n_components=k)
    pca_result = pca.fit_transform(L)
    
    if vid == 'inscapes':
    
    # Flip PC1 (CHECK SIGN)
        pca_result[:, 0] *= -1
        pca.components_[0, :] *= -1
            
    # Continue exactly as before
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = explained_variance_ratio.cumsum()

    
    # Plot explained variance ratio
    plt.figure(figsize=(8, 5))
    plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.7, align='center',
            label='Individual Explained Variance')
    plt.step(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, where='mid',
             label='Cumulative Explained Variance')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title(f'{vid} Robust PCA Explained Variance')
    plt.legend(loc='best')
    plt.tight_layout()
    
    num_timepoints = num_timepoints  # Number of timepoints per patient

    
    # Save PCA results back into the normalized DataFrame
    eye_df_w_pcs = normed_eye_df.copy()
    for i in range(pca_result.shape[1]):
        eye_df_w_pcs[f'PC{i+1}'] = pca_result[:, i]
    
    eye_df_w_pcs["Time"] = np.tile(np.arange(1, num_timepoints + 1), len(eye_df_w_pcs) // num_timepoints)
    eye_df_w_pcs.to_csv(os.path.join(fig_dir, f"{vid}_normed_features_w_pcs_Robust_PCA.csv"), index=False)
    
    # Save PCA results into the unnormalized DataFrame
    unnormed_eye_df_w_pcs = final_eye_df.copy() 
    for i in range(pca_result.shape[1]):
        unnormed_eye_df_w_pcs[f'PC{i+1}'] = pca_result[:, i]
    
    unnormed_eye_df_w_pcs["Time"] = np.tile(np.arange(1, num_timepoints + 1), len(unnormed_eye_df_w_pcs) // num_timepoints)
    unnormed_eye_df_w_pcs.to_csv(os.path.join(fig_dir, f"{vid}_unnormed_features_w_pcs_Robust_PCA.csv"), index=False)


#%% Extract loadings and save to copy of df with eye features

if pca_type == 'regular':
    # Create loadings DataFrame from PCA components
    loadings = pca.components_.T
    loadings_df = pd.DataFrame(
        loadings,
        index=data_for_pca.columns,
        columns=[f'PC{i+1}' for i in range(loadings.shape[1])]
    )
    
    # Reorder and rename features
    loadings_df = loadings_df.loc[feature_order]
    loadings_df.index = [feature_renames[feat] for feat in feature_order]
    
    # Show all PCs
    loadings_df = loadings_df.iloc[:, :11]
    
    # Save loadings
    pca_loadings_path = os.path.join(fig_dir, f"{vid}_pca_loadings.csv")
    loadings_df.to_csv(pca_loadings_path, index=True)
    
    # Plot the heatmap
    plt.figure(figsize=(8, 8))
    sns.heatmap(loadings_df,
                cmap='coolwarm',
                annot=True,
                fmt=".2f",
                cbar_kws={'label': 'Loading Value'},
                annot_kws={'fontsize': 10})  # larger numbers in cells
    
    # Style adjustments
    plt.title(f'PCA Loadings for {vid} (N = {n_pats} window size: {window_len}s)', fontsize=14)
    plt.xlabel('Principal Components', fontsize=14)
    plt.ylabel('Features', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    
    # Save the figure
    pc_loadings_heatmap_path = os.path.join(fig_dir, f"{vid}_pc_loadings_heatmap.png")
    plt.savefig(pc_loadings_heatmap_path, dpi=300, bbox_inches='tight')
    # plt.show()  # Optional

if pca_type == 'robust':
# Create loadings DataFrame from PCA components (run on L)
    loadings = pca.components_.T
    loadings_df = pd.DataFrame(
        loadings,
        #index = data_for_pca.columns,
        index=feature_order,  # instead of data_for_pca.columns
        columns=[f'PC{i+1}' for i in range(loadings.shape[1])]
    )
    
    # Reorder and rename features
    loadings_df = loadings_df.loc[feature_order]
    loadings_df.index = [feature_renames[feat] for feat in feature_order]
    
    # Show all PCs
    loadings_df = loadings_df.iloc[:, :11]
    
    # Save loadings
    pca_loadings_path = os.path.join(fig_dir, f"{vid}_pca_loadings.csv")
    loadings_df.to_csv(pca_loadings_path, index=True)
    
    # Plot the heatmap
    plt.figure(figsize=(8, 8))
    sns.heatmap(loadings_df,
                cmap='coolwarm',
                annot=True,
                fmt=".2f",
                cbar_kws={'label': 'Loading Value'},
                annot_kws={'fontsize': 12})
    
    plt.title(f'PCA Loadings for {vid} (N = {n_pats} window size: {window_len}s)', fontsize=14)
    plt.xlabel('Principal Components', fontsize=14)
    plt.ylabel('Features', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    
    # Save the figure
    pc_loadings_heatmap_path = os.path.join(fig_dir, f"{vid}_pc_loadings_heatmap_robust.png")
    plt.savefig(pc_loadings_heatmap_path, dpi=300, bbox_inches='tight')


    #%% Visualize and quantify variation in particular pc/factor
rest = True
if rest == True:    
    if vid == 'despicable_me_english':
        factor = 'PC1'
    elif vid == 'inscapes':
        factor = 'PC1'
    elif vid == 'both_movies':
        factor ='PC1'
    
    eye_df_w_pcs["patient"] = eye_df_w_pcs["patient"].astype(str)
    
    # Plot selected factor over time for each patient
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=eye_df_w_pcs, x=eye_df_w_pcs.groupby("patient").cumcount(), y=f"{factor}", hue="patient", palette="tab10", alpha=0.4)
    
    plt.title(f"{factor} over time across patients - {vid}")
    plt.xlabel("Time Point")
    plt.ylabel(f"{factor} Score")
    plt.legend(title="Patient", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    #plt.show()
    
    #%%
    from scipy.stats import f_oneway, kruskal
    
    # generate a histogram of each patient's factor loadings for the factor defined (e.g. 'PC1' or 'PC2' as defined above) 
    #across the full time series (num_timepoints timepoints)
    # not in order of Time, but in order of increasing loading values, to see if there are any patients with similar distributions
    # of loading values 
    
    # Loop through each video
    #for video_id in eye_df_w_pcs['Video'].unique():
    #    print(f"\nProcessing video: {video_id}")
    
    # Subset data for this video
    if vid == "both_movies":
        video_df = eye_df_w_pcs[eye_df_w_pcs['Video'] == video_id.copy()]
    else:
        video_df = eye_df_w_pcs
        video_id = vid 
    
    # Prepare time counter
    video_df['Time_Point'] = video_df.groupby("patient").cumcount()
    
    plt.figure(figsize=(8, 4))
    
    patients = video_df['patient'].unique()
    palette = sns.color_palette("tab10", n_colors=len(patients))
    color_dict = dict(zip(patients, palette))
    
    # highlighted_patients = {
    #     'inscapes': 'NS155_02_ses-02_run-01',
    #     'despicable_me_english': 'NS127_02_ses-02_run-01'
    # }
    
    # highlight_patient = highlighted_patients.get(video_id, None)
    
    # Plot individual lines
  
    # Plot mean of non-highlighted patients in black
    # Exclude the highlighted patient from the mean
    #non_highlight_df = video_df[video_df['patient'] != highlight_patient]
    
    # Compute mean across non-highlighted patients
    mean_df = video_df.groupby('Time_Point')[factor].mean()
    
    # Plot individual patients
    for patient in patients:
        patient_df = video_df[video_df['patient'] == patient]
        plt.plot(
            patient_df['Time_Point'],
            patient_df[factor],
            label=patient,
            alpha=0.25,
            color=color_dict[patient]
        )
    
    # Plot group mean
    plt.plot(
        mean_df.index,
        mean_df.values,
        color='black',
        linewidth=2,
        alpha=1.0,
        label='Non-highlighted Mean'
    )
    
    plt.title(f"{factor} over time across patients - {video_id}")
    plt.xlabel("Time Point")
    plt.ylabel(f"{factor} Score")
    plt.ylim(-3, 3)
    plt.grid(True)
    plt.legend(title="Patient", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    
    #plt.show()
    
    # ----------------------------
    # HISTOGRAM PLOTS PER PATIENT
    # ----------------------------
    plot_hist = True
    if plot_hist:
        unique_patients = video_df['patient'].unique()
        num_patients = len(unique_patients)
        num_cols = 3
        num_rows = (num_patients + num_cols - 1) // num_cols
    
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(5, 2 * num_rows))
        fig.suptitle(f'Distribution of {factor} Loadings by Patient in {video_id}', fontsize=10)
    
        axes = axes.flatten() if num_rows > 1 else axes.reshape(1, -1).flatten()
    
        for idx, patient in enumerate(unique_patients):
            patient_data = video_df[video_df['patient'] == patient]
            loadings = patient_data[factor]
        
            axes[idx].hist(loadings, bins=30, alpha=0.7)
            axes[idx].set_title(f'Patient {patient}')
            axes[idx].set_xlabel(f'{factor} Loading Value')
            axes[idx].set_ylabel('Count')
            axes[idx].grid(True)
            axes[idx].set_xlim(-3, 3)   # Consistent x-axis
            axes[idx].set_ylim(0, 25)   # Consistent y-axis
    
        for idx in range(len(unique_patients), len(axes)):
            axes[idx].set_visible(False)
    
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)  # Make room for suptitle
        plt.savefig(os.path.join(fig_dir, f"{video_id}_{factor}_histograms_by_patient.png"), dpi=300)
    
    # ----------------------------
    # SUMMARY STATS & BAR PLOTS
    # ----------------------------
    factor_summary = video_df.groupby("patient")[factor].agg(
        mean='mean',
        abs_mean=lambda x: np.mean(np.abs(x)),
        std='std',
        min='min',
        max='max',
        median='median',
        count='count'
    ).reset_index()
    
    factor_summary = factor_summary.sort_values(by='mean')
    
    # Statistical tests
    factor_by_patient = [group[factor].values for name, group in video_df.groupby("patient")]
    anova_result = f_oneway(*factor_by_patient)
    kruskal_result = kruskal(*factor_by_patient)
    
    print(f"ANOVA p-value for {factor}: {anova_result.pvalue:.4f}")
    print(f"Kruskal-Wallis p-value for {factor}: {kruskal_result.pvalue:.4f}")
    
    # 3-panel barplot
    fig, axes = plt.subplots(3, 1, figsize=(6, 6), sharex=True)
    fig.suptitle(f'Distribution of {factor} projections by patient in {video_id}', fontsize=10)
    
    sns.barplot(data=factor_summary, x='patient', y='mean', ax=axes[0], palette='coolwarm')
    axes[0].set_title(f"Mean {factor} Score per Patient (Direction)")
    axes[0].set_ylabel(f"Mean {factor}")
    axes[0].axhline(0, color='black', linestyle='--')
    
    sns.barplot(data=factor_summary, x='patient', y='abs_mean', ax=axes[1], palette='magma')
    axes[1].set_title(f"Mean Absolute {factor} Score per Patient (Strength)")
    axes[1].set_ylabel(f"Mean |{factor}|")
    
    sns.barplot(data=factor_summary, x='patient', y='std', ax=axes[2], palette='viridis')
    axes[2].set_title(f"Standard Deviation of {factor} per Patient (Variability)")
    axes[2].set_ylabel(f"{factor} Std Dev")
    axes[2].set_xlabel("patient")
    
    # Make x-axis tick labels smaller
    for ax in axes:
        ax.tick_params(axis='x', labelsize=8)  # Adjust font size here
    
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"{video_id}_{factor}_patient_summary_bars.png"), dpi=300)
    
    
    influence_df, interp = compute_patient_pca_influence(video_df, factor)
    
    print("LOPO-PCA Influence Summary:")
    print(f"Mean similarity: {interp['mean_similarity']:.3f}")
    print(f"Standard deviation: {interp['std_similarity']:.3f}")
    print(f"Minimum similarity: {interp['min_similarity']:.3f}")
    print(f"Influence threshold (mean - 2*SD): {interp['influence_threshold']:.3f}")
    print(f"Patients flagged as influential: {interp['influential_patients']}")
    
    
    plt.figure(figsize=(8, 4))
    sns.barplot(data=influence_df, x='patient', y='PC1_similarity', palette='coolwarm')
    plt.axhline(interp['influence_threshold'], color='red', linestyle='--', label='Influence Threshold')
    plt.axhline(1.0, color='gray', linestyle=':')
    plt.title(f'LOPO-PCA: Influence of Patients on {factor} ({video_id})')
    plt.ylabel("Cosine Similarity to Full PC1")
    plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"{video_id}_{factor}_PC1_influence_barplot.png"), dpi=300)
    
    
    # Save summary to CSV
    summary_save_path = os.path.join(fig_dir, f"{video_id}_{factor}_patient_summary.csv")
    factor_summary.to_csv(summary_save_path, index=False)
    
    
    #%%
    # Loop through each video and run compute_factor_variability_general on the subsetted df
    results_all_videos = {}
    
    #for video_id in eye_df_w_pcs['Video'].unique():
       # print(f"\nProcessing video: {video_id}")
    
        # Subset data for this video
       # video_df = eye_df_w_pcs[eye_df_w_pcs['Video'] == video_id].copy()
    
    video_df = eye_df_w_pcs
    
    # Select appropriate factor for the video
    if video_id == 'despicable_me_english':
        factor = 'PC1'
    elif video_id == 'inscapes':
        factor = 'PC1'
    else:
        print(f"Skipping video {video_id} (unknown factor)")
        #continue
    
    # Compute and plot variability
    missing, mean_isc = compute_factor_variability_general(video_df, factor, video_id, plot=True)
    
    # Optionally store results
    results_all_videos[video_id] = {
        "factor": factor,
        "missing_data_count": missing,
        "mean_isc": mean_isc
    }
    
    
    #%% Compute moments of individual variation 
    ## SECOND VERSION WITH "GROUP MEAN" as "external period"
    compute_periods = False
    if compute_periods:
        my_factor_df = unnormed_eye_df_w_pcs.copy()
        
        # Compute Z-scores within each individual for group-demeaned factor
        factor_demeaned = f"{factor}_demeaned_group"
        factor_z_individual = f"{factor}_Z_GroupDemeaned_Indiv"
        
        # Compute Z-score within each individual for the group-demeaned factor
        my_factor_df[factor_z_individual] = my_factor_df.groupby("Patient")[factor_demeaned].transform(
            lambda x: (x - x.mean()) / x.std()
        )
        
        # Define thresholds for internal and external periods
        z_threshold_int = 1  # Z-score threshold for internal periods (1 SD above individual's mean)
        z_threshold_ext = 0.5  # Z-score threshold for external periods (0.5 SD from group mean)
        z_threshold = f"{z_threshold_int}_{z_threshold_ext}"
        
        # Define internal periods based on individual Z-scores of group-demeaned values
        my_factor_df[f"{factor}_Int_GroupDemeaned_Indiv"] = my_factor_df[factor_z_individual] > z_threshold_int
        
        # Define external periods based on tracking the group mean
        # A value is considered "external" if it falls within ±0.5 standard deviation of the group mean
        my_factor_df[f"{factor}_Ext_GroupWindow_Indiv"] = (
            (my_factor_df[factor] >= (my_factor_df[f"Group_Mean_{factor}"] - z_threshold_ext * my_factor_df[f"Group_Std_{factor}"])) &
            (my_factor_df[factor] <= (my_factor_df[f"Group_Mean_{factor}"] + z_threshold_ext * my_factor_df[f"Group_Std_{factor}"]))
        )
        
        # Define the file path to save the df
        factor_intext_csv = os.path.join(fig_dir, f"{vid}_{factor}_IntExt_count_{z_threshold}.csv")
        
        # Save normed eye features w factors as csv
        my_factor_df.to_csv(factor_intext_csv, index=False)
        
        # Count the number of "Int" and "Ext" periods per patient 
        int_ext_counts_group_demeaned = my_factor_df.groupby("Patient")[[f"{factor}_Int_GroupDemeaned_Indiv", f"{factor}_Ext_GroupWindow_Indiv"]].sum()
        
        # Plot the number of "Int" and "Ext" periods per patient for the selected factor
        plt.figure(figsize=(12, 8))
        int_ext_counts_group_demeaned[[f"{factor}_Int_GroupDemeaned_Indiv", 
                                     f"{factor}_Ext_GroupWindow_Indiv"]].plot(kind="bar", stacked=True)
        
        plt.title(f"{vid}-{factor} Int/Ext window count by patient (thresh:{z_threshold})", fontsize=8)
        plt.ylabel("Count of Int/Ext Periods")
        plt.xlabel("Patient")
        plt.legend(["Int Periods", "Ext Periods"])
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        
        # Define the file path to save the figure
        factor_int_and_ext_count = os.path.join(fig_dir, f"{vid}_{factor}_IntExt_count_{z_threshold}.png")
        
        # Save the plot as a PNG file
        plt.savefig(factor_int_and_ext_count, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Define the factor to plot
        factor_to_plot = factor
        
        # Prepare data for plotting
        my_factor_df["Time"] = my_factor_df.groupby("Patient").cumcount()
        
        # Create the plot
        plt.figure(figsize=(12, 6))
        
        # Plot factor over time for each patient with high and low values scatter
        sns.lineplot(data=my_factor_df, x="Time", y=factor, hue="Patient", palette="tab10", alpha=0.7)
        
        # Overlay dots for internal and external points
        for patient in my_factor_df["Patient"].unique():
            patient_data = my_factor_df[my_factor_df["Patient"] == patient]
        
            # Internal periods (red dots)
            plt.scatter(patient_data["Time"][patient_data[f"{factor_to_plot}_Int_GroupDemeaned_Indiv"]],
                        patient_data[factor][patient_data[f"{factor_to_plot}_Int_GroupDemeaned_Indiv"]], s=8,
                        color="red", label="Int Period" if patient == my_factor_df["Patient"].unique()[0] else "", alpha=0.7)
        
            # External periods (blue dots)
            plt.scatter(patient_data["Time"][patient_data[f"{factor_to_plot}_Ext_GroupWindow_Indiv"]],
                        patient_data[factor][patient_data[f"{factor_to_plot}_Ext_GroupWindow_Indiv"]], s=8,
                        color="blue", label="Ext Period" if patient == my_factor_df["Patient"].unique()[0] else "", alpha=0.7)
        
        plt.title(f"{vid}-{factor} Int/Ext windows by patient (thresh:{z_threshold})", fontsize=14)
        plt.xlabel("Time Point")
        plt.ylabel(f"{factor_to_plot} Score")
        plt.legend(title="Patient", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True)
        plt.tight_layout()
        
        # Define the file path to save the figure
        factor_int_and_ext_time = os.path.join(fig_dir, f"{vid}_{factor}_IntExt_{z_threshold}.png")
        
        # Save the plot as a PNG file
        plt.savefig(factor_int_and_ext_time, dpi=300, bbox_inches='tight')
        plt.show()
    

