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


import os, re,sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import seaborn as sns  
from statsmodels.stats.multitest import fdrcorrection
import matplotlib.patches as patches
from scipy.spatial.distance import norm
from scipy.stats import ttest_1samp,  ttest_ind
from matplotlib.colors import Normalize, TwoSlopeNorm

# sys.path.insert(0, '/Users/christinechesebrough/Documents/EPIPE-movie_nwb/Python')
sys.path.insert(0,'/media/christine/Samsung/EPIPE-movie_nwb/Python')
from epipe import plot_brain_surf

Y7_REGION_MAP = {
    "7Networks_1": "Visual",
    "7Networks_2": "Somatomotor",
    "7Networks_3": "Dorsal Attention",
    "7Networks_4": "Ventral Attention",
    "7Networks_5": "Limbic",
    "7Networks_6": "Frontoparietal",
    "7Networks_7": "Default",
}


machine_path = 'media/christine'

def permute_split(vals, n_int, rng):
    """
    Randomly assign n_int samples to 'int' and the rest to 'ext' from pooled vals.
    """
    idx = rng.permutation(len(vals))
    int_idx = idx[:n_int]
    ext_idx = idx[n_int:]
    return vals[int_idx], vals[ext_idx]

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
normalize = False 

# Lists of frequency bands, movies, and atlases
freq_bands = ['alpha']#,'HFA']  # Example: alpha and HFA bands
#freq_bands = ['alpha','HFA']

lfp_type = 'power'

entropy_type = 'mssd'

movies = ['inscapes']  #['inscapes']
PC_type = 'shared_PC1'#'shared_PC1'#'sep_PCs'#'shared_PC1'
atlases = ['Y17_Atlas_Region']#,'Y17_Atlas_Region','DK_Atlas_Region']#,'Y7_Atlas_Region','Y17_Atlas_Region']

method = 'power_z'

# Choose which z-scoring approach to use: 'individual' or 'group'
z_scoring_approach = 'individual'  # Options: 'individual' or 'group'

# Note: These thresholds are used for documentation only as the actual thresholds
# are now determined by examining_shared_PC_features.py

# Define threshold label - this should match the one used in examining_shared_PC_features.py
#threshold_label = "PC1z_high0.75_low0.5_dev_high.7_low.3"  # Update this to match your actual thresholds

threshold = (.5,.5)

thr_str = f"[{threshold[0]:.1f}, {threshold[1]:.1f}]"

#if PC_type == 'shared_PC1':
#    z_threshold = z_threshold_both
    
#if PC_type == 'sep_PCs':
#    z_threshold = z_threshold_both

correction_method = 'fdr'  # Options: 'fdr', 'bonferroni', or 'none'

#%%
# Output directory
if lfp_type == 'entropy': 
    fig_dir = os.path.join(f'/{machine_path}/Samsung/Movie_data/PC_derived_freq_x_atlas_matrices_ENTROPY/')
elif lfp_type == 'power':
    fig_dir = os.path.join(f'/{machine_path}/Samsung/Movie_data/PC_derived_freq_x_atlas_matrices_7Jan26_withinPats')
os.makedirs(fig_dir, exist_ok=True)


for movie in movies:
    if movie == 'despicable_me_english':
        vidname = 'Narrative'
        #patients = ['NS127_02','NS135','NS137','NS136','NS138','NS140','NS153','NS164','NS166','NS174_02']
        #entry_ids = ['NS140_ses-01_run-01']
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
            #et_file = '/{machine_path}/Samsung/Movie_data/combined_PC_features_output_21Apr25/combined_features_despicable_me_english.csv'
            et_file = f'/{machine_path}/Samsung/Movie_data/shared_PC_features_9Jan26/{movie}_features_df_{thr_str}.csv'
        elif PC_type == "sep_PCs":
           #et_file = f'/{machine_path}/Samsung/Movie_data/combined_PC_features_output_21Apr25/despicable_me_english_features_df_{z_threshold}.csv'
           et_file = f'/{machine_path}/Samsung/Movie_data/separate_PC_features_7Jan26/{movie}_features_df_{thr_str}.csv'
     
        # Read et file as df
        et_df = pd.read_csv(et_file)
        factor = 'PC1'

    elif movie == 'inscapes':
        vidname = 'Ambient'
        #patients = ['NS127_02','NS135','NS136','NS137','NS138','NS140','NS140_02','NS164','NS166']
        entry_ids = ['NS140_ses-01_run-01']
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
            #et_file = '/{machine_path}/Samsung/Movie_data/combined_PC_features_output_21Apr25/combined_features_despicable_me_english.csv'
            et_file = f'/{machine_path}/Samsung/Movie_data/shared_PC_features_9Jan26/{movie}_features_df_{thr_str}.csv'
        elif PC_type == "sep_PCs":
           #et_file = f'/{machine_path}/Samsung/Movie_data/combined_PC_features_output_21Apr25/despicable_me_english_features_df_{z_threshold}.csv'
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
                #data_dir = f'/{machine_path}/Samsung/Movie_data/{freq_band}_all_{movie}_all_cortContacts_wLabels_y17_4Mar25/'
                data_dir = f'/{machine_path}/Samsung/Movie_data/{freq_band}_{movie}_4Jan26'
            elif lfp_type == 'entropy':
                data_dir = f'/{machine_path}/Samsung/Movie_data/entropy_extracted/{freq_band}_all_{movie}_all_cortContacts_entropy_29Mar25'

            # Initialize dictionary to store data for each atlas group across all conditions
           # atlas_group_data = {}
            # patient_region_diffs = {}  # atlas_group -> list of patient-level diffs
            
            # # BEFORE loops: create collector
            # patient_delta_rows = []   # put near top of script (outside loops)
            # electrode_rows = []
            
            electrode_rows = []
            patient_delta_rows = []
            patient_region_diffs = {}

            # Loop through patients
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
               # et_patient_rows = et_df[et_df['patient'] == rec].reset_index(drop=True)
                
                patient_col = next(
                    c for c in et_df.columns if c.lower() == 'patient'
                )
                et_patient_rows = et_df[et_df[patient_col] == rec].reset_index(drop=True)

                
                n_lfp = len(data_rows)
                n_et  = len(et_patient_rows)
                if n_lfp != n_et:
                   # print(f"[WARN] Length mismatch {rec} {movie} {freq_band}: LFP={n_lfp}, ET={n_et}. Skipping.")
                    continue
        
                int_idx = et_patient_rows[et_patient_rows[f'Attention_Label_{thr_str}'] == 'Internal_HighConfidence'].index.to_numpy()
                ext_idx = et_patient_rows[et_patient_rows[f'Attention_Label_{thr_str}'] == 'External_HighConfidence'].index.to_numpy()


                # Normalize electrode columns if required
                electrode_cols = [col for col in lfp_values.columns if col[0] in ['L', 'R'] and col[1:].isalnum()]
                if normalize:
                    data_rows[electrode_cols] = data_rows[electrode_cols].apply(pd.to_numeric, errors='coerce')
                    data_rows[electrode_cols] = (data_rows[electrode_cols] - data_rows[electrode_cols].mean()) / data_rows[electrode_cols].std()

  
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
                    excluded_regions = ['bankssts']

                # Process each atlas group
                for atlas_group, cols in grouped_cols.items():
                    if atlas_group in excluded_regions:
                        continue
                
                    elec_int_means = []
                    elec_ext_means = []
                
                    for elec in cols:
                        if elec not in data_rows.columns:
                            continue
                
                        # robust numeric conversion
                        col_data = pd.to_numeric(data_rows[elec], errors="coerce").to_numpy(float)
                
                        iv = col_data[int_idx]
                        ev = col_data[ext_idx]
                
                        # keep only finite values (handles NaN and inf)
                        iv = iv[np.isfinite(iv)]
                        ev = ev[np.isfinite(ev)]
                
                        # if iv.size < 10 or ev.size < 10:
                        #     continue
                
                        mean_int_e = float(iv.mean())
                        mean_ext_e = float(ev.mean())
                        delta_e = mean_int_e - mean_ext_e
                
                        # Welch t-test (recommended)
                        t_e, p_e = ttest_ind(iv, ev, equal_var=False)
                
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
                            "t": float(t_e),
                            "p": float(p_e),
                            # NOTE: this is not Welch df; keep only if you truly want it as a rough descriptor
                            "df_approx": float(iv.size + ev.size - 2),
                            "n_int": int(iv.size),
                            "n_ext": int(ev.size),
                        })
                
                        # keep for region aggregation (one entry per electrode)
                        elec_int_means.append(mean_int_e)
                        elec_ext_means.append(mean_ext_e)
                
                    # region-level summary for this patient
                    if len(elec_int_means) > 0:
                        mean_int = float(np.nanmean(elec_int_means))
                        mean_ext = float(np.nanmean(elec_ext_means))
                        delta = mean_int - mean_ext
                
                        # for later inferential stats across patients
                        patient_region_diffs.setdefault(atlas_group, []).append(delta)
                
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
                            "n_int": int(len(int_idx)),  # windows labeled internal (all, not per-electrode finite)
                            "n_ext": int(len(ext_idx)),
                        })
            electrodes_df = pd.DataFrame(electrode_rows)
            electrodes_df.to_csv(os.path.join(fig_dir, f"{movie}_electrode_stats_{threshold}_{freq_band}_{atlas}.csv"), index=False)
            
                                
                  #%%
import mne
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

machine_path= 'media/christine'
base_dir = f'/{machine_path}/Samsung/Movie_data'
recon_dir = f'/{machine_path}/Samsung/anatomy' 
raw_dir = f'/{machine_path}/Samsung/Movie_data/rawdata_for_conversion' 
nwb_dir = f'/{machine_path}/Samsung/Movie_data/movies_nwb_standard' 
img_dir = f'/{machine_path}/Samsung/Movie_data/figures/dataset_summary/'

#movie = 'inscapes'
#threshold = (0.7, 0.7)
#freq_band = 'HFA'
#atlas = 'Y17_Atlas_Region'

#if "electrodes_df" not in locals():
electrodes_df = pd.read_csv(
        os.path.join(fig_dir, f"{movie}_electrode_stats_{threshold}_{freq_band}_{atlas}.csv")
    )                  
        
master_path = f"/{machine_path}/Samsung/anatomy/shared_correspondence/movie_subs_master_updated.csv"
corr = pd.read_csv(master_path)

coords_df = corr[["SubID","Contact","Hem","FSAverageInf_1","FSAverageInf_2","FSAverageInf_3"]].copy()
coords_df = coords_df.rename(columns={
    "SubID": "pat_base",
    "Contact": "electrode",
    "Hem": "hem",
    "FSAverageInf_1": "x",
    "FSAverageInf_2": "y",
    "FSAverageInf_3": "z",
})

coords_df["pat_base"] = coords_df["pat_base"].astype(str).str.strip()
coords_df["electrode"] = coords_df["electrode"].astype(str).str.strip()
coords_df["hem"] = coords_df["hem"].astype(str).str.upper().str.strip()
coords_df.loc[coords_df["hem"].isin(["LEFT","LH"]), "hem"] = "L"
coords_df.loc[coords_df["hem"].isin(["RIGHT","RH"]), "hem"] = "R"

coords_df[["x","y","z"]] = coords_df[["x","y","z"]].apply(pd.to_numeric, errors="coerce")
coords_df = coords_df.dropna(subset=["x","y","z"])
coords_df = coords_df.drop_duplicates(subset=["pat_base","electrode"], keep="first")

print(coords_df.shape)
print(coords_df.head())

vis_df = electrodes_df.merge(coords_df, how="inner", on=["pat_base","electrode"])
print("vis_df:", vis_df.shape)

# sanity check
merge_rate = vis_df.shape[0] / max(1, electrodes_df.shape[0])
print(f"merge_rate rows: {merge_rate:.1%}")


miss = electrodes_df.merge(coords_df, on=["pat_base","electrode"], how="left", indicator=True)
print(miss["_merge"].value_counts())
print(miss.loc[miss["_merge"]=="left_only", ["pat_base","electrode"]].head(30))
                
#%%

# def plot_electrode_deltas_fsavg_inf(vis_df, *, movie, freq_band, region=None, atlas=None,
#                                    value_col="delta", min_obs=2, n_bins=8,
#                                    views=("lateral","medial","ventral")):#"dorsal",

#     d = vis_df[(vis_df["movie"] == movie) & (vis_df["freq_band"] == freq_band)].copy()
#     if atlas is not None:
#         d = d[d["atlas"] == atlas]
#     if region is not None:
#         d = d[d["region"] == region]

#     d = d.dropna(subset=["x","y","z", value_col])
#     d = d[(d["n_int"] >= min_obs) & (d["n_ext"] >= min_obs)]
#     if d.empty:
#         raise ValueError("No electrodes left after filters; relax min_obs or check labels/merge.")

#     coords = d[["x","y","z"]].to_numpy(float)
#     hem = d["hem"].astype(str).str.upper().to_numpy()
#     vals = d[value_col].to_numpy(float)

#     vmax = np.nanpercentile(np.abs(vals), 95)
#     vmax = float(vmax) if (vmax and not np.isnan(vmax)) else float(np.nanmax(np.abs(vals)))
#     vmin = -vmax

#     bins = np.linspace(vmin, vmax, n_bins + 1)
#     bin_id = np.clip(np.digitize(vals, bins) - 1, 0, n_bins - 1)

#     cmap = plt.colormaps["coolwarm"]
#     colors = cmap(np.linspace(0, 1, n_bins))[:, :3]

#     tag = f"{movie}_{freq_band}"
#     if atlas:  tag += f"_{atlas}"
#     if region: tag += f"_{region.replace(' ', '')}"

#     # LH
#     mne_fig = mne.viz.create_3d_figure((1000,1000), show=True, bgcolor="white")
#     brain = mne.viz.Brain("fsaverage", hemi="lh", subjects_dir=recon_dir,
#                           surf="inflated", figure=mne_fig, cortex="classic",
#                           background="white", alpha=1)
#     for k in range(n_bins):
#         sel = (bin_id == k) & (hem == "L")
#         if np.any(sel):
#             brain.add_foci(coords[sel], scale_factor=0.4, color=colors[k])
#     for v in views:
#         brain.show_view(v)
#         brain.save_image(os.path.join(img_dir, f"{tag}_lh_{v}_{atlas}_{region}.png"))
#     mne.viz.close_all_3d_figures()

#     # RH
#     mne_fig = mne.viz.create_3d_figure((1000,1000), show=True, bgcolor="white")
#     brain = mne.viz.Brain("fsaverage", hemi="rh", subjects_dir=recon_dir,
#                           surf="inflated", figure=mne_fig, cortex="classic",
#                           background="white", alpha=1)
#     for k in range(n_bins):
#         sel = (bin_id == k) & (hem == "R")
#         if np.any(sel):
#             brain.add_foci(coords[sel], scale_factor=0.4, color=colors[k])
#     for v in views:
#         brain.show_view(v)
#         save_path =os.path.join(img_dir, f"{tag}_rh_{v}_{atlas}_{region}.png")
#         brain.save_image(save_path)
#         print(save_path)
#        # print("image saved to:",save path)
#     #mne.viz.close_all_3d_figures()
#     print("Saved:", tag)

def plot_electrode_deltas_fsavg_inf_abs(
    vis_df, *, movie, freq_band, region=None, atlas=None,
    value_col="t", min_obs=2,
    views=("lateral", "medial", "ventral"),
    bin_values=True,
    n_bins=8,
    cmap_name="coolwarm",
    clim="auto",
    clip_percentile=95,
    scale_factor=0.4,
    rgb_round=2,
    max_color_groups=300
):
    """
    Plot electrode-level ABS(delta) values on fsaverage inflated surface.

    Compatible with MNE versions where Brain.add_foci() requires a single RGB color per call.

    - bin_values=True: bins into n_bins (fast).
    - bin_values=False: continuous-looking via value->RGB, then grouping by rounded RGB and
      calling add_foci once per color group.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import mne
    from matplotlib.colors import Normalize

    # ---- filter ----
    d = vis_df[(vis_df["movie"] == movie) & (vis_df["freq_band"] == freq_band)].copy()
    if atlas is not None:
        d = d[d["atlas"] == atlas]
    if region is not None:
        d = d[d["region"] == region]

    d = d.dropna(subset=["x", "y", "z", value_col])
    d = d[(d["n_int"] >= min_obs) & (d["n_ext"] >= min_obs)]
    if d.empty:
        raise ValueError("No electrodes left after filters; relax min_obs or check labels/merge.")

    coords = d[["x", "y", "z"]].to_numpy(float)
    hem = d["hem"].astype(str).str.upper().to_numpy()

    vals = np.abs(d[value_col].to_numpy(float))

    # ---- scaling ----
    if clim != "auto":
        raise ValueError("This implementation supports clim='auto' only; control range with clip_percentile.")

    vmax = np.nanpercentile(vals, clip_percentile)
    vmax = float(vmax) if np.isfinite(vmax) and vmax > 0 else float(np.nanmax(vals))
    if not np.isfinite(vmax) or vmax <= 0:
        raise ValueError("vmax is zero/NaN; check vals (all zeros or all NaN after filtering).")

    vals_clip = np.clip(vals, 0.0, vmax)

    # ---- filename tag ----
    tag = f"{movie}_{freq_band}"
    if atlas:
        tag += f"_{atlas}"
    if region:
        tag += f"_{region.replace(' ', '')}"

    cmap = plt.colormaps[cmap_name]

    # ---- define point-adding helper ----
    if bin_values:
        bins = np.linspace(0.0, vmax, n_bins + 1)
        bin_id = np.clip(np.digitize(vals_clip, bins) - 1, 0, n_bins - 1)
        bin_colors = cmap(np.linspace(0, 1, n_bins))[:, :3]  # (n_bins, 3)

        def add_points(brain, hemi_code):
            for k in range(n_bins):
                sel = (hem == hemi_code) & (bin_id == k)
                if np.any(sel):
                    brain.add_foci(
                        coords[sel],
                        scale_factor=scale_factor,
                        color=tuple(float(x) for x in bin_colors[k])
                    )
    else:
        norm = Normalize(vmin=0.0, vmax=vmax)
        rgb = cmap(norm(vals_clip))[:, :3]  # (N, 3) floats in [0,1]

        # Coarsen rounding if too many unique colors
        dec = int(rgb_round)
        while True:
            rgb_q = np.round(rgb, decimals=dec)
            color_keys = [tuple(float(x) for x in row) for row in rgb_q]  # list of tuples
            n_unique = len(set(color_keys))
            if n_unique <= max_color_groups or dec <= 0:
                break
            dec -= 1

        def add_points(brain, hemi_code):
            sel_idx = np.where(hem == hemi_code)[0]
            if sel_idx.size == 0:
                return

            groups = {}  # key: (r,g,b) tuple -> list of coords
            for i in sel_idx:
                k = color_keys[i]          # guaranteed hashable tuple
                groups.setdefault(k, []).append(coords[i])

            for k, pts in groups.items():
                pts = np.asarray(pts, float)
                brain.add_foci(pts, scale_factor=scale_factor, color=k)

    # ---- LH ----
    mne_fig = mne.viz.create_3d_figure((1000, 1000), show=True, bgcolor="white")
    brain = mne.viz.Brain(
        "fsaverage", hemi="lh", subjects_dir=recon_dir,
        surf="inflated", figure=mne_fig, cortex="classic",
        background="white", alpha=1
    )
    add_points(brain, "L")
    for v in views:
        brain.show_view(v)
        brain.save_image(os.path.join(img_dir, f"{tag}_lh_{v}.png"))
    mne.viz.close_all_3d_figures()

    # ---- RH ----
    mne_fig = mne.viz.create_3d_figure((1000, 1000), show=True, bgcolor="white")
    brain = mne.viz.Brain(
        "fsaverage", hemi="rh", subjects_dir=recon_dir,
        surf="inflated", figure=mne_fig, cortex="classic",
        background="white", alpha=1
    )
    add_points(brain, "R")
    for v in views:
        brain.show_view(v)
        save_path = os.path.join(img_dir, f"{tag}_rh_{v}.png")
        brain.save_image(save_path)
        print("Saved:", save_path)

    print("Done:", tag)


#plot_electrode_deltas_fsavg_inf(vis_df,movie = "despicable_me_english",freq_band = "HFA",atlas = "Y17_Atlas_Region",region = "Default B")
# plot_electrode_deltas_fsavg_inf(vis_df, movie="inscapes", freq_band="HFA",
#                                 atlas="Y17_Atlas_Region", region="Default A")


# plot_electrode_deltas_fsavg_inf_abs(
#     vis_df, movie="despicable_me_english", freq_band="HFA",
#     atlas="Y17_Atlas_Region",region = "Dorsal Attention A",
#     bin_values=False, clip_percentile=95
# )

#%%
import numpy as np

def vis_df_to_plot_brain_inputs(
    vis_df, *, movie, freq_band, atlas=None, region=None,
    value_col="t", abs_val=False, min_obs=2
):
    d = vis_df[(vis_df["movie"] == movie) & (vis_df["freq_band"] == freq_band)].copy()
    if atlas is not None:
        d = d[d["atlas"] == atlas]
    if region is not None:
        d = d[d["region"] == region]

    # match your earlier plotting filters
    d = d.dropna(subset=["x","y","z", value_col, "hem"])
    d = d[(d["n_int"] >= min_obs) & (d["n_ext"] >= min_obs)]
    if d.empty:
        raise ValueError("No rows left after filtering vis_df.")

    coords = d[["x","y","z"]].to_numpy(float)

    # plot_brain_surf expects 'l'/'r'
    elec_hem = (
        d["hem"].astype(str).str.upper().map({"L":"l", "R":"r"})
        .to_numpy()
    )
    if np.any([h not in ("l","r") for h in elec_hem]):
        bad = d.loc[~d["hem"].astype(str).str.upper().isin(["L","R"]), "hem"].unique()
        raise ValueError(f"Unexpected hem values in vis_df: {bad}")

    # names: any unique-ish labels
    if "pat_base" in d.columns and "electrode" in d.columns:
        elec_names = (d["pat_base"].astype(str) + "_" + d["electrode"].astype(str)).to_list()
    else:
        elec_names = [f"e{i:04d}" for i in range(len(d))]

    vals = d[value_col].to_numpy(float)
    if abs_val:
        vals = np.abs(vals)

    return elec_names, coords, elec_hem, vals

elec_names, coords, elec_hem, vals = vis_df_to_plot_brain_inputs(
    vis_df,
    movie=movie,
    freq_band=freq_band,
    atlas=atlas,
    value_col="t",
    abs_val=False
)

vals = vals.astype(float)
v = np.nanpercentile(np.abs(vals), 99)
vals_clip = np.clip(vals, -v, v)

# fig = plot_brain_surf(
#     elec_names=elec_names,
#     coords=coords,
#     elec_hem=elec_hem.tolist(),
#     subject="fsaverage",
#     subjects_dir=recon_dir,
#     surf="inflated",
#     #views = "omni",
#     views=[["l_lateral", "l_medial"], ["r_lateral", "r_medial"]],
#     elec_colors=vals_clip,
#     elec_size=0.5,
#     cmap="coolwarm",
#     cbar=True,
#     cbar_title="t (Internal vs External)",
#     cbar_minmax=(-v, v),
#     max_dist=2000,  # diagnostic; once confirmed, lower it if you want
# )

# fig = plot_brain_surf(
#     elec_names=elec_names,
#     coords=coords,
#     elec_hem=elec_hem_list,
#     subject="fsaverage",
#     subjects_dir=recon_dir,
#     surf="pial",                 # <- key change: avoids forced snap_to_surf
#     views=[["l_lateral","l_medial"],["r_lateral","r_medial"]],
#     elec_colors=vals,
#     elec_size=0.6,               # temporarily larger to confirm visibility
#     cmap="coolwarm",
#     cbar=True,
#     cbar_title="t (int-ext)",
#     cbar_minmax=(0, np.nanpercentile(vals, 95)),
#     max_dist=200
# )

#%%

# given by you
if atlas == 'Y17_Atlas_Region':
    network_mapping = {
        "17Networks_1": "Visual Central (Visual A)",
        "17Networks_2": "Visual Peripheral (Visual B)",
        "17Networks_3": "Somatomotor A",
        "17Networks_4": "Somatomotor B",
        "17Networks_5": "Dorsal Attention A",
        "17Networks_6": "Dorsal Attention B",
        "17Networks_7": "Salience / Ventral Attention A",
        "17Networks_8": "Salience / Ventral Attention B",
        "17Networks_9": "Limbic A",
        "17Networks_10": "Limbic B",
        "17Networks_11": "Control C",
        "17Networks_12": "Control A",
        "17Networks_13": "Control B",
        "17Networks_14": "Temporal Parietal",
        "17Networks_15": "Default C",
        "17Networks_16": "Default A",
        "17Networks_17": "Default B"
    }
    
    name_to_17 = {v: k for k, v in network_mapping.items()}

if atlas == 'Y7_Atlas_Region':
    network_mapping = {
     "7Networks_1": "Visual", 
     "7Networks_2": "Somatomotor", 
     "7Networks_3": "Dorsal Attention", 
     "7Networks_4": "Ventral Attention", 
     "7Networks_5": "Limbic", 
     "7Networks_6": "Frontoparietal", 
     "7Networks_7": "Default"}
    
    name_to_7 = {v: k for k, v in network_mapping.items()}


# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors

# # Filter to the slice you want to visualize
# df = electrodes_df[
#     (electrodes_df["movie"] == "despicable_me_english") &
#     (electrodes_df["freq_band"] == "HFA") &
#     (electrodes_df["atlas"] == "Y17_Atlas_Region")
# ].copy()

# # Optional: drop medial wall or any non-parcel labels you don't want colored
# excluded = {"FreeSurfer_Defined_Medial_Wall"}
# df = df[~df["region"].isin(excluded)]

# # Aggregate electrode-level values to one value per parcel
# # Choose one:
# #   value = mean t-statistic
# #   value = mean delta
# #   value = mean abs(t) etc.
# parcel_df = (
#     df.groupby("region", as_index=False)
#       .agg(
#           value=("t", "mean"),
#           n_electrodes=("t", "size"),
#           n_recordings=("patient", "nunique"),
#       )
# )

# print(parcel_df.sort_values("value").head())
# print(parcel_df.sort_values("value", ascending=False).head())


# vals = parcel_df["value"].to_numpy(float)

# Robust symmetric scaling
# v = float(np.nanpercentile(np.abs(vals), 95))
# v = v if np.isfinite(v) and v > 0 else float(np.nanmax(np.abs(vals)))

# norm = mcolors.Normalize(vmin=-v, vmax=v)
# cmap = plt.get_cmap("coolwarm")
# rgba = cmap(norm(vals))  # shape (n_parcels, 4)

# parcs_to_show = parcel_df["region"].tolist()
# parc_colors = [tuple(c) for c in rgba]  # plot_brain_surf accepts matplotlib colors

# fig = plot_brain_surf(
#     elec_names=[],                 # no electrodes
#     coords=np.empty((0, 3)),        # no electrodes
#     elec_hem=[],                    # no electrodes
#     subject="fsaverage",
#     subjects_dir=recon_dir,
#     surf="inflated",
#     views=[["l_lateral", "l_medial"], ["r_lateral", "r_medial"]],
#     parc="y17",                     # IMPORTANT: must match ATLASES key used in that codebase
#     parcs_to_show=parcs_to_show,
#     parc_colors=parc_colors,
#     parc_alpha=1.0,
#     parc_borders=False,
#     clear_overlay=True,
#     title="Inscapes HFA: parcel mean t (Internal vs External)",
#     annotations=True
# )
#%%
movie = movie
freq_band = freq_band
atlas = atlas
value_col = "delta"

d = electrodes_df[
    (electrodes_df["movie"] == movie) &
    (electrodes_df["freq_band"] == freq_band) &
    (electrodes_df["atlas"] == atlas)
].copy()

# optionally restrict to electrodes that have enough windows
min_obs = 10
d = d[(d["n_int"] >= min_obs) & (d["n_ext"] >= min_obs)]

# aggregate per your human-readable network name (your existing region labels)
agg = (
    d.groupby("region", as_index=False)[value_col]
     .mean()
     .rename(columns={value_col: "val"})
)

if atlas == 'Y17_Atlas_Region':
# map to 17Networks_k IDs
    agg["parc_id"] = agg["region"].map(name_to_17)

if atlas == 'Y7_Atlas_Region':
# map to 17Networks_k IDs
    agg["parc_id"] = agg["region"].map(name_to_7)

# keep only networks that successfully mapped
agg = agg.dropna(subset=["parc_id"]).copy()

# optional: enforce minimum electrodes per network
min_elecs_per_net = 5
counts = d.groupby("region").size().rename("n_elecs").reset_index()
agg = agg.merge(counts, on="region", how="left")
agg = agg[agg["n_elecs"] >= min_elecs_per_net]
#%%

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

vals = agg["val"].to_numpy(float)

# symmetric scaling (robust)
v = np.nanpercentile(np.abs(vals), 95)
v = float(v) if np.isfinite(v) and v > 0 else float(np.nanmax(np.abs(vals)))
norm = mcolors.TwoSlopeNorm(vmin=-v, vcenter=0.0, vmax=v)

cmap = plt.get_cmap("coolwarm")

# plot_brain_surf expects actual colors per parcel (strings or rgba)
# safest: give RGBA tuples
agg["rgba"] = [cmap(norm(x)) for x in vals]

parcs_to_show = agg["parc_id"].tolist()
parc_colors   = agg["rgba"].tolist()

elec_names, coords, elec_hem, elec_vals = vis_df_to_plot_brain_inputs(
    vis_df, movie=movie, freq_band=freq_band, atlas=atlas, value_col=value_col, abs_val=False
)

if atlas == 'Y17_Atlas_Region':
    parc_key = "y17"
if atlas == 'Y7_Atlas_Region':
    parc_key = 'y7'

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
    parc=parc_key,
    parcs_to_show=parcs_to_show,
    parc_colors=parc_colors,
    parc_alpha=1,        # translucent parcels so electrodes remain visible
    parc_borders=False,
    clear_overlay=True,
    title=f"{movie} {freq_band} {threshold} {PC_type} {atlas}: parcels + electrodes",
)

out_path = os.path.join(
    img_dir,
    f"{movie}_{freq_band}_{atlas}_{value_col}_{threshold}_{PC_type}_parcels_plus_electrodes.png"
)

fig.savefig(out_path, dpi=300, bbox_inches="tight")
print("Saved:", out_path)
