#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 20:00:43 2026

@author: christinechesebrough
"""

        
freq_bands = ['delta']
movies     = ['inscapes','despicable_me_english']#inscapes

# =============================================================================
# ENTRY-ID TABLES  (add/remove entries here)
# =============================================================================

MOVIE_ENTRIES = {
    'despicable_me_english': [
        'NS127_02_ses-02_run-01',
        'NS135_ses-01_run-01',
        'NS136_ses-01_run-01',
        'NS137_ses-01_run-01',
        'NS138_ses-01_run-01',
        'NS140_ses-01_run-01',
        'NS153_ses-01_run-01',
        'NS155_02_ses-02_run-01',
        'NS164_ses-01_run-01',
        'NS174_02_ses-02_run-01',
        'NS174_03_ses-03_run-01',
        'NS178_ses-01_run-01',
        'NS190_ses-01_run-01',
        'NS190_ses-01_run-02',
        'NS191_ses-01_run-01',
        'NS193_ses-01_run-01',
        'NS193_ses-01_run-02',
        'NS194_ses-01_run-01',
        'NS205_ses-01_run-01',
    ],
    'inscapes': [
        'NS127_02_ses-02_run-01',
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
        'NS178_ses-01_run-01',
        'NS205_ses-01_run-01',
        'NS210_ses-01_run-01',
    ],
}



import os
import re
import pandas as pd
import numpy as np

machine_path = 'Volumes'

freq_bands = ['delta','theta','alpha','beta','gamma','HFA']
movies = ['despicable_me_english','inscapes']

corr_dir = f'/{machine_path}/Samsung/Movie_data/data/movie_elec_corr_sheets'
method = 'power_log'
z_val = ''
WINDOW_SEC = 10  # set this if not already defined


def extract_pat_id(text: str) -> str:
    m = re.search(r'(NS\d+(?:_\d+)?)', text)
    if not m:
        raise ValueError(f"Could not extract patient ID from: {text}")
    return m.group(1)


def extract_run_label(text: str) -> str:
    m = re.search(r'run[-_]?(\d+)', text, flags=re.IGNORECASE)
    return f"run-{int(m.group(1)):02d}" if m else None


def extract_sub_label(text: str) -> str:
    """
    Extract sub-NS127 or sub-NS127_02 if present.
    """
    m = re.search(r'(sub-NS\d+(?:_\d+)?)', text, flags=re.IGNORECASE)
    return m.group(1) if m else None


def standardize_corr_columns(df):
    """
    Make column names case-insensitive and normalize likely variants.
    """
    rename_map = {}
    for col in df.columns:
        c = col.strip().lower()

        if c == 'label':
            rename_map[col] = 'label'
        elif c in {'aparc_aseg', 'aparc-aseg', 'aparc aseg', 'aparc_aseg_atlas'}:
            rename_map[col] = 'aparc_aseg'

    df = df.rename(columns=rename_map)
    return df


for freq_band in freq_bands:
    for movie in movies:

        lfp_dir = (
            f'/{machine_path}/Samsung/Movie_data/windowed_power_10s/'
            f'windowed_unnormed_{method}_{movie}_{freq_band}_1Apr26'
        )

        if not os.path.isdir(lfp_dir):
            print(f"[SKIP] lfp_dir not found: {lfp_dir}")
            continue

        # ------------------------------------------------------------------
        # Loop through all patient subfolders in lfp_dir
        # ------------------------------------------------------------------
        for pat in sorted(os.listdir(lfp_dir)):
            pat_dir = os.path.join(lfp_dir, pat)
            if not os.path.isdir(pat_dir) or pat.startswith('.'):
                continue

            csv_files = sorted([
                f for f in os.listdir(pat_dir)
                if f.endswith('.csv') and not f.startswith('._')
            ])

            if not csv_files:
                print(f"[SKIP] No CSVs in {pat_dir}")
                continue

            # ------------------------------------------------------------------
            # Load correspondence sheet once per patient
            # ------------------------------------------------------------------
            excel_files = sorted([
                f for f in os.listdir(corr_dir)
                if pat in f and f.endswith('.xlsx') and not f.startswith('.')
            ])

            if not excel_files:
                print(f"[SKIP] No correspondence sheet found for {pat}")
                continue

            excel_path = max(
                [os.path.join(corr_dir, f) for f in excel_files],
                key=os.path.getmtime
            )
            print(f"\nUsing corr sheet for {pat}: {os.path.basename(excel_path)}")

            try:
                elecs_subs = pd.read_excel(excel_path)
            except Exception as e:
                print(f"[SKIP] Could not read corr sheet for {pat}: {e}")
                continue

            elecs_subs = standardize_corr_columns(elecs_subs)

            required = {'label', 'aparc_aseg','Desikan_Killiany'}
            missing = required - set(elecs_subs.columns)
            if missing:
                print(f"[SKIP] Missing required cols in {excel_path}: {missing}")
                continue

            # normalize label strings for matching
            elecs_subs['label'] = elecs_subs['label'].astype(str).str.strip()
            elecs_subs['aparc_aseg'] = elecs_subs['aparc_aseg'].astype(str).str.strip()

            aparc_lookup = dict(zip(elecs_subs['label'], elecs_subs['aparc_aseg']))

            # ------------------------------------------------------------------
            # Process each CSV in this patient folder
            # ------------------------------------------------------------------
            for lfp_file in csv_files:
                full_lfp_path = os.path.join(pat_dir, lfp_file)

                try:
                    pat_from_file = extract_pat_id(lfp_file)
                    run_from_file = extract_run_label(lfp_file)
                    sub_from_file = extract_sub_label(lfp_file)
                except Exception as e:
                    print(f"[SKIP] Could not parse filename {lfp_file}: {e}")
                    continue

                # optional sanity check
                if pat_from_file != pat:
                    print(f"[SKIP] Filename patient mismatch: folder={pat}, file={pat_from_file}")
                    continue

                print(f"  Processing {lfp_file}")
                print(f"    parsed -> pat={pat_from_file}, run={run_from_file}, sub={sub_from_file}")

                try:
                    raw = pd.read_csv(full_lfp_path)
                except Exception as e:
                    print(f"  [ERROR] Reading {lfp_file}: {e}")
                    continue

                if raw.empty:
                    print(f"  [SKIP] Empty CSV: {lfp_file}")
                    continue

                # ------------------------------------------------------------------
                # Find the first column containing row labels / atlas names
                # ------------------------------------------------------------------
                atlas_col = 'Atlas'
        
                atlas_row_mask = raw[atlas_col].astype(str).str.strip().eq('AparcAseg_Atlas_Region')
                
                if atlas_row_mask.sum() == 0:
                    print(f"  [SKIP] No 'Aparc_Aseg_Atlas' row found in {lfp_file}")
                    continue

                atlas_row_idx = raw.index[atlas_row_mask][0]

                # Electrode columns: everything except the first label column
                electrode_cols = [
                    col for col in raw.columns
                    if col and col[0] in ('L', 'R') and col[1:].isalnum()
                ]
                n_updated = 0
                n_missing = 0

                for elec in electrode_cols:
                    elec_clean = str(elec).strip()

                    if elec_clean in aparc_lookup:
                        raw.at[atlas_row_idx, elec] = aparc_lookup[elec_clean]
                        n_updated += 1
                    else:
                        n_missing += 1
                        # optional: leave existing value unchanged
                        # raw.at[atlas_row_idx, elec] = np.nan

                try:
                    raw.to_csv(full_lfp_path, index=False)
                    print(f"    saved: updated {n_updated} electrodes, missing {n_missing}")
                except Exception as e:
                    print(f"  [ERROR] Saving {lfp_file}: {e}")
                    
                    
                    #%%
                    
import os
import pandas as pd

data_dir = '/Volumes/Samsung/Movie_data/data'

all_dfs = []

def standardize_columns(df):
    rename_map = {}
    
    for col in df.columns:
        c = col.strip().lower()
        
        if c == 'label':
            rename_map[col] = 'label'
        elif c in {'aparc_aseg', 'aparc-aseg', 'aparc aseg'}:
            rename_map[col] = 'aparc_aseg'
        elif c in {'desikan_killiany', 'desikan-killiany', 'dk'}:
            rename_map[col] = 'desikan_killiany'
    
    df = df.rename(columns=rename_map)
    return df


for fname in os.listdir(data_dir):
    if not fname.endswith('.xlsx') or fname.startswith('~$'):
        continue
    
    fpath = os.path.join(data_dir, fname)
    print(f"Loading {fname}")
    
    try:
        df = pd.read_excel(fpath)
    except Exception as e:
        print(f"  [SKIP] Could not read {fname}: {e}")
        continue
    
    df = standardize_columns(df)
    
    # --- extract patient ID from filename ---
    pat = fname.split('.')[0]
    df['patient'] = pat
    
    # --- clean label ---
    if 'label' in df.columns:
        df['label'] = df['label'].astype(str).str.strip()
    
    all_dfs.append(df)


# =============================================================================
# COMBINE
# =============================================================================

master = pd.concat(all_dfs, ignore_index=True)

print("\nCombined shape:", master.shape)

master = master.drop_duplicates(subset=['patient', 'label'])

print("After dedup:", master.shape)


#%%

def get_unique_clean(series):
    return (
        series
        .astype(str)
        .str.strip()
        .replace(['', 'nan', 'None'], pd.NA)
        .dropna()
        .unique()
    )

aparc_vals = get_unique_clean(corr_master['aparcaseg_atlas'])
dk_vals    = get_unique_clean(corr_master['dk_atlas'])

print(f"\nAparc+aseg ({len(aparc_vals)} unique):")
print(sorted(aparc_vals))

print(f"\nDesikan-Killiany ({len(dk_vals)} unique):")
print(sorted(dk_vals))

pd.Series(sorted(aparc_vals)).to_csv('unique_aparc_aseg_labels.csv', index=False)
pd.Series(sorted(dk_vals)).to_csv('unique_desikan_labels.csv', index=False)
             

        

#%%

corr_dir = '/Volumes/Samsung/Movie_data/data/movie_elec_corr_sheets'

#for every .xlsx file in corr_dir
#open it, and run map_network to produce a new 'network' column, then save as is. 

def map_network(dk, aparc):
    dk = str(dk).lower()
    aparc = str(aparc)

    # -------------------------
    # EXCLUDE non-gray matter
    # -------------------------
    if any(x in aparc for x in [
        'White-Matter', 'Ventricle', 'CSF', 'choroid', 'Unknown', 'WM-hypointensities'
    ]):
        return 'EXCLUDE'

    # -------------------------
    # DMN (MTL first)
    # -------------------------
    if aparc in [
        'Left-Hippocampus', 'Right-Hippocampus',
        'Left-Amygdala', 'Right-Amygdala'
    ]:
        return 'DMN'

    if dk in {
        'medialorbitofrontal',
        'rostralanteriorcingulate',
        'posteriorcingulate',
        'isthmuscingulate',
        'precuneus',
        'inferiorparietal',
        'middletemporal',
        'temporalpole',
        'parahippocampal',
        'entorhinal',
        'frontalpole'
    }:
        return 'DMN'

    # -------------------------
    # FPCN-A
    # -------------------------
    if dk in {
        'rostralmiddlefrontal',
        'superiorfrontal',
        'parsopercularis',
        'parsorbitalis',
        'parstriangularis'
    }:
        return 'FPCN_A'

    # -------------------------
    # FPCN-B
    # -------------------------
    if dk in {
        'caudalmiddlefrontal',
        'precentral',
        'postcentral',
        'superiorparietal',
        'supramarginal'
    }:
        return 'FPCN_B'

    return 'OTHER'


corr_master['network'] = corr_master.apply(
    lambda row: map_network(row['dk_atlas'], row['aparcaseg_atlas']),
    axis=1
)

corr_master['network'].value_counts()

#%%
import os
import pandas as pd

corr_dir = '/Volumes/Samsung/Movie_data/data/movie_elec_corr_sheets'


def standardize_columns(df):
    rename_map = {}

    for col in df.columns:
        c = str(col).strip().lower()

        if c == 'label':
            rename_map[col] = 'label'
        elif c in {'aparc_aseg', 'aparc-aseg', 'aparc aseg', 'aparc_aseg_atlas'}:
            rename_map[col] = 'aparc_aseg'
        elif c in {'desikan_killiany', 'desikan-killiany', 'desikan killiany', 'dk'}:
            rename_map[col] = 'desikan_killiany'
        elif c in {'y7_atlas', 'y7 atlas', 'yeo7', 'yeo_7'}:
            rename_map[col] = 'y7_atlas'

    return df.rename(columns=rename_map)


def map_network(dk, aparc):
    dk = str(dk).strip().lower()
    aparc = str(aparc).strip()

    # -------------------------
    # EXCLUDE non-gray matter
    # -------------------------
    if any(x in aparc for x in [
        'White-Matter', 'Ventricle', 'CSF', 'choroid', 'Unknown', 'WM-hypointensities'
    ]):
        return 'EXCLUDE'

    # -------------------------
    # DMN (MTL first)
    # -------------------------
    if aparc in [
        'Left-Hippocampus', 'Right-Hippocampus',
        'Left-Amygdala', 'Right-Amygdala'
    ]:
        return 'DMN'

    if dk in {
        'medialorbitofrontal',
        'rostralanteriorcingulate',
        'posteriorcingulate',
        'isthmuscingulate',
        'precuneus',
        'inferiorparietal',
        'middletemporal',
        'temporalpole',
        'parahippocampal',
        'entorhinal',
        'frontalpole'
    }:
        return 'DMN'

    # -------------------------
    # FPCN-A
    # -------------------------
    if dk in {
        'rostralmiddlefrontal',
        'superiorfrontal',
        'parsopercularis',
        'parsorbitalis',
        'parstriangularis'
    }:
        return 'FPCN_A'

    # -------------------------
    # FPCN-B
    # -------------------------
    if dk in {
        'caudalmiddlefrontal',
        'precentral',
        'postcentral',
        'superiorparietal',
        'supramarginal'
    }:
        return 'FPCN_B'

    return 'OTHER'


excel_files = sorted([
    f for f in os.listdir(corr_dir)
    if f.endswith('.xlsx') and not f.startswith('~$') and not f.startswith('.')
])

for fname in excel_files:
    fpath = os.path.join(corr_dir, fname)
    print(f'\nProcessing: {fname}')

    try:
        df = pd.read_excel(fpath)
    except Exception as e:
        print(f'  [SKIP] Could not read file: {e}')
        continue

    df = standardize_columns(df)

    required = {'aparc_aseg', 'desikan_killiany'}
    missing = required - set(df.columns)
    if missing:
        print(f'  [SKIP] Missing required columns: {missing}')
        continue

    df['aparc_aseg'] = df['aparc_aseg'].astype(str).str.strip()
    df['desikan_killiany'] = df['desikan_killiany'].astype(str).str.strip()

    df['network'] = df.apply(
        lambda row: map_network(row['desikan_killiany'], row['aparc_aseg']),
        axis=1
    )

    try:
        df.to_excel(fpath, index=False)
        print('  saved')
        print(df['network'].value_counts(dropna=False))
    except Exception as e:
        print(f'  [ERROR] Could not save file: {e}')
        
        
        #%%
        
import os
import re
import pandas as pd
import numpy as np

machine_path = 'Volumes'

freq_bands = ['delta']#, 'theta', 'alpha', 'beta', 'gamma', 'HFA']
movies = ['despicable_me_english', 'inscapes']

corr_dir = f'/{machine_path}/Samsung/Movie_data/data/movie_elec_corr_sheets'
method = 'power_log'
z_val = ''
WINDOW_SEC = 10


def extract_pat_id(text: str) -> str:
    m = re.search(r'(NS\d+(?:_\d+)?)', text)
    if not m:
        raise ValueError(f"Could not extract patient ID from: {text}")
    return m.group(1)


def extract_run_label(text: str) -> str:
    m = re.search(r'run[-_]?(\d+)', text, flags=re.IGNORECASE)
    return f"run-{int(m.group(1)):02d}" if m else None


def extract_sub_label(text: str) -> str:
    m = re.search(r'(sub-NS\d+(?:_\d+)?)', text, flags=re.IGNORECASE)
    return m.group(1) if m else None


def standardize_corr_columns(df):
    rename_map = {}
    for col in df.columns:
        c = str(col).strip().lower()

        if c == 'label':
            rename_map[col] = 'label'
        elif c in {'aparc_aseg', 'aparc-aseg', 'aparc aseg', 'aparc_aseg_atlas'}:
            rename_map[col] = 'aparc_aseg'
        elif c in {'desikan_killiany', 'desikan-killiany', 'desikan killiany', 'dk'}:
            rename_map[col] = 'desikan_killiany'

    return df.rename(columns=rename_map)


def map_network(dk, aparc):
    dk = str(dk).strip().lower()
    aparc = str(aparc).strip()

    # -------------------------
    # EXCLUDE non-gray matter
    # -------------------------
    if any(x in aparc for x in [
        'White-Matter', 'Ventricle', 'CSF', 'choroid', 'Unknown', 'WM-hypointensities'
    ]):
        return 'EXCLUDE'

    # -------------------------
    # DMN (MTL first)
    # -------------------------
    if aparc in [
        'Left-Hippocampus', 'Right-Hippocampus',
        'Left-Amygdala', 'Right-Amygdala'
    ]:
        return 'DMN'

    if dk in {
        'medialorbitofrontal',
        'rostralanteriorcingulate',
        'posteriorcingulate',
        'isthmuscingulate',
        'precuneus',
        'inferiorparietal',
        'middletemporal',
        'temporalpole',
        'parahippocampal',
        'entorhinal',
        'frontalpole'
    }:
        return 'DMN'

    # -------------------------
    # FPCN-A
    # -------------------------
    if dk in {
        'rostralmiddlefrontal',
        'superiorfrontal',
        'parsopercularis',
        'parsorbitalis',
        'parstriangularis'
    }:
        return 'FPCN_A'

    # -------------------------
    # FPCN-B
    # -------------------------
    if dk in {
        'caudalmiddlefrontal',
        'precentral',
        'postcentral',
        'superiorparietal',
        'supramarginal'
    }:
        return 'FPCN_B'

    return 'OTHER'


for freq_band in freq_bands:
    for movie in movies:

        lfp_dir = (
            f'/{machine_path}/Samsung/Movie_data/windowed_power_10s/'
            f'windowed_unnormed_{method}_{movie}_{freq_band}_1Apr26'
        )

        if not os.path.isdir(lfp_dir):
            print(f"[SKIP] lfp_dir not found: {lfp_dir}")
            continue

        for pat in sorted(os.listdir(lfp_dir)):
            pat_dir = os.path.join(lfp_dir, pat)
            if not os.path.isdir(pat_dir) or pat.startswith('.'):
                continue

            csv_files = sorted([
                f for f in os.listdir(pat_dir)
                if f.endswith('.csv') and not f.startswith('._')
            ])

            if not csv_files:
                print(f"[SKIP] No CSVs in {pat_dir}")
                continue

            excel_files = sorted([
                f for f in os.listdir(corr_dir)
                if pat in f and f.endswith('.xlsx') and not f.startswith('.')
            ])

            if not excel_files:
                print(f"[SKIP] No correspondence sheet found for {pat}")
                continue

            excel_path = max(
                [os.path.join(corr_dir, f) for f in excel_files],
                key=os.path.getmtime
            )
            print(f"\nUsing corr sheet for {pat}: {os.path.basename(excel_path)}")

            try:
                elecs_subs = pd.read_excel(excel_path)
            except Exception as e:
                print(f"[SKIP] Could not read corr sheet for {pat}: {e}")
                continue

            elecs_subs = standardize_corr_columns(elecs_subs)

            required = {'label', 'aparc_aseg', 'desikan_killiany'}
            missing = required - set(elecs_subs.columns)
            if missing:
                print(f"[SKIP] Missing required cols in {excel_path}: {missing}")
                continue

            elecs_subs['label'] = elecs_subs['label'].astype(str).str.strip()
            elecs_subs['aparc_aseg'] = elecs_subs['aparc_aseg'].astype(str).str.strip()
            elecs_subs['desikan_killiany'] = elecs_subs['desikan_killiany'].astype(str).str.strip()

            network_lookup = {}
            for _, row in elecs_subs.iterrows():
                elec = row['label']
                network_lookup[elec] = map_network(
                    row['desikan_killiany'],
                    row['aparc_aseg']
                )

            for lfp_file in csv_files:
                full_lfp_path = os.path.join(pat_dir, lfp_file)

                try:
                    pat_from_file = extract_pat_id(lfp_file)
                    run_from_file = extract_run_label(lfp_file)
                    sub_from_file = extract_sub_label(lfp_file)
                except Exception as e:
                    print(f"[SKIP] Could not parse filename {lfp_file}: {e}")
                    continue

                if pat_from_file != pat:
                    print(f"[SKIP] Filename patient mismatch: folder={pat}, file={pat_from_file}")
                    continue

                print(f"  Processing {lfp_file}")
                print(f"    parsed -> pat={pat_from_file}, run={run_from_file}, sub={sub_from_file}")

                try:
                    raw = pd.read_csv(full_lfp_path)
                except Exception as e:
                    print(f"  [ERROR] Reading {lfp_file}: {e}")
                    continue

                if raw.empty:
                    print(f"  [SKIP] Empty CSV: {lfp_file}")
                    continue

                atlas_col = 'Atlas'
                if atlas_col not in raw.columns:
                    print(f"  [SKIP] No '{atlas_col}' column in {lfp_file}")
                    continue

                electrode_cols = [
                    col for col in raw.columns
                    if isinstance(col, str) and len(col) > 1 and col[0] in ('L', 'R') and col[1:].isalnum()
                ]

                # remove an existing network row if already present
                existing_network_mask = raw[atlas_col].astype(str).str.strip().str.lower().eq('network')
                if existing_network_mask.any():
                    raw = raw.loc[~existing_network_mask].copy()

                # build new network row
                network_row = {col: np.nan for col in raw.columns}
                network_row[atlas_col] = 'network'

                n_assigned = 0
                n_missing = 0

                for elec in electrode_cols:
                    elec_clean = str(elec).strip()
                    if elec_clean in network_lookup:
                        network_row[elec] = network_lookup[elec_clean]
                        n_assigned += 1
                    else:
                        n_missing += 1

                # insert after AparcAseg_Atlas_Region if present, otherwise append
                aparc_mask = raw[atlas_col].astype(str).str.strip().eq('AparcAseg_Atlas_Region')

                if aparc_mask.any():
                    insert_idx = raw.index[aparc_mask][0] + 1
                    top = raw.iloc[:insert_idx].copy()
                    bottom = raw.iloc[insert_idx:].copy()
                    raw = pd.concat([top, pd.DataFrame([network_row]), bottom], ignore_index=True)
                else:
                    raw = pd.concat([raw, pd.DataFrame([network_row])], ignore_index=True)

                try:
                    raw.to_csv(full_lfp_path, index=False)
                    print(f"    saved: network row added, assigned {n_assigned} electrodes, missing {n_missing}")
                except Exception as e:
                    print(f"  [ERROR] Saving {lfp_file}: {e}")