#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract band-limited power, ONE RECORDING OPENED ONCE (TODO A12).

Restructured from extract_power_fc.py. Identical processing; only the loop
order differs.

    BEFORE   for vid > for freq_band > for pat > for f in lfp_files
             -> each .fif is opened and get_data() called ONCE PER BAND.
                With 6 bands that is 6x the file reads and 6x the memory churn.

    AFTER    for vid > for pat > for f in lfp_files > for freq_band
             -> each .fif is opened once; all bands are extracted from the
                already-loaded array.

What moved:
    - the freq_band loop and its freq_range/bin_width block moved inside the
      file loop, after the data is loaded and re-referenced
    - fig_dir and fig_patient_dir moved with it (they embed freq_band)
    - `if len(ip_contacts) == 0: pass / else: <everything>` was flattened to
      `if len(ip_contacts) == 0: continue`. That guard was purely structural,
      but it meant band-specific code lived at two different indentation
      levels - part inside the else, part after it - so the band loop could not
      wrap it without flattening first.

What did NOT change:
    Any computation. Bandpass filtering, Hilbert, the log transform, the
    600->300 Hz decimation, rounding, atlas rows and the CSV write are byte
    identical to extract_power_fc.py.

VALIDATE BEFORE TRUSTING:
    Run one recording through both scripts and diff the CSVs. They should be
    identical. This was a mechanical restructure of a 1200-line script and has
    only been checked structurally, not numerically.
"""

"""
Created on Tue May 20 07:06:08 2025

@author: christinechesebrough
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Updated December 16th
Only includes neural data (not eye features, computed elsewhere)
Added finding of peak freqs using FOOOF

Correlate eye movement based ISC measures with neural signals suspected to
index attentional state changes 
"""
import os, re, sys
from scipy.stats import pearsonr
from itertools import compress
import numpy as np
import pandas as pd
from scipy import stats, signal, interpolate
import matplotlib.pyplot as plt
import mne
import seaborn as sns
from mne.time_frequency import psd_array_welch
import multiprocessing

from scipy.signal import decimate
from scipy.signal import correlate

from joblib import Parallel, delayed
from fooof import FOOOF
from fooof.plts.spectra import plot_spectrum

machine_path = 'media/christine'#'Volumes' #'media/christine'


#from antropy import sample_entropy, spectral_entropy, perm_entropy, lziv_complexity

# Add Linux library paths BEFORE importing epipe
sys.path.insert(0, f'/{machine_path}/Samsung/EPIPE/Python')
sys.path.insert(0, f'/{machine_path}/Samsung/iEEG2NWB-main')

#vids = ['inscapes','despicable_me_english']#,'despicable_me_english']
vids = ['despicable_me_hungarian','inscapes','despicable_me_english']#,'despicable_me_english']
freq_bands = ['theta','alpha','beta','gamma','HFA']#['delta','theta','alpha','gamma','HFA']#'beta','gamma','HFA'] #'delta','theta','alpha','beta','gamma'
#freq_bands = ['theta_alpha','all_gamma']

ref = 'avg'
#freq_band = 'HFA'
region = 'all'


data_dir = f'/{machine_path}/Samsung/Movie_data/movies_prep_standard'      #new_dmh_prep_standard'
isc_dir = f'/{machine_path}/SamsungMovie_data/data/isc'
mne_data_dir = f'/{machine_path}/Samsung/Movie_data/movies_prep_standard'   # new_dmh_prep_standard'
elec_dir = f'/{machine_path}/Samsung/Movie_data/data/electrode_localization'
fs_dir = f'/{machine_path}/Samsung/anatomy'

fooof_dir = '/Volumes/Samsung/Movie_data/fooof_theta_peak_lookup_all.csv'

corr_dir = f'/{machine_path}/Samsung/Movie_data/data/movie_elec_corr_sheets'

fs_eye = 300

visualize_mne_steps = False
condense_to_isc = False
rolling_average = False
lowpass = False
find_peaks =False
plot_power = False
plot_power_subsets = False
use_interpolation = False
window_compare = False
extract_power =True
output = 'power'
pow_type = 'log'

entropy_type = 'mssd'


wd = '/Volumes/Samsung/scripts/movies_ET_attn_main'
src_dir = os.path.join(wd, 'src')
src_dir = os.path.abspath(src_dir)

# Add src to path if not already there
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)


# 1) Put src on sys.path (at the front)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
print('on sys.path?', src_dir in sys.path)


from epipe import nwb2mne, inspectNwb
from epipe import inspectNwb, nwb2mne, read_ielvis
import sys

sys.path.insert(0, f'/{machine_path}/Samsung/scripts/movies_ET_attn_main/src')
# Import EEG preprocessing helper functions
from eeg_preproc_helpers import (
    plot_power_spectra,  plot_psd_batched, 
    create_file_paths, apply_highpass_filter,
    save_bad_channels, load_bad_channels, check_bad_channels_integrity, load_preprocessed_data, validate_mne_structure, preserve_mne_structure, restore_mne_structure, load_data_with_bad_channels, summarize_bad_channels, 
    interpolate_spikes, make_groups_from_prefix, regress_out_noise_by_group,
    detect_spikes_all_channels, reref_avg_by_group,
    ProcessingLogger, detect_spikes_ref1)

###### EXTRACT NORMALIZED AND NON-NORMALIZED VALUES FOR EACH EYE MOVEMENT FOR EACH WINDOW?????
#### AND TRY TO FIGURE OUT WHICH EYE MOVEMENT FEATURES ARE THE BEST PREDICTORS / FIT

#%%
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

def _extract_run_label(fname: str) -> str:
    """
    Try to extract a run label like 'run-01' or 'run-1' from a filename.
    Fallback: 'run-01'.
    """
    m = re.search(r'run[-_]?(\d+)', fname, flags=re.IGNORECASE)
    if m:
        return f"run-{int(m.group(1)):02d}"
    return "run-01"

def _extract_ses_label(fname: str) -> str:
    """
    Optional: extract 'ses-02' etc. Fallback: ''.
    """
    m = re.search(r'ses[-_]?(\d+)', fname, flags=re.IGNORECASE)
    if m:
        return f"ses-{int(m.group(1)):02d}"
    return ""

def _sort_key(f):
    ses = _extract_ses_label(f)
    run = _extract_run_label(f)
    return (ses, run, f)


def atlas_labels_for_channels(corr, pat, labels_ip, atlas_col="DK_Atlas"):
    atlas_df = corr.rename(columns={"SubID": "pat_base", "Contact": "electrode"}).copy()
    atlas_df["pat_base"] = atlas_df["pat_base"].astype(str).str.strip()
    atlas_df["electrode"] = atlas_df["electrode"].astype(str).str.strip()

    lab = pd.DataFrame({"electrode": [str(x).strip() for x in labels_ip]})
    lab["pat_base"] = str(pat).strip()

    lab = lab.merge(
        atlas_df[["pat_base", "electrode", "Hem", "DK_Atlas", "DK_Lobe", "Y7_Atlas", "Y17_Atlas", "AparcAseg_Atlas"]],
        on=["pat_base", "electrode"],
        how="left"
    )

    if atlas_col not in lab.columns:
        raise ValueError(f"atlas_col='{atlas_col}' not found. Available: {list(lab.columns)}")

    lab["atlas_label"] = lab[atlas_col].astype("string").fillna("Unknown").str.strip()

    # optional hemi standardization
    lab["hem"] = lab["Hem"].astype(str).str.upper().str.strip()
    lab.loc[lab["hem"].isin(["LEFT","LH"]), "hem"] = "L"
    lab.loc[lab["hem"].isin(["RIGHT","RH"]), "hem"] = "R"
    lab["hem"] = lab["hem"].where(lab["hem"].isin(["L","R"]), "")

    return lab

def reorder_pow_by_atlas(pow_plot_win, labels_ip, lab_df, *, sort_cols=("atlas_label", "hem", "electrode")):
    lab_df = lab_df.sort_values(list(sort_cols)).reset_index(drop=True)
    idx = [labels_ip.index(e) for e in lab_df["electrode"].tolist()]
    pow_re = pow_plot_win[idx, :]
    ylabels = (lab_df["atlas_label"] + " | " + lab_df["electrode"]).tolist()
    return pow_re, ylabels, lab_df


#%% Main script

vids.sort()
freq_bands.sort()

processed_lfp_files = []  # initialize once


for vid in vids:
    
    if vid == 'despicable_me_english':
        #good_ET_pats = ["NS190"]#['NS140_02','NS153','NS164','NS166','NS174_02']#['NS127_02','NS135','NS136','NS137','NS138','NS140','NS140_02'] #'NS153','NS164','NS166','NS174_02',"NS178","NS190","NS191"]#,'NS193',"NS194","NS201",'NS205']
       # good_ET_pats = ['NS193',"NS194","NS201",'NS205']
       # good_ET_pats = ['NS127_02','NS135','NS136','NS137','NS138','NS140','NS151','NS153','NS154','NS155_02','NS164','NS174_02','NS174_03','NS190','NS191','NS193','NS194','NS178','NS201_02','NS204','NS205']
      # good_ET_pats = ['NS193'] 
       good_ET_pats = [
         # 'NS127_02',
         # 'NS135',
         # 'NS136',
         # 'NS137',
         'NS138',
         'NS140',
         'NS151',
         'NS153',
         'NS154',
         'NS155_02',
         'NS164',
         'NS174_02',
         'NS174_03',
         'NS178',
         'NS190',
         'NS191',
         'NS193',
         'NS194',
         'NS201_02',
         'NS204',
         'NS205'
         ]
       
    if vid == 'the_present':
        patients = [
         # 'NS135',
         'NS140',
         'NS137',
         'NS144_02',
         'NS149',
         'NS153',
         'NS154',
         'NS155',
         'NS155_02',
         'NS164',
         'NS174_03',
         'NS178',
         'NS190',
         'NS192',
         'NS193',
         'NS208',
         'NS210']
         #['NS174_03','NS210']

         
    elif vid == 'inscapes':
        good_ET_pats = [
         'NS127_02',
         'NS135',
         'NS136',
         'NS137',
         'NS138',
         'NS140',
         'NS140_02',
         'NS151',
         'NS153',
         'NS155',
         'NS155_02',
         'NS164',
         'NS178',
         'NS205',
         'NS210']
    
    elif vid == 'despicable_me_hungarian':
        good_ET_pats = [
            #'LH010',
        'NS127_02',
        'NS128_02',
        'NS135',
        'NS136',
        'NS137',
       'NS138',
       'NS140',
       'NS140_02',
        #'NS144',
        #'NS145',
        #'NS151',
        'NS145',
      #  'NS153',
         'NS154',
         'NS164',
        #'NS166',
         'NS167',
         'NS174_02',
         'NS174_03',
         'NS178'
        ]
        
    


    patients = good_ET_pats
    
    patients.sort()
        
    if vid == 'despicable_me_english':
        keys = ['despicable_me_english','dme']
    if vid == 'despicable_me_hungarian':
        keys = ['despicable_me_hungarian','dmh']
    if vid == 'inscapes':
        keys = ['inscapes']
    if vid == 'the_present':
        keys = ['present','the_present']


    
    
    freq_band_count = 0
    
    for pat in patients:
        pat_dir = os.path.join(data_dir, pat)

        lfp_pat_dir = '{:s}/{:s}/Neural_prep'.format(mne_data_dir, pat)

        lfp_files = os.listdir(lfp_pat_dir)


        lfp_files = [
            f for f in os.listdir(lfp_pat_dir)
            if f.endswith(".fif")
            and ref in f
            and ('referenced' in f)
            and 'aic' not in f
            and any(k.lower() in f.lower() for k in keys)
            and not f.startswith("._")
        ]

        # Deterministic sort: by session then run then filename

        lfp_files = sorted(lfp_files, key=_sort_key)

        print(f"Found {len(lfp_files)} matching runs for {pat}:")
        for f in lfp_files:
            print("  ", f)

            processed_lfp_files.append(f)

        # # Iterate over each file/run as an independent entry
        for lfp_file in lfp_files:
            # Build an entry label that will propagate to outputs
            ses_label = _extract_ses_label(lfp_file)
            run_label = _extract_run_label(lfp_file)
            if ses_label:
                entry_id = f"{pat}_{ses_label}_{run_label}"
            else:
                entry_id = f"{pat}_{run_label}"

            print(f"Loading data for entry {entry_id} from {lfp_file} ...")

            if run_label == 'run-01':
                run_keys = ['run-01','run-1']
            if run_label == 'run-02':
                run_keys = ['run-02','run-2']

            mne_data = mne.io.read_raw(os.path.join(lfp_pat_dir, lfp_file), preload=False)

            sub_fs_dir = '{:s}/{:s}'.format(fs_dir, pat)

            subid = os.path.basename(sub_fs_dir)

            elec_recon_dir = corr_dir    

            excel_files = sorted([
                f for f in os.listdir(elec_recon_dir)
                if pat in f
                and f.endswith('.xlsx')
                and not f.startswith('.')
            ])
            if not excel_files:
                print(f"  [SKIP] No correspondence .xlsx for {pat} found in {elec_recon_dir}")
                continue

            # Use most recently modified if multiple exist (matches Python HFO script)
            excel_path = max(
                [os.path.join(elec_recon_dir, f) for f in excel_files],
                key=os.path.getmtime
            )
            print(f"  Using: {os.path.basename(excel_path)}")
            elec_ref_table = pd.read_excel(excel_path)

            elecs_subs =pd.read_excel(excel_path)
            required_cols = {"label"}
            col_map = {col.lower(): col for col in elecs_subs.columns}
            elecs_subs.rename(columns={col_map[req]: req for req in required_cols}, inplace=True)


            # Find candidate files already in the patient dir
            bad_channel_files = [
                f for f in os.listdir(lfp_pat_dir)
                if (
                    ('bad_channels' in f)
                    and f.endswith('.txt')
                    and any(k.lower() in f.lower() for k in keys)
                    and not f.startswith('._')
                )
            ]

            # Narrow to candidates matching ref + run
            ref_bad_channel_files = [
                f for f in bad_channel_files
                if (
                    (ref in f)
                    and any(r in f for r in run_keys)
                )
            ]

            # Helper: pick most recent if multiple
            def pick_most_recent(files):
                if not files:
                    return None
                return max(files, key=lambda fn: os.path.getmtime(os.path.join(lfp_pat_dir, fn)))

            picked = pick_most_recent(ref_bad_channel_files)

            # If none found, define a NEW file name we will create
            # Make sure this naming is unique enough for your workflow
            if picked is None:
                # You can include ses_label/run_label if you want; run_keys might be list like ["run-01", "run-1"]
                # Use run_label if you have a single normalized run label available.
                picked = f'{pat}_{ses_label}_{vid}_{run_label}_{ref}_bad_channels.txt'

            bad_channel_path = os.path.join(lfp_pat_dir, picked)

            # Read existing file (if any)
            #    Preserve header/comments, parse channel lines robustly.
                           # Read existing file (if any)
            header_lines = []
            existing_bads = []

            if os.path.exists(bad_channel_path):
                with open(bad_channel_path, "r") as f:
                    for ln in f:
                        s = ln.strip()
                        if not s:
                            continue
                        if s.startswith("#"):
                            header_lines.append(ln.rstrip("\n"))
                            continue
                        existing_bads.append(s)

            # Safely get FIF bads (empty list if none)
            fif_bads = list(mne_data.info.get("bads", []))

            # --- NEW: filter to only channels that exist in this Raw ---
            ch_set = set(mne_data.ch_names)  # or mne_data.info["ch_names"]

            existing_bads_valid = [ch for ch in existing_bads if ch in ch_set]
            existing_bads_missing = [ch for ch in existing_bads if ch not in ch_set]

            # Optional: log missing bads once (useful for debugging)
            if existing_bads_missing:
                print(
                    f"Warning: {len(existing_bads_missing)} bad channels from file are not in this recording and will be ignored. "
                    f"Examples: {existing_bads_missing[:10]}"
                )

            # Merge (dedupe, preserve order) using only valid names
            merged_bads = list(dict.fromkeys(fif_bads + existing_bads_valid))

            # Assign back
            mne_data.info["bads"] = merged_bads

            # Interactive marking

            if visualize_mne_steps:
                mne_data.plot(
                    scalings=dict(seeg=200e-6),
                    n_channels=32,
                    remove_dc=True,
                    show_scrollbars=True,
                    duration=12.0,
                    block=True
                )

            # Save bads after closing plot
            final_bads = list(dict.fromkeys(mne_data.info.get("bads", [])))  # dedupe again

            os.makedirs(lfp_pat_dir, exist_ok=True)

            tmp_path = bad_channel_path + ".tmp"
            with open(tmp_path, "w") as f:
                # If file had no header, write a minimal one
                if not header_lines:
                    f.write(f"# Bad channels for {pat} {vid} {run_label} {ref}\n")
                else:
                    f.write("\n".join(header_lines) + "\n")

                # Write channel names, one per line
                if final_bads:
                    f.write("\n".join(final_bads) + "\n")

            os.replace(tmp_path, bad_channel_path)
            print(f"Bad channels saved to: {bad_channel_path}")

            # Drop them for downstream processing

            if final_bads:
                mne_data.drop_channels(final_bads)

            labels = mne_data.ch_names

            exclude_strings = ['bankssts']

            #ip_contacts = elecs_subs.Contact.values
            ip_contacts = elecs_subs.label.values


           # ip_contacts = elecs_subs.Contact.values[(elecs_subs.AparcAseg_Atlas != 'Right-Cerebral-White-Matter') & (elecs_subs.AparcAseg_Atlas != 'Left-Cerebral-White-Matter')]
            # ip_contacts = elecs_subs.loc[elecs_subs['DK_Lobe'].isin(['Right-Hippocampus', 'Left-Hippocampus']),
            #     'Contact'
            # ].values

            if len(ip_contacts) == 0:
                continue          # no in-patient contacts for this recording

            if ref == 'wm_bip':
                labels_split = [l.split('-') for l in labels]
                idx_ip = np.unique(np.concatenate([np.where([np.sum([ld == ls for ls in lf]) for lf in labels_split])[0] for ld in ip_contacts]))
                idx_ip = np.in1d(np.arange(len(labels)), idx_ip)
            if ref in ['avg','wm']:
                idx_ip = np.array([label in ip_contacts for label in labels])

            lfp = mne_data.get_data()
            fs_lfp = mne_data.info['sfreq']
            fs_lfp_raw = fs_lfp      # A12: preserved; fs_lfp is decimated per band
            time_lfp = mne_data.times

            lfp_ip = lfp[idx_ip, :]

            labels_ip = list(compress(labels, idx_ip))

            info = mne.create_info(ch_names=labels_ip,sfreq = fs_lfp)


            # ---- all bands for THIS recording; .fif already loaded ----
            for freq_band in freq_bands:
                # A12 FIX: the decimation below reduces fs_lfp. The .fif is now
                # loaded ONCE and this loop runs per band, so that reduction would
                # compound (600 -> 300 -> 150 -> 75 ...) and every band after the
                # first would design its bandpass filter against the wrong rate.
                # Reset from the preserved raw value on each iteration.
                fs_lfp = fs_lfp_raw

                if freq_band == 'alpha':
                    freq_range = (8, 13)
                elif freq_band == 'HFA':
                    freq_range = (51, 150)
                elif freq_band == 'delta':
                    freq_range = (1, 3)
                elif freq_band == 'theta':
                    freq_range = (4, 7)
                elif freq_band == 'beta':
                    freq_range = (14, 30)
                elif freq_band == 'gamma':
                    freq_range = (31, 50)
                elif freq_band == 'all_gamma':
                    freq_range = (31,150)
                elif freq_band == 'mid_gamma':
                    freq_range = (50,69)
                elif freq_band == 'theta_alpha':
                    freq_range = (4,13)
                elif freq_band == 'all':
                    freq_range = (0,170)
                else:
                    raise Exception("no range assigned")

                if freq_band == 'alpha':
                    bin_width = 2
                elif freq_band == 'beta':
                    bin_width = 4
                elif freq_band in ['gamma', 'all_gamma','mid_gamma']:
                    bin_width = 8
                elif freq_band == 'HFA':
                    bin_width = 10
                else:
                    bin_width = 2  # default


                print(f"Processing data for {vid} in {freq_band} with range: {freq_range} Hz")
                if output == 'entropy':
                    fig_dir = f'/{machine_path}/Samsung/Movie_data/full_raw_log_power/{freq_band}/{freq_band}_{vid}_all_cortContacts_entropy_29Mar25/entropy_extracted'
                else:
                   fig_dir = f'/{machine_path}/Samsung/Movie_data/full_raw_log_power_rescale/{freq_band}/{freq_band}_{output}_{pow_type}_{freq_band}_{vid}'
                if not os.path.exists(fig_dir):
                    os.makedirs(fig_dir)

                fig_patient_dir = os.path.join(fig_dir, pat)
                if not os.path.exists(fig_patient_dir):
                    os.makedirs(fig_patient_dir)
                f_start, f_end = freq_range
                freq_bins = [(f, min(f + bin_width, f_end)) for f in range(f_start, f_end, bin_width)]

                # Remove flat channels before processing (std < threshold)
                flat_std_thresh = 1e-6
                stds = np.std(lfp_ip, axis=1)
                nonflat_idx = stds > flat_std_thresh
                lfp_ip = lfp_ip[nonflat_idx]
                labels_ip = [label for i, label in enumerate(labels_ip) if nonflat_idx[i]]

                fooof_region = None

                if find_peaks:
                    fooof_fig_dir = f'/{machine_path}/Samsung/Movie_data/fooof_peaks_{freq_band}_{vid}_23Jun26'
                    if not os.path.exists(fooof_fig_dir):
                        os.makedirs(fooof_fig_dir)

                    fooof_fig_patient_dir = os.path.join(fooof_fig_dir, pat)
                    if not os.path.exists(fooof_fig_patient_dir):
                        os.makedirs(fooof_fig_patient_dir)

                    fooof_results = []

                    # The band of interest is already defined earlier in the script
                    peak_search_range = freq_range

                    # Broader fitting ranges for estimating aperiodic background + candidate peaks
                    fooof_fit_ranges = {
                        'delta': (1, 13),
                        'theta': (1, 13),
                        'alpha': (4, 30),
                        'beta':  (4, 40),
                        'gamma': (20, 80),
                        'HFA':   (30, 170),
                        'all_gamma': (20, 170),
                        'mid_gamma': (20, 90),
                        'theta_alpha': (1, 30),
                        'all': (1, 170),
                    }

                    if freq_band not in fooof_fit_ranges:
                        raise ValueError(f"No FOOOF fit range defined for freq_band={freq_band}")

                    if freq_band in ['gamma', 'HFA', 'all_gamma', 'mid_gamma']:
                        peak_width_limits = [2, 20]
                    else:
                        peak_width_limits = [1, 6]

                    fooof_fmin, fooof_fmax = fooof_fit_ranges[freq_band]

                    # Compute PSD over the broader FOOOF fitting range
                    psd_fooof, freqs_fooof = psd_array_welch(
                        lfp_ip,
                        sfreq=fs_lfp,
                        fmin=fooof_fmin,
                        fmax=fooof_fmax,
                        n_fft=int(fs_lfp * 2),
                        n_overlap=int(fs_lfp),
                        average='mean'
                    )

                    for ch_idx, label in enumerate(labels_ip):

                        fm = FOOOF(
                            peak_width_limits=peak_width_limits,
                            max_n_peaks=6,
                            min_peak_height=0.1,
                            aperiodic_mode='fixed',
                            verbose=False
                        )

                        try:
                            fm.fit(freqs_fooof, psd_fooof[ch_idx, :])

                            all_peaks = fm.peak_params_

                            if all_peaks.shape[0] > 0:
                                in_band = (
                                    (all_peaks[:, 0] >= peak_search_range[0]) &
                                    (all_peaks[:, 0] <= peak_search_range[1])
                                )

                                band_peaks = all_peaks[in_band]

                                # if band_peaks.shape[0] > 0:
                                #     # Pick the strongest in-band peak
                                #     best_peak = band_peaks[np.argmax(band_peaks[:, 1])]

                                #     peak_freq = best_peak[0]
                                #     peak_amp = best_peak[1]
                                #     peak_width = best_peak[2]
                                #     found_peak = True
                                #     n_peaks_total = all_peaks.shape[0]
                                #     n_peaks_in_band = band_peaks.shape[0]
                                # else:
                                #     peak_freq = np.nan
                                #     peak_amp = np.nan
                                #     peak_width = np.nan
                                #     found_peak = False
                                #     n_peaks_total = all_peaks.shape[0]
                                #     n_peaks_in_band = 0

                                if band_peaks.shape[0] > 0:
                                    # Keep all in-band peaks
                                    peak_freq = band_peaks[:, 0].tolist()
                                    peak_amp = band_peaks[:, 1].tolist()
                                    peak_width = band_peaks[:, 2].tolist()

                                    found_peak = True
                                    n_peaks_total = all_peaks.shape[0]
                                    n_peaks_in_band = band_peaks.shape[0]
                                else:
                                    peak_freq = []
                                    peak_amp = []
                                    peak_width = []

                                    found_peak = False
                                    n_peaks_total = all_peaks.shape[0]
                                    n_peaks_in_band = 0

                            else:
                                peak_freq = np.nan
                                peak_amp = np.nan
                                peak_width = np.nan
                                found_peak = False
                                n_peaks_total = 0
                                n_peaks_in_band = 0

                            r_squared = fm.r_squared_
                            fit_error = fm.error_

                        except Exception as e:
                            peak_freq = np.nan
                            peak_amp = np.nan
                            peak_width = np.nan
                            found_peak = False
                            n_peaks_total = np.nan
                            n_peaks_in_band = np.nan
                            r_squared = np.nan
                            fit_error = np.nan
                            print(f"FOOOF failed for {pat} {run_label} {label}: {e}")

                        # Look up atlas region
                        fooof_region = None
                        match_row = elecs_subs[elecs_subs['label'] == label]

                        if not match_row.empty:
                            if 'Y17_Atlas' in match_row.columns:
                                fooof_region = match_row['Y17_Atlas'].values[0]
                            elif 'Y7_Atlas' in match_row.columns:
                                fooof_region = match_row['Y7_Atlas'].values[0]

                        fooof_results.append({
                            'Patient': pat,
                            'Run': run_label,
                            'Video': vid,
                            'FreqBand': freq_band,
                            'Channel': label,
                            'Region': fooof_region,

                            'FOOOF_Fit_Range_Low': fooof_fmin,
                            'FOOOF_Fit_Range_High': fooof_fmax,
                            'Peak_Search_Range_Low': peak_search_range[0],
                            'Peak_Search_Range_High': peak_search_range[1],

                            'Found_Peak': found_peak,
                            'Peak_Frequency': peak_freq,
                            'Peak_Amplitude': peak_amp,
                            'Peak_Width': peak_width,

                            'N_Peaks_Total': n_peaks_total,
                            'N_Peaks_In_Band': n_peaks_in_band,
                            'FOOOF_R2': r_squared,
                            'FOOOF_Error': fit_error
                        })

                    fooof_df = pd.DataFrame(fooof_results)

                    fooof_csv = os.path.join(
                        fooof_fig_patient_dir,
                        f'{pat}_{vid}_{run_label}_{region}_{freq_band}_fooof_peaks.csv'
                    )

                    fooof_df.to_csv(fooof_csv, index=False)
                    print(f'Saved FOOOF peak results to {fooof_csv}')

                    if not extract_power:
                        continue

                if plot_power:
                    # Create an MNE Info object (still useful for later if you want)
                    info = mne.create_info(ch_names=labels_ip, sfreq=fs_lfp,
                                           ch_types=['ecog'] * len(labels_ip))
                    lfp_dat = mne.io.RawArray(lfp_ip, info)

                    # Compute the PSD directly from the NumPy array
                    psd, freqs = psd_array_welch(
                        lfp_ip,                    # <--- use array, not RawArray
                        sfreq=fs_lfp,
                        fmin=freq_range[0],
                        fmax=freq_range[1],
                        n_fft=int(fs_lfp * 2),
                        n_overlap=int(fs_lfp),
                        average='mean'
                    )

                if output == 'power_z':

                    zscored_power_bins = []

                    for f_low, f_high in freq_bins:
                        # 1. Bandpass filter for each bin
                        sos = signal.butter(5, [f_low, f_high], btype='bandpass', fs=fs_lfp, output='sos')
                        band_bin = signal.sosfiltfilt(sos, lfp_ip, axis=1)

                        # 2. Hilbert transform -> amplitude
                        analytic = signal.hilbert(band_bin, axis=1)
                        power = np.abs(analytic)

                        # 3. Choose raw or log representation before z-scoring
                        if pow_type == 'raw':
                            power_for_z = power
                        elif pow_type == 'log':
                           # power_for_z = np.log10(power + 1e-6)
                            power_for_z = np.log10(np.maximum(power, np.finfo(power.dtype).tiny))
                        else:
                            raise ValueError(f"Unknown pow_type: {pow_type}")

                        # 4. Z-score across time (per channel)
                        mean_power = np.mean(power_for_z, axis=1, keepdims=True)
                        std_power = np.std(power_for_z, axis=1, keepdims=True)
                        z_power = (power_for_z - mean_power) / std_power
                        zscored_power_bins.append(z_power)

                    # 5. Average z-scored power across bins to get frequency-band-level power
                    pow_dat_z = np.mean(np.stack(zscored_power_bins, axis=0), axis=0)  # shape: (n_channels, n_times)

                    pow_dat = pow_dat_z


                elif output == 'power':
                    raw_power_bins = []

                    for f_low, f_high in freq_bins:
                        sos = signal.butter(5, [f_low, f_high], btype='bandpass', fs=fs_lfp, output='sos')
                        band_bin = signal.sosfiltfilt(sos, lfp_ip, axis=1)

                        analytic = signal.hilbert(band_bin, axis=1)
                        power = np.abs(analytic)  # raw envelope
                        raw_power_bins.append(power)

                    pow_dat_raw = np.mean(np.stack(raw_power_bins, axis=0), axis=0)  # (n_channels, n_times)

                    if pow_type == 'raw':
                        pow_dat = pow_dat_raw
                    elif pow_type == 'log':
                       # pow_dat = np.log10(pow_dat_raw + 1e-6)
                       pow_dat = np.log10(np.maximum(pow_dat_raw, np.finfo(pow_dat_raw.dtype).tiny))
                    else:
                        raise ValueError(f"Unknown pow_type: {pow_type}")

                    # Decimate 600 -> 300 Hz. The Hilbert envelope of a bandpassed
                    # signal is bandlimited by the PASSBAND WIDTH, not the carrier
                    # frequency: 99 Hz for HFA, the widest band. A 150 Hz Nyquist
                    # therefore has 1.5x margin, and no anti-alias filter is needed
                    # because nothing exists above the passband width to fold back.
                    #
                    # Placed AFTER the pow_type branches so it applies to both 'raw'
                    # and 'log'. fs_lfp is reassigned so every downstream index that
                    # derives from it - window_idx(), total_samples, new_window_samples
                    # - recomputes correctly. fs_lfp is re-read from
                    # mne_data.info['sfreq'] per recording, so this does not compound.
                    DECIM_POWER = 2
                    pow_dat = pow_dat[:, ::DECIM_POWER]
                    fs_lfp = fs_lfp / DECIM_POWER

                    # Round before the DataFrame is built. float_format on
                    # to_csv does NOT work here: df_w_atlas concatenates the
                    # string atlas rows with the numeric data, so every
                    # column becomes dtype=object and pandas falls back to
                    # str() per value, writing all 17 digits.
                    # 5 decimals on log10 power (~-4.8) is 6 significant
                    # figures, far beyond the measurement precision, and
                    # shortens each value from 19 chars to 8.
                    pow_dat = np.round(pow_dat, 5)

                else:
                    raise ValueError(f"Unknown output: {output}")

                t_lfp = np.arange(lfp_ip.shape[1]) / fs_lfp


                if window_compare:

                    def window_idx(t0, t1, fs, n_times):
                        i0 = max(0, int(np.floor(t0 * fs)))
                        i1 = min(n_times, int(np.ceil(t1 * fs)))
                        if i1 <= i0:
                            raise ValueError(f"Empty window: {t0}-{t1}s mapped to samples {i0}:{i1}")
                        return i0, i1

                    # Define windows
                    t_base0, t_base1 = 100, 125   # baseline
                    t_win0,  t_win1  = 170, 190   # target

                    n_times = pow_dat.shape[1]
                    b0, b1 = window_idx(t_base0, t_base1, fs_lfp, n_times)
                    w0, w1 = window_idx(t_win0,  t_win1,  fs_lfp, n_times)

                    # Mean power in each window per channel
                    base_mean = np.nanmean(pow_dat[:, b0:b1], axis=1)
                    win_mean  = np.nanmean(pow_dat[:, w0:w1], axis=1)

                    # Relative increase metrics
                    diff = win_mean - base_mean
                    pct  = (win_mean - base_mean) / (np.abs(base_mean) + 1e-12) * 100.0  # percent change

                    # Optional: effect size (robust-ish) using baseline std
                    base_std = np.nanstd(pow_dat[:, b0:b1], axis=1) + 1e-12
                    z_like = (win_mean - base_mean) / base_std

                    results = pd.DataFrame({
                        "Channel": labels_ip,
                        "BaseMean_100_125": base_mean,
                        "WinMean_160_185": win_mean,
                        "Diff": diff,
                        "PctChange": pct,
                        "Zlike": z_like
                    }).sort_values("Diff", ascending=False)

                    print(results.head(20))

                    cols_to_save = [
                        "Channel",
                        "BaseMean_100_125",
                        "WinMean_160_185",
                        "Diff",
                        "Zlike",
                        "PctChange"
                    ]

                    results_to_save = results[cols_to_save].copy()
                    results_to_save["Patient"] = pat
                    results_to_save["Run"] = run_label
                    results_to_save["Video"] = vid
                    results_to_save["FreqBand"] = freq_band
                    results_to_save["Window"] = "170-180"
                    results_to_save["Baseline"] = "160-170"
                    csv_fname = (
                        f"{pat}_{run_label}_{vid}_"
                        f"{freq_band}_{ref}_increase_100_125_vs_164_185.csv"
                    )

                    csv_path = os.path.join(fig_patient_dir, csv_fname)
                    results_to_save.to_csv(csv_path, index=False)
                    print(f"Saved HFA window comparison to:\n{csv_path}")
#

                # Legacy rolling average using z-scored data (preserved for future use)
                if condense_to_isc:

                    total_samples = int(fs_lfp * 600)  # 10 minutes = 600s → 360,000 samples
                    new_window_samples = int(10 * fs_lfp)  # 3000 samples
                    target_num_steps = 236

                    step_size_samples = int((total_samples - new_window_samples) / (target_num_steps - 1))  # ~1519
                    overlap_samples = new_window_samples - step_size_samples  # could be negative if windows don’t overlap

                    #total_samples = band_dat.shape[1]
                    total_duration = total_samples / fs_lfp 

                    #window_duration = total_duration / 236  # Duration of each window in seconds
                    #window_size_samples = int(fs_lfp * window_duration)  # Convert to samples
                    #window_samples = int(10*fs_lfp)  # 10 seconds
                    #overlap_samples = int(7.5*fs_lfp)  # 7.5 seconds 

                    #step_size_samples = window_samples - overlap_samples

                    num_steps = int(np.floor((total_samples - new_window_samples) / step_size_samples) + 1)

                    if output != 'entropy':
                        # Apply rolling average with non-overlapping windows
                        rolling_avg = np.apply_along_axis(
                            lambda x: pd.Series(x).rolling(window=new_window_samples, min_periods=new_window_samples//2, center = True).mean()[::new_window_samples].to_numpy(),
                            axis=1,
                            arr=pow_dat
                        )                        
                        # Generate time axis for the rolling averages
                        rolling_time = np.linspace(0, total_duration, 236)

                        # Existing rolling time and rolling average (length 237)
                        current_time = np.linspace(0, total_duration, len(rolling_avg[0]))  # 237 points
                        target_time = np.linspace(0, total_duration, 236)  # 236 points

                        # Interpolate each channel
                        rolling_avg_interpolated = np.array([
                            interpolate.interp1d(current_time, channel_avg, kind='linear')(target_time)
                            for channel_avg in rolling_avg
                        ])

                        fig, ax = plt.subplots(figsize=(10, 5))

                        for i, channel in enumerate(labels_ip):
                            ax.plot(rolling_time, rolling_avg_interpolated[i, :], label=channel)

                        ax.set_title(f"Rolling Average of Power Data for {pat} for {freq_band} ({freq_range}) {vid}")
                        ax.set_xlabel("Time (s)")
                        ax.set_ylabel("Power (Smoothed)")
                        ax.grid(True)

                        pow_fig_path = f"{pat}_{run_label}_{vid}_{freq_band}_all_elecs_power_over_time.png"
                        fig.savefig(os.path.join(fig_patient_dir, pow_fig_path), dpi=300, bbox_inches="tight")

                        plt.show()
                        plt.close(fig)


                        pow_dat = rolling_avg_interpolated

                # New rolling average for functional connectivity using log power
                if rolling_average:
                    # Use 10-second windows with 7.5s overlap, 236 observations (legacy logic)
                    window_length_sec = 10
                    overlap_sec = 7.5
                    step_size_sec = window_length_sec - overlap_sec  # 2.5s
                    fs = fs_lfp
                    new_window_samples = int(window_length_sec * fs)
                    step_size_samples = int(step_size_sec * fs)
                    total_samples = pow_dat.shape[1]
                    target_num_steps = 236

                    connectivity_matrices = []
                    rolling_times = []

                    for i in range(target_num_steps):
                        start = i * step_size_samples
                        end = start + new_window_samples
                        if end > total_samples:
                            break
                        window_data = pow_dat[:, start:end]
                        fc_matrix = np.corrcoef(window_data)
                        connectivity_matrices.append(fc_matrix)
                        rolling_times.append((start + end) / 2 / fs)

                    connectivity_matrices = np.stack(connectivity_matrices, axis=0)  # shape (n_windows, n_channels, n_channels)
                    rolling_times = np.array(rolling_times)

                # Save or visualize connectivity_matrices as needed
                
                # (Optional) Re-assign pow_dat for regression/plotting
                # Rolling mean for visualization (not needed for FC)
                if use_interpolation:
                    rolling_avg_interpolated = []
                    for x in pow_dat:
                        # Compute rolling mean at each window
                        vals = []
                        for i in range(target_num_steps):
                            start = i * step_size_samples
                            end = start + new_window_samples
                            if end > total_samples:
                                break
                            vals.append(np.mean(x[start:end]))
                        rolling_avg_interpolated.append(vals)
                    rolling_avg_interpolated = np.array(rolling_avg_interpolated)
    
                    # Use this for visualization or regression alignment
                    pow_dat = rolling_avg_interpolated    
                
                # if lowpass:
                #     #lowpass filter
                #     sos = signal.butter(5, 0.5*fs_isc, btype='lowpass', output='sos', fs=fs_lfp)
                #     pow_dat = signal.sosfiltfilt(sos, pow_dat, axis=1)
                                  
                #     # Create a RawArray with the filtered data
                #     lowpass_mne = mne.io.RawArray(pow_dat, info)
                #     #lowpass_mne.plot(block = True)
                    
                #     f = interpolate.interp1d(time_lfp, pow_dat)
                #     pow_dat = f(time_isc)
                
                if plot_power:
                   time_isc = np.arange(1, pow_dat.shape[1] + 1)
               
                   # Downsample only for plotting if not using rolling average
                   if not rolling_average:
                       ds_factor = 2400  # adjust (e.g., 5, 10, 20 depending on density)
                       time_isc_plot = time_isc[::ds_factor]
                       pow_dat_plot = pow_dat[:, ::ds_factor]
                   else:
                       time_isc_plot = time_isc
                       pow_dat_plot = pow_dat
               
                   fig, ax = plt.subplots(figsize=(10, 5))
               
                   for i, channel in enumerate(labels_ip):
                       ax.plot(time_isc_plot, pow_dat_plot[i, :], label=channel)
               
                   ax.set_title('Interpolated Power Data')
                   ax.set_xlabel('Window')
                   ax.set_ylabel('Power')
                   ax.grid(True)
               
                   plt.show()
    
                # continue otherwise
                pow_ip = pow_dat
                labels = labels_ip
                pow_ip = pow_ip.T
#%%

                # Prepare data lists for electrodes
                data_list = []
                variable_names = []
                
                if 'pow_ip' in locals() and 'labels_ip' in locals():
                    if pow_ip.ndim == 1:
                        data_list.append(pow_ip)
                        variable_names.append(f'{labels_ip[0]}')
                    else:
                        for i, label in enumerate(labels_ip):
                            data_list.append(pow_ip[:, i])
                            variable_names.append(f'{label}')
                
                # Generate the DataFrame from the data_matrix
                data_matrix = np.column_stack(data_list)
                df = pd.DataFrame(data_matrix, columns=variable_names)
            
                # Initialize lists for DK_Atlas, Y7_Atlas, and AparcAseg_Atlas regions
                dk_regions = []
                y7_regions = []
                y17_regions = []
                aparc_aseg_regions = []
                            

                # Make a copy so you don't permanently alter the original dataframe
                elecs_subs_lc = elecs_subs.copy()
                
                # Lowercase all column names
                elecs_subs_lc.columns = elecs_subs_lc.columns.str.strip().str.lower()
                
                # Lowercase electrode labels
                elecs_subs_lc["label_lc"] = elecs_subs_lc["label"].astype(str).str.strip().str.lower()
                
                for label in df.columns:
                
                    label_lc = str(label).strip().lower()
                
                    match_row = elecs_subs_lc[elecs_subs_lc["label_lc"] == label_lc]
                
                    if not match_row.empty:
                
                        dk_col = "dk_atlas" if "dk_atlas" in elecs_subs_lc.columns else "desikan_killiany"
                        y7_col = "y7_atlas" if "y7_atlas" in elecs_subs_lc.columns else "yeo7"
                        y17_col = "y17_atlas" if "y17_atlas" in elecs_subs_lc.columns else "yeo17"
                        aparc_col = "aparcaseg_atlas" if "aparcaseg_atlas" in elecs_subs_lc.columns else "aparc_aseg"
                
                        dk_regions.append(match_row[dk_col].values[0])
                        y7_regions.append(match_row[y7_col].values[0])
                        y17_regions.append(match_row[y17_col].values[0])
                        aparc_aseg_regions.append(match_row[aparc_col].values[0])
                
                    else:
                        dk_regions.append("Unknown")
                        y7_regions.append("Unknown")
                        y17_regions.append("Unknown")
                        aparc_aseg_regions.append("Unknown")
                    
     
                # Create a DataFrame to hold the atlas region rows
                atlas_rows = pd.DataFrame({
                    'Atlas': ['DK_Atlas_Region', 'Y7_Atlas_Region', 'Y17_Atlas_Region', 'AparcAseg_Atlas_Region'],
                    **{col: [dk_regions[i], y7_regions[i], y17_regions[i], aparc_aseg_regions[i]] for i, col in enumerate(df.columns)}
                })
                
                
                # Combine atlas rows and the main data into a single DataFrame
                df_w_atlas = pd.concat([atlas_rows, df], ignore_index=True)
                df_w_atlas = df_w_atlas[['Atlas'] + [col for col in df_w_atlas.columns if col != 'Atlas']]
                
                # Add a column with SubID and move it to the first position
                df_w_atlas['SubID'] = pat
                df_w_atlas = df_w_atlas[['SubID'] + [col for col in df_w_atlas.columns if col != 'SubID']]
                
    
                #### Relabel Y17 Atlas rows                 
                # Find the row where 'Atlas' is 'Y17_Atlas_Region'
                atlas_row = df_w_atlas[df_w_atlas['Atlas'] == 'Y17_Atlas_Region']
                
                if not atlas_row.empty:
                    # Loop through each column that contains network labels (LTs1, LTs2, ..., RPc1, RPc2, ...)
                    for column in df_w_atlas.columns:
                        # Replace values in relevant columns based on the network mapping
                        df_w_atlas.loc[atlas_row.index, column] = df_w_atlas.loc[atlas_row.index, column].replace(network_mapping)
                                    
                if rolling_average:
                    method = 'rolling_avg'
                if lowpass:
                    method ='lowpass'
    
                # Save the DataFrame with new rows to a CSV file
                if output == 'entropy':
                    csv_filename = os.path.join(fig_patient_dir, f'{pat}_{vid}_{run_label}_{freq_band}_cortical_{entropy_type}.csv')
                else:
                    csv_filename = os.path.join(fig_patient_dir, f'{pat}_{vid}_{run_label}_{freq_band}_cortical_{output}_{pow_type}.csv')
                df_w_atlas.to_csv(csv_filename, header=True, index=False,
                                                      float_format='%.6g')            
            
