#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 20:40:27 2024

@author: max

Edit (Christine request, Dec 2025):
- Use ALL runs from ALL patients as independent entries when computing ISC.
  i.e., if a patient has run-01 and run-02 for the same movie, both are included
  as separate “subjects” in the ISC pool.
- Keep the rest of the script’s behavior as close to the original as possible
  (same computations, same saving logic), but add entry metadata so you can tell
  which patient/run each ISC value corresponds to.

Notes:
- This script still equalizes length across entries by truncating to len_min,
  exactly like the original did across patients.
- It will also include file naming variations for the movie (e.g., dme) via aliases.
"""

import os
from itertools import compress
import numpy as np
import pandas as pd
from scipy import interpolate, signal
from tqdm import tqdm
import matplotlib.pyplot as plt

import re
from collections import defaultdict

#%%
data_dir = '/Volumes/Samsung/Movie_data/movies_prep_standard'
fs_dir = '/Volumes/Samsung/Movie_data/anatomy'
eloc_dir = '/Volumes/Samsung/Movie_data/data/electrode_localization'
#prep_dir = '/Volumes/Expansion/Movie_data/movies_prep_standard'

vid = 'despicable_me_hungarian'

# Create results directory
out_dir = f'/Volumes/Samsung/Movie_data/ISC_{vid}_10s_windows'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# (kept for compatibility with your old helper that prints matching dirs)
search_string = f'{vid}_run-1_et_prep.npz'

patients = os.listdir(data_dir)
patients.sort()

if vid == 'inscapes':
    patients = ['NS127_02','NS135','NS136','NS137','NS138','NS140','NS140_02','NS144','NS151','NS153','NS154','NS155','NS155_02','NS164','NS178', "NS205", 'NS210','NS211']

elif vid == 'despicable_me_english':
    patients = ['NS127_02',
             'NS135',
             'NS136',
             'NS137',
             'NS138',
             'NS140',
             'NS140_02',
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
    
elif vid == 'despicable_me_hungarian':
    patients = [
     'LH010',
     'NS127_02',
     'NS128',
     'NS135',
     'NS136',
     'NS137',
     'NS138',
     'NS140',
     'NS140_02',
     'NS144',
     'NS145',
     'NS151',
     'NS153',
     'NS154',
     'NS164',
     'NS166',
     'NS167',
     'NS174_02',
     'NS174_03',
     'NS178']

patients.sort()

# Eyetracking sampling rate
fs_eye = 300

# Downsampling factor
dsf = 3

# Time resolved ISC
window = 1000 #500  # 1000
overlap = 750 #250  # 750

# Number of permutations
n_perm = 100

#%% ---------------------------------------------------------------------
# Utils (unchanged)
# ---------------------------------------------------------------------
def interp_nans(data):

    s = np.arange(len(data))
    idx_nan = np.isnan(data)

    f = interpolate.interp1d(
        s[np.invert(idx_nan)],
        data[np.invert(idx_nan)],
        fill_value='extrapolate'
    )
    data[idx_nan] = f(np.where(idx_nan)[0])

    return data

def compute_isc(xy):

    corr_x = np.corrcoef(xy[:, 0, :].T)
    corr_y = np.corrcoef(xy[:, 1, :].T)

    corr = np.mean(
        np.concatenate(
            (np.expand_dims(corr_x, 2), np.expand_dims(corr_y, 2)),
            axis=2
        ),
        axis=2
    )

    corr[np.eye(corr.shape[0]) == 1] = np.nan

    isc = np.nanmean(corr, axis=0)

    return isc

def time_resolved_isc(data, fs, window, overlap):

    step = window - overlap

    time_isc = np.arange(window/2, len(data) - window/2, step) / fs

    n_pat = data.shape[1]
    x_win = [None] * n_pat

    for ip in range(n_pat):

        ts = data[:, ip]

        shape = (ts.size - window + 1, window)
        strides = ts.strides * 2
        ts_win = np.lib.stride_tricks.as_strided(ts, shape=shape, strides=strides)

        x_win[ip] = ts_win[:-1:step, :]

    corr_time = np.empty((n_pat, x_win[ip].shape[0]))
    for ip in range(n_pat):
        corr_pat = np.empty((n_pat, x_win[ip].shape[0]))
        for jp in range(n_pat):
            corr_mat = np.corrcoef(x_win[ip], x_win[jp])
            corr_pat[jp, :] = np.diag(corr_mat[:len(x_win[ip]), len(x_win[ip]):])

        corr_time[ip, :] = np.mean(corr_pat, axis=0)

    return corr_time, time_isc

def plot_isc_pat(isc, title_str):

    mean_isc = np.mean(isc)

    plt.figure()
    plt.hist(isc, color='tab:Gray', ec='k')

    ylim = plt.ylim()

    plt.plot([mean_isc, mean_isc], ylim, 'r', linewidth=2)
    plt.ylim(ylim)

    plt.xlabel('ISC')
    plt.ylabel('# of Patients')
    plt.legend(['Mean ISC', 'ISC for each patient'])
    plt.title(title_str)

    plt.grid()
    plt.tight_layout()

# ---------------------------------------------------------------------
# Helper: find patient directories that contain a specific file string (kept)
# ---------------------------------------------------------------------
def find_patient_directories_with_file(root_dir, search_string):
    matching_directories = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if search_string in filename:
                parent_directory = os.path.dirname(os.path.normpath(dirpath))
                matching_directories.append(parent_directory)
                break
    return matching_directories

matching_directories = find_patient_directories_with_file(data_dir, search_string)
print(matching_directories)

# ---------------------------------------------------------------------
# NEW: run-aware file listing (minimal, robust)
# ---------------------------------------------------------------------
RUN_RE = re.compile(r"_run-(\d+)", re.IGNORECASE)

def extract_run_number(fname: str):
    m = RUN_RE.search(fname)
    return m.group(1).zfill(2) if m else None

def movie_aliases(vid: str):
    """
    Map canonical movie name -> acceptable filename variants.
    Extend as needed.
    """
    v = vid.lower()
    if v in {"despicable_me_english", "despicable_me"}:
        return {"despicable_me_english", "despicable_me", "dme"}
    return {v}

def list_et_entries_for_patient(pat_dir: str, vid: str):
    """
    Return list of entries for ALL runs found for this patient.
    Each entry: {'run': '01', 'path': '/.../file.npz'}

    Preference (if both exist for same run):
      1) *_et_prep_updated.npz
      2) *_et_prep.npz
    """
    eye_dir = os.path.join(pat_dir, "Eye_prep")
    if not os.path.isdir(eye_dir):
        return []

    aliases = movie_aliases(vid)
    files = [f for f in os.listdir(eye_dir) if not f.startswith("._")]

    # Accept both *_et_prep.npz and *_et_prep_updated.npz
    cand = []
    for f in files:
        f_low = f.lower()
        if not f_low.endswith(".npz"):
            continue
        if "_et_prep" not in f_low:
            continue

        run = extract_run_number(f)
        if run is None:
            continue

        # match on task-<alias> OR _<alias>_ (legacy)
        if not any((f"_task-{a}_" in f_low) or (f"_{a}_" in f_low) for a in aliases):
            continue

        cand.append((run, f))

    # pick one per run (prefer updated)
    by_run = defaultdict(list)
    for run, f in cand:
        by_run[run].append(f)

    entries = []
    for run, flist in sorted(by_run.items()):
        updated = [f for f in flist if f.lower().endswith("_et_prep_updated.npz")]
        pick = sorted(updated or flist)[0]
        entries.append({"run": run, "path": os.path.join(eye_dir, pick)})

    return entries

#%% ---------------------------------------------------------------------
# Collect all data, but now across ALL (patient, run) entries
# ---------------------------------------------------------------------
entries = []  # each element is one "subject" in ISC: patient-run

for pat in patients:
    pat_dir = os.path.join(data_dir, pat)
    these = list_et_entries_for_patient(pat_dir, vid)
    for e in these:
        e["pat"] = pat
        e["id"] = f"{pat}_run-{e['run']}"
        entries.append(e)

print(f"Found {len(entries)} total entries (patient-run) for vid={vid}")
if len(entries) < 2:
    raise RuntimeError("Need >= 2 total entries to compute ISC.")

xy = [None] * len(entries)
gaze_var = [None] * len(entries)

xy_perm = [None] * len(entries)
gaze_var_perm = [None] * len(entries)

for i, ent in enumerate(entries):

    pat = ent["pat"]
    et_path = ent["path"]
    et_file = os.path.basename(et_path)

    print(f"Loading {ent['id']}: {et_file}")

    et_data = np.load(et_path)
    t = et_data['t_gaze']

    # Interpolate missing data
    x = interp_nans(et_data['xy'][:, 0])
    y = interp_nans(et_data['xy'][:, 1])

    xy_pat = np.vstack((x, y)).T

    # Interpolate gaps in time axis
    t_res = np.arange(t[0], t[-1], 1 / fs_eye)
    f = interpolate.interp1d(t, xy_pat.T)  # keep original behavior (no explicit fill_value)
    xy_pat = f(t_res).T
    t = t_res

    # Compute gaze variation
    xy_pat -= np.mean(xy_pat, axis=0)

    env = np.abs(signal.hilbert(xy_pat, axis=0))
    gaze_var_pat = np.sqrt(np.sum(env**2, axis=1))

    # Cut around movie
    idx_mov = np.logical_and(t >= et_data['t_pupil'][0], t < et_data['t_pupil'][-1])

    xy[i] = xy_pat[idx_mov, :]
    gaze_var[i] = gaze_var_pat[idx_mov]

    # Downsample
    xy[i] = signal.resample(xy[i], int(len(xy[i]) / dsf))
    gaze_var[i] = signal.resample(gaze_var[i], int(len(gaze_var[i]) / dsf))

    # Permute the time series
    xy_perm[i] = np.empty((xy[i].shape[0], xy[i].shape[1], n_perm))
    gaze_var_perm[i] = np.empty((len(gaze_var[i]), n_perm))

    idx_perm = [np.random.randint(len(xy[i])) for j in range(n_perm)]

    for p in range(n_perm):
        xy_perm[i][:, :, p] = np.vstack((xy[i][idx_perm[p]:, :], xy[i][:idx_perm[p], :]))
        gaze_var_perm[i][:, p] = np.hstack((gaze_var[i][idx_perm[p]:], gaze_var[i][:idx_perm[p]]))

# ---------------------------------------------------------------------
# Reorganize data (same as original, but for entries instead of patients)
# ---------------------------------------------------------------------
idx_not_none = [c is not None for c in xy]

entries = list(compress(entries, idx_not_none))
xy = list(compress(xy, idx_not_none))
gaze_var = list(compress(gaze_var, idx_not_none))

xy_perm = list(compress(xy_perm, idx_not_none))
gaze_var_perm = list(compress(gaze_var_perm, idx_not_none))

print(f"Kept {len(entries)} entries after filtering.")
if len(entries) < 2:
    raise RuntimeError("Need >= 2 entries after filtering to compute ISC.")

# Equalize length across entries
len_min = np.min([len(c) for c in xy])

xy = [c[:len_min, :] for c in xy]
gaze_var = [c[:len_min] for c in gaze_var]

xy_perm = [c[:len_min, :, :] for c in xy_perm]
gaze_var_perm = [c[:len_min, :] for c in gaze_var_perm]

xy = np.concatenate([np.expand_dims(c, 2) for c in xy], axis=2)               # (T, 2, N)
gaze_var = np.concatenate([np.expand_dims(c, 1) for c in gaze_var], axis=1)   # (T, N)

xy_perm = np.concatenate([np.expand_dims(c, 3) for c in xy_perm], axis=3)     # (T, 2, n_perm, N)
gaze_var_perm = np.concatenate([np.expand_dims(c, 2) for c in gaze_var_perm], axis=2)  # (T, n_perm, N)

# Entry metadata (NEW, minimal)
entry_ids = [e["id"] for e in entries]
entry_pats = [e["pat"] for e in entries]
entry_runs = [e["run"] for e in entries]

# ---------------------------------------------------------------------
# Compute ISC (same as original)
# ---------------------------------------------------------------------
isc = compute_isc(xy)

isc_perm = np.empty((n_perm, len(isc)))
for p in range(n_perm):
    isc_perm[p, :] = compute_isc(xy_perm[:, :, p, :])

# ---------------------------------------------------------------------
# Gaze variation (same as original; keep isc_gaze_var)
# ---------------------------------------------------------------------
corr = np.corrcoef(gaze_var.T)
corr[np.eye(corr.shape[0]) == 1] = np.nan
isc_gaze_var = np.nanmean(corr, axis=0)

isc_gaze_perm = np.empty((n_perm, gaze_var.shape[1]))
for p in range(n_perm):
    corr = np.corrcoef(gaze_var_perm[:, p, :].T)
    corr[np.eye(corr.shape[0]) == 1] = np.nan
    isc_gaze_perm[p, :] = np.nanmean(corr, axis=0)

# ---------------------------------------------------------------------
# Slow fluctuations (same as original)
# ---------------------------------------------------------------------
sos = signal.butter(5, [0.05, 0.15], btype='bandpass', fs=fs_eye, output='sos')
xy_slow = signal.sosfiltfilt(sos, xy, axis=0)
isc_slow = compute_isc(xy_slow)

# ---------------------------------------------------------------------
# Time resolved ISC (same as original)
# ---------------------------------------------------------------------
# Gaze position
# IMPORTANT: negative-dimensions error occurs if len_min < window.
# The original script implicitly assumed long enough series; we add a minimal guard.
if len_min <= window:
    raise ValueError(
        f"Time-resolved ISC window={window} is longer than available timepoints len_min={len_min}. "
        f"Reduce window/overlap or ensure longer recordings."
    )

isc_time_x, time_isc = time_resolved_isc(xy[:, 0, :], (fs_eye / dsf), window, overlap)
isc_time_y, _ = time_resolved_isc(xy[:, 1, :], (fs_eye / dsf), window, overlap)

isc_time_x_perm = np.empty(np.hstack((isc_time_x.shape, n_perm)))
isc_time_y_perm = np.empty(np.hstack((isc_time_x.shape, n_perm)))

print('Computing surrogate distribution of time resolved ISC (Gaze position)')

for ip in tqdm(range(n_perm)):
    isc_time_x_perm[:, :, ip], _ = time_resolved_isc(xy_perm[:, 0, ip, :], (fs_eye / dsf), window, overlap)
    isc_time_y_perm[:, :, ip], _ = time_resolved_isc(xy_perm[:, 1, ip, :], (fs_eye / dsf), window, overlap)

isc_time_gaze = np.mean(
    np.concatenate((np.expand_dims(isc_time_x, 2), np.expand_dims(isc_time_y, 2)), axis=2),
    axis=2
)

isc_time_gaze_perm = np.mean(
    np.concatenate((np.expand_dims(isc_time_x_perm, 3), np.expand_dims(isc_time_y_perm, 3)), axis=3),
    axis=3
)

# Gaze variation time-resolved
isc_time_var, _ = time_resolved_isc(gaze_var, (fs_eye / dsf), window, overlap)

isc_time_var_perm = np.empty(np.hstack((isc_time_x.shape, n_perm)))

print('Computing surrogate distribution of time resolved ISC (Gaze variation)')

for ip in tqdm(range(n_perm)):
    isc_time_var_perm[:, :, ip], _ = time_resolved_isc(gaze_var_perm[:, ip, :], (fs_eye / dsf), window, overlap)

# ---------------------------------------------------------------------
# Plots (same as original; titles now reflect "entries" but kept labels)
# ---------------------------------------------------------------------
plt.rcParams.update({'font.size': 14})

plot_isc_pat(isc, 'Gaze position ISC')

plt.figure()
plt.hist(np.mean(isc_perm, axis=1), color='tab:Gray', ec='k')
ylim = plt.ylim()
plt.plot([np.mean(isc), np.mean(isc)], ylim, 'r', linewidth=2)
plt.ylim(ylim)
plt.xlabel('ISC')
plt.ylabel('# of permutations')
plt.legend(['Original Data', 'Permutations'])
plt.title('Permutation test')
plt.xticks(rotation=45)
plt.tight_layout()

plot_isc_pat(isc_slow, 'ISC of slow fluctuations')

plot_isc_pat(isc_gaze_var, 'Gaze Variation ISC')
plt.savefig(f'{out_dir}/{vid}_gaze_variation_isc_patients.png', dpi=300)

plt.figure()
plt.plot(time_isc, isc_time_gaze.T, color='tab:Gray')
plt.plot(time_isc, np.mean(isc_time_gaze, axis=0), 'k', linewidth=2)
plt.xlim([time_isc[0], time_isc[-1]])
plt.xlabel('Time [s]')
plt.ylabel('ISC')
plt.title('Time resolved ISC of Gaze Position')
plt.tight_layout()
plt.grid()
plt.tight_layout()

plt.figure()
plt.plot(time_isc, isc_time_var.T, color='tab:Gray')
plt.plot(time_isc, np.mean(isc_time_var, axis=0), 'k', linewidth=2)
plt.xlim([time_isc[0], time_isc[-1]])
plt.xlabel('Time [s]')
plt.ylabel('ISC')
plt.title('Time resolved ISC of Gaze Variance')
plt.tight_layout()
plt.grid()
plt.tight_layout()

# Stats of time resolved ISC

plt.figure()
plt.plot(time_isc, np.mean(isc_time_gaze_perm, axis=0), color='tab:Gray')
plt.plot(time_isc, np.mean(isc_time_gaze, axis=0), color='k', linewidth=2)
plt.xlim([time_isc[0], time_isc[-1]])
ylim = plt.ylim()
plt.ylim([0, ylim[1]])
plt.xlabel('Time [s]')
plt.ylabel('ISC')
plt.title('Mean ISC (Gaze variation) with surrogate distribution')
plt.grid()
plt.tight_layout()

plt.figure()
plt.plot(time_isc, np.mean(isc_time_var_perm, axis=0), color='tab:Gray')
plt.plot(time_isc, np.mean(isc_time_var, axis=0), color='k', linewidth=2)
plt.xlim([time_isc[0], time_isc[-1]])
ylim = plt.ylim()
plt.ylim([0, ylim[1]])
plt.xlabel('Time [s]')
plt.ylabel('ISC')
plt.title('Mean ISC (Gaze variation) with surrogate distribution')
plt.grid()
plt.tight_layout()

# ---------------------------------------------------------------------
# Save data (same files, but include entry metadata so you can map results)
# ---------------------------------------------------------------------
# ISC for whole recording
np.savez(
    f'{out_dir}/{vid}_isc_updated.npz',
    isc_gaze_pos=isc,
    isc_gaze_pos_slow=isc_slow,
    isc_gaze_var=isc_gaze_var,
    isc_gaze_pos_perm=isc_perm,
    patients=entry_pats,     # now one per entry
    runs=entry_runs,         # NEW
    entry_ids=entry_ids,     # NEW
    n_perm=n_perm,
    fs=fs_eye/dsf
)

# Gaze position
np.savez(
    f'{out_dir}/{vid}_isc_gaze_position_time_updated.npz',
    isc_time_gaze=isc_time_gaze,
    isc_time_gaze_perm=isc_time_gaze_perm,
    time_isc=time_isc,
    patients=entry_pats,     # now one per entry
    runs=entry_runs,         # NEW
    entry_ids=entry_ids,     # NEW
    window=window,
    overlap=overlap,
    n_perm=n_perm,
    fs=fs_eye/dsf
)

# Gaze variation
np.savez(
    f'{out_dir}/{vid}_isc_gaze_variation_time_updated.npz',
    isc_time_var=isc_time_var,
    isc_time_var_perm=isc_time_var_perm,
    time_isc=time_isc,
    patients=entry_pats,     # now one per entry
    runs=entry_runs,         # NEW
    entry_ids=entry_ids,     # NEW
    window=window,
    overlap=overlap,
    n_perm=n_perm,
    fs=fs_eye/dsf
)

print(f"DONE vid={vid}: saved outputs with N_entries={len(entries)}")
