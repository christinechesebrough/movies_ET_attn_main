#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract continuous canonical-band log power from full-resolution
linear Morlet-wavelet HDF5 files.

Input HDF5 structure expected:
    pow_tf_dat : channels x frequencies x time
    freqs_tf   : wavelet frequencies
    labels_ip  : channel labels

Expected HDF5 attributes:
    pat
    run_label
    vid
    fs_tf

Processing:
    1. Loop through recordings.
    2. Loop sequentially through canonical frequency bands.
    3. Divide each canonical band into sub-bins using the same
       bin-width scheme as the previous Hilbert extraction.
    4. Within each sub-bin, average LINEAR wavelet power over
       available wavelet frequencies.
    5. Average those sub-bin timecourses equally.
    6. Log10-transform the resulting continuous band-power signal.
    7. Save continuous log-band-power as HDF5.

No temporal downsampling, smoothing, z-scoring, or windowing is done here.
"""

import os
import glob
import h5py
import numpy as np

WINDOW_SEC = 10
OVERLAP_SEC = 7.5
STEP_SEC = WINDOW_SEC - OVERLAP_SEC

robust_zscore = True

def robust_zscore_channels(pow_dat):
    median = np.nanmedian(pow_dat, axis=1, keepdims=True)
    mad = np.nanmedian(np.abs(pow_dat - median), axis=1, keepdims=True)
    robust_sd = 1.4826 * mad
    robust_sd[robust_sd == 0] = np.nan
    z_dat = (pow_dat - median) / robust_sd
    return z_dat, median[:, 0], robust_sd[:, 0]


def rolling_average_windows(pow_dat, fs, window_sec, overlap_sec):
    window_samples = int(window_sec * fs)
    step_samples = int((window_sec - overlap_sec) * fs)

    n_channels, n_samples = pow_dat.shape
    starts = range(0, n_samples - window_samples + 1, step_samples)

    means = []
    centers = []

    for start in starts:
        end = start + window_samples
        means.append(np.nanmean(pow_dat[:, start:end], axis=1))
        centers.append((start + end) / 2.0 / fs)

    condensed = np.column_stack(means)
    window_centers_sec = np.asarray(centers)

    return condensed, window_centers_sec
# ============================================================
# USER SETTINGS
# ============================================================

machine_path = "media/christine"

vids = [#"despicable_me_hungarian",
        "despicable_me_english","inscapes"]

freq_bands = [
    "theta","alpha", "beta", "gamma", "HFA"]

wavelet_root = f"/{machine_path}/Data/Movie_data"
band_power_root = f"/{machine_path}/Data/Movie_data/wavelet_band_power"

os.makedirs(band_power_root, exist_ok=True)

channel_batch_size = 8
overwrite = False


# ============================================================
# PATIENT LISTS
# ============================================================

patient_lists = {
    "despicable_me_hungarian": [
        "LH010", "NS127_02",
       "NS135", 
        "NS136", "NS137", 
        "NS138", "NS140", "NS140_02",
        "NS142", "NS144", "NS145", "NS154", "NS155_02", "NS164", "NS174_02", "NS174_03",
        "NS178", "NS211"
    ],

    "despicable_me_english": [
      #  "NS127_02", "NS135",
        "NS136", "NS137", "NS138", "NS140", "NS140_02", "NS153",
        "NS154", "NS155_02", "NS164", "NS174_02", "NS174_03", "NS178", "NS191", "NS193",
        "NS194", "NS201_02", "NS205"
    ],

    "inscapes": [
        "NS127_02", "NS135", "NS136", "NS137", "NS138", "NS140", "NS140_02", "NS153",
        "NS155_02", "NS164", "NS178", "NS205", "NS210"
    ],
}


# ============================================================
# FREQUENCY-BAND DEFINITIONS
# ============================================================

def get_band_settings(freq_band):
    if freq_band == "alpha":
        freq_range = (8, 13)
        bin_width = 2
    elif freq_band == "HFA":
        freq_range = (51, 150)
        bin_width = 10
    elif freq_band == "delta":
        freq_range = (1, 3)
        bin_width = 2
    elif freq_band == "theta":
        freq_range = (4, 7)
        bin_width = 2
    elif freq_band == "beta":
        freq_range = (14, 30)
        bin_width = 4
    elif freq_band == "gamma":
        freq_range = (31, 50)
        bin_width = 8
    elif freq_band == "all_gamma":
        freq_range = (31, 150)
        bin_width = 8
    elif freq_band == "mid_gamma":
        freq_range = (50, 69)
        bin_width = 8
    elif freq_band == "theta_alpha":
        freq_range = (4, 13)
        bin_width = 2
    else:
        raise ValueError(f"No frequency range assigned for {freq_band}")

    return freq_range, bin_width


# ============================================================
# CREATE FREQUENCY SUB-BINS
# ============================================================

def make_frequency_bins(freq_range, bin_width):
    f_start, f_end = freq_range
    bins = []
    f_low = f_start

    while f_low < f_end:
        f_high = min(f_low + bin_width, f_end)
        bins.append((f_low, f_high))
        f_low += bin_width

    return bins


# ============================================================
# FIND WAVELET FILES
# ============================================================

def find_wavelet_files(wavelet_root, pat, vid):
    candidates = glob.glob(os.path.join(wavelet_root, "**", pat, "*.h5"), recursive=True)
    candidates = [f for f in candidates if vid.lower() in os.path.basename(f).lower() and "wavelet" in os.path.basename(f).lower() and "tf" in os.path.basename(f).lower()]
    return sorted(candidates)


# ============================================================
# MAIN SCRIPT
# ============================================================
#%%
for vid in vids:

    print("\n" + "=" * 70)
    print(f"PROCESSING VIDEO: {vid}")
    print("=" * 70)

    patients = patient_lists[vid]
    patients.sort()

    for pat in patients:

        print("\n" + "-" * 70)
        print(f"Patient: {pat}")
        print("-" * 70)

        wavelet_files = find_wavelet_files(wavelet_root, pat, vid)

        print(f"Found {len(wavelet_files)} wavelet file(s)")

        for f in wavelet_files:
            print("   ", os.path.basename(f))

        if len(wavelet_files) == 0:
            print(f"[SKIP] No wavelet H5 files found for {pat} {vid}")
            continue

        for wavelet_file in wavelet_files:

            print(f"\nLoading wavelet file:\n{wavelet_file}")

            with h5py.File(wavelet_file, "r") as h5in:

                required_datasets = ["pow_tf_dat", "freqs_tf", "labels_ip"]

                for dataset in required_datasets:
                    if dataset not in h5in:
                        raise KeyError(f"{dataset} not found in {wavelet_file}")

                freqs_tf = h5in["freqs_tf"][:].astype(float)
                labels_raw = h5in["labels_ip"][:]
                labels_ip = [x.decode() if isinstance(x, bytes) else str(x) for x in labels_raw]

                fs_tf = float(h5in.attrs["fs_tf"])
                pat_file = h5in.attrs.get("pat", pat)
                vid_file = h5in.attrs.get("vid", vid)
                run_label = h5in.attrs.get("run_label", "run-unknown")

                if isinstance(pat_file, bytes):
                    pat_file = pat_file.decode()

                if isinstance(vid_file, bytes):
                    vid_file = vid_file.decode()

                if isinstance(run_label, bytes):
                    run_label = run_label.decode()

                tf_dataset = h5in["pow_tf_dat"]

                n_channels = tf_dataset.shape[0]
                n_freqs = tf_dataset.shape[1]
                n_times = tf_dataset.shape[2]

                print("Wavelet shape:", tf_dataset.shape)
                print(f"Sampling rate: {fs_tf} Hz")
                print(f"Frequency range: {freqs_tf[0]}-{freqs_tf[-1]} Hz")
                print(f"Duration: {n_times / fs_tf:.2f} s")

                # ============================================================
                # LOOP SEQUENTIALLY THROUGH CANONICAL BANDS
                # ============================================================

                for freq_band in freq_bands:

                    freq_range, bin_width = get_band_settings(freq_band)
                    freq_bins = make_frequency_bins(freq_range, bin_width)

                    print(f"\nProcessing {freq_band}: {freq_range[0]}-{freq_range[1]} Hz")
                    print(f"Bin width: {bin_width} Hz")
                    print("Frequency bins:")

                    bin_indices = []

                    for bin_i, (f_low, f_high) in enumerate(freq_bins):

                        is_last_bin = bin_i == len(freq_bins) - 1

                        if is_last_bin:
                            idx = np.where((freqs_tf >= f_low) & (freqs_tf <= f_high))[0]
                        else:
                            idx = np.where((freqs_tf >= f_low) & (freqs_tf < f_high))[0]

                        if len(idx) == 0:
                            print(f"  WARNING: {f_low}-{f_high} Hz contains no wavelet frequencies.")
                            continue

                        bin_indices.append({"f_low": f_low, "f_high": f_high, "indices": idx})

                        print(f"  {f_low:>5.1f}-{f_high:<5.1f} Hz -> {freqs_tf[idx]}")

                    if len(bin_indices) == 0:
                        print(f"[SKIP] No usable frequencies for {freq_band}")
                        continue

                    # ========================================================
                    # OUTPUT DIRECTORY
                    # ========================================================

                    band_dir = os.path.join(band_power_root, vid, freq_band, pat)
                    os.makedirs(band_dir, exist_ok=True)

                    base_name = os.path.basename(wavelet_file)

                    if base_name.endswith(".h5"):
                        base_name = base_name[:-3]
                    
                    # ============================================================
                    # CREATE CONTINUOUS LOG-POWER OUTPUT FILE
                    # ============================================================
                    
                    continuous_file = os.path.join(
                        band_dir,
                        f"{base_name}_{freq_band}_log_band_power.h5"
                    )
                    
       
                    
                    continuous_exists = os.path.exists(continuous_file)
                    
                    if continuous_exists and not overwrite:
                        print(f"Continuous output already exists, reusing:\n{continuous_file}")
                    
                    else:
                        with h5py.File(continuous_file, "w") as h5out:
                                            
                            h5out.attrs["pat"] = str(pat_file)
                            h5out.attrs["vid"] = str(vid_file)
                            h5out.attrs["run_label"] = str(run_label)
                            h5out.attrs["freq_band"] = freq_band
                            h5out.attrs["freq_min"] = freq_range[0]
                            h5out.attrs["freq_max"] = freq_range[1]
                            h5out.attrs["bin_width"] = bin_width
                            h5out.attrs["fs"] = fs_tf
                            h5out.attrs["representation"] = "log10_mean_linear_wavelet_power_equal_weight_frequency_bins"
                            h5out.attrs["source_wavelet_file"] = wavelet_file
                        
                            h5out.create_dataset("labels_ip", data=np.asarray(labels_ip, dtype="S"))
                        
                            frequencies_used = np.concatenate([b["indices"] for b in bin_indices])
                            frequencies_used = np.unique(frequencies_used)
                        
                            h5out.create_dataset("freqs_used", data=freqs_tf[frequencies_used].astype(np.float32))
                        
                            sub_bin_ranges = np.asarray([[b["f_low"], b["f_high"]] for b in bin_indices], dtype=np.float32)
                            h5out.create_dataset("frequency_bins", data=sub_bin_ranges)
                        
                            band_dataset = h5out.create_dataset(
                                "pow_dat",
                                shape=(n_channels, n_times),
                                dtype=np.float32,
                                chunks=(1, min(int(fs_tf * 10), n_times)),
                                compression="gzip",
                                compression_opts=4,
                                shuffle=True
                            )
                        
                            # ========================================================
                            # PROCESS CHANNELS IN BATCHES
                            # ========================================================
                        
                            for batch_start in range(0, n_channels, channel_batch_size):
                        
                                batch_end = min(batch_start + channel_batch_size, n_channels)
                        
                                print(f"  Channels {batch_start + 1}-{batch_end} of {n_channels}")
                        
                                band_sum = np.zeros((batch_end - batch_start, n_times), dtype=np.float64)
                                n_valid_bins = 0
                        
                                for bin_info in bin_indices:
                        
                                    idx = bin_info["indices"]
                        
                                    bin_tf = tf_dataset[batch_start:batch_end, idx, :]
                                    bin_power = np.mean(bin_tf, axis=1, dtype=np.float64)
                        
                                    band_sum += bin_power
                                    n_valid_bins += 1
                        
                                    del bin_tf, bin_power
                        
                                band_power_linear = band_sum / n_valid_bins
                        
                                del band_sum
                        
                                tiny = np.finfo(band_power_linear.dtype).tiny
                                band_power_log = np.log10(np.maximum(band_power_linear, tiny))
                        
                                band_dataset[batch_start:batch_end, :] = band_power_log.astype(np.float32)
                        
                                h5out.flush()
                        
                                del band_power_linear, band_power_log
                        
                        
                        print(f"Saved continuous {freq_band} log power to:\n{continuous_file}")
                        
                    
                    # ============================================================
                    # CREATE SEPARATE WINDOWED Z-SCORED FILE
                    # ============================================================
                    
                    windowed_file = os.path.join(
                        band_dir,
                        f"{base_name}_{freq_band}_robust_z_windowed_{WINDOW_SEC}s.h5"
                    )
                    
                    if os.path.exists(windowed_file) and not overwrite:
                        print(f"[SKIP] Windowed output already exists:\n{windowed_file}")
                        continue
                    
                    
                    # ============================================================
                    # LOAD CONTINUOUS LOG POWER
                    # ============================================================
                    
                    with h5py.File(continuous_file, "r") as h5cont:
                    
                        pow_log = h5cont["pow_dat"][:]
                        labels_windowed = h5cont["labels_ip"][:]
                    
                        fs_window = float(h5cont.attrs["fs"])
                    
                    
                    # ============================================================
                    # ROBUST Z-SCORE ACROSS CONTINUOUS TIME
                    # ============================================================
                    
                    print(f"  Robust z-scoring continuous {freq_band} log power...")
                    
                    pow_z, robust_median, robust_sd = robust_zscore_channels(pow_log)
                    
                    print(f"  Continuous z-scored shape: {pow_z.shape}")
                    
                    
                    # ============================================================
                    # ROLLING WINDOW AVERAGE
                    # ============================================================
                    
                    windowed_z, window_centers = rolling_average_windows(
                        pow_z,
                        fs_window,
                        WINDOW_SEC,
                        OVERLAP_SEC
                    )
                    
                    print(f"  Windowed shape: {windowed_z.shape}")
                    print(f"  Number of windows: {windowed_z.shape[1]}")
                    
                    
                    # ============================================================
                    # SAVE WINDOWED FILE
                    # ============================================================
                    
                    with h5py.File(windowed_file, "w") as h5win:
                    
                        h5win.attrs["pat"] = str(pat_file)
                        h5win.attrs["vid"] = str(vid_file)
                        h5win.attrs["run_label"] = str(run_label)
                        h5win.attrs["freq_band"] = freq_band
                        h5win.attrs["fs_original"] = fs_window
                    
                        h5win.attrs["zscore_method"] = "median_MAD"
                        h5win.attrs["robust_sd_scale"] = 1.4826
                    
                        h5win.attrs["window_sec"] = WINDOW_SEC
                        h5win.attrs["overlap_sec"] = OVERLAP_SEC
                        h5win.attrs["step_sec"] = STEP_SEC
                    
                        h5win.attrs["source_continuous_file"] = continuous_file
                    
                        h5win.create_dataset("labels_ip", data=labels_windowed)
                    
                        h5win.create_dataset(
                            "robust_median",
                            data=robust_median.astype(np.float32)
                        )
                    
                        h5win.create_dataset(
                            "robust_sd",
                            data=robust_sd.astype(np.float32)
                        )
                    
                        h5win.create_dataset(
                            "window_centers_sec",
                            data=window_centers.astype(np.float32)
                        )
                    
                        h5win.create_dataset(
                            "pow_dat_z_windowed",
                            data=windowed_z.astype(np.float32),
                            compression="gzip",
                            compression_opts=4,
                            shuffle=True
                        )
                    
                    
                    print(f"Saved robust-z windowed {freq_band} power to:\n{windowed_file}")
                    
                    
                    del pow_log, pow_z, robust_median, robust_sd, windowed_z, window_centers