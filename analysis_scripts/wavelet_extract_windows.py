#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 05:50:29 2026

@author: christine
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract continuous log-transformed and robust-z-scored Morlet wavelet power
while retaining the full channel x frequency x time representation.

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
    2. Load wavelet power in channel x frequency batches.
    3. Log10-transform wavelet power separately at every frequency.
    4. Robust-z-score every channel x frequency time course across the full recording.
    5. Save the full continuous standardized wavelet representation.
    6. Define rolling 10-second windows with 7.5-second overlap.
    7. Save window start/end sample indices and times.

Important:
    - No averaging across frequencies.
    - No averaging across time.
    - No temporal downsampling or smoothing.
    - Windows retain the continuous time-frequency representation.
    - Overlapping window samples are not duplicated in the HDF5 file.
"""

import os
import glob
import h5py
import numpy as np


# ============================================================
# USER SETTINGS
# ============================================================

WINDOW_SEC = 10
OVERLAP_SEC = 7.5
STEP_SEC = WINDOW_SEC - OVERLAP_SEC

machine_path = "media/christine"

vids = [
     "despicable_me_hungarian",
    "despicable_me_english",
    "inscapes"
]

# wavelet_root = f"/{machine_path}/Data/Movie_data/wavelet_{vid}_all_cortContacts_tf_10s_(1, 150)_26Aug26"
wavelet_output_root = f"/{machine_path}/Data/Movie_data/wavelet_continuous_z"

# os.makedirs(wavelet_output_root, exist_ok=True)

channel_batch_size = 16
frequency_batch_size = 16

overwrite = False


# ============================================================
# PATIENT LISTS
# ============================================================

patient_lists = {
    "despicable_me_hungarian": [
        "LH010",
        "NS127_02",
        "NS135",
        "NS136",
        "NS137",
        "NS138",
        "NS140",
        "NS140_02",
        "NS142",
        "NS144",
        "NS145",
        "NS154",
        "NS155_02",
        "NS164",
        "NS174_02",
        "NS174_03",
        "NS178",
        "NS211"
    ],

    "despicable_me_english": [
        "NS127_02",
        "NS135",
        "NS136",
        "NS137",
        "NS138",
        "NS140",
        "NS140_02",
        "NS153",
        "NS154",
        "NS155_02",
        "NS164",
        "NS174_02",
        "NS174_03",
        "NS178",
        "NS191",
        "NS193",
        "NS194",
        "NS201_02",
        "NS205"
    ],

    "inscapes": [
        "NS127_02",
        "NS135",
        "NS136",
        "NS137",
        "NS138",
        "NS140",
        "NS140_02",
        "NS153",
        "NS155_02",
        "NS164",
        "NS178",
        "NS205",
        "NS210"
    ]
}


# ============================================================
# ROBUST Z-SCORE
# ============================================================

def robust_zscore_tf(tf_log):
    """
    Robust-z-score every channel x frequency time course across continuous time.

    Parameters
    ----------
    tf_log : ndarray
        Shape:
            channels x frequencies x time

    Returns
    -------
    tf_z : ndarray
        Robust-z-scored time-frequency representation.
        Shape:
            channels x frequencies x time

    median : ndarray
        Median for each channel x frequency.
        Shape:
            channels x frequencies

    robust_sd : ndarray
        MAD-derived robust standard deviation for each channel x frequency.
        Shape:
            channels x frequencies
    """

    median = np.nanmedian(tf_log, axis=2, keepdims=True)
    mad = np.nanmedian(np.abs(tf_log - median), axis=2, keepdims=True)
    robust_sd = 1.4826 * mad
    robust_sd[robust_sd == 0] = np.nan
    tf_z = (tf_log - median) / robust_sd

    return tf_z, median[:, :, 0], robust_sd[:, :, 0]


# ============================================================
# DEFINE ROLLING WINDOW INDICES
# ============================================================

def get_rolling_window_indices(n_samples, fs, window_sec, overlap_sec):
    """
    Define rolling windows without averaging or copying the underlying data.

    Returns
    -------
    starts : ndarray
        Inclusive window start sample.

    ends : ndarray
        Exclusive window end sample.

    centers_sec : ndarray
        Window center times in seconds.
    """

    window_samples = int(round(window_sec * fs))
    step_samples = int(round((window_sec - overlap_sec) * fs))
    starts = np.arange(0, n_samples - window_samples + 1, step_samples, dtype=np.int64)
    ends = starts + window_samples
    centers_sec = ((starts + ends) / 2.0) / fs

    return starts, ends, centers_sec


# ============================================================
# FIND WAVELET FILES
# ============================================================


def find_wavelet_files(wavelet_root, pat, vid):
    candidates = glob.glob(os.path.join(wavelet_root, pat, "*_wavelet_raw_tf.h5"))
    candidates = [f for f in candidates if vid.lower() in os.path.basename(f).lower()]
    return sorted(candidates)
# ============================================================
# MAIN SCRIPT
# ============================================================

#%%

for vid in vids:

    wavelet_root = f"/{machine_path}/Data/Movie_data/wavelet_{vid}_all_cortContacts_tf_10s_(1, 150)_26Aug26"

    print("\n" + "=" * 70)
    print(f"PROCESSING VIDEO: {vid}")
    print(f"WAVELET ROOT: {wavelet_root}")
    print("=" * 70)

    patients = sorted(patient_lists[vid])

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

                n_channels, n_freqs, n_times = tf_dataset.shape

                print(f"Wavelet shape: {tf_dataset.shape}")
                print(f"Sampling rate: {fs_tf} Hz")
                print(f"Number of channels: {n_channels}")
                print(f"Number of frequencies: {n_freqs}")
                print(f"Frequency range: {freqs_tf[0]}-{freqs_tf[-1]} Hz")
                print(f"Number of time samples: {n_times}")
                print(f"Duration: {n_times / fs_tf:.2f} s")


                # ========================================================
                # DEFINE ROLLING WINDOWS
                # ========================================================

                window_starts, window_ends, window_centers = get_rolling_window_indices(n_times, fs_tf, WINDOW_SEC, OVERLAP_SEC)

                window_samples = int(round(WINDOW_SEC * fs_tf))
                step_samples = int(round(STEP_SEC * fs_tf))
                n_windows = len(window_starts)

                print(f"Window length: {WINDOW_SEC} s")
                print(f"Window overlap: {OVERLAP_SEC} s")
                print(f"Window step: {STEP_SEC} s")
                print(f"Samples per window: {window_samples}")
                print(f"Step samples: {step_samples}")
                print(f"Number of complete windows: {n_windows}")


                # ========================================================
                # OUTPUT DIRECTORY
                # ========================================================

                output_dir = os.path.join(wavelet_output_root, vid, pat)
                os.makedirs(output_dir, exist_ok=True)

                base_name = os.path.basename(wavelet_file)

                if base_name.endswith(".h5"):
                    base_name = base_name[:-3]

                output_file = os.path.join(output_dir, f"{base_name}_log_robust_z_wavelets_{WINDOW_SEC}s_windows.h5")

                if os.path.exists(output_file) and not overwrite:
                    print(f"[SKIP] Output already exists:\n{output_file}")
                    continue


                # ========================================================
                # CREATE OUTPUT FILE
                # ========================================================

                print(f"Creating output file:\n{output_file}")

                with h5py.File(output_file, "w") as h5out:

                    # ====================================================
                    # METADATA
                    # ====================================================

                    h5out.attrs["pat"] = str(pat_file)
                    h5out.attrs["vid"] = str(vid_file)
                    h5out.attrs["run_label"] = str(run_label)
                    h5out.attrs["fs"] = fs_tf
                    h5out.attrs["representation"] = "log10_wavelet_power_robust_zscore_per_channel_frequency"
                    h5out.attrs["log_transform"] = "log10"
                    h5out.attrs["zscore_method"] = "median_MAD"
                    h5out.attrs["robust_sd_scale"] = 1.4826
                    h5out.attrs["zscore_axis"] = "continuous_time"
                    h5out.attrs["window_sec"] = WINDOW_SEC
                    h5out.attrs["overlap_sec"] = OVERLAP_SEC
                    h5out.attrs["step_sec"] = STEP_SEC
                    h5out.attrs["window_samples"] = window_samples
                    h5out.attrs["step_samples"] = step_samples
                    h5out.attrs["n_windows"] = n_windows
                    h5out.attrs["source_wavelet_file"] = wavelet_file


                    # ====================================================
                    # CHANNELS AND FREQUENCIES
                    # ====================================================

                    h5out.create_dataset("labels_ip", data=np.asarray(labels_ip, dtype="S"))
                    h5out.create_dataset("freqs_tf", data=freqs_tf.astype(np.float32))


                    # ====================================================
                    # WINDOW INDICES
                    # ====================================================

                    h5out.create_dataset("window_start_samples", data=window_starts)
                    h5out.create_dataset("window_end_samples", data=window_ends)
                    h5out.create_dataset("window_start_sec", data=(window_starts / fs_tf).astype(np.float32))
                    h5out.create_dataset("window_end_sec", data=(window_ends / fs_tf).astype(np.float32))
                    h5out.create_dataset("window_centers_sec", data=window_centers.astype(np.float32))


                    # ====================================================
                    # CONTINUOUS LOG + Z-SCORED WAVELET DATASET
                    #
                    # shape:
                    # channels x frequencies x continuous time
                    # ====================================================

                    z_dataset = h5out.create_dataset(
                        "pow_tf_log_z",
                        shape=(n_channels, n_freqs, n_times),
                        dtype=np.float32,
                        chunks=(1, min(frequency_batch_size, n_freqs), min(window_samples, n_times)),
                        compression="gzip",
                        compression_opts=4,
                        shuffle=True
                    )


                    # ====================================================
                    # NORMALIZATION PARAMETERS
                    #
                    # shape:
                    # channels x frequencies
                    # ====================================================

                    median_dataset = h5out.create_dataset("robust_median", shape=(n_channels, n_freqs), dtype=np.float32)
                    sd_dataset = h5out.create_dataset("robust_sd", shape=(n_channels, n_freqs), dtype=np.float32)


                    # ====================================================
                    # PROCESS CHANNELS IN BATCHES
                    # ====================================================

                    for ch_start in range(0, n_channels, channel_batch_size):

                        ch_end = min(ch_start + channel_batch_size, n_channels)

                        print(f"Processing channels {ch_start + 1}-{ch_end} of {n_channels}")


                        # =================================================
                        # PROCESS FREQUENCIES IN BATCHES
                        # =================================================

                        for freq_start in range(0, n_freqs, frequency_batch_size):

                            freq_end = min(freq_start + frequency_batch_size, n_freqs)

                            print(f"    Frequencies {freq_start + 1}-{freq_end} of {n_freqs} ({freqs_tf[freq_start]:.1f}-{freqs_tf[freq_end - 1]:.1f} Hz)")


                            # =============================================
                            # LOAD LINEAR WAVELET POWER
                            #
                            # channels x frequencies x continuous time
                            # =============================================

                            tf_linear = tf_dataset[ch_start:ch_end, freq_start:freq_end, :].astype(np.float64)


                            # =============================================
                            # LOG10 TRANSFORM
                            # =============================================

                            tiny = np.finfo(np.float64).tiny
                            tf_log = np.log10(np.maximum(tf_linear, tiny))

                            del tf_linear


                            # =============================================
                            # ROBUST Z-SCORE EACH CHANNEL x FREQUENCY
                            # ACROSS FULL CONTINUOUS RECORDING
                            # =============================================

                            tf_z, robust_median, robust_sd = robust_zscore_tf(tf_log)

                            del tf_log


                            # =============================================
                            # WRITE CONTINUOUS STANDARDIZED DATA TO DISK
                            # =============================================

                            z_dataset[ch_start:ch_end, freq_start:freq_end, :] = tf_z.astype(np.float32)
                            median_dataset[ch_start:ch_end, freq_start:freq_end] = robust_median.astype(np.float32)
                            sd_dataset[ch_start:ch_end, freq_start:freq_end] = robust_sd.astype(np.float32)

                            del tf_z, robust_median, robust_sd

                            h5out.flush()


                print(f"Saved log-transformed, robust-z-scored continuous wavelets to:\n{output_file}")


                # ========================================================
                # VERIFY SAVED FILE
                # ========================================================

                with h5py.File(output_file, "r") as h5check:

                    print("\nSaved datasets:")

                    for key in h5check.keys():
                        print(f"    {key}: {h5check[key].shape}")

                    saved_shape = h5check["pow_tf_log_z"].shape

                    if saved_shape != (n_channels, n_freqs, n_times):
                        raise RuntimeError(f"Saved pow_tf_log_z shape {saved_shape} does not match expected {(n_channels, n_freqs, n_times)}")

                    saved_starts = h5check["window_start_samples"][:]
                    saved_ends = h5check["window_end_samples"][:]

                    if len(saved_starts) != n_windows or len(saved_ends) != n_windows:
                        raise RuntimeError("Saved window indices do not match expected number of windows.")

                    if n_windows > 0 and np.any((saved_ends - saved_starts) != window_samples):
                        raise RuntimeError("One or more saved rolling windows has an incorrect number of samples.")

                print("Output verification passed.")


print("\n" + "=" * 70)
print("ALL REQUESTED RECORDINGS COMPLETE")
print("=" * 70)