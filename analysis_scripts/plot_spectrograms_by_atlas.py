#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Config
# ----------------------------
ROOT = "/Volumes/Samsung/Movie_data/HFA_the_present_14Jan26"
GLOB_PATTERN = "**/*_cortical_power_z_wLabels.csv"

ATLAS_ROW = "AparcAseg_Atlas_Region"          # e.g., "DK_Atlas", "DK_Lobe", "Y17_Atlas", "Y7_Atlas", "AparcAseg_Atlas"
N_LABEL_ROWS = 4               # you said head(4)
LABEL_ATLAS_COL = "Atlas"       # column that stores atlas row names
MIN_ELEC_PER_REGION = 1

# If your time is not explicitly present in the file, set FS_HZ to compute seconds from sample index.
# If you *do* have a time column, the script will use it automatically.
FS_HZ = 600                   # e.g., 600 for raw LFP power, or after decimation your plotting fs; set if needed.

# Optional time window (seconds); set to None for full
T_WINDOW = (160, 190.0)      # or None

OUT_DIR = os.path.join(ROOT, "_atlas_group_averages")
os.makedirs(OUT_DIR, exist_ok=True)

FIG_DPI = 300
MAX_YTICKS = 40

from scipy import signal

def smooth_and_decimate_time(mat, fs, smooth_sec=0.05, decim=2):
    """
    mat: (n_regions, n_times)
    smooth_sec: moving-average window in seconds (e.g., 0.05 = 50 ms)
    decim: keep every Nth sample for plotting (e.g., 2 or 4)
    """
    out = mat

    # moving-average smoothing (zero-phase)
    if smooth_sec is not None and smooth_sec > 0:
        win = int(round(smooth_sec * fs))
        win = max(win, 1)
        kernel = np.ones(win) / win
        out = signal.filtfilt(kernel, [1.0], out, axis=1)

    # decimate for plotting (optional)
    if decim is not None and decim > 1:
        out = out[:, ::decim]
        fs = fs / decim

    return out, fs



# ----------------------------
# Plotting
# ----------------------------
def plot_heatmap(t, mat, labels, title, out_png):
    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(
        mat,
        aspect="auto",
        origin="lower",
        extent=[t[0], t[-1], 0, mat.shape[0]],
        interpolation="nearest",
    )

    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Atlas region")

    # --- FIX: center y-ticks on rows ---
    step = max(1, len(labels) // MAX_YTICKS)
    yticks = np.arange(0, len(labels), step) + 0.5

    ax.set_yticks(yticks)
    ax.set_yticklabels(
        [labels[i] for i in range(0, len(labels), step)],
        fontsize=8
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Power (z)")

    plt.tight_layout()
    fig.savefig(out_png, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


# ----------------------------
# Loading
# ----------------------------
def load_power_labels_and_data(path, *, n_label_rows=N_LABEL_ROWS, atlas_col=LABEL_ATLAS_COL):
    """
    Reads CSV where:
      - First n_label_rows are label rows with an 'Atlas' column naming the label type (e.g., DK_Atlas)
      - Remaining rows are numeric time-series (power_z)
      - Electrodes are columns (except metadata columns)
    Returns:
      atlas_rows: df of label rows
      data_rows: df of numeric data rows
      electrode_cols: list of electrode columns
      time: numpy array of seconds (best effort)
    """
    df = pd.read_csv(path, low_memory=False)

    if atlas_col not in df.columns:
        raise ValueError(f"Expected a column named '{atlas_col}' for atlas row names. Found: {list(df.columns)[:20]}...")

    atlas_rows = df.head(n_label_rows).copy()
    data_rows = df.iloc[n_label_rows:].reset_index(drop=True).copy()

    # Identify electrode columns: everything except obvious metadata columns
    # At minimum exclude 'Atlas'. Also exclude common non-electrode columns if present.
    exclude = {atlas_col, "time", "Time", "t", "T", "sample", "Sample", "index", "Index","SubID"}
    electrode_cols = [c for c in df.columns if c not in exclude]

    if not electrode_cols:
        raise ValueError(f"No electrode columns detected after excluding {exclude}.")

    # Coerce numeric data
    data_rows[electrode_cols] = data_rows[electrode_cols].apply(pd.to_numeric, errors="coerce")

    # Time axis:
    # Prefer an explicit time column if present; otherwise sample index -> seconds if FS_HZ set; else samples.
    time_col = None
    for c in ["time", "Time", "t", "T"]:
        if c in df.columns:
            time_col = c
            break

    if time_col is not None:
        t = pd.to_numeric(data_rows[time_col], errors="coerce").to_numpy()
        if np.all(~np.isfinite(t)):
            # fallback
            t = np.arange(len(data_rows), dtype=float)
        time = t
        # If time_col was in exclude set, it is not in electrode_cols; fine.
    else:
        if FS_HZ is None:
            time = np.arange(len(data_rows), dtype=float)  # samples
        else:
            time = np.arange(len(data_rows), dtype=float) / float(FS_HZ)

    return atlas_rows, data_rows, electrode_cols, time


def region_ts_for_recording(atlas_rows, data_rows, electrode_cols, time, atlas_name=ATLAS_ROW, min_elec=MIN_ELEC_PER_REGION):
    """
    Uses atlas_rows (top rows) to map each electrode -> region label for atlas_name,
    then averages data_rows within each region to produce region time series.
    """
    # pick the atlas label row
    match = atlas_rows[atlas_rows[LABEL_ATLAS_COL].astype(str).str.strip() == str(atlas_name).strip()]
    if match.empty:
        raise ValueError(
            f"atlas '{atlas_name}' not found in top label rows. "
            f"Available: {atlas_rows[LABEL_ATLAS_COL].astype(str).tolist()}"
        )
    atlas_row = match.iloc[0]

    # atlas_row[electrode] should be region label
    labels = {}
    for e in electrode_cols:
        lab = atlas_row.get(e, None)
        if lab is None or (isinstance(lab, float) and np.isnan(lab)):
            lab = "Unknown"
        lab = str(lab).strip()
        if lab == "" or lab.lower() == "nan":
            lab = "Unknown"
        labels[e] = lab

    # group electrodes -> region
    region_to_elecs = {}
    for e, lab in labels.items():
        region_to_elecs.setdefault(lab, []).append(e)

    # compute region mean over time
    region_ts = {}
    X = data_rows[electrode_cols].to_numpy(float)  # (n_times, n_elecs)
    # transpose for easier indexing by elecs
    data_by_e = {e: data_rows[e].to_numpy(float) for e in electrode_cols}

    for region, elecs in region_to_elecs.items():
        if len(elecs) < min_elec:
            continue
        Y = np.vstack([data_by_e[e] for e in elecs])  # (n_elecs, n_times)
        region_ts[region] = np.nanmean(Y, axis=0)

    return region_ts, time


# ----------------------------
# Time alignment across recordings
# ----------------------------
def build_common_time_grid(times, t_window=None):
    starts = [float(t[0]) for t in times]
    ends = [float(t[-1]) for t in times]

    t0 = max(starts)
    t1 = min(ends)
    if t_window is not None:
        t0 = max(t0, float(t_window[0]))
        t1 = min(t1, float(t_window[1]))

    if t1 <= t0:
        raise ValueError(f"No overlapping time range across recordings. t0={t0}, t1={t1}")

    dts = []
    for t in times:
        if len(t) >= 3:
            dt = np.diff(t)
            dt = dt[np.isfinite(dt)]
            if len(dt):
                dts.append(np.nanmedian(dt))

    dt_common = float(np.nanmedian(dts)) if dts else 1.0
    if not np.isfinite(dt_common) or dt_common <= 0:
        dt_common = 1.0

    n = int(np.floor((t1 - t0) / dt_common)) + 1
    return t0 + np.arange(n) * dt_common


def interp_to_grid(t_src, y_src, t_dst):
    t_src = np.asarray(t_src, float)
    y_src = np.asarray(y_src, float)
    m = np.isfinite(t_src) & np.isfinite(y_src)
    if m.sum() < 2:
        return np.full_like(t_dst, np.nan, dtype=float)
    return np.interp(t_dst, t_src[m], y_src[m])


# ----------------------------
# Main
# ----------------------------
def main():
    paths = sorted(glob.glob(os.path.join(ROOT, GLOB_PATTERN)))
    print(f"Found {len(paths)} files matching pattern under:\n  {ROOT}\n  {GLOB_PATTERN}")
    if not paths:
        raise SystemExit("No files found. Check ROOT and GLOB_PATTERN.")

    recs = []
    for p in paths:
        try:
            atlas_rows, data_rows, elec_cols, time = load_power_labels_and_data(p)

            region_ts, time = region_ts_for_recording(
                atlas_rows, data_rows, elec_cols, time,
                atlas_name=ATLAS_ROW, min_elec=MIN_ELEC_PER_REGION
            )

            if not region_ts:
                print(f"Skipping (no regions after grouping): {p}")
                continue

            fn = os.path.basename(p)
            rec_id = fn.replace("_HFA_cortical_power_z_wLabels.csv", "").replace(".csv", "")
            recs.append({"id": rec_id, "path": p, "time": time, "region_ts": region_ts})
        except Exception as e:
            print(f"ERROR loading {p}\n  {e}")
            continue

    print(f"Loaded {len(recs)} recordings successfully.")
    if not recs:
        raise SystemExit("No recordings loaded successfully.")

    # time grid
    t_common = build_common_time_grid([r["time"] for r in recs], t_window=T_WINDOW)
    print(f"Common time grid: {t_common[0]:.3f} to {t_common[-1]:.3f} s, n={len(t_common)}")

    # union of regions
    all_regions = sorted(set().union(*[set(r["region_ts"].keys()) for r in recs]))
    print(f"Total regions (union): {len(all_regions)}")

    mats = []
    for r in recs:
        mat = np.full((len(all_regions), len(t_common)), np.nan, dtype=float)
        for i, region in enumerate(all_regions):
            if region in r["region_ts"]:
                mat[i] = interp_to_grid(r["time"], r["region_ts"][region], t_common)
        mats.append(mat)

    mats = np.stack(mats, axis=0)          # (n_rec, n_regions, n_times)
    mat_mean = np.nanmean(mats, axis=0)    # (n_regions, n_times)
    
    # ---- smoothing for visualization ----
    SMOOTH_SEC = 0.5   # 50 ms; try 0.1 if still too jagged
    DECIM = 6           # try 4 if you want lighter files / smoother appearance
    
    mat_plot, fs_plot = smooth_and_decimate_time(mat_mean, FS_HZ, smooth_sec=SMOOTH_SEC, decim=DECIM)
    t_plot = np.arange(mat_plot.shape[1]) / fs_plot + t_common[0]

    n_contrib = np.sum(np.isfinite(mats), axis=0)

    # save
    tag = f"group_{ATLAS_ROW}"
    if T_WINDOW is not None:
        tag += f"_{int(T_WINDOW[0])}-{int(T_WINDOW[1])}s"

    out_mean_csv = os.path.join(OUT_DIR, f"{tag}_region_time_mean.csv")
    out_n_csv = os.path.join(OUT_DIR, f"{tag}_region_time_ncontrib.csv")
    out_png = os.path.join(OUT_DIR, f"{tag}_heatmap.png")
    out_manifest = os.path.join(OUT_DIR, f"{tag}_recordings_used.csv")

    pd.DataFrame(mat_mean, index=all_regions, columns=t_common).to_csv(out_mean_csv)
    pd.DataFrame(n_contrib, index=all_regions, columns=t_common).to_csv(out_n_csv)
    pd.DataFrame({
        "recording_id": [r["id"] for r in recs],
        "path": [r["path"] for r in recs]
    }).to_csv(out_manifest, index=False)

    plot_heatmap(
        t_plot,
        mat_plot,
        all_regions,
        title=f"Mean power_z over time grouped by {ATLAS_ROW} (n_rec={len(recs)}), smooth={SMOOTH_SEC}s, decim={DECIM}",
        out_png=out_png
    )


    print("Saved:")
    print(" ", out_mean_csv)
    print(" ", out_n_csv)
    print(" ", out_png)
    print(" ", out_manifest)


if __name__ == "__main__":
    main()
