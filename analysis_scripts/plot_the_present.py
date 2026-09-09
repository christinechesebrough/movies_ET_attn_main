#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob, re
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt

base_dir = '/Volumes/Samsung/Movie_data'
recon_dir = '/Volumes/Samsung/anatomy'
img_dir = '/Volumes/Samsung/Movie_data/the_present_figures'
os.makedirs(img_dir, exist_ok=True)

# ----------------------------
# Plotting
# ----------------------------

def make_atlas_map(corr, pat_base, labels_ip, atlas_col="DK_Atlas"):
    atlas_df = corr.rename(columns={"SubID": "pat_base", "Contact": "electrode"}).copy()
    atlas_df["pat_base"] = atlas_df["pat_base"].astype(str).str.strip()
    atlas_df["electrode"] = atlas_df["electrode"].astype(str).str.strip()

    # keep only relevant columns (plus hem if you want)
    keep_cols = ["pat_base", "electrode", "Hem", "DK_Atlas", "DK_Lobe", "Y7_Atlas", "Y17_Atlas", "AparcAseg_Atlas"]
    keep_cols = [c for c in keep_cols if c in atlas_df.columns]
    atlas_df = atlas_df[keep_cols]

    # filter to this patient and the channels we’re plotting
    lab = pd.DataFrame({"electrode": [str(x).strip() for x in labels_ip]})
    lab["pat_base"] = str(pat_base).strip()

    lab = lab.merge(atlas_df, on=["pat_base", "electrode"], how="left")

    # standardize hem if present
    if "Hem" in lab.columns:
        lab["hem"] = lab["Hem"].astype(str).str.upper().str.strip()
        lab.loc[lab["hem"].isin(["LEFT","LH"]), "hem"] = "L"
        lab.loc[lab["hem"].isin(["RIGHT","RH"]), "hem"] = "R"
    else:
        lab["hem"] = ""

    # choose atlas label
    if atlas_col not in lab.columns:
        raise ValueError(f"atlas_col='{atlas_col}' not found. Available: {list(lab.columns)}")
    lab["atlas_label"] = lab[atlas_col].astype("string").fillna("Unknown").str.strip()

    return lab


def plot_heatmap(t, pow_dat, labels, title="Power heatmap"):
    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(
        pow_dat,
        aspect='auto',
        origin='lower',
        extent=[t[0], t[-1], 0, pow_dat.shape[0]],
        interpolation='nearest'
    )
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channel index")
    # Optional: label every Nth channel to avoid clutter
    step = max(1, pow_dat.shape[0] // 20)
    ax.set_yticks(np.arange(0, pow_dat.shape[0], step))
    ax.set_yticklabels([labels[i] for i in range(0, pow_dat.shape[0], step)], fontsize=8)
    fig.colorbar(im, ax=ax, label="Power (env / log / z)")
    plt.tight_layout()
    return fig

def plot_fsavg_inf_for_tag(
    vis_df, *, tag, value_col="Zlike",
    recon_dir=None, out_dir=None,
    views=("lateral","medial","ventral"),
    vlim=3.0, scale_factor=0.4
):
    if recon_dir is None or out_dir is None:
        raise ValueError("Provide recon_dir and out_dir.")

    d = vis_df[vis_df["tag"] == tag].copy()

    # Ensure score exists
    if value_col not in d.columns:
        print(f"Skipping {tag}: missing value_col '{value_col}'")
        return

    d = d.dropna(subset=["x","y","z",value_col,"hem"])
    if d.empty:
        print(f"Skipping {tag}: no data after filters")
        return

    coords = d[["x","y","z"]].to_numpy(float)
    hem = d["hem"].astype(str).str.upper().to_numpy()
    vals = d[value_col].to_numpy(float)

    vmin, vmax = -float(vlim), float(vlim)
    vals_clip = np.clip(vals, vmin, vmax)
    cmap = plt.colormaps["coolwarm"]

    def add_points(brain, hemi_label):
        sel = (hem == hemi_label)
        if not np.any(sel):
            return
        normed = (vals_clip[sel] - vmin) / (vmax - vmin + 1e-12)
        colors = cmap(normed)[:, :3]
        for xyz, col in zip(coords[sel], colors):
            brain.add_foci(xyz[None, :], scale_factor=scale_factor, color=col)

    # LH
    fig3d = mne.viz.create_3d_figure((1000, 1000), show=True, bgcolor="white")
    brain = mne.viz.Brain("fsaverage", hemi="lh", subjects_dir=recon_dir,
                          surf="inflated", figure=fig3d, cortex="classic",
                          background="white", alpha=1)
    add_points(brain, "L")
    for v in views:
        brain.show_view(v)
        brain.save_image(os.path.join(out_dir, f"{tag}_{value_col}_lh_{v}.png"))
    mne.viz.close_all_3d_figures()

    # RH
    fig3d = mne.viz.create_3d_figure((1000, 1000), show=True, bgcolor="white")
    brain = mne.viz.Brain("fsaverage", hemi="rh", subjects_dir=recon_dir,
                          surf="inflated", figure=fig3d, cortex="classic",
                          background="white", alpha=1)
    add_points(brain, "R")
    for v in views:
        brain.show_view(v)
        brain.save_image(os.path.join(out_dir, f"{tag}_{value_col}_rh_{v}.png"))
    mne.viz.close_all_3d_figures()

    print("Saved:", tag)

# ----------------------------
# Loading / tagging
# ----------------------------
def parse_tag_from_filename(path):
    fn = os.path.basename(path).replace(".csv", "")
    # remove redundant token if present
    fn = fn.replace("_HFA_HFA_", "_HFA_")
    return fn

def load_zdiff_csvs(csv_paths):
    dfs = []
    for p in csv_paths:
        df = pd.read_csv(p)

        # Standardize electrode column
        if "electrode" not in df.columns:
            if "Channel" in df.columns:
                df = df.rename(columns={"Channel": "electrode"})
            elif "Contact" in df.columns:
                df = df.rename(columns={"Contact": "electrode"})
            else:
                raise ValueError(f"No electrode column in {p}")

        # Standardize patient id
        if "pat_base" not in df.columns:
            if "Patient" in df.columns:
                df = df.rename(columns={"Patient": "pat_base"})
            else:
                m = re.match(r"(NS\d+(?:_\d+)?)", os.path.basename(p))
                df["pat_base"] = m.group(1) if m else "UNKNOWN"

        df["pat_base"] = df["pat_base"].astype(str).str.strip()
        df["electrode"] = df["electrode"].astype(str).str.strip()

        df["source_csv"] = p
        df["tag"] = parse_tag_from_filename(p)

        dfs.append(df)

    if not dfs:
        raise ValueError("No CSVs loaded. Check your glob pattern and root path.")
    return pd.concat(dfs, ignore_index=True)


def make_atlas_map(corr, pat_base, labels_ip, atlas_col="DK_Atlas"):
    atlas_df = corr.rename(columns={"SubID": "pat_base", "Contact": "electrode"}).copy()
    atlas_df["pat_base"] = atlas_df["pat_base"].astype(str).str.strip()
    atlas_df["electrode"] = atlas_df["electrode"].astype(str).str.strip()

    # keep only relevant columns (plus hem if you want)
    keep_cols = ["pat_base", "electrode", "Hem", "DK_Atlas", "DK_Lobe", "Y7_Atlas", "Y17_Atlas", "AparcAseg_Atlas"]
    keep_cols = [c for c in keep_cols if c in atlas_df.columns]
    atlas_df = atlas_df[keep_cols]

    # filter to this patient and the channels we’re plotting
    lab = pd.DataFrame({"electrode": [str(x).strip() for x in labels_ip]})
    lab["pat_base"] = str(pat_base).strip()

    lab = lab.merge(atlas_df, on=["pat_base", "electrode"], how="left")

    # standardize hem if present
    if "Hem" in lab.columns:
        lab["hem"] = lab["Hem"].astype(str).str.upper().str.strip()
        lab.loc[lab["hem"].isin(["LEFT","LH"]), "hem"] = "L"
        lab.loc[lab["hem"].isin(["RIGHT","RH"]), "hem"] = "R"
    else:
        lab["hem"] = ""

    # choose atlas label
    if atlas_col not in lab.columns:
        raise ValueError(f"atlas_col='{atlas_col}' not found. Available: {list(lab.columns)}")
    lab["atlas_label"] = lab[atlas_col].astype("string").fillna("Unknown").str.strip()

    return lab

def reorder_by_atlas(pow_plot, labels_ip, atlas_map, *, group_cols=("atlas_label", "hem", "electrode")):
    # atlas_map has one row per electrode in labels_ip
    atlas_map = atlas_map.copy()

    # sort electrodes by atlas label (then hem, then name)
    atlas_map = atlas_map.sort_values(list(group_cols)).reset_index(drop=True)

    # reorder power rows
    idx = [labels_ip.index(e) for e in atlas_map["electrode"].tolist()]
    pow_re = pow_plot[idx, :]

    # y labels
    ylab = (atlas_map["atlas_label"] + " | " + atlas_map["electrode"]).tolist()

    return pow_re, ylab, atlas_map

#%%
# ----------------------------
# Main
# ----------------------------


# root = "/Volumes/Samsung/Movie_data/HFA_the_present_14Jan26"
# csv_paths = sorted(glob.glob(os.path.join(root, "**", "*_increase_100_125_vs_164_185.csv"), recursive=True))

root = "/Volumes/Samsung/Movie_data/HFA_the_present_14Jan26/NS135"
csv_paths = sorted(glob.glob(os.path.join(root,"*_increase_100_125_vs_164_185.csv"), recursive=True))

print(f"Found {len(csv_paths)} CSVs")

electrodes_df = load_zdiff_csvs(csv_paths)
print("Loaded rows:", electrodes_df.shape)
print(electrodes_df[["pat_base","tag"]].drop_duplicates().head())

# Coordinates
master_path = "/Volumes/Samsung/anatomy/shared_correspondence/movie_subs_master_updated.csv"
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
coords_df = coords_df.dropna(subset=["x","y","z"]).drop_duplicates(subset=["pat_base","electrode"], keep="first")

# Merge
vis_df = electrodes_df.merge(coords_df, how="inner", on=["pat_base","electrode"])
print("vis_df:", vis_df.shape)
print(f"merge_rate rows: {vis_df.shape[0] / max(1, electrodes_df.shape[0]):.1%}")

# Optional: report missing coords
miss = electrodes_df.merge(coords_df, on=["pat_base","electrode"], how="left", indicator=True)
print(miss["_merge"].value_counts())
print(miss.loc[miss["_merge"]=="left_only", ["pat_base","electrode","tag"]].head(30))

# Choose score column
value_col = "Zlike"  # or "Diff"

# Sanity: ensure the score column exists in at least some rows
if value_col not in vis_df.columns:
    raise ValueError(f"value_col='{value_col}' not found. Available columns include: {list(vis_df.columns)}")

# Plot each recording (tag)
unique_tags = sorted(vis_df["tag"].dropna().unique())
print(f"Plotting {len(unique_tags)} tags...")

for tag in unique_tags:
    plot_fsavg_inf_for_tag(
        vis_df,
        tag=tag,
        value_col=value_col,
        recon_dir=recon_dir,
        out_dir=img_dir,
        vlim=.3
    )
    
    
