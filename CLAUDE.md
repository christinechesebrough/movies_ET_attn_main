# movies_ET_attn_main

iEEG + eye-tracking analysis of attention states during movie watching.
Research code for an in-progress paper.

---

## How Christine works — read this first

- **Scripts are run interactively in Spyder, chunk by chunk.** They are not
  batch jobs and not a CLI. Top-level executable code is intentional.
- **Do NOT refactor scripts into `main()` functions, argparse, or modules.**
  That breaks the chunk-by-chunk workflow. Keep configuration as top-level
  variables near the top of the file.
- **`vids = [...]` and patient lists are hand-edited knobs**, not bugs. A
  script processing one video is normal, not a scoping error.
- Reusable functions live in `src/`. Scripts import them by bare module name
  (`from eeg_preproc_helpers import ...`) via `sys.path` manipulation.

Working backlog with dependencies and ordering: **`TODO.md`**.

## Current state (as of 2026-09-09)

Recreating a previously working pipeline against **newly added and
re-preprocessed data**. The strategy is to extract large continuous wavelets
once, then derive every other needed representation from them.

Active workstream: the wavelet chain (Stages 1–3 below).

---

## Pipeline

```
STAGE 0  preprocessing
         movies_ieeg_preprocess_batch.py      -> handles bad channels itself
         label_bad_windows_continous.py       -> manual bad-window marking on
                                                 continuous .fif, BEFORE the
                                                 wavelet transform
         [save_with_bad_chans.py]             -> backfill only, see below
              |
STAGE 1  master time-frequency representation
         extract_wavelet_hdf5.py
         -> HDF5: pow_tf_dat (ch x freq x time), freqs_tf, labels_ip, t_tf
            attrs: pat, run_label, vid, output, pow_type, fs_lfp, fs_tf,
                   n_cycles, n_cycles_mode
              |
STAGE 2  derived representations (both read Stage 1 HDF5)
         bandpass_from_wavelet.py
         -> HDF5: pow_dat, pow_dat_z_windowed, frequency_bins, freqs_used,
                  labels_ip, robust_median, robust_sd
         wavelet_extract_windows.py
         -> HDF5: pow_tf_log_z, freqs_tf, labels_ip, robust_median, robust_sd,
                  window_start_samples/_sec, window_end_samples/_sec,
                  window_centers_sec
              |
STAGE 3  *** MISSING: bridge to long-format CSV ***
              |
STAGE 4  analyses (all read CSV via pd.read_csv)
         extract_all_fooof.py, aggregate_oscillatory_peaks.py
         robust_pca_gaze_features.py, examining_shared_PC_features.py
         compare_attn_states_*.py
         find_artifactual_windows.py          -> automated MAD detector, runs on
                                                 extracted CSV; QC cross-check
```

### The CSV boundary — most important fact in this repo

**Every Stage 4 script reads CSV.** None touch `.npz` or HDF5. They are
indifferent to the npz -> HDF5 migration. The migration did not break Stage 4;
it broke the chain that produced the CSVs.

Verified against real files on 2026-09-09 under
`/media/christine/Samsung/Movie_data/`. **Four tiers, not one:**

**Tier 1 — full-resolution power, WIDE** (~1 GB per file)
`full_raw_log_power_1Apr26/power_log_{band}_{vid}_{date}/{pat}/`
`{pat}_{vid}_{run}_{band}_{ref}_power_log.csv`
```
cols: SubID, Atlas, <one column per electrode: RDa3, RDa4, ...>   (149 cols)
rows 1-4 : atlas metadata, keyed by the `Atlas` column —
           DK_Atlas_Region, Y7_Atlas_Region, Y17_Atlas_Region, AparcAseg_Atlas_Region
rows 5+  : timepoints @ 600 Hz, `Atlas` blank   (359,428 rows ~= 10 min)
```
One file **per band** — bands are separate directory trees, NOT columns.
Bands: delta, theta, alpha, beta, gamma, HFA. `ref` = `cortical`.

**Tier 2 — windowed, WIDE** (~676 KB; produced by `lowpass_power_to_windows.py`)
`windowed_power_10s/windowed_{normed|unnormed}_power_log_{vid}_{band}_1Apr26/{pat}/`
`{pat}_{run}_{vid}_{band}_power_log_{z}_rolling_avg.csv`
```
same wide shape, plus an `Unnamed: 0` index column       (150 cols, ~242 rows)
rows 1-5 : atlas metadata — the four above PLUS a `network` row (custom network)
rows 6+  : windows (10 s window, 7.5 s overlap, 2.5 s step @ 600 Hz)
```
Note the extra `network` metadata row that Tier 1 lacks.

**Tier 3 — aggregated, LONG** (`windowed_power_10s/all_power_wide.csv`, 333 MB)
```
pat_base, run, movie, timepoint, electrode, elec_id,
dk_region, y7_network, y17_network, aparc_aseg_region, custom_network,
alpha_power_log, beta_power_log, gamma_power_log, HFA_power_log,
theta_power_log, delta_power_log                          (18 cols)
```
Bands become **columns** here; atlas metadata rows become **columns**.
One row per (patient, run, movie, timepoint, electrode).

**Tier 4 — power + eye merged** (`dme_power_eye_merged.csv`,
`ins_power_eye_merged.csv`; 618 MB; identical columns)
```
Tier 3 columns, plus:
  patient_run
  eye features : Saccade_Rate, Vergence, Vergence_Std, Abs_Vergence,
                 Saccade_Dispersion, Saccade_Dispersion_Std, Blink_Rate,
                 Blink_Duration, Pupil_Avg, Pupil_Std, ISC
  PCs          : PC1..PC4, PC1_z
  group refs   : *_groupmean for each eye feature
  deviation    : group_deviation_mahal, group_dev_mahal_z, mahal_time_z,
                 group_dev_z, group_dev_time_z
  labels       : Attention_Label_within_subject_dev_[0.6 0.6]
                 Internal_HighConf_within_subject_dev_0.6
                 External_HighConf_within_subject_dev_0.6
                 Attention_Label_within_timepoint_dev_[0.6 0.6]
                 Internal_HighConf_within_timepoint_dev_0.6
                 External_HighConf_within_timepoint_dev_0.6   (55 cols)
```
This is what Stage 4 actually consumes. **`0.6` is a hard-coded threshold
baked into the column names** — changing it renames columns downstream.

#### Two producers are MISSING from this repo

Nothing in the repo writes `all_power_wide.csv` (Tier 2 -> 3) or
`*_power_eye_merged.csv` (Tier 3 -> 4, including all attention labels and the
Mahalanobis group-deviation measures). Nine Stage 4 scripts *read* these
columns; none *create* them. Those steps were likely run interactively or live
outside this repo. **Recovering or rewriting them is required for
reproducibility** — see TODO D2.

Related eye-side producers that ARE present:
`eyetracking_process_scripts/prePCA_agg_norm.py` (aggregate+normalize eye
features) and `analysis_scripts/robust_pca_gaze_features.py` (PCs).

#### Bridge target for the new wavelet pipeline

`wavelet_extract_windows.py` already uses the **same window grid** as
`lowpass_power_to_windows.py` (10 s / 7.5 s / 2.5 s), so the new Stage 3 should
emit **Tier 2**: wide, one file per band, with the five atlas metadata rows
prepended. Tiers 3 and 4 then proceed unchanged — once their missing producers
are recovered.

Earlier drafts of this file claimed a single long CSV with `Is_Bad_Window` /
`Window_Start_Sec` / `Attention_Window_Index` columns. **That was inferred and
is wrong** — no such columns exist in the real files.

### Vocabulary drift — three names for the same things

| Old (`extract_power_es.py`, npz) | Stage 1 wavelet HDF5 | Stage 2 band HDF5 |
|---|---|---|
| `pow_ip`    | `pow_tf_dat` | `pow_dat` / `pow_dat_z_windowed` |
| `freq_bins` | `freqs_tf`   | `frequency_bins` / `freqs_used` |
| `labels`    | `labels_ip`  | `labels_ip` |
| `t_lfp`     | `t_tf`       | via `window_*_sec` |
| `fs_lfp`    | attr `fs_tf` | attr `fs` |

Structurally identical, three vocabularies. Standardize only *after* Stage 3 is
validated — otherwise renames and numerical differences get debugged together.

---

## Bad channels and bad windows

**Bad channels** are handled inside `movies_ieeg_preprocess_batch.py`
(loads `bad_channels_file`, sets `info['bads']`, supports manual marking).

`save_with_bad_chans.py` is **not a pipeline stage.** It reconciles legacy data:
finds `*bad_channels*.txt`, picks the most recent, merges into the FIF's
`info['bads']`. Use only to backfill older recordings. Note it also carries
unrelated audio-export code (`scipy.io.wavfile.write`, `audio_dir`) — that is
the origin of the `.wav` files in this repo.

**Bad windows have two tools at different pipeline positions.** They are
complementary, not alternatives:

| | `find_artifactual_windows.py` | `label_bad_windows_continous.py` |
|---|---|---|
| reads | CSV of extracted LFP | continuous `.fif` |
| method | automated (`median_abs_deviation`) | manual marking |
| writes | `mask`, `mask_padded`, `bad_overlap_sec` | `bad_channels` txt + QC CSV |
| position | after extraction | **before** extraction |

Marking before the wavelet transform is the more defensible default here:
Morlet convolution smears a transient across roughly +/- n_cycles/f seconds, so
masking windows post hoc hides the window the artifact sat in without removing
the contamination that leaked into its neighbours. Cost: manual marking does not
scale, and excised segments create discontinuities the transform can ring on.

Recommended: manual continuous marking as the real cleaning step, automated MAD
detection retained downstream as a QC cross-check.

## Conventions

### Script docstring contract

`bandpass_from_wavelet.py` and `wavelet_extract_windows.py` define the house
style. Every pipeline script should carry a header stating inputs, outputs, and
**what it deliberately does not do**:

```python
"""
One-line purpose.

Input HDF5 structure expected:
    pow_tf_dat : channels x frequencies x time
    freqs_tf   : wavelet frequencies
    labels_ip  : channel labels

Expected HDF5 attributes:
    pat, run_label, vid, fs_tf

Processing:
    1. ...

No temporal downsampling, smoothing, z-scoring, or windowing is done here.
"""
```

The "does not do" line is the most useful sentence for reasoning about stage
order. Currently present in 2 of 43 scripts.

The pipeline map above should be **derived from these headers**, not maintained
separately, so it cannot silently drift from the code.

---

## Repo layout

```
analysis_scripts/          iEEG/EEG pipeline + analyses (43 scripts, active)
eyetracking_process_scripts/  eye-tracking pipeline (8 scripts, active)
src/                       reusable helpers (eeg_preproc_helpers, eye_helpers)
docs/                      Sphinx docs (docs/_build/ is gitignored)
```

Both script directories are active. The eye-tracking branch joins the iEEG
pipeline at **Tier 4** (`*_power_eye_merged.csv`), where eye features, PCs and
Mahalanobis group deviation are merged onto long-format windowed power.

## Git setup

- `origin` -> personal repo (`christinechesebrough/movies_ET_attn_main`) — **work here**
- `lab` -> `IEEG/movies_ET_attn_main` — do not push without asking

Lab alignment is deliberately deferred until the repo is in better shape.
`lab/max_temp` is a colleague's branch; **alignment with it is not a goal** —
do not propose merging it or preserving compatibility with its file naming.

`lab/master` contains `remove_single_sample_artifact()`, which is **not** in the
local `src/eeg_preproc_helpers.py`. It overlaps functionally with local
`detect_spikes_*` / `interpolate_spikes*`. Deciding which spike handling is
canonical is an open scientific question. <!-- VERIFY -->

## Known issues

- **39 of 55 scripts hardcode absolute paths**, split across two machines
  (`/Users/christinechesebrough/...` Mac, `/media/christine/Samsung/...` Linux,
  plus a stale `/Volumes/Samsung`). Only urgent if running off this laptop.
- `label_bad_windows_continous.py` is misspelled (missing `u` in
  "continuous"). Worth renaming before it gets imported or referenced widely.
- `review_power` has **no file extension** and no `.py` twin — it is the only
  copy of that script (1134 lines).
- Signatures of `plot_power_spectra` and `plot_psd_batched` in
  `src/eeg_preproc_helpers.py` gained parameters; callers using the old
  argument order may break. <!-- VERIFY -->
- Dated/ad-hoc filenames (`compute_norm_eye_features_4Jan26.py`,
  `compare_attn_states_heatmap_fromPCs_MATRIX_21Apr25.py`) — unclear which are
  current. <!-- VERIFY -->
- 4 near-sibling preprocessing scripts (`movie_ieeg_preprocess{,_es,_wm_bip}.py`,
  `movies_ieeg_preprocess_batch.py`), ~1400 lines each. **Do not propose
  consolidating these** until the paper is out.

## Deferred until after the paper

Tests, CI, packaging, consolidating the preprocessing siblings, unifying the
three vocabularies, README rewrite (the current README describes a `scripts/`,
`config/`, `tests/` layout that does not exist).
