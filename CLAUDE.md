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

Full pipeline structure — four pipelines, shared front-end, known weaknesses:
**`PIPELINE.md`**. Working backlog with dependencies and ordering: **`TODO.md`**.

## Current state (as of 2026-09-09)

Recreating a previously working pipeline against **newly added and
re-preprocessed data**. The strategy is to extract large continuous wavelets
once, then derive every other needed representation from them.

Active workstream: the wavelet chain (Stages 1–3 below).

### Design: three conditions, hungarian is the bridge

```
inscapes                 abstract, non-narrative
despicable_me_hungarian  narrative visuals, unintelligible speech   <- BRIDGE
despicable_me_english    narrative visuals, intelligible speech
```

`despicable_me_hungarian` was added to bridge the previously analysed
`inscapes` and `despicable_me_english` data.

**Consequence — this drives the whole refactor:** a bridge condition only works
if all three are processed *identically*. English and inscapes were originally
derived through the old Hilbert/bandpass chain; deriving hungarian through the
new wavelet chain would make any three-condition gradient partly an artifact of
method rather than of stimulus.

This is why the pipeline is being rebuilt from the top for all three videos.
**Do not propose backfilling hungarian through the old pipeline** — uniform
derivation is the point.

**The old pipeline is to be PRESERVED, not replaced.** Its outputs are the
reference against which the wavelet results get validated. Do not propose
deleting, rewriting, or "cleaning up" the old `extract_power_*` /
`lowpass_power_to_windows.py` chain.

It also means the existing Tier 3/4 files (english + inscapes, 2026-05-17) are
**reference and validation artifacts, not the target**. They are what the new
outputs get checked against, not what gets extended.

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
This is what Stage 4 actually consumes. The `[0.6 0.6]` in the label column
names is `z_thresh` — a **tunable parameter interpolated into the column name**,
not a hardcoded constant. Changing it renames columns downstream.

#### The attention-label branch (eye-tracking side)

Labels are produced by the eye-tracking chain, not the power chain:

```
eyetracking_process_scripts/
  compute_eye_measures.py        per-patient eye measures
  prePCA_agg_norm.py             aggregate across patients + normalize
        |
analysis_scripts/
  robust_pca_gaze_features.py    robust PCA -> PC1..PC4
        |
  examine_separate_PCs_together.py
        computes group_deviation_mahal, group_dev_mahal_z, mahal_time_z
        two deviation schemes:
            within_subject_dev   -> group_dev_mahal_z
            within_timepoint_dev -> mahal_time_z
        emits per scheme:
            Attention_Label_{scheme}_{z_thresh}
            Internal_HighConf_{scheme}_{z_thresh_int}
            External_HighConf_{scheme}_{z_thresh_ext}
        writes {vid}_features_df_{z_thresh}.csv
               all_subject_attention_counts_both_schemes_{z_thresh}.csv
```

This branch **needs cleanup** (Christine, 2026-09-09) — treat its current
ordering as provisional.

#### Data staleness: hungarian is missing from Tiers 3 and 4

Verified 2026-09-09 on disk:

```
all_power_wide.csv         2026-05-17 16:47   movies: despicable_me_english, inscapes
dme_power_eye_merged.csv   2026-05-17 17:40
ins_power_eye_merged.csv   2026-05-17 17:40
windowed_..._hungarian_*   2026-06-26 10:51   <- 40 days AFTER the aggregation
```

Tier 2 windowed data exists for all three videos (english 17 patients,
hungarian 13, inscapes 15), but **the Tier 3 aggregate contains only
despicable_me_english and inscapes**, and there is no `dmh_power_eye_merged.csv`.

Hungarian is absent from **this branch only**. The old pipeline has two
parallel Stage 4 branches and hungarian completed the other one:

```
BRANCH A  power + eye merge          BRANCH B  rolling FOOOF / oscillatory peaks
  windowed_power_10s/                  rolling_fooof_low_mid_{vid}_26Jun26/
  -> all_power_wide.csv                -> rolling_fooof_aggregated_26Jun26/
  -> *_power_eye_merged.csv            (english 13 pat, hungarian 16 pat)
  (english + inscapes only, May 17)    Jun-Jul 2026 — hungarian COMPLETE here
  -> compare_attn_states_*             -> aggregate_oscillatory_peaks.py
```

So hungarian was pushed through the FOOOF branch but not re-merged into
Branch A's aggregates. Both branches are **old-pipeline outputs to preserve**,
not gaps to backfill — the new wavelet pipeline will re-derive all three
conditions uniformly, and these files serve as validation references (B3).

#### What is genuinely missing from the repo

Two joins are unaccounted for:

1. **Tier 2 -> Tier 3** — whatever builds `all_power_wide.csv`: reshaping
   per-band wide windowed CSVs into one long table with bands as columns and
   the atlas metadata rows lifted into columns.
2. **Tier 3 -> Tier 4** — the join that merges the labeled eye-feature frame
   from `examine_separate_PCs_together.py` onto long power, producing
   `*_power_eye_merged.csv`.

The label *computation* is version controlled; these two *reshape/merge* steps
are not. See TODO B4.

#### Channel metadata — where the atlas rows come from

Each wavelet output directory carries a sibling metadata CSV:

```
wavelet_power_10s/wavelet_{vid}_all_cortContacts_tf/{pat}/
    {pat}_{ses}_{run}_{vid}_channel_metadata.csv
    {pat}_{ses}_{run}_{vid}_wavelet_log_tf.npz
```

`*_channel_metadata.csv` — one row per channel (147 for NS127_02, matching the
147 electrode columns in Tier 1/2):

| column | -> Tier 2 metadata row |
|---|---|
| `label` | the electrode column names |
| `DK_Atlas` | `DK_Atlas_Region` |
| `Y7_Atlas` | `Y7_Atlas_Region` |
| `Y17_Atlas` | `Y17_Atlas_Region` |
| `AparcAseg_Atlas` | `AparcAseg_Atlas_Region` |

**WARNING:** in all 76 wavelet-side `*_channel_metadata.csv` files the
`AparcAseg_Atlas` column is mispopulated — it is a verbatim copy of `Y7_Atlas`
(Yeo-7 network names) rather than FreeSurfer regions. Tier 1/2 power CSVs are
correct. Use `src/channel_metadata.py`, which reads the correspondence sheets
directly, rather than these files.

That covers four of the five rows. The fifth, `network`, comes from
`define_custom_network_atlas.py`, which in the old chain **rewrites the Tier 1
CSV in place** (line 245) to insert it; `lowpass_power_to_windows.py` then just
carries all five rows through. Old chain order:

```
Tier 1 CSV (4 atlas rows)
   -> define_custom_network_atlas.py   adds 'network' row, rewrites in place
   -> lowpass_power_to_windows.py      windows; carries 5 rows through
   -> Tier 2
```

For the wavelet bridge the network-assignment logic
(`define_custom_network_atlas.py` ~lines 348-535) needs applying to
`channel_metadata.csv` instead of to a Tier 1 CSV.

#### Bridge target for the new wavelet pipeline

`wavelet_extract_windows.py` already uses the **same window grid** as
`lowpass_power_to_windows.py` (10 s / 7.5 s / 2.5 s), so the new Stage 3 should
emit **Tier 2**: wide, one file per band, with the five atlas metadata rows
prepended — four joined from `channel_metadata.csv` on `label`, the fifth
derived via the network logic in `define_custom_network_atlas.py`.
Tiers 3 and 4 then proceed unchanged — once their missing producers are
recovered.

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

**Current behaviour (Christine, 2026-09-09): bad windows are INCLUDED in
wavelet extraction and in subsequent norming/windowing.** They are filtered only
later — from statistical grouping in the power pipeline and from plotting in the
spectrogram pipeline. So marking exists; exclusion at extraction does not.
Whether that is right is an open question (PIPELINE.md).

An earlier draft of this file asserted that marking happens before the transform.
That was wrong.

Argument for excluding earlier:
Morlet convolution smears a transient across roughly +/- n_cycles/f seconds, so
masking windows post hoc hides the window the artifact sat in without removing
the contamination that leaked into its neighbours. Cost: manual marking does not
scale, and excised segments create discontinuities the transform can ring on.

Recommended: manual continuous marking as the real cleaning step, automated MAD
detection retained downstream as a QC cross-check.

## The `+ 1e-6` epsilon bug (old bandpass pipeline)

`extract_power_*.py` computed `np.log10(pow_dat_raw + 1e-6)`. That additive
constant is NOT negligible against iEEG amplitudes, and because power follows
1/f the damage scales with frequency. Measured on NS127_02 english, 6 channels:

| band | true amplitude | 1e-6 / signal | dynamic range lost |
|---|---|---|---|
| delta | 1.28e-05 | 0.08 | 9.4% |
| theta | 6.33e-06 | 0.16 | 14.0% |
| alpha | 4.99e-06 | 0.20 | 17.2% |
| beta | 2.92e-06 | 0.34 | 25.6% |
| gamma | 1.37e-06 | 0.73 | **42.8%** |
| **HFA** | 6.28e-07 | **1.59** | **59.4%** |

**For HFA the epsilon exceeds the signal itself.** Roughly 60% of HFA's
dynamic range was destroyed, and 43% of gamma's.

The compression is NONLINEAR - it flattens the low end while barely touching
the high end - so it is not removable by rescaling and downstream z-scoring
does not undo it. It is also channel-dependent, hitting low-amplitude
electrodes hardest, so it distorts spatial patterns as well as effect sizes.

**Any HFA or gamma result from `full_raw_log_power_1Apr26` is attenuated.**

Fixed 2026-09-09 by `np.log10(np.maximum(x, np.finfo(x.dtype).tiny))` - a floor
rather than an addition, so values above it pass through unchanged. Verified:
the re-extracted delta matches `log10(10^old - 1e-6)` to 0.003 log units
(r = 0.99987), confirming the fix changed exactly one thing.

**Note this is the OPPOSITE end of the spectrum from the wavelet grid problem:**

```
epsilon       worst at HFA/gamma  (59%, 43% of dynamic range lost)
wavelet grid  worst at delta      (40% band coverage, 1.2 Hz gap)
```

## Channel metadata: use src/channel_metadata.py

`src/channel_metadata.py` is the canonical accessor (TODO A6). It reads the
electrode correspondence sheets — the authoritative source — and normalizes
atlas labels.

```python
from channel_metadata import load_channel_metadata, as_atlas_rows
meta = load_channel_metadata('NS127_02', channels=data_columns)  # order-aligned
rows = as_atlas_rows(meta)   # the 5-row block Tier 1/2 CSVs carry
```

It raises rather than silently misaligning if a data column has no metadata row.

Background: the Yeo code->name mapping was duplicated across 9 scripts, and
three Y7 vocabularies were in simultaneous use — raw codes (`7Networks_5`),
short names (`Limbic`), long names (`Limbic Network (LN)`). All three are
retained as explicit maps; `long` is the default because it matches the
existing Tier 1/2 CSVs. A fourth, separate scheme assigns networks from
hand-curated DK region lists in `compare_attn_states_*` — not reconciled.

## Wavelet frequency grid — a low-frequency limitation

Stage 1 (`extract_wavelet_hdf5.py`) uses a **linear** frequency grid, 1-151 Hz
in 2 Hz steps (76 bins), with **fixed `n_cycles = 5`**. Consequences:

| band | Hz | wavelet freqs inside it |
|---|---|---|
| delta | 1-3 | **2** (exactly 1.0 and 3.0 - the endpoints; 2 Hz is not sampled) |
| theta | 4-7 | **2** (5.0, 7.0) |
| alpha | 8-13 | 3 |
| beta | 14-30 | 8 |
| gamma | 31-50 | 10 |
| HFA | 51-150 | **50** |

A linear grid puts two thirds of its resolution above 50 Hz and almost none
below 10 Hz. Neural oscillatory bands are logarithmically spaced, so a log grid
is the usual choice. Changing this means re-running Stage 1 over 750 GB, so it
is recorded rather than fixed.

Fixed `n_cycles = 5` also means wavelet duration scales as 5/f:

```
   1 Hz  -> 5.00 s long, 0.40 Hz bandwidth
   3 Hz  -> 1.67 s long, 1.20 Hz bandwidth
  50 Hz  -> 0.10 s long, 20.0 Hz bandwidth
```

At 1 Hz the wavelet spans **5 s**, so a 10 s analysis window holds only ~2
independent estimates and each draws on data +/- 2.5 s beyond its own centre.
Adjacent 10 s windows at delta are therefore smeared into one another well
beyond the nominal 75% overlap.

### Two separate defects, often confused

**Spectral** — is the band measured properly in frequency? Set by the grid.
**Temporal** — how much of a window's value comes from outside it? Set by
`n_cycles/f` relative to the analysis window.

Current extraction (linear 2 Hz, nc=5), and the proposed 0.5-30 Hz log
re-extraction (nc 3 -> 10 -> 5):

| band | spectral coverage | spectral leak | temporal leak (10 s win) |
|---|---|---|---|
| sub-delta 0.5-1 | 40% -> **100%** | 50% -> 45% | 60% (unchanged - intrinsic) |
| delta 1-3 | 40% -> **100%** | 50% -> **22%** | 40% |
| theta 4-7 | 100% | 41% -> 29% | 18% |
| alpha 8-13 | 100% | 44% -> 22% | 12% |
| beta 14-30 | 100% | 32% -> 18% | 3% |

**Delta is usable after re-extraction** - spectrally it becomes the best low
band (22% leak, better than theta). Its 40% temporal leak means delta changes
resolve on a ~15-20 s timescale, not 10 s; state that rather than dropping the
band. Delta is additionally available from the bandpass route at 0.75 s.

**Sub-delta is the band with a real constraint**: 60% temporal leak in a 10 s
window, because `nc=3` at 0.5 Hz is a 6 s wavelet. Fix by windowing that range
at 30 s, not by changing the extraction.

### Full-range re-extraction, if ever done

0.5-151 Hz, ~100 log points, `n_cycles` 3 -> 15 would fix every band:
HFA 50 freqs/30% leak -> 18 freqs/**6%** leak; gamma 46% -> 22%. At nc=5 a
150 Hz wavelet has 60 Hz bandwidth and is barely frequency-specific - the
linear grid's cost at the top mirrors its cost at the bottom.
Cost ~21 GB/recording, ~1.1 TB total. Deferred: gamma and HFA are currently
usable, the low bands were not.

**Interpretation guidance:** delta and theta estimates from the CURRENT grid
are thin and temporally smeared. Treat low-frequency results as coarse. This is a
property of the frequency grid and `n_cycles`, not of the windowing or the
band-averaging.

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

## Paths: use src/paths.py

**Never hardcode absolute paths.** `src/paths.py` resolves them (TODO E2).

```python
from paths import MOVIE_DATA, ANATOMY, CORR_SHEETS, find, wavelet_raw_tf
d = find('wavelet_continuous_z')          # searches every root
s1 = wavelet_raw_tf('despicable_me_english')
```
`python3 src/paths.py` prints what resolved where.

Data is split across two drives, **per resource, not per tree** — there is a
`Movie_data` on both:

```
/media/christine/Samsung/Movie_data     1.9T, 95% FULL, 107G free
    movies_prep_standard        CURRENT (49 patients, newest 2026-09-08)
    full_raw_log_power_*, windowed_power_*, rolling_fooof_*,
    shared_PC_features_*, movies_bad_windows, data/

/media/christine/Data/Movie_data        11T, 6.6T free
    wavelet_*_26Aug26/          Stage 1 raw TF HDF5   750 G
    wavelet_continuous_z/       Stage 2               612 G
    wavelet_band_power/         Stage 2                38 G
    movies_prep_standard        STALE 2024 copy (25 patients) - DO NOT USE

/media/christine/Data/anatomy           FreeSurfer tree + shared_correspondence
```

New wavelet output goes to the Data drive because single files are 9-27 GB and
the set is ~1.4 TB. It cannot fit on Samsung. A `/Volumes/Samsung` ->
`/media/christine/Samsung` rewrite silently breaks both the anatomy and wavelet
paths — resolve per resource instead.

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

- **39 of 55 scripts hardcode absolute paths** (`/Users/christinechesebrough/...`,
  `/Volumes/Samsung/...`, `/media/christine/Samsung/...`). `src/paths.py` now
  exists; migrate scripts to it as they are touched. Seven scripts reference
  `/Volumes/Samsung/anatomy/` — that content is on the **Data** drive.
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
