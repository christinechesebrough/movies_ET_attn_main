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

## Current state (as of 2026-09-09)

Recreating a previously working pipeline against **newly added and
re-preprocessed data**. The strategy is to extract large continuous wavelets
once, then derive every other needed representation from them.

Active workstream: the wavelet chain (Stages 1–3 below).

---

## Pipeline

```
STAGE 0  preprocessing
         movies_ieeg_preprocess_batch.py
         save_with_bad_chans.py
         label_bad_windows_continous.py       -> bad-window labels
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
```

### The CSV boundary — most important fact in this repo

**Every Stage 4 script reads CSV.** None of them touch `.npz` or HDF5. They are
completely indifferent to the npz -> HDF5 migration.

This means the old analysis scripts are still usable with new data, provided
Stage 3 emits the same CSV schema. The migration did not break Stage 4; it
broke the script that used to produce the CSV.

Expected CSV columns (recovered from Stage 4 usage; <!-- VERIFY -->):

```
Patient · Movie/Video · Session · Run · Entry_ID · Channel · Atlas ·
Y17_Atlas_Region · Window_Start_Sec · Attention_Window_Index ·
Attention_Time_Index · Is_Bad_Window · Is_Good_Window ·
band columns: delta, alpha, beta, gamma, all_gamma, ...
```

`Is_Bad_Window` comes from `label_bad_windows_continous.py` and must be joined
in **at Stage 3**, not later.

Template for writing this CSV: the `to_csv` block in `extract_power_fc.py`
around lines 880–925 already produces the canonical column set.

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

Both script directories are active. <!-- VERIFY: how the eye-tracking branch
connects to the iEEG pipeline is not yet mapped. -->

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
