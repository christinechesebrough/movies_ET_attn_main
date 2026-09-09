# TODO

Working backlog. Organizing principle: **settle everything upstream of Stage 1
before committing to the final wavelet extraction.** Re-extracting continuous
wavelets over all subjects is the most expensive operation in this pipeline, so
decisions that change its input are the ones that must be made first.

Status: `[ ]` open · `[~]` in progress · `[x]` done · `[?]` needs a decision

---

## A. Upstream decisions — settle BEFORE the final extraction run

These change what goes into Stage 1. Getting them wrong means re-extracting.

- [?] **A1. Spike / artifact handling: pick one.**
  `lab/master` has `remove_single_sample_artifact()`; local `src/` has
  `detect_spikes_*` / `interpolate_spikes*`. Both target single-sample
  transients. Decide which is canonical, or how they compose.
  *Scientific decision, not a merge decision.*

- [ ] **A2. Hybrid bad-window marking.**
  Run the MAD detector on continuous `.fif` to *propose* windows, then confirm
  or reject manually. Gets reproducibility (a stated threshold survives adding
  subjects) plus judgment. Modest change to `label_bad_windows_continous.py`
  rather than a new script — reuse the detection logic from
  `find_artifactual_windows.py`, which currently runs post-extraction.

- [ ] **A3. Confirm bad channels are fully handled by
  `movies_ieeg_preprocess_batch.py`** for the newly re-preprocessed data, so
  `save_with_bad_chans.py` is genuinely backfill-only. Christine believes this
  is true; not yet verified against the new data.

## B. Critical path — completing the pipeline

Strictly ordered; each blocks the next.

- [x] **B1. Resolve the CSV schema.** DONE 2026-09-09, verified against real
  files. Result: **four tiers**, not one — see CLAUDE.md. The previously
  documented long format with `Is_Bad_Window` / `Window_Start_Sec` columns was
  inferred and wrong; no such columns exist.

- [ ] **B2. Write the Stage 3 bridge** (`wavelet_windows_to_csv.py`).
  Reads `wavelet_extract_windows.py` HDF5 → emits **Tier 2**: wide CSV, one
  file per band, electrodes as columns, five atlas metadata rows prepended
  (`DK_Atlas_Region`, `Y7_Atlas_Region`, `Y17_Atlas_Region`,
  `AparcAseg_Atlas_Region`, `network`).
  Template: `lowpass_power_to_windows.py` (~lines 290–335) — it already writes
  exactly this format and uses the same 10 s / 7.5 s / 2.5 s window grid the
  wavelet scripts use.
  Open question: where do the atlas/network row labels come from when the
  source is HDF5 rather than a Tier 1 CSV that already carried them?

- [ ] **B3. Validate new bands against old bandpass power.**
  Run B2 on one patient/movie previously analysed with `extract_power_*`, and
  diff the CSVs. If band power agrees within tolerance, all of Stage 4 is
  certified against new data for free.
  **Highest-value single step in this list.** It is also a real scientific
  check: wavelet-derived bands and Hilbert/bandpass power can differ
  legitimately because the filters differ — a mismatch is not automatically
  a bug, and needs interpreting rather than "fixing".
  *Blocked by B2.*

- [ ] **B4. Recover the two MISSING producers.** Nothing in this repo writes
  `all_power_wide.csv` (Tier 2 -> 3 aggregation) or `*_power_eye_merged.csv`
  (Tier 3 -> 4 merge, which creates every attention label and the Mahalanobis
  group-deviation measures). Nine Stage 4 scripts read those columns; none
  create them. They were run interactively or live outside this repo.
  **This is a reproducibility hole, not just a convenience gap** — the paper's
  dependent measures are produced by code that is not version controlled.
  Note `0.6` is hard-coded into the label column names.
  *Blocks B5. Independent of B2/B3 — can start now.*

- [ ] **B5. Run Stage 4 unchanged** on the new data.
  *Blocked by B3 and B4.*

## C. Correctness fixes — small, do when convenient

- [ ] **C1. `plot_power_spectra` / `plot_psd_batched` signature change.**
  Both gained parameters in `src/eeg_preproc_helpers.py`. Check the ~16 callers
  for positional-argument breakage. Silent wrong-plot risk, not a crash.
- [ ] **C2. `review_power` has no `.py` extension** and no twin — the only copy
  of 1134 lines. Rename to `.py`.
- [ ] **C3. Rename `label_bad_windows_continous.py`** → `..._continuous.py`
  before it gets imported or referenced widely.
- [ ] **C4. Triage dated filenames** — `compute_norm_eye_features_4Jan26.py`,
  `compare_attn_states_heatmap_fromPCs_MATRIX_21Apr25.py`, etc. Which are
  current? Archive or rename the rest.

## D. Unmapped territory

- [x] **D1. Map the eye-tracking branch.** DONE — joins at **Tier 4**
  (`*_power_eye_merged.csv`), where eye features, PC1–PC4 and Mahalanobis group
  deviation merge onto long windowed power. Producers present in repo:
  `prePCA_agg_norm.py` (aggregate/normalize), `robust_pca_gaze_features.py` (PCs).
  Still open: `compute_eye_measures.py` exists in **both** script directories at
  different lengths (516 vs 618 lines) — two diverged copies, unclear which is current.
- [x] **D2. Attention-label provenance.** DONE — labels are created in the
  Tier 3 -> 4 merge, which is missing from the repo. Promoted to **B4**.

## E. Development velocity — pays back over weeks

- [ ] **E1. Roll the docstring contract across active scripts** (currently 2 of
  43). Highest-leverage documentation work: it makes the pipeline map derivable
  from code instead of separately maintained, so it cannot silently drift.
  Do the ~15 active pipeline scripts first, not all 43.
- [ ] **E2. Single path config.** 39 of 55 scripts hardcode absolute paths
  across two machines plus a stale `/Volumes/Samsung`.
  *Only urgent if running off this laptop.*
- [ ] **E3. Make `src/` importable** via `pyproject.toml` + `pip install -e .`.
  Removes `sys.path` hacks from ~16 scripts **with zero edits to import lines**,
  because the bare module names stay valid. Cheapest structural win available.

## F. Deferred until after the paper

Listed so they stop being re-proposed:

- Unifying the three vocabularies (`pow_ip` / `pow_tf_dat` / `pow_dat`) — do
  *after* B3, or renames and numerical differences get debugged simultaneously.
- README rewrite (currently describes a `scripts/`, `config/`, `tests/` layout
  that does not exist).
- Consolidating the 4 near-sibling preprocessing scripts.
- Tests, CI, packaging.
- Lab alignment: merging with `lab/master` or `lab/max_temp`, and pushing back
  to `IEEG`. Deliberately deferred until the repo is in better shape.

---

## Suggested order

1. ~~**B1**~~ — done. Schema verified against real files.
2. **B4** — recover the missing Tier 2->3 and Tier 3->4 producers. Now the
   biggest risk in the project: the paper's dependent measures come from code
   that is not in version control. Independent of B2/B3, so it can start now.
3. **A1, A2, A3** — settle upstream decisions before the final extraction run.
4. **B2 → B3** — bridge, then validate. B3 is the moment the old scripts are
   proven reusable against new data.
4. **C1–C4** — cheap, do between longer tasks.
5. **E1** — as each script is touched for another reason, add its header.
6. **D1, D2** — before writing the paper's methods section, since neither the
   eye-tracking branch nor the attention-label provenance is currently documented.
