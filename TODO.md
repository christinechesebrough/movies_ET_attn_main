# TODO

Working backlog. Organizing principle: **settle everything upstream of Stage 1
before committing to the final wavelet extraction.** Re-extracting continuous
wavelets over all subjects is the most expensive operation in this pipeline, so
decisions that change its input are the ones that must be made first.

Status: `[ ]` open · `[~]` in progress · `[x]` done · `[?]` needs a decision

---

Full pipeline structure: **`PIPELINE.md`**.

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

- [ ] **A4. Clean up the eye-tracking pipeline.** Christine flagged it as
  needing work. Current provisional order: `compute_eye_measures.py` ->
  `prePCA_agg_norm.py` -> `robust_pca_gaze_features.py` ->
  `examine_separate_PCs_together.py` (labels). Two diverged copies of
  `compute_eye_measures.py` exist (516 vs 618 lines, different directories) plus
  a `compute_eye_measures_unified.py` — resolve which is canonical first.
  *Upstream of every attention label, so it gates the dependent measures.*

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
  RESOLVED 2026-09-09: atlas rows come from the sibling
  `{pat}_{ses}_{run}_{vid}_channel_metadata.csv` in each wavelet patient dir
  (columns: label, DK_Atlas, Y7_Atlas, Y17_Atlas, AparcAseg_Atlas; one row per
  channel). Join on `label`. The fifth `network` row comes from
  `define_custom_network_atlas.py`, which in the old chain rewrote the Tier 1
  CSV in place — that logic needs applying to channel_metadata.csv instead.

- [ ] **B3. Validate new bands against old bandpass power.**
  Run B2 on one patient/movie previously analysed with `extract_power_*`, and
  diff the CSVs. If band power agrees within tolerance, all of Stage 4 is
  certified against new data for free.
  **Highest-value single step in this list.** It is also a real scientific
  check: wavelet-derived bands and Hilbert/bandpass power can differ
  legitimately because the filters differ — a mismatch is not automatically
  a bug, and needs interpreting rather than "fixing".
  *Blocked by B2.*

- [ ] **B4. Recover the two missing reshape/merge steps.**
  Corrected 2026-09-09: attention labels ARE version controlled — they come from
  `examine_separate_PCs_together.py` (Mahalanobis deviation + `z_thresh`), which
  sits downstream of the eye-tracking pipeline. What is *not* in the repo:
  1. **Tier 2 -> 3**: whatever builds `all_power_wide.csv` (per-band wide
     windowed CSVs -> one long table, bands as columns, atlas rows -> columns).
  2. **Tier 3 -> 4**: the join merging the labeled eye-feature frame onto long
     power to produce `*_power_eye_merged.csv`.
  Both were likely done interactively. Smaller than first assessed, but still
  the undocumented seam between the power and eye branches.
  **Purpose is to re-derive, not to backfill.** Hungarian IS complete in the
  old pipeline's rolling-FOOOF branch (16 patients, Jun-Jul 2026); it is absent
  only from Branch A's May 17 power+eye aggregates. Do not push hungarian
  through the old Branch A chain to patch that — as the bridge condition it must
  be derived identically to english and inscapes, which means all three go
  through the wavelet pipeline. B4 needs the reshape/merge *logic*,
  reimplemented on wavelet-derived Tier 2 for all three videos.
  Old outputs are preserved as validation references for B3.
  *Blocks B5. Depends on B2.*

- [ ] **B5. Run Stage 4 unchanged** on the new data.
  *Blocked by B3 and B4.*

- [~] **A5. Inclusion/exclusion logging.** SCAFFOLD DONE.
  `analysis_scripts/scan_recording_coverage.py` walks the data tree and writes
  `Movie_data/recording_coverage.csv`: one row per (patient, session, video,
  run) with has_preprocessed / bad_channels / bad_windows / wavelet /
  power_tier1 / power_tier2 / fooof, plus eye-quality metrics joined from the
  existing `missing_data_{video}.csv` tables.
  First run: **87 recordings, 39 patients.** Attrition 87 preprocessed -> 54
  wavelet -> 47 tier2 -> 29 fooof; **33 recordings are preprocessed with no
  downstream output at all.**
  The `include` / `exclude_reason` / `exclude_modality` columns are left BLANK
  by design — file presence records what was processed, never what should be.
  Gaze heuristic implemented (Christine's rule: exclude if under 70% gaze data
  present in EACH eye). Emitted as `suggested_include`, advisory only, since she
  has stated there are exceptions. Per-eye present fractions are reported for
  both the pre- and post-interp metrics.

  **[?] OPEN DECISION — pre- or post-interp?** The choice moves 9 recordings:
  pre-interp passes 60 / fails 25; post-interp passes 51 / fails 34. The 9 that
  flip sit at 0.72-0.80 pre but 0.49-0.68 post (NS151, NS151_02, NS153,
  NS174_02, NS174_03 english; NS144, NS151 hungarian; NS136, NS151 inscapes).
  Note post-interp missing is HIGHER than pre in 97.8% of rows, so it is the
  conservative metric, not a gap-filled one. `suggested_include` currently uses
  pre-interp.

  **6 recordings fail the 70% rule but were processed downstream anyway** —
  either the stated exceptions or oversights; worth confirming which:
  NS155_02 english (0.52/0.53), NS190 english run-2 (0.76/0.54),
  NS128_02 hungarian (0.67/0.66), NS167 hungarian (0.56/0.68),
  NS153 inscapes (0.61/0.64), NS210 inscapes (0.77/0.66).
  NS190 and NS210 fail on ONE eye only — the per-eye rule catches asymmetry
  that an averaged-across-eyes rule would hide.

  Cross-referenced against what actually reached the PC/attention-label stage
  (`shared_PC_features_10s_20Apr26/*features_df*[0.6, 0.6]*.csv`), now emitted
  as `in_pca_stage`. **No script carries a patient list** — inclusion is purely
  "whichever input files happened to exist."

  Results: 18 english / 12 hungarian / 17 inscapes reached the PC stage.
  - **4 included despite failing the 70% rule:** NS155_02 english (0.52/0.53),
    NS190 english run-2 (0.76/0.54), NS153 inscapes (0.61/0.64),
    NS210 inscapes (0.77/0.66).
  - **17 pass the rule but never reached the PC stage**, several with excellent
    gaze: NS201_02 english (0.98/0.93), NS204 english (0.85/0.98),
    LH010 hungarian (0.93/0.93), NS155/NS155_02 hungarian (0.92/0.93).
  - **[?] NS190 english: the WORSE run was included.** run-1 (0.887/0.889,
    passes, has wavelet) is absent; run-2 (0.762/0.538, fails on the left eye)
    is in. Looks like an error rather than a judgement call — worth checking.

  Remaining: settle the pre/post question, resolve NS190, fill `include` by
  hand, then make the pipeline scripts read this table.

- [ ] **A7. Anatomy paths point at a different drive.** CORRECTED 2026-09-09:
  `movie_subs_master_updated.csv` is NOT missing and is NOT an inclusion
  mechanism. It is a concatenated table of the electrode correspondence sheets
  (11313 rows x 56 cols, 44 subjects, 2004 contacts) - an older way of pulling
  data out of those sheets. It lives at
  `/media/christine/Data/anatomy/shared_correspondence/movie_subs_master_updated.csv`.

  `anatomy/` is the FreeSurfer directory tree and lives on the **Data** drive,
  not the Samsung drive. Seven scripts still reference the Mac path
  `/Volumes/Samsung/anatomy/...`.

  **Consequence for E2:** the Mac->Linux path map is NOT a single prefix swap.
  On the Mac both `anatomy` and `Movie_data` sat under `/Volumes/Samsung`; on
  Linux they are on different physical drives:
  ```
  /Volumes/Samsung/anatomy/...     -> /media/christine/Data/anatomy/...
  /Volumes/Samsung/Movie_data/...  -> /media/christine/Samsung/Movie_data/...
  ```
  A naive `/Volumes/Samsung` -> `/media/christine/Samsung` substitution silently
  breaks every anatomy path. The path config must map per resource.

- [x] **A8. Disk space.** RESOLVED — new wavelet output already targets the
  Data drive (6.6 TB free). Current footprint 1.4 TB: Stage 1 raw TF 750 G
  (english 318 G / hungarian 252 G / inscapes 180 G), wavelet_continuous_z
  612 G, wavelet_band_power 38 G. Single files run 9-27 GB.
  Samsung remains at 95% full / 107 GB free — do not write new large outputs
  there.

- [ ] **A9. inscapes is behind on Stage 2.** Coverage scan shows inscapes with
  12 Stage 1 HDF5 files but only **2** carrying derived Stage 2 output, versus
  english 22/24 and hungarian 18/19. Either the derivation has not been run for
  inscapes or it failed partway.

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
  RESOLVED 2026-09-09: atlas rows come from the sibling
  `{pat}_{ses}_{run}_{vid}_channel_metadata.csv` in each wavelet patient dir
  (columns: label, DK_Atlas, Y7_Atlas, Y17_Atlas, AparcAseg_Atlas; one row per
  channel). Join on `label`. The fifth `network` row comes from
  `define_custom_network_atlas.py`, which in the old chain rewrote the Tier 1
  CSV in place — that logic needs applying to channel_metadata.csv instead.

- [ ] **B3. Validate new bands against old bandpass power.**
  Run B2 on one patient/movie previously analysed with `extract_power_*`, and
  diff the CSVs. If band power agrees within tolerance, all of Stage 4 is
  certified against new data for free.
  **Highest-value single step in this list.** It is also a real scientific
  check: wavelet-derived bands and Hilbert/bandpass power can differ
  legitimately because the filters differ — a mismatch is not automatically
  a bug, and needs interpreting rather than "fixing".
  *Blocked by B2.*

- [ ] **B4. Recover the two missing reshape/merge steps.**
  Corrected 2026-09-09: attention labels ARE version controlled — they come from
  `examine_separate_PCs_together.py` (Mahalanobis deviation + `z_thresh`), which
  sits downstream of the eye-tracking pipeline. What is *not* in the repo:
  1. **Tier 2 -> 3**: whatever builds `all_power_wide.csv` (per-band wide
     windowed CSVs -> one long table, bands as columns, atlas rows -> columns).
  2. **Tier 3 -> 4**: the join merging the labeled eye-feature frame onto long
     power to produce `*_power_eye_merged.csv`.
  Both were likely done interactively. Smaller than first assessed, but still
  the undocumented seam between the power and eye branches.
  **Purpose is to re-derive, not to backfill.** Hungarian IS complete in the
  old pipeline's rolling-FOOOF branch (16 patients, Jun-Jul 2026); it is absent
  only from Branch A's May 17 power+eye aggregates. Do not push hungarian
  through the old Branch A chain to patch that — as the bridge condition it must
  be derived identically to english and inscapes, which means all three go
  through the wavelet pipeline. B4 needs the reshape/merge *logic*,
  reimplemented on wavelet-derived Tier 2 for all three videos.
  Old outputs are preserved as validation references for B3.
  *Blocks B5. Depends on B2.*

- [ ] **B5. Run Stage 4 unchanged** on the new data.
  *Blocked by B3 and B4.*

- [~] **A5. Inclusion/exclusion logging.** SCAFFOLD DONE.
  `analysis_scripts/scan_recording_coverage.py` walks the data tree and writes
  `Movie_data/recording_coverage.csv`: one row per (patient, session, video,
  run) with has_preprocessed / bad_channels / bad_windows / wavelet /
  power_tier1 / power_tier2 / fooof, plus eye-quality metrics joined from the
  existing `missing_data_{video}.csv` tables.
  First run: **87 recordings, 39 patients.** Attrition 87 preprocessed -> 54
  wavelet -> 47 tier2 -> 29 fooof; **33 recordings are preprocessed with no
  downstream output at all.**
  The `include` / `exclude_reason` / `exclude_modality` columns are left BLANK
  by design — file presence records what was processed, never what should be.
  Gaze heuristic implemented (Christine's rule: exclude if under 70% gaze data
  present in EACH eye). Emitted as `suggested_include`, advisory only, since she
  has stated there are exceptions. Per-eye present fractions are reported for
  both the pre- and post-interp metrics.

  **[?] OPEN DECISION — pre- or post-interp?** The choice moves 9 recordings:
  pre-interp passes 60 / fails 25; post-interp passes 51 / fails 34. The 9 that
  flip sit at 0.72-0.80 pre but 0.49-0.68 post (NS151, NS151_02, NS153,
  NS174_02, NS174_03 english; NS144, NS151 hungarian; NS136, NS151 inscapes).
  Note post-interp missing is HIGHER than pre in 97.8% of rows, so it is the
  conservative metric, not a gap-filled one. `suggested_include` currently uses
  pre-interp.

  **6 recordings fail the 70% rule but were processed downstream anyway** —
  either the stated exceptions or oversights; worth confirming which:
  NS155_02 english (0.52/0.53), NS190 english run-2 (0.76/0.54),
  NS128_02 hungarian (0.67/0.66), NS167 hungarian (0.56/0.68),
  NS153 inscapes (0.61/0.64), NS210 inscapes (0.77/0.66).
  NS190 and NS210 fail on ONE eye only — the per-eye rule catches asymmetry
  that an averaged-across-eyes rule would hide.

  Cross-referenced against what actually reached the PC/attention-label stage
  (`shared_PC_features_10s_20Apr26/*features_df*[0.6, 0.6]*.csv`), now emitted
  as `in_pca_stage`. **No script carries a patient list** — inclusion is purely
  "whichever input files happened to exist."

  Results: 18 english / 12 hungarian / 17 inscapes reached the PC stage.
  - **4 included despite failing the 70% rule:** NS155_02 english (0.52/0.53),
    NS190 english run-2 (0.76/0.54), NS153 inscapes (0.61/0.64),
    NS210 inscapes (0.77/0.66).
  - **17 pass the rule but never reached the PC stage**, several with excellent
    gaze: NS201_02 english (0.98/0.93), NS204 english (0.85/0.98),
    LH010 hungarian (0.93/0.93), NS155/NS155_02 hungarian (0.92/0.93).
  - **[?] NS190 english: the WORSE run was included.** run-1 (0.887/0.889,
    passes, has wavelet) is absent; run-2 (0.762/0.538, fails on the left eye)
    is in. Looks like an error rather than a judgement call — worth checking.

  Remaining: settle the pre/post question, resolve NS190, fill `include` by
  hand, then make the pipeline scripts read this table.

- [ ] **A7. `movie_subs_master_updated.csv` is MISSING from this drive.**
  Seven scripts load it —
  `robust_pca_gaze_features.py`, `compute_eye_measures.py` (both copies),
  `prePCA_agg_norm.py`, `plot_the_present.py`,
  `plot_elec_anatomy_attn_effects.py`, `compute_norm_eye_features_4Jan26.py` —
  all at `/Volumes/Samsung/anatomy/shared_correspondence/movie_subs_master_updated.csv`.
  The whole `anatomy/` tree is absent here; it lived on the Mac.
  **Those scripts cannot run on this machine as-is**, which likely explains why
  parts of the eye branch have not been re-run on the new data.
  `Movie_data/data/movie_subs_table.xlsx` is NOT a substitute — it is a
  contact-level anatomy table (6328 electrodes x 56 cols), not a
  recording-level table.
  *Blocks A4 (eye-pipeline cleanup) and the whole eye branch. Locate it on the
  Mac and copy it across, or reconstruct what it provided.*

  Note for A6: that xlsx is a THIRD source of channel metadata, carrying its own
  `AparcAseg_Atlas` / `DK_Atlas` / `Y7_Atlas` / `Y17_Atlas` columns alongside the
  correspondence sheets and the wavelet-side channel_metadata.csv.
  Re-running never clobbers annotations; it writes `_rescan.csv` alongside.

- [ ] **A5b. Original (superseded) item: inclusion/exclusion logging.** Which recordings are in or out, and
  why (iEEG quality vs eye-tracking quality), is not recorded anywhere — it is
  implicit in hand-edited patient lists per step. Effective N is not recoverable
  from outputs, and steps can silently run on different subject sets.
  Wanted: one per-recording inclusion table (patient/session/run/video,
  included, reason, failing modality) written once and read everywhere.
  *Cross-cutting; affects every pipeline and every reported N.*

- [~] **A6. Canonical channel-metadata access.** IN PROGRESS.
  Done: `src/channel_metadata.py` — reads the correspondence sheets
  (`Movie_data/data/movie_elec_corr_sheets/`, 106 sheets, 48 cols) and emits
  normalized `label / DK_Atlas / Y7_Atlas / Y17_Atlas / AparcAseg_Atlas /
  network` plus quality flags. Verified against NS127_02's existing
  channel_metadata.csv: DK, Y7 and Y17 all match 100%.
  Consolidates a Yeo mapping that was duplicated across 9 scripts, and three
  competing Y7 vocabularies (raw codes / short names / long names) - all three
  retained as explicit maps.
  Remaining: adopt it in scripts as they are touched for other reasons. No
  existing script has been modified.

  **BUG FOUND — `AparcAseg_Atlas` is mispopulated in ALL 76 wavelet-side
  `*_channel_metadata.csv` files.** It holds a verbatim copy of `Y7_Atlas`
  (Yeo-7 network names) instead of FreeSurfer regions. The Tier 1/2 power CSVs
  are CORRECT (`Right-Amygdala`, `Right-Cerebral-White-Matter`), so the old
  power pipeline is unaffected - the fault is in whatever wrote the wavelet-side
  metadata. **B2 must not source atlas rows from those files**; use
  `src/channel_metadata.py`, which reads the correspondence sheet directly.
  Any analysis that used AparcAseg from the wavelet side needs rechecking.

- [ ] **A6b. Original (superseded) item: canonical channel-metadata access.** Atlas/channel metadata comes
  from each patient's electrode correspondence sheet but is imported
  inconsistently across preprocessing, analysis and plotting steps. Wanted: one
  canonical per-recording channel table plus a single accessor.
  `*_channel_metadata.csv` beside the wavelet outputs is the obvious candidate.
  *Cross-cutting. Also unblocks B2, which needs these rows.*
  Note there are FOUR sources of channel metadata in circulation:
  the correspondence sheets (authoritative, per patient),
  `movie_subs_master_updated.csv` (11313 rows, concatenation of those sheets),
  `movie_subs_table.xlsx` (6328 rows, older concatenation),
  and the wavelet-side `*_channel_metadata.csv` (which has the AparcAseg bug).

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
- [~] **E2. Single path config.** DONE: `src/paths.py`. Resolves roots by
  probing candidates, searches all roots per resource via `find()`, and
  `wavelet_raw_tf(vid)` picks the current HDF5 run by mtime rather than by the
  date-stamp string (which does not sort chronologically). Env overrides:
  `MOVIES_DATA_ROOT`, `MOVIES_ANATOMY_ROOT`. `python3 src/paths.py` reports.
  Adopted by `src/channel_metadata.py` and `scan_recording_coverage.py`.
  Remaining: migrate the 39 scripts as they are touched. No existing script has
  been modified.
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
2. **A4** — clean up the eye-tracking pipeline and settle which
   `compute_eye_measures.py` is canonical. It gates every attention label.
3. **A1, A2, A3** — settle upstream decisions before the final extraction run.
4. **B2 -> B4** — bridge to Tier 2, then reimplement the Tier 2->3->4
   reshape/merge on wavelet-derived data for all three videos.
4. **B2 → B3** — bridge, then validate. B3 is the moment the old scripts are
   proven reusable against new data.
4. **C1–C4** — cheap, do between longer tasks.
5. **E1** — as each script is touched for another reason, add its header.
6. **D1, D2** — before writing the paper's methods section, since neither the
   eye-tracking branch nor the attention-label provenance is currently documented.

---

## G. Directory organization

The layout is genuinely disorganized, but the expensive fix (relocating ~2.8 TB)
is the low-value half. `src/paths.py` already removes the need for scripts to
know where anything lives. What is worth doing, cheapest first:

- [ ] **G1. Delete the duplicated 2024 snapshot on the Data drive — 374 GB.**
  `Data/Movie_data/movies_nwb_standard` (198 GB) and
  `Data/Movie_data/movies_prep_standard` (176 GB) are **strict subsets** of the
  Samsung copies: 0 entries present on Data and absent on Samsung (verified
  2026-09-09 by top-level entry name, NOT by file content — confirm before
  deleting, e.g. `diff <(cd A && find . -type f | sort) <(cd B && find . -type f | sort)`).
  Deleting them also removes the trap where a path resolving to the Data drive
  silently yields 25 patients instead of 49, with no error.

- [ ] **G2. Move the rest of the 2024 material into `_archive/`.**
  ~57 GB of dirs unique to the Data drive whose newest file is 2024:
  `both_movies_extract` 13 G, `movie_prep_good_ET_2` 27 G,
  `movies_nwb_good_ET` 22 G, `ET_prep` 2.2 G, `movies_task` 888 M,
  plus ~40 small `*_test_*`, `*_24Jul24`, `alpha_all_*`, `HFA_all_*`,
  `new_verg*`, `networks_test_*` directories.
  Moving into `Movie_data/_archive/` is reversible and nothing references them.
  Do NOT delete outright — some are the only copy.

- [ ] **G3. Adopt a convention for NEW outputs only.**
  Do not rename existing directories: 39 scripts hardcode paths and renaming
  mid-analysis will break them silently. Instead fix the shape going forward:

  ```
  <root>/Movie_data/
      raw/            nwb, rawdata_for_conversion
      preprocessed/   movies_prep_standard
      wavelet/        {vid}/          Stage 1 raw TF HDF5
      derived/        band_power/ continuous_z/   Stage 2
      tabular/        tier2/ tier3/ tier4/        CSVs
      results/        figures, stats
      _archive/       superseded output
  ```

  Naming rules for new output directories:
    - no date stamps in directory names — the filesystem records mtime, and
      `_21Apr26` vs `_26Aug26` does not sort chronologically as a string
      (this already caused a wrong-directory bug in `paths.py`)
    - no parentheses or spaces — `tf_10s_(1, 150)` is awkward to glob and
      quote; use `tf_10s_f1-150`
    - parameters that identify a *variant* belong in the name; parameters that
      identify a *run* belong in a sidecar metadata file

- [ ] **G4. Settle which drive owns what, then record it in `paths.py`.**
  Current de facto split, which is defensible on size grounds:
    Samsung (107 GB free) — legacy power pipeline, raw/nwb, prep, PC features
    Data (6.6 TB free)    — all new wavelet output, anatomy
  Samsung being 95% full means new large output must go to Data regardless.
  The decision to record is whether Samsung eventually becomes archive-only.
