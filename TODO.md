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

- [~] **A11. Full-spectrum wavelet re-extraction.** IN PROGRESS.
  Replaces the original grid (linear 2 Hz, fixed nc=5, 600 Hz) with a single
  log-spaced extraction: **100 freqs 0.5-151 Hz, n_cycles log-ramped 3 -> 15,
  decim 6 (stored at 100 Hz)**. See `src/wavelet_grid.py`.

  Decided to do this as ONE extraction rather than a low-frequency patch plus
  the existing high-frequency data. Doing it in two would have meant two
  parameter sets, two sample rates (100 vs 600 Hz), a splice rule at 30 Hz, and
  concatenation only being possible after windowing.

  | band | old freqs / coverage / leakage | new |
  |---|---|---|
  | sub-delta 0.5-1 | 1 / 40% / 50% | 13 / 100% / 48% |
  | delta 1-3 | 2 / 40% / 50% | 19 / 100% / 31% |
  | theta 4-7 | 2 / 100% / 41% | 9 / 100% / 33% |
  | alpha 8-13 | 3 / 100% / 44% | 8 / 100% / 32% |
  | beta 14-30 | 8 / 100% / 32% | 13 / 100% / 18% |
  | gamma 31-50 | 10 / 100% / 46% | 8 / 100% / 22% |
  | HFA 51-150 | 50 / 100% / 30% | 18 / 100% / **6%** |

  **Better in every band AND 4.6x smaller**: 15.84 -> 3.47 GB per recording,
  ~174 GB for 50 against ~792 GB now.

  **Decimation is only valid because of the log n_cycles ramp.** Power envelope
  bandwidth equals filter bandwidth, so the worst case is 40.3 Hz at 151 Hz
  (nc=15) and 100 Hz gives 2.5x margin. Verified empirically with a 25 Hz
  amplitude modulation: 99% of envelope power below 25 Hz, only 0.0026% above
  the 50 Hz Nyquist. With the ORIGINAL nc=5 the same wavelet is 60 Hz wide,
  needs >=121 Hz, and puts 0.35% above that Nyquist - 130x more. MNE's `decim`
  is plain slicing with no anti-alias filter, so that would have aliased.

  Also caught during patching: `n_times` was taken from the raw sample count,
  which with decim would have allocated every array 6x too large with 5/6 left
  as zeros - silently, since nothing would error.

  Patient lists: english 19, inscapes 13, hungarian 18 (15 had been commented
  out from a previous run and were restored). **50 recordings total.**

  Remaining: verify the single-recording test, then run all 50. Supersedes the
  original extraction entirely - do not mix, n_cycles differs at every
  frequency.

## B. Critical path — completing the pipeline

Strictly ordered; each blocks the next.

- [x] **B1. Resolve the CSV schema.** DONE 2026-09-09, verified against real
  files. Result: **four tiers**, not one — see CLAUDE.md. The previously
  documented long format with `Is_Bad_Window` / `Window_Start_Sec` columns was
  inferred and wrong; no such columns exist.

- [x] **B2. Stage 3 bridge — DONE.**
  `analysis_scripts/wavelet_windows_to_csv.py`. Reads
  `wavelet_band_power/{vid}/{band}/{pat}/*_log_band_power.h5`, applies the
  identical rolling-mean windowing as `lowpass_power_to_windows.py`
  (6000-sample windows, 1500-sample step @ 600 Hz), joins the five atlas
  metadata rows via `src/channel_metadata.py`, and writes Tier 2 wide CSV.
  Verified on NS127_02 english gamma: **241 rows x same structure as the old
  Tier 2 file, exact window-count match (236 windows + 5 metadata rows).**
  `SOURCE='log_band_power'` reproduces old Tier 2 (unnormed rolling mean);
  `SOURCE='robust_z'` reads the pre-z-scored file instead and is NOT comparable
  to old outputs.

- [x] **B3a. `bandpass_from_wavelet.py` windowing VERIFIED** 2026-09-09
  (NS127_02, english, gamma). Independently reimplemented the full path —
  robust median/MAD over the continuous series, then rolling-mean windowing —
  and reproduced the stored `pow_dat_z_windowed` **bit-for-bit** (max abs
  difference 0.000e+00). Window centres identical; stored `robust_median` and
  `robust_sd` match recomputation exactly.
  Establishes: no implementation bug in the windowing or robust-z arithmetic,
  and the z-then-window order is as documented.
  Does NOT establish: correctness of the method choices (see B3), or that this
  holds beyond one recording/band.

- [x] **B3b. `wavelet_extract_windows.py` (`wavelet_continuous_z`) VERIFIED**
  2026-09-09 (NS127_02, english). Recomputed log10 + per-(channel,frequency)
  robust z from the source `pow_tf_dat` for 3 channels x 76 freqs x 359428
  samples (82 million values) and matched the stored output to float32
  precision: `pow_tf_log_z` max abs diff 4.8e-07, `robust_median` 4.7e-07,
  `robust_sd` 3.0e-08. Frequency axis (76 bins, 1-151 Hz) and channel labels
  identical between source and derived.

  **Note — the filename is misleading.** `*_log_robust_z_wavelets_10s_windows.h5`
  contains `pow_tf_log_z` at shape (145, 76, 359428): **continuous, full time
  resolution, NOT windowed.** The 236 window boundaries are stored alongside as
  metadata (`window_start_samples`, `window_end_samples`, `window_*_sec`,
  `window_centers_sec`) but are not applied to the data. Anything consuming
  these files must apply the windowing itself. Same misleading-name pattern as
  `lowpass_power_to_windows.py`, which does no lowpass filtering.
  (See TODO G3: no misleading names in new outputs.)

- [~] **B3. Validation — FIRST RESULT (NS127_02, english, gamma).**
  Wavelet-derived vs Hilbert/bandpass band power, same recording, same windows:

  | measure | value |
  |---|---|
  | global r | 0.981 (inflated by between-channel variance) |
  | **per-channel median r** | **0.840** (range 0.68-0.98; 114/145 below 0.9) |
  | slope, new = a*old + b | **3.33** (a=1 would be a pure units offset) |
  | SD | old 0.12, new 0.41 |
  | r after per-channel z-scoring | 0.839 |

  **Not interchangeable.** ~30% unshared variance within-channel. The slope of
  3.33 means the wavelet-derived power has ~3.3x the dynamic range, so this is
  a genuine methodological difference, not scaling.
  **RESOLVED 2026-09-09. Two compounding causes, neither one filtering:**

  1. **Amplitude vs power.** The old chain computes
     `power = np.abs(signal.hilbert(...))` — the *amplitude envelope*, despite
     the variable name (`extract_power_fc.py:810-811`, `extract_power_es.py:626`,
     the code's own comment says "raw envelope"). The new chain computes wavelet
     POWER (`representation = log10_mean_linear_wavelet_power_...`).
     Power = amplitude^2, so in log units this alone is a factor of exactly 2
     (verified by simulation: slope 2.00, r 1.0000).

  2. **The `+ 1e-6` epsilon compresses the old values.** The old chain does
     `np.log10(pow_dat_raw + 1e-6)`. Typical old amplitude is ~2.75e-6, so the
     epsilon is **36% of the signal** — not a numerical guard at this scale. It
     flattens the low end nonlinearly, cutting SD from ~0.42 to ~0.12 and
     pushing the slope above 2. Simulation with the epsilon: slope 2.74, SD 0.15
     vs 0.42 (measured: 3.33, 0.12 vs 0.41; residual gap is the frequency-bin
     averaging, which also differs).

  **Ruled out:** `lowpass_power_to_windows.py` performs NO filtering — no
  butter/filtfilt/savgol/decimate anywhere in it, despite the name; it only does
  a rolling mean. It also did NOT z-score these files: `normalize = False`, and
  the outputs are named `windowed_unnormed_*`.

  **Implication.** The old Tier 1/2 values lost roughly **41%** of the dynamic
  range of log amplitude to that epsilon, nonlinearly and per-channel
  (measured log-power SD 0.41 -> implied log-amplitude SD 0.205; old
  log(amp+eps) SD 0.12; 1 - 0.12/0.205 = 41%. An earlier note in this file said
  70%, which wrongly compared against the log-POWER SD without halving it) — which is why
  per-channel r is 0.84 rather than 1.0. Downstream z-scoring removes the offset
  but cannot undo nonlinear compression. **The new pipeline is cleaner, not just
  different; do not tune it to reproduce the old output.** Old and new results
  are not directly comparable, which reinforces deriving all three video
  conditions afresh (see the bridge-condition note in CLAUDE.md).
  Also: the new extraction has **145 channels vs the old 147** for this
  recording (missing RDh15, RDh16) - worth understanding.
  Remaining: repeat across bands, recordings and videos before drawing a
  conclusion. One recording x one band is not a validation.

- [ ] **B2-OLD (superseded). Write the Stage 3 bridge** (`wavelet_windows_to_csv.py`).
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

- [~] **A11. Full-spectrum wavelet re-extraction.** IN PROGRESS.
  Replaces the original grid (linear 2 Hz, fixed nc=5, 600 Hz) with a single
  log-spaced extraction: **100 freqs 0.5-151 Hz, n_cycles log-ramped 3 -> 15,
  decim 6 (stored at 100 Hz)**. See `src/wavelet_grid.py`.

  Decided to do this as ONE extraction rather than a low-frequency patch plus
  the existing high-frequency data. Doing it in two would have meant two
  parameter sets, two sample rates (100 vs 600 Hz), a splice rule at 30 Hz, and
  concatenation only being possible after windowing.

  | band | old freqs / coverage / leakage | new |
  |---|---|---|
  | sub-delta 0.5-1 | 1 / 40% / 50% | 13 / 100% / 48% |
  | delta 1-3 | 2 / 40% / 50% | 19 / 100% / 31% |
  | theta 4-7 | 2 / 100% / 41% | 9 / 100% / 33% |
  | alpha 8-13 | 3 / 100% / 44% | 8 / 100% / 32% |
  | beta 14-30 | 8 / 100% / 32% | 13 / 100% / 18% |
  | gamma 31-50 | 10 / 100% / 46% | 8 / 100% / 22% |
  | HFA 51-150 | 50 / 100% / 30% | 18 / 100% / **6%** |

  **Better in every band AND 4.6x smaller**: 15.84 -> 3.47 GB per recording,
  ~174 GB for 50 against ~792 GB now.

  **Decimation is only valid because of the log n_cycles ramp.** Power envelope
  bandwidth equals filter bandwidth, so the worst case is 40.3 Hz at 151 Hz
  (nc=15) and 100 Hz gives 2.5x margin. Verified empirically with a 25 Hz
  amplitude modulation: 99% of envelope power below 25 Hz, only 0.0026% above
  the 50 Hz Nyquist. With the ORIGINAL nc=5 the same wavelet is 60 Hz wide,
  needs >=121 Hz, and puts 0.35% above that Nyquist - 130x more. MNE's `decim`
  is plain slicing with no anti-alias filter, so that would have aliased.

  Also caught during patching: `n_times` was taken from the raw sample count,
  which with decim would have allocated every array 6x too large with 5/6 left
  as zeros - silently, since nothing would error.

  Patient lists: english 19, inscapes 13, hungarian 18 (15 had been commented
  out from a previous run and were restored). **50 recordings total.**

  Remaining: verify the single-recording test, then run all 50. Supersedes the
  original extraction entirely - do not mix, n_cycles differs at every
  frequency.

## B. Critical path — completing the pipeline

Strictly ordered; each blocks the next.

- [x] **B1. Resolve the CSV schema.** DONE 2026-09-09, verified against real
  files. Result: **four tiers**, not one — see CLAUDE.md. The previously
  documented long format with `Is_Bad_Window` / `Window_Start_Sec` columns was
  inferred and wrong; no such columns exist.

- [x] **B2. Stage 3 bridge — DONE.**
  `analysis_scripts/wavelet_windows_to_csv.py`. Reads
  `wavelet_band_power/{vid}/{band}/{pat}/*_log_band_power.h5`, applies the
  identical rolling-mean windowing as `lowpass_power_to_windows.py`
  (6000-sample windows, 1500-sample step @ 600 Hz), joins the five atlas
  metadata rows via `src/channel_metadata.py`, and writes Tier 2 wide CSV.
  Verified on NS127_02 english gamma: **241 rows x same structure as the old
  Tier 2 file, exact window-count match (236 windows + 5 metadata rows).**
  `SOURCE='log_band_power'` reproduces old Tier 2 (unnormed rolling mean);
  `SOURCE='robust_z'` reads the pre-z-scored file instead and is NOT comparable
  to old outputs.

- [x] **B3a. `bandpass_from_wavelet.py` windowing VERIFIED** 2026-09-09
  (NS127_02, english, gamma). Independently reimplemented the full path —
  robust median/MAD over the continuous series, then rolling-mean windowing —
  and reproduced the stored `pow_dat_z_windowed` **bit-for-bit** (max abs
  difference 0.000e+00). Window centres identical; stored `robust_median` and
  `robust_sd` match recomputation exactly.
  Establishes: no implementation bug in the windowing or robust-z arithmetic,
  and the z-then-window order is as documented.
  Does NOT establish: correctness of the method choices (see B3), or that this
  holds beyond one recording/band.

- [x] **B3b. `wavelet_extract_windows.py` (`wavelet_continuous_z`) VERIFIED**
  2026-09-09 (NS127_02, english). Recomputed log10 + per-(channel,frequency)
  robust z from the source `pow_tf_dat` for 3 channels x 76 freqs x 359428
  samples (82 million values) and matched the stored output to float32
  precision: `pow_tf_log_z` max abs diff 4.8e-07, `robust_median` 4.7e-07,
  `robust_sd` 3.0e-08. Frequency axis (76 bins, 1-151 Hz) and channel labels
  identical between source and derived.

  **Note — the filename is misleading.** `*_log_robust_z_wavelets_10s_windows.h5`
  contains `pow_tf_log_z` at shape (145, 76, 359428): **continuous, full time
  resolution, NOT windowed.** The 236 window boundaries are stored alongside as
  metadata (`window_start_samples`, `window_end_samples`, `window_*_sec`,
  `window_centers_sec`) but are not applied to the data. Anything consuming
  these files must apply the windowing itself. Same misleading-name pattern as
  `lowpass_power_to_windows.py`, which does no lowpass filtering.
  (See TODO G3: no misleading names in new outputs.)

- [~] **B3. Validation — FIRST RESULT (NS127_02, english, gamma).**
  Wavelet-derived vs Hilbert/bandpass band power, same recording, same windows:

  | measure | value |
  |---|---|
  | global r | 0.981 (inflated by between-channel variance) |
  | **per-channel median r** | **0.840** (range 0.68-0.98; 114/145 below 0.9) |
  | slope, new = a*old + b | **3.33** (a=1 would be a pure units offset) |
  | SD | old 0.12, new 0.41 |
  | r after per-channel z-scoring | 0.839 |

  **Not interchangeable.** ~30% unshared variance within-channel. The slope of
  3.33 means the wavelet-derived power has ~3.3x the dynamic range, so this is
  a genuine methodological difference, not scaling.
  **RESOLVED 2026-09-09. Two compounding causes, neither one filtering:**

  1. **Amplitude vs power.** The old chain computes
     `power = np.abs(signal.hilbert(...))` — the *amplitude envelope*, despite
     the variable name (`extract_power_fc.py:810-811`, `extract_power_es.py:626`,
     the code's own comment says "raw envelope"). The new chain computes wavelet
     POWER (`representation = log10_mean_linear_wavelet_power_...`).
     Power = amplitude^2, so in log units this alone is a factor of exactly 2
     (verified by simulation: slope 2.00, r 1.0000).

  2. **The `+ 1e-6` epsilon compresses the old values.** The old chain does
     `np.log10(pow_dat_raw + 1e-6)`. Typical old amplitude is ~2.75e-6, so the
     epsilon is **36% of the signal** — not a numerical guard at this scale. It
     flattens the low end nonlinearly, cutting SD from ~0.42 to ~0.12 and
     pushing the slope above 2. Simulation with the epsilon: slope 2.74, SD 0.15
     vs 0.42 (measured: 3.33, 0.12 vs 0.41; residual gap is the frequency-bin
     averaging, which also differs).

  **Ruled out:** `lowpass_power_to_windows.py` performs NO filtering — no
  butter/filtfilt/savgol/decimate anywhere in it, despite the name; it only does
  a rolling mean. It also did NOT z-score these files: `normalize = False`, and
  the outputs are named `windowed_unnormed_*`.

  **Implication.** The old Tier 1/2 values lost roughly **41%** of the dynamic
  range of log amplitude to that epsilon, nonlinearly and per-channel
  (measured log-power SD 0.41 -> implied log-amplitude SD 0.205; old
  log(amp+eps) SD 0.12; 1 - 0.12/0.205 = 41%. An earlier note in this file said
  70%, which wrongly compared against the log-POWER SD without halving it) — which is why
  per-channel r is 0.84 rather than 1.0. Downstream z-scoring removes the offset
  but cannot undo nonlinear compression. **The new pipeline is cleaner, not just
  different; do not tune it to reproduce the old output.** Old and new results
  are not directly comparable, which reinforces deriving all three video
  conditions afresh (see the bridge-condition note in CLAUDE.md).
  Also: the new extraction has **145 channels vs the old 147** for this
  recording (missing RDh15, RDh16) - worth understanding.
  Remaining: repeat across bands, recordings and videos before drawing a
  conclusion. One recording x one band is not a validation.

- [ ] **B2-OLD (superseded). Write the Stage 3 bridge** (`wavelet_windows_to_csv.py`).
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

- [?] **A10. BUG — `wavelet_extract_windows.py` never applies its windows.**
  Confirmed by Christine 2026-09-09: windowing was intended; the step is missing.

  `get_rolling_window_indices()` computes 236 window boundaries (line 303) and
  they are written to the output (lines 378-382), but **no averaging is ever
  performed**. There is no `mean` anywhere in the file. `pow_tf_log_z` is
  allocated at `(n_channels, n_freqs, n_times)` and the continuous `tf_z` is
  written straight to it.

  Consequences:
  - 53 files / 621 GB hold continuous data where ~408 MB of windowed data was
    intended — **1523x the storage** (15.8 GB vs 10.4 MB per recording).
  - The filename `*_log_robust_z_wavelets_10s_windows.h5` is actively
    misleading.
  - Nothing downstream has consumed these yet, and a consumer expecting 236
    windows would hit a 359428-length axis — a shape error, so it would fail
    loudly rather than silently produce wrong numbers.

  The log10 and per-(channel,frequency) robust z ARE correct and verified
  (B3b), so the existing data is sound, just unreduced.

  **FIX 1 DONE 2026-09-09:** `analysis_scripts/window_continuous_wavelets.py`
  windows the existing files. Verified on NS127_02 english: mean/median/sd match
  an independent recompute to float32 (4.8e-07), robust_median/sd carried
  forward, z recovery exact (9.8e-07). **12.5 GB -> 25.3 MB in 393 s** (494x
  reduction). Full set: 53 recordings, ~5.8 h.

  Design decision — it stores windowed **log power**, not windowed z. The
  z-scoring is an affine transform with constants fixed per (channel,
  frequency), so it is exactly invertible and windowing commutes with it
  (verified to 1e-15). Storing log power keeps absolute scale and leaves any
  alternative normalisation available; z is recoverable in one line via the
  carried-forward `robust_median`/`robust_sd`. Storing z would have frozen one
  normalisation choice and pinned the grand mean at ~0, forcing every state
  contrast to be symmetric about zero.

  Also adds `frac_bad` per window from the 1 s QC files, so bad-data handling
  becomes an analysis-time threshold rather than all-or-nothing rejection.
  On NS127_02, 26 of 236 windows contain some artifact but none exceed 50%.

  **STATISTICS CAVEAT baked into the output attrs:** windows are 10 s with a
  2.5 s step, so 75% overlap. The 236 windows are NOT independent - a 599 s
  recording holds 60 non-overlapping 10 s epochs. Use every 4th window
  (stride 4 -> 59 observations) for statistics; all 236 for plotting. Treating
  all 236 as independent inflates df ~4x and makes p-values anticonservative.
  This applies to the OLD pipeline too, which uses the same grid.

  **FIX 2 still open:** `wavelet_extract_windows.py` itself is unchanged, so
  future runs will again produce unwindowed output. Allocate `pow_tf_log_z` at
  `(n_channels, n_freqs, n_windows)` and average within
  `window_starts`/`window_ends` before writing.

  Original options as assessed:
  1. *Window the existing files* (cheap). The continuous z-scored data is
     verified; averaging within the window boundaries already stored in each
     file is a pure reduction. No need to re-read the 750 GB of Stage 1 raw TF
     or recompute log/z.
  2. *Fix the script* so future runs are correct: allocate `pow_tf_log_z` at
     `(n_channels, n_freqs, n_windows)` and average `tf_z` within
     `window_starts`/`window_ends` before writing.

## C. Correctness fixes — small, do when convenient

- [x] **C0b. FOOOF light pipeline — RUN 2026-09-10.**
  `analysis_scripts/extract_fooof_light.py` + `launch_fooof_light.sh` ->
  `/media/christine/Samsung/Movie_data/rolling_fooof_light/{pat}/`
  `{pat}_{ses}_{run}_{vid}_fooof_light.csv`, 71 recordings (25 english,
  26 hungarian, 20 inscapes), 3.08 M channel-windows, 831 MB. Knee aperiodic
  fit 1-57 Hz + per-band flattened-spectrum readout (theta/alpha/beta/gamma),
  10 s / 2.5 s grid starting at 0 s (the June `rolling_fooof_*_26Jun26` grid
  starts at 5 s — join on `Window_Center_Sec`, not `Window_Index`).
  Column rename done first: `{band}_ArgmaxHz` (never NaN) and
  `{band}_CF_if_present` (NaN below PRESENCE_THRESHOLD); no `{band}_CF`.
  Validation: R2 0.97 all three videos; median f_knee 7.8-8.3 Hz; knee
  undefined in 0.5-1.4% of fits; theta present 70-77%, alpha 89-90%,
  beta/gamma >99% (binary presence useless there, as the docstring says).
  NS127_02 english knee-exponent vs old fixed-exponent r = 0.43, consistent
  with the 0.46 measured in the misspecification study.
  Caveats: `max_n_peaks=6` is reached in 41-49% of fits (by design — the cap
  protects the aperiodic fit, per-band readout is cap-free); `Region` is
  empty for LH010, LH012, NS167 (no usable correspondence sheet).
  **Duplicate preprocessed files found:** NS166, NS178, NS210, NS211 each
  have two inscapes `.fif`s (`_ieeg` vs not, `run-01` vs `run-1`) with
  DIFFERENT data (r 0.94-0.999, different bads). Every extraction script
  selects on `'referenced' in f` and so processes both, later writer wins.
  Decision (Christine 2026-09-10): use the most recently modified file.
  Implemented in `extract_fooof_light.list_recordings()` only; the other
  extractors (`extract_power_per_recording`, `extract_wavelet_hdf5`,
  `extract_all_fooof`, `extract_fooof_knee`) still have the ambiguity.
  Note the same collision affected `full_raw_log_power_rescale` inscapes
  for NS178/NS210 — which copy those hold is unknown.

- [x] **C0c. Power extraction — DONE 2026-09-10.** All 324 Tier 1 files present
  in `/media/christine/Data/Movie_data/full_raw_log_power_rescale`
  (54 recordings x 6 bands). The last 20 (NS127_02/NS135/NS136/NS137 english,
  non-delta bands) were completed by Christine in Spyder.

- [x] **C0d. Tier 1 -> Tier 2 for the rescaled power — DONE 2026-09-10.**
  `analysis_scripts/window_bandpass_power_robust.py` ->
  `/media/christine/Samsung/Movie_data/windowed_power_10s_rescale/`
  324 files x 2 window statistics, 286 MB each, 0% NaN, verified against a
  hand computation to 0.00e+00. Old `windowed_power_10s` untouched.

- [x] **C0e. DECIDE the window statistic: `trim20` vs `median` vs `mean`.**
  DECIDED 2026-09-13, all three on disk (`*_rolling_{trim20,median,mean}_10s.csv`).
  At the level that matters - the Internal vs External contrast matrices
  (`compare_attn_states_lmm.py`, pooled labels, within_subject, z = 0.6,
  STRIDE = 4 for every run) - the statistic is immaterial:
    trim20 vs median  r(est) 0.999 / 0.999 (Y17 / Y7), slope 1.00-1.01
    trim20 vs mean    r(est) 0.998 / 0.998, slope 0.98-1.01
    median vs mean    r(est) 0.994 / 0.995
  sign agreement 96-100%, 95-99 of ~102 significant Y17 cells shared, no
  sign flips. The burst-attenuation concern does not materialise (the mean
  enlarges no cell). Keep trim20; report "unchanged under mean, 20% trimmed
  mean and median windowing".
  **CAUTION (found 2026-09-13):** an earlier version of this note compared
  runs made at different STRIDE values after the repo script was edited to
  STRIDE = 1 in Spyder on 2026-09-11; those directories were regenerated at
  stride 1 (trim20 z0.5/z0.6 Y17, z0.6 Y7) and are preserved as `*_stride1`.
  STRIDE matters more than the statistic: stride 4 vs stride 1 (both trim20)
  r(est) = 0.894, slope 0.78, sign agreement 87%, ~70% of significant cells
  shared. Stride 4 (non-overlapping windows) remains the reported design;
  pin STRIDE with every launch. Contrast outputs for the three statistics:
  `attn_state_contrasts_10s/{Y17,Y7}_pooled_within_subject_dev_z0.6{,_median,_mean}`.

- [ ] **C0f. Window counts vary 236-239 across recordings.** Recording length,
  not a bug — the OLD Tier 2 shows the same spread. But a downstream merge must
  not assume a fixed grid: attention labels come from `time_isc` with 237
  entries, so NS174_02 (238) and NS174_03 (239) will silently lose their extra
  windows on a positional join. Handle explicitly in B4.

- [ ] **C0. FOOOF `aperiodic_mode` is misspecified — refit required.**
  Measured 2026-09-10 (`analysis_scripts/compare_fooof_aperiodic_mode.py`,
  10,000 fits): a knee sits inside the 1-57 Hz fit range in **95.1%** of
  windows (median 5.19 Hz), the fixed-mode exponent correlates only **0.46**
  with the knee-mode exponent, and the bias tracks knee position at
  **r = -0.833** — a confound in the attention-state contrast, not just an
  offset. The same misfit suppresses delta and inflates theta/alpha
  (delta periodic power flips sign, -0.029 -> +0.196).
  Also `max_n_peaks=12` binds in 30.1% of fits.
  Blocks any use of `rolling_fooof_*_26Jun26` exponents or low-frequency peaks.
  Details in CLAUDE.md. Fix: `aperiodic_mode='knee'`, raise `max_n_peaks`,
  carry `f_knee` through as its own measure, refit.
  Note LH010's correspondence sheet has no `label` column (KeyError in
  `channel_metadata.py`) — unrelated, but it silently drops that patient.

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

## I. Exploration track `attn_explore_14Sep26` (dummy name, 2026-09-14)

Parallel to the legacy chain, for testing new ways of identifying attention
states. Outputs under `{MOVIE_DATA}/attn_explore_14Sep26/`; never writes into
legacy directories. Scripts prefixed `attn_explore_14Sep26_`.

- [x] **I1. Eye PCA without ISC** (`attn_explore_14Sep26_eye_pca.py` ->
  `eye_pca/`). Robust per-recording standardisation (H9; shared helper
  `src/eye_normalise.py`), stage-2 z per video, robust PCA on 8 gaze/pupil
  features, pooled and per video. ISC carried as a column (orients PC1: high
  PC1 = low ISC) but NOT in the PCA, retained for later feature sets.
  No Gaze_Valid_Frac masking yet (needs the compute_norm rerun).
  Pooled: PC1 64.8 % (PC2 25.2), cos 0.974 with legacy pooled PC1 on the
  same 8, r(PC1, ISC) -0.12. Loadings Saccade_Rate -0.67, Blink_Rate +0.63,
  Vergence +0.17, Vergence_Std +0.18, Pupil +0.19/+0.21.
  Per video vs pooled: cos 0.955 / 0.941 / 0.953, score r 0.96 / 0.95 / 0.95
  (English / Hungarian / Inscapes); PC1 var 64 / 65 / 53 %. English PC1 is
  saccade-blink only (vergence ~0); Hungarian and Inscapes add vergence and
  pupil. r(PC1, ISC) -0.09 to -0.12 in every fit.
  I1b. PC1 vs ISC, per viewer (eye_pca/pc1_vs_isc_per_recording.csv,
  _summary.csv): one r per recording across its 236 windows (Christine:
  the unit is the individual's windows, not pooled windows). Raw and
  within-rec z ISC give identical per-recording r. Mean r (Fisher) -0.08 to
  -0.13 for every source (pooled PC1 within each video, and per-video PC1),
  negative in 10-14 of 14-18 recordings, t on Fisher z p 0.004-0.02; range
  -0.46 to +0.27. Timepoint-demeaned ISC: means -0.03 to -0.14, significant
  only for Hungarian (both fits). Log entry 2.
- [x] **I2. Mahalanobis group deviation** (`attn_explore_14Sep26_deviation.py`
  -> `deviation/`). 9 features (PCA 8 + ISC), leave-one-out group mean per
  timepoint, empirical covariance, d z-scored within recording (mahal_z) and
  within timepoint (mahal_time_z). FINDING: with raw 0-1 ISC + Ledoit-Wolf
  (the legacy derive_deviation recipe) ISC's share of d^2 is ~1/30 instead
  of 1/9 - shrinkage toward the mean variance suppresses a feature whose
  residual variance is 0.015 vs ~1. Track version z-scores ISC across the
  video and drops shrinkage (3-4k windows x 9). Legacy is under-weighting
  ISC in its deviation. Results: d rises MONOTONICALLY with PC1 (mean
  mahal_z by PC1_z bin -0.2 -> +0.4/+0.6), not a U; per-rec r(d, PC1)
  Eng +0.25, Hun +0.37, Ins +0.17, negative
  in 1-4 of 14-18; r(d, ISC) -0.14 Eng, -0.20 Hun, ~0 Ins; eta2(Time) of d
  6-13 %; top-decile d carried mostly by Vergence_Std / Vergence (~20 %),
  ISC 6-8 %. Implication for labels: high-dev for Internal + low-dev for
  External follows the data; high-dev at both poles would starve External.
  Log entry 3.
- [ ] **I3. Vergence extremes vs gaze validity (2026-09-14, assessment only, no
  code changed).** Question: would the planned `Gaze_Valid_Frac` mask (H9c,
  fraction of finite combined-gaze samples per window, threshold 0.5) clean
  the tail-inflated vergence windows? Measured on 45 of 49 recordings
  (4 English Eye_prep files not matched by name: NS178, NS193 run-02, NS194,
  NS205), window-level, `eye_pca/vergence_validity_windows.csv`:
  165 windows in 14 recordings have |robust z(Vergence_Std)| > 6.
  Gaze_Valid_Frac < 0.5 catches 17 of them; < 0.8 catches 97 but discards
  27 % of all windows; < 0.9 catches 135 at 46 % discarded. NOT an adequate
  vergence filter. Reason: xy is the COMBINED gaze, valid when either eye
  is valid (combine_left_right fills from one eye); vergence needs BOTH eyes.
  The both-eyes-valid fraction (NaN pattern of dva_gaze_disp_x in the
  et_prep.csv) predicts the extremes far better: interpolated fraction > 0.2
  catches 121 of 165. But 87 of 165 stay extreme on measured samples alone,
  and 26 have good validity, little interpolation and are still extreme
  (e.g. NS140 English 185-190 s: measured disparity range 0.39 screen
  widths, ~10x typical SD) - real disparity jumps or one-eye miscalibration
  with valid flags. NS164 Hungarian alone has 39 of the 165 and its
  Vergence_Std tracks gaze validity at r = -0.86.
  Also noted: compute_norm uses `gaze_dist_x_interp` (uncapped, interpolated,
  screen-fraction units) with plain np.mean/np.std; the capped column caps
  interpolated samples only; both vergence_calc.py and eye_helpers.py
  interpolate the RIGHT eye's distance with `val_left` (copy-paste bug).
  Checked: mask_spikes_by_derivative and interp_nans_maxgap are defined in
  vergence_calc.py but never called (only commented-out calls); outliers
  come in RUNS up to 2-3 s (NS140 Eng 70 runs, 99 % measured; NS164 Hun
  28 runs, 76 % interpolated) and only 1-8 % of outlying samples are
  derivative spikes, so a step detector would miss the plateaus anyway.
  DECIDED (Christine, 2026-09-14): treat ONLY interpolated samples; a
  measured value stands even when extreme. IMPLEMENTED in
  compute_norm_eye_features_4Jan26.py under NORM_SCHEME = 'robust':
  `VERG_MEASURED_ONLY = True` excludes samples that are NaN in the raw
  dva_gaze_disp_x column (= one or both eyes missing) from the window
  mean / SD; windows with < `VERG_MIN_MEASURED` (0.5) measured get NaN ->
  recording median after standardisation; new `Verg_Valid_Frac` column
  (measured fraction per window) carried through build_both_movies and
  make_attention_labels. Legacy path unchanged. Preview on the two worst
  recordings: NS164 Hun extreme Vergence_Std windows 39 -> 16 (SD/MAD 6.2
  -> 3.8, 7 windows NaN); NS140 Eng 6 -> 6 (measured plateaus, kept by
  design). Not yet rerun in Spyder. Still open: the val_left copy-paste
  bug for the right eye in vergence_calc.py / eye_helpers.py (affects the
  saved files; a fix needs vergence_calc rerun). Exploration outputs
  untouched.
- [x] **I4. Full recomputation with measured-only vergence (2026-09-14).**
  compute_norm_eye_features_4Jan26.py run non-interactively for all three
  videos with NORM_SCHEME = 'robust' (driver in the session scratchpad; it
  execs the script with `vid` / `NORM_SCHEME` overridden, Agg backend) ->
  `6Apr26_norm_eye_features_by_rec_10s_robust/`. Two additions during the
  run: (a) VERG_MIN_MEASURED lowered 0.5 -> 0.3; (b) new
  VERG_MIN_REC_MEDIAN = 0.5: a recording whose median measured fraction is
  below it gets all three vergence features set missing (NS190 r1, NS194
  English; NS174_03 Hungarian; NS155, NS210 Inscapes) - otherwise a
  mostly-median bimodal series enters the PCA. Also fixed: Verg_Valid_Frac
  was not trimmed/stretched with the other window arrays when num_steps !=
  236 (English crashed). Track scripts now read the robust file
  (`SOURCE = 'compute_norm_robust'`), restricted to the legacy 17/14/18
  recordings (`INCLUDE_LEGACY_SET`; the script's English list has 4 more:
  NS166, NS190 r1/r2, NS193 r1). Whole track rerun: eye_pca, pc1-vs-ISC,
  deviation, time series; log entries 1-4 updated, entry 5 added.
  RESULTS vs first pass: English / Hungarian PC1 unchanged (cos 0.98-0.99),
  pooled 0.95, INSCAPES 0.82 - Saccade_Rate -0.53 -> -0.13, PC1 now
  Vergence 0.48 / Vergence_Std 0.49 / Pupil_Avg 0.38 / Blink 0.44 /
  Dispersion 0.33. Attribution (Inscapes refits): mask alone -> SR -0.11,
  measured vergence alone -> -0.27, dropping masked windows -> -0.37,
  removing NS151/153/155 (92-106 no-gaze windows each) -> -0.27; the two
  changes compound, none restores the first pass. Reading: no-gaze windows
  had been masquerading as low-saccade (internal) windows. PC1-ISC per-rec
  means -0.07 to -0.09 (was -0.08 to -0.13; pooled-in-Hungarian p 0.09);
  deviation-PC1 +0.26/+0.30/+0.28 (0-2 negative); deviation extremes still
  vergence-carried (measured samples, kept by design). OPEN: whether the
  Inscapes vergence/pupil axis is the right one to label from.
- [ ] **I5. ROBUST PCA IS DEGENERATE AT ITS DEFAULT LAMBDA (2026-09-14) - affects
  the LEGACY PC1 too.** `r_pca.R_pca` (principal component pursuit) uses
  lambda = 1/sqrt(max(n, m)) = 0.009 for n ~ 11.5k windows x 8-9 features.
  At that value the L1 penalty on the "sparse" part S is nearly free: S
  absorbs 87 % of the data's energy and is 91 % nonzero, L keeps 2-3.5 %
  (rank ~3), and the PCA is then fit on L. "PC1 explains 58-72 %" is 58-72 %
  of that 2 %; on the data it is ~20 %. Legacy pooled fit: identical
  behaviour (||L||^2/||X||^2 = 0.035; stored PC1 reproduced exactly). The
  DIRECTION survives roughly (cos robust vs plain PCA 0.95 legacy, 0.86
  track pooled, 0.81-0.94 per video; per RECORDING median 0.78, min 0.11)
  but the SCORES are shrunk projections of L: legacy stored PC1 scores r
  0.87 with X @ loadings and 0.83 with plain-PCA scores (this is the "r
  0.87" already noted under H7, now explained). Lambda sensitivity (track
  pooled): lambda 0.05 -> S is genuinely sparse (0.3 % of entries, max
  |S| 14.6 = the spikes), L keeps 93 %, cos(PC1, plain) 0.93; lambda >=
  0.2 -> S empty, identical to plain PCA. So a properly tuned robust PCA
  is plain PCA with a few hundred gross outliers removed. Cost: default
  R_pca 6 s per pooled fit; plain PCA 5 ms.
  DECISION NEEDED (Christine): for the track, (a) plain PCA on the
  robust-standardised features, (b) R_pca with lambda ~0.05 (sparse S),
  or (c) keep the default for continuity with legacy. For the paper: the
  legacy PC1 scores are not the projection of the eye data on the
  reported loadings. Not yet applied anywhere; no outputs changed.
- [x] **I6. Subject-level vs pooled PC1, and the structure of the feature set
  (2026-09-14, plain PCA on the recomputed robust features;
  `eye_pca/subject_level_pc1_plain.csv`, `_loadings_plain.csv`).**
  Plain PCA: PC1 explains only 20 % (per video 19-22 %; uniform 12.5 %),
  PC2 17 % - a near tie. The 8 features form TWO nearly independent
  clusters within viewer: Saccade_Rate <-> Blink_Rate (r -0.39) and
  Vergence <-> Vergence_Std <-> Pupil_Avg (r 0.30-0.38); all other |r| <
  0.2. Which cluster is PC1 depends on the sample (English: saccade-blink;
  Hungarian / Inscapes / pooled: vergence-pupil); per-video PC1 bootstrap
  5th-pct cos 0.05 (Eng) / 0.06 (Hun) / 0.56 (Ins) because the two swap;
  pooled 0.78; LOO min 0.97. The legacy PC1 (SR -0.58 / BR +0.58 + V/P
  0.2-0.3) is a shrinkage-determined BLEND of the two clusters (I5), not a
  dominant axis. Subject-level PC1s: own axis captures median 33 % of the
  recording vs 15-20 % for the pooled axis, but split-half (first vs
  second half) cos median 0.29-0.47; cos with pooled median 0.43-0.65,
  < 0.6 in 29 of 49; leading feature scattered across all 8 (= whichever
  cluster is larger in that person). => subject-level axes are unreliable
  and assign viewers to different constructs. Not recommended.
  OPEN (the real question): one axis vs two. Options: treat saccade-blink
  and vergence-pupil as separate dimensions and define states in that
  plane; pick one on theory; or a supervised reduction (toward ISC or the
  neural data). Log entry 6.
- [x] **I7. Track PCA switched to R_pca lambda = 0.05 (Christine, 2026-09-14)**
  and downstream recomputed (ISC, deviation, time series; default-lambda
  eye_pca kept as `eye_pca_lambda_default/`). Sparse part 0.25 % (pooled),
  1.4-2.0 % per video; L keeps 76-93 %. PC1 20-23 % of the DATA, PC2
  18-20 %. Pooled PC1: SR -0.50, BR +0.46, V +0.43, VS +0.24, PA +0.47, PS
  +0.24 - an even blend of the two clusters; cos 0.955 with legacy PC1 on
  8; per-video cos with pooled Eng 0.83 / Hun 0.99 / Ins 0.96. Per-viewer
  r(PC1, ISC) -0.08 to -0.12 (Ins p 0.002, Eng 0.04-0.07, Hun n.s.);
  timepoint-demeaned ISC strengthens Hun (-0.14, p 0.004), weakens others.
  DEVIATION vs PC1 is now an ASYMMETRIC U: mean mahal_z by PC1_z bin
  ~0 / -0.3 / -0.3 / 0 / +1.0; r(d, PC1) +0.36-0.44, r(d, |PC1|)
  +0.42-0.47. Gate at z 0.6: Internal keeps 41-46 % of PC1>0.6 windows
  (base 20 %) -> informative; External keeps 31-35 % of PC1<-0.6 (base
  26-30 %) -> nearly random. 12-15 % of Internal windows have residual
  deviation > 2 (binocular events). IMPLICATION for labels (log entry 7):
  internal = PC1 AND deviation; external = PC1 alone or a positive
  synchrony criterion; report/label which cluster carries each window.
  Legacy blend decomposed: cos 0.83 with the saccade-blink axis, 0.51 with
  vergence-pupil; both clusters relate to ISC (-0.09 / -0.05) and the
  blend more (-0.11) - a defensible composite, not a discovered axis.
- [x] **I8. Fine-bin deviation x PC1, within viewer AND within timepoint
  (2026-09-14; `deviation/pc1_dev_fine_bins.csv`, `pc1_dev_thresholds.csv`).**
  Mean mahal_z by PC1_z bin: ~-0.3 through the middle, crosses 0 at PC1_z
  0.4-0.5 (timepoint scheme 0.5-0.6), ~0 up to 0.8, then +0.25 (0.8-1.0),
  +0.35 (1.0-1.2), +0.5-0.7 (1.2-1.5), +1.1-1.6 (> 1.5); external side
  stays below average to -1.5, turns positive only < -1.5. Gate at
  matched threshold t = 0.4..1.0: internal pass rate flat 41-48 % vs base
  25 -> 12 % (informative at every t); external pass rate tracks base
  45 -> 7 % and is BELOW base for t >= 0.7. Both schemes identical in
  shape. => the 0.6 cut sits on the flat part; deviation separates
  internal windows only above ~0.8; no t at which a low-deviation gate
  adds anything for external. Log entry 8.
- [x] **I9. Region analysis: which windows are internal / external (2026-09-14;
  log entry 9).** Six PC1_z regions (< -1.5 | -1.5..-0.4 | -0.4..0.4 |
  0.4..0.8 | 0.8..1.2 | > 1.2; shares ~4 / 33 / 31 / 11 / 8 / 12 %).
  Per-viewer means, t-tests: deviation z -0.2..-0.3 in external + middle
  (p < 0.001), ~0 in 0.4-0.8 (n.s.), +0.32-0.35 in 0.8-1.2 (p < 0.004),
  +1.05-1.24 above 1.2 (p < 1e-5); far-external tail +0.04..+0.39 (sig in
  Inscapes only), paired vs external band p 0.04-0.0001 in all three.
  ISC z: +0.03..+0.10 external, -0.20..-0.25 far internal (p 0.009-0.047).
  Feature profile: internal bands = few saccades / many blinks / raised
  vergence & pupil scaling with distance; external band = the reverse at
  a fraction of the amplitude; far-external tail = saccade-rate burst
  with the other features near normal. Gate comparison at 0.6: internal
  PC1+dev doubles within-timepoint deviation of selected windows (+0.6 ->
  +1.4/1.5, p < 1e-5) and lowers ISC in English (p 0.002); PC1 > 0.8
  alone does much less. External PC1+low-dev leaves ISC unchanged (p
  0.5-0.9) while dropping 2/3 of windows; PC1 + ISC_time_z > 0 selects
  windows also more scene-typical (dev_t -0.2..-0.3, p 0.02-0.07).
  => internal = PC1 AND deviation; external = PC1 alone or + synchrony;
  far-external tail to be inspected separately.
- [x] **I10. The PC1 x deviation plane (2026-09-14; log entry 10).** 2-D
  histograms per video; GMM k=1..4: BIC drop 4-7 % for k=2, < 1 % after;
  k=2 = ordinary mass 61-75 % at (PC1 -0.2..-0.3, dev -0.3..-0.5) + internal
  lobe 25-39 % at (+0.5..+0.7, +0.8..+1.0); k=3 splits the ordinary mass
  along PC1, never a low/low lobe. Within-viewer r(dev, PC1): all 0.34-0.41;
  PC1 > median 0.51-0.60 (positive in every recording); PC1 < median
  -0.14..-0.19. Reverse conditional P(PC1>0.6 | dev>2) 65-71 % vs 5-10 %
  at dev<-1; P(PC1<-0.6 | dev) barely varies (30-38 % -> 13-17 %).
  Per-viewer counts: int PC1+dev 24-27 ± 7 [7-39]; ext PC1-only 65-70 ± 7
  [49-82]; ext PC1+low-dev 20-24 ± 10-12 [0-53]. => PC1 and deviation
  cluster together on the internal side only; external = low side of the
  ordinary mass, PC1 alone; the low-deviation gate makes external counts
  erratic.
- [x] **I11. ISC as a third criterion for external (2026-09-14; log entry 11).**
  Within-viewer r(ISC_z, PC1) on the external half (PC1 < median): -0.04 /
  +0.04 / -0.03, all n.s.; on the internal half -0.11 / -0.14 / -0.09 (p
  0.003-0.04) - the ISC-PC1 relation lives on the internal side only.
  Mean ISC_z by external PC1 bin flat at +0.05..+0.15, no gradient. GMM on
  PC1 x ISC: BIC drop 0.4-2.9 % (vs 4-7 % for PC1 x dev); 3-D adds nothing;
  no low-PC1/high-ISC lobe. ISC gate on PC1 < -0.6: pass 3-6 points above
  base (within-viewer), = base (within-timepoint). ISC_time_z > 0 gate
  gives erratic counts (0-61 per viewer). => external side homogeneous on
  deviation, ISC and time-ISC; ISC is a check / positive definition, not a
  sharpening criterion. Asymmetry stands.
- [x] **I12. Labels built (`attn_explore_14Sep26_labels.py`, 2026-09-14) ->
  `attention_labels_10s/attention_labels_10s_pooled_explore14Sep26.csv`**
  (compare_attn_states_lmm.py reads it with VARIANT = 'explore14Sep26',
  SCHEME = rule name, THRESHOLDS = [0.6] / [0.8]; FarExternal is ignored
  by the Internal-vs-External stage 1). Rules: conj (int PC1&dev, ext PC1
  alone, tail < -1.5 = FarExternal), legacy (symmetric), pc1only, syncext
  (ext PC1 & ISC_z > 0), conj_tp (within-timepoint), clusterA / clusterB
  (conj rule on the saccade-blink / vergence-pupil score). Per-viewer
  counts (mean): conj_0.6 int 24-27 / ext 55-60; conj_0.8 int 19-21 / ext
  38-42; legacy_0.6 ext 20-24 (min 0); legacy_0.8 ext 8-10; syncext_0.6
  ext 27-31; conj_tp int SD 10-20 (unstable in English). Counts in
  `attn_explore_14Sep26/labels/label_counts_per_viewer.csv`.
  JUSTIFIED NEURAL COMPARISONS (each tests one claim from the log):
  (1) conj_0.6 vs legacy_0.6 - does the external low-dev gate matter
  (same Internal set); (2) conj_0.6 vs pc1only_0.6 - does the internal
  deviation gate matter (same External set); (3) conj_0.8 vs conj_0.6 -
  threshold where deviation begins to rise; (4) syncext vs conj - external
  as positive synchrony; (5) clusterA vs clusterB vs conj - which cluster
  carries the contrast; (6) conj_tp - scheme sensitivity; (7) FarExternal
  vs External (needs a 3-level option in the contrast script). NOT RUN.
- [x] **I13. First neural contrast under conj_0.6 (2026-09-14; log entry 12).**
  compare_attn_states_lmm.py driven non-interactively (scratch driver) with
  VARIANT explore14Sep26, SCHEME conj, z 0.6, Y17 + Y7, mean, stride 1 ->
  `attn_state_contrasts_10s_explore14Sep26/{Y17,Y7}_pooled_conj_z0.6_mean_stride1/`.
  Y17 raw, sig cells / hypothesis hits-contra of 18: Eng 17 / 4-0 (legacy
  29 / 6-0); Hun 50 / 6-3 (38 / 5-3); Ins 34 / 8-0 (35 / 5-3). Y7 raw: Eng
  9 / 1-0 (14 / 3-0); Hun 25 / 3-3 (20 / 1-0); Ins 12 / 3-0 (17 / 2-0).
  Inscapes cleaner, Hungarian more cells, English fewer cells (the film
  whose own axis is most saccade-blink). Next: comparisons (1)-(5) of I12.
- [x] **I14. Six-state neural profile + three-state pairwise contrasts
  (2026-09-15; log entries 13-14).** `attn_explore_14Sep26_state_profile.py`:
  contact-centred amplitude by PC1 region (PC1 only, no deviation gate),
  Y17 + Y7 -> `state_profile/`. Network-averaged: alpha / beta / theta rise
  monotonically far-ext -> far-int by ~0.05-0.11 SD; HFA / gamma rise only
  at the internal end; delta flat; far-external tail = lowest point on
  low bands (extension of external, not a distinct profile).
  `attn_explore_14Sep26_state_contrasts3.py`: External (< -0.4) / Middle /
  Internal (>= 0.8), pairwise MixedLM contrasts; hypothesis networks (Y7
  DAN, DMN, VN; Y17 DAN-A, Default A, Vis Central, Vis Peripheral) Int-Ext
  UNCORRECTED, everything else FDR within (network, film). Int-Ext
  significant: alpha/beta/theta positive in DAN-A and Vis Central in all
  three films, Y7 VN in all three; hypothesised negative gamma/HFA only one
  uncorrected cell (Eng DAN-A gamma); Default A theta NEGATIVE in Hun and
  Ins (opposite to hypothesis); Default A / Vis Peripheral gamma-HFA
  positive in Hun.
- [ ] **I15. Report window counts per bin, per scheme (Christine, 2026-09-15;
  to come back to).** For every binning / categorisation used in the log -
  six PC1 regions (PC1-only), six regions deviation-gated (int > 0.6, ext
  <= 0), three collapsed states (both versions), the conj / legacy /
  pc1only / syncext / clusterA / clusterB label rules at 0.6 and 0.8, and
  the fine bins of entry 8 - report (a) the number of windows per bin per
  film and (b) windows per participant per bin (mean, SD, min, max), in
  the log and as a CSV. Accompany with the theoretical justification for
  each scheme (what the bin is meant to capture, why its edges sit where
  they do - lobe centre, deviation zero-crossing, rise point) and the
  statistical justification (minimum windows per participant per state
  for a usable stage-1 estimate given 75 % window overlap, effect of
  unequal counts on the mixed model, and how the per-participant minimum
  interacts with MIN_WIN in compare_attn_states_lmm.py). Partial counts
  already exist: `attn_explore_14Sep26/labels/label_counts_per_viewer.csv`
  (label rules) and the n_windows column in
  `deviation/pc1_dev_fine_bins.csv` (fine bins, pooled over viewers).
- [ ] **I16. Log rework, deferred (2026-09-15).** Before the log is treated as
  a reference: (a) extend the guide (stops at entry 9) to cover the joint
  plane, ISC-gate test, neural contrast, six-state profile and the
  three-state contrasts; (b) update the justification's labelling
  paragraph for the region-based internal boundary at 0.8 and the
  three-state results; (c) fold in I15 window counts. Timepoint-gate
  caveat already recorded in entry 14 (matched strictness at 0.6;
  per-viewer pass rate 0.05-0.93 vs 0.19-0.72; sensitivity only).
- [x] **I17. Planned contrasts over the six regions (2026-09-15;
  `attn_explore_14Sep26_state_trend6.py` -> `state_profile/trend6[_gated|
  _gated_tp].csv/.json`; log entry 13).** Replaces 15 pairwise tests per
  network x band with two: linear TREND (per-contact weighted slope of
  region means on rank 1-6, MixedLM with person intercept) and ENDPOINT
  (far internal - far external). Hypothesis networks trend uncorrected;
  all else FDR within (network, film) over 12 tests. PC1-only: significant
  trends Y17 62 (31 hyp + 31 other), all rising except delta 1 / gamma 1 /
  theta 3 falling; Y7 30, all rising but one. Gated versions similar
  (Y17 67 / 63, Y7 35 / 32). Endpoint: Y17 49-57, Y7 22-28. Trend arrows
  with stars drawn on the six-bar panels; summary + hypothesis tables in
  the log.
- [x] **I18. FOOOF aperiodic / periodic features by gaze state (2026-09-15;
  `attn_explore_14Sep26_fooof_states.py` -> `fooof_states/`; log entry 15).**
  Adapts compare_attn_states_fooof.py (9/10-11, legacy labels, stride 4) to
  the track: rolling_fooof_light knee-mode features (exponent, offset, knee
  Hz, periodic theta/alpha/beta/gamma; peak presence / CF left out), 47 of
  49 recordings (NS140_02 + NS205 English missing), contact-relative means
  per region, states = deviation-gated with the within-timepoint internal
  gate (internal mahal_time_z > 0.6; external within-subject mahal_z <= 0);
  two-state (conj_0.6), six-region trend + endpoint, three-state pairwise;
  same correction scheme as I14/I17. Significant trends: exponent and
  offset RISE external -> internal (Y17 8 rising / 1 falling each; Y7 4/0),
  knee Hz up where significant, periodic alpha rises (Y17 8/0), theta /
  beta / gamma ~0. Two-state weaker and mixed in sign. ODDITY to inspect:
  network-averaged exponent / offset dip in the gated 'internal' band
  (0.8-1.2) below both neighbours (-0.017 vs +0.015 ambiguous / +0.013 far
  internal) - possibly the timepoint gate selecting a different subset there.
- [x] **I19. Peak presence by state (2026-09-15; `fooof_states/presence/`,
  PRESENCE knob in attn_explore_14Sep26_fooof_states.py; companion page).**
  Has_theta / Has_alpha as % of a state's windows with a fitted peak, Y7,
  not contact-centred. Nearly flat: alpha peaks in ~92-93 % of windows,
  theta ~69-70 %, in every state; one significant trend each (theta 1 up /
  1 down), two-state 1 cell each. States differ in periodic POWER, not in
  whether a peak is fitted.
  NOTE (found while checking coverage): every NEURAL analysis in the track
  (entries 12-15) uses `descriptives_power_10s/included_recordings.csv` =
  42 recordings (Eng 16 / Hun 13 / Ins 13), NOT the 49 of the eye-side
  analyses; 7 eye-included recordings (NS140_02 Eng; LH010 Hun; NS144,
  NS151, NS154, NS155, NS211 Ins) are absent from it. FOOOF: 41 (NS205 Eng
  missing). Entry 12 text corrected; the legacy primary uses the same
  list, so the comparison there is like-for-like. Whether the 7 should
  be added to the neural list is an open question (H3 covers two of them).
- [~] **I20. State-conditioned wavelet spectrograms per Y7 network (2026-09-15;
  `attn_explore_14Sep26_spectrograms.py` -> `spectrograms/maps.npz` +
  `reports/spectrograms/*.png`; third page
  `reports/attn_explore_14Sep26_spectrograms.html`, published
  https://claude.ai/artifact/7GVmDijQdST3wMtJnnknnV).** Windowed wavelets
  (`wavelet_windowed_10s`, pow_log_mean 76 freqs x 236 win) -> contact
  robust z -> network mean -> minus the recording's mean spectrum over good
  windows -> split by conj_0.6 state (Internal / External / Neutral;
  FarExternal excluded) -> nanmean across recordings; plus per-state mean
  spectra. Descriptive only. STATUS: English 12 recordings done; windowed
  files existed for English only, so `window_continuous_wavelets.py` is
  running (vids hungarian, inscapes, english; OVERWRITE False) to build the
  rest - ~12.5 GB read per recording, disk-bound, hours. Four English
  windowed files (NS174_03, NS193, NS194, NS201_02) were TRUNCATED (bad
  object header) and were deleted so the job regenerates them (TODO A11
  case). When the job finishes: rerun the spectrogram script and republish.
  Note: raw robust z is offset negative in every state because the window
  MEAN of log power sits below the continuous MEDIAN used as the z centre;
  the recording-relative subtraction removes that offset.
- [x] **I21. State-averaged spectra + Int-Ext contrast per frequency (2026-09-15;
  `attn_explore_14Sep26_spectra_states.py` -> `spectra_states/`; now the
  DEFAULT view of the spectrograms page).** Time collapsed: heatmap of the
  mean contact-relative z spectrum per six-region state (x) x frequency
  (y) per Y7 network, plus Internal (int + far int) minus External (ext +
  far ext) per frequency via MixedLM over contacts with person intercept,
  FDR over 76 freqs within network x film; summary panel networks x freq.
  Gate (Christine): by-timepoint deviation at BOTH ends, internal > 0.6,
  external <= 0 (knobs GATE_INT_COL / GATE_EXT_COL). Films drawn only with
  >= 5 windowed recordings (MIN_REC). English (12 rec): Int-Ext positive in
  delta-beta in DAN (band-mean z 2.1 / 2.8 / 2.2 / 1.5), VN (1.6 / 2.3 /
  2.6 / 2.0), SMN delta-theta (2.2 / 1.4); gamma / HFA slightly negative in
  DAN, VN, SMN; DMN / FPN / VAN / LN positive but weaker. Per-frequency FDR
  mostly not reached at 11-12 persons. Hungarian / Inscapes pending the
  windowing job. Time-resolved maps kept on the page as secondary.
  Running log for this track: `reports/attn_explore_14Sep26.html`
  (published artifact https://claude.ai/code/artifact/819b3762-7a83-47f9-b495-dde471173f35;
  republish the same file to update it). Entry 1 = feature generation +
  PC1/PC2 loading figures.
  PAUSED here (Christine, 2026-09-14). Next candidates: deviation + labels
  on this PC1; feature sets that add ISC back as its own axis; ISC-by-label
  validation.

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

- [x] **G5. `rawdata_for_conversion` moved to the Data drive.** DONE 2026-09-09.
  339 GB / 3038 files. Verified before deleting the source: full filename+size
  comparison identical, 6 random content checksums matched (incl. a 766 MB
  .tev), rsync exit 0. Samsung source then deleted.
  **Samsung: 107 GB free -> 446 GB free (95% -> 77% used).**
  Now at `/media/christine/Data/Movie_data/rawdata_for_conversion`.

- [ ] **G6. `plot_elec_anatomy_attn_effects.py:463` is now BROKEN.**
  It hardcodes `f'/{machine_path}/Samsung/Movie_data/rawdata_for_conversion'`,
  which no longer exists — the data is on the Data drive. Deliberately deferred
  (not a priority 2026-09-09), so this is a known breakage, not a surprise.
  Fix: `raw_dir = paths.find('rawdata_for_conversion')`.
  (`find_ekgs.py:16` matches the same name but points at `AV40_data/`, a
  different tree — unaffected.)

- [ ] **G4. Settle which drive owns what, then record it in `paths.py`.**
  Current de facto split, which is defensible on size grounds:
    Samsung (107 GB free) — legacy power pipeline, raw/nwb, prep, PC features
    Data (6.6 TB free)    — all new wavelet output, anatomy
  Samsung being 95% full means new large output must go to Data regardless.
  The decision to record is whether Samsung eventually becomes archive-only.

## H. Return to for paper reporting

- [ ] **H1. Bootstrap CIs on PC1 congruence across movies.** Parked
  2026-09-10 (Christine: important, not now). `compare_pc1_across_movies.py`
  resamples recordings within each video (N_BOOT = 100, env-overridable),
  refits the robust PCA, and writes `pc1_congruence_bootstrap.csv` with 95%
  percentile CIs on Tucker's phi vs the pooled PC1 and vs the video's own
  full-sample PC1. Report the CI next to each point estimate (English 0.83,
  Hungarian 0.99, Inscapes 0.88); `phi_vs_own` is the within-condition
  stability reference the pooled value should be read against.
  Point estimates, the PC1/PC2 cross-congruence (the "same plane, rotated
  axis" result) and the figure are done and in
  `Movie_data/pc1_across_movies_10s/`.
  **NOTE:** the `pc1_congruence_bootstrap.csv` on disk was produced BEFORE
  NS190 run-02 was excluded (A5c) and is stale; re-run with the default
  N_BOOT=100 (`python3 analysis_scripts/compare_pc1_across_movies.py`,
  ~15 min) before reporting.

- [ ] **A5c. Eye-measure outlier log (2026-09-10, from item 3 descriptives).**
  Per-recording means, |z| > 2.5 within video, `descriptives_eye_10s/`:
  - **NS190 run-02 english — EXCLUDED.** Mean gaze disparity −0.18 vs ~+0.02
    typical: sign-flipped and an order of magnitude off (vergence −3.9 SD,
    vergence SD +3.8). Calibration / eye-swap artefact, not behaviour. Also
    the run A5 flagged as the worse of NS190's two; neither run has a wavelet.
    Removed from `robust_pca_gaze_features.py` both_movies list; eye set is
    now english 17 / hungarian 14 / inscapes 18 = 49 recording-videos, 25
    entries in the PCA. Neural usable set unchanged at 43.
  - **NS210 inscapes — INCLUDED for now.** Same disparity pattern (−3.3 SD),
    lowest pupil SD. Christine: gaze was likely recorded outside the screen,
    inflating the plane-axis values. Keep; revisit if it drives a contrast.
  - **NS174_03 english — INCLUDED, watch.** Lowest ISC in the whole set
    (0.015; −3.1 SD), high blink rate, pupil mean and pupil SD.
  Downstream of the exclusion, re-run: robust PCA -> threshold sweep ->
  PC1 comparison (N_BOOT=0; bootstrap stays parked under H1) -> eye
  descriptives. Done 2026-09-10 in that order.

- [~] **H2. Between-movie mixed-model comparisons of the FOOOF features.**
  FIRST PASS DONE 2026-09-10 on the `rolling_fooof_light` outputs (knee mode,
  1-57 Hz; 40 of 42 usable recordings, NS205 english/inscapes still running):
  `compare_fooof_across_movies.py` -> `descriptives_fooof_10s/`. Result: NO
  aperiodic difference between movies (exponent p = 0.08, offset 0.12, knee
  0.98; ICC 0.92-0.95, person-dominated); the movie effect is PERIODIC THETA -
  Inscapes > English in theta periodic power (d = 0.50), theta peak presence
  (d = 0.52) and peaks fitted (d = 0.53), Holm p <= 0.018; theta CF omnibus
  p = 0.039. English and Hungarian never differ. Has_beta is 1.0 in every
  recording (skipped). Re-run when NS205 lands and when the knee refit
  (`rolling_fooof_knee`, not yet written) exists. Original spec follows.
  Flagged 2026-09-10. Once the per-window aperiodic / periodic extraction
  finishes (the knee-mode refit, C0 / C0b / extract_fooof_knee.py running
  now), run the same `value ~ movie + (1 | person)` analysis used for the eye
  measures and band amplitude (`compare_eye_measures_across_movies.py`,
  `compare_power_across_movies.py` - the latter shows how to reuse
  `fit_measure` and the figure layout on a new table). Recording-level unit:
  median across contacts of each per-window feature's mean, and its SD across
  windows for the variability version.
  Features: aperiodic exponent and offset, **f_knee as its own measure**
  (CLAUDE.md: it may be the state-sensitive parameter), per-band periodic
  power and peak presence/CF (use `{band}_CF_if_present`, NaN when absent,
  never the raw argmax - C0b naming note).
  Same inclusion as item 4 (eye-included, wavelet-extracted; 42 or 43
  recordings depending on NS140_02 english / LH010 hungarian), same NS145
  hungarian caveat (16.5% bad windows, 3x delta/theta variability).
  Expectation to test, from the amplitude result: Inscapes differs from both
  narratives, English and Hungarian do not - if that holds for the exponent /
  knee too, the narrative-vs-abstract split is aperiodic, not oscillatory.

- [ ] **H3. Add NS140_02 english and LH010 hungarian to the rescale power
  chain, then re-run everything that reads it.** Flagged 2026-09-10. Both are
  eye-included with wavelets but have NO files in `full_raw_log_power_rescale`
  or `windowed_power_10s_rescale` (0/6 bands each): LH010 is commented out of
  the hungarian list in `extract_power_per_recording.py` (line ~320) and
  NS140_02 is absent from its english list (pared to NS127_02, NS135-NS138
  after the last C0c batch). Christine is running the two extractions in
  Spyder (all six bands each) followed by `window_bandpass_power_robust.py`.
  When both Tier 2 directories exist, re-run in order:
    1. `summarize_power_descriptives.py`   -> 44 recordings (17 / 14 / 13)
    2. `compare_power_across_movies.py`    -> amplitude + variability LMMs
    3. item 6 / item 7 outputs, once they exist
  Until then every rescale-chain result is on 42 recordings (16 / 13 / 13)
  and item 6 should be built to pick up the two automatically from
  `included_recordings.csv`, not from a hand list.

- [x] **H4. Attention states over movie time (2026-09-10) - reporting note.**
  `attention_state_time_density.py` -> `attention_labels_10s/state_time_clustering.csv`,
  `state_density_{scheme}_z0.6.png`. Circular-shift permutation null (each
  recording's label series rolled by a random offset: preserves within-viewer
  autocorrelation and base rate, destroys cross-viewer alignment).
  **External clusters at a few timepoints** (English 1 at ~310 s, 59% of
  recordings; Hungarian 1 at ~228 s, 50%; Inscapes 5 at ~70 s and 315-335 s,
  up to 55%; null ceiling ~28%; global SD ratios 1.16-1.51, p <= 0.004).
  **Internal exceeds chance at NO timepoint in any movie after FDR**; global
  SD ratios 1.00 / 1.12 / 1.14. Christine: External being stimulus-driven is
  fine; Internal being stimulus-driven ("zoning-out scenes") would be the
  concern, and it is not. Report as: External states cluster in a few
  places, Internal states do not.
  Consequence for item 6: expect raw vs tl deltas to differ for External-
  driven cells and agree for Internal-driven ones; describe the contrast as
  Internal vs baseline. `within_timepoint_dev` removes the clustering by
  construction (Internal SD ratio 0.78-0.82) - not evidence of anything.

- [ ] **H5. STRIDE: use ALL windows in stage 1 of the attention-state contrast
  (2026-09-13).** The English "disappearance" under the mixed model was the
  stride-4 subsampling, not the hierarchy and not individual drivers.
  Diagnostic (`attn_state_pooled_driver_diagnostic.py`,
  `attn_state_contrasts_pooled_t_10s/driver_diagnostic/`): in the pooled
  English cells the patients AGREE - DAN-A alpha 12/14 same sign (top patient
  15% of windows, leave-one-out |t| >= 13.5), DAN-A HFA 13/14, Default C theta
  13/13, Temporal Parietal alpha 12/16. No cell in the hypothesis set depends
  on one person.
  Same LMM, same cells, English:
      stride 4  DAN-A alpha est 0.093 se 0.055 z 1.7 (per-patient dsd 0.195)
      stride 1  DAN-A alpha est 0.144 se 0.048 z 3.0 FDR 0.013 (dsd 0.137);
                HFA -3.0, gamma -2.6, Visual A alpha 2.5, Default B alpha 3.4,
                Default C theta 5.0, Temporal Parietal alpha 4.7
  i.e. the hypothesised DAN-A alpha up / gamma-HFA down / visual / DMN pattern.
  Why: stride 4 leaves ~6 windows per state per contact, so each contact's
  delta is noisy; stage 2's SE correctly reflects that. Overlap between
  windows is NOT a pseudo-replication problem in the two-stage design -
  stage 2 never uses within-contact df, it estimates the variance of delta
  across contacts and persons empirically. Stride 4 was inherited from the
  single-stage (window-level) design where it IS needed. => STRIDE = 1 is the
  right default for compare_attn_states_lmm.py; keep stride 4 as a robustness
  note. Also: English-only refit gives larger z (4.7 vs 3.0) than the joint
  three-video fit because the joint model pools one residual variance across
  videos - consider per-video variance (fit per video, or vc by video).
  DONE 2026-09-13: full stride-1 set generated - Y17/Y7 x pooled/per_video x
  within_subject/within_timepoint x 0.5/0.6/0.7, trim20 - as
  `attn_state_contrasts_10s/*_stride1/`, with `stability_*_stride1.csv/.png`.
  Threshold stability at stride 1 is r(est) 0.98-0.99 for EVERY configuration
  and every movie (English 0.98-0.99 vs 0.52-0.64 at stride 4), sign
  agreement 93-99%: the threshold "instability" was under-sampling. Cells
  significant at all three thresholds: Y17 pooled 82 (was 33 at stride 4).
  Stride-4 runs keep the un-suffixed dir names and are the robustness set.
  DONE 2026-09-14: STRIDE = 1 and STAT = 'mean' are the repo defaults in
  compare_attn_states_lmm.py (Christine's decision, 2026-09-14: mean is the
  reported window statistic for the contrast and gain scripts; the range and
  gain scripts already had it). A plain run now writes the
  '..._mean_stride1' primary directories.

- [~] **H6. `centred_global` eye-normalisation variant (2026-09-14).** Legacy
  normalisation z-scores 8 features per recording (pupil, ISC untouched) and
  then z-scores 10 across movies; it removes each viewer's attentional *range*
  as well as level, so every recording yields ~25 Internal and ~25 External
  windows at z 0.6 (SD 5-7) regardless of how much their gaze actually varied.
  The variant centres all 11 features per recording and scales each feature
  once across the movie, so range differences between viewers survive into
  PC1 and the labels. Knobs (`NORM_SCHEME` / `VARIANT = 'centred_global'`) in
  `compute_norm_eye_features_4Jan26.py`, `build_both_movies_eye_matrix.py`,
  `robust_pca_gaze_features.py`, `make_attention_labels.py`,
  `compare_attn_states_lmm.py`. Legacy is the default everywhere; variant
  outputs go to sibling `*_centred_global` directories and never overwrite
  legacy results. Deviation and PC1 are z-scored within movie (symmetric
  within-subject / within-timepoint columns), fixing the legacy asymmetry.
  Findings so far: PC1 cosine 0.97 with legacy, 73 % variance, per-video PC1s
  converge (English-Inscapes congruence 0.48 -> 0.82). Labels at z 0.6: same
  mean counts (17-23 per state) but SD 11-16 vs 5-7, range 0-58, person ICC of
  External count 0.5-0.6 (legacy 0.02), same-person English-Hungarian label
  rate r 0.6 (legacy -0.1), 15-19 of 49 recordings under 10 windows of a
  state (legacy 1). Pooled vs per-video PC1 labels kappa 0.84-0.93.
  Sweep + per-recording spread tables/figures:
  `attention_threshold_sweep_10s_centred_global/`.
  Stage-2 consequence: with n per state ranging 0-58 the equal-weight
  MixedLM is mis-specified, so `WEIGHTS = 'nwin'` (precision weight
  n_int*n_ext/(n_int+n_ext), sqrt-w transform of endog/exog/random design)
  is the variant default. MIN_WIN 8 was tried first and REJECTED: it drops
  English from 15 persons to 7 (Hungarian 11 -> 8, Inscapes 12 -> 9) because
  sparse labellers have few windows per state on every contact - the very
  selection on attentional range the variant exists to avoid. MIN_WIN 3 +
  weights keeps 14/9/11 persons; weighting is what handles the sparse
  contacts (weighted vs unweighted r 0.99 once MIN_WIN 8 has trimmed).
  Contrast runs (stride 1, mean, z 0.6, Y17 + Y7, both schemes, raw + tl,
  pooled primary, per-video sensitivity) in
  `attn_state_contrasts_10s_centred_global/`, dir tag `_wnwin` = primary,
  `_minwin8_wnwin` / `_minwin8` = sensitivities.
  Legacy vs variant (`compare_legacy_vs_centred_global.py` ->
  `attn_state_legacy_vs_centred_global/`): within-subject raw r(z) 0.80-0.88
  per movie, no sign reversals among cells significant in both, fewer
  significant cells (English 29 -> 18, Hungarian 38 -> 26, Inscapes 35 -> 35),
  English hypothesis cells retained (5/7 hits, 0 contra), Inscapes hypothesis
  support lost (2 hits, 3 contra). Within-timepoint labels: English becomes
  null under pooled PC1 (0 sig, hyp mean z 0.7) but not per-video PC1 (16
  sig); Hungarian/Inscapes pick up gamma/HFA contradictions. Same ordering of
  evidence as legacy: within-subject raw strongest, tl decomposition second,
  within-timepoint weakest.
  Open: decide legacy vs variant as the reported pipeline (variant is the
  more defensible normalisation; legacy gives more power because every
  recording contributes ~25 windows per state); pupil covariate still not
  done.

- [x] **H7. Attentional range as a person-level covariate (2026-09-14).**
  Replaces the centred_global *labelling* route (H6, now a documented
  sensitivity only) after Christine judged it conceptually further from the
  goal: range should be *accounted for*, not used to decide what a state is.
  `compare_attn_states_range.py` refits stage 2 from the legacy stride-1
  mean contact tables. RANGE IS MEASURED ON THE LEGACY PC1 AXIS (revised
  2026-09-14 after Christine flagged that the first version measured range
  along the centred_global variant's PC1 while labels were cut on legacy
  PC1): raw 9 PCA features, centred per recording, one SD per feature
  across the movie's viewers, projected on the legacy pooled loadings,
  oriented high = Internal. Primary moderator `eye_delta` = mean score
  over the recording's Internal windows minus its External windows (the
  eye contrast matched to the neural delta); `range` = SD over all windows
  (r 0.91 with eye_delta). The earlier variant-based measure (`range_cg`)
  correlates only 0.61 with the legacy-axis range - the mismatch was real,
  not cosmetic (variant uses 11 features and different scaling; also the
  stored legacy scores are the PCA of the robust low-rank part, r 0.87
  with X @ loadings). eye_delta: gaze_valid r 0.31, isc_sd r 0.19;
  same-person across movies r -0.02 / -0.20 / 0.45, not a stable trait.
  Model
  `delta ~ 0 + video + video:range_c + (1|person)`, range centred within
  movie, so intercepts are the contrast for an average-range viewer.
  Intercepts reproduce legacy (r 0.96-0.99; intercept SE +2-4%).
  Slope inference is at the PERSON level: the mixed-model slope for a
  person-level covariate was inflated (median |z| 1.7-1.8 vs person-level
  |t| 0.5-0.9, sign agreement 67-79%), so the reported slope is a
  random-effects meta-regression of person-mean delta on range with
  pooled within-person SD for the weights (a person's own SD with 2
  contacts gave one person 100% of the weight) and t on n_persons-2 df.
  Result (legacy-axis eye_delta): intercepts reproduce legacy (r 0.95-
  0.99). Slopes: FDR-significant in 0 of 102 cells for English in every
  design; Hungarian 2 cells in every Y17 design (beta and gamma Limbic A,
  t 5-6.7); Inscapes 0-1. Y7: none. Uncorrected: English 0-4/102 within-
  subject (14-16 within-timepoint, mostly negative), Hungarian 10-16,
  Inscapes 7-12, median t positive in Hungarian/Inscapes (+0.3 to +0.8) and
  negative in English (-0.2 to -0.4). Hypothesis cells: no moderation
  except Hungarian Visual Peripheral / Central HFA (t 6.2 / 3.7,
  uncorrected; wider eye contrast -> larger Int-Ext HFA, opposite to the
  hypothesised HFA sign; survives gaze_valid + isc_sd adjustment).
  Outputs: `attn_state_contrasts_range_10s/{run}/` (cells with int_* and
  slope_*, cells_*_sens, interaction with state x movie and range x movie,
  recording_covariates, covariate_correlations, intercept + slope heatmaps,
  scatter_hyp).
  Reporting: the individual-differences claim becomes "the within-person
  contrast does not scale with the size of the viewer's eye-movement
  contrast, at n = 11-15 persons per movie, except in Hungarian Limbic A
  beta/gamma and, uncorrected, Hungarian visual HFA". Eye contrast size is
  not a stable trait across movies. Open (Christine's question, not yet
  run): the continuous per-contact slope of amplitude on the legacy-axis
  score (gain) as the direct test of whether coupling differs across
  people - delta ~ gain x eye contrast, and the flat delta-vs-eye_delta
  result implies gain falls with eye contrast or the state is categorical.

- [~] **H8. Continuous gain analysis, stage 1 done (2026-09-14).**
  `compute_gaze_gain_contacts.py`: per contact x band, slope of amplitude
  (robust z, mean, stride 1, bad windows out, all windows not just
  labelled) on the legacy-axis eye score (H7 definition), raw and tl; plus
  quadratic and 3-level step (legacy labels) fits with BIC.
  Outputs `attn_state_gain_10s/stride1_mean/` (contacts_gain_{raw,tl},
  recording_scores, summary_by_band_video, gain_distributions.png).
  7302 contacts x 6 bands. Findings: gains are small and mostly positive
  (median +0.01 to +0.02 z per unit eye score in alpha/beta/theta, 63-67%
  positive; HFA/gamma near 0 in English, positive in Hungarian); median
  r2 0.018 (90th pct 0.12) - the eye score explains very little window-to-
  window amplitude variance. BIC prefers linear in 59-73% of contacts,
  step in 9-16%, quadratic 15-30% (BIC penalty favours the 2-parameter
  linear model; not evidence of gradedness by itself). Per contact,
  r(gain, delta from the state contrast) 0.45-0.60 by band, and
  multiplying gain by the recording's eye contrast does not improve the
  match - delta and gain are related but the contrast is not simply
  gain x eye contrast at contact level (both are noisy; r2 ~ 0.02).
  Descriptive person-level r(gain, eye_delta) varies in sign across
  bands/movies with 11-15 persons; no inference yet.
  Stage 2a run as a separate GROUPING analysis (`compare_gaze_gain_regions.py`,
  outputs `regions_{Y17,Y7}/`): gain ~ 0 + video + (1|person) per network x
  band, FDR within video. Gain cells correlate r 0.85-0.94 with the legacy
  within-subject contrast z per movie and are MORE often significant
  (Y17 raw: English 48 vs 29, Hungarian 40 vs 38, Inscapes 34 vs 35; every
  contrast-significant cell is gain-significant with the same sign except
  a handful). Hypothesis cells Y17 raw: 22 hits, 4 contra of 54 (contrast:
  16/6); Y7 raw 12 hits, 0 contra of 24. Alpha DAN-A positive in all three
  movies (z 5.2 / 6.7 / 8.9), alpha Visual Central positive in all three,
  Default C alpha and theta positive in all three, HFA DAN-A negative in
  English and Inscapes (z -3.6 / -3.7). Significant gain cells are almost
  all positive in delta-beta (100%); HFA 71% positive, gamma 80% - the
  hypothesised negative high-frequency coupling appears only in DAN-A and
  is contradicted in DAN-B / Visual Peripheral (Hungarian, Inscapes).
  Gain x movie interaction FDR-significant in 60/102 Y17 cells (raw).
  Stage 2b NOT run: gain vs eye_delta meta-regression across persons
  (the individual-differences question in gain form).
  H8 addendum (2026-09-14): `SCORE = 'mahal'` knob in both gain scripts
  runs the same chain on the legacy Mahalanobis group deviation
  (group_deviation_mahal from the pooled label file: distance from the
  other viewers' mean at the same timepoint, 8 legacy-normalised features,
  unsigned, z-scored within movie). Outputs `stride1_mean_mahal/` (+
  regions_{Y17,Y7}). Deviation gain is a different quantity from PC1 gain:
  cell z correlate r 0.57-0.76 (raw) with PC1-gain cells, no opposite-sign
  cells among those significant in both. Y17 raw significant cells:
  English 21 (PC1 48), Hungarian 46 (40), Inscapes 29 (34). Signature is
  almost uniformly POSITIVE across bands (HFA 100%, gamma 94%, alpha 100%
  of significant cells): amplitude rises in every band as a viewer departs
  from the group, strongest in salience/ventral attention gamma (Hungarian
  z 7.7, Inscapes 6.6) and temporal-parietal alpha (English 5.8). So the
  deviation axis carries a broadband "unusual viewer" signal, not the
  low-up / high-down pattern of the PC1 axis. Hypothesis cells (framed for
  PC1 direction): 10 hits, 5 contra of 54 (PC1 gain: 22 / 4); HFA DAN-A no
  longer negative anywhere; DAN-B / Visual Peripheral HFA positive in
  Hungarian and Inscapes. tl removes most of English (21 -> 1) but not
  Hungarian (46 -> 46), so the English deviation coupling is largely
  stimulus-locked.

- [x] **H7/H8 CORRECTION (2026-09-14, caught by Christine).** The legacy
  chain's `both_movies_unnormed_features_w_pcs_Robust_PCA.csv` is NOT raw:
  its 8 gaze features are already z-scored per recording (mean 0, SD 1 in
  every recording); "unnormed" means before the across-movie stage-2 z.
  Pupil (rescaled upstream) and ISC (0-1) are the only columns carrying
  between-viewer scale. Consequences:
  (a) The legacy Mahalanobis deviation and legacy PC1 both come from this
      per-recording-standardised file, so the PC1-gain and deviation-gain
      analyses in H8 ARE on the same footing (the "9 centred vs 8
      uncentred features" flag in the session was wrong). H8 stands; the
      `pc1` docstring now says so. A `pc1_range` option (range-preserving,
      from the centred_global raw-unit file) exists but has not been run.
  (b) The second H7 version ("range on the legacy PC1 axis") read this file
      and therefore carried range only through pupil and ISC - which is
      why it correlated just 0.61 with the variant range. Fixed:
      `legacy_axis_scores(source='centred')` now projects the
      centred-per-recording RAW-unit features (centred_global dir,
      `..._unnormed_...`; per-recording SD of saccade rate 3.2-13.5),
      scaled per feature across the movie, onto the legacy 9-feature PC1
      loadings. Per-recording SD of the corrected score spans 0.54-2.64
      (4.8x) vs 0.98-1.82 (1.8x) for the standardised version; the two
      recording-level ranges correlate only 0.41, though window-level
      scores correlate 0.92.
  Corrected H7 result (eye_delta on the range-preserving score, 1.2-6.1):
  intercepts still reproduce legacy (r 0.92-0.99); slopes FDR-significant
  in 0 cells in every design (previously the Hungarian Limbic A pair);
  uncorrected 0-4/102 English, 3-8 Hungarian, 3-7 Inscapes, i.e. chance;
  the only notable cell is Hungarian Visual Central HFA (t 6.3,
  uncorrected). eye_delta: gaze_valid r -0.29, isc_sd r 0.08; same-person
  across movies 0.42 / -0.51 / 0.64 (n 8-11). The H7 conclusion is
  unchanged and now rests on a measure that actually preserves range: the
  within-person contrast does not scale with the size of the viewer's
  eye-movement contrast.

- [~] **H9. Robust per-recording eye normalisation (`NORM_SCHEME = 'robust'`,
  2026-09-14).** Christine's decision after questioning the legacy z-scoring:
  per-recording standardisation STAYS (raw per-recording spread is mostly
  tracking quality - Vergence SD varies 9-23x across recordings and
  correlates -0.25 to -0.47 with gaze validity, Blink_Rate -0.6), but three
  stage-1 details change in `compute_norm_eye_features_4Jan26.py`:
  (a) scale = 1.4826 * MAD about the median instead of SD (SD tail-inflated in
  11 Vergence / 21 Vergence_Std recordings of 53, up to 12x MAD; SD fallback
  when MAD = 0, which happens for Blink_Rate in 8 recordings);
  (b) missing windows filled AFTER standardisation at the recording median,
  not with raw 0 before it (raw 0 put 258 no-saccade windows 1.1-3.7 SD below
  the mean in Saccade_Dispersion, up to 52/236 in one recording);
  (c) new per-window `Gaze_Valid_Frac` column (fraction of finite gaze
  samples); windows below `GAZE_VALID_MIN` = 0.5 have all 8 gaze features set
  missing -> 0 and are flagged, distinguishing "no gaze data" from "gaze but
  no saccades" (the latter keeps dispersion missing -> 0; Saccade_Rate = 0
  carries it). Pupil untouched in stage 1 (already rescaled per individual
  upstream). `ISC_MODE` knob: 'raw' (default) | 'within_rec'.
  Threaded through build_both_movies_eye_matrix (carries Gaze_Valid_Frac),
  robust_pca_gaze_features, make_attention_labels (carries the flag into the
  label file), compare_attn_states_lmm; output dirs get the `_robust` suffix,
  legacy untouched. NOT YET RUN in Spyder.
  Prototype from the saved raw per-recording files (scratch; no gaze-validity
  masking, which needs the full rerun): legacy replica reproduces the stored
  PC1 (cosine 1.000, label kappa 0.97). Robust vs legacy: PC1 cosine 0.975,
  score r 0.92, label kappa 0.69 (English 0.71, Hungarian 0.71, Inscapes
  0.65); per-recording counts unchanged (Internal 23 +- 7 vs 24 +- 6). The
  loading shift is where expected: Vergence 0.33 -> 0.17, Vergence_Std
  0.27 -> 0.17, Saccade_Rate -0.58 -> -0.66, Blink_Rate 0.59 -> 0.62; PC1
  explains 62% vs 60%. So the SD-inflated recordings were pulling vergence
  into PC1. ISC: 'within_rec' vs 'raw' changes PC1 by cosine 0.9995 and the
  ISC loading -0.15 -> -0.14 - ISC is nearly orthogonal to the gaze axis
  regardless of its scale, so the scaling choice is not what keeps it out of
  PC1. Decision on ISC_MODE still open; it does not matter for PC1.
  Per-video PCAs (same prototype; legacy replica reproduces the stored
  per-video labels, kappa 0.98): robust vs legacy PC1 cosine / score r /
  label kappa - English 0.996 / 0.95 / 0.75, Hungarian 0.950 / 0.89 / 0.67,
  Inscapes 0.874 / 0.85 / 0.60. Not a uniform shift: Vergence loading falls
  in Inscapes (0.57 -> 0.22) and pooled, but RISES in Hungarian
  (0.32 -> 0.47); Blink_Rate/Saccade_Rate go the opposite way. PC1 variance
  explained drops in every video (63 -> 61, 69 -> 61, 58 -> 52 %). Under MAD
  scaling a tail-heavy recording keeps SD > 1, so its spikes are left for
  R_pca's sparse term instead of being compressed into the bulk; which
  recordings dominate each video's PC1 therefore changes, most in Inscapes
  (widest tails). Per-recording label counts unchanged in all three.
  Per-video with ISC_MODE = 'within_rec' (ISC robust-z per recording):
  vs legacy cosine / score r / kappa - English 0.994 / 0.94 / 0.70,
  Hungarian 0.985 / 0.91 / 0.68, Inscapes 0.866 / 0.85 / 0.60 (same as
  raw-ISC robust). vs raw-ISC robust: PC1 cosine 0.979-0.999 and score r
  0.97-0.997, yet label kappa only 0.81-0.85 - because ISC is one of the 8
  FEATURE_COLS in the Mahalanobis group deviation (sweep_attention_thresholds
  .derive_deviation), so standardising ISC changes the deviation gate, not
  just PC1. ISC loading stays -0.06 to -0.15 in every version; corr(PC1, ISC)
  -0.11 to -0.21. PC1 variance explained: within-ISC 64 / 64 / 53 % vs
  raw-ISC 61 / 61 / 52 %.
  Next: run the chain with NORM_SCHEME = 'robust' (compute_norm ->
  build_both_movies -> robust_pca -> make_attention_labels ->
  compare_attn_states_lmm), then compare against the legacy primary as in
  compare_legacy_vs_centred_global.py, and decide whether robust replaces
  legacy as the reported pipeline. Downstream exclusion of low
  Gaze_Valid_Frac windows is not implemented yet.
