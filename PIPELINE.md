# Pipeline

Authoritative description of the analysis pipelines, as described by Christine
2026-09-09. Where this document and the code disagree, this document states the
intent and the code is what needs fixing.

`CLAUDE.md` holds working constraints and file-format detail. This file holds
structure.

---

## Overview

**Four pipelines share one front-end.** iEEG and eye-tracking data are recorded
in the same sessions and preprocessed in parallel; the eye branch produces
attention-state labels that the neural branches are grouped by.

```
                    RECORDING SESSION (iEEG + eye tracking, simultaneous)
                                    |
              +---------------------+---------------------+
              |                                           |
      iEEG PREPROCESSING                          EYE PREPROCESSING
      filter / channel rejection / reref          gaze feature extraction
              |                                           |
      ARTIFACTUAL WINDOW REJECTION                normalize across recordings
      (separate step)                             PCA over eye features
              |                                   Mahalanobis distance w/ PCs
              |                                           |
              |                                   ATTENTION STATE per window
              |                                           |
              +---------------------+---------------------+
                                    |
     +----------------+-------------+--------------+----------------+
     |                |                            |                |
  POWER         SPECTROGRAM                 FUNCTIONAL           FOOOF
  pipeline      PLOTTING pipeline           CONNECTIVITY         pipeline
  (main)        (visualization)             pipeline             (periodic vs
                                            (partial)             aperiodic)
                                                                  (partial)
```

Attention states from the eye branch are the grouping variable for statistics
and plotting in the neural branches.

---

## Shared front-end

### iEEG preprocessing
Filtering, channel rejection, re-referencing.
`movies_ieeg_preprocess_batch.py` (and the `movie_ieeg_preprocess*` siblings).
Bad channels are handled here.

### Artifactual window rejection — a separate step
`label_bad_windows_continous.py` (manual marking on continuous `.fif`)
`find_artifactual_windows.py` (automated MAD detection, currently post-extraction)

**Current behaviour:** bad windows are **included** in wavelet extraction and in
the subsequent norming and windowing. They are excluded later — from statistical
grouping and inference in the power pipeline, and from plotting in the
spectrogram pipeline.

**Open question (Christine):** whether that is the right place to handle them.
See "Cross-cutting problems" below.

### Eye preprocessing
Eye-movement features extracted from raw gaze data, independently of the iEEG
branch. `compute_eye_measures.py`, `eye_vergence.py`, helpers in `src/eye_helpers.py`.

---

## Cross-cutting problems

These are known weaknesses that affect every pipeline. Christine flagged all
three; they are the highest-value structural fixes.

### 1. Inclusion / exclusion is not logged

Which recordings are included or excluded — for iEEG signal quality *or* eye
tracking quality — **is not robustly recorded anywhere.** It is implicit in the
hand-edited patient lists at each step, so:

- the effective N for any result cannot be recovered from the outputs
- a recording dropped for eye-quality reasons is indistinguishable from one
  dropped for iEEG reasons, or from one simply not yet processed
- different steps can silently operate on different subject sets

**Wanted:** a single per-recording inclusion table (patient, session, run, video,
included Y/N, reason, which modality failed), written once and read by every
step instead of hand-maintained lists.

### 2. Channel metadata access is inconsistent

Atlas and other channel metadata originates in each patient's **electrode
correspondence sheet**, and is imported into the power CSVs — but "not always in
the same manner." Different preprocessing, analysis and plotting steps reach for
it differently.

**Wanted:** one canonical per-recording channel table and one accessor used
everywhere. Related: `*_channel_metadata.csv` already exists beside the wavelet
outputs (columns `label`, `DK_Atlas`, `Y7_Atlas`, `Y17_Atlas`, `AparcAseg_Atlas`)
and is the obvious candidate for that canonical form.

### 3. Bad-window handling is ambiguous

Bad windows currently flow *through* extraction and norming, and are filtered
only at the analysis and plotting stages. Consequences worth deciding on:

- **Norming is contaminated.** Robust z-scoring and rolling means computed over
  data that includes artifactual windows shift the reference for every window,
  including good ones.
- **Wavelet leakage.** Morlet convolution spreads a transient across roughly
  +/- n_cycles/f seconds, so an artifact contaminates neighbouring windows that
  are not themselves marked bad.

Excluding earlier avoids both but creates discontinuities the transform can ring
on, and manual marking does not scale. Not yet decided.

---

## Pipeline 1 — Power (main)

**Old (bandpass) route, to be PRESERVED as reference:**

```
preprocessed iEEG (.fif)
   -> bandpass filter per canonical band
   -> power, log-transformed                 -> Tier 1 CSV (full-res, ~600 Hz)
   -> z-score + robust mean per rolling window (10 s / 7.5 s overlap)
                                             -> Tier 2 CSV (downsampled)
   -> atlas/metadata joined in from electrode correspondence sheet
   -> aggregate across bands + recordings    -> Tier 3 (all_power_wide.csv)
   -> merge with eye/attention labels        -> Tier 4 (*_power_eye_merged.csv)
   -> grouping, plotting, statistics         compare_attn_states_*.py
```

Scripts: `extract_power_es.py`, `extract_power_fc.py`,
`lowpass_power_to_windows.py`, `define_custom_network_atlas.py`,
`compare_attn_states_*.py`.

**New (wavelet-derived) route** — same downstream, different source:
see "The wavelet-first refactor" below.

## Pipeline 2 — Spectrogram plotting (visualization)

```
preprocessed iEEG (.fif)
   -> wavelet extraction AND 10 s windowing in a single step
      (full continuous wavelet was NOT saved — judged too large)
   -> spectrogram plots                      plot_spectrograms_by_atlas.py
                                             plot_spectrograms_by_pat_condition.py
```

Bad windows are removed at the plotting stage.

**This is the pipeline the refactor changes most:** the full continuous wavelet
is now saved, so extraction and windowing are separate steps and the same
wavelet serves plotting and analysis.

## Pipeline 3 — Functional connectivity (partially built)

Connectivity across contacts. `extract_power_fc.py`, `compare_attn_states_fc.py`.
Not yet mapped in detail.

## Pipeline 4 — FOOOF (partially built)

Periodic vs aperiodic components per contact. Reads preprocessed `.fif`
directly rather than the power CSVs — a genuinely separate branch.

```
preprocessed iEEG (.fif)
   -> extract_all_fooof.py        -> rolling_fooof_low_mid_{vid}_{date}/{pat}/
                                     *_rolling_fooof_peaks_long_*.csv
                                     *_rolling_fooof_aperiodic_windows_*.csv
   -> aggregate_oscillatory_peaks.py -> rolling_fooof_aggregated_{date}/
```

Coverage note: hungarian is complete here (16 patients, Jun-Jul 2026) while
english has 13 — this branch is *ahead* of the power branch for hungarian.

## The eye branch (feeds all neural pipelines)

```
raw gaze
   -> compute_eye_measures.py        per-recording eye features
   -> prePCA_agg_norm.py             aggregate + normalize across recordings
   -> robust_pca_gaze_features.py    PCA over normalized features -> PC1..PC4
   -> examine_separate_PCs_together.py
         Mahalanobis distance combined with PCs
         -> attention state per rolling window
         schemes: within_subject_dev, within_timepoint_dev
         thresholded by z_thresh (interpolated into column names)
   -> used for grouping, plotting, statistical tests in the neural pipelines
```

Needs cleanup (Christine). Three candidate copies of `compute_eye_measures.py`
exist; canonical one not yet determined.

---

## The wavelet-first refactor

**The change:** extract and save the full continuous Morlet wavelet once, then
derive every other representation from it, instead of each pipeline extracting
its own.

```
                 preprocessed iEEG (.fif)
                            |
              extract_wavelet_hdf5.py            <- ONE master representation
              pow_tf_dat : ch x freq x time         (full continuous, saved)
                            |
     +----------------+-----+------------+----------------+
     |                |                  |                |
 canonical        windowed           spectrograms      FOOOF /
 band power       log + robust-z     (Pipeline 2)      connectivity
 bandpass_from_   wavelet_extract_                     (Pipelines 3, 4)
 wavelet.py       windows.py
     |                |
     +-------> Tier 2 CSV (bridge, NOT YET WRITTEN)
                      |
              Tier 3 -> Tier 4 -> analyses (unchanged)
```

**Why it matters beyond tidiness:**

- Pipeline 2 previously discarded the continuous wavelet; now plotting and
  analysis derive from the *same* numbers.
- All three video conditions get derived identically — required because
  `despicable_me_hungarian` is the bridge condition between `inscapes` and
  `despicable_me_english` (see CLAUDE.md).
- Pipelines 3 and 4 could also source from the master wavelet rather than
  re-reading `.fif`, though that is not yet decided.

**What is not yet built:** the Tier 2 bridge (TODO B2), and the Tier 2->3->4
reshape/merge (TODO B4).

---

## Open questions

1. Where should bad windows be excluded — before extraction, before norming, or
   at analysis as now? Affects norming validity and wavelet leakage.
2. Should Pipelines 3 and 4 source from the master wavelet, or keep reading
   `.fif` directly?
3. What is the canonical form of the inclusion table, and which step writes it?
4. Which `compute_eye_measures.py` is canonical?
5. Does `bandpass_from_wavelet.py`'s output land on the same 10 s grid as the
   old Tier 2, making it the direct bridge input?
