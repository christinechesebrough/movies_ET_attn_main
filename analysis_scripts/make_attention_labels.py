#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Write the attention-state labels used by every downstream neural contrast,
from the CURRENT robust-PCA output, at the reported threshold and its two
sensitivity values, for both deviation schemes.

Replaces reading `shared_PC_features_10s_20Apr26/{vid}_features_df_[z, z].csv`,
which were produced from the pre-2026-09-10 PC file (old pupil grid, no LH010,
NS190 run-02 included) and are stale.

Input:
    {EYE_DIR}/both_movies_unnormed_features_w_pcs_Robust_PCA.csv

Processing: sweep_attention_thresholds.derive_deviation() and .label() -
the exact logic of examine_separate_PCs_together.py, verified 100% identical
on the old inputs (see that script's docstring).

PC_SOURCE:
    'pooled'     PC1 from the single robust PCA over all three videos (the
                 pipeline default; robust_pca_gaze_features.py vid='both_movies')
    'per_video'  PC1 from a robust PCA fit separately to each video (the
                 'separate' mode of that script), oriented so corr(PC1, ISC) < 0.
                 Sensitivity analysis: per-video PC1s agree with the pooled
                 labels at kappa 0.71-0.82 (2026-09-10) but define Internal by
                 different feature weights in English (oculomotor) vs Inscapes
                 (vergence/pupil) - see pc1_across_movies_10s/.

Output:
    {OUT_DIR}/attention_labels_10s_{PC_SOURCE}.csv   long, one row per (video, patient, Time):
        video, patient, pat, run, Time, window_idx_10s (= Time - 1), PC1, PC1_z,
        group_deviation_mahal, group_dev_mahal_z, mahal_time_z,
        label_{scheme}_{z}  for scheme in (within_subject_dev, within_timepoint_dev)
                            and z in THRESHOLDS  ('Internal' / 'External' / 'Neutral')

Deliberately NOT done: no bad-window information (join on window_idx_10s
against bad_windows_10s/), no neural data.
"""

import os
import re
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sweep_attention_thresholds as sw
import compare_pc1_across_movies as C

machine_path = 'media/christine'
drive = 'Samsung'
# VARIANT (2026-09-14):
#   'legacy'          eye features from the original two-stage z; PC1 and the
#                     deviation z-scored within patient (subject-relative);
#                     within_timepoint columns re-normalise the deviation only.
#   'centred_global'  eye features centred per recording and scaled per movie
#                     (compute_norm NORM_SCHEME='centred_global'); PC1 and the
#                     deviation z-scored within MOVIE, so label rate per viewer
#                     reflects their attentional range; the within_timepoint
#                     columns are symmetric (PC1 AND deviation within timepoint).
VARIANT = 'legacy'            # 'legacy' | 'robust' | 'centred_global'  (compute_norm_eye_features_4Jan26.py NORM_SCHEME)
EYE_DIR = f'/{machine_path}/{drive}/Movie_data/6Apr26_norm_eye_features_by_rec_10s' + ('' if VARIANT == 'legacy' else f'_{VARIANT}')
OUT_DIR = f'/{machine_path}/{drive}/Movie_data/attention_labels_10s'
THRESHOLDS = [0.5, 0.6, 0.7]
PC_SOURCE = 'pooled'          # 'pooled' | 'per_video'

if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    feats = sw.load_features(os.path.join(EYE_DIR, 'both_movies_unnormed_features_w_pcs_Robust_PCA.csv'))
    if PC_SOURCE == 'per_video':
        # replace the pooled PC1 with each video's own robust-PCA PC1 (ISC-oriented)
        X, _ = C.load_inputs()
        X['Time'] = X.groupby(['video', 'patient']).cumcount() + 1
        sc = []
        for vid in sw.VIDEOS:
            dv = X[X['video'] == vid]
            _, scores, evr = C.fit_robust_pca(dv[C.FEATURE_ORDER].to_numpy(float))
            print(f'  per-video PCA {vid}: PC1 explains {evr[0]*100:.1f}%')
            sc.append(dv[['video', 'patient', 'Time']].assign(PC1_pv=scores[:, 0]))
        feats = feats.merge(pd.concat(sc), on=['video', 'patient', 'Time'], how='inner')
        feats['PC1'] = feats['PC1_pv']
    parts = []
    for vid in sw.VIDEOS:
        df = sw.derive_deviation(feats[feats['video'] == vid], scope='movie' if VARIANT == 'centred_global' else 'subject')
        for scheme, dev_col in sw.SCHEMES.items():
            # legacy: PC1_z (within patient) for both schemes.  centred_global:
            # PC1_z (within movie) with the subject scheme, PC1_time_z with the
            # timepoint scheme, so the two axes are always normalised alike.
            pc_col = 'PC1_time_z' if (VARIANT == 'centred_global' and scheme == 'within_timepoint_dev') else 'PC1_z'
            pc = df[pc_col].to_numpy(float); dev = df[dev_col].to_numpy(float)
            for z in THRESHOLDS:
                df[f'label_{scheme}_{z}'] = sw.label(pc, dev, z, z)
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    m = out['patient'].str.extract(r'^(?P<pat>.+?)(?:_ses-\d+)?(?:_(?P<run>run-\d+))?$')
    out['pat'] = m['pat']; out['run'] = m['run'].fillna('run-01')
    out['window_idx_10s'] = out['Time'].astype(int) - 1
    keep = ['video', 'patient', 'pat', 'run', 'Time', 'window_idx_10s', 'PC1', 'PC1_z', 'PC1_time_z',
            'group_deviation_mahal', 'group_dev_mahal_z', 'mahal_time_z'] + \
           [c for c in ['Gaze_Valid_Frac', 'Verg_Valid_Frac'] if c in out.columns] + \
           [c for c in out.columns if c.startswith('label_')]
    tag = '' if VARIANT == 'legacy' else f'_{VARIANT}'
    out[keep].to_csv(os.path.join(OUT_DIR, f'attention_labels_10s_{PC_SOURCE}{tag}.csv'), index=False)
    print(out.groupby('video').patient.nunique().to_dict(), len(out), 'rows')
    for z in THRESHOLDS:
        print(f'  z={z}:', out[f'label_within_subject_dev_{z}'].value_counts().to_dict())
    print(f'-> {OUT_DIR}/attention_labels_10s_{PC_SOURCE}{tag}.csv')
