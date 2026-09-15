#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Assemble the cross-video eye-feature matrix that robust_pca_gaze_features.py
reads (`many_normed_et_features_for_pca_both_movies.csv`).

Until 2026-09-10 this step existed in no script: the both_movies file was
concatenated by hand. Verified against the hand-built file of 2026-06-22: it is
a plain row-wise concatenation of the three per-video files with a `video`
column appended, and NO re-normalization across videos (per-video mean 0 /
sd 1 is preserved; max abs diff vs the hand-built file 5e-08, CSV rounding).

Input (per video), from compute_norm_eye_features_4Jan26.py:
    {EYE_DIR}/many_normed_et_features_for_pca_{vid}.csv
        patient, <11 feature columns>            normalized within video
    {EYE_DIR}/many_et_features_{vid}.csv
        patient, <11 feature columns>            within-recording z only

Output:
    {EYE_DIR}/many_normed_et_features_for_pca_both_movies.csv
    {EYE_DIR}/many_et_features_both_movies.csv
        same columns + `video`

Processing:
    1. read each per-video file, append video = vid
    2. concatenate in the order of `vids`
    3. assert every recording has exactly N_WINDOWS rows and no NaN

No normalization, filtering by inclusion list, or PCA is done here. Inclusion
is applied downstream by robust_pca_gaze_features.py from its `good_entries`.
"""

import os
import sys
import pandas as pd

machine_path = 'media/christine'
drive = 'Samsung'
window_len = 10

vids = ['despicable_me_english', 'despicable_me_hungarian', 'inscapes']

NORM_SCHEME = 'legacy'        # 'legacy' | 'robust' | 'centred_global'  (compute_norm_eye_features_4Jan26.py)
EYE_DIR = f'/{machine_path}/{drive}/Movie_data/6Apr26_norm_eye_features_by_rec_{window_len}s' + ('' if NORM_SCHEME == 'legacy' else f'_{NORM_SCHEME}')
N_WINDOWS = 236 if window_len == 10 else 238

FEATURES = ['Saccade_Rate', 'Vergence', 'Vergence_Std', 'Abs_Vergence',
            'Saccade_Dispersion', 'Saccade_Dispersion_Std', 'Blink_Rate',
            'Blink_Duration', 'Pupil_Avg', 'Pupil_Std', 'ISC']


def build(stem):
    parts = []
    for vid in vids:
        path = os.path.join(EYE_DIR, f'{stem}_{vid}.csv')
        df = pd.read_csv(path)
        missing = [c for c in ['patient'] + FEATURES if c not in df.columns]
        if missing:
            raise KeyError(f'{path}: missing columns {missing}')
        extra = [c for c in ['Gaze_Valid_Frac', 'Verg_Valid_Frac'] if c in df.columns]   # 'robust' scheme flag, carried through
        df = df[['patient'] + FEATURES + extra].copy()
        df['video'] = vid

        counts = df.groupby('patient').size()
        bad = counts[counts != N_WINDOWS]
        if len(bad):
            raise ValueError(f'{vid}: recordings without {N_WINDOWS} rows:\n{bad}')
        n_nan = int(df[FEATURES].isna().sum().sum())
        if n_nan:
            raise ValueError(f'{vid}: {n_nan} NaN feature values in {path}')

        print(f'  {vid:26s} {df.patient.nunique():3d} recordings  {len(df):6d} rows')
        parts.append(df)

    out = pd.concat(parts, ignore_index=True)
    out_path = os.path.join(EYE_DIR, f'{stem}_both_movies.csv')
    out.to_csv(out_path, index=False)
    print(f'  -> {out_path}  {out.shape}\n')
    return out


if __name__ == '__main__':
    print(f'EYE_DIR: {EYE_DIR}\n')
    print('many_normed_et_features_for_pca (for robust_pca_gaze_features.py):')
    build('many_normed_et_features_for_pca')
    print('many_et_features (within-recording z only):')
    build('many_et_features')
