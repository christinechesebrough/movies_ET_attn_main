"""
Exploration track 'attn_explore_14Sep26' (dummy name), step 1: robust PCA of
the per-recording-standardised gaze features, pooled across videos and per
video, WITHOUT ISC in the feature set. ISC is carried alongside as its own
column for later feature sets and for orienting PC1.

Parallel to the legacy chain (compute_norm_eye_features_4Jan26 ->
build_both_movies_eye_matrix -> robust_pca_gaze_features); never writes into
legacy directories.

Inputs (SOURCE knob):
  SOURCE = 'compute_norm_robust'  (default since the 2026-09-14 rerun)
    {MOVIE_DATA}/6Apr26_norm_eye_features_by_rec_10s_robust/many_normed_et_features_for_pca_{vid}.csv
        produced by compute_norm_eye_features_4Jan26.py with NORM_SCHEME =
        'robust': stage 1 robust z per recording (8 gaze features, MAD scale,
        gaze-validity mask, missing -> recording median), vergence from
        MEASURED samples only, stage 2 z across the video; plus ISC (raw),
        Gaze_Valid_Frac, Verg_Valid_Frac. Steps 1-2 below are then skipped.
  SOURCE = 'raw_legacy_files'  (the first pass, before the rerun)
    {MOVIE_DATA}/6Apr26_norm_eye_features_by_rec_10s/{rec}/many_normed_et_features_{vid}_{pat}.csv
        RAW windowed features per recording (legacy interpolated vergence);
        steps 1-2 applied here.
    {MOVIE_DATA}/6Apr26_norm_eye_features_by_rec_10s/both_movies_unnormed_features_w_pcs_Robust_PCA.csv
        the legacy PC1 loadings for comparison (both sources), and the
        inclusion set for 'raw_legacy_files'

Processing:
    1. stage 1  per recording, 8 gaze features: (x - median) / (1.4826 * MAD),
                missing -> 0 after (src/eye_normalise.standardise_recording).
                Pupil_Avg / Pupil_Std untouched (already rescaled per
                individual upstream). ISC untouched.
    2. stage 2  per video, z across all windows of the 10 non-ISC features
                (near-identity for the 8, matters for pupil) - as legacy.
    3. robust PCA (R_pca with lmbda = LAMBDA -> PCA on L, 6 components) on FEATURES (8, no ISC):
                pooled over the three videos, and per video.
                PC1 oriented so that it correlates negatively with ISC
                (high PC1 = Internal, the legacy convention); PC2 so that
                Vergence loads positively (display only).

Outputs ({MOVIE_DATA}/attn_explore_14Sep26/eye_pca/):
    features_robust_{vid}.csv          stage-1+2 features + ISC + Gaze_Valid_Frac if present
    pooled_features_w_pcs.csv          all videos, features + ISC + PC1..PC6 (pooled fit)
    pooled_loadings.csv                8 features x PC1..PC6
    pervideo_features_w_pcs_{vid}.csv  per-video fit scores
    pervideo_loadings_{vid}.csv
    summary.csv                        variance explained, cosines vs legacy and pooled vs per-video
    summary.txt                        same, readable

Deliberately NOT done: no Gaze_Valid_Frac masking (that column comes from the
compute_norm rerun and is absent from the current raw files), no deviation /
labels, no legacy-style ISC in the PCA.
"""
import os, sys, glob
import numpy as np
import pandas as pd

machine_path = 'media/christine'
sys.path.append(f'/{machine_path}/Samsung/scripts')                            # r_pca lives here
sys.path.append(f'/{machine_path}/Samsung/scripts/movies_ET_attn_main/src')
from r_pca import R_pca
from eye_normalise import standardise_recording
from sklearn.decomposition import PCA

MOVIE_DATA = f'/{machine_path}/Samsung/Movie_data'
TRACK = 'attn_explore_14Sep26'
LEGACY_DIR = f'{MOVIE_DATA}/6Apr26_norm_eye_features_by_rec_10s'
ROBUST_DIR = f'{MOVIE_DATA}/6Apr26_norm_eye_features_by_rec_10s_robust'
SOURCE = 'compute_norm_robust'   # 'compute_norm_robust' | 'raw_legacy_files'
INCLUDE_LEGACY_SET = True        # restrict to the recordings of the legacy pooled PCA (17/14/18), for
                                 # comparability; the robust feature file has 4 more English recordings
                                 # (NS166, NS190 r1, NS190 r2, NS193 r1) that robust_pca_gaze_features
                                 # excluded via its good_entries list
OUT_DIR = f'{MOVIE_DATA}/{TRACK}/eye_pca'
VIDS = ['despicable_me_english', 'despicable_me_hungarian', 'inscapes']

GAZE8 = ['Saccade_Rate', 'Vergence', 'Vergence_Std', 'Abs_Vergence', 'Saccade_Dispersion',
         'Saccade_Dispersion_Std', 'Blink_Rate', 'Blink_Duration']        # stage-1 robust z
STAGE2 = GAZE8 + ['Pupil_Avg', 'Pupil_Std']                                # across-movie z
FEATURES = ['Saccade_Rate', 'Saccade_Dispersion', 'Vergence', 'Vergence_Std',
            'Blink_Rate', 'Blink_Duration', 'Pupil_Avg', 'Pupil_Std']      # PCA set, legacy order minus ISC
CARRY = ['ISC', 'Gaze_Valid_Frac']                                         # kept, not in the PCA
N_COMP = 6
LAMBDA = 0.05     # R_pca sparse-term weight. Default 1/sqrt(n) ~ 0.009 is degenerate here (S absorbs 87 % of
                  # the data, L keeps 2 %; TODO I5). 0.05 gives a genuinely sparse S (~0.3 % of entries).


def load_raw(vid, include):
    parts = []
    for f in sorted(glob.glob(f'{LEGACY_DIR}/*/many_normed_et_features_{vid}_*.csv')):
        d = pd.read_csv(f)
        if d['patient'].iloc[0] in include:
            parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    missing = include - set(df['patient'])
    if missing:
        raise FileNotFoundError(f'{vid}: no raw file for {sorted(missing)}')
    return df


def normalise(df):
    out = pd.concat([standardise_recording(g, GAZE8, 'robust', verbose_name=rec)
                     for rec, g in df.groupby('patient', sort=False)], ignore_index=True)
    out[STAGE2] = (out[STAGE2] - out[STAGE2].mean()) / out[STAGE2].std()
    out['Time'] = out.groupby('patient').cumcount() + 1
    return out


def fit_robust_pca(df):
    X = df[FEATURES].to_numpy(float)
    L, S = R_pca(X, lmbda=LAMBDA).fit(max_iter=10000, iter_print=10**9)
    print(f'    R_pca lambda {LAMBDA}: sparse part {np.mean(np.abs(S) > 1e-6)*100:.2f}% of entries nonzero, '
          f'low-rank part keeps {(L**2).sum()/(X**2).sum()*100:.1f}% of the energy')
    pca = PCA(n_components=N_COMP)
    scores = pca.fit_transform(L)
    comps = pca.components_.copy()
    if np.corrcoef(scores[:, 0], df['ISC'].to_numpy(float))[0, 1] > 0:   # high PC1 = low ISC
        scores[:, 0] *= -1; comps[0] *= -1
    if comps[1, FEATURES.index('Vergence')] < 0:
        scores[:, 1] *= -1; comps[1] *= -1
    return comps, scores, pca.explained_variance_ratio_


def cosine(a, b):
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    legacy = pd.read_csv(f'{LEGACY_DIR}/both_movies_unnormed_features_w_pcs_Robust_PCA.csv')
    leg_load = pd.read_csv(f'{LEGACY_DIR}/both_movies_pca_loadings.csv', index_col=0)['PC1'].to_numpy()[:8]
    # legacy loadings file order = FEATURES + ISC; drop ISC and compare on the 8

    feats = {}
    for vid in VIDS:
        if SOURCE == 'compute_norm_robust':
            df = pd.read_csv(f'{ROBUST_DIR}/many_normed_et_features_for_pca_{vid}.csv')
            if INCLUDE_LEGACY_SET:
                include = set(legacy.loc[legacy['video'] == vid, 'patient'])
                dropped = sorted(set(df['patient']) - include)
                df = df[df['patient'].isin(include)].copy()
                if dropped:
                    print(f'{vid}: excluded (not in legacy pooled set): {dropped}')
            df['Time'] = df.groupby('patient').cumcount() + 1
            n_nan = int(df[FEATURES].isna().sum().sum())
            if n_nan:
                raise ValueError(f'{vid}: {n_nan} NaN feature values in the robust file')
        else:
            include = set(legacy.loc[legacy['video'] == vid, 'patient'])
            df = normalise(load_raw(vid, include))
        df['video'] = vid
        carry = [c for c in CARRY if c in df.columns]
        df = df[['video', 'patient', 'Time'] + FEATURES + ['Abs_Vergence', 'Saccade_Dispersion_Std'] + carry]
        df.to_csv(f'{OUT_DIR}/features_robust_{vid}.csv', index=False)
        feats[vid] = df
        print(f'{vid}: {df.patient.nunique()} recordings, {len(df)} windows')

    rows, lines = [], []
    pooled = pd.concat(feats.values(), ignore_index=True)
    comps_p, scores_p, evr_p = fit_robust_pca(pooled)
    for i in range(N_COMP):
        pooled[f'PC{i+1}'] = scores_p[:, i]
    pooled.to_csv(f'{OUT_DIR}/pooled_features_w_pcs.csv', index=False)
    pd.DataFrame(comps_p.T, index=FEATURES, columns=[f'PC{i+1}' for i in range(N_COMP)]).to_csv(f'{OUT_DIR}/pooled_loadings.csv')
    r_isc = np.corrcoef(scores_p[:, 0], pooled['ISC'])[0, 1]
    rows.append(dict(fit='pooled', pc1_var=evr_p[0], pc2_var=evr_p[1], cos_pc1_vs_legacy=cosine(comps_p[0], leg_load),
                     cos_pc1_vs_pooled=1.0, r_pc1_isc=r_isc, n_rec=pooled.patient.nunique()))
    lines.append(f'POOLED  PC1 {evr_p[0]*100:.1f}%  PC2 {evr_p[1]*100:.1f}%  cos(PC1, legacy PC1 on 8) {cosine(comps_p[0], leg_load):.3f}  r(PC1, ISC) {r_isc:.3f}')
    lines.append('  loadings PC1: ' + '  '.join(f'{f} {c:+.2f}' for f, c in zip(FEATURES, comps_p[0])))

    for vid in VIDS:
        df = feats[vid].copy()
        comps, scores, evr = fit_robust_pca(df)
        for i in range(N_COMP):
            df[f'PC{i+1}'] = scores[:, i]
        df.to_csv(f'{OUT_DIR}/pervideo_features_w_pcs_{vid}.csv', index=False)
        pd.DataFrame(comps.T, index=FEATURES, columns=[f'PC{i+1}' for i in range(N_COMP)]).to_csv(f'{OUT_DIR}/pervideo_loadings_{vid}.csv')
        r_isc = np.corrcoef(scores[:, 0], df['ISC'])[0, 1]
        pv = pooled.loc[pooled['video'] == vid, 'PC1'].to_numpy()
        rows.append(dict(fit=vid, pc1_var=evr[0], pc2_var=evr[1], cos_pc1_vs_legacy=cosine(comps[0], leg_load),
                         cos_pc1_vs_pooled=cosine(comps[0], comps_p[0]), r_pc1_isc=r_isc,
                         r_score_vs_pooled=np.corrcoef(scores[:, 0], pv)[0, 1], n_rec=df.patient.nunique()))
        lines.append(f'{vid:24s} PC1 {evr[0]*100:.1f}%  PC2 {evr[1]*100:.1f}%  cos vs legacy {cosine(comps[0], leg_load):.3f}  '
                     f'cos vs pooled {cosine(comps[0], comps_p[0]):.3f}  score r vs pooled {np.corrcoef(scores[:, 0], pv)[0, 1]:.3f}  r(PC1, ISC) {r_isc:.3f}')
        lines.append('  loadings PC1: ' + '  '.join(f'{f} {c:+.2f}' for f, c in zip(FEATURES, comps[0])))

    summary = pd.DataFrame(rows)
    summary.to_csv(f'{OUT_DIR}/summary.csv', index=False)
    open(f'{OUT_DIR}/summary.txt', 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'\n-> {OUT_DIR}')
