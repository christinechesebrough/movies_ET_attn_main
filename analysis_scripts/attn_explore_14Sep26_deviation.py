"""
Exploration track 'attn_explore_14Sep26', step 2: Mahalanobis deviation of each
viewer from the group at each window, and diagnostics of what it captures.

Inputs ({MOVIE_DATA}/attn_explore_14Sep26/eye_pca/):
    features_robust_{vid}.csv          stage-1+2 features (8 gaze/pupil, robust per
                                       recording), ISC raw, Time
    pooled_features_w_pcs.csv          pooled-fit PC1..PC6
    pervideo_features_w_pcs_{vid}.csv  per-video-fit PC1..PC6

Processing (per video):
    1. Feature set for the deviation: the 8 PCA features PLUS ISC (9). ISC
       enters here and not in the PCA: the deviation subtracts the group at the
       same timepoint, which removes ISC's stimulus-locked half. ISC is
       z-scored across the video first (ISC_SCALE = 'z_video'): with raw 0-1
       ISC its residual variance is ~0.015 against ~1 for the others, and
       covariance shrinkage then cuts its weight in d^2 from the nominal 1/9
       to ~1/30 (measured 2026-09-14; the legacy derive_deviation, raw ISC +
       Ledoit-Wolf, has the same under-weighting).
    2. Group reference at each Time = mean over the OTHER viewers of the video
       (leave-one-out, GROUP_REF = 'loo'); 'all' reproduces the legacy inclusive
       mean for comparison.
    3. Residual R = x - group mean. Covariance = empirical covariance of all
       residuals of the video (COV = 'empirical'; ~3-4k windows x 9 features
       needs no shrinkage, and without it every feature contributes exactly
       1/9 of the mean d^2). 'ledoit' reproduces the legacy estimator.
       d = sqrt(R' S^-1 R).
    4. mahal_z      = z of d within recording        (person-relative)
       mahal_time_z = z of d within timepoint        (group-relative)
    5. Diagnostics per recording: r(d, PC1), r(d, |PC1_z|), r(d, ISC);
       eta^2 of d on Time (stimulus-locked share); d(loo) vs d(all) r;
       per-feature share of d^2 in the TOP 10 % windows of each recording
       (the overall share is 1/9 per feature by construction, so only the
       extremes are informative); mean mahal_z in 5 PC1_z bins.

Outputs ({MOVIE_DATA}/attn_explore_14Sep26/deviation/):
    deviation_{vid}.csv          video, patient, Time, PC1_pooled, PC1_pervideo,
                                 ISC_raw, mahal, mahal_z, mahal_time_z, mahal_all
    per_recording_diagnostics.csv, summary_by_video.csv, feature_share.csv,
    pc1_bins.csv, summary.txt

Deliberately NOT done: no labels, no thresholds, no neural data.
"""
import os, sys
import numpy as np
import pandas as pd
from numpy.linalg import inv
from sklearn.covariance import LedoitWolf, EmpiricalCovariance
from scipy import stats

machine_path = 'media/christine'
MOVIE_DATA = f'/{machine_path}/Samsung/Movie_data'
TRACK = 'attn_explore_14Sep26'
IN_DIR = f'{MOVIE_DATA}/{TRACK}/eye_pca'
OUT_DIR = f'{MOVIE_DATA}/{TRACK}/deviation'
VIDS = ['despicable_me_english', 'despicable_me_hungarian', 'inscapes']
FEATURES = ['Saccade_Rate', 'Saccade_Dispersion', 'Vergence', 'Vergence_Std',
            'Blink_Rate', 'Blink_Duration', 'Pupil_Avg', 'Pupil_Std']
DEV_FEATURES = FEATURES + ['ISC']
GROUP_REF = 'loo'            # 'loo' | 'all'
ISC_SCALE = 'z_video'        # 'z_video' | 'raw'
COV = 'empirical'            # 'empirical' | 'ledoit'
BINS = [-np.inf, -1, -0.3, 0.3, 1, np.inf]
BIN_NAMES = ['< -1', '-1 to -0.3', '-0.3 to 0.3', '0.3 to 1', '> 1']


def zs(s):
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd > 0 else s * np.nan


def deviation(df, cols, ref):
    X = df[cols].to_numpy(float)
    if ref == 'all':
        mu = df.groupby('Time')[cols].transform('mean').to_numpy(float)
    else:                                            # leave-one-out
        g = df.groupby('Time')[cols]
        s = g.transform('sum').to_numpy(float); n = g.transform('count').to_numpy(float)
        mu = (s - X) / (n - 1)
    R = X - mu
    ok = np.isfinite(R).all(axis=1)
    est = EmpiricalCovariance() if COV == 'empirical' else LedoitWolf()
    Sinv = inv(est.fit(R[ok]).covariance_)
    d2 = np.einsum('ij,jk,ik->i', R, Sinv, R)
    contrib = R * (R @ Sinv)                         # per-feature share of d2 (sums to d2)
    return np.sqrt(np.maximum(d2, 0)), contrib


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    pooled = pd.read_csv(f'{IN_DIR}/pooled_features_w_pcs.csv')
    recs, shares, bins, lines = [], [], [], []
    for vid in VIDS:
        f = pd.read_csv(f'{IN_DIR}/features_robust_{vid}.csv')
        pv = pd.read_csv(f'{IN_DIR}/pervideo_features_w_pcs_{vid}.csv')[['patient', 'Time', 'PC1']].rename(columns={'PC1': 'PC1_pervideo'})
        pp = pooled.loc[pooled.video == vid, ['patient', 'Time', 'PC1']].rename(columns={'PC1': 'PC1_pooled'})
        df = f.merge(pp, on=['patient', 'Time']).merge(pv, on=['patient', 'Time'])
        df['ISC_raw'] = df['ISC']
        if ISC_SCALE == 'z_video':
            df['ISC'] = zs(df['ISC'])
        d, contrib = deviation(df, DEV_FEATURES, GROUP_REF)
        d_all, _ = deviation(df, DEV_FEATURES, 'all')
        df['mahal'] = d; df['mahal_all'] = d_all
        df['mahal_z'] = df.groupby('patient')['mahal'].transform(zs)
        df['mahal_time_z'] = df.groupby('Time')['mahal'].transform(zs)
        df['PC1_z'] = df.groupby('patient')['PC1_pooled'].transform(zs)
        df[['video', 'patient', 'Time', 'PC1_pooled', 'PC1_pervideo', 'PC1_z', 'ISC_raw', 'mahal', 'mahal_z', 'mahal_time_z', 'mahal_all']].to_csv(f'{OUT_DIR}/deviation_{vid}.csv', index=False)

        # feature share of squared distance in each recording's top-10 % windows
        top = df.groupby('patient')['mahal'].transform(lambda s: s >= s.quantile(0.9)).to_numpy(bool)
        sh = pd.Series(contrib[top].sum(axis=0) / (d[top] ** 2).sum(), index=DEV_FEATURES)
        shares.append(sh.rename(vid))

        # per-recording diagnostics
        for rec, g in df.groupby('patient'):
            recs.append(dict(video=vid, patient=rec,
                             r_pc1=np.corrcoef(g.mahal, g.PC1_pooled)[0, 1],
                             r_abs_pc1=np.corrcoef(g.mahal, g.PC1_z.abs())[0, 1],
                             r_isc=np.corrcoef(g.mahal, g.ISC_raw)[0, 1],
                             r_loo_all=np.corrcoef(g.mahal, g.mahal_all)[0, 1],
                             mean_d=g.mahal.mean(), sd_d=g.mahal.std(), skew_d=stats.skew(g.mahal)))
            b = pd.cut(g.PC1_z, BINS, labels=BIN_NAMES)
            bins.append(g.groupby(b, observed=False)['mahal_z'].mean().rename((vid, rec)))

        # stimulus-locked share of d
        grp = df.groupby('Time')['mahal']
        eta = ((grp.transform('mean') - df.mahal.mean()) ** 2).sum() / ((df.mahal - df.mahal.mean()) ** 2).sum()
        R = pd.DataFrame([r for r in recs if r['video'] == vid])
        def tt(col):
            z = np.arctanh(R[col].clip(-0.999, 0.999)); return np.tanh(z.mean()), stats.ttest_1samp(z, 0).pvalue, int((R[col] < 0).sum())
        m1, p1, n1 = tt('r_pc1'); m2, p2, n2 = tt('r_abs_pc1'); m3, p3, n3 = tt('r_isc')
        lines.append(f'{vid}: {len(R)} recordings | eta2(Time) of d = {eta:.3f} | r(d_loo, d_all) median {R.r_loo_all.median():.3f} | '
                     f'mean d {R.mean_d.mean():.2f} skew {R.skew_d.median():.2f} | per-rec r(d, PC1) mean {m1:+.3f} p {p1:.3g} ({n1} neg) | '
                     f'r(d, |PC1_z|) mean {m2:+.3f} p {p2:.3g} ({len(R)-n2} pos) | r(d, ISC) mean {m3:+.3f} p {p3:.3g} ({n3} neg)')
        lines.append('   feature share of d^2 in top-10% windows: ' + '  '.join(f'{k} {v:.2f}' for k, v in sh.items()))

    pd.DataFrame(recs).to_csv(f'{OUT_DIR}/per_recording_diagnostics.csv', index=False)
    pd.concat(shares, axis=1).to_csv(f'{OUT_DIR}/feature_share.csv')
    B = pd.concat(bins, axis=1).T; B.index = pd.MultiIndex.from_tuples(B.index, names=['video', 'patient'])
    B.to_csv(f'{OUT_DIR}/pc1_bins.csv')
    bm = B.groupby(level='video').agg(['mean', 'sem'])
    for vid in VIDS:
        lines.append(f'{vid}: mean mahal_z by PC1_z bin: ' + '  '.join(f'{b} {bm.loc[vid, (b, "mean")]:+.2f}±{bm.loc[vid, (b, "sem")]:.2f}' for b in BIN_NAMES))
    open(f'{OUT_DIR}/summary.txt', 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines)); print(f'\n-> {OUT_DIR}')
