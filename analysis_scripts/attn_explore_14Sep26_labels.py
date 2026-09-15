"""
Exploration track 'attn_explore_14Sep26', step 3: attention-state labels from
the PC1 x deviation plane, under the rules the log's entries 8-11 justify.

Inputs ({MOVIE_DATA}/attn_explore_14Sep26/):
    deviation/deviation_{vid}.csv    PC1_pooled, PC1_z (within viewer), mahal,
                                     mahal_z, mahal_time_z, ISC_raw, Time
    eye_pca/features_robust_{vid}.csv  the 8 standardised features (for the two
                                     cluster scores) + Gaze_Valid_Frac, Verg_Valid_Frac

Rules (t = threshold; z = within-viewer z unless noted; every rule also marks
PC1_z < -1.5 as 'FarExternal', a separate category, except 'legacy'):
    conj       Internal: PC1_z > t AND mahal_z > t        External: -1.5 <= PC1_z < -t
               (the justified rule: conjunction for internal, axis alone for external)
    legacy     Internal: PC1_z > t AND mahal_z > t        External: PC1_z < -t AND mahal_z < -t
               (the legacy symmetric rule, on the new axis; for comparability)
    pc1only    Internal: PC1_z > t                        External: -1.5 <= PC1_z < -t
               (no deviation at all: measures what the internal gate adds)
    syncext    Internal as conj                           External: -1.5 <= PC1_z < -t AND ISC_z > 0
               (external as a positive synchrony definition)
    conj_tp    within-TIMEPOINT z: PC1_time_z > t AND mahal_time_z > t / PC1_time_z < -t
               (scheme sensitivity)
    clusterA   conj rule with the saccade-blink score in place of PC1
    clusterB   conj rule with the vergence-pupil score in place of PC1
               (which cluster carries the contrast)
    Thresholds: 0.6 (legacy comparability) and 0.8 (where deviation begins to
    rise along PC1, entry 8).

Output:
    {MOVIE_DATA}/attention_labels_10s/attention_labels_10s_pooled_explore14Sep26.csv
        columns as make_attention_labels.py writes them (video, patient, pat, run,
        Time, window_idx_10s, PC1, PC1_z, PC1_time_z, group_deviation_mahal,
        group_dev_mahal_z, mahal_time_z, ISC_z, clusterA_z, clusterB_z,
        Gaze_Valid_Frac, Verg_Valid_Frac) + label_{rule}_{t}
        -> usable by compare_attn_states_lmm.py with VARIANT = 'explore14Sep26',
           SCHEME = one of the rule names, THRESHOLDS = [0.6] or [0.8].
    {MOVIE_DATA}/attn_explore_14Sep26/labels/label_counts_per_viewer.csv

Deliberately NOT done: no neural data, no bad-window exclusion (the contrast
script applies its own), no cross-rule reconciliation.
"""
import os, re
import numpy as np
import pandas as pd

machine_path = 'media/christine'
MOVIE_DATA = f'/{machine_path}/Samsung/Movie_data'
TRACK = f'{MOVIE_DATA}/attn_explore_14Sep26'
OUT_LABELS = f'{MOVIE_DATA}/attention_labels_10s/attention_labels_10s_pooled_explore14Sep26.csv'
OUT_DIR = f'{TRACK}/labels'
VIDS = ['despicable_me_english', 'despicable_me_hungarian', 'inscapes']
THRESHOLDS = [0.6, 0.8]
FAR_EXT = -1.5


def zs(s):
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd > 0 else s * np.nan


def lab(int_mask, ext_mask, far_mask=None):
    out = np.full(len(int_mask), 'Neutral', dtype=object)
    out[ext_mask.to_numpy()] = 'External'
    out[int_mask.to_numpy()] = 'Internal'
    if far_mask is not None:
        out[far_mask.to_numpy()] = 'FarExternal'
    return out


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    parts = []
    for vid in VIDS:
        d = pd.read_csv(f'{TRACK}/deviation/deviation_{vid}.csv')
        f = pd.read_csv(f'{TRACK}/eye_pca/features_robust_{vid}.csv')
        d = d.merge(f[['patient', 'Time', 'Saccade_Rate', 'Blink_Rate', 'Vergence', 'Vergence_Std', 'Pupil_Avg']
                      + [c for c in ['Gaze_Valid_Frac', 'Verg_Valid_Frac'] if c in f.columns]], on=['patient', 'Time'])
        d['video'] = vid
        g = d.groupby('patient')
        d['PC1_time_z'] = d.groupby('Time')['PC1_pooled'].transform(zs)
        d['ISC_z'] = g['ISC_raw'].transform(zs)
        d['clusterA_z'] = g[['Blink_Rate', 'Saccade_Rate']].transform(lambda s: s).pipe(lambda x: (x['Blink_Rate'] - x['Saccade_Rate']) / np.sqrt(2))
        d['clusterA_z'] = d.groupby('patient')['clusterA_z'].transform(zs)
        d['clusterB_z'] = (d['Vergence'] + d['Vergence_Std'] + d['Pupil_Avg']) / np.sqrt(3)
        d['clusterB_z'] = d.groupby('patient')['clusterB_z'].transform(zs)
        far = d.PC1_z < FAR_EXT
        for t in THRESHOLDS:
            hi, lo = d.PC1_z > t, (d.PC1_z < -t) & ~far
            d[f'label_conj_{t}'] = lab(hi & (d.mahal_z > t), lo, far)
            d[f'label_legacy_{t}'] = lab(hi & (d.mahal_z > t), (d.PC1_z < -t) & (d.mahal_z < -t))
            d[f'label_pc1only_{t}'] = lab(hi, lo, far)
            d[f'label_syncext_{t}'] = lab(hi & (d.mahal_z > t), lo & (d.ISC_z > 0), far)
            d[f'label_conj_tp_{t}'] = lab((d.PC1_time_z > t) & (d.mahal_time_z > t), (d.PC1_time_z < -t) & ~(d.PC1_time_z < FAR_EXT), d.PC1_time_z < FAR_EXT)
            for cl in ['clusterA', 'clusterB']:
                c = d[f'{cl}_z']; farc = c < FAR_EXT
                d[f'label_{cl}_{t}'] = lab((c > t) & (d.mahal_z > t), (c < -t) & ~farc, farc)
        parts.append(d)
    out = pd.concat(parts, ignore_index=True)
    m = out['patient'].str.extract(r'^(?P<pat>.+?)(?:_ses-\d+)?(?:_(?P<run>run-\d+))?$')
    out['pat'] = m['pat']; out['run'] = m['run'].fillna('run-01')
    out['window_idx_10s'] = out['Time'].astype(int) - 1
    out = out.rename(columns={'PC1_pooled': 'PC1', 'mahal': 'group_deviation_mahal', 'mahal_z': 'group_dev_mahal_z'})
    keep = ['video', 'patient', 'pat', 'run', 'Time', 'window_idx_10s', 'PC1', 'PC1_z', 'PC1_time_z',
            'group_deviation_mahal', 'group_dev_mahal_z', 'mahal_time_z', 'ISC_raw', 'ISC_z', 'clusterA_z', 'clusterB_z'] + \
           [c for c in ['Gaze_Valid_Frac', 'Verg_Valid_Frac'] if c in out.columns] + [c for c in out.columns if c.startswith('label_')]
    out[keep].to_csv(OUT_LABELS, index=False)
    rules = [c for c in out.columns if c.startswith('label_')]
    cnt = out.groupby(['video', 'patient'])[rules].agg(lambda s: pd.Series({'Internal': (s == 'Internal').sum(), 'External': (s == 'External').sum(), 'FarExternal': (s == 'FarExternal').sum()}).to_dict())
    rows = []
    for (vid, pat), r in cnt.iterrows():
        for rule in rules:
            rows.append(dict(video=vid, patient=pat, rule=rule.replace('label_', ''), **r[rule]))
    C = pd.DataFrame(rows); C.to_csv(f'{OUT_DIR}/label_counts_per_viewer.csv', index=False)
    summ = C.groupby(['rule', 'video'])[['Internal', 'External', 'FarExternal']].agg(['mean', 'std', 'min']).round(1)
    pd.set_option('display.width', 250); print(summ.to_string())
    print(f'\n-> {OUT_LABELS}\n-> {OUT_DIR}/label_counts_per_viewer.csv')
