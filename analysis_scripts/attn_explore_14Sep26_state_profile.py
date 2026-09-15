"""
Exploration track, step 5: neural amplitude across the six PC1 regions (far
external -> far internal), per network and band, Y17 and Y7.

Reuses compare_attn_states_lmm.py's loaders (Tier 2 robust-z windowed power,
atlas rows, bad-window masks, included recordings). For every contact, the
mean amplitude in each region minus the contact's mean over all kept windows
(so contact offsets cancel); then mean and SEM across contacts per (video,
atlas, network, band, region), with the number of persons.

Regions (within-viewer PC1 z, from log entry 9):
    far external < -1.5 | external -1.5..-0.4 | middle -0.4..0.4 |
    ambiguous 0.4..0.8 | internal 0.8..1.2 | far internal > 1.2

GATE (2026-09-15): with GATE = True the two internal regions additionally
require deviation z > GATE_T (0.6) on GATE_INT_COL, and the two external
regions deviation z <= GATE_EXT_MAX (0, at or below the viewer's average
distance from the group) on GATE_EXT_COL (within-subject in both versions);
windows failing the gate are dropped, middle and ambiguous are unchanged.
Outputs carry '_gated' (internal gate within-subject) or '_gated_tp'
(internal gate within-timepoint). Note a deviation z below 0 is not a negative
distance: it is a distance below the viewer's own average distance.

Outputs: {MOVIE_DATA}/attn_explore_14Sep26/state_profile/contacts_by_region[_gated].csv,
         profile_by_region[_gated].csv, profile[_gated].json
Deliberately NOT done: no inference.
"""
import os, sys, json
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(v, '1')
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_attn_states_lmm as L

MOVIE_DATA = L.MOVIE_DATA
LABELS = f'{MOVIE_DATA}/attention_labels_10s/attention_labels_10s_pooled_explore14Sep26.csv'
OUT = f'{MOVIE_DATA}/attn_explore_14Sep26/state_profile'
EDGES = [-np.inf, -1.5, -0.4, 0.4, 0.8, 1.2, np.inf]
REG = ['far external', 'external', 'middle', 'ambiguous', 'internal', 'far internal']
ATLASES = ['Y17_Atlas_Region', 'Y7_Atlas_Region']
GATE = False
GATE_T = 0.6
GATE_EXT_MAX = 0.0
GATE_INT_COL = 'group_dev_mahal_z'   # deviation used for the INTERNAL gate: 'group_dev_mahal_z' (within-subject z) | 'mahal_time_z' (within-timepoint z)
GATE_EXT_COL = 'group_dev_mahal_z'   # deviation used for the EXTERNAL gate (Christine 2026-09-15: within-subject in both versions;
                                     # external and far external differ only in PC1)
SUF = ('_gated_tp' if GATE_INT_COL == 'mahal_time_z' else '_gated') if GATE else ''

if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    labels = pd.read_csv(LABELS); included = pd.read_csv(L.INCLUDED)
    rows = []
    for band in L.BANDS:
        for r in included.itertuples():
            atlas, z = L.read_tier2(r.video, band, r.patient, r.run)
            lab = labels[(labels.video == r.video) & (labels.pat == r.patient)].set_index('window_idx_10s')
            l = lab['PC1_z']
            n = min(len(z), int(l.index.max()) + 1)
            pc = l.reindex(range(n)).to_numpy(float)
            dev_int = lab[GATE_INT_COL].reindex(range(n)).to_numpy(float)
            dev_ext = lab[GATE_EXT_COL].reindex(range(n)).to_numpy(float)
            bad, _ = L.load_bad(r.video, r.patient, r.run, n)
            keep = ~bad & np.isfinite(pc)
            region = pd.cut(pc, EDGES, labels=REG).astype(object)
            if GATE:
                is_int = np.isin(region, ['internal', 'far internal']); is_ext = np.isin(region, ['external', 'far external'])
                region[is_int & ~(dev_int > GATE_T)] = None
                region[is_ext & ~(dev_ext <= GATE_EXT_MAX)] = None
            Z = z.iloc[:n]
            for c in Z.columns:
                y = Z[c].to_numpy(float); ok = keep & np.isfinite(y)
                if ok.sum() < 50:
                    continue
                base = y[ok].mean()
                for a in ATLASES:
                    L.ATLAS = a; reg = L.region_of(atlas, c)
                    if reg in L.EXCLUDE_REGIONS:
                        continue
                    for k in REG:
                        m = ok & (region == k)
                        if m.sum() >= 3:
                            rows.append(dict(atlas=a.split('_')[0], band=band, video=r.video, patient=r.patient, person=L.person_of(r.patient), contact=c, network=reg, region=k, n_win=int(m.sum()), amp=float(y[m].mean() - base)))
        print(band, 'done', flush=True)
    C = pd.DataFrame(rows)
    C.to_csv(f'{OUT}/contacts_by_region{SUF}.csv', index=False)
    S = C.groupby(['atlas', 'video', 'network', 'band', 'region'], observed=True).agg(mean=('amp', 'mean'), sem=('amp', 'sem'), n_contacts=('amp', 'size'), n_persons=('person', 'nunique')).reset_index()
    S.to_csv(f'{OUT}/profile_by_region{SUF}.csv', index=False)
    out = {}
    for a, sa in S.groupby('atlas'):
        out[a] = {'networks': sorted(sa.network.unique()), 'data': {}}
        for (vid, net, band), g in sa.groupby(['video', 'network', 'band']):
            g = g.set_index('region').reindex(REG)
            out[a]['data'][f'{vid}|{net}|{band}'] = {'mean': [None if np.isnan(x) else round(x, 4) for x in g['mean']], 'sem': [None if np.isnan(x) else round(x, 4) for x in g['sem']], 'n': [0 if np.isnan(x) else int(x) for x in g['n_contacts']], 'np': [0 if np.isnan(x) else int(x) for x in g['n_persons']]}
    json.dump({'regions': REG, 'bands': L.BANDS, 'videos': L.VIDEOS, 'atlases': out}, open(f'{OUT}/profile{SUF}.json', 'w'))
    print(S.groupby(['atlas', 'band', 'region'], observed=True)['mean'].mean().unstack().round(3).to_string())
    print(f'-> {OUT}')
