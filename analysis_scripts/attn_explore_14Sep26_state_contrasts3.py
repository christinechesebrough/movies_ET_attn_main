"""
Exploration track, step 6: three collapsed states and pairwise contrasts.

Collapses the six PC1 regions of attn_explore_14Sep26_state_profile.py into
    External = far external + external      (PC1 z < -0.4)
    Middle   = middle + ambiguous            (-0.4 <= PC1 z < 0.8)
    Internal = internal + far internal      (PC1 z >= 0.6 since 2026-09-15; was 0.8)
per contact (window-count-weighted), then for every (atlas, video, network,
band) fits the three pairwise contrasts Internal-External, Internal-Middle,
Middle-External with the pipeline's two-stage model: per-contact difference,
then MixedLM with a person random intercept (z, p).

Multiple comparisons (Christine, 2026-09-15):
    hypothesis networks - Y7: DAN, DMN, VN; Y17: Dorsal Attention A, Default A,
    Visual Central, Visual Peripheral - Internal-External is reported
    UNCORRECTED across bands (7 networks x 6 bands x 3 videos = 126 planned tests);
    every other test (other networks, and the Int-Mid / Mid-Ext contrasts in
    the hypothesis networks) is FDR-corrected (Benjamini-Hochberg) within its
    (network, video) cell, i.e. over the tests run in that network for that film.

Inputs:  {MOVIE_DATA}/attn_explore_14Sep26/state_profile/contacts_by_region[_gated].csv  (GATED knob)
Outputs: {MOVIE_DATA}/attn_explore_14Sep26/state_profile/contrasts3.csv, means3.csv, contrasts3.json
"""
import os, json, warnings
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(v, '1')
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings('ignore')

MOVIE_DATA = '/media/christine/Samsung/Movie_data'
D = f'{MOVIE_DATA}/attn_explore_14Sep26/state_profile'
GROUP = {'far external': 'External', 'external': 'External', 'middle': 'Middle', 'ambiguous': 'Middle', 'internal': 'Internal', 'far internal': 'Internal'}
PAIRS = [('Internal', 'External'), ('Internal', 'Middle'), ('Middle', 'External')]
HYP_NETS = {'Y7': ['Dorsal Attention Network (DAN)', 'Default Mode Network (DMN)', 'Visual Network (VN)'],
            'Y17': ['Dorsal Attention A', 'Default A', 'Visual Central (Visual A)', 'Visual Peripheral (Visual B)']}
MIN_CONTACTS, MIN_PERSONS = 5, 3
GATED = False          # True: use the deviation-gated regions, '_gated' outputs
GATE_TP = False        # with GATED: use the within-timepoint gated regions ('_gated_tp')
SUF = ('_gated_tp' if GATE_TP else '_gated') if GATED else ''

if __name__ == '__main__':
    C = pd.read_csv(f'{D}/contacts_by_region{SUF}.csv')
    C['group'] = C.region.map(GROUP)
    C['w'] = C.amp * C.n_win
    G = C.groupby(['atlas', 'band', 'video', 'patient', 'person', 'contact', 'network', 'group'], observed=True).agg(w=('w', 'sum'), n=('n_win', 'sum')).reset_index()
    G['amp'] = G.w / G.n
    W = G.pivot_table(index=['atlas', 'band', 'video', 'patient', 'person', 'contact', 'network'], columns='group', values='amp').reset_index()
    means = G.groupby(['atlas', 'video', 'network', 'band', 'group'], observed=True).agg(mean=('amp', 'mean'), sem=('amp', 'sem'), n_contacts=('amp', 'size'), n_persons=('person', 'nunique')).reset_index()
    means.to_csv(f'{D}/means3{SUF}.csv', index=False)
    rows = []
    for (atlas, vid, net, band), g in W.groupby(['atlas', 'video', 'network', 'band']):
        for a, b in PAIRS:
            d = g[[a, b, 'person']].dropna()
            if len(d) < MIN_CONTACTS or d.person.nunique() < MIN_PERSONS:
                continue
            d = d.assign(delta=d[a] - d[b])
            try:
                m = smf.mixedlm('delta ~ 1', d, groups=d['person']).fit(reml=True)
                est, se = float(m.params['Intercept']), float(m.bse['Intercept'])
            except Exception:
                est, se = d.delta.mean(), d.delta.sem()
            z = est / se if se > 0 else np.nan; p = 2 * (1 - __import__('scipy').stats.norm.cdf(abs(z))) if np.isfinite(z) else np.nan
            rows.append(dict(atlas=atlas, video=vid, network=net, band=band, contrast=f'{a}-{b}', est=est, se=se, z=z, p=p, n_contacts=len(d), n_persons=d.person.nunique()))
    R = pd.DataFrame(rows)
    R['hyp_uncorrected'] = R.apply(lambda r: (r.network in HYP_NETS[r.atlas]) and r.contrast == 'Internal-External', axis=1)
    R['p_adj'] = np.nan
    for (atlas, vid, net), idx in R.groupby(['atlas', 'video', 'network']).groups.items():
        sub = R.loc[idx]; corr = sub[~sub.hyp_uncorrected]
        if len(corr):
            R.loc[corr.index, 'p_adj'] = multipletests(corr.p.fillna(1), method='fdr_bh')[1]
        R.loc[sub[sub.hyp_uncorrected].index, 'p_adj'] = sub[sub.hyp_uncorrected].p
        R.loc[idx, 'n_tests_cell'] = len(corr)
    R['sig'] = R.p_adj < 0.05
    R.to_csv(f'{D}/contrasts3{SUF}.csv', index=False)
    out = {'groups': ['External', 'Middle', 'Internal'], 'bands': ['delta', 'theta', 'alpha', 'beta', 'gamma', 'HFA'], 'videos': sorted(W.video.unique()), 'hyp_nets': HYP_NETS, 'atlases': {}}
    for atlas in ['Y17', 'Y7']:
        nets = sorted(means[means.atlas == atlas].network.unique()); data = {}
        for (vid, net, band), g in means[means.atlas == atlas].groupby(['video', 'network', 'band']):
            g = g.set_index('group').reindex(out['groups'])
            rr = R[(R.atlas == atlas) & (R.video == vid) & (R.network == net) & (R.band == band)].set_index('contrast')
            data[f'{vid}|{net}|{band}'] = {'mean': [None if np.isnan(x) else round(x, 4) for x in g['mean']], 'sem': [None if np.isnan(x) else round(x, 4) for x in g['sem']], 'n': [0 if np.isnan(x) else int(x) for x in g.n_contacts], 'np': [0 if np.isnan(x) else int(x) for x in g.n_persons],
                                         'tests': {c: {'z': round(float(rr.z[c]), 2), 'p': float(f'{rr.p[c]:.2g}'), 'p_adj': float(f'{rr.p_adj[c]:.2g}'), 'sig': bool(rr.sig[c]), 'unc': bool(rr.hyp_uncorrected[c])} for c in rr.index}}
        out['atlases'][atlas] = {'networks': nets, 'data': data}
    json.dump(out, open(f'{D}/contrasts3{SUF}.json', 'w'))
    S = R[R.sig].groupby(['atlas', 'contrast', 'hyp_uncorrected']).size()
    print(S.to_string()); print()
    print('Internal-External significant cells (hypothesis networks, uncorrected):')
    print(R[R.hyp_uncorrected & R.sig][['atlas', 'video', 'network', 'band', 'z', 'p']].sort_values(['atlas', 'network', 'video', 'band']).round(3).to_string(index=False))
    print(f'\n-> {D}/contrasts3{SUF}.csv')
