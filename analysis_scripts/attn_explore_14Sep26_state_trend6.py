"""
Exploration track, step 7: planned contrasts over the six ordered PC1 regions.

Instead of pairwise tests between the six categories (15 pairs x 6 bands per
network and film), two planned contrasts per (network, band):
    trend     per contact, weighted least-squares slope of the six region means
              on region rank 1..6 (weights = windows per region); one number per
              contact; then MixedLM slope ~ 1 + (1|person). Positive = amplitude
              rises from external to internal across the axis.
    endpoint  per contact, far internal minus far external; same second stage.
Correction (as in step 6): hypothesis networks (Y7 DAN, DMN, VN; Y17 DAN-A,
Default A, Vis Central, Vis Peripheral) trend UNCORRECTED; every other test
FDR within (network, film) over the 12 tests run there (6 bands x 2 contrasts).

VERSION knob: '' (PC1-only regions) | '_gated' (within-subject deviation gate)
| '_gated_tp' (timepoint-relative internal gate); reads
contacts_by_region{VERSION}.csv, writes trend6{VERSION}.csv / .json.
"""
import os, json, warnings
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(v, '1')
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings('ignore')

D = '/media/christine/Samsung/Movie_data/attn_explore_14Sep26/state_profile'
VERSION = ''
REG = ['far external', 'external', 'middle', 'ambiguous', 'internal', 'far internal']
RANK = {r: i + 1 for i, r in enumerate(REG)}
HYP_NETS = {'Y7': ['Dorsal Attention Network (DAN)', 'Default Mode Network (DMN)', 'Visual Network (VN)'],
            'Y17': ['Dorsal Attention A', 'Default A', 'Visual Central (Visual A)', 'Visual Peripheral (Visual B)']}
MIN_CONTACTS, MIN_PERSONS, MIN_REGIONS = 5, 3, 4


def stage2(d, col):
    try:
        m = smf.mixedlm(f'{col} ~ 1', d, groups=d['person']).fit(reml=True)
        est, se = float(m.params['Intercept']), float(m.bse['Intercept'])
    except Exception:
        est, se = d[col].mean(), d[col].sem()
    z = est / se if se > 0 else np.nan
    return est, se, z, (2 * (1 - stats.norm.cdf(abs(z))) if np.isfinite(z) else np.nan)


if __name__ == '__main__':
    C = pd.read_csv(f'{D}/contacts_by_region{VERSION}.csv')
    rows = []
    for (atlas, vid, net, band), g in C.groupby(['atlas', 'video', 'network', 'band']):
        per = []
        for (pat, person, contact), gc in g.groupby(['patient', 'person', 'contact']):
            if len(gc) < MIN_REGIONS:
                continue
            x = gc.region.map(RANK).to_numpy(float); y = gc.amp.to_numpy(float); w = gc.n_win.to_numpy(float)
            xm = np.average(x, weights=w); slope = np.sum(w * (x - xm) * y) / np.sum(w * (x - xm) ** 2)
            ends = gc.set_index('region').amp
            per.append(dict(person=person, slope=slope, endpoint=(ends.get('far internal', np.nan) - ends.get('far external', np.nan))))
        P = pd.DataFrame(per)
        if len(P) < MIN_CONTACTS or P.person.nunique() < MIN_PERSONS:
            continue
        for cname, col in [('trend', 'slope'), ('endpoint', 'endpoint')]:
            d = P.dropna(subset=[col])
            if len(d) < MIN_CONTACTS or d.person.nunique() < MIN_PERSONS:
                continue
            est, se, z, p = stage2(d, col)
            rows.append(dict(atlas=atlas, video=vid, network=net, band=band, contrast=cname, est=est, se=se, z=z, p=p, n_contacts=len(d), n_persons=d.person.nunique()))
    R = pd.DataFrame(rows)
    R['hyp_uncorrected'] = R.apply(lambda r: (r.network in HYP_NETS[r.atlas]) and r.contrast == 'trend', axis=1)
    R['p_adj'] = np.nan
    for (atlas, vid, net), idx in R.groupby(['atlas', 'video', 'network']).groups.items():
        sub = R.loc[idx]; corr = sub[~sub.hyp_uncorrected]
        if len(corr):
            R.loc[corr.index, 'p_adj'] = multipletests(corr.p.fillna(1), method='fdr_bh')[1]
        R.loc[sub[sub.hyp_uncorrected].index, 'p_adj'] = sub[sub.hyp_uncorrected].p
        R.loc[idx, 'n_tests_cell'] = len(corr)
    R['sig'] = R.p_adj < 0.05
    R.to_csv(f'{D}/trend6{VERSION}.csv', index=False)
    out = {}
    for r in R.itertuples():
        out.setdefault(r.atlas, {})[f'{r.video}|{r.network}|{r.band}|{r.contrast}'] = dict(z=round(float(r.z), 2), p=float(f'{r.p:.2g}'), p_adj=float(f'{r.p_adj:.2g}'), sig=bool(r.sig), unc=bool(r.hyp_uncorrected), est=round(float(r.est), 4))
    json.dump(out, open(f'{D}/trend6{VERSION}.json', 'w'))
    print(f'VERSION "{VERSION}": tests {len(R)}; significant:'); print(R[R.sig].groupby(['atlas', 'contrast', 'hyp_uncorrected']).size().to_string())
    print('\nsign of significant trends:'); print(R[R.sig & (R.contrast == 'trend')].groupby(['atlas', 'band']).z.agg(pos=lambda x: (x > 0).sum(), neg=lambda x: (x < 0).sum()).to_string())
    print(f'\n-> {D}/trend6{VERSION}.csv')
