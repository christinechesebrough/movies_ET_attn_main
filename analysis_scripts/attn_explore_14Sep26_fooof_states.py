"""
Exploration track, step 8: FOOOF aperiodic and periodic features by gaze state.

Same states as entries 13-14 of the log, deviation-gated with the within-
timepoint internal gate (the third row there):
    six regions on within-viewer PC1 z: far external < -1.5 | external -1.5..-0.4 |
    middle -0.4..0.4 | ambiguous 0.4..0.8 | internal 0.8..1.2 | far internal > 1.2
    gate: internal regions require mahal_time_z > 0.6; external regions require
    within-subject mahal_z <= 0; failing windows dropped
    three states: External (< -0.4) / Middle / Internal (>= 0.8), same gate
    two states: the conj_0.6 label (Internal PC1_z > 0.6 & mahal_z > 0.6;
    External -1.5 <= PC1_z < -0.6) for a like-for-like with entry 12

Features (rolling_fooof_light, knee mode, 1-57 Hz): Aperiodic_Exponent,
Aperiodic_Offset, Knee_Freq_Hz, Periodic_{theta,alpha,beta,gamma}. Peak
presence / centre-frequency features deliberately left out for now.

Per contact: mean of each feature per region minus the contact's mean over all
kept windows (bad windows excluded), so values are contact-relative.
Tests (all MixedLM with person random intercept on per-contact quantities):
    two-state    Internal - External per contact (conj_0.6), per network x feature
    trend        weighted slope of the six region means on rank 1..6
    endpoint     far internal - far external
    three-state  the three pairwise differences
Correction: hypothesis networks (Y7 DAN / DMN / VN; Y17 DAN-A / Default A /
Visual Central / Visual Peripheral) - two-state, trend and three-state
Internal-External uncorrected; everything else FDR within (network, film)
over the tests run there. Networks: Y17 and Y7 from the Tier 2 atlas rows.

PRESENCE (2026-09-15): with PRESENCE = True the features are Has_theta and
Has_alpha, expressed as PERCENT of a state's windows with a fitted peak; values
are NOT contact-centred (a bar reads as the raw percentage), contrasts are
still per-contact differences. Y17 and Y7. Outputs go to fooof_states/presence/.

Outputs ({MOVIE_DATA}/attn_explore_14Sep26/fooof_states/):
    contacts_by_region.csv, profile_by_region.csv, profile.json
    trend6.csv/.json, contrasts3.csv/.json, twostate.csv/.json
"""
import os, sys, glob, json, warnings
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(v, '1')
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_attn_states_lmm as L
warnings.filterwarnings('ignore')

MOVIE_DATA = L.MOVIE_DATA
FOOOF_DIR = f'{MOVIE_DATA}/rolling_fooof_light'
LABELS = f'{MOVIE_DATA}/attention_labels_10s/attention_labels_10s_pooled_explore14Sep26.csv'
PRESENCE = False        # True: Has_theta / Has_alpha as % of windows, Y7 only, no centring -> fooof_states/presence/
OUT = f'{MOVIE_DATA}/attn_explore_14Sep26/fooof_states' + ('/presence' if PRESENCE else '')
FEATURES = ['Has_theta', 'Has_alpha'] if PRESENCE else ['Aperiodic_Exponent', 'Aperiodic_Offset', 'Knee_Freq_Hz', 'Periodic_theta', 'Periodic_alpha', 'Periodic_beta', 'Periodic_gamma']
ATLASES = ['Y17', 'Y7']       # presence run originally Y7 only; Y17 added 2026-09-15
EDGES = [-np.inf, -1.5, -0.4, 0.4, 0.8, 1.2, np.inf]
REG = ['far external', 'external', 'middle', 'ambiguous', 'internal', 'far internal']
RANK = {r: i + 1 for i, r in enumerate(REG)}
GROUP = {'far external': 'External', 'external': 'External', 'middle': 'Middle', 'ambiguous': 'Middle', 'internal': 'Internal', 'far internal': 'Internal'}
GATE_T, GATE_EXT_MAX = 0.6, 0.0
GATE_INT_COL, GATE_EXT_COL = 'mahal_time_z', 'group_dev_mahal_z'
TWO_STATE_LABEL = 'label_conj_0.6'
HYP_NETS = {'Y7': ['Dorsal Attention Network (DAN)', 'Default Mode Network (DMN)', 'Visual Network (VN)'],
            'Y17': ['Dorsal Attention A', 'Default A', 'Visual Central (Visual A)', 'Visual Peripheral (Visual B)']}
MIN_WIN, MIN_CONTACTS, MIN_PERSONS, MIN_REGIONS = 3, 5, 3, 4


def find_file(pat, run, vid):
    n = int(run.split('-')[1])
    hits = [f for f in glob.glob(f'{FOOOF_DIR}/{pat}/{pat}_*{vid}_fooof_light.csv') if not os.path.basename(f).startswith('._')
            and (f'_run-{n:02d}_' in os.path.basename(f) or f'_run-{n}_' in os.path.basename(f))]
    return hits[0] if hits else None


def networks_of_contacts(vid, pat, run):
    f = f'{L.T2_DIR}/windowed_robustz_power_log_{vid}_alpha/{pat}/{pat}_{run}_{vid}_alpha_power_log_robustz_rolling_{L.STAT}_10s.csv'
    if not os.path.exists(f):
        return {}
    a = pd.read_csv(f, nrows=4).set_index('Atlas').iloc[:, 2:]
    return {c: {'Y17': str(a.loc['Y17_Atlas_Region', c]), 'Y7': L.Y7_TO_LONG.get(str(a.loc['Y7_Atlas_Region', c]), str(a.loc['Y7_Atlas_Region', c]))} for c in a.columns}


def stage2(d, col):
    try:
        m = smf.mixedlm(f'{col} ~ 1', d, groups=d['person']).fit(reml=True)
        est, se = float(m.params['Intercept']), float(m.bse['Intercept'])
    except Exception:
        est, se = d[col].mean(), d[col].sem()
    z = est / se if se > 0 else np.nan
    return est, se, z, (2 * (1 - stats.norm.cdf(abs(z))) if np.isfinite(z) else np.nan)


def correct(R, unc_mask):
    R = R.copy(); R['hyp_uncorrected'] = unc_mask; R['p_adj'] = np.nan
    for key, idx in R.groupby(['atlas', 'video', 'network']).groups.items():
        sub = R.loc[idx]; corr = sub[~sub.hyp_uncorrected]
        if len(corr):
            R.loc[corr.index, 'p_adj'] = multipletests(corr.p.fillna(1), method='fdr_bh')[1]
        R.loc[sub[sub.hyp_uncorrected].index, 'p_adj'] = sub[sub.hyp_uncorrected].p
        R.loc[idx, 'n_tests_cell'] = len(corr)
    R['sig'] = R.p_adj < 0.05
    return R


def to_json(R, keyfields):
    out = {}
    for r in R.itertuples():
        out.setdefault(r.atlas, {})['|'.join(str(getattr(r, k)) for k in keyfields)] = dict(z=round(float(r.z), 2), p=float(f'{r.p:.2g}'), p_adj=float(f'{r.p_adj:.2g}'), sig=bool(r.sig), unc=bool(r.hyp_uncorrected), est=round(float(r.est), 5))
    return out


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    labels = pd.read_csv(LABELS); included = pd.read_csv(L.INCLUDED)
    rows, two = [], []
    for r in included.itertuples():
        f = find_file(r.patient, r.run, r.video)
        if f is None:
            print(f'  no FOOOF file: {r.video} {r.patient} {r.run}'); continue
        d = pd.read_csv(f, usecols=['Window_Index', 'Channel'] + FEATURES)
        if PRESENCE:
            for feat in FEATURES:
                d[feat] = d[feat].astype(str).str.lower().map({'true': 100.0, 'false': 0.0, '1': 100.0, '0': 0.0, '1.0': 100.0, '0.0': 0.0})
        lab = labels[(labels.video == r.video) & (labels.pat == r.patient)].set_index('window_idx_10s')
        n = min(int(d.Window_Index.max()) + 1, int(lab.index.max()) + 1)
        pc = lab['PC1_z'].reindex(range(n)).to_numpy(float)
        dev_int = lab[GATE_INT_COL].reindex(range(n)).to_numpy(float); dev_ext = lab[GATE_EXT_COL].reindex(range(n)).to_numpy(float)
        two_lab = lab[TWO_STATE_LABEL].reindex(range(n)).to_numpy(object)
        bad, _ = L.load_bad(r.video, r.patient, r.run, n)
        keep = ~bad & np.isfinite(pc)
        region = pd.cut(pc, EDGES, labels=REG).astype(object)
        is_int = np.isin(region, ['internal', 'far internal']); is_ext = np.isin(region, ['external', 'far external'])
        region[is_int & ~(dev_int > GATE_T)] = None; region[is_ext & ~(dev_ext <= GATE_EXT_MAX)] = None
        nets = networks_of_contacts(r.video, r.patient, r.run)
        for ch, g in d.groupby('Channel'):
            g = g.set_index('Window_Index').reindex(range(n))
            net = nets.get(ch, {'Y17': 'unknown', 'Y7': 'unknown'})
            for feat in FEATURES:
                y = g[feat].to_numpy(float); ok = keep & np.isfinite(y)
                if ok.sum() < 50:
                    continue
                base = 0.0 if PRESENCE else y[ok].mean()
                for k in REG:
                    m = ok & (region == k)
                    if m.sum() >= MIN_WIN:
                        rows.append(dict(band=feat, video=r.video, patient=r.patient, person=L.person_of(r.patient), contact=f'{r.patient}_{r.run}_{ch}', Y17=net['Y17'], Y7=net['Y7'], region=k, n_win=int(m.sum()), amp=float(y[m].mean() - base)))
                mi = ok & (two_lab == 'Internal'); me = ok & (two_lab == 'External')
                if mi.sum() >= MIN_WIN and me.sum() >= MIN_WIN:
                    two.append(dict(band=feat, video=r.video, patient=r.patient, person=L.person_of(r.patient), contact=f'{r.patient}_{r.run}_{ch}', Y17=net['Y17'], Y7=net['Y7'], n_int=int(mi.sum()), n_ext=int(me.sum()), delta=float(y[mi].mean() - y[me].mean())))
        print(f'  {r.video:24s} {r.patient:22s} {d.Channel.nunique():4d} contacts', flush=True)
    C = pd.DataFrame(rows); C.to_csv(f'{OUT}/contacts_by_region.csv', index=False)
    TW = pd.DataFrame(two); TW.to_csv(f'{OUT}/contacts_twostate.csv', index=False)
    Cl = pd.concat([C.assign(atlas=a, network=C[a]) for a in ATLASES], ignore_index=True)
    Cl = Cl[~Cl.network.isin(L.EXCLUDE_REGIONS)]
    TWl = pd.concat([TW.assign(atlas=a, network=TW[a]) for a in ATLASES], ignore_index=True)
    TWl = TWl[~TWl.network.isin(L.EXCLUDE_REGIONS)]
    # profile
    S = Cl.groupby(['atlas', 'video', 'network', 'band', 'region'], observed=True).agg(mean=('amp', 'mean'), sem=('amp', 'sem'), n_contacts=('amp', 'size'), n_persons=('person', 'nunique')).reset_index()
    S.to_csv(f'{OUT}/profile_by_region.csv', index=False)
    prof = {'regions': REG, 'bands': FEATURES, 'videos': L.VIDEOS, 'atlases': {}}
    for a, sa in S.groupby('atlas'):
        prof['atlases'][a] = {'networks': sorted(sa.network.unique()), 'data': {}}
        for (vid, net, band), g in sa.groupby(['video', 'network', 'band']):
            g = g.set_index('region').reindex(REG)
            prof['atlases'][a]['data'][f'{vid}|{net}|{band}'] = {'mean': [None if np.isnan(x) else round(float(x), 5) for x in g['mean']], 'sem': [None if np.isnan(x) else round(float(x), 5) for x in g['sem']], 'n': [0 if np.isnan(x) else int(x) for x in g.n_contacts], 'np': [0 if np.isnan(x) else int(x) for x in g.n_persons]}
    json.dump(prof, open(f'{OUT}/profile.json', 'w'))
    # two-state
    rows2 = []
    for (atlas, vid, net, band), g in TWl.groupby(['atlas', 'video', 'network', 'band']):
        if len(g) < MIN_CONTACTS or g.person.nunique() < MIN_PERSONS:
            continue
        est, se, z, p = stage2(g, 'delta'); rows2.append(dict(atlas=atlas, video=vid, network=net, band=band, contrast='Internal-External', est=est, se=se, z=z, p=p, n_contacts=len(g), n_persons=g.person.nunique()))
    R2 = correct(pd.DataFrame(rows2), pd.DataFrame(rows2).apply(lambda r: r.network in HYP_NETS[r.atlas], axis=1))
    R2.to_csv(f'{OUT}/twostate.csv', index=False); json.dump(to_json(R2, ['video', 'network', 'band']), open(f'{OUT}/twostate.json', 'w'))
    # trend + endpoint
    rows3 = []
    for (atlas, vid, net, band), g in Cl.groupby(['atlas', 'video', 'network', 'band']):
        per = []
        for (contact, person), gc in g.groupby(['contact', 'person']):
            if len(gc) < MIN_REGIONS:
                continue
            x = gc.region.map(RANK).to_numpy(float); y = gc.amp.to_numpy(float); w = gc.n_win.to_numpy(float)
            xm = np.average(x, weights=w); slope = np.sum(w * (x - xm) * y) / np.sum(w * (x - xm) ** 2)
            e = gc.set_index('region').amp; per.append(dict(person=person, slope=slope, endpoint=e.get('far internal', np.nan) - e.get('far external', np.nan)))
        P = pd.DataFrame(per)
        if len(P) < MIN_CONTACTS or P.person.nunique() < MIN_PERSONS:
            continue
        for cname, col in [('trend', 'slope'), ('endpoint', 'endpoint')]:
            dd = P.dropna(subset=[col])
            if len(dd) < MIN_CONTACTS or dd.person.nunique() < MIN_PERSONS:
                continue
            est, se, z, p = stage2(dd, col); rows3.append(dict(atlas=atlas, video=vid, network=net, band=band, contrast=cname, est=est, se=se, z=z, p=p, n_contacts=len(dd), n_persons=dd.person.nunique()))
    R3 = pd.DataFrame(rows3); R3 = correct(R3, R3.apply(lambda r: (r.network in HYP_NETS[r.atlas]) and r.contrast == 'trend', axis=1))
    R3.to_csv(f'{OUT}/trend6.csv', index=False); json.dump(to_json(R3, ['video', 'network', 'band', 'contrast']), open(f'{OUT}/trend6.json', 'w'))
    # three-state
    Cl['group'] = Cl.region.map(GROUP); Cl['w'] = Cl.amp * Cl.n_win
    G = Cl.groupby(['atlas', 'band', 'video', 'patient', 'person', 'contact', 'network', 'group'], observed=True).agg(w=('w', 'sum'), n=('n_win', 'sum')).reset_index(); G['amp'] = G.w / G.n
    W = G.pivot_table(index=['atlas', 'band', 'video', 'patient', 'person', 'contact', 'network'], columns='group', values='amp').reset_index()
    means3 = G.groupby(['atlas', 'video', 'network', 'band', 'group'], observed=True).agg(mean=('amp', 'mean'), sem=('amp', 'sem'), n_contacts=('amp', 'size'), n_persons=('person', 'nunique')).reset_index()
    rows4 = []
    for (atlas, vid, net, band), g in W.groupby(['atlas', 'video', 'network', 'band']):
        for a, b in [('Internal', 'External'), ('Internal', 'Middle'), ('Middle', 'External')]:
            if a not in g or b not in g:
                continue
            dd = g[[a, b, 'person']].dropna()
            if len(dd) < MIN_CONTACTS or dd.person.nunique() < MIN_PERSONS:
                continue
            dd = dd.assign(delta=dd[a] - dd[b]); est, se, z, p = stage2(dd, 'delta')
            rows4.append(dict(atlas=atlas, video=vid, network=net, band=band, contrast=f'{a}-{b}', est=est, se=se, z=z, p=p, n_contacts=len(dd), n_persons=dd.person.nunique()))
    R4 = pd.DataFrame(rows4); R4 = correct(R4, R4.apply(lambda r: (r.network in HYP_NETS[r.atlas]) and r.contrast == 'Internal-External', axis=1))
    R4.to_csv(f'{OUT}/contrasts3.csv', index=False)
    c3 = {'groups': ['External', 'Middle', 'Internal'], 'bands': FEATURES, 'videos': L.VIDEOS, 'hyp_nets': HYP_NETS, 'atlases': {}}
    for atlas in ATLASES:
        nets = sorted(means3[means3.atlas == atlas].network.unique()); data = {}
        for (vid, net, band), g in means3[means3.atlas == atlas].groupby(['video', 'network', 'band']):
            g = g.set_index('group').reindex(c3['groups']); rr = R4[(R4.atlas == atlas) & (R4.video == vid) & (R4.network == net) & (R4.band == band)].set_index('contrast')
            data[f'{vid}|{net}|{band}'] = {'mean': [None if np.isnan(x) else round(x, 5) for x in g['mean']], 'sem': [None if np.isnan(x) else round(x, 5) for x in g['sem']], 'n': [0 if np.isnan(x) else int(x) for x in g.n_contacts], 'np': [0 if np.isnan(x) else int(x) for x in g.n_persons],
                                         'tests': {c: {'z': round(float(rr.z[c]), 2), 'p': float(f'{rr.p[c]:.2g}'), 'p_adj': float(f'{rr.p_adj[c]:.2g}'), 'sig': bool(rr.sig[c]), 'unc': bool(rr.hyp_uncorrected[c])} for c in rr.index}}
        c3['atlases'][atlas] = {'networks': nets, 'data': data}
    json.dump(c3, open(f'{OUT}/contrasts3.json', 'w'))
    pd.set_option('display.width', 250)
    print('\nrecordings per video:', C.groupby('video').patient.nunique().to_dict())
    print('\nTWO-STATE (conj_0.6) significant Internal-External by feature, hypothesis networks uncorrected / other FDR:'); print(R2[R2.sig].groupby(['atlas', 'band', 'hyp_uncorrected']).size().unstack(fill_value=0).to_string())
    print('\nTREND significant, sign by feature:'); print(R3[R3.sig & (R3.contrast == 'trend')].groupby(['atlas', 'band']).z.agg(pos=lambda x: (x > 0).sum(), neg=lambda x: (x < 0).sum()).to_string())
    print('\nwhole-picture: network-averaged region means per feature (Y17):')
    print(S[S.atlas == ATLASES[0]].groupby(['band', 'region'], observed=True)['mean'].mean().unstack()[REG].round(4).to_string())
    print(f'\n-> {OUT}')
