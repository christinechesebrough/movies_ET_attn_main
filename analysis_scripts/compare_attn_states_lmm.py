#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Internal vs External attention-state contrasts in band amplitude, with a
hierarchical model that replaces both the pooled t-test
(compare_attn_states_heatmap_fromPCs_MATRIX_21Apr25.py,
compare_attn_states_bargraph_fromPCs.py) and the within-patient t-test
(compare_attn_states_within_pats.py), and adds the three-movie interaction
those scripts cannot test. Outputs keep their figure conventions.

Inputs:
    {MOVIE_DATA}/windowed_power_10s_rescale/windowed_robustz_power_log_{vid}_{band}/{pat}/
        {pat}_{run}_{vid}_{band}_power_log_robustz_rolling_{STAT}_10s.csv
        4 atlas rows (DK, Y7, Y17, AparcAseg) + 236-239 windows; robust z per
        contact of log10 Hilbert amplitude, one value per 10 s window (STAT =
        'mean' is the reported default; 'trim20' and 'median' are sensitivities)
    {MOVIE_DATA}/attention_labels_10s/attention_labels_10s_{LABEL_SOURCE}.csv
        make_attention_labels.py; 'pooled' (shared PCA) or 'per_video'
    {MOVIE_DATA}/bad_windows_10s/{vid}/{pat}/{pat}_{run}_{vid}_10s_window_artifact_mask.csv
    {MOVIE_DATA}/descriptives_power_10s/included_recordings.csv   (item 4; 42 recordings)

Design (per band, per atlas region):
    windows   every STRIDE-th 10 s window (STRIDE = 4 -> non-overlapping),
              window labelled Internal or External at the chosen scheme and z,
              frac_bad == 0 on the manual QC grid
    stage 1   per contact c:  delta_c = mean(z | Internal) - mean(z | External)
              over those windows, requiring >= MIN_WIN of each state.
              Two versions:
                raw       on the stored z
                tl        after subtracting, at every window index, the mean
                          across ALL included contacts of that video and band
                          - the stimulus-locked (time-locked) component. This
                          is the model-free equivalent of a categorical time
                          covariate: whatever is shared across patients at a
                          timepoint cannot enter the contrast.
    stage 2   MixedLM  delta_c ~ 0 + C(video), random intercept per person
              (contacts nested in person; contact count per person is handled
              by the nesting, not by weighting). Per-video estimate, SE, z, p
              -> the heatmap cell. Wald test of equal video effects -> the
              state x movie interaction cell.
              WEIGHTS = 'nwin' (2026-09-14): each delta_c enters with
              precision weight w = n_int*n_ext/(n_int+n_ext), the inverse of
              Var(delta_c) up to a common sigma^2 (contacts are already on a
              robust-z scale). Implemented by the sqrt(w) transform of endog,
              exog and the random-intercept design, so the fitted model is the
              weighted mixed model, not a frequency-weighted one. Needed for
              the centred_global labels, where labelled windows per state
              range 0-58 across recordings; under legacy labels (~25 per
              state everywhere) it is nearly the identity.
    also      naive pooled t (all contact-windows, ttest_ind) and the
              within-patient t (per-patient mean delta, ttest_1samp), for
              comparison with the old outputs.
    FDR       Benjamini-Hochberg across the region x band matrix, per video.

Outputs ({OUT_DIR}/{atlas}_{label_source}_{scheme}_z{z}{tag}/):
    tag appends _{STAT} if not trim20, _stride{STRIDE} if not 4, _minwin{MIN_WIN}
    (so the reported default, mean + stride 1, is '..._mean_stride1'; trim20 and
    stride 4 keep the un-suffixed names from the earlier robustness runs),
    if not 3, _w{WEIGHTS} if weighted. VARIANT = 'centred_global' switches
    LABELS to the *_centred_global label file and OUT_DIR to
    attn_state_contrasts_10s_centred_global/, so variant runs never share a
    directory with legacy runs.
    cells_{delta}.csv            every band x region x video: est, se, z, p, p_fdr,
                                 n_contacts, n_persons, naive_t, naive_p, within_t,
                                 within_p, n_patients_within
    interaction_{delta}.csv      band x region: chi2, df, p, p_fdr
    contacts_{delta}.csv         stage-1 table (one row per contact)
    heatmap_{vid}_{delta}.png    region x band, model z, FDR stars   (MATRIX style)
    heatmap_interaction_{delta}.png
    boxplots_{vid}_{delta}.png   all bands stacked: per-person mean Int / Ext per
                                 region with paired lines           (within_pats style)

Deliberately NOT done: no Neutral windows, no contact-level covariates, no
region-level custom 'network' grouping (needs define_custom_network_atlas),
no reading of the old windowed_power_10s outputs.
"""

import os
import sys
import re
import warnings
import itertools
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import matplotlib
import matplotlib.pyplot as plt

machine_path = 'media/christine'
drive = 'Samsung'
MOVIE_DATA = f'/{machine_path}/{drive}/Movie_data'
T2_DIR = f'{MOVIE_DATA}/windowed_power_10s_rescale'
LABEL_SOURCE = 'pooled'           # 'pooled' | 'per_video'   (make_attention_labels.py PC_SOURCE)
VARIANT = 'legacy'                # 'legacy' | 'robust' | 'centred_global'   (make_attention_labels.py VARIANT)
_vtag = '' if VARIANT == 'legacy' else f'_{VARIANT}'
LABELS = f'{MOVIE_DATA}/attention_labels_10s/attention_labels_10s_{LABEL_SOURCE}{_vtag}.csv'
BAD_DIR = f'{MOVIE_DATA}/bad_windows_10s'
INCLUDED = f'{MOVIE_DATA}/descriptives_power_10s/included_recordings.csv'
OUT_DIR = f'{MOVIE_DATA}/attn_state_contrasts_10s{_vtag}'

VIDEOS = ['despicable_me_english', 'despicable_me_hungarian', 'inscapes']
VID_SHORT = {'despicable_me_english': 'DM English', 'despicable_me_hungarian': 'DM Hungarian',
             'inscapes': 'Inscapes'}
BANDS = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'HFA']
STAT = 'mean'                     # 'mean' | 'trim20' | 'median'  window statistic; mean is the reported default

ATLAS = 'Y7_Atlas_Region'        # 'Y17_Atlas_Region' | 'Y7_Atlas_Region' | 'DK_Atlas_Region'
SCHEME = 'within_timepoint_dev'     # 'within_subject_dev' | 'within_timepoint_dev'
THRESHOLDS = [0.6]      # first = reported; others = sensitivity (CSV + heatmaps only)
DELTAS = ['raw', 'tl']
STRIDE = 1
MIN_WIN = 3                       # labelled windows per state per contact, after stride
                                  # (8 recommended for centred_global: ~2 independent windows at stride 1)
WEIGHTS = None                    # None | 'nwin'  stage-2 precision weights, see docstring
MIN_PERSONS = 3                   # persons per region cell to fit
EXCLUDE_REGIONS = {'FreeSurfer_Defined_Medial_Wall', 'Out', 'unknown', 'nan', ''}
BOXPLOTS_FOR = [0.6]              # thresholds that also get the per-band boxplot figures

Y7_TO_LONG = {  # the three Y7 vocabularies in the atlas rows -> one
    '7Networks_1': 'Visual Network (VN)', '7Networks_2': 'Somatomotor Network (SMN)',
    '7Networks_3': 'Dorsal Attention Network (DAN)', '7Networks_4': 'Ventral Attention Network (VAN)',
    '7Networks_5': 'Limbic Network (LN)', '7Networks_6': 'Frontoparietal Network (FPN)',
    '7Networks_7': 'Default Mode Network (DMN)',
    'Visual': 'Visual Network (VN)', 'Somatomotor': 'Somatomotor Network (SMN)',
    'Dorsal Attention': 'Dorsal Attention Network (DAN)', 'Ventral Attention': 'Ventral Attention Network (VAN)',
    'Limbic': 'Limbic Network (LN)', 'Frontoparietal': 'Frontoparietal Network (FPN)',
    'Default': 'Default Mode Network (DMN)'}
COLOR_INT, COLOR_EXT = '#2a78d6', '#8a8985'


def person_of(pat):
    return re.sub(r'_0\d$', '', pat)


def read_tier2(vid, band, pat, run):
    f = f'{T2_DIR}/windowed_robustz_power_log_{vid}_{band}/{pat}/{pat}_{run}_{vid}_{band}_power_log_robustz_rolling_{STAT}_10s.csv'
    d = pd.read_csv(f)
    atlas = d.iloc[:4].set_index('Atlas').iloc[:, 2:]
    z = d.iloc[4:].iloc[:, 3:].astype(float).reset_index(drop=True)   # windows x contacts
    return atlas, z


def region_of(atlas_rows, contact):
    v = str(atlas_rows.loc[ATLAS, contact])
    if ATLAS == 'Y7_Atlas_Region':
        v = Y7_TO_LONG.get(v, v)
    return v


def load_bad(vid, pat, run, n_win):
    f = f'{BAD_DIR}/{vid}/{pat}/{pat}_{run}_{vid}_10s_window_artifact_mask.csv'
    bad = np.zeros(n_win, bool)
    if os.path.exists(f):
        m = pd.read_csv(f)
        idx = m.loc[m.frac_bad > 0, 'window_idx_10s'].to_numpy()
        bad[idx[idx < n_win]] = True
        return bad, True
    return bad, False


# -----------------------------------------------------------------------------
def stage1(labels, included, scheme, z_thr):
    """Per contact deltas for every band. Returns (contacts_df, pooled_df)."""
    lab_col = f'label_{scheme}_{z_thr}'
    contact_rows, pooled_rows = [], []
    for band in BANDS:
        series = {}   # (vid, pat, run) -> (atlas, z, labels array, bad array)
        for r in included.itertuples():
            atlas, z = read_tier2(r.video, band, r.patient, r.run)
            l = labels[(labels.video == r.video) & (labels.pat == r.patient)].set_index('window_idx_10s')[lab_col]
            n = min(len(z), int(l.index.max()) + 1)
            lab = l.reindex(range(n)).to_numpy(object)
            bad, has_qc = load_bad(r.video, r.patient, r.run, n)
            series[(r.video, r.patient, r.run)] = (atlas, z.iloc[:n], lab, bad, has_qc)
        # stimulus-locked component per video: mean over all included contacts at each window
        tl_mean = {}
        for vid in VIDEOS:
            mats = [v[1] for k, v in series.items() if k[0] == vid]
            n = min(len(m) for m in mats)
            tl_mean[vid] = np.nanmean(np.concatenate([m.iloc[:n].to_numpy() for m in mats], axis=1), axis=1)
        for (vid, pat, run), (atlas, z, lab, bad, has_qc) in series.items():
            n = len(z)
            keep = np.zeros(n, bool); keep[::STRIDE] = True
            keep &= ~bad
            is_int = keep & (lab == 'Internal'); is_ext = keep & (lab == 'External')
            tl = tl_mean[vid][:n]
            for c in z.columns:
                reg = region_of(atlas, c)
                if reg in EXCLUDE_REGIONS:
                    continue
                y = z[c].to_numpy(float)
                for dname, yy in (('raw', y), ('tl', y - tl)):
                    yi, ye = yy[is_int], yy[is_ext]
                    yi, ye = yi[np.isfinite(yi)], ye[np.isfinite(ye)]
                    if len(yi) < MIN_WIN or len(ye) < MIN_WIN:
                        continue
                    contact_rows.append(dict(band=band, video=vid, patient=pat, run=run,
                                             person=person_of(pat), contact=f'{pat}_{run}_{c}',
                                             region=reg, delta_type=dname, has_qc=has_qc,
                                             n_int=len(yi), n_ext=len(ye),
                                             sd_int=yi.std(ddof=1), sd_ext=ye.std(ddof=1),
                                             mean_int=yi.mean(), mean_ext=ye.mean(), delta=yi.mean() - ye.mean()))
                    if dname == 'raw':
                        for v_ in yi: pooled_rows.append((band, vid, reg, 'Internal', v_))
                        for v_ in ye: pooled_rows.append((band, vid, reg, 'External', v_))
        print(f'  stage 1 {band}: {sum(1 for r in contact_rows if r["band"] == band and r["delta_type"] == "raw")} contacts')
    return (pd.DataFrame(contact_rows),
            pd.DataFrame(pooled_rows, columns=['band', 'video', 'region', 'state', 'y']))


def fit_stage2(s):
    """delta ~ 0 + C(video) + (1 | person), optionally precision-weighted."""
    if WEIGHTS is None:
        return smf.mixedlm('delta ~ 0 + C(video)', s, groups=s['person']).fit(reml=True)
    if WEIGHTS != 'nwin':
        raise ValueError(WEIGHTS)
    w = (s.n_int * s.n_ext / (s.n_int + s.n_ext)).to_numpy(float)
    sw = np.sqrt(w / w.mean())                       # mean weight 1 keeps 'scale' interpretable
    X = pd.get_dummies(s['video'], prefix='C(video)', prefix_sep='[', dtype=float)
    X.columns = [f'{c}]' for c in X.columns]         # same names as the formula fit
    X = X.loc[:, X.sum() > 0]
    return sm.MixedLM(endog=pd.Series(s['delta'].to_numpy(float) * sw, index=s.index, name='delta'),
                      exog=X.mul(sw, axis=0), groups=s['person'],
                      exog_re=pd.DataFrame({'person': sw}, index=s.index)).fit(reml=True)


def stage2(contacts, pooled, delta_type):
    d = contacts[contacts.delta_type == delta_type]
    cells, inter = [], []
    for band, reg in itertools.product(BANDS, sorted(d.region.unique())):
        s = d[(d.band == band) & (d.region == reg)]
        s = s[s.groupby('video').person.transform('nunique') >= MIN_PERSONS]
        if s.empty:
            continue
        s = s.copy(); s['video'] = pd.Categorical(s['video'], categories=[v for v in VIDEOS if v in set(s.video)])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = fit_stage2(s)
        names = list(m.fe_params.index); b = m.fe_params.values
        V = m.cov_params().loc[names, names].values
        for i, nme in enumerate(names):
            vid = re.search(r'\[(.+)\]', nme).group(1)
            sv = s[s.video == vid]
            # comparisons: naive pooled t, within-patient t
            p_ = pooled[(pooled.band == band) & (pooled.region == reg) & (pooled.video == vid)]
            nt, npv = stats.ttest_ind(p_[p_.state == 'Internal'].y, p_[p_.state == 'External'].y) if len(p_) else (np.nan, np.nan)
            pm = sv.groupby('patient').delta.mean()
            wt, wp = stats.ttest_1samp(pm, 0) if len(pm) >= 3 else (np.nan, np.nan)
            cells.append(dict(band=band, region=reg, video=vid, est=b[i], se=np.sqrt(V[i, i]),
                              z=b[i] / np.sqrt(V[i, i]), p=2 * stats.norm.sf(abs(b[i] / np.sqrt(V[i, i]))),
                              n_contacts=len(sv), n_persons=sv.person.nunique(),
                              mean_int=sv.mean_int.mean(), mean_ext=sv.mean_ext.mean(),
                              naive_t=nt, naive_p=npv, within_t=wt, within_p=wp, n_patients_within=len(pm)))
        if len(names) >= 2:   # equal-effects Wald test = state x movie interaction
            L = np.zeros((len(names) - 1, len(names)))
            for k in range(len(names) - 1):
                L[k, k] = 1; L[k, k + 1] = -1
            chi2 = float((L @ b) @ np.linalg.solve(L @ V @ L.T, L @ b))
            inter.append(dict(band=band, region=reg, chi2=chi2, df=len(names) - 1,
                              p=stats.chi2.sf(chi2, len(names) - 1), n_videos=len(names)))
    cells, inter = pd.DataFrame(cells), pd.DataFrame(inter)
    cells['p_fdr'] = np.nan
    for vid in VIDEOS:
        m = (cells.video == vid) & cells.p.notna()
        if m.any():
            cells.loc[m, 'p_fdr'] = multipletests(cells.loc[m, 'p'], method='fdr_bh')[1]
    if len(inter):
        inter['p_fdr'] = np.nan
        m = inter.p.notna()
        inter.loc[m, 'p_fdr'] = multipletests(inter.loc[m, 'p'], method='fdr_bh')[1]
    return cells, inter


# -----------------------------------------------------------------------------
def stars(p):
    return '' if pd.isna(p) else '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''


def heatmap(mat, ann, title, cbar_label, out, vlim=None, cmap='coolwarm'):
    fig, ax = plt.subplots(figsize=(4.6, 0.32 * len(mat) + 1.4))
    v = vlim if vlim is not None else np.nanmax(np.abs(mat.values))
    im = ax.imshow(mat.values, cmap=cmap, vmin=-v if cmap == 'coolwarm' else 0, vmax=v, aspect='auto')
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if pd.notna(mat.values[i, j]):
                ax.text(j, i, ann.values[i, j], ha='center', va='center', fontsize=8,
                        color='white' if abs(mat.values[i, j]) > 0.6 * v else '#0b0b0b')
    ax.set_xticks(range(mat.shape[1])); ax.set_xticklabels(mat.columns, fontsize=8)
    ax.set_yticks(range(mat.shape[0])); ax.set_yticklabels(mat.index, fontsize=8)
    ax.set_title(title, fontsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label(cbar_label, fontsize=8); cb.ax.tick_params(labelsize=7)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout(); fig.savefig(out, dpi=300, bbox_inches='tight'); plt.close(fig)


def boxplots_all_bands(contacts, cells, vid, delta_type, out):
    """One figure per video: rows = bands, per-person region means Int vs Ext."""
    d_all = contacts[(contacts.delta_type == delta_type) & (contacts.video == vid)]
    regs = sorted(cells[cells.video == vid].region.unique())
    if not regs:
        return
    fig, axes = plt.subplots(len(BANDS), 1, figsize=(max(7, 0.62 * len(regs)), 2.3 * len(BANDS)), sharex=True)
    for ax, band in zip(axes, BANDS):
        d = d_all[d_all.band == band]
        pm = d.groupby(['region', 'patient'])[['mean_int', 'mean_ext']].mean().reset_index()
        cb = cells[(cells.video == vid) & (cells.band == band)].set_index('region')
        for i, reg in enumerate(regs):
            r = pm[pm.region == reg]
            if r.empty:
                continue
            xi, xe = i - 0.18, i + 0.18
            for vals, x, col in ((r.mean_int, xi, COLOR_INT), (r.mean_ext, xe, COLOR_EXT)):
                ax.boxplot(vals, positions=[x], widths=0.28, patch_artist=True, showfliers=False,
                           boxprops=dict(facecolor=col, alpha=0.35, lw=0.7), medianprops=dict(color='#0b0b0b'),
                           whiskerprops=dict(lw=0.7), capprops=dict(lw=0.7))
            for _, row in r.iterrows():
                ax.plot([xi, xe], [row.mean_int, row.mean_ext], color='#b8b7b3', lw=0.7, zorder=1)
            if reg in cb.index and stars(cb.loc[reg, 'p_fdr']):
                ax.text(i, ax.get_ylim()[1], stars(cb.loc[reg, 'p_fdr']), ha='center', va='bottom', fontsize=8)
        ax.axhline(0, color='#52514e', lw=0.6, ls='--')
        ax.set_ylabel(f'{band}\n(robust z)', fontsize=9)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        ax.tick_params(labelsize=8)
    axes[-1].set_xticks(range(len(regs))); axes[-1].set_xticklabels(regs, rotation=45, ha='right', fontsize=8)
    axes[0].set_title(f'{VID_SHORT[vid]} - per-patient region means, Internal (blue) vs External (grey); '
                      f'stars = mixed-model FDR ({LABEL_SOURCE} PC, delta {delta_type})', fontsize=9)
    fig.tight_layout(h_pad=0.6)
    fig.savefig(out, dpi=200, bbox_inches='tight'); plt.close(fig)


# -----------------------------------------------------------------------------
if __name__ == '__main__':
    labels = pd.read_csv(LABELS)
    included = pd.read_csv(INCLUDED)
    print(f'{len(included)} recordings; atlas {ATLAS}; labels {LABEL_SOURCE} ({VARIANT}); scheme {SCHEME}; '
          f'stat {STAT}; stride {STRIDE}; min_win {MIN_WIN}; weights {WEIGHTS}')
    for z_thr in THRESHOLDS:
        tag = (('' if STAT == 'trim20' else f'_{STAT}') + ('' if STRIDE == 4 else f'_stride{STRIDE}')
               + ('' if MIN_WIN == 3 else f'_minwin{MIN_WIN}') + ('' if WEIGHTS is None else f'_w{WEIGHTS}'))
        # trim20 / stride 4 keep the original dir names; e.g. '..._z0.6_mean', '..._z0.6_stride1'
        od = os.path.join(OUT_DIR, f'{ATLAS.split("_")[0]}_{LABEL_SOURCE}_{SCHEME}_z{z_thr}{tag}')
        os.makedirs(od, exist_ok=True)
        print(f'\n=== z = {z_thr} ===')
        contacts, pooled = stage1(labels, included, SCHEME, z_thr)
        for dt in DELTAS:
            cells, inter = stage2(contacts, pooled, dt)
            contacts[contacts.delta_type == dt].to_csv(os.path.join(od, f'contacts_{dt}.csv'), index=False)
            cells.to_csv(os.path.join(od, f'cells_{dt}.csv'), index=False)
            inter.to_csv(os.path.join(od, f'interaction_{dt}.csv'), index=False)
            regs = sorted(cells.region.unique())
            vmax = np.nanmax(np.abs(cells.z))
            for vid in VIDEOS:
                cv = cells[cells.video == vid]
                if cv.empty:
                    continue
                Z = cv.pivot(index='region', columns='band', values='z').reindex(index=regs, columns=BANDS)
                A = cv.pivot(index='region', columns='band', values='p_fdr').reindex(index=regs, columns=BANDS).applymap(stars)
                heatmap(Z, A, f'{VID_SHORT[vid]}: Internal − External, {ATLAS.split("_")[0]}, {LABEL_SOURCE} PC\n'
                        f'mixed model z (contacts in persons), {SCHEME}, z_thr {z_thr}, delta {dt}',
                        'z (Int − Ext)', os.path.join(od, f'heatmap_{vid}_{dt}.png'), vlim=vmax)
            if len(inter):
                C = inter.pivot(index='region', columns='band', values='chi2').reindex(index=regs, columns=BANDS)
                A = inter.pivot(index='region', columns='band', values='p_fdr').reindex(index=regs, columns=BANDS).applymap(stars)
                heatmap(C, A, f'State × movie interaction (Wald χ², equal Int−Ext across movies)\n'
                        f'{ATLAS.split("_")[0]}, {SCHEME}, z_thr {z_thr}, delta {dt}',
                        'χ²', os.path.join(od, f'heatmap_interaction_{dt}.png'), cmap='Blues')
            if z_thr in BOXPLOTS_FOR:
                for vid in VIDEOS:
                    boxplots_all_bands(contacts, cells, vid, dt, os.path.join(od, f'boxplots_{vid}_{dt}.png'))
            sig = cells[cells.p_fdr < 0.05]
            print(f'  delta={dt}: {len(cells)} cells, {len(sig)} FDR<0.05; '
                  f'interaction FDR<0.05: {(inter.p_fdr < 0.05).sum() if len(inter) else 0}')
        print(f'  -> {od}')
