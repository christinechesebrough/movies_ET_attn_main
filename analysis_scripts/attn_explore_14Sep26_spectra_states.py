"""
Exploration track, step 10: state-averaged wavelet spectra per Y7 network
(not time-resolved) and the Internal - External contrast per frequency.

Inputs: as attn_explore_14Sep26_spectrograms.py (windowed wavelets, Tier 2 atlas
rows, the track's label file, the neural inclusion list).

States: the six PC1 regions (within-viewer PC1 z: < -1.5 | -1.5..-0.4 |
-0.4..0.4 | 0.4..0.8 | 0.8..1.2 | > 1.2), deviation-gated with the
BY-TIMEPOINT deviation for both ends (Christine 2026-09-15): internal regions
require mahal_time_z > GATE_INT_T (0.6); external regions require
mahal_time_z <= GATE_EXT_MAX (0). Knobs below switch either gate to the
within-subject deviation. Windows failing a gate are dropped.

Per recording: contact z (robust, per contact x frequency) minus that contact's
mean over all good windows per frequency (contact-relative); per contact and
region the mean spectrum (>= MIN_WIN windows); network = mean over contacts.
Heatmap = mean over recordings of the network means (frequency x region).
Contrast = Internal (internal + far internal) minus External (external + far
external) per contact, window-weighted; per frequency a MixedLM with person
random intercept over contacts; FDR over the 76 frequencies within network and
film. Peak-free summary: mean contrast per band.

Outputs: {MOVIE_DATA}/attn_explore_14Sep26/spectra_states/{spectra.npz, contrast.csv}
         reports/spectrograms/spectra_{network_slug}.png, spectra_summary.png
"""
import os, sys, glob, re, warnings
import numpy as np, pandas as pd, h5py
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_attn_states_lmm as L
from attn_explore_14Sep26_spectrograms import find_windowed, WIN_DIR, VN, Y7
warnings.filterwarnings('ignore')
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(v, '1')

MOVIE_DATA = L.MOVIE_DATA
LABELS = f'{MOVIE_DATA}/attention_labels_10s/attention_labels_10s_pooled_explore14Sep26.csv'
OUT = f'{MOVIE_DATA}/attn_explore_14Sep26/spectra_states'
FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'reports', 'spectrograms')
VIDS = L.VIDEOS
ATLAS = 'Y7'           # 'Y7' | 'Y17'   (figure files get a _y17 suffix for Y17)
SUF = '' if ATLAS == 'Y7' else '_y17'
EDGES = [-np.inf, -1.5, -0.4, 0.4, 0.8, 1.2, np.inf]
REG = ['far external', 'external', 'middle', 'ambiguous', 'internal', 'far internal']
GATE_INT_COL, GATE_INT_T = 'mahal_time_z', 0.6
GATE_EXT_COL, GATE_EXT_MAX = 'mahal_time_z', 0.0
MIN_WIN, MIN_CONTACTS, MIN_PERSONS = 3, 5, 3
MIN_REC = 5            # films with fewer windowed recordings than this are shown as pending
BANDS = [('delta', 1, 3), ('theta', 4, 7), ('alpha', 8, 13), ('beta', 14, 30), ('gamma', 31, 50), ('HFA', 51, 150)]
VLIM = 0.12


def stage2(d):
    try:
        m = smf.mixedlm('delta ~ 1', d, groups=d['person']).fit(reml=True); est, se = float(m.params['Intercept']), float(m.bse['Intercept'])
    except Exception:
        est, se = d.delta.mean(), d.delta.sem()
    z = est / se if se > 0 else np.nan
    return est, se, z, (2 * (1 - stats.norm.cdf(abs(z))) if np.isfinite(z) else np.nan)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True); os.makedirs(FIG, exist_ok=True)
    def nets_of(vid, pat, run):
        f = f'{L.T2_DIR}/windowed_robustz_power_log_{vid}_alpha/{pat}/{pat}_{run}_{vid}_alpha_power_log_robustz_rolling_{L.STAT}_10s.csv'
        if not os.path.exists(f):
            return {}
        a = pd.read_csv(f, nrows=4).set_index('Atlas').iloc[:, 2:]
        row = a.loc['Y7_Atlas_Region'] if ATLAS == 'Y7' else a.loc['Y17_Atlas_Region']
        return {c: (L.Y7_TO_LONG.get(str(v), str(v)) if ATLAS == 'Y7' else str(v)) for c, v in row.items()}
    labels = pd.read_csv(LABELS); included = pd.read_csv(L.INCLUDED)
    NETS = Y7 if ATLAS == 'Y7' else sorted({v for r in included.itertuples() for v in nets_of(r.video, r.patient, r.run).values()} - L.EXCLUDE_REGIONS)
    net_state = {}   # (net, vid) -> list of (regions x freq) arrays per recording
    contact_rows = []  # per contact: Internal, External mean spectra (freq arrays)
    freqs = None; nrec = {}
    for r in included.itertuples():
        f = find_windowed(r.video, r.patient, r.run)
        if f is None:
            continue
        try:
            hh = h5py.File(f, 'r')
        except OSError as exc:
            print(f'  SKIP unreadable {os.path.basename(f)[:60]}'); continue
        with hh as h:
            P = h['pow_log_mean'][:]; m = h['robust_median'][:]; s = h['robust_sd'][:]
            labs = [x.decode() if isinstance(x, bytes) else str(x) for x in h['labels_ip'][:]]; freqs = h['freqs_tf'][:]; fbad = h['frac_bad'][:]
        Z = ((P - m[:, :, None]) / s[:, :, None])[:, :, :236]; fbad = fbad[:236]; nwin = Z.shape[2]
        lab = labels[(labels.video == r.video) & (labels.pat == r.patient)].set_index('window_idx_10s')
        pc = lab['PC1_z'].reindex(range(nwin)).to_numpy(float); di = lab[GATE_INT_COL].reindex(range(nwin)).to_numpy(float); de = lab[GATE_EXT_COL].reindex(range(nwin)).to_numpy(float)
        good = ~(fbad > 0) & np.isfinite(pc)
        region = pd.cut(pc, EDGES, labels=REG).astype(object)
        region[np.isin(region, ['internal', 'far internal']) & ~(di > GATE_INT_T)] = None
        region[np.isin(region, ['external', 'far external']) & ~(de <= GATE_EXT_MAX)] = None
        Z = Z - np.nanmean(Z[:, :, good], axis=2, keepdims=True)                 # contact-relative
        nets = nets_of(r.video, r.patient, r.run); nrec[r.video] = nrec.get(r.video, 0) + 1
        person = L.person_of(r.patient)
        for net in NETS:
            idx = [i for i, c in enumerate(labs) if nets.get(c) == net]
            if not idx:
                continue
            S = np.full((len(REG), len(freqs)), np.nan)
            for k, reg in enumerate(REG):
                sel = good & (region == reg)
                if sel.sum() >= MIN_WIN:
                    S[k] = np.nanmean(np.nanmean(Z[idx][:, :, sel], axis=2), axis=0)
            net_state.setdefault((net, r.video), []).append(S)
            si = good & np.isin(region, ['internal', 'far internal']); se_ = good & np.isin(region, ['external', 'far external'])
            if si.sum() >= MIN_WIN and se_.sum() >= MIN_WIN:
                for i in idx:
                    contact_rows.append(dict(network=net, video=r.video, person=person, contact=f'{r.patient}_{r.run}_{labs[i]}', n_int=int(si.sum()), n_ext=int(se_.sum()),
                                             delta=np.nanmean(Z[i][:, si], axis=1) - np.nanmean(Z[i][:, se_], axis=1)))
        print(f'  {r.video:24s} {r.patient:10s} states {dict(zip(*np.unique(region[good].astype(str), return_counts=True)))}', flush=True)
    print('recordings used:', nrec)
    # contrast per frequency
    rows = []
    for (net, vid), grp in pd.DataFrame(contact_rows).groupby(['network', 'video']):
        if len(grp) < MIN_CONTACTS or grp.person.nunique() < MIN_PERSONS or nrec.get(vid, 0) < MIN_REC:
            continue
        D = np.stack(grp.delta.to_numpy())
        for j, fr in enumerate(freqs):
            d = pd.DataFrame({'delta': D[:, j], 'person': grp.person.to_numpy()}).dropna()
            est, se, z, p = stage2(d); rows.append(dict(network=net, video=vid, freq=float(fr), est=est, se=se, z=z, p=p, n_contacts=len(d), n_persons=d.person.nunique()))
    R = pd.DataFrame(rows); R['p_fdr'] = np.nan
    for key, idx in R.groupby(['network', 'video']).groups.items():
        R.loc[idx, 'p_fdr'] = multipletests(R.loc[idx, 'p'].fillna(1), method='fdr_bh')[1]
    R['sig'] = R.p_fdr < 0.05
    R.to_csv(f'{OUT}/contrast{SUF}.csv', index=False)
    net_state = {k: v for k, v in net_state.items() if len(v) >= MIN_REC}
    H = {k: np.nanmean(np.stack(v), axis=0) for k, v in net_state.items()}
    np.savez_compressed(f'{OUT}/spectra{SUF}.npz', freqs=freqs, regions=np.array(REG), keys=np.array([f'{k[0]}|{k[1]}' for k in H]), heat=np.stack(list(H.values())), n=np.array([len(net_state[k]) for k in H]))
    # per-band summary of the contrast
    R['band'] = pd.cut(R.freq, [0] + [b[2] + 0.5 for b in BANDS], labels=[b[0] for b in BANDS])
    print('\nInternal - External per band (mean z across frequencies in band):'); print(R.groupby(['video', 'network', 'band'], observed=True).z.mean().unstack().round(2).to_string())
    # figures per network
    vids_present = [v for v in VIDS if any(k[1] == v for k in H)]
    for net in NETS:
        fig, axes = plt.subplots(len(VIDS), 2, figsize=(11, 3.3 * len(VIDS)), gridspec_kw={'width_ratios': [1.15, 1.6]}, squeeze=False)
        for i, vid in enumerate(VIDS):
            ax = axes[i, 0]
            if (net, vid) in H:
                fe = np.r_[freqs - 1, freqs[-1] + 1]; im = ax.pcolormesh(np.arange(len(REG) + 1), fe, H[(net, vid)].T, cmap='RdBu_r', norm=TwoSlopeNorm(vcenter=0, vmin=-VLIM, vmax=VLIM), shading='flat', rasterized=True)
                ax.set_yscale('log'); ax.set_ylim(1, 152); ax.set_yticks([2, 4, 8, 16, 32, 64, 150]); ax.set_yticklabels(['2', '4', '8', '16', '32', '64', '150']); ax.set_xticks(np.arange(len(REG)) + 0.5); ax.set_xticklabels([r.replace(' ', '\n') for r in REG], fontsize=7.5)
                ax.set_title(f'{VN[vid]}: mean spectrum per state ({len(net_state[(net, vid)])} recordings)', fontsize=8.5); ax.set_ylabel('frequency (Hz)', fontsize=8)
            else:
                ax.text(0.5, 0.5, 'windowed wavelets pending', ha='center', va='center', fontsize=9, color='#8a9391', transform=ax.transAxes); ax.set_title(VN[vid], fontsize=8.5); ax.set_xticks([]); ax.set_yticks([])
            ax.tick_params(labelsize=7)
            ax = axes[i, 1]; rr = R[(R.network == net) & (R.video == vid)].sort_values('freq')
            if len(rr):
                ax.fill_between(rr.freq, rr.est - rr.se, rr.est + rr.se, color='#5546b8', alpha=0.18, lw=0); ax.plot(rr.freq, rr.est, color='#5546b8', lw=1.6)
                sg = rr[rr.sig]; ax.scatter(sg.freq, sg.est, s=14, color='#151b1a', zorder=3, label='FDR < 0.05')
                ax.axhline(0, color='#c9d0ce', lw=0.8); ax.set_xscale('log'); ax.set_xticks([2, 4, 8, 16, 32, 64, 150]); ax.set_xticklabels(['2', '4', '8', '16', '32', '64', '150'])
                for b, lo, hi in BANDS:
                    ax.axvspan(lo - 0.5, hi + 0.5, color='#eef1f0' if BANDS.index((b, lo, hi)) % 2 else 'white', zorder=0, lw=0); ax.text(np.sqrt(max(lo, 1) * hi), ax.get_ylim()[1] * 0.92 if ax.get_ylim()[1] > 0 else 0, b, ha='center', fontsize=7, color='#8a9391')
                ax.set_title(f'Internal minus External, mixed-model estimate ± SE over contacts ({rr.n_contacts.iloc[0]} contacts, {rr.n_persons.iloc[0]} persons)', fontsize=8.5)
                ax.set_xlabel('Hz', fontsize=8); ax.set_ylabel('z difference', fontsize=8); ax.legend(fontsize=7, frameon=False, loc='lower left')
            else:
                ax.set_xticks([]); ax.set_yticks([])
            ax.tick_params(labelsize=7)
        fig.suptitle(f'{net}: wavelet power by gaze state (six PC1 regions, by-timepoint deviation gate: internal > {GATE_INT_T}, external <= {GATE_EXT_MAX:g})', fontsize=10)
        fig.tight_layout(rect=[0, 0, 0.92, 0.96]); cax = fig.add_axes([0.935, 0.2, 0.012, 0.6]); fig.colorbar(im, cax=cax).set_label('z, contact-relative, network mean', fontsize=8); cax.tick_params(labelsize=7)
        slug = re.sub(r'[^a-z0-9]+', '_', net.lower()).strip('_'); fig.savefig(f'{FIG}/spectra{SUF}_{slug}.png', dpi=130); plt.close(fig); print('wrote spectra' + SUF + '_' + slug)
    # summary: networks x frequency, Internal - External z, one panel per film
    fig, axes = plt.subplots(1, len(VIDS), figsize=(5.2 * len(VIDS), 3.6 if ATLAS == 'Y7' else 6.2), squeeze=False)
    for j, vid in enumerate(VIDS):
        ax = axes[0, j]; rr = R[R.video == vid]
        if len(rr):
            M = rr.pivot(index='network', columns='freq', values='z').reindex(NETS); Sg = rr.pivot(index='network', columns='freq', values='sig').reindex(NETS)
            im = ax.pcolormesh(np.r_[freqs - 1, freqs[-1] + 1], np.arange(len(NETS) + 1), M.values, cmap='RdBu_r', norm=TwoSlopeNorm(vcenter=0, vmin=-5, vmax=5), shading='flat', rasterized=True)
            ys, xs = np.where(Sg.values == True); ax.scatter(freqs[xs], ys + 0.5, s=5, color='#151b1a')
            ax.set_xscale('log'); ax.set_xticks([2, 4, 8, 16, 32, 64, 150]); ax.set_xticklabels(['2', '4', '8', '16', '32', '64', '150']); ax.set_yticks(np.arange(len(NETS)) + 0.5); ax.set_yticklabels([n.split(' (')[0] for n in NETS], fontsize=7)
            ax.set_title(f'{VN[vid]}: Internal minus External, z per frequency (dots FDR < 0.05)', fontsize=8.5); ax.set_xlabel('Hz', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'pending', ha='center', va='center', color='#8a9391', transform=ax.transAxes); ax.set_title(VN[vid], fontsize=8.5); ax.set_xticks([]); ax.set_yticks([])
        ax.tick_params(labelsize=7)
    fig.tight_layout(); fig.savefig(f'{FIG}/spectra{SUF}_summary.png', dpi=130); plt.close(fig); print('wrote spectra' + SUF + '_summary'); print(f'-> {OUT}')
