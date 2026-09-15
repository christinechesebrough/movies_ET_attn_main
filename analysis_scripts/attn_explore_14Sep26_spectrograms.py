"""
Exploration track, step 9: state-conditioned wavelet spectrograms per Y7 network.

Inputs:
    {WAVELET_WINDOWED}/{vid}/{pat}/*_wavelet_raw_tf_windowed_10s.h5   (window_continuous_wavelets.py)
        pow_log_mean (ch x 76 freqs x 236 windows), robust_median / robust_sd (ch x freqs),
        freqs_tf, labels_ip, frac_bad, window_centers_sec
    Tier 2 alpha file atlas rows -> Y7 network per channel (labels match)
    {MOVIE_DATA}/attention_labels_10s/attention_labels_10s_pooled_explore14Sep26.csv
        STATE_COL (default label_conj_0.6): Internal / External / Neutral; FarExternal excluded
    {MOVIE_DATA}/descriptives_power_10s/included_recordings.csv   (neural inclusion list)

Processing, per recording:
    z = (pow_log_mean - robust_median) / robust_sd           per channel x frequency
    network map = mean of z over the network's channels     -> 76 x 236,
    then minus that recording's mean over all good windows per frequency (so a
    map shows departure from the recording's own average spectrum; the raw z is
    offset negative everywhere because the window MEAN of log power sits below
    the continuous MEDIAN the z uses as centre)
    windows with frac_bad > 0 set to NaN
    per state: windows not in that state set to NaN
    across recordings: nanmean per (network, video, state)   -> 76 x 236 (+ count of viewers per window)
    state spectrum: per recording mean over the state's windows, then mean +- SEM across recordings

Outputs:
    {MOVIE_DATA}/attn_explore_14Sep26/spectrograms/maps.npz        the arrays
    reports/spectrograms/{network_slug}.png                        one figure per network:
        rows = films, columns = Internal / External / Neutral maps + state spectra
Deliberately NOT done: no statistics; the figures are descriptive.
"""
import os, sys, glob, re, json
import numpy as np, pandas as pd, h5py
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_attn_states_lmm as L

MOVIE_DATA = L.MOVIE_DATA
WIN_DIR = '/media/christine/Data/Movie_data/wavelet_windowed_10s'
LABELS = f'{MOVIE_DATA}/attention_labels_10s/attention_labels_10s_pooled_explore14Sep26.csv'
OUT = f'{MOVIE_DATA}/attn_explore_14Sep26/spectrograms'
FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'reports', 'spectrograms')
STATE_COL = 'label_conj_0.6'
STATES = ['Internal', 'External', 'Neutral']
VIDS = L.VIDEOS
VN = {'despicable_me_english': 'Despicable Me, English', 'despicable_me_hungarian': 'Despicable Me, Hungarian', 'inscapes': 'Inscapes'}
Y7 = ['Visual Network (VN)', 'Somatomotor Network (SMN)', 'Dorsal Attention Network (DAN)', 'Ventral Attention Network (VAN)', 'Limbic Network (LN)', 'Frontoparietal Network (FPN)', 'Default Mode Network (DMN)']
VLIM = 0.5


def find_windowed(vid, pat, run):
    n = int(run.split('-')[1])
    hits = [f for f in glob.glob(f'{WIN_DIR}/{vid}/{pat}/*{vid}*windowed_10s.h5') if f'_run-{n:02d}_' in os.path.basename(f) or f'_run-{n}_' in os.path.basename(f)]
    return hits[0] if hits else None


def y7_of(vid, pat, run):
    f = f'{L.T2_DIR}/windowed_robustz_power_log_{vid}_alpha/{pat}/{pat}_{run}_{vid}_alpha_power_log_robustz_rolling_{L.STAT}_10s.csv'
    if not os.path.exists(f):
        return {}
    a = pd.read_csv(f, nrows=4).set_index('Atlas').iloc[:, 2:]
    return {c: L.Y7_TO_LONG.get(str(v), str(v)) for c, v in a.loc['Y7_Atlas_Region'].items()}


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True); os.makedirs(FIG, exist_ok=True)
    labels = pd.read_csv(LABELS); included = pd.read_csv(L.INCLUDED)
    acc = {}   # (net, vid, state) -> list of 76x236 arrays (NaN outside state)
    spec = {}  # (net, vid, state) -> list of 76 vectors
    freqs = None; tcent = None; nrec = {}
    for r in included.itertuples():
        f = find_windowed(r.video, r.patient, r.run)
        if f is None:
            continue
        try:
            hh = h5py.File(f, 'r')
        except OSError as exc:
            print(f'  SKIP unreadable {os.path.basename(f)[:60]}: {exc}'); continue
        with hh as h:
            P = h['pow_log_mean'][:]; m = h['robust_median'][:]; s = h['robust_sd'][:]
            labs = [x.decode() if isinstance(x, bytes) else str(x) for x in h['labels_ip'][:]]
            freqs = h['freqs_tf'][:]; tcent = h['window_centers_sec'][:]; fbad = h['frac_bad'][:]
        Z = (P - m[:, :, None]) / s[:, :, None]                                  # ch x freq x win
        Z = Z[:, :, :236]; fbad = fbad[:236]; tcent = tcent[:236]                # 236 is the label grid; a 237th window exists in some recordings
        nwin = Z.shape[2]
        lab = labels[(labels.video == r.video) & (labels.pat == r.patient)].set_index('window_idx_10s')[STATE_COL].reindex(range(nwin)).to_numpy(object)
        good = ~(fbad > 0)
        nets = y7_of(r.video, r.patient, r.run)
        nrec[r.video] = nrec.get(r.video, 0) + 1
        for net in Y7:
            idx = [i for i, c in enumerate(labs) if nets.get(c) == net]
            if len(idx) == 0:
                continue
            M = np.nanmean(Z[idx], axis=0)                                        # freq x win
            M = M - np.nanmean(M[:, good], axis=1, keepdims=True)                   # recording-relative: subtract this recording's mean over all good windows, per frequency
            for st in STATES:
                sel = good & (lab == st)
                A = np.full_like(M, np.nan); A[:, sel] = M[:, sel]
                acc.setdefault((net, r.video, st), []).append(A)
                if sel.sum() >= 3:
                    spec.setdefault((net, r.video, st), []).append(M[:, sel].mean(axis=1))
        print(f'  {r.video:24s} {r.patient:10s} {len(labs)} ch, states {dict(zip(*np.unique(lab[good].astype(str), return_counts=True)))}', flush=True)
    print('recordings used:', nrec)
    maps = {}
    for (net, vid, st), lst in acc.items():
        arr = np.stack(lst); maps[(net, vid, st)] = (np.nanmean(arr, axis=0), np.sum(~np.isnan(arr[:, 0, :]), axis=0))
    np.savez_compressed(f'{OUT}/maps.npz', freqs=freqs, t=tcent, keys=np.array([f'{k[0]}|{k[1]}|{k[2]}' for k in maps]),
                        maps=np.stack([maps[k][0] for k in maps]), counts=np.stack([maps[k][1] for k in maps]),
                        spec_keys=np.array([f'{k[0]}|{k[1]}|{k[2]}' for k in spec]), spec_mean=np.stack([np.mean(spec[k], axis=0) for k in spec]),
                        spec_sem=np.stack([np.std(spec[k], axis=0) / np.sqrt(len(spec[k])) for k in spec]), spec_n=np.array([len(spec[k]) for k in spec]))
    # figures
    COL = {'Internal': '#5546b8', 'External': '#9a6d00', 'Neutral': '#8e9491'}
    vids_present = [v for v in VIDS if any(k[1] == v for k in maps)]
    for net in Y7:
        fig, axes = plt.subplots(len(VIDS), 4, figsize=(15, 2.9 * len(VIDS)), gridspec_kw={'width_ratios': [3, 3, 3, 1.6]}, squeeze=False)
        for i, vid in enumerate(VIDS):
            for j, st in enumerate(STATES):
                ax = axes[i, j]
                if (net, vid, st) in maps:
                    M, cnt = maps[(net, vid, st)]
                    im = ax.pcolormesh(tcent, freqs, M, cmap='RdBu_r', norm=TwoSlopeNorm(vcenter=0, vmin=-VLIM, vmax=VLIM), shading='nearest', rasterized=True)
                    ax.set_yscale('log'); ax.set_ylim(freqs.min(), freqs.max()); ax.set_yticks([2, 4, 8, 16, 32, 64, 128]); ax.set_yticklabels(['2', '4', '8', '16', '32', '64', '128'])
                    ax.set_title(f'{VN[vid]} · {st}\n{nrec.get(vid, 0)} recordings, {cnt[cnt > 0].mean():.1f} viewers per window on average', fontsize=8)
                else:
                    ax.text(0.5, 0.5, 'windowed wavelets pending', ha='center', va='center', fontsize=9, color='#8a9391', transform=ax.transAxes); ax.set_title(f'{VN[vid]} · {st}', fontsize=8); ax.set_xticks([]); ax.set_yticks([])
                if j == 0:
                    ax.set_ylabel('frequency (Hz)', fontsize=8)
                if i == len(VIDS) - 1:
                    ax.set_xlabel('time in film (s)', fontsize=8)
                ax.tick_params(labelsize=7)
            ax = axes[i, 3]
            for st in STATES:
                k = (net, vid, st)
                if k in spec:
                    mu = np.mean(spec[k], axis=0); se = np.std(spec[k], axis=0) / np.sqrt(len(spec[k]))
                    ax.plot(freqs, mu, color=COL[st], lw=1.6, label=st); ax.fill_between(freqs, mu - se, mu + se, color=COL[st], alpha=0.18, lw=0)
            ax.axhline(0, color='#c9d0ce', lw=0.8); ax.set_xscale('log'); ax.set_xticks([2, 4, 8, 16, 32, 64, 128]); ax.set_xticklabels(['2', '4', '8', '16', '32', '64', '128'])
            ax.set_title('state spectrum\nmean ± SEM over recordings', fontsize=8); ax.set_xlabel('Hz', fontsize=8); ax.set_ylabel('z, recording-relative', fontsize=8); ax.tick_params(labelsize=7)
            if not any((net, vid, st) in spec for st in STATES):
                ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.legend(fontsize=7, frameon=False)
        fig.suptitle(f'{net}: robust-z log wavelet power by gaze state ({STATE_COL})', fontsize=11)
        cax = fig.add_axes([0.905, 0.15, 0.008, 0.7]); fig.colorbar(im, cax=cax).set_label('z, network mean, relative to the recording average', fontsize=8); cax.tick_params(labelsize=7)
        fig.tight_layout(rect=[0, 0, 0.9, 0.97])
        slug = re.sub(r'[^a-z]+', '_', net.lower()).strip('_')
        fig.savefig(f'{FIG}/{slug}.png', dpi=130); plt.close(fig); print('wrote', slug)
    print(f'-> {OUT}, {FIG}')
