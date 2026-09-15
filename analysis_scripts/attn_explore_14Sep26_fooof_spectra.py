#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exploration track: the AVERAGE SPECTRAL COMPONENTS per gaze state, instead of
scalar FOOOF features.

For every kept 10 s window of every contact this rebuilds the three curves
that the broadband FOOOF fit decomposes the spectrum into,

    total      log10 Welch PSD, notch dips interpolated (exactly as fitted)
    aperiodic  offset - log10(knee + f^exponent), from the stored per-window
               parameters of extract_fooof_broadband_interp.py
    periodic   total - aperiodic   (the flattened spectrum FOOOF fits peaks to)

and averages each over the windows of a gaze state. States are the ones in
attn_explore_14Sep26_fooof_states*.py, imported from there so they cannot
drift: six PC1 regions with the deviation gate, the three collapsed groups,
and the two-state conj_0.6 label. Two versions of every curve are kept:

    absolute   the state's mean curve for the contact
    relative   the state's mean curve minus the contact's mean over ALL its
               kept windows (the contact-relative convention of the states
               script, so a curve reads "this state minus this contact's own
               average")

Stage 1 (per recording, parallel by FOOOF_WORKER / FOOOF_N_WORKERS):
    npz per recording with contacts x states x 3 components x freqs, plus
    window counts, network labels from the Tier 2 atlas rows.
Stage 2 (AGGREGATE = True, after every recording exists):
    per (atlas, video, network, state): mean and SEM across contacts of the
    absolute and relative curves, contact and person counts. Written at full
    0.5 Hz resolution to spectra_by_state.npz and, on a log-spaced grid
    (LOG_BINS_PER_DECADE, each contact binned before averaging), to
    spectra_by_state.json for the page.

No statistics are computed here - the tests on the scalar features are in the
states script; these curves are what those features summarise.

Output ({MOVIE_DATA}/attn_explore_14Sep26/fooof_spectra/):
    rec/{entry_id}_{vid}.npz, spectra_by_state.npz, spectra_by_state.json
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import sys, glob, json, time, warnings
import numpy as np, pandas as pd
from mne.time_frequency import psd_array_welch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_attn_states_lmm as L
import attn_explore_14Sep26_fooof_states_broadband as S
import extract_fooof_broadband_interp as X
warnings.filterwarnings('ignore')

MOVIE_DATA = L.MOVIE_DATA
OUT = f'{MOVIE_DATA}/attn_explore_14Sep26/fooof_spectra'
REC_DIR = f'{OUT}/rec'
AGGREGATE = os.environ.get('FOOOF_SPECTRA_AGGREGATE', '0') == '1'
WORKER = int(os.environ.get('FOOOF_WORKER', 0)); N_WORKERS = int(os.environ.get('FOOOF_N_WORKERS', 1))
OVERWRITE = True   # 2026-09-15: params added to the npz; set False to resume a partial run

COMPONENTS = ['total', 'aperiodic', 'periodic']
STATES = S.REG + ['External', 'Middle', 'Internal', 'Internal2', 'External2', 'all']
# Internal2 / External2 = the two-state conj_0.6 label; 'all' = every kept window
PAGE_NETS = {'Y7': None,                          # None -> every network
             'Y17': S.HYP_NETS['Y17']}            # only the hypothesis networks in the JSON
MIN_WIN = S.MIN_WIN
LOG_BINS_PER_DECADE = 20      # page curves; the npz keeps the full 0.5 Hz grid


def aperiodic_curve(freqs, offset, knee, exponent):
    """fooof knee form, log10 power. Shapes: freqs (F,), params (N,) -> (N, F)."""
    return offset[:, None] - np.log10(knee[:, None] + freqs[None, :] ** exponent[:, None])


def state_masks(lab, n, bad):
    pc = lab['PC1_z'].reindex(range(n)).to_numpy(float)
    dev_int = lab[S.GATE_INT_COL].reindex(range(n)).to_numpy(float)
    dev_ext = lab[S.GATE_EXT_COL].reindex(range(n)).to_numpy(float)
    two = lab[S.TWO_STATE_LABEL].reindex(range(n)).to_numpy(object)
    keep = ~bad & np.isfinite(pc)
    region = pd.cut(pc, S.EDGES, labels=S.REG).astype(object)
    is_int = np.isin(region, ['internal', 'far internal']); is_ext = np.isin(region, ['external', 'far external'])
    region[is_int & ~(dev_int > S.GATE_T)] = None; region[is_ext & ~(dev_ext <= S.GATE_EXT_MAX)] = None
    masks = {k: keep & (region == k) for k in S.REG}
    for g in ('External', 'Middle', 'Internal'):
        masks[g] = keep & np.isin(region, [k for k, v in S.GROUP.items() if v == g])
    masks['Internal2'] = keep & (two == 'Internal'); masks['External2'] = keep & (two == 'External')
    masks['all'] = keep
    return masks


def process_recording(r, labels):
    f = S.find_file(r.patient, r.run, r.video)
    if f is None:
        print(f'  no FOOOF file: {r.video} {r.patient} {r.run}', flush=True); return
    d = pd.read_csv(f, usecols=['Window_Index', 'Channel', 'Aperiodic_Offset', 'Aperiodic_Knee', 'Aperiodic_Exponent', 'Notch_Ranges_Hz'])
    entry = os.path.basename(f).replace(f'_{r.video}_{S.FILE_SUFFIX}.csv', '')
    out = f'{REC_DIR}/{entry}_{r.video}.npz'
    if os.path.exists(out) and not OVERWRITE:
        print(f'  exists: {entry} {r.video}', flush=True); return
    t0 = time.time()
    # --- the fif, same loader and PSD settings as the extraction ---
    fifs = [p for v, p_, p in X.list_recordings(vid_list=[r.video], patients=[r.patient]) if X._run_label(os.path.basename(p)) == r.run]
    if not fifs:
        print(f'  no fif: {r.video} {r.patient} {r.run}', flush=True); return
    lfp, labels_ch, fs, meta = X.load_lfp(r.patient, fifs[0])
    n_fft = int(round(fs * X.N_FFT_SEC)); psd_hi = min(X.FIT_HI + X.PSD_PAD_HZ, fs / 2)
    ranges = [tuple(map(float, s.split('-'))) for s in str(d.Notch_Ranges_Hz.iloc[0]).split(';') if s and s != 'nan']
    bounds = X.window_bounds(lfp.shape[1], fs)
    lab = labels[(labels.video == r.video) & (labels.pat == r.patient)].set_index('window_idx_10s')
    n = min(len(bounds), int(lab.index.max()) + 1)
    bad, _ = L.load_bad(r.video, r.patient, r.run, n)
    masks = state_masks(lab, n, bad)
    # --- per-window PSD for all channels ---
    logpsd = None
    for wi, s0, s1 in bounds[:n]:
        psd, freqs = psd_array_welch(lfp[:, s0:s1], sfreq=fs, fmin=X.FIT_LO, fmax=psd_hi, n_fft=n_fft, n_overlap=n_fft // 2, average='mean')
        if ranges:
            psd, _ = X.interpolate_notches(freqs, psd, ranges)
        if logpsd is None:
            fsel = (freqs >= X.FIT_LO) & (freqs <= X.FIT_HI); fr = freqs[fsel]
            logpsd = np.full((len(labels_ch), n, fsel.sum()), np.nan, np.float32)
        logpsd[:, wi, :] = np.log10(psd[:, fsel])
    # --- aperiodic curves from the stored fits ---
    ap = np.full_like(logpsd, np.nan)
    ch_index = {c: i for i, c in enumerate(labels_ch)}
    d = d[d.Window_Index < n]
    for ch, g in d.groupby('Channel'):
        i = ch_index.get(ch)
        if i is None:
            continue
        w = g.Window_Index.to_numpy(int)
        ap[i, w, :] = aperiodic_curve(fr, g.Aperiodic_Offset.to_numpy(float), g.Aperiodic_Knee.to_numpy(float), g.Aperiodic_Exponent.to_numpy(float))
    # channels without any fit (dropped in extraction) are excluded
    fitted = np.isfinite(ap).any(axis=(1, 2))
    logpsd, ap = logpsd[fitted], ap[fitted]; chans = [c for c, k in zip(labels_ch, fitted) if k]
    per = logpsd - ap
    comp = np.stack([logpsd, ap, per], axis=2)              # ch x win x comp x freq
    valid = np.isfinite(comp).all(axis=(2, 3))              # ch x win
    # per-window aperiodic parameters, aligned to (channel, window); knee also as f_knee
    P = np.full((len(chans), n, 4), np.nan, np.float32)
    ch_index2 = {c: i for i, c in enumerate(chans)}
    for ch, g in d.groupby('Channel'):
        i = ch_index2.get(ch)
        if i is None:
            continue
        w = g.Window_Index.to_numpy(int); k = g.Aperiodic_Knee.to_numpy(float); e = g.Aperiodic_Exponent.to_numpy(float)
        P[i, w, 0] = g.Aperiodic_Offset.to_numpy(float); P[i, w, 1] = k; P[i, w, 2] = e
        P[i, w, 3] = np.where((k > 0) & (e > 0), k ** (1.0 / np.maximum(e, 1e-6)), np.nan)
    curves = np.full((len(chans), len(STATES), 3, len(fr)), np.nan, np.float32)
    params = np.full((len(chans), len(STATES), 4), np.nan, np.float32)   # offset, knee, exponent, f_knee
    n_win = np.zeros((len(chans), len(STATES)), np.int16)
    for si, st in enumerate(STATES):
        m = masks[st]
        for ci in range(len(chans)):
            mm = m & valid[ci]
            if mm.sum() >= MIN_WIN:
                curves[ci, si] = comp[ci, mm].mean(axis=0); n_win[ci, si] = mm.sum()
                params[ci, si] = np.nanmean(P[ci, mm], axis=0)
    nets = S.networks_of_contacts(r.video, r.patient, r.run)
    y17 = np.array([nets.get(c, {}).get('Y17', 'unknown') for c in chans]); y7 = np.array([nets.get(c, {}).get('Y7', 'unknown') for c in chans])
    os.makedirs(REC_DIR, exist_ok=True)
    np.savez_compressed(out, freqs=fr, states=np.array(STATES), components=np.array(COMPONENTS), contacts=np.array(chans),
                        Y17=y17, Y7=y7, curves=curves, params=params, n_win=n_win, patient=r.patient, person=L.person_of(r.patient), video=r.video, run=r.run)
    print(f'  {r.video:24s} {r.patient:10s} {len(chans):4d} contacts, {n} windows, {time.time()-t0:.0f}s', flush=True)


def aggregate():
    files = sorted(glob.glob(f'{REC_DIR}/*.npz'))
    print(f'aggregating {len(files)} recordings', flush=True)
    rows = []   # one row per contact x state: curves kept in arrays
    curves_abs, curves_rel, par_abs, par_rel, meta = [], [], [], [], []
    for f in files:
        z = np.load(f, allow_pickle=True)
        fr = z['freqs']; cv = z['curves']; nw = z['n_win']; pr = z['params']
        i_all = list(z['states']).index('all')
        for ci in range(cv.shape[0]):
            base = cv[ci, i_all]; pbase = pr[ci, i_all]
            for si, st in enumerate(z['states']):
                if nw[ci, si] < MIN_WIN or not np.isfinite(cv[ci, si]).all():
                    continue
                meta.append(dict(video=str(z['video']), patient=str(z['patient']), person=str(z['person']), contact=f"{z['patient']}_{z['run']}_{z['contacts'][ci]}",
                                 Y17=str(z['Y17'][ci]), Y7=str(z['Y7'][ci]), state=str(st), n_win=int(nw[ci, si])))
                curves_abs.append(cv[ci, si]); curves_rel.append(cv[ci, si] - base)
                par_abs.append(pr[ci, si]); par_rel.append(pr[ci, si] - pbase)
    M = pd.DataFrame(meta); A = np.stack(curves_abs); R = np.stack(curves_rel); PA = np.stack(par_abs); PR = np.stack(par_rel)
    print(f'  {len(M):,} contact-states, {M.contact.nunique():,} contacts', flush=True)
    agg = {}; keys = []
    for atlas in S.ATLASES:
        nets = sorted(n for n in M[atlas].unique() if n not in L.EXCLUDE_REGIONS)
        for vid in L.VIDEOS:
            for net in ['all'] + nets:
                sel = (M.video == vid) & ((M[atlas] == net) if net != 'all' else ~M[atlas].isin(L.EXCLUDE_REGIONS))
                for st in STATES:
                    idx = np.flatnonzero(sel & (M.state == st))
                    if len(idx) < S.MIN_CONTACTS:
                        continue
                    a, rr = A[idx], R[idx]
                    agg[(atlas, vid, net, st)] = dict(abs_mean=a.mean(0), abs_sem=a.std(0, ddof=1) / np.sqrt(len(idx)),
                                                     rel_mean=rr.mean(0), rel_sem=rr.std(0, ddof=1) / np.sqrt(len(idx)),
                                                     n_contacts=len(idx), n_persons=int(M.person.iloc[idx].nunique()))
                    keys.append((atlas, vid, net, st))
    K = np.array(keys, dtype=object)
    np.savez_compressed(f'{OUT}/spectra_by_state.npz', freqs=fr, keys=K, components=np.array(COMPONENTS),
                        abs_mean=np.stack([agg[k]['abs_mean'] for k in keys]), abs_sem=np.stack([agg[k]['abs_sem'] for k in keys]),
                        rel_mean=np.stack([agg[k]['rel_mean'] for k in keys]), rel_sem=np.stack([agg[k]['rel_sem'] for k in keys]),
                        n_contacts=np.array([agg[k]['n_contacts'] for k in keys]), n_persons=np.array([agg[k]['n_persons'] for k in keys]))
    # ---- JSON for the page: LOG-SPACED bins, LOG_BINS_PER_DECADE, selected networks ----
    # Each contact's curve is binned first (mean of the 0.5 Hz bins that fall in
    # a log bin), then mean and SEM are taken across contacts, so the SEM band
    # belongs to the plotted value. A 100 Hz bin thereby averages ~10x as many
    # raw bins as a 10 Hz bin, which is what makes the high band readable: the
    # Welch estimation noise is the same size at every frequency (~0.002 log10
    # after averaging) while the state effect above 30 Hz is only ~0.005.
    edges = 10 ** np.arange(np.log10(fr[0]), np.log10(fr[-1]) + 1e-9, 1.0 / LOG_BINS_PER_DECADE)
    edges = np.append(edges, fr[-1] + 0.25)
    which = np.digitize(fr, edges) - 1
    bins = [np.flatnonzero(which == b) for b in range(len(edges) - 1)]
    bins = [b for b in bins if b.size]                      # drop log bins narrower than the 0.5 Hz grid
    fj = [round(float(10 ** np.log10(fr[b]).mean()), 3) for b in bins]   # geometric centre
    def binned(arr):                                         # (N, 3, F) -> (N, 3, B)
        return np.stack([arr[:, :, b].mean(axis=2) for b in bins], axis=2)
    Ab, Rb = binned(A), binned(R)
    def rnd(v):
        return [round(float(x), 4) for x in v]
    J = {'freqs': fj, 'bins_per_decade': LOG_BINS_PER_DECADE, 'states': STATES, 'regions': S.REG, 'components': COMPONENTS, 'videos': L.VIDEOS, 'atlases': {}}
    for atlas in S.ATLASES:
        want = PAGE_NETS[atlas]
        nets = ['all'] + (sorted(n for n in M[atlas].unique() if n not in L.EXCLUDE_REGIONS) if want is None else list(want))
        J['atlases'][atlas] = {'networks': nets, 'data': {}}
        for vid in L.VIDEOS:
            for net in nets:
                sel = (M.video == vid) & ((M[atlas] == net) if net != 'all' else ~M[atlas].isin(L.EXCLUDE_REGIONS))
                for st in STATES:
                    idx = np.flatnonzero(sel & (M.state == st))
                    if len(idx) < S.MIN_CONTACTS:
                        continue
                    a, rr = Ab[idx], Rb[idx]
                    J['atlases'][atlas]['data'][f'{vid}|{net}|{st}'] = {
                        'abs': [rnd(a.mean(0)[c]) for c in range(3)],
                        'rel': [rnd(rr.mean(0)[c]) for c in range(3)],
                        'rel_sem': [rnd((rr.std(0, ddof=1) / np.sqrt(len(idx)))[c]) for c in range(3)],
                        # mean aperiodic parameters [offset, knee, exponent, f_knee]: absolute, and state minus contact mean
                        'par_abs': rnd(np.nanmean(PA[idx], axis=0)), 'par_rel': rnd(np.nanmean(PR[idx], axis=0)),
                        'n': int(len(idx)), 'np': int(M.person.iloc[idx].nunique())}
    json.dump(J, open(f'{OUT}/spectra_by_state.json', 'w'), separators=(',', ':'))
    print(f'  -> {OUT}/spectra_by_state.json  {os.path.getsize(f"{OUT}/spectra_by_state.json")/1e6:.1f} MB', flush=True)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    if AGGREGATE:
        aggregate()
    else:
        labels = pd.read_csv(S.LABELS, usecols=['video', 'pat', 'window_idx_10s', 'PC1_z', S.GATE_INT_COL, S.GATE_EXT_COL, S.TWO_STATE_LABEL])
        included = pd.read_csv(L.INCLUDED)
        recs = list(included.itertuples())
        if N_WORKERS > 1:
            recs = [r for i, r in enumerate(recs) if i % N_WORKERS == WORKER]
        for r in recs:
            try:
                process_recording(r, labels)
            except Exception as exc:
                print(f'  FAILED {r.video} {r.patient} {r.run}: {type(exc).__name__}: {exc}', flush=True)
        print('done', flush=True)
