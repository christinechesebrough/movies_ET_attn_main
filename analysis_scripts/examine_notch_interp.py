#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Before/after figures for the notch interpolation in
`extract_fooof_broadband_interp.py`.

Reads the same .fif files and the two CSVs that script writes per recording
(`*_fooof_broadband.csv` = interpolated, `*_fooof_broadband_nointerp.csv` =
reference) and produces, per recording:

    1. whole-recording channel-median PSD, raw vs interpolated, with the
       detected notch ranges shaded and the knee-mode aperiodic fit on each
    2. zoom on every notch: raw bins, interpolated bins, buffer bins
    3. one channel-window example: full FOOOF model (aperiodic + peaks),
       raw vs interpolated
    4. from the CSVs: distributions of exponent, knee frequency, R2, peak
       count; peak centre-frequency histogram; per-band periodic power

No fitting parameters are chosen here - everything is imported from the
extraction script so the figures show exactly what production does.

Output: reports/notch_interp_15Sep26/*.png
"""

import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mne.time_frequency import psd_array_welch
from fooof import FOOOF

warnings.filterwarnings('ignore')

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
import extract_fooof_broadband_interp as X   # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(_here), 'reports', 'notch_interp_15Sep26')
os.makedirs(OUT_DIR, exist_ok=True)

EXAMPLE_WINDOW = 40          # window index for the single-window example
EXAMPLE_CHANNELS = 2         # how many channels to show

RAW_C, INT_C = '#b3572a', '#2b6f9e'


def make_fm():
    return FOOOF(peak_width_limits=X.PEAK_WIDTH_LIMITS, max_n_peaks=X.MAX_N_PEAKS,
                 min_peak_height=X.MIN_PEAK_HEIGHT, aperiodic_mode=X.APERIODIC_MODE,
                 peak_threshold=X.PEAK_THRESHOLD, verbose=False)


def fit_pair(freqs, p_raw, p_int):
    fr, fi = make_fm(), make_fm()
    fr.fit(freqs, p_raw, [X.FIT_LO, X.FIT_HI])
    fi.fit(freqs, p_int, [X.FIT_LO, X.FIT_HI])
    return fr, fi


def fm_label(fm, name):
    ap = fm.aperiodic_params_
    return (f'{name}: exp {ap[-1]:.2f}, f_knee {X.knee_freq(ap):.1f} Hz, '
            f'R2 {fm.r_squared_:.3f}, {fm.n_peaks_} peaks')


def shade(ax, tab):
    for r in tab.itertuples():
        ax.axvspan(r.lo_hz, r.hi_hz, color='0.85', zorder=0)


# =============================================================================
summary_rows = []
for vid, pat, fif_path in X.list_recordings(patients=X.PATIENTS):
    fname = os.path.basename(fif_path)
    ses, run = X._ses_label(fname), X._run_label(fname)
    entry_id = f'{pat}_{ses}_{run}' if ses else f'{pat}_{run}'
    print(entry_id, flush=True)

    lfp, labels, fs, meta = X.load_lfp(pat, fif_path)
    n_fft = int(round(fs * X.N_FFT_SEC))
    psd_hi = min(X.FIT_HI + X.PSD_PAD_HZ, fs / 2)
    psd_rec, f = psd_array_welch(lfp, sfreq=fs, fmin=X.FIT_LO, fmax=psd_hi,
                                 n_fft=n_fft, n_overlap=n_fft // 2, average='mean')
    tab = X.detect_notches(f, psd_rec)
    ranges = [(r.lo_hz, r.hi_hz) for r in tab.itertuples()]
    psd_int, _ = X.interpolate_notches(f, psd_rec, ranges)

    med_raw = np.median(psd_rec, axis=0)
    med_int = np.median(psd_int, axis=0)
    fr, fi = fit_pair(f, med_raw, med_int)

    # ---- 1 + 2: recording-median PSD, full range and notch zooms ----
    n_z = len(ranges)
    fig = plt.figure(figsize=(13, 4.2 + 3.2))
    gs = fig.add_gridspec(2, max(n_z, 1), height_ratios=[4.2, 3.2])
    ax = fig.add_subplot(gs[0, :])
    shade(ax, tab)
    ax.loglog(f, med_raw, color=RAW_C, lw=1.2, label='raw (notched) median PSD')
    ax.loglog(f, med_int, color=INT_C, lw=1.2, label='interpolated')
    ax.loglog(fr.freqs, 10 ** fr._ap_fit, color=RAW_C, ls='--', lw=1,
              label=fm_label(fr, 'aperiodic, raw'))
    ax.loglog(fi.freqs, 10 ** fi._ap_fit, color=INT_C, ls='--', lw=1,
              label=fm_label(fi, 'aperiodic, interp'))
    ax.axvline(X.FIT_HI, color='k', ls=':', lw=0.8)
    ax.set_xlim(X.FIT_LO, psd_hi)
    ax.set_xlabel('Hz'); ax.set_ylabel('power (V$^2$/Hz)')
    ax.set_title(f'{entry_id}  {vid}  channel-median PSD over the whole recording '
                 f'({len(labels)} ch)  |  notches: '
                 + ', '.join(f'{c:g}' for c in tab.centre_hz))
    ax.legend(fontsize=8, loc='lower left')
    for k, (lo, hi) in enumerate(ranges):
        axz = fig.add_subplot(gs[1, k])
        m = (f >= lo - 6) & (f <= hi + 6)
        axz.semilogy(f[m], med_raw[m], 'o-', color=RAW_C, ms=3, lw=0.8, label='raw')
        axz.semilogy(f[m], med_int[m], 's-', color=INT_C, ms=3, lw=0.8, label='interpolated')
        inside = (f >= lo) & (f <= hi)
        i0, i1 = np.flatnonzero(inside)[[0, -1]]
        buf = np.r_[np.arange(i0 - X.INTERP_BUFFER, i0), np.arange(i1 + 1, i1 + 1 + X.INTERP_BUFFER)]
        buf = buf[(buf >= 0) & (buf < len(f))]
        axz.semilogy(f[buf], med_raw[buf], 'o', mfc='none', mec='k', ms=8, label='buffer bins')
        axz.axvspan(lo, hi, color='0.85', zorder=0)
        axz.set_title(f'notch {tab.centre_hz.iloc[k]:g} Hz  ({lo:g}-{hi:g} interpolated)', fontsize=9)
        axz.set_xlabel('Hz')
        if k == 0:
            axz.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f'{entry_id}_psd_before_after.png'), dpi=130)
    plt.close(fig)

    # ---- 3: single channel-window examples with the full FOOOF model ----
    bounds = X.window_bounds(lfp.shape[1], fs)
    wi = min(EXAMPLE_WINDOW, len(bounds) - 1)
    _, s0, s1 = bounds[wi]
    pw, fw = psd_array_welch(lfp[:, s0:s1], sfreq=fs, fmin=X.FIT_LO, fmax=psd_hi,
                             n_fft=n_fft, n_overlap=n_fft // 2, average='mean')
    pw_int, _ = X.interpolate_notches(fw, pw, ranges)
    # pick channels: the median-power channel and the highest-alpha channel
    alpha = (fw >= 8) & (fw <= 12)
    order = np.argsort(np.log10(pw[:, alpha]).mean(1) - np.log10(pw).mean(1))
    chans = [order[-1], order[len(order) // 2]][:EXAMPLE_CHANNELS]
    fig, axes = plt.subplots(len(chans), 2, figsize=(13, 4 * len(chans)), squeeze=False)
    for r, ci in enumerate(chans):
        fr_, fi_ = fit_pair(fw, pw[ci], pw_int[ci])
        for c, (fm, col, name, pp) in enumerate(((fr_, RAW_C, 'raw', pw[ci]),
                                                  (fi_, INT_C, 'interpolated', pw_int[ci]))):
            a = axes[r, c]
            shade(a, tab)
            a.plot(fm.freqs, fm.power_spectrum, color='k', lw=1, label='spectrum (log10)')
            a.plot(fm.freqs, fm._ap_fit, color=col, ls='--', lw=1.2, label='aperiodic fit')
            a.plot(fm.freqs, fm.fooofed_spectrum_, color=col, lw=1.2, alpha=0.8, label='full model')
            for cf in fm.peak_params_[:, 0]:
                a.axvline(cf, color=col, lw=0.6, alpha=0.5)
            a.set_xscale('log'); a.set_xlim(X.FIT_LO, X.FIT_HI)
            a.set_title(f'{labels[ci]}  window {wi} ({s0/fs:.0f}-{s1/fs:.0f} s)  {name}\n'
                        + fm_label(fm, '').lstrip(': '), fontsize=9)
            a.set_xlabel('Hz'); a.set_ylabel('log10 power')
            if r == 0 and c == 0:
                a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f'{entry_id}_window_example.png'), dpi=130)
    plt.close(fig)

    # ---- 4: CSV-level distributions ----
    pat_dir = os.path.join(X.OUT_ROOT, pat)
    f_int = os.path.join(pat_dir, f'{entry_id}_{vid}_fooof_broadband.csv')
    f_raw = os.path.join(pat_dir, f'{entry_id}_{vid}_fooof_broadband_nointerp.csv')
    if not (os.path.exists(f_int) and os.path.exists(f_raw)):
        print('   CSVs not both present yet, skipping distributions', flush=True)
        continue
    di, dr = pd.read_csv(f_int), pd.read_csv(f_raw)
    key = ['Window_Index', 'Channel']
    m = dr.merge(di, on=key, suffixes=('_raw', '_int'))

    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    for a, col, lab, rng in zip(axes[0],
                                ['Aperiodic_Exponent', 'Knee_Freq_Hz', 'FOOOF_R2', 'N_Peaks_Fit'],
                                ['exponent', 'f_knee (Hz)', 'R2', 'n peaks'],
                                [(0, 5), (0, 40), (0.85, 1.0), (-0.5, X.MAX_N_PEAKS + 0.5)]):
        bins = np.linspace(*rng, 41) if col != 'N_Peaks_Fit' else np.arange(-0.5, X.MAX_N_PEAKS + 1)
        a.hist(dr[col].clip(*rng), bins=bins, color=RAW_C, alpha=0.6, label='raw')
        a.hist(di[col].clip(*rng), bins=bins, color=INT_C, alpha=0.6, label='interpolated')
        a.set_title(f'{lab}   raw med {dr[col].median():.3g} | interp med {di[col].median():.3g}', fontsize=9)
        a.legend(fontsize=7)
    # peak CF histogram
    def cfs(d):
        return np.concatenate([np.array(s.split(';'), float) for s in d.Peak_CFs.dropna() if s])
    a = axes[1, 0]
    shade(a, tab)
    bins = np.arange(0, X.FIT_HI + 1, 1.0)
    a.hist(cfs(dr), bins=bins, color=RAW_C, alpha=0.6, label='raw')
    a.hist(cfs(di), bins=bins, color=INT_C, alpha=0.6, label='interpolated')
    a.set_title('fitted peak centre frequencies (all channel-windows)', fontsize=9)
    a.set_xlabel('Hz'); a.legend(fontsize=7)
    # exponent scatter
    a = axes[1, 1]
    a.plot(m.Aperiodic_Exponent_raw, m.Aperiodic_Exponent_int, '.', ms=2, alpha=0.3, color='0.3')
    lim = (0, 5); a.plot(lim, lim, 'k--', lw=0.8); a.set_xlim(lim); a.set_ylim(lim)
    r_ = np.corrcoef(m.Aperiodic_Exponent_raw, m.Aperiodic_Exponent_int)[0, 1]
    a.set_title(f'exponent per channel-window, r = {r_:.3f}', fontsize=9)
    a.set_xlabel('raw'); a.set_ylabel('interpolated')
    # knee scatter
    a = axes[1, 2]
    a.plot(m.Knee_Freq_Hz_raw, m.Knee_Freq_Hz_int, '.', ms=2, alpha=0.3, color='0.3')
    lim = (0, 40); a.plot(lim, lim, 'k--', lw=0.8); a.set_xlim(lim); a.set_ylim(lim)
    ok = m[['Knee_Freq_Hz_raw', 'Knee_Freq_Hz_int']].dropna()
    r_ = np.corrcoef(ok.Knee_Freq_Hz_raw, ok.Knee_Freq_Hz_int)[0, 1]
    a.set_title(f'f_knee per channel-window, r = {r_:.3f}', fontsize=9)
    a.set_xlabel('raw'); a.set_ylabel('interpolated')
    # per-band periodic power
    a = axes[1, 3]
    bands = list(X.BANDS)
    xr = np.arange(len(bands))
    a.bar(xr - 0.18, [dr[f'Periodic_{b}'].median() for b in bands], 0.36, color=RAW_C, label='raw')
    a.bar(xr + 0.18, [di[f'Periodic_{b}'].median() for b in bands], 0.36, color=INT_C, label='interpolated')
    a.set_xticks(xr); a.set_xticklabels(bands); a.axhline(0, color='k', lw=0.6)
    a.set_title('median Periodic_{band} (flattened spectrum mean)', fontsize=9)
    a.legend(fontsize=7)
    fig.suptitle(f'{entry_id}  {vid}  {len(di):,} channel-windows', y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f'{entry_id}_fooof_distributions.png'), dpi=130)
    plt.close(fig)

    for name, d in (('raw', dr), ('interp', di)):
        summary_rows.append(dict(
            entry=entry_id, condition=name, n=len(d),
            exponent_med=d.Aperiodic_Exponent.median(),
            exponent_sd=d.Aperiodic_Exponent.std(),
            f_knee_med=d.Knee_Freq_Hz.median(),
            r2_mean=d.FOOOF_R2.mean(),
            n_peaks_mean=d.N_Peaks_Fit.mean(),
            at_cap_pct=100 * (d.N_Peaks_Fit >= X.MAX_N_PEAKS).mean(),
            **{f'periodic_{b}': d[f'Periodic_{b}'].median() for b in bands}))

if summary_rows:
    s = pd.DataFrame(summary_rows)
    s.to_csv(os.path.join(OUT_DIR, 'summary.csv'), index=False, float_format='%.4g')
    print(s.to_string(index=False))
print('->', OUT_DIR)
