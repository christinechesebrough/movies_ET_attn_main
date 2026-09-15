#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
How much did extending the FOOOF fit from 1-57 Hz to 1-150 Hz change the
estimates?

Joins, per (Entry_ID, Video, Channel, Window_Index), the light fits
(rolling_fooof_light, 1-57 Hz, knee, cap 6) to the broadband fits
(rolling_fooof_broadband_interp, 1-150 Hz notch-interpolated, knee, cap 8) and
reports for each shared column:

    pooled r          across every channel-window
    within-channel r  median over channels of r across that channel's windows
                      (the quantity a within-channel state contrast depends on)
    mean shift        broadband - light

Nothing is refit here. Output:
    reports/notch_interp_15Sep26/broadband_vs_light_{per_recording,per_column}.csv
"""
import os, glob, sys
import numpy as np
import pandas as pd

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_here), 'src'))
from paths import MOVIE_DATA, p   # noqa: E402

LIGHT = p(MOVIE_DATA, 'rolling_fooof_light')
BROAD = p(MOVIE_DATA, 'rolling_fooof_broadband_interp')
OUT = os.path.join(os.path.dirname(_here), 'reports', 'notch_interp_15Sep26')

COLS = ['Aperiodic_Offset', 'Aperiodic_Exponent', 'Knee_Freq_Hz', 'FOOOF_R2',
        'theta_FlatMax', 'Periodic_theta', 'alpha_FlatMax', 'Periodic_alpha',
        'beta_FlatMax', 'Periodic_beta', 'gamma_FlatMax', 'Periodic_gamma']
KEY = ['Entry_ID', 'Video', 'Channel', 'Window_Index']


def within_channel_r(m, col):
    rs = []
    for _, g in m.groupby('Channel'):
        a, b = g[col + '_light'].values, g[col + '_broad'].values
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() >= 20 and a[ok].std() > 0 and b[ok].std() > 0:
            rs.append(np.corrcoef(a[ok], b[ok])[0, 1])
    return float(np.median(rs)) if rs else np.nan


rows = []
pooled = []
for fb in sorted(glob.glob(os.path.join(BROAD, '*', '*_fooof_broadband.csv'))):
    base = os.path.basename(fb).replace('_fooof_broadband.csv', '')
    pat = os.path.basename(os.path.dirname(fb))
    fl = os.path.join(LIGHT, pat, base + '_fooof_light.csv')
    if not os.path.exists(fl):
        print('no light fit for', base)
        continue
    b = pd.read_csv(fb, usecols=KEY + COLS)
    l = pd.read_csv(fl, usecols=KEY + COLS)
    m = l.merge(b, on=KEY, suffixes=('_light', '_broad'))
    if m.empty:
        print('no overlap for', base)
        continue
    pooled.append(m)
    r = dict(entry=base, video=m.Video.iloc[0], n=len(m))
    for c in COLS:
        a, bb = m[c + '_light'], m[c + '_broad']
        ok = a.notna() & bb.notna()
        r[f'r_{c}'] = np.corrcoef(a[ok], bb[ok])[0, 1]
        r[f'rwc_{c}'] = within_channel_r(m, c)
        r[f'shift_{c}'] = float((bb - a)[ok].mean())
    rows.append(r)
    print(f"{base:45s} n={len(m):6d}  exp r={r['r_Aperiodic_Exponent']:.3f} "
          f"wc={r['rwc_Aperiodic_Exponent']:.3f} shift={r['shift_Aperiodic_Exponent']:+.2f}  "
          f"knee r={r['r_Knee_Freq_Hz']:.3f}  alpha r={r['r_alpha_FlatMax']:.3f}  "
          f"gamma r={r['r_gamma_FlatMax']:.3f}", flush=True)

per_rec = pd.DataFrame(rows)
per_rec.to_csv(os.path.join(OUT, 'broadband_vs_light_per_recording.csv'), index=False, float_format='%.4g')

M = pd.concat(pooled, ignore_index=True)
summ = []
for c in COLS:
    a, bb = M[c + '_light'], M[c + '_broad']
    ok = a.notna() & bb.notna()
    summ.append(dict(column=c,
                     pooled_r=np.corrcoef(a[ok], bb[ok])[0, 1],
                     within_channel_r_median=per_rec[f'rwc_{c}'].median(),
                     per_recording_r_min=per_rec[f'r_{c}'].min(),
                     per_recording_r_median=per_rec[f'r_{c}'].median(),
                     light_mean=a[ok].mean(), broad_mean=bb[ok].mean(),
                     shift=(bb - a)[ok].mean(),
                     shift_in_light_sd=(bb - a)[ok].mean() / a[ok].std()))
summ = pd.DataFrame(summ)
summ.to_csv(os.path.join(OUT, 'broadband_vs_light_per_column.csv'), index=False, float_format='%.4g')
pd.set_option('display.width', 200)
print('\n', len(per_rec), 'recordings,', len(M), 'channel-windows joined')
print(summ.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
