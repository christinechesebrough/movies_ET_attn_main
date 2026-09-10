#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reshape FOOOF output into a per-window time series (TODO A13).

The FOOOF extraction writes two tables:

    *_aperiodic_windows_*.csv   one row per (channel, window)   - already regular
    *_peaks_long_*.csv          one row per (channel, window, PEAK) - irregular

The peaks table cannot be used as a time series directly: a window may contain
two alpha-range peaks, or none, so the number of rows per window varies. This
collapses it to ONE ROW PER (recording, channel, window) with a fixed column
set, giving a regular temporal readout of both components.

Input:
    The aggregated FOOOF peaks file (atlas labels already joined), e.g.
    rolling_fooof_aggregated_*/{vid}_all_recordings_low_mid_rolling_fooof_peaks_with_atlas.csv

Output:
    One row per (Patient, Run, Video, Channel, Window_Index) with:

        Window_Center_Sec, atlas columns          identity / anatomy
        Aperiodic_Offset, Aperiodic_Exponent      the 1/f background
        FOOOF_R2, FOOOF_Error                     fit quality
        {band}_CF, {band}_Amp, {band}_BW          largest peak in that band
        {band}_N                                  how many peaks fell in it
        N_Peaks_Total                             across all bands

    A band with no peak in a window gets NaN for CF/Amp/BW and 0 for N. That
    absence is itself informative - it means no oscillation was detectable
    above the aperiodic background - so it is preserved rather than filled.

BAND EDGES
    The canonical bands used elsewhere in this repo leave gaps (delta 1-3,
    theta 4-7, so nothing owns 3-4 Hz). Peaks are continuous and land in those
    gaps regularly, so contiguous edges are used here by default, split at the
    midpoints of the canonical gaps:

        delta   1.0 -  3.5
        theta   3.5 -  7.5
        alpha   7.5 - 13.5
        beta   13.5 - 30.5
        gamma  30.5 - 57.0     (57 = the FOOOF fit ceiling, below the 60 Hz notch)

    Set CONTIGUOUS_BANDS = False to use the strict canonical edges instead and
    discard peaks in the gaps.

PEAK QUALITY FILTERING
    Two filters are applied by default, both recording what they removed:

    1. Peaks whose Peak_Bandwidth_Hz sits at the peak_width_limits ceiling
       (10 Hz for the low_mid fits). A Gaussian pinned at its maximum width is
       usually the model absorbing a spectral slope rather than fitting an
       oscillation.
    2. Peaks within EDGE_MARGIN_HZ of the fit range boundaries, where the
       aperiodic fit is least constrained and edge artifacts collect.

    Set them to None to keep everything.

WHY THE LARGEST PEAK PER BAND
    When a band contains several peaks, the largest by Peak_Amplitude is taken
    - amplitude here is height ABOVE the aperiodic background, so it is the
    most oscillation-like. {band}_N preserves the fact that others were present,
    so a window with 3 alpha peaks is distinguishable from one with a single
    clean peak.
"""

import os
import sys
import glob
import numpy as np
import pandas as pd

_src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
if _src not in sys.path:
    sys.path.insert(0, _src)

# =============================================================================
# PARAMETERS
# =============================================================================
CONTIGUOUS_BANDS = True

BANDS_CONTIGUOUS = {          # no gaps: every peak lands in exactly one band
    'delta': (1.0, 3.5),
    'theta': (3.5, 7.5),
    'alpha': (7.5, 13.5),
    'beta':  (13.5, 30.5),
    'gamma': (30.5, 57.0),
}
BANDS_CANONICAL = {           # matches the rest of the repo; leaves gaps
    'delta': (1, 3), 'theta': (4, 7), 'alpha': (8, 13),
    'beta': (14, 30), 'gamma': (31, 50),
}

MAX_BANDWIDTH_HZ = 10.0       # the peak_width_limits ceiling for low_mid fits
EDGE_MARGIN_HZ = 1.0          # drop peaks this close to the fit boundaries

ID_COLS = ['Patient', 'Run', 'Video', 'Channel', 'Window_Index']
CARRY_COLS = ['Window_Start_Sec', 'Window_End_Sec', 'Window_Center_Sec',
              'Aperiodic_Offset', 'Aperiodic_Exponent', 'FOOOF_R2', 'FOOOF_Error',
              'Entry_ID', 'Session', 'Desikan_Killiany', 'Yeo17', 'Yeo7',
              'Aparc_Aseg', 'FreqBand']


def assign_band(cf, bands):
    """Which band a peak centre frequency falls in, or None."""
    for name, (lo, hi) in bands.items():
        if lo <= cf < hi:
            return name
    return None


def filter_peaks(df, max_bw=MAX_BANDWIDTH_HZ, edge=EDGE_MARGIN_HZ):
    """Drop implausible peaks; return (kept, report dict)."""
    n0 = len(df)
    rep = {}
    keep = pd.Series(True, index=df.index)
    if max_bw is not None:
        at_ceiling = df['Peak_Bandwidth_Hz'] >= max_bw - 1e-6
        rep['at_bandwidth_ceiling'] = int(at_ceiling.sum())
        keep &= ~at_ceiling
    if edge is not None and {'FOOOF_Fit_Range_Low', 'FOOOF_Fit_Range_High'} <= set(df.columns):
        near_edge = ((df['Peak_CF_Hz'] <= df['FOOOF_Fit_Range_Low'] + edge) |
                     (df['Peak_CF_Hz'] >= df['FOOOF_Fit_Range_High'] - edge))
        rep['near_fit_edge'] = int(near_edge.sum())
        keep &= ~near_edge
    rep['kept'] = int(keep.sum())
    rep['dropped_pct'] = 100.0 * (n0 - keep.sum()) / max(n0, 1)
    return df[keep], rep


def to_timeseries(peaks_df, bands=None, verbose=True):
    """
    Collapse a long peaks table to one row per (recording, channel, window).
    """
    bands = bands or (BANDS_CONTIGUOUS if CONTIGUOUS_BANDS else BANDS_CANONICAL)

    kept, rep = filter_peaks(peaks_df)
    if verbose:
        print(f"  peaks: {len(peaks_df):,} -> {rep['kept']:,} "
              f"({rep['dropped_pct']:.1f}% dropped)")
        for k, v in rep.items():
            if k.startswith(('at_', 'near_')):
                print(f"      {k}: {v:,}")

    kept = kept.copy()
    kept['band'] = kept['Peak_CF_Hz'].map(lambda c: assign_band(c, bands))
    unassigned = kept['band'].isna().sum()
    if verbose and unassigned:
        print(f"      outside all bands: {unassigned:,}")
    kept = kept.dropna(subset=['band'])

    # the per-window skeleton: identity, timing, aperiodic, anatomy
    carry = [c for c in CARRY_COLS if c in peaks_df.columns]
    skeleton = (peaks_df.groupby(ID_COLS, as_index=False)[carry].first())
    skeleton['N_Peaks_Total'] = (kept.groupby(ID_COLS).size()
                                 .reindex(pd.MultiIndex.from_frame(skeleton[ID_COLS]))
                                 .fillna(0).astype(int).values)

    # largest peak per (window, band), plus a count
    kept = kept.sort_values('Peak_Amplitude', ascending=False)
    top = kept.groupby(ID_COLS + ['band'], as_index=False).first()
    counts = kept.groupby(ID_COLS + ['band'], as_index=False).size()

    out = skeleton
    for name in bands:
        t = top[top['band'] == name][ID_COLS + ['Peak_CF_Hz', 'Peak_Amplitude',
                                                'Peak_Bandwidth_Hz']]
        t = t.rename(columns={'Peak_CF_Hz': f'{name}_CF',
                              'Peak_Amplitude': f'{name}_Amp',
                              'Peak_Bandwidth_Hz': f'{name}_BW'})
        c = counts[counts['band'] == name][ID_COLS + ['size']].rename(
            columns={'size': f'{name}_N'})
        out = out.merge(t, on=ID_COLS, how='left').merge(c, on=ID_COLS, how='left')
        out[f'{name}_N'] = out[f'{name}_N'].fillna(0).astype(int)

    return out.sort_values(ID_COLS).reset_index(drop=True)


if __name__ == '__main__':
    M = '/media/christine/Samsung/Movie_data'
    files = [p for p in glob.glob(os.path.join(M, 'rolling_fooof_aggregated_*',
                                               '*peaks_with_atlas.csv'))
             if not os.path.basename(p).startswith('._')]
    if not files:
        print('no aggregated FOOOF peaks file found')
        sys.exit(1)
    for src in files:
        print(f'\n{os.path.basename(src)[:70]}')
        df = pd.read_csv(src)
        ts = to_timeseries(df)
        out = src.replace('_peaks_with_atlas.csv', '_window_timeseries.csv')
        ts.to_csv(out, index=False, float_format='%.6g')
        print(f"  -> {ts.shape[0]:,} windows x {ts.shape[1]} cols")
        print(f"  -> {os.path.basename(out)[:70]}")
