#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reconstruct discrete oscillatory EVENTS from overlapping FOOOF windows (A14).

The windowed FOOOF output is redundant: windows are 10 s stepping 2.5 s, so
every sample contributes to 4 windows and a single oscillatory episode appears
in several consecutive rows. This collapses those runs into one row per event.

WHY THE OVERLAP IS NOT PURELY REDUNDANT

    A window reports only "a peak was present somewhere in these 10 s", so from
    one window the onset is unknown to within 10 s. But consecutive windows
    constrain each other. If window k-1 has no peak and window k does:

        window k-1  [ 7.5 ---------- 17.5 ]   absent
        window k         [ 10.0 ---------- 20.0 ]   present

    absent throughout 7.5-17.5 and present somewhere in 10-20 means the peak
    lies in 17.5-20.0. Onset is bounded to ONE STEP (2.5 s), not one window.

    Symmetrically, if window j has a peak and j+1 does not, the peak must lie
    in the first step of window j, so offset is bounded to [start_j,
    start_j + step].

    So the temporal precision of event boundaries is the STEP size, and the
    overlap buys real information rather than only smoothing.

WHAT COUNTS AS ONE EVENT

    A maximal run of consecutive windows in which the band has a peak, with the
    centre frequency staying within CF_TOLERANCE_HZ of the run's running median.
    A jump beyond that starts a new event, so two sequential oscillations at
    different frequencies are not merged into one.

    Runs may be interrupted by up to MAX_GAP_WINDOWS missing windows without
    splitting, since a genuine oscillation can dip below the detection
    threshold briefly.

OUTPUT (one row per event)

    Patient, Run, Video, Channel, band, atlas columns
    onset_lo_sec, onset_hi_sec      the 2.5 s bracket containing onset
    offset_lo_sec, offset_hi_sec    the bracket containing offset
    onset_sec, offset_sec           bracket midpoints, for convenience
    duration_lo_sec, duration_hi_sec   bracket on the event's true duration
    duration_sec                    bracket midpoint
    shorter_than_window             True when the run spans fewer windows than
                                    window/step, so the neighbouring windows
                                    should have seen it too - a brief burst or
                                    detection noise rather than a clean episode
    n_windows                       how many windows the run spanned
    cf_median, cf_min, cf_max       peak frequency during the event
    amp_max, amp_mean               peak amplitude during the event
    exponent_mean                   mean aperiodic exponent across the run

CAVEAT

    A window shows a peak only if the oscillation contributes enough to the
    10 s averaged spectrum. A short burst may not register, and one that does
    will bias the window's spectrum for the whole 10 s. So these are episodes
    of DETECTABLE oscillation, not precise burst boundaries. The bracket
    columns make that uncertainty explicit rather than hiding it in a midpoint.
"""

import os
import sys
import numpy as np
import pandas as pd

STEP_SEC = 2.5
WINDOW_SEC = 10.0
CF_TOLERANCE_HZ = 2.0      # a jump beyond this starts a new event
MAX_GAP_WINDOWS = 1        # allow this many missing windows inside an event
MIN_WINDOWS = 1            # discard events shorter than this

ID_COLS = ['Patient', 'Run', 'Video', 'Channel']
CARRY = ['Entry_ID', 'Session', 'Desikan_Killiany', 'Yeo17', 'Yeo7', 'Aparc_Aseg']


def _runs(present, max_gap):
    """Index ranges of contiguous True, tolerating gaps up to max_gap."""
    idx = np.flatnonzero(present)
    if idx.size == 0:
        return []
    splits = np.flatnonzero(np.diff(idx) > max_gap + 1)
    groups = np.split(idx, splits + 1)
    return [(g[0], g[-1]) for g in groups]


def events_for_series(df, band, step=STEP_SEC, window=WINDOW_SEC,
                      cf_tol=CF_TOLERANCE_HZ, max_gap=MAX_GAP_WINDOWS):
    """
    df: rows for ONE (patient, run, video, channel), sorted by window index,
        with columns {band}_CF, {band}_Amp, Window_Start_Sec, Window_End_Sec.
    """
    cf = df[f'{band}_CF'].to_numpy()
    amp = df[f'{band}_Amp'].to_numpy()
    starts = df['Window_Start_Sec'].to_numpy()
    ends = df['Window_End_Sec'].to_numpy()
    exps = df['Aperiodic_Exponent'].to_numpy() if 'Aperiodic_Exponent' in df else np.full(len(df), np.nan)
    present = ~np.isnan(cf)

    out = []
    for a, b in _runs(present, max_gap):
        # split the run where CF jumps beyond tolerance
        seg_start = a
        run_idx = [i for i in range(a, b + 1) if present[i]]
        for pos, i in enumerate(run_idx):
            last = (pos == len(run_idx) - 1)
            med = np.nanmedian(cf[seg_start:i + 1])
            jump = abs(cf[i] - med) > cf_tol if pos else False
            if jump or last:
                end_i = (i - 1) if jump else i
                sel = slice(seg_start, end_i + 1)
                nwin = int(present[sel].sum())
                if nwin >= MIN_WINDOWS:
                    s0, e0 = starts[seg_start], ends[seg_start]
                    s1, e1 = starts[end_i], ends[end_i]
                    at_start = (seg_start == 0)
                    at_end = (end_i == len(df) - 1)

                    # Onset bracket. If the preceding window had no peak, the
                    # oscillation cannot have started before that window ended,
                    # so onset is pinned to the final step of the first window.
                    on_lo = s0 if at_start else e0 - step
                    on_hi = e0
                    # Offset bracket, symmetrically.
                    off_lo = s1
                    off_hi = e1 if at_end else s1 + step

                    # Duration is a BRACKET, not a point. The lower bound can be
                    # negative arithmetically when the run spans fewer windows
                    # than window/step, which for a contiguous oscillation is a
                    # contradiction: the neighbouring windows overlap enough
                    # that they should also have seen it. Such runs are either
                    # brief bursts that tipped a single window's spectrum, or
                    # detection noise. Clamp at 0 and flag them.
                    dur_lo = max(0.0, s1 - e0)
                    dur_hi = (e1 - s0)
                    short = (s1 - e0) < 0

                    out.append(dict(
                        band=band,
                        onset_lo_sec=on_lo, onset_hi_sec=on_hi,
                        offset_lo_sec=off_lo, offset_hi_sec=off_hi,
                        onset_sec=(on_lo + on_hi) / 2,
                        offset_sec=(off_lo + off_hi) / 2,
                        duration_lo_sec=dur_lo, duration_hi_sec=dur_hi,
                        duration_sec=(dur_lo + dur_hi) / 2,
                        shorter_than_window=bool(short),
                        n_windows=nwin,
                        cf_median=np.nanmedian(cf[sel]),
                        cf_min=np.nanmin(cf[sel]), cf_max=np.nanmax(cf[sel]),
                        amp_max=np.nanmax(amp[sel]), amp_mean=np.nanmean(amp[sel]),
                        exponent_mean=np.nanmean(exps[sel]),
                    ))
                seg_start = i
    return out


def to_events(ts, bands=('theta', 'alpha', 'beta', 'gamma'), verbose=True):
    """ts: the per-window table from fooof_windows_to_timeseries.py"""
    ts = ts.sort_values(ID_COLS + ['Window_Index'])
    carry = [c for c in CARRY if c in ts.columns]
    rows = []
    for key, g in ts.groupby(ID_COLS, sort=False):
        base = dict(zip(ID_COLS, key))
        for c in carry:
            base[c] = g[c].iloc[0]
        for band in bands:
            if f'{band}_CF' not in g:
                continue
            for ev in events_for_series(g, band):
                rows.append({**base, **ev})
    ev = pd.DataFrame(rows)
    if verbose and len(ev):
        print(f"  {len(ts):,} windows -> {len(ev):,} events "
              f"({len(ts)/max(len(ev),1):.1f} windows per event)")
    return ev


if __name__ == '__main__':
    import glob
    src = [p for p in glob.glob('/media/christine/Samsung/Movie_data/'
                                'rolling_fooof_aggregated_*/*_window_timeseries.csv')
           if not os.path.basename(p).startswith('._')]
    if not src:
        print('run fooof_windows_to_timeseries.py first'); sys.exit(1)
    for s in src:
        print(os.path.basename(s)[:70])
        ev = to_events(pd.read_csv(s))
        out = s.replace('_window_timeseries.csv', '_oscillatory_events.csv')
        ev.to_csv(out, index=False, float_format='%.6g')
        print(f'  -> {os.path.basename(out)[:70]}')
