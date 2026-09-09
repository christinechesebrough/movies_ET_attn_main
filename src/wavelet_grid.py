#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Frequency grid, n_cycles profile and decimation for the full-spectrum
wavelet re-extraction (TODO A11).

Replaces the original grid (linear 2 Hz steps, fixed n_cycles = 5, stored at
600 Hz) with a single log-spaced extraction covering 0.5-151 Hz.

WHY THE ORIGINAL GRID FAILS AT BOTH ENDS

    Morlet bandwidth scales as 2f/n_cycles, so non-redundant frequency spacing
    also scales with f. A fixed 2 Hz step is therefore undersampled at the
    bottom and oversampled at the top. With n_cycles = 5:

        1 Hz wavelet covers 0.8-1.2 Hz
                            <-- 1.2 Hz GAP, nothing measures this
        3 Hz wavelet covers 2.4-3.6 Hz

    so only ~40% of the 1-3 Hz delta band is measured, while HFA gets 50
    near-duplicate frequencies whose 60 Hz bandwidth makes them barely
    frequency-selective.

DESIGN

    frequencies   100 log-spaced points, 0.5-151 Hz
    n_cycles      log-spaced 3 -> 15
                  3 at the bottom is the floor: below ~3 cycles a Morlet is not
                  frequency-selective and leaks toward DC.
                  15 at the top is affordable because a 151 Hz wavelet at
                  n_cycles = 15 is still only 0.1 s long.
    decim         6, i.e. output stored at 100 Hz rather than 600 Hz

WHY ONE DECIMATION RATE WORKS ACROSS THE WHOLE SPECTRUM

    We store POWER, and a power envelope's bandwidth equals the filter's
    bandwidth (power beats the signal against itself within the passband).
    The log n_cycles ramp keeps relative bandwidth roughly constant, so the
    envelope's Nyquist requirement stays bounded:

        0.5 Hz  nc  3.0   BW  0.3 Hz   needs   0.7 Hz
      151.0 Hz  nc 15.0   BW 20.1 Hz   needs  40.3 Hz   <- worst case

    100 Hz gives 2.5x margin on the worst case. Verified empirically at 151 Hz
    with a deliberately fast (25 Hz) amplitude modulation: 99% of envelope
    power falls below 25 Hz and only 0.0026% exceeds the 50 Hz Nyquist.

    The guarantee is structural rather than empirical: a 20 Hz-wide filter
    cannot produce envelope content beyond 20 Hz for ANY input.

    NOTE this would NOT hold for the original nc = 5 grid, where a 151 Hz
    wavelet is 60 Hz wide, needs >=121 Hz, and puts 0.35% of envelope power
    above a 50 Hz Nyquist - 130x more. MNE's `decim` is plain slicing with no
    anti-alias filter, so that would fold back as aliasing.

WHAT IT ACHIEVES (spectral coverage / spectral leakage, old -> new)

    sub-delta 0.5-1     40% / 50%  ->  100% / 48%
    delta     1-3       40% / 50%  ->  100% / 31%
    theta     4-7      100% / 41%  ->  100% / 33%
    alpha     8-13     100% / 44%  ->  100% / 32%
    beta      14-30    100% / 32%  ->  100% / 18%
    gamma     31-50    100% / 46%  ->  100% / 22%
    HFA       51-150   100% / 30%  ->  100% /  6%

    Storage 15.84 -> 3.47 GB per recording; ~174 GB for 50 recordings against
    ~792 GB now. Better in every band AND 4.5x smaller.

WHAT IT DOES NOT FIX

    Temporal smearing at sub-delta. n_cycles = 3 at 0.5 Hz is a 6 s wavelet,
    so 60% of a 10 s analysis window's value comes from outside it. That is
    intrinsic to the frequency, not the grid: window 0.5-1 Hz at 30 s instead.
    Delta's 40% temporal leak means delta resolves on a ~15-20 s timescale.

    Also note this extraction supersedes the original entirely - do not mix
    the two, since n_cycles differs at every frequency.
"""

import numpy as np

# ---------------------------------------------------------------------------
# GRID
# ---------------------------------------------------------------------------
F_LO, F_HI = 0.5, 151.0
N_FREQS = 100

NC_LO, NC_HI = 3.0, 15.0        # log-ramped across the range

DECIM = 6                       # 600 Hz -> 100 Hz
FS_RAW = 600.0
FS_OUT = FS_RAW / DECIM


def frequencies(n=N_FREQS, f_lo=F_LO, f_hi=F_HI):
    """Log-spaced frequency grid."""
    return np.logspace(np.log10(f_lo), np.log10(f_hi), n)


def n_cycles_for(freqs, nc_lo=NC_LO, nc_hi=NC_HI, f_lo=F_LO, f_hi=F_HI):
    """
    Log-ramped n_cycles from nc_lo at f_lo to nc_hi at f_hi.

    Log rather than linear because frequency and bandwidth both scale
    multiplicatively; a linear ramp would spend most of its range on the
    top octave.
    """
    f = np.asarray(freqs, dtype=float)
    return nc_lo * (nc_hi / nc_lo) ** (np.log(f / f_lo) / np.log(f_hi / f_lo))


def envelope_nyquist_check(freqs=None, ncs=None, fs_out=FS_OUT):
    """
    Confirm the chosen output rate is above twice the worst-case envelope
    bandwidth. Returns (required_fs, margin). Call before any long run.
    """
    f = frequencies() if freqs is None else np.asarray(freqs)
    nc = n_cycles_for(f) if ncs is None else np.asarray(ncs)
    required = float(np.max(2 * (2 * f / nc)))
    return required, fs_out / required


def describe():
    """Print the grid and safety checks, for review before a multi-hour run."""
    f = frequencies()
    nc = n_cycles_for(f)
    print(f'{len(f)} frequencies, {f[0]:.2f}-{f[-1]:.1f} Hz, '
          f'n_cycles {nc[0]:.1f}-{nc[-1]:.1f}, decim {DECIM} -> {FS_OUT:.0f} Hz\n')
    print(f"  {'Hz':>8} {'n_cycles':>9} {'BW (Hz)':>9} {'duration':>10}")
    for t in (0.5, 1, 3, 7, 13, 30, 60, 100, 151):
        i = int(np.argmin(np.abs(f - t)))
        print(f'  {f[i]:8.2f} {nc[i]:9.1f} {2*f[i]/nc[i]:8.2f} {nc[i]/f[i]:9.3f}s')

    req, margin = envelope_nyquist_check()
    print(f'\n  worst-case envelope Nyquist requirement: {req:.1f} Hz')
    print(f'  output rate {FS_OUT:.0f} Hz -> margin {margin:.2f}x '
          f'{"OK" if margin >= 1.5 else "TOO TIGHT"}')

    print('\n  frequencies per band:')
    for name, (lo, hi) in [('sub-delta', (0.5, 1)), ('delta', (1, 3)),
                           ('theta', (4, 7)), ('alpha', (8, 13)),
                           ('beta', (14, 30)), ('gamma', (31, 50)),
                           ('HFA', (51, 150))]:
        old = int(((np.arange(1, 152, 2.) >= lo) &
                   (np.arange(1, 152, 2.) <= hi)).sum())
        new = int(((f >= lo) & (f <= hi)).sum())
        print(f'    {name:10s} {old:2d} -> {new:2d}')

    ch, raw = 145, 359428
    old_b = ch * 76 * raw * 4
    new_b = ch * len(f) * len(range(0, raw, DECIM)) * 4
    print(f'\n  storage per recording: {old_b/1e9:.2f} GB -> {new_b/1e9:.2f} GB '
          f'({old_b/new_b:.1f}x smaller)')


if __name__ == '__main__':
    describe()
