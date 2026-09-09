#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Frequency grid and n_cycles profile for the low-frequency re-extraction (A11).

Defines the 0.5-30 Hz log-spaced grid that replaces the linear 2 Hz / fixed
n_cycles=5 grid below 30 Hz, and blends seamlessly into the existing >30 Hz
extraction.

Why the existing grid fails at low frequencies:
    Morlet bandwidth scales as 2f/n_cycles, so the non-redundant frequency
    spacing also scales with f. A fixed 2 Hz step is therefore undersampled at
    the bottom and oversampled at the top. Concretely, with n_cycles=5:

        1 Hz wavelet covers 0.8-1.2 Hz
                            <-- 1.2 Hz GAP, nothing measures this
        3 Hz wavelet covers 2.4-3.6 Hz

    so only ~40% of the 1-3 Hz delta band is measured at all, and sub-delta
    (0.5-1 Hz) has a single wavelet sitting at its edge.

Design:
    freqs      62 log-spaced points, 0.5-30 Hz
               spacing at 30 Hz is 1.95 Hz, matching the old 2 Hz linear step,
               so grid DENSITY is continuous across the join
    n_cycles   3.0 at 0.5 Hz   (the floor - below ~3 cycles a Morlet is not
                                frequency-selective)
               rising to 10.0 by 8 Hz
               held at 10.0 across alpha (8-13 Hz), where leakage matters most
               falling to 5.0 by 30 Hz, matching the old fixed value, so
               BANDWIDTH is continuous across the join (3% residual)

What it achieves (spectral coverage / spectral leakage):
    sub-delta 0.5-1   40% / 50%  ->  100% / 45%
    delta     1-3     40% / 50%  ->  100% / 22%
    theta     4-7    100% / 41%  ->  100% / 29%
    alpha     8-13   100% / 44%  ->  100% / 22%
    beta      14-30  100% / 32%  ->  100% / 18%

What it does NOT fix:
    Temporal smearing at sub-delta. n_cycles=3 at 0.5 Hz is a 6 s wavelet, so
    60% of a 10 s window's value comes from outside it. That is intrinsic to
    the frequency, not the grid - address it by windowing 0.5-1 Hz at 30 s
    rather than 10 s. Delta's 40% temporal leak means delta resolves on a
    ~15-20 s timescale; report that rather than dropping the band.

Splice rule:
    use THIS extraction for <= 30 Hz, the existing one for > 30 Hz.
    Do not mix - beta is re-derived here with finer n_cycles.
"""

import numpy as np

# ---------------------------------------------------------------------------
# GRID
# ---------------------------------------------------------------------------
F_LO, F_HI = 0.5, 30.0
N_FREQS = 62                    # gives 1.95 Hz spacing at 30 Hz

NC_FLOOR = 3.0                  # at F_LO; below ~3 cycles a Morlet is not
                                # frequency-selective and leaks toward DC
NC_PEAK = 10.0                  # held across alpha
NC_JOIN = 5.0                   # at F_HI, matching the existing extraction
F_PEAK_LO, F_PEAK_HI = 8.0, 13.0


def frequencies(n=N_FREQS, f_lo=F_LO, f_hi=F_HI):
    """Log-spaced frequency grid."""
    return np.logspace(np.log10(f_lo), np.log10(f_hi), n)


def n_cycles_for(freqs):
    """
    n_cycles profile: NC_FLOOR -> NC_PEAK -> NC_JOIN, log-interpolated.

    Log interpolation (not linear) because both frequency and bandwidth scale
    multiplicatively; a linear ramp would spend most of its range on the top
    octave.
    """
    f = np.asarray(freqs, dtype=float)
    nc = np.empty_like(f)

    rise = f < F_PEAK_LO
    nc[rise] = NC_FLOOR * (NC_PEAK / NC_FLOOR) ** (
        np.log(f[rise] / F_LO) / np.log(F_PEAK_LO / F_LO))

    hold = (f >= F_PEAK_LO) & (f <= F_PEAK_HI)
    nc[hold] = NC_PEAK

    fall = f > F_PEAK_HI
    nc[fall] = NC_PEAK * (NC_JOIN / NC_PEAK) ** (
        np.log(f[fall] / F_PEAK_HI) / np.log(F_HI / F_PEAK_HI))

    return nc


def describe(freqs=None, ncs=None):
    """Print the grid, for sanity-checking before a multi-hour run."""
    f = frequencies() if freqs is None else np.asarray(freqs)
    nc = n_cycles_for(f) if ncs is None else np.asarray(ncs)
    print(f'{len(f)} frequencies, {f[0]:.2f}-{f[-1]:.1f} Hz\n')
    print(f"  {'Hz':>7} {'n_cycles':>9} {'BW (Hz)':>9} {'duration':>10}")
    for t in (0.5, 0.8, 1.5, 3, 5, 8, 10, 13, 20, 30):
        i = int(np.argmin(np.abs(f - t)))
        print(f'  {f[i]:7.2f} {nc[i]:9.1f} {2*f[i]/nc[i]:8.2f} {nc[i]/f[i]:9.2f}s')
    print(f'\n  spacing at {f[0]:.2f} Hz: {f[1]-f[0]:.3f} Hz')
    print(f'  spacing at {f[-1]:.1f} Hz: {f[-1]-f[-2]:.2f} Hz  '
          f'(old grid: 2.00 Hz linear)')
    print(f'  BW at {f[-1]:.1f} Hz: {2*f[-1]/nc[-1]:.2f} Hz  '
          f'(old at 31 Hz: {2*31/5:.2f} Hz)')
    print('\n  frequencies per band:')
    for name, (lo, hi) in [('sub-delta', (0.5, 1)), ('delta', (1, 3)),
                           ('theta', (4, 7)), ('alpha', (8, 13)),
                           ('beta', (14, 30))]:
        print(f'    {name:10s} {int(((f >= lo) & (f <= hi)).sum()):2d}')


if __name__ == '__main__':
    describe()
