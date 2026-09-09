#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Centralized path management (TODO E2).

One place that knows where data lives, so scripts stop hardcoding absolute
paths. Resolves at import time by probing candidate locations, so the same
script runs unchanged as data moves between drives or machines.

Why this is not a single prefix substitution:
    On the Mac everything sat under /Volumes/Samsung. On this PC the data is
    split across two physical drives, and split PER RESOURCE - not per tree.
    There are Movie_data directories on BOTH drives holding different things:

        /media/christine/Samsung/Movie_data   1.9T, 95% full, 107G free
            movies_prep_standard (CURRENT: 49 patients, newest 2026-09-08)
            full_raw_log_power_*, windowed_power_*, rolling_fooof_*
            shared_PC_features_*, movies_bad_windows, data/

        /media/christine/Data/Movie_data      11T, 6.6T free
            wavelet_*_26Aug26/       Stage 1 raw TF HDF5  (750 G)
            wavelet_band_power/      Stage 2               (38 G)
            wavelet_continuous_z/    Stage 2               (612 G)
            movies_prep_standard     STALE 2024 copy - 25 patients, do not use

        /media/christine/Data/anatomy         FreeSurfer tree

    New wavelet output goes to the Data drive because individual files are
    9-27 GB and the full set is ~1.4 TB - it cannot fit on Samsung.

    So rewriting '/Volumes/Samsung' -> '/media/christine/Samsung' looks correct
    and silently breaks both the anatomy paths and the wavelet paths. Resources
    are resolved individually, by searching the roots below in order.

Usage:
    from paths import MOVIE_DATA, ANATOMY, CORR_SHEETS, p

    prep = p(MOVIE_DATA, 'movies_prep_standard')
    sheet_dir = CORR_SHEETS

Overriding:
    Set an environment variable to force a location, e.g.

        export MOVIES_DATA_ROOT=/media/christine/Data/Movie_data

    Or, in Spyder, set the module attribute before deriving paths:

        import paths; paths.MOVIE_DATA = '/somewhere/else'

Checking what resolved:
    python3 src/paths.py        # prints the resolved table and flags problems

Note (2026-09-09): both drives carry a Movie_data. The Samsung copy is current
(49 patients, newest file 2026-09-08); the Data copy is a 2024 archive
(25 patients, newest 2024-06-27). Resolution order prefers whichever exists
first in CANDIDATES below, so if the pipeline data is later moved to the Data
drive wholesale, reorder that list rather than editing any script.
"""

import os

# ---------------------------------------------------------------------------
# CANDIDATE LOCATIONS, most-preferred first.
# Add to these rather than editing scripts when data moves.
# ---------------------------------------------------------------------------
# Roots searched, in order, when locating a named resource.
DATA_ROOTS = [r for r in [
    os.environ.get('MOVIES_DATA_ROOT'),
    '/media/christine/Samsung/Movie_data',   # legacy + current power pipeline
    '/media/christine/Data/Movie_data',      # new wavelet output (large files)
    '/Volumes/Samsung/Movie_data',           # Mac
] if r and os.path.exists(r)]

# Resources known to live on the big drive regardless of search order.
BIG_DATA_ROOT = next((r for r in ['/media/christine/Data/Movie_data']
                      if os.path.exists(r)), None)

CANDIDATES = {
    'MOVIE_DATA': [
        os.environ.get('MOVIES_DATA_ROOT'),
        '/media/christine/Samsung/Movie_data',   # current pipeline data
        '/media/christine/Data/Movie_data',
        '/Volumes/Samsung/Movie_data',           # Mac
    ],
    'ANATOMY': [
        os.environ.get('MOVIES_ANATOMY_ROOT'),
        '/media/christine/Data/anatomy',         # FreeSurfer tree, on the PC
        '/media/christine/Samsung/anatomy',
        '/Volumes/Samsung/anatomy',              # Mac
    ],
    'SCRIPTS': [
        os.environ.get('MOVIES_SCRIPTS_ROOT'),
        '/media/christine/Samsung/scripts/movies_ET_attn_main',
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ],
}


def _resolve(key):
    """First candidate that exists on disk, else None."""
    for cand in CANDIDATES[key]:
        if cand and os.path.exists(cand):
            return cand
    return None


MOVIE_DATA = _resolve('MOVIE_DATA')
ANATOMY    = _resolve('ANATOMY')
SCRIPTS    = _resolve('SCRIPTS')


def p(*parts):
    """
    Join path parts, skipping None.

    Raises if the root is unresolved, rather than silently building a path
    starting with 'None/' that fails later and further away.
    """
    if parts and parts[0] is None:
        raise RuntimeError(
            'Unresolved path root. Run `python3 src/paths.py` to see what '
            'resolved, or set MOVIES_DATA_ROOT / MOVIES_ANATOMY_ROOT.')
    return os.path.join(*[str(x) for x in parts])


def find(*relative, roots=None, required=False):
    """
    Locate a resource by searching DATA_ROOTS in order.

    Use this instead of joining onto MOVIE_DATA when a resource might live on
    either drive - which, after the wavelet outputs moved, is most of them.

        find('wavelet_continuous_z')   -> the Data-drive copy
        find('windowed_power_10s')     -> the Samsung copy

    Returns the first existing path, or None. Pass required=True to raise
    instead of returning None, so a missing input fails at the point of lookup
    rather than deep inside a loop.
    """
    for root in (roots or DATA_ROOTS):
        cand = os.path.join(root, *[str(x) for x in relative])
        if os.path.exists(cand):
            return cand
    if required:
        raise FileNotFoundError(
            f'{os.path.join(*[str(x) for x in relative])!r} not found under any '
            f'of: {DATA_ROOTS}')
    return None


# ---------------------------------------------------------------------------
# NAMED RESOURCES
# Derived once here so a move needs one edit, not a search across 55 scripts.
# ---------------------------------------------------------------------------
PREP_STANDARD   = p(MOVIE_DATA, 'movies_prep_standard') if MOVIE_DATA else None
BAD_WINDOWS     = p(MOVIE_DATA, 'movies_bad_windows') if MOVIE_DATA else None
DATA_DIR        = p(MOVIE_DATA, 'data') if MOVIE_DATA else None
CORR_SHEETS     = p(MOVIE_DATA, 'data', 'movie_elec_corr_sheets') if MOVIE_DATA else None
WAVELET_10S     = p(MOVIE_DATA, 'wavelet_power_10s') if MOVIE_DATA else None
WINDOWED_10S    = p(MOVIE_DATA, 'windowed_power_10s') if MOVIE_DATA else None
COVERAGE_TABLE  = p(MOVIE_DATA, 'recording_coverage.csv') if MOVIE_DATA else None

SHARED_CORRESPONDENCE = p(ANATOMY, 'shared_correspondence') if ANATOMY else None
SUBS_MASTER = (p(ANATOMY, 'shared_correspondence', 'movie_subs_master_updated.csv')
               if ANATOMY else None)

# --- new wavelet pipeline: lives on the Data drive (~1.4 TB) ---------------
WAVELET_BAND_POWER   = find('wavelet_band_power')
WAVELET_CONTINUOUS_Z = find('wavelet_continuous_z')

def wavelet_raw_tf(vid, roots=None, prefer_hdf5=True):
    """
    Stage 1 raw TF directory for one video.

    Directory names are date-stamped ("..._21Apr26", "..._26Aug26"), and those
    stamps do not sort chronologically as strings, so candidates are gathered
    from EVERY root and compared by modification time. Searching roots in order
    and stopping at the first hit would return the superseded April .npz
    directories on Samsung rather than the current August HDF5 ones on Data.

    prefer_hdf5: if any candidate contains .h5 files, restrict to those. The
    npz-era directories are the pre-migration format and are not interchangeable.
    """
    import glob as _glob
    hits = []
    for root in (roots or DATA_ROOTS):
        hits.extend(_glob.glob(
            os.path.join(root, f'wavelet_{vid}_all_cortContacts_tf_*')))
    hits = [h for h in hits if os.path.isdir(h)]
    if not hits:
        return None
    if prefer_hdf5:
        with_h5 = [h for h in hits
                   if _glob.glob(os.path.join(h, '*', '*.h5'))]
        if with_h5:
            hits = with_h5
    return max(hits, key=os.path.getmtime)

# Date-stamped roots, globbed rather than named. Pattern lives in one place.
TIER1_POWER_GLOB = p(MOVIE_DATA, 'full_raw_log_power_*') if MOVIE_DATA else None
TIER2_POWER_GLOB = p(MOVIE_DATA, 'windowed_power_*') if MOVIE_DATA else None
FOOOF_GLOB       = p(MOVIE_DATA, 'rolling_fooof_*') if MOVIE_DATA else None


def report():
    """Print what resolved where, and flag anything missing or ambiguous."""
    rows = [
        ('MOVIE_DATA', MOVIE_DATA), ('ANATOMY', ANATOMY), ('SCRIPTS', SCRIPTS),
        ('PREP_STANDARD', PREP_STANDARD), ('CORR_SHEETS', CORR_SHEETS),
        ('SUBS_MASTER', SUBS_MASTER), ('WAVELET_10S', WAVELET_10S),
        ('WINDOWED_10S', WINDOWED_10S), ('BAD_WINDOWS', BAD_WINDOWS),
        ('WAVELET_BAND_POWER', WAVELET_BAND_POWER),
        ('WAVELET_CONTINUOUS_Z', WAVELET_CONTINUOUS_Z),
        ('wavelet_raw_tf(english)', wavelet_raw_tf('despicable_me_english')),
        ('wavelet_raw_tf(hungarian)', wavelet_raw_tf('despicable_me_hungarian')),
        ('wavelet_raw_tf(inscapes)', wavelet_raw_tf('inscapes')),
    ]
    width = max(len(n) for n, _ in rows)
    print('Resolved paths')
    print('-' * (width + 60))
    for name, val in rows:
        mark = 'OK ' if val and os.path.exists(val) else 'MISSING'
        print(f'  {name:<{width}}  {mark}  {val}')

    print()
    print(f'DATA_ROOTS searched in order:')
    for r in DATA_ROOTS:
        print(f'    {r}')
    print()
    for key in CANDIDATES:
        seen, found = set(), []
        for c in CANDIDATES[key]:
            if c and os.path.exists(c):
                real = os.path.realpath(c)
                if real not in seen:
                    seen.add(real)
                    found.append(c)
        if len(found) > 1:
            print(f'NOTE: {key} has {len(found)} candidates present; '
                  f'using the first:')
            for c in found:
                print(f'        {"->" if c == _resolve(key) else "  "} {c}')


if __name__ == '__main__':
    report()
