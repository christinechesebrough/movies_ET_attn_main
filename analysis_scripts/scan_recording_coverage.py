#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a per-recording coverage table (TODO A5).

Answers "which recordings exist, and how far has each one got through each
pipeline?" by scanning the data tree, so effective N stops being implicit in
hand-edited patient lists.

Input:
    Directory structure under DATA_ROOT. Nothing is read into memory beyond
    file listings and the small missing_data_*.csv eye-quality tables.

Output:
    recording_coverage.csv  — one row per (patient, session, video, run):

        patient, session, video, run
        has_preprocessed        preprocessed / referenced .fif present
        has_bad_channels        bad-channel .txt present
        has_bad_windows         manual window-QC .csv present
        has_wavelet             wavelet output (.npz or .h5) present
        has_power_tier1         full-resolution power CSV present (per band)
        has_power_tier2         windowed power CSV present (per band)
        has_fooof               rolling-FOOOF output present
        eye_missing_frac        worst pre-interpolation missing fraction
        eye_missing_post_interp worst post-interpolation missing fraction
        include                 BLANK - to be filled in by hand
        exclude_reason          BLANK - to be filled in by hand
        exclude_modality        BLANK - ieeg / eye / both

    The last three columns are deliberately empty: presence of files says what
    was processed, never whether it *should* be. That judgement is Christine's,
    and recording it is the entire point of the table.

Processing:
    1. Enumerate recordings from preprocessed .fif filenames.
    2. Normalize video aliases (dme / DespMeEng -> despicable_me_english).
    3. Probe each pipeline stage directory for matching outputs.
    4. Join eye-quality metrics from movies_prep_standard/missing_data_*.csv.
    5. Write the table; never overwrite an existing annotated copy in place.

This script only READS the data tree and writes one CSV. It modifies no
existing data file and no other script.
"""

import os
import re
import glob
import pandas as pd

# =============================================================================
# PARAMETERS
# =============================================================================
DATA_ROOT   = '/media/christine/Samsung/Movie_data'
OUT_PATH    = os.path.join(DATA_ROOT, 'recording_coverage.csv')

# Videos of interest, with the filename aliases each one appears under.
VIDEO_ALIASES = {
    'despicable_me_english':   ['despicable_me_english', 'dme', 'DespMeEng'],
    'despicable_me_hungarian': ['despicable_me_hungarian'],
    'inscapes':                ['inscapes'],
}

PREP_DIR      = os.path.join(DATA_ROOT, 'movies_prep_standard')
BAD_WIN_DIR   = os.path.join(DATA_ROOT, 'movies_bad_windows')
# Wavelet outputs live in two places: under wavelet_power_*/ and, for the
# 21Apr26 runs, as top-level wavelet_{vid}_* directories.
WAVELET_GLOBS = [
    os.path.join(DATA_ROOT, 'wavelet_power_*', 'wavelet_{vid}_*', '{pat}', '*'),
    os.path.join(DATA_ROOT, 'wavelet_{vid}_*', '{pat}', '*'),
]
TIER1_GLOB    = os.path.join(DATA_ROOT, 'full_raw_log_power_*', 'power_log_*_{vid}_*', '{pat}', '*.csv')
TIER2_GLOB    = os.path.join(DATA_ROOT, 'windowed_power_*', '*_{vid}_*', '{pat}', '*.csv')
FOOOF_GLOB    = os.path.join(DATA_ROOT, 'rolling_fooof_*{vid}*', '{pat}', '*.csv')

# .fif suffixes that indicate a usable preprocessed/referenced recording
PREPROC_MARKERS = ('referenced_avg', 'preprocessed', 'prep_ref_avg', 'referenced_wm')


# =============================================================================
# HELPERS
# =============================================================================

def _real(paths):
    """Drop macOS ._ resource-fork files, which otherwise double every count."""
    return [p for p in paths if not os.path.basename(p).startswith('._')]


def canonical_video(token):
    """Map a filename task token to a canonical video name, or None."""
    for canon, aliases in VIDEO_ALIASES.items():
        if token in aliases:
            return canon
    return None


def parse_fif(path):
    """
    Pull (session, video, run) out of a .fif filename.

    Handles both conventions seen in the tree:
        sub-LH012_ses-01_task-despicable_me_hungarian_run-01_ieeg_preprocessed.fif
        sub-NS144_ses-implant02_task-DespMeEng_ieeg_prep_ref_avg.fif   (no run)
    Returns None if the file is not one of the videos of interest.
    """
    name = os.path.basename(path)
    m_task = re.search(r'task-([A-Za-z_0-9]+?)(?:_run-|_ieeg)', name)
    if not m_task:
        return None
    vid = canonical_video(m_task.group(1))
    if vid is None:
        return None
    m_ses = re.search(r'ses-([A-Za-z0-9]+)', name)
    m_run = re.search(r'run-([0-9]+)', name)
    return {
        'session': m_ses.group(1) if m_ses else '',
        'video':   vid,
        # recordings without an explicit run token are treated as run 01
        'run':     f"{int(m_run.group(1)):02d}" if m_run else '01',
    }


def has_any(patterns, **kw):
    """True if any of the given glob patterns matches a real file."""
    if isinstance(patterns, str):
        patterns = [patterns]
    return any(_real(glob.glob(p.format(**kw))) for p in patterns)


def load_eye_quality():
    """
    Read the existing missing_data_{video}.csv eye-quality tables.

    These already log eye-tracking quality per recording (missing-sample
    fractions before and after interpolation) but carry no include/exclude
    decision. Returns {(patient, video): (worst_pre, worst_post)}.
    """
    out = {}
    for vid in VIDEO_ALIASES:
        path = os.path.join(PREP_DIR, f'missing_data_{vid}.csv')
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        pre_cols = [c for c in df.columns if c.endswith('_missing')]
        post_cols = [c for c in df.columns if c.endswith('_missing_post_interp')]
        for _, row in df.iterrows():
            pat = str(row['Patient'])
            pre = max([row[c] for c in pre_cols if pd.notna(row[c])], default=None)
            post = max([row[c] for c in post_cols if pd.notna(row[c])], default=None)
            out[(pat, vid)] = (pre, post)
    return out


# =============================================================================
# SCAN
# =============================================================================

def scan():
    eye = load_eye_quality()
    records = {}

    for pat_dir in sorted(_real(glob.glob(os.path.join(PREP_DIR, '*')))):
        if not os.path.isdir(pat_dir):
            continue
        pat = os.path.basename(pat_dir)

        for fif in _real(glob.glob(os.path.join(pat_dir, '**', '*.fif'), recursive=True)):
            info = parse_fif(fif)
            if info is None:
                continue
            key = (pat, info['session'], info['video'], info['run'])
            rec = records.setdefault(key, {
                'patient': pat, 'session': info['session'],
                'video': info['video'], 'run': info['run'],
                'has_preprocessed': False,
            })
            if any(mark in os.path.basename(fif) for mark in PREPROC_MARKERS):
                rec['has_preprocessed'] = True

    for (pat, ses, vid, run), rec in records.items():
        np_dir = os.path.join(PREP_DIR, pat, 'Neural_prep')
        rec['has_bad_channels'] = len(_real(glob.glob(
            os.path.join(np_dir, '*bad_channels*.txt')))) > 0
        rec['has_bad_windows'] = len(_real(glob.glob(
            os.path.join(BAD_WIN_DIR, pat, f'*{vid}*qc*.csv')))) > 0
        rec['has_wavelet']     = has_any(WAVELET_GLOBS, vid=vid, pat=pat)
        rec['has_power_tier1'] = has_any(TIER1_GLOB,   vid=vid, pat=pat)
        rec['has_power_tier2'] = has_any(TIER2_GLOB,   vid=vid, pat=pat)
        rec['has_fooof']       = has_any(FOOOF_GLOB,   vid=vid, pat=pat)

        pre, post = eye.get((pat, vid), (None, None))
        # correspondence sheets and eye tables sometimes drop the _02 suffix
        if pre is None:
            pre, post = eye.get((pat.split('_')[0], vid), (None, None))
        rec['eye_missing_frac'] = pre
        rec['eye_missing_post_interp'] = post

        rec['include'] = ''
        rec['exclude_reason'] = ''
        rec['exclude_modality'] = ''

    cols = ['patient', 'session', 'video', 'run',
            'has_preprocessed', 'has_bad_channels', 'has_bad_windows',
            'has_wavelet', 'has_power_tier1', 'has_power_tier2', 'has_fooof',
            'eye_missing_frac', 'eye_missing_post_interp',
            'include', 'exclude_reason', 'exclude_modality']
    df = pd.DataFrame(list(records.values()))
    return df.reindex(columns=cols).sort_values(['video', 'patient', 'session', 'run'])


if __name__ == '__main__':
    table = scan()

    # Never clobber hand-annotated work: write beside it instead.
    out = OUT_PATH
    if os.path.exists(out):
        existing = pd.read_csv(out)
        annotated = (existing.get('include', pd.Series(dtype=str))
                     .astype(str).str.strip()
                     .replace({'nan': '', 'None': ''}))
        if (annotated != '').any():
            out = OUT_PATH.replace('.csv', '_rescan.csv')
            print(f'Existing table has annotations; writing rescan to {out}')

    table.to_csv(out, index=False)
    print(f'Wrote {len(table)} recordings to {out}')
    print()
    print(table.groupby('video')[
        ['has_preprocessed', 'has_wavelet', 'has_power_tier1',
         'has_power_tier2', 'has_fooof']].sum().to_string())
