#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check wavelet HDF5 files for truncated writes (TODO A11).

A killed extraction leaves a partially-written HDF5 that opens without error
and reads as ZEROS for the channels it never reached. Because the extraction
scripts use `overwrite = False`, such a file is treated as done on the next run
and silently becomes part of the dataset. This finds them.

Input:
    Any directory tree containing *.h5 with a `pow_tf_dat` dataset.

Output:
    A report. With --delete, removes files that fail, so a re-run regenerates
    them. Nothing is deleted without that flag.

How truncation is detected:
    Extraction fills the array channel by channel, so an interrupted write
    leaves a contiguous tail of all-zero channels. Sampling channels across the
    array and flagging any that are entirely zero catches this reliably. A
    genuinely dead electrode is possible but would have been dropped upstream
    by the flat-channel filter (std > 1e-6), so an all-zero channel here means
    an unfinished write.

Usage:
    python3 analysis_scripts/check_wavelet_integrity.py
    python3 analysis_scripts/check_wavelet_integrity.py --delete
"""

import os
import sys
import glob
import numpy as np
import h5py

_src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
if _src not in sys.path:
    sys.path.insert(0, _src)

try:
    from paths import DATA_ROOTS
except ImportError:
    DATA_ROOTS = ['/media/christine/Data/Movie_data',
                  '/media/christine/Samsung/Movie_data']

# Directory globs to scan, relative to each data root.
SCAN_PATTERNS = [
    'wavelet_*_all_cortContacts_tf_*/*/*.h5',
    'wavelet_continuous_z/*/*/*.h5',
    'wavelet_band_power/*/*/*/*.h5',
    'wavelet_windowed_10s/*/*/*.h5',
]

N_PROBE_CHANNELS = 12      # channels sampled per file
N_PROBE_SAMPLES = 300      # time samples per probe


def find_files():
    out = []
    for root in DATA_ROOTS:
        for pat in SCAN_PATTERNS:
            out += [p for p in glob.glob(os.path.join(root, pat))
                    if not os.path.basename(p).startswith('._')]
    return sorted(set(out))


def check(path):
    """
    Return (status, detail). status is 'ok', 'partial', or 'unreadable'.
    """
    try:
        with h5py.File(path, 'r') as h:
            key = next((k for k in ('pow_tf_dat', 'pow_tf_log_z', 'pow_dat',
                                    'pow_dat_z_windowed', 'pow_log_mean')
                        if k in h), None)
            if key is None:
                return 'ok', 'no power dataset (metadata-only file)'
            d = h[key]
            n_ch = d.shape[0]
            idx = np.unique(np.linspace(0, n_ch - 1, N_PROBE_CHANNELS).astype(int))
            empty = []
            for i in idx:
                block = d[i, ..., :N_PROBE_SAMPLES]
                if not np.any(block):
                    empty.append(int(i))
            if empty:
                return 'partial', (f'{len(empty)}/{len(idx)} sampled channels '
                                   f'all-zero (first at ch {empty[0]} of {n_ch})')
            if not np.all(np.isfinite(d[idx[0], ..., :N_PROBE_SAMPLES])):
                return 'partial', 'non-finite values in first probed channel'
            return 'ok', f'{d.shape}'
    except BlockingIOError:
        # HDF5 holds a write lock: this file is being written RIGHT NOW by an
        # active extraction. Not corruption - never delete these.
        return 'locked', 'in use by a running extraction'
    except OSError as exc:
        if 'unable to lock' in str(exc).lower():
            return 'locked', 'in use by a running extraction'
        return 'unreadable', f'{type(exc).__name__}: {exc}'
    except Exception as exc:
        return 'unreadable', f'{type(exc).__name__}: {exc}'


def main(delete=False):
    files = find_files()
    print(f'scanning {len(files)} wavelet HDF5 files\n')
    bad, locked = [], []
    for p in files:
        status, detail = check(p)
        if status == 'locked':
            locked.append(p)
            continue
        if status != 'ok':
            bad.append((p, status, detail))
            print(f'  {status.upper():10s} {os.path.getsize(p)/1e9:6.2f} GB  '
                  f'{os.path.basename(p)[:56]}')
            print(f'             {detail}')
    print()
    if locked:
        print(f'  {len(locked)} file(s) locked by a running extraction - skipped, '
              f'not evaluated:')
        for p in locked[:8]:
            print(f'      {os.path.basename(p)[:60]}')
        print('  re-run this check once the extraction finishes.\n')
    if not bad:
        print(f'  {len(files)-len(locked)} evaluated, all OK')
        return 0
    print(f'  {len(bad)} of {len(files)-len(locked)} evaluated files FAILED')
    if delete:
        for p, _, _ in bad:
            os.remove(p)
            print(f'  deleted {os.path.basename(p)[:60]}')
        print(f'\n  {len(bad)} deleted; re-run the extraction to regenerate them')
    else:
        print('\n  re-run with --delete to remove them, then re-run the extraction')
    return 1


if __name__ == '__main__':
    sys.exit(main(delete='--delete' in sys.argv))
