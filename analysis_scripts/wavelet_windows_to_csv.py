#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 3 bridge: wavelet-derived band power -> Tier 2 wide CSV (TODO B2).

Produces the same CSV format the old Hilbert/bandpass chain produced, so every
downstream Stage 4 analysis runs unchanged on wavelet-derived data.

Input HDF5 (per patient x video x band), written by bandpass_from_wavelet.py:
    {WAVELET_BAND_POWER}/{vid}/{band}/{pat}/
        *_{band}_log_band_power.h5
            pow_dat    : channels x samples   continuous log band power @ 600 Hz
            labels_ip  : channel labels
            attrs      : pat, vid, run_label, freq_band, fs
        *_{band}_robust_z_windowed_10s.h5
            pow_dat_z_windowed : channels x windows   (already robust-z scored)

Output CSV, matching lowpass_power_to_windows.py exactly:
    row 0..4 : atlas metadata, keyed by the `Atlas` column
               DK_Atlas_Region, Y7_Atlas_Region, Y17_Atlas_Region,
               AparcAseg_Atlas_Region, network
    row 5..  : one row per window, `Atlas` blank
    columns  : SubID, Atlas, then one column per electrode

Processing:
    1. Read continuous log band power.
    2. Rolling-mean window it: 10 s window, 7.5 s overlap, 2.5 s step @ 600 Hz
       (WINDOW_SAMPLES=6000, STEP_SAMPLES=1500) - the identical computation in
       lowpass_power_to_windows.rolling_average_windows.
    3. Join atlas metadata via src/channel_metadata.py, which reads the
       electrode correspondence sheets.
    4. Transpose to windows x electrodes and write CSV.

Which source to window, and why:
    SOURCE = 'log_band_power' reproduces the OLD Tier 2 files, which are
    `windowed_unnormed_*` - a rolling MEAN of log power, not z-scored. This is
    the setting to use for B3 validation against existing outputs.

    SOURCE = 'robust_z' instead reads pow_dat_z_windowed directly. That is
    robust-z scored (median/MAD) and is NOT equivalent to the old Tier 2 files.
    Do not compare it against them and expect agreement.

Deliberately NOT done here:
    No z-scoring (when SOURCE='log_band_power'), no bad-window exclusion, no
    aggregation across bands or recordings, no eye-feature merge. Those are
    Tier 3/4 concerns.

Atlas metadata comes from the correspondence sheets, NOT from the wavelet-side
*_channel_metadata.csv files - those have AparcAseg_Atlas mispopulated with a
copy of Y7_Atlas in all 76 files (see CLAUDE.md).
"""

import os
import sys
import glob
import h5py
import numpy as np
import pandas as pd

_src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
if _src not in sys.path:
    sys.path.insert(0, _src)

from paths import find                              # noqa: E402
from channel_metadata import load_channel_metadata   # noqa: E402

# =============================================================================
# PARAMETERS
# =============================================================================
SOURCE = 'log_band_power'      # 'log_band_power' (old Tier 2 parity) or 'robust_z'

WINDOW_SEC  = 10
OVERLAP_SEC = 7.5
STEP_SEC    = WINDOW_SEC - OVERLAP_SEC     # 2.5
FS_LFP      = 600                          # Hz, matches the old pipeline

WINDOW_SAMPLES = int(WINDOW_SEC * FS_LFP)  # 6000
STEP_SAMPLES   = int(STEP_SEC   * FS_LFP)  # 1500

BAND_POWER_DIR = find('wavelet_band_power')
OUT_ROOT = os.path.join(os.path.dirname(BAND_POWER_DIR or '.'),
                        f'windowed_power_{WINDOW_SEC}s_from_wavelet')

# Hand-edited knobs, as elsewhere in this repo.
vids  = ['despicable_me_english']
bands = ['gamma']
pats  = None          # None = every patient found

ATLAS_ROWS = ['DK_Atlas', 'Y7_Atlas', 'Y17_Atlas', 'AparcAseg_Atlas', 'network']


# =============================================================================
# HELPERS
# =============================================================================

def rolling_average_windows(pow_dat, window_samples, step_samples):
    """
    Mean of each electrode within each window.

    Identical to lowpass_power_to_windows.rolling_average_windows: windows that
    would run past the end of the recording are dropped rather than padded, so
    window counts match the old outputs exactly.

    pow_dat : (n_electrodes, n_samples)
    returns : (n_electrodes, n_windows), window_centers_sec
    """
    n_samples = pow_dat.shape[1]
    means, centers = [], []
    for start in range(0, n_samples - window_samples + 1, step_samples):
        end = start + window_samples
        means.append(pow_dat[:, start:end].mean(axis=1))
        centers.append((start + end) / 2.0 / FS_LFP)
    return np.column_stack(means), np.asarray(centers)


def _decode(labels):
    """HDF5 stores channel labels as bytes; the CSVs use str."""
    return [l.decode() if isinstance(l, (bytes, np.bytes_)) else str(l)
            for l in labels]


def build_tier2_frame(pat, labels, condensed):
    """
    Assemble the wide Tier 2 frame: 5 atlas metadata rows above the data rows.

    Raises if any data channel is missing from the correspondence sheet -
    silent misalignment between metadata and columns is exactly the failure this
    is meant to prevent.
    """
    meta = load_channel_metadata(pat, channels=labels)

    atlas_block = pd.DataFrame(
        [meta[col].astype(str).tolist() for col in ATLAS_ROWS],
        columns=labels)
    atlas_block.insert(0, 'Atlas',
                       [c if c == 'network' else f'{c}_Region' for c in ATLAS_ROWS])
    atlas_block.insert(0, 'SubID', pat)

    data = pd.DataFrame(condensed.T, columns=labels)   # windows x electrodes
    data.insert(0, 'Atlas', '')
    data.insert(0, 'SubID', pat)

    return pd.concat([atlas_block, data], ignore_index=True)


def process_file(h5_path, vid, band, out_root=OUT_ROOT, source=SOURCE,
                 dry_run=False):
    """Convert one HDF5 to one Tier 2 CSV. Returns the output path."""
    with h5py.File(h5_path, 'r') as h:
        pat       = h.attrs.get('pat', 'UNKNOWN')
        run_label = h.attrs.get('run_label', 'run-01')
        labels    = _decode(h['labels_ip'][:])

        if source == 'log_band_power':
            condensed, _ = rolling_average_windows(
                h['pow_dat'][:], WINDOW_SAMPLES, STEP_SAMPLES)
            normed_val = 'unnormed'
        else:
            condensed = h['pow_dat_z_windowed'][:]
            normed_val = 'robustz'

    frame = build_tier2_frame(pat, labels, condensed)

    out_dir = os.path.join(out_root, f'windowed_{normed_val}_power_log_{vid}_{band}',
                           str(pat))
    out_name = (f'{pat}_{run_label}_{vid}_{band}_power_log_'
                f'_rolling_avg_{WINDOW_SEC}s.csv')
    out_path = os.path.join(out_dir, out_name)

    if not dry_run:
        os.makedirs(out_dir, exist_ok=True)
        frame.to_csv(out_path)
    return out_path, frame


def find_inputs(vid, band, source=SOURCE, pats=None):
    suffix = ('log_band_power.h5' if source == 'log_band_power'
              else 'robust_z_windowed_10s.h5')
    hits = sorted(glob.glob(os.path.join(
        BAND_POWER_DIR, vid, band, '*', f'*{suffix}')))
    hits = [h for h in hits if not os.path.basename(h).startswith('._')]
    if pats:
        hits = [h for h in hits if os.path.basename(os.path.dirname(h)) in pats]
    return hits


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    print(f'source        : {SOURCE}')
    print(f'band power in : {BAND_POWER_DIR}')
    print(f'writing to    : {OUT_ROOT}\n')

    for vid in vids:
        for band in bands:
            inputs = find_inputs(vid, band, SOURCE, pats)
            print(f'{vid} / {band}: {len(inputs)} recordings')
            for h5_path in inputs:
                try:
                    out_path, frame = process_file(h5_path, vid, band)
                    print(f'   {frame.shape[0]-len(ATLAS_ROWS):4d} windows x '
                          f'{frame.shape[1]-2:3d} elec -> {os.path.basename(out_path)}')
                except Exception as exc:
                    print(f'   FAILED {os.path.basename(h5_path)}: '
                          f'{type(exc).__name__}: {exc}')
    print('\nDone.')
