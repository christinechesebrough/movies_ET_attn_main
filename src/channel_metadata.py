#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Canonical channel-metadata access (TODO A6).

Single place to load per-recording channel metadata and translate atlas labels,
so every preprocessing / analysis / plotting step sees the same values.

Input:
    Electrode correspondence sheets (authoritative source), one per patient:
        {CORR_DIR}/{pat}_Electrodes_Natus_TDT_correspondence[_renamed_networks].xlsx
    48 columns; the ones used here:
        label              channel name, joins to data columns
        desikan_killiany   DK atlas region
        y7_atlas           Yeo-7  as raw codes  ("7Networks_5")
        Yeo17              Yeo-17 as raw codes  ("17Networks_9")
        aparc_aseg         AparcAseg region
        network            custom network (DMN / FPCN_A / FPCN_B / OTHER / EXCLUDE)
        Good, Bad, SOZ, Spikey, Out, WMvsGM, hem, sEEG_ECoG   quality/type flags

Output:
    A DataFrame with normalized columns matching the derived
    *_channel_metadata.csv convention already used beside the wavelet outputs:
        label, DK_Atlas, Y7_Atlas, Y17_Atlas, AparcAseg_Atlas, network
    plus the quality flags, and raw codes retained as Y7_code / Y17_code.

Why this exists:
    The Yeo code -> name mapping was duplicated across 9 scripts, and three
    different Y7 vocabularies were in circulation simultaneously:
        raw codes   "7Networks_5"            (correspondence sheets)
        short names "Limbic"                 (plot_elec_anatomy_attn_effects.py)
        long names  "Limbic Network (LN)"    (channel_metadata.csv, compare_*)
    All three are preserved here as explicit maps so a caller can ask for the
    vocabulary it needs instead of hardcoding one.

This module only READS. It does not modify any existing script or data file.
No caching, no path guessing beyond the documented default directory.
"""

import os
import glob
import pandas as pd

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
try:
    from paths import CORR_SHEETS as CORR_DIR
except ImportError:                      # standalone use without src on path
    CORR_DIR = '/media/christine/Samsung/Movie_data/data/movie_elec_corr_sheets'

# ---------------------------------------------------------------------------
# LABEL VOCABULARIES
# Values copied verbatim from the existing scripts so nothing is renamed.
# ---------------------------------------------------------------------------

# Yeo-17: identical in all 8 scripts that carried a copy (verified 2026-09-09).
YEO17_CODE_TO_NAME = {
    "17Networks_1":  "Visual Central (Visual A)",
    "17Networks_2":  "Visual Peripheral (Visual B)",
    "17Networks_3":  "Somatomotor A",
    "17Networks_4":  "Somatomotor B",
    "17Networks_5":  "Dorsal Attention A",
    "17Networks_6":  "Dorsal Attention B",
    "17Networks_7":  "Salience / Ventral Attention A",
    "17Networks_8":  "Salience / Ventral Attention B",
    "17Networks_9":  "Limbic A",
    "17Networks_10": "Limbic B",
    "17Networks_11": "Control C",
    "17Networks_12": "Control A",
    "17Networks_13": "Control B",
    "17Networks_14": "Temporal Parietal",
    "17Networks_15": "Default C",
    "17Networks_16": "Default A",
    "17Networks_17": "Default B",
}

# Yeo-7 short form, as used in plot_elec_anatomy_attn_effects.py.
YEO7_CODE_TO_SHORT = {
    "7Networks_1": "Visual",
    "7Networks_2": "Somatomotor",
    "7Networks_3": "Dorsal Attention",
    "7Networks_4": "Ventral Attention",
    "7Networks_5": "Limbic",
    "7Networks_6": "Frontoparietal",
    "7Networks_7": "Default",
}

# Yeo-7 long form: the vocabulary in the derived *_channel_metadata.csv files
# and in compare_attn_states_*. This is the default here, for consistency with
# existing Tier 1/2 outputs.
YEO7_CODE_TO_LONG = {
    "7Networks_1": "Visual Network (VN)",
    "7Networks_2": "Somatomotor Network (SMN)",
    "7Networks_3": "Dorsal Attention Network (DAN)",
    "7Networks_4": "Ventral Attention Network (VAN)",
    "7Networks_5": "Limbic Network (LN)",
    "7Networks_6": "Frontoparietal Network (FPN)",
    "7Networks_7": "Default Mode Network (DMN)",
}

# Values that are not network assignments and should pass through untranslated.
PASSTHROUGH_LABELS = {"Out", "FreeSurfer_Defined_Medial_Wall", "unknown", "nan"}


# ---------------------------------------------------------------------------
# LOADING
# ---------------------------------------------------------------------------

def find_correspondence_sheet(pat, corr_dir=CORR_DIR, prefer_renamed=False):
    """
    Locate a patient's correspondence sheet.

    Two variants exist per patient: a base sheet and a '_renamed_networks'
    sheet. For NS127_02 their 'network' columns were byte-identical, but that
    has not been checked for every patient, so the choice is explicit rather
    than assumed. Default is the base sheet.

    Returns a path, or None if no sheet exists for this patient.
    """
    hits = [p for p in glob.glob(os.path.join(corr_dir, f'{pat}_Electrodes*.xlsx'))
            if not os.path.basename(p).startswith('._')]
    if not hits:
        return None
    renamed = [p for p in hits if 'renamed_networks' in os.path.basename(p)]
    base = [p for p in hits if 'renamed_networks' not in os.path.basename(p)]
    if prefer_renamed and renamed:
        return renamed[0]
    return (base or renamed)[0]


def translate_y7(code, vocabulary='long'):
    """Translate a raw Yeo-7 code. Unknown/non-network values pass through."""
    mapping = YEO7_CODE_TO_LONG if vocabulary == 'long' else YEO7_CODE_TO_SHORT
    if vocabulary == 'raw':
        return code
    return mapping.get(str(code), code)


def translate_y17(code):
    """Translate a raw Yeo-17 code. Unknown/non-network values pass through."""
    return YEO17_CODE_TO_NAME.get(str(code), code)


def load_channel_metadata(pat, corr_dir=CORR_DIR, y7_vocabulary='long',
                          prefer_renamed=False, channels=None):
    """
    Load normalized channel metadata for one patient.

    pat            patient id as used in directory names, e.g. 'NS127_02'
    y7_vocabulary  'long' (default, matches existing CSVs), 'short', or 'raw'
    channels       optional list of channel labels to restrict/order the result
                   to; use the data's own column order to guarantee alignment

    Returns a DataFrame with columns:
        label, DK_Atlas, Y7_Atlas, Y17_Atlas, AparcAseg_Atlas, network,
        Y7_code, Y17_code, Good, Bad, SOZ, Spikey, Out, WMvsGM, hem, sEEG_ECoG

    Raises FileNotFoundError if the patient has no correspondence sheet, and
    ValueError if `channels` contains labels absent from the sheet - silent
    misalignment between metadata rows and data columns is the failure mode
    this module exists to prevent.
    """
    path = find_correspondence_sheet(pat, corr_dir, prefer_renamed)
    if path is None:
        raise FileNotFoundError(
            f'No correspondence sheet for {pat!r} in {corr_dir}')

    raw = pd.read_excel(path)
    raw['label'] = raw['label'].astype(str)

    out = pd.DataFrame({
        'label':           raw['label'],
        'DK_Atlas':        raw.get('desikan_killiany'),
        'Y7_code':         raw.get('y7_atlas'),
        'Y17_code':        raw.get('Yeo17'),
        'AparcAseg_Atlas': raw.get('aparc_aseg'),
        'network':         raw.get('network'),
    })
    out['Y7_Atlas'] = out['Y7_code'].map(lambda c: translate_y7(c, y7_vocabulary))
    out['Y17_Atlas'] = out['Y17_code'].map(translate_y17)

    for flag in ['Good', 'Bad', 'SOZ', 'Spikey', 'Out', 'WMvsGM', 'hem', 'sEEG_ECoG']:
        if flag in raw.columns:
            out[flag] = raw[flag]

    out = out[['label', 'DK_Atlas', 'Y7_Atlas', 'Y17_Atlas', 'AparcAseg_Atlas',
               'network', 'Y7_code', 'Y17_code']
              + [c for c in out.columns if c not in
                 {'label', 'DK_Atlas', 'Y7_Atlas', 'Y17_Atlas',
                  'AparcAseg_Atlas', 'network', 'Y7_code', 'Y17_code'}]]

    if channels is not None:
        channels = [str(c) for c in channels]
        missing = [c for c in channels if c not in set(out['label'])]
        if missing:
            raise ValueError(
                f'{pat}: {len(missing)} channel(s) absent from correspondence '
                f'sheet: {missing[:10]}{"..." if len(missing) > 10 else ""}')
        out = out.set_index('label').loc[channels].reset_index()

    return out


def as_atlas_rows(meta, atlas_rows=('DK_Atlas', 'Y7_Atlas', 'Y17_Atlas',
                                    'AparcAseg_Atlas', 'network')):
    """
    Reshape metadata into the wide 'atlas rows' block that Tier 1/2 CSVs carry
    above their data rows: one row per atlas, one column per channel.

    Row labels use the Tier 1/2 convention ('DK_Atlas' -> 'DK_Atlas_Region');
    'network' is emitted as-is, matching the existing Tier 2 files.
    """
    block = meta.set_index('label')[list(atlas_rows)].T
    block.index = [r if r == 'network' else f'{r}_Region' for r in block.index]
    return block
