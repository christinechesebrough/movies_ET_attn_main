"""
Per-recording standardisation of windowed eye features.

Shared by compute_norm_eye_features_4Jan26.py (NORM_SCHEME = 'robust') and the
exploration track scripts (attn_explore_14Sep26_*), so the robust scale is
defined in exactly one place.

robust_scale(x)            -> (median, scale)   scale = 1.4826 * MAD, SD fallback
standardise_recording(df, cols, scheme)  -> copy with cols standardised

No windowing, no across-movie (stage 2) scaling, no PCA is done here.
"""
import numpy as np
import pandas as pd

MAD_TO_SD = 1.4826   # 1 / Phi^-1(0.75): makes the MAD consistent with the SD under normality


def robust_scale(x):
    """Median and MAD-based scale of a Series (NaN ignored). Falls back to the
    SD when the MAD is 0 (more than half the windows identical, e.g. blink
    rate 0). Returns (median, scale, used_fallback)."""
    x = pd.Series(x, dtype=float)
    med = x.median(skipna=True)
    scale = MAD_TO_SD * (x - med).abs().median(skipna=True)
    fallback = (not np.isfinite(scale)) or scale == 0
    if fallback:
        scale = x.std(skipna=True)
    return med, scale, fallback


def standardise_recording(df, cols, scheme='robust', verbose_name=None):
    """
    Standardise `cols` of ONE recording's window table.

    scheme='robust'  (x - median) / (1.4826 * MAD); missing windows filled with 0
                     AFTER standardisation (= at the recording median).
    scheme='legacy'  missing filled with raw 0 BEFORE (x - mean) / SD, as in the
                     original chain. Kept for like-for-like comparisons.
    """
    out = df.copy()
    for c in cols:
        x = out[c].astype(float)
        if scheme == 'robust':
            med, scale, fb = robust_scale(x)
            if fb and verbose_name:
                print(f'  {verbose_name} {c}: MAD is 0, falling back to SD ({scale:.4g})')
            out[c] = ((x - med) / scale).fillna(0)
        elif scheme == 'legacy':
            x = x.fillna(0)
            out[c] = (x - x.mean()) / x.std()
        else:
            raise ValueError(scheme)
    return out
