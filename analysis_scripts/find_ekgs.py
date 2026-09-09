#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 23 01:26:37 2026

@author: christinechesebrough
"""

import tdt
import numpy as np
from scipy.signal import welch
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch

blk = tdt.read_block("/Volumes/Samsung/AV40_data/rawdata_for_conversion/NS219/Neural/B3_NS219_rest", t1=0, t2=30)  # short read


print("Streams:", list(getattr(blk, "streams", {}).keys()))
print("Epocs:", list(getattr(blk, "epocs", {}).keys()))


def ekg_score(x, fs):
    # x is 1D (samples,)
    f, Pxx = welch(x, fs=fs, nperseg=min(len(x), int(fs * 4)))
    low = Pxx[(f >= 0.5) & (f <= 5)].mean()
    hi  = Pxx[(f >= 20) & (f <= 80)].mean()
    return float(low / (hi + 1e-12))

# rows = []

# for sname, st in blk.streams.items():
#     data = np.asarray(st.data)
#     fs = float(st.fs)
#     # ensure (ch, time)
#     if data.ndim == 1:
#         data = data[None, :]
#     for ch in range(data.shape[0]):
#         x = data[ch]
#         rows.append({
#             "stream": sname,
#             "ch": ch,
#             "fs": fs,
#             "score": ekg_score(x, fs),
#             "std": float(np.std(x)),
#         })
        
rows = []

for sname, st in blk.streams.items():

    if not hasattr(st, "data"):
        print(f"Skipping {sname}: no .data (has {list(st.__dict__.keys())})")
        continue

    data = np.asarray(st.data)
    fs = float(st.fs)

    # ensure (ch, time)
    if data.ndim == 1:
        data = data[None, :]

    for ch in range(data.shape[0]):
        x = data[ch]
        rows.append({
            "stream": sname,
            "ch": ch,
            "fs": fs,
            "score": ekg_score(x, fs),
            "std": float(np.std(x)),
        })


df = pd.DataFrame(rows).sort_values("score", ascending=False)
print(df.head(40))

top = df.head(10)

for _, r in top.iterrows():
    sname, ch, fs = r["stream"], int(r["ch"]), float(r["fs"])
    x = np.asarray(blk.streams[sname].data)[ch]

    # time plot (first 10 seconds)
    n = int(fs * 10)
    t = np.arange(min(len(x), n)) / fs
    plt.figure()
    plt.plot(t, x[:len(t)])
    plt.title(f"{sname} ch{ch}  score={r['score']:.2f}")
    plt.xlabel("Time (s)")
    plt.show()

    # PSD
    f, Pxx = welch(x, fs=fs, nperseg=min(len(x), int(fs * 4)))
    plt.figure()
    plt.semilogy(f, Pxx)
    plt.xlim(0, 80)
    plt.title(f"PSD: {sname} ch{ch}")
    plt.xlabel("Hz")
    plt.show()


#%%
# ----------------------------
# Config: channels to plot
# ----------------------------
# Config: store + channels
# ----------------------------
sname = "EEG2"                 # the store/stream to pull from
channels_to_plot = range(70,80) # or e.g. [0, 2, 5, 7]

# Pull data once
stream = blk.streams[sname]
data = np.asarray(stream.data)
fs = float(stream.fs) if hasattr(stream, "fs") else float(stream.fs[0])  # defensive

for ch in channels_to_plot:
    x = data[ch]

    # ---- Time plot (first 10 seconds)
    n = int(fs * 10)
    t = np.arange(min(len(x), n)) / fs

    plt.figure()
    plt.plot(t, x[:len(t)])
    plt.title(f"{sname} ch{ch}")
    plt.xlabel("Time (s)")
    plt.show()

    # ---- PSD
    f, Pxx = welch(x, fs=fs, nperseg=min(len(x), int(fs * 4)))

    plt.figure()
    plt.semilogy(f, Pxx)
    plt.xlim(0, 80)
    plt.title(f"PSD: {sname} ch{ch}")
    plt.xlabel("Hz")
    plt.show()

#%%
stream_name = "EEG2"   # or "Wav5"
ekg_chs = [2,3]  # example indices you find

fs = float(blk.streams[stream_name].fs)
X = np.asarray(blk.streams[stream_name].data)[ekg_chs, :]  # (n_ch, n_samp)

import mne
info = mne.create_info(
    ch_names=[f"EKG{i+1}" for i in range(len(ekg_chs))],
    sfreq=fs,
    ch_types=["ecg"] * len(ekg_chs),
)
raw_ekg = mne.io.RawArray(X, info)

import numpy as np
mont = mne.channels.make_dig_montage(
    ch_pos={ch: np.array([np.nan, np.nan, np.nan]) for ch in raw_ekg.ch_names},
    coord_frame="mri",
)
raw_ekg.set_montage(mont, on_missing="ignore")  # affects only raw_ekg



