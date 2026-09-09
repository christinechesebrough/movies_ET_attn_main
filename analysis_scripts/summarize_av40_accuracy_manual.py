#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 12:28:33 2026

@author: christinechesebrough
"""

import pandas as pd
import numpy as np

input_dir = "/Volumes/Samsung/AV40_data/timing_info"
pat = "NS224"
recording = "B9_NS224_AV40AV" #"B5_NS224_AV40-AA"
csv_name = "B9_NS224_AV40AV_Wav5_ch4_vis_and_button_combined.csv"


if 'AA' in recording:
    condition = "AA"
    target_modality = "auditory"
    deviant_name = "tone_deviant"
elif 'AV' in recording:
    condition = "AV"
    target_modality = "visual"
    deviant_name = "flash_deviant"
elif 'AB' in recording:
    condition = "AB"
    target_modality = "breathing"

pat_dir = '{:s}/{:s}'.format(input_dir, pat)
recording_dir = '{:s}/{:s}'.format(pat_dir, recording)
input_path = '{:s}/{:s}'.format(recording_dir, csv_name)

df = pd.read_csv(input_path)


deviant_times = np.sort(
    df.loc[df["event_type"] == deviant_name, "time_original_s"].to_numpy()
)
button_times = np.sort(
    df.loc[df["event_type"] == "button", "time_original_s"].to_numpy()
)

response_window_s = 1.5

# One row per deviant tone
deviant_results = []
for deviant_number, deviant_time in enumerate(deviant_times, start=1):
    matching_idx = np.where(
        (button_times >= deviant_time)
        & (button_times <= deviant_time + response_window_s)
    )[0]

    first_button_time = (
        button_times[matching_idx[0]] if len(matching_idx) > 0 else np.nan
    )

    deviant_results.append({
        "deviant_number": deviant_number,
        "deviant_time_s": deviant_time,
        "hit": len(matching_idx) > 0,
        "first_button_time_s": first_button_time,
        "response_latency_s": (
            first_button_time - deviant_time if len(matching_idx) > 0 else np.nan
        ),
        "n_buttons_within_1p5s": len(matching_idx),
        "all_matching_button_times_s": "; ".join(
            f"{button_times[i]:.6f}" for i in matching_idx
        ),
    })

deviant_results = pd.DataFrame(deviant_results)

# One row per button press
button_results = []
for button_number, button_time in enumerate(button_times, start=1):
    matching_idx = np.where(
        (deviant_times <= button_time)
        & (deviant_times >= button_time - response_window_s)
    )[0]

    button_results.append({
        "button_number": button_number,
        "button_time_s": button_time,
        "within_1p5s_after_deviant": len(matching_idx) > 0,
        "n_matching_deviants": len(matching_idx),
        "matching_deviant_numbers": "; ".join(
            str(i + 1) for i in matching_idx
        ),
        "matching_deviant_times_s": "; ".join(
            f"{deviant_times[i]:.6f}" for i in matching_idx
        ),
        "response_latencies_s": "; ".join(
            f"{button_time - deviant_times[i]:.6f}" for i in matching_idx
        ),
    })

button_results = pd.DataFrame(button_results)

deviant_output = "B5_NS224_deviant_button_matches_1p5s.csv"
deviant_path = '{:s}/{:s}'.format(recording_dir,deviant_output)

button_output = "B5_NS224_button_classification_1p5s.csv"
button_path = '{:s}/{:s}'.format(recording_dir,button_output)

deviant_results.to_csv(deviant_path, index=False)
button_results.to_csv(button_path, index=False)

summary = {
    "n_deviants": len(deviant_results),
    "n_deviants_with_button": int(deviant_results["hit"].sum()),
    "hit_rate": float(deviant_results["hit"].mean()),
    "n_buttons": len(button_results),
    "n_buttons_after_deviant": int(
        button_results["within_1p5s_after_deviant"].sum()
    ),
    "n_buttons_not_after_deviant": int(
        (~button_results["within_1p5s_after_deviant"]).sum()
    ),
    "mean_latency_s": float(
        deviant_results.loc[deviant_results["hit"], "response_latency_s"].mean()
    ),
    "median_latency_s": float(
        deviant_results.loc[deviant_results["hit"], "response_latency_s"].median()
    ),
    "min_latency_s": float(
        deviant_results.loc[deviant_results["hit"], "response_latency_s"].min()
    ),
    "max_latency_s": float(
        deviant_results.loc[deviant_results["hit"], "response_latency_s"].max()
    ),
}


n_deviants = len(deviant_results)
n_hits = int(deviant_results["hit"].sum())
n_button_presses = len(button_results)

# Button presses that did not occur within response_window_s after any deviant
n_false_alarms = int(
    (~button_results["within_1p5s_after_deviant"]).sum()
)

hit_rate = (
    n_hits / n_deviants
    if n_deviants > 0
    else np.nan
)

hit_rts = deviant_results.loc[
    deviant_results["hit"],
    "response_latency_s"
]

mean_rt_s = hit_rts.mean() if len(hit_rts) > 0 else np.nan
median_rt_s = hit_rts.median() if len(hit_rts) > 0 else np.nan


accuracy_summary = pd.DataFrame([{
    "log_base": recording,
    "condition": condition,
    "target_modality": target_modality,
    "n_deviants": n_deviants,
    "response_window_s": response_window_s,
    "n_hits": n_hits,
    "hit_rate": hit_rate,
    "n_button_presses": n_button_presses,
    "n_false_alarms": n_false_alarms,
    "mean_rt_s": mean_rt_s,
    "median_rt_s": median_rt_s,
}])


accuracy_output = f"{recording}_accuracy_summary.csv"
accuracy_path = "{:s}/{:s}".format(
    recording_dir,
    accuracy_output
)

accuracy_summary.to_csv(accuracy_path, index=False)

print(accuracy_summary.to_string(index=False))
print(f"\nSaved accuracy summary to:\n{accuracy_path}")
