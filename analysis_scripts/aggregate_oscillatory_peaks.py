from pathlib import Path

script = r'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-'''

"""
Create time-aligned attention-window labels and merge them onto aggregated
rolling-FOOOF peak data.

Designed to be run cell-by-cell in Spyder.

Main outputs:
    1. One wide attention-label table per movie
    2. Bad/good-window flags added to that table
    3. Attention and bad-window information merged onto the aggregated
       FOOOF peak-level dataframe using explicit Window_Start_Sec
@author: Christine Chesebrough
"""

#%% Imports

import os
import re
import numpy as np
import pandas as pd


#%% User settings

machine_path = "Volumes"  # use "media/christine" on Linux

movies = [
    "despicable_me_english",
    "despicable_me_hungarian",
]

# Threshold pairs to search for
threshold_values = [
    (0.5, 0.5),
    (0.6, 0.6),
]

win_len_sec = 10
window_step_sec = 2.5
method = "power_log"

# Attention-label directories
separate_pc_dir = (
    f"/{machine_path}/Samsung/Movie_data/"
    "separate_PC_features_7Jan26"
)

shared_pc_dir = (
    f"/{machine_path}/Samsung/Movie_data/"
    "shared_PC_features_10s_20Apr26"
)

attention_dirs = {
    "separate_PC": separate_pc_dir,
    "shared_PC1": shared_pc_dir,
}

# Aggregated FOOOF files created previously
fooof_aggregated_dir = (
    f"/{machine_path}/Samsung/Movie_data/"
    "rolling_fooof_aggregated_26Jun26"
)

output_dir = (
    f"/{machine_path}/Samsung/Movie_data/"
    "rolling_fooof_aggregated_10Jul26/"
    "attention_label_merges"
)

os.makedirs(output_dir, exist_ok=True)


#%% Small helper functions only for repeated string parsing

def extract_patient(entry_id):
    """
    Extract patient ID such as:
        NS127_02
        NS190
        LH010
    """
    match = re.match(
        r"((?:NS|LH)\d+(?:_\d+)?)",
        str(entry_id),
        flags=re.IGNORECASE,
    )
    if not match:
        raise ValueError(f"Could not extract patient from: {entry_id}")

    return match.group(1).upper()


def extract_run(entry_id):
    """Return run label in run-01 format."""
    match = re.search(
        r"run[-_]?(\d+)",
        str(entry_id),
        flags=re.IGNORECASE,
    )

    if match:
        return f"run-{int(match.group(1)):02d}"

    return "run-01"


def extract_session(entry_id, patient):
    """
    Use explicit ses-XX when available.
    Otherwise infer session from patient suffix:
        NS127_02 -> ses-02
        NS178    -> ses-01
    """
    match = re.search(
        r"ses[-_]?(\d+)",
        str(entry_id),
        flags=re.IGNORECASE,
    )

    if match:
        return f"ses-{int(match.group(1)):02d}"

    suffix_match = re.search(r"_(\d+)$", str(patient))

    if suffix_match:
        return f"ses-{int(suffix_match.group(1)):02d}"

    return "ses-01"


#%% Initialize combined outputs

all_movie_label_tables = []
attention_processing_log = []
bad_window_processing_log = []


#%% Main movie loop

for movie in movies:

    print("\n" + "=" * 80)
    print(f"PROCESSING {movie}")
    print("=" * 80)

    # This dataframe will grow horizontally as each threshold/PC file is merged
    movie_labels_wide = None

    #%% Load and merge all attention-label files for this movie

    for pc_type, attention_dir in attention_dirs.items():

        for threshold in threshold_values:

            thr_str = f"[{threshold[0]:.1f}, {threshold[1]:.1f}]"

            # Machine-safe version for output column names
            thr_name = (
                f"{threshold[0]:.1f}_{threshold[1]:.1f}"
                .replace(".", "p")
            )

            attention_file = os.path.join(
                attention_dir,
                f"{movie}_features_df_{thr_str}.csv",
            )

            print("\nLooking for:")
            print(attention_file)

            if not os.path.isfile(attention_file):

                print(
                    f"[WARNING] Missing attention file for "
                    f"{movie}, {pc_type}, threshold {threshold}"
                )

                attention_processing_log.append({
                    "Movie": movie,
                    "PC_Type": pc_type,
                    "Threshold": str(threshold),
                    "Source_File": attention_file,
                    "Status": "missing",
                    "N_Rows": 0,
                    "N_Entries": 0,
                })

                continue

            # Load source file
            attention_df = pd.read_csv(attention_file)

            print(
                f"Loaded {len(attention_df)} rows from "
                f"{os.path.basename(attention_file)}"
            )

            # Find patient/entry column case-insensitively
            patient_col = next(
                (
                    col for col in attention_df.columns
                    if col.lower() in ["patient", "entry_id", "entryid"]
                ),
                None,
            )

            if patient_col is None:
                raise KeyError(
                    f"No patient/entry column found in {attention_file}"
                )

            # Find Time column case-insensitively
            time_col = next(
                (
                    col for col in attention_df.columns
                    if col.lower() == "time"
                ),
                None,
            )

            if time_col is None:
                raise KeyError(
                    f"No Time column found in {attention_file}"
                )

            # Expected attention-label columns
            within_subject_col = (
                f"Attention_Label_within_subject_dev_{thr_str}"
            )

            within_timepoint_col = (
                f"Attention_Label_within_timepoint_dev_{thr_str}"
            )

            # Case-insensitive fallback for column matching
            if within_subject_col not in attention_df.columns:
                subject_matches = [
                    col for col in attention_df.columns
                    if (
                        "attention_label" in col.lower()
                        and "within_subject" in col.lower()
                        and thr_str.replace(" ", "") in col.replace(" ", "")
                    )
                ]

                if len(subject_matches) != 1:
                    raise KeyError(
                        f"Could not uniquely find within-subject column in "
                        f"{attention_file}\nMatches: {subject_matches}"
                    )

                within_subject_col = subject_matches[0]

            if within_timepoint_col not in attention_df.columns:
                timepoint_matches = [
                    col for col in attention_df.columns
                    if (
                        "attention_label" in col.lower()
                        and "within_timepoint" in col.lower()
                        and thr_str.replace(" ", "") in col.replace(" ", "")
                    )
                ]

                if len(timepoint_matches) != 1:
                    raise KeyError(
                        f"Could not uniquely find within-timepoint column in "
                        f"{attention_file}\nMatches: {timepoint_matches}"
                    )

                within_timepoint_col = timepoint_matches[0]

            # Keep only needed fields
            current_labels = attention_df[
                [
                    patient_col,
                    time_col,
                    within_subject_col,
                    within_timepoint_col,
                ]
            ].copy()

            current_labels.rename(
                columns={
                    patient_col: "Source_Entry_ID",
                    time_col: "Attention_Time_Index",
                    within_subject_col: (
                        f"Attention_Label_{pc_type}_"
                        f"within_subject_thr_{thr_name}"
                    ),
                    within_timepoint_col: (
                        f"Attention_Label_{pc_type}_"
                        f"within_timepoint_thr_{thr_name}"
                    ),
                },
                inplace=True,
            )

            # Build explicit zero-based attention window index
            current_labels["Attention_Time_Index"] = pd.to_numeric(
                current_labels["Attention_Time_Index"],
                errors="raise",
            ).astype(int)

            current_labels["Attention_Window_Index"] = (
                current_labels["Attention_Time_Index"] - 1
            )

            # Earlier attention windows began at 0 s
            current_labels["Window_Start_Sec"] = (
                current_labels["Attention_Window_Index"]
                * window_step_sec
            ).round(6)

            # Parse identifiers
            current_labels["Patient"] = (
                current_labels["Source_Entry_ID"]
                .apply(extract_patient)
            )

            current_labels["Run"] = (
                current_labels["Source_Entry_ID"]
                .apply(extract_run)
            )

            current_labels["Session"] = [
                extract_session(entry_id, patient)
                for entry_id, patient in zip(
                    current_labels["Source_Entry_ID"],
                    current_labels["Patient"],
                )
            ]

            current_labels["Entry_ID"] = (
                current_labels["Patient"]
                + "_"
                + current_labels["Session"]
                + "_"
                + current_labels["Run"]
            )

            current_labels["Movie"] = movie

            # Check uniqueness
            duplicate_check = current_labels.duplicated(
                [
                    "Movie",
                    "Entry_ID",
                    "Window_Start_Sec",
                ]
            )

            if duplicate_check.any():
                print(
                    current_labels.loc[
                        duplicate_check,
                        [
                            "Movie",
                            "Entry_ID",
                            "Window_Start_Sec",
                        ],
                    ].head()
                )

                raise ValueError(
                    f"Duplicate entry/window rows found in {attention_file}"
                )

            # Keep identifiers and renamed label columns
            label_cols = [
                col for col in current_labels.columns
                if col.startswith("Attention_Label_")
            ]

            current_labels = current_labels[
                [
                    "Movie",
                    "Entry_ID",
                    "Patient",
                    "Session",
                    "Run",
                    "Attention_Time_Index",
                    "Attention_Window_Index",
                    "Window_Start_Sec",
                ]
                + label_cols
            ]

            # Merge horizontally
            merge_cols = [
                "Movie",
                "Entry_ID",
                "Patient",
                "Session",
                "Run",
                "Attention_Time_Index",
                "Attention_Window_Index",
                "Window_Start_Sec",
            ]

            if movie_labels_wide is None:
                movie_labels_wide = current_labels.copy()

            else:
                movie_labels_wide = movie_labels_wide.merge(
                    current_labels,
                    how="outer",
                    on=merge_cols,
                    validate="one_to_one",
                )

            attention_processing_log.append({
                "Movie": movie,
                "PC_Type": pc_type,
                "Threshold": str(threshold),
                "Source_File": attention_file,
                "Status": "loaded",
                "N_Rows": len(current_labels),
                "N_Entries": current_labels["Entry_ID"].nunique(),
            })

            print(
                f"Current wide table shape: "
                f"{movie_labels_wide.shape}"
            )

    #%% Check merged attention-label table before bad-window merge

    if movie_labels_wide is None:
        print(
            f"[WARNING] No attention files loaded for {movie}. "
            f"Skipping this movie."
        )
        continue

    movie_labels_wide = movie_labels_wide.sort_values(
        [
            "Entry_ID",
            "Window_Start_Sec",
        ]
    ).reset_index(drop=True)

    print("\nAttention-label table summary:")
    print(movie_labels_wide.shape)
    print(
        movie_labels_wide.groupby("Entry_ID")
        .size()
        .describe()
    )

    print("\nAttention label columns:")
    print([
        col for col in movie_labels_wide.columns
        if col.startswith("Attention_Label_")
    ])

    #%% Add bad-window flags

    movie_labels_wide["Is_Bad_Window"] = False
    movie_labels_wide["Is_Good_Window"] = True
    movie_labels_wide["Bad_Window_Source_File"] = pd.NA

    bad_windows_dir = (
        f"/{machine_path}/Samsung/Movie_data/"
        f"1secEpochs_for_review_power_log_{movie}_HFA_1Apr26"
    )

    for entry_id in sorted(
        movie_labels_wide["Entry_ID"].dropna().unique()
    ):

        print("\nFinding bad windows for:")
        print(entry_id)

        entry_rows = movie_labels_wide[
            movie_labels_wide["Entry_ID"] == entry_id
        ]

        patient = entry_rows["Patient"].iloc[0]
        session = entry_rows["Session"].iloc[0]
        run = entry_rows["Run"].iloc[0]

        bad_windows_pat_dir = os.path.join(
            bad_windows_dir,
            patient,
        )

        if not os.path.isdir(bad_windows_pat_dir):

            print(
                f"[WARNING] Missing bad-window directory: "
                f"{bad_windows_pat_dir}"
            )

            bad_window_processing_log.append({
                "Movie": movie,
                "Entry_ID": entry_id,
                "Bad_Window_File": np.nan,
                "Status": "missing_patient_directory",
                "N_Bad_Indices": 0,
                "N_Bad_Times_Matched": 0,
            })

            continue

        bad_window_files = sorted([
            filename
            for filename in os.listdir(bad_windows_pat_dir)
            if (
                method.lower() in filename.lower()
                and filename.endswith(".csv")
                and f"{win_len_sec}s_bad_window_indices" in filename
                and not filename.startswith("._")
            )
        ])

        if len(bad_window_files) == 0:

            print(
                f"[WARNING] No bad-window file found for {entry_id}"
            )

            bad_window_processing_log.append({
                "Movie": movie,
                "Entry_ID": entry_id,
                "Bad_Window_File": np.nan,
                "Status": "missing_bad_window_file",
                "N_Bad_Indices": 0,
                "N_Bad_Times_Matched": 0,
            })

            continue

        # Prefer run/session match when multiple files exist
        if len(bad_window_files) > 1:

            run_variants = [
                run.lower(),
                run.lower().replace("run-0", "run-"),
            ]

            session_variants = [
                session.lower(),
                session.lower().replace("ses-0", "ses-"),
            ]

            scored_files = []

            for filename in bad_window_files:

                filename_lower = filename.lower()
                score = 0

                if any(
                    value in filename_lower
                    for value in run_variants
                ):
                    score += 4

                if any(
                    value in filename_lower
                    for value in session_variants
                ):
                    score += 2

                if movie.lower() in filename_lower:
                    score += 1

                scored_files.append((score, filename))

            scored_files = sorted(
                scored_files,
                reverse=True,
            )

            best_score = scored_files[0][0]

            best_files = [
                filename
                for score, filename in scored_files
                if score == best_score
            ]

            if len(best_files) != 1:

                print("Candidate files:")
                print(bad_window_files)

                raise RuntimeError(
                    f"Could not uniquely select bad-window file "
                    f"for {entry_id}"
                )

            bad_window_file = best_files[0]

        else:
            bad_window_file = bad_window_files[0]

        bad_window_path = os.path.join(
            bad_windows_pat_dir,
            bad_window_file,
        )

        print(f"Using: {bad_window_path}")

        bad_windows = pd.read_csv(bad_window_path)

        bad_index_col = next(
            (
                col for col in bad_windows.columns
                if col.lower() == f"window_idx_{win_len_sec}s"
            ),
            None,
        )

        if bad_index_col is None:
            raise KeyError(
                f"Could not find window_idx_{win_len_sec}s "
                f"in {bad_window_path}"
            )

        bad_idx = (
            pd.to_numeric(
                bad_windows[bad_index_col],
                errors="coerce",
            )
            .dropna()
            .astype(int)
            .unique()
        )

        # Earlier bad-window indexing also began at 0 s
        bad_window_start_times = (
            bad_idx * window_step_sec
        ).round(6)

        entry_mask = (
            movie_labels_wide["Entry_ID"] == entry_id
        )

        bad_match_mask = (
            entry_mask
            & movie_labels_wide["Window_Start_Sec"].isin(
                bad_window_start_times
            )
        )

        movie_labels_wide.loc[
            bad_match_mask,
            "Is_Bad_Window",
        ] = True

        movie_labels_wide.loc[
            bad_match_mask,
            "Is_Good_Window",
        ] = False

        movie_labels_wide.loc[
            entry_mask,
            "Bad_Window_Source_File",
        ] = bad_window_path

        matched_bad_times = (
            movie_labels_wide.loc[
                bad_match_mask,
                "Window_Start_Sec",
            ]
            .drop_duplicates()
            .to_numpy()
        )

        unmatched_bad_times = np.setdiff1d(
            bad_window_start_times,
            matched_bad_times,
        )

        print(
            f"Bad indices: {len(bad_idx)} | "
            f"Matched times: {len(matched_bad_times)} | "
            f"Unmatched times: {len(unmatched_bad_times)}"
        )

        bad_window_processing_log.append({
            "Movie": movie,
            "Entry_ID": entry_id,
            "Bad_Window_File": bad_window_path,
            "Status": "loaded",
            "N_Bad_Indices": len(bad_idx),
            "N_Bad_Times_Matched": len(matched_bad_times),
            "N_Bad_Times_Unmatched": len(unmatched_bad_times),
        })

    #%% Inspect bad-window results before saving

    print("\nBad-window counts by entry:")
    print(
        movie_labels_wide
        .groupby("Entry_ID")["Is_Bad_Window"]
        .sum()
    )

    #%% Save movie-level label table

    label_output_path = os.path.join(
        output_dir,
        f"{movie}_all_attention_window_labels.csv",
    )

    movie_labels_wide.to_csv(
        label_output_path,
        index=False,
    )

    print(f"\nSaved attention-label table:\n{label_output_path}")

    all_movie_label_tables.append(
        movie_labels_wide.copy()
    )

    #%% Load aggregated FOOOF peak table

    fooof_file = os.path.join(
        fooof_aggregated_dir,
        (
            f"{movie}_all_recordings_low_mid_"
            "rolling_fooof_peaks_with_atlas.csv"
        ),
    )

    if not os.path.isfile(fooof_file):

        print(
            f"[WARNING] FOOOF file not found:\n{fooof_file}"
        )

        continue

    fooof_df = pd.read_csv(fooof_file)

    print("\nLoaded aggregated FOOOF table:")
    print(fooof_df.shape)

    #%% Create matching Entry_ID in FOOOF table

    if "Session" not in fooof_df.columns:
        fooof_df["Session"] = pd.NA

    fooof_df["Patient"] = (
        fooof_df["Patient"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    fooof_df["Run"] = (
        fooof_df["Run"]
        .astype(str)
        .apply(extract_run)
    )

    fooof_sessions = []

    for patient, session_value in zip(
        fooof_df["Patient"],
        fooof_df["Session"],
    ):

        if pd.notna(session_value):
            session = extract_session(
                str(session_value),
                patient,
            )
        else:
            session = extract_session(
                "",
                patient,
            )

        fooof_sessions.append(session)

    fooof_df["Session"] = fooof_sessions

    fooof_df["Entry_ID"] = (
        fooof_df["Patient"]
        + "_"
        + fooof_df["Session"]
        + "_"
        + fooof_df["Run"]
    )

    fooof_df["Movie"] = movie

    fooof_df["Window_Start_Sec"] = pd.to_numeric(
        fooof_df["Window_Start_Sec"],
        errors="raise",
    ).round(6)

    print("\nFOOOF entry IDs:")
    print(sorted(fooof_df["Entry_ID"].unique()))

    #%% Merge labels onto FOOOF by movie, entry, and start time

    merge_keys = [
        "Movie",
        "Entry_ID",
        "Window_Start_Sec",
    ]

    label_columns_to_merge = [
        col for col in movie_labels_wide.columns
        if col not in [
            "Patient",
            "Session",
            "Run",
        ]
    ]

    fooof_with_labels = fooof_df.merge(
        movie_labels_wide[label_columns_to_merge],
        how="left",
        on=merge_keys,
        validate="many_to_one",
        indicator="_Attention_Merge",
    )

    print("\nMerge result:")
    print(
        fooof_with_labels["_Attention_Merge"]
        .value_counts(dropna=False)
    )

    #%% Inspect unmatched FOOOF windows

    unmatched_fooof_windows = (
        fooof_with_labels[
            fooof_with_labels["_Attention_Merge"] != "both"
        ][
            [
                "Movie",
                "Patient",
                "Session",
                "Run",
                "Entry_ID",
                "Window_Index",
                "Window_Start_Sec",
                "_Attention_Merge",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "Entry_ID",
                "Window_Start_Sec",
            ]
        )
    )

    print("\nUnmatched FOOOF windows:")
    print(unmatched_fooof_windows.head(20))
    print(f"Total unmatched windows: {len(unmatched_fooof_windows)}")

    #%% Save merged FOOOF table

    merged_output_path = os.path.join(
        output_dir,
        (
            f"{movie}_all_recordings_low_mid_rolling_fooof_"
            "peaks_with_atlas_attention_badwindows.csv"
        ),
    )

    fooof_with_labels.to_csv(
        merged_output_path,
        index=False,
    )

    print(f"\nSaved merged FOOOF table:\n{merged_output_path}")

    unmatched_output_path = os.path.join(
        output_dir,
        f"{movie}_unmatched_fooof_windows.csv",
    )

    unmatched_fooof_windows.to_csv(
        unmatched_output_path,
        index=False,
    )


#%% Save combined attention-label table

if len(all_movie_label_tables) > 0:

    all_movies_labels = pd.concat(
        all_movie_label_tables,
        ignore_index=True,
        sort=False,
    )

    all_movies_output_path = os.path.join(
        output_dir,
        "all_movies_attention_window_labels.csv",
    )

    all_movies_labels.to_csv(
        all_movies_output_path,
        index=False,
    )

    print(
        f"\nSaved combined attention-label table:\n"
        f"{all_movies_output_path}"
    )


#%% Save processing logs

attention_processing_log_df = pd.DataFrame(
    attention_processing_log
)

bad_window_processing_log_df = pd.DataFrame(
    bad_window_processing_log
)

attention_log_path = os.path.join(
    output_dir,
    "attention_label_processing_log.csv",
)

bad_window_log_path = os.path.join(
    output_dir,
    "bad_window_processing_log.csv",
)

attention_processing_log_df.to_csv(
    attention_log_path,
    index=False,
)

bad_window_processing_log_df.to_csv(
    bad_window_log_path,
    index=False,
)

print("\nFinished.")