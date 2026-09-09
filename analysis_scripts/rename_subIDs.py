#!/usr/bin/env python3
import os
import sys
import argparse
import shutil
from pathlib import Path

from typing import Optional

os.chdir('/Volumes/Samsung/scripts/movies_ET_attn_main/scripts/')
print("Current working directory:", os.getcwd())


# ---- CONFIG (default root, can override with --root) ----
ROOT = Path("/Volumes/Samsung/Movie_data/movies_nwb_standard")

# Mapping from your table: nwb_label -> NS_label
# MAPPING = {
#     "sub-0101": "LH010",
#     "sub-0201": "LH012",
#     "sub-0301": "NS086_02",
#     "sub-0401": "NS127_02",
#     "sub-0501": "NS128_02",
#     "sub-0601": "NS132",
#     "sub-0701": "NS133_02",
#     "sub-0801": "NS134",
#     "sub-0901": "NS135",
#     "sub-1001": "NS136",
#     "sub-1101": "NS137",
#     "sub-1201": "NS138",
#     "sub-1301": "NS140",
#     "sub-1302": "NS140_02",
#     "sub-1401": "NS141",
#     "sub-1501": "NS142",
#     "sub-1601": "NS144",
#     "sub-1602": "NS144_02",
#     "sub-1701": "NS145",
#     "sub-1801": "NS148_02",
#     "sub-1901": "NS149",
#     "sub-2001": "NS151",
#     "sub-2101": "NS153",
#     "sub-2201": "NS154",
#     "sub-2301": "NS155",
#     "sub-23": "NS155",
#     "sub-2302": "NS155_02",
#     "sub-2401": "NS164",
#     "sub-2501": "NS166",
#     "sub-2601": "NS167",
#     "sub-2701": "NS170",
#     "sub-2801": "NS173",
#     "sub-2901": "NS174_02",
#     "sub-2902": "NS174_03",
#     "sub-3001": "NS178",
#     "sub-3101": "NS184_02",
#     "sub-3201": "NS189",
#     "sub-3301": "NS190",
#     "sub-3401": "NS191",
#     "sub-3501": "NS192",
#     "sub-3601": "NS193",
#     "sub-3701": "NS194",
#     "sub-3801": "NS196",
#     "sub-3901": "NS201_02",
#     "sub-4001": "NS203",
#     "sub-4101": "NS204",
#     "sub-4201": "NS205",
#     "sub-4301": "NS206",
#     "sub-4401": "NS208",
#     "sub-4501": "NS210",
#     "sub-4601": "NS211",
# }

MAPPING = {
    "sub-01": "LH010",
    "sub-02": "LH012",
    "sub-03": "NS086_02",
    "sub-04": "NS127_02",
    "sub-05": "NS128_02",
    "sub-06": "NS132",
    "sub-07": "NS133_02",
    "sub-08": "NS134",
    "sub-09": "NS135",
    "sub-10": "NS136",
    "sub-11": "NS137",
    "sub-12": "NS138",
    "sub-13": "NS140",
    "sub-13": "NS140_02",
    "sub-14": "NS141",
    "sub-15": "NS142",
    "sub-16": "NS144",
    "sub-16": "NS144_02",
    "sub-17": "NS145",
    "sub-18": "NS148_02",
    "sub-19": "NS149",
    "sub-20": "NS151",
    "sub-21": "NS153",
    "sub-22": "NS154",
    "sub-23": "NS155",
    "sub-23": "NS155_02",
    "sub-24": "NS164",
    "sub-25": "NS166",
    "sub-26": "NS167",
    "sub-27": "NS170",
    "sub-28": "NS173",
    "sub-29": "NS174_02",
    "sub-29": "NS174_03",
    "sub-30": "NS178",
    "sub-31": "NS184_02",
    "sub-32": "NS189",
    "sub-33": "NS190",
    "sub-34": "NS191",
    "sub-35": "NS192",
    "sub-36": "NS193",
    "sub-37": "NS194",
    "sub-38": "NS196",
    "sub-39": "NS201_02",
    "sub-40": "NS203",
    "sub-41": "NS204",
    "sub-42": "NS205",
    "sub-43": "NS206",
    "sub-44": "NS208",
    "sub-45": "NS210",
    "sub-46": "NS211",
}

# --- Lazy import so the script can still run without pynwb if only doing dry-run of renames
def _load_pynwb():
    try:
        from pynwb import NWBHDF5IO
        from pynwb.file import Subject
        return NWBHDF5IO, Subject
    except Exception as e:
        print(f"[ERROR] pynwb not available: {e}", file=sys.stderr)
        return None, None


def rename_inside_dir(dir_path: Path, old_token: str, new_token: str, apply: bool) -> int:
    """
    Rename files and (optionally) subdirectories containing `old_token` → `new_token` under dir_path.
    Returns count of rename operations performed (or that would be performed in dry-run).
    """
    count = 0
    if not dir_path.is_dir():
        return 0

    for root, dirs, files in os.walk(dir_path, topdown=False):
        root_p = Path(root)

        # Files
        for name in files:
            if old_token in name:
                src = root_p / name
                dst = root_p / name.replace(old_token, new_token)
                print(f"FILE: {src}  ->  {dst}")
                if apply:
                    if dst.exists():
                        print(f"  [SKIP] Destination exists: {dst}")
                    else:
                        src.rename(dst)
                count += 1

        # Subdirectories
        for dname in dirs:
            if old_token in dname:
                src_d = root_p / dname
                dst_d = root_p / dname.replace(old_token, new_token)
                print(f"DIR : {src_d}  ->  {dst_d}")
                if apply:
                    if dst_d.exists():
                        print(f"  [SKIP] Destination exists: {dst_d}")
                    else:
                        src_d.rename(dst_d)
                count += 1
    return count


def ensure_backup(path: Path):
    """Create a one-time .bak copy next to `path` (filename.nwb.bak) if it doesn't exist."""
    bak = path.with_suffix(path.suffix + ".bak")
    if bak.exists():
        return bak
    shutil.copy2(path, bak)
    return bak


def update_nwb_subject_id(nwb_path: Path, new_subject_id: str, apply: bool, backup: bool) -> Optional[str]:
    """
    Add/Update a custom field with the NS label without changing subject.subject_id:
      - nwbfile.ns_subject_id = <NS_LABEL>
      - if subject exists, also try: nwbfile.subject.ns_subject_id = <NS_LABEL>
    """
    NWBHDF5IO, Subject = _load_pynwb()
    if NWBHDF5IO is None:
        return "[ERROR] pynwb is required to update NWB files."

    if not nwb_path.is_file():
        return f"[WARN] Not a file: {nwb_path}"

    try:
        # Do a quick read to show current values (optional, but helpful in logs)
        with NWBHDF5IO(str(nwb_path), mode='r') as io:
            nwb_peek = io.read()
            current_sid = getattr(getattr(nwb_peek, "subject", None), "subject_id", None)
            current_ns_root = getattr(nwb_peek, "ns_subject_id", None)
            current_ns_subj = getattr(getattr(nwb_peek, "subject", None), "ns_subject_id", None)

        msg = (f"NWB: {nwb_path.name}  subject_id: {current_sid!r} | "
               f"ns_subject_id (root): {current_ns_root!r} -> {new_subject_id!r} | "
               f"ns_subject_id (subject): {current_ns_subj!r} -> {new_subject_id!r}")
        print(msg)

        if not apply:
            return msg

        if backup:
            ensure_backup(nwb_path)

        # Single open in r+ to modify & write
        with NWBHDF5IO(str(nwb_path), mode='r+') as io:
            nwbfile = io.read()

            # Root-level custom field
            setattr(nwbfile, "ns_subject_id", new_subject_id)

            # Mirror on Subject if present; ignore if PyNWB disallows dynamic attr here
            if getattr(nwbfile, "subject", None) is not None:
                try:
                    setattr(nwbfile.subject, "ns_subject_id", new_subject_id)
                except Exception:
                    # Not critical; some versions may not persist arbitrary attrs on Subject
                    pass

            io.write(nwbfile)

        return None

    except Exception as e:
        return f"[ERROR] Failed to update {nwb_path}: {e}"


def main():
    ap = argparse.ArgumentParser(
        description="Rename subject folders and filenames from nwb_label to NS_label, and update NWB subject_id."
    )
    ap.add_argument("--apply", action="store_true",
                    help="Perform changes (default: dry-run / print only).")
    ap.add_argument("--root", type=str, default=str(ROOT),
                    help="Root directory containing subject folders (default: %(default)s)")
    ap.add_argument("--no-nwb", action="store_true",
                    help="Skip updating NWB subject_id (only do renames).")
    ap.add_argument("--backup", action="store_true",
                    help="When applying NWB edits, create filename.nwb.bak before writing.")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        print(f"[ERROR] Root not found: {root}", file=sys.stderr)
        sys.exit(1)

    print(f"Root: {root}")
    print("Mode:", "APPLY (will rename + edit NWBs)" if args.apply else "DRY-RUN (no filesystem changes)")
    if not args.no_nwb:
        print("NWB subject_id updates: ENABLED")
    else:
        print("NWB subject_id updates: DISABLED")

    for nwb_label, ns_label in MAPPING.items():
        old_dir = root / nwb_label
        new_dir = root / ns_label

        candidates = []
        if old_dir.is_dir():
            candidates.append((old_dir, nwb_label, ns_label))
        if new_dir.is_dir():
            candidates.append((new_dir, nwb_label, ns_label))

        if not candidates:
            #print(f"[WARN] Neither {old_dir} nor {new_dir} exists. Skipping {nwb_label} -> {ns_label}.")
            continue

        # Update filenames/directories that still contain the old token
        for d, old_tok, new_tok in candidates:
            print(f"\nScanning: {d}  (replace '{old_tok}' -> '{new_tok}')")
            rename_inside_dir(d, old_tok, new_tok, apply=args.apply)

        # If the old dir still exists after inner renames, rename the subject folder itself
        if old_dir.is_dir():
            print(f"\nDIR : {old_dir}  ->  {new_dir}")
            if args.apply:
                if new_dir.exists():
                    print(f"  [SKIP] Destination exists: {new_dir}")
                else:
                    old_dir.rename(new_dir)

        # After renames, determine where to look for NWBs (prefer new_dir if present)
        subject_dir = new_dir if new_dir.is_dir() else (old_dir if old_dir.is_dir() else None)
        if subject_dir is None:
            continue

        # NWB updates (subject_id -> ns_label)
        if not args.no_nwb:
            for nwb_path in subject_dir.rglob("*.nwb"):
                status = update_nwb_subject_id(nwb_path, ns_label, apply=args.apply, backup=args.backup)
                if status:
                    print(status)

    print("\nDone.")



# --- everything above stays the same ---

if __name__ == "__main__":
    # Option A: normal CLI entry
    main()

# --- Option B: local call section for Spyder / notebooks ---
# Comment out the "main()" above if you only want this to run when you hit Run in Spyder.
if True:  # change to True when you want to run from Spyder interactively
    from pathlib import Path

    # >>> customize your parameters here <<<
    root = Path("/Volumes/Samsung/Movie_data/movies_nwb_standard")
    apply = True      # True = actually rename/edit files, False = dry run
    backup = False     # backup .nwb files before writing
    skip_nwb = False  # True = skip updating NWBs

    # Simulate command-line args
    import sys
    sys.argv = [
        "rename_subjects_and_update_nwb.py",
        "--root", str(root),
        *(["--apply"] if apply else []),
        *(["--backup"] if backup else []),
        *(["--no-nwb"] if skip_nwb else []),
    ]

    # Run main() with those parameters
    main()
