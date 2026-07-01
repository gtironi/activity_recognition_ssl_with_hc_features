#!/usr/bin/env python3
"""WISDM human activity dataset → windowed parquet + HC features.

Input:  dataset/raw/WISDM/wisdm-dataset/raw/{phone,watch}/{accel,gyro}/
Output: dataset/processed/WISDM/
          windowed_train.parquet
          windowed_test.parquet
          hc_manifest.json
          split_report.json

12 channels: phone_accel_{x,y,z}, phone_gyro_{x,y,z},
             watch_accel_{x,y,z}, watch_gyro_{x,y,z}.

Line format: subject_id,activity_code,timestamp_ns,x,y,z;
18 activities (A–S, no N), subjects 1600–1650, ~20 Hz.

The 4 streams (phone/watch × accel/gyro) are aligned per subject+activity segment
by resampling all streams to a common 20 Hz grid using nearest-neighbour merge.

Usage:
    python scripts/datasets/wisdm.py \\
        --raw dataset/raw/WISDM/wisdm-dataset \\
        --out dataset/processed/WISDM
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from core.windowing import window_dataframe
from core.hc_features import compute_dataframe
from core.manifest import build_manifest, save_manifest
from core.splits import split_by_gen_split, save_split_report

FS = 20
WINDOW_SIZE = 60    # 3 s @ 20 Hz
STRIDE = 30         # 50 % overlap
PURITY = 0.9

CHANNEL_COLS = [
    "phone_acc_x", "phone_acc_y", "phone_acc_z",
    "phone_gyr_x", "phone_gyr_y", "phone_gyr_z",
    "watch_acc_x", "watch_acc_y", "watch_acc_z",
    "watch_gyr_x", "watch_gyr_y", "watch_gyr_z",
]

_ACTIVITY_MAP_PATH = "activity_key.txt"


def _load_activity_key(wisdm_dir: Path) -> dict[str, str]:
    key_file = wisdm_dir / "activity_key.txt"
    mapping = {}
    if key_file.exists():
        for line in key_file.read_text().splitlines():
            if "=" in line:
                name, code = line.split("=")
                mapping[code.strip()] = name.strip()
    return mapping


def _read_stream(txt_path: Path) -> pd.DataFrame:
    """Parse one stream file → DataFrame with [subject, activity, t_ns, x, y, z]."""
    rows = []
    for line in txt_path.read_text().splitlines():
        line = line.strip().rstrip(";")
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 6:
            continue
        try:
            subject = int(parts[0])
            activity = parts[1].strip()
            t_ns = int(parts[2])
            x, y, z = float(parts[3]), float(parts[4]), float(parts[5])
        except (ValueError, IndexError):
            continue
        rows.append((subject, activity, t_ns, x, y, z))
    cols = ["subject", "activity", "t_ns", "x", "y", "z"]
    return pd.DataFrame(rows, columns=cols)


def _load_all_streams(raw_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all 4 streams, return dict keyed by (device, sensor)."""
    streams: dict[str, pd.DataFrame] = {}
    for device in ("phone", "watch"):
        for sensor in ("accel", "gyro"):
            folder = raw_dir / "raw" / device / sensor
            if not folder.exists():
                print(f"  Warning: {folder} not found")
                continue
            frames = []
            for txt in sorted(folder.glob("*.txt")):
                df = _read_stream(txt)
                if not df.empty:
                    frames.append(df)
            if frames:
                streams[(device, sensor)] = pd.concat(frames, ignore_index=True)
                print(f"  {device}/{sensor}: {len(streams[(device,sensor)]):,} rows")
    return streams


def _merge_streams(streams: dict, activity_map: dict) -> pd.DataFrame:
    """
    Merge the 4 streams per (subject, activity) segment to a common 20 Hz grid.
    Strategy: for each segment, build a uniform time grid at FS Hz spanning the
    shortest common range, then nearest-neighbour fill each stream.
    """
    all_subjects = set()
    for df in streams.values():
        all_subjects |= set(df["subject"].unique())

    all_rows = []
    for subj in sorted(all_subjects):
        # Gather activities present in ALL four streams for this subject
        act_sets = []
        for df in streams.values():
            sub_df = df[df["subject"] == subj]
            act_sets.append(set(sub_df["activity"].unique()))
        common_acts = act_sets[0]
        for s in act_sets[1:]:
            common_acts &= s

        for act in sorted(common_acts):
            label = activity_map.get(act, act)
            # Slice each stream to this subject+activity, and re-base each
            # stream's clock to its own start (phone and watch use
            # independent, unsynchronized device clocks — only the
            # elapsed/relative time within a segment is meaningful).
            slices: dict = {}
            for key, df in streams.items():
                sl = df[(df["subject"] == subj) & (df["activity"] == act)].sort_values("t_ns")
                if len(sl) > 0:
                    sl = sl.copy()
                    sl["t_ns"] = sl["t_ns"] - sl["t_ns"].iloc[0]
                slices[key] = sl

            # Build common (relative) time grid from the overlapping range
            t_starts = [sl["t_ns"].iloc[0] for sl in slices.values() if len(sl) > 0]
            t_ends   = [sl["t_ns"].iloc[-1] for sl in slices.values() if len(sl) > 0]
            if not t_starts:
                continue
            t0 = max(t_starts)
            t1 = min(t_ends)
            if t1 <= t0:
                continue

            dt_ns = int(1e9 / FS)
            grid = np.arange(t0, t1, dt_ns, dtype=np.int64)
            if len(grid) < WINDOW_SIZE:
                continue

            # Nearest-neighbour resample each stream onto grid
            resampled: dict[str, np.ndarray] = {}
            for (device, sensor), sl in slices.items():
                if len(sl) == 0:
                    break
                t_arr = sl["t_ns"].to_numpy()
                for axis, col in zip("xyz", [f"{device}_{sensor[0]}_{ax}" for ax in "xyz"]):
                    v_arr = sl[axis].to_numpy()
                    idx = np.searchsorted(t_arr, grid)
                    idx = np.clip(idx, 0, len(t_arr) - 1)
                    resampled[col] = v_arr[idx]
            else:
                # Build rows (one per sample on the grid)
                for i, t in enumerate(grid):
                    row = {"dateTime": int(t), "calf_id": subj,
                           "label": label, "seg_id": 0}
                    for col, arr in resampled.items():
                        row[col] = float(arr[i])
                    all_rows.append(row)

    df_merged = pd.DataFrame(all_rows)
    # Rename sensor shorthand to canonical channel names
    rename = {}
    for device in ("phone", "watch"):
        for sensor, short in (("accel", "acc"), ("gyro", "gyr")):
            for ax in "xyz":
                rename[f"{device}_{sensor[0]}_{ax}"] = f"{device}_{short}_{ax}"
    df_merged = df_merged.rename(columns=rename)
    return df_merged


def build(raw_dir: Path, out_dir: Path, test_fraction: float = 0.2) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    activity_map = _load_activity_key(raw_dir)
    print(f"Activity key: {activity_map}")

    streams = _load_all_streams(raw_dir)
    if len(streams) < 4:
        print("Warning: fewer than 4 streams found. Output may be incomplete.")

    print("\nMerging streams to 20 Hz grid (this may take a few minutes)...")
    df_all = _merge_streams(streams, activity_map)
    df_all["seg_id"] = 0
    print(f"Merged: {len(df_all):,} rows, subjects: {sorted(df_all['calf_id'].unique())}")
    print(f"Labels: {sorted(df_all['label'].unique())}")

    train_raw, test_raw, split_meta = split_by_gen_split(
        df_all, "calf_id", "label", test_fraction,
        existing_split_report=out_dir / "split_report.json",
    )
    known = set(train_raw["label"].unique())
    test_raw = test_raw[test_raw["label"].isin(known)].reset_index(drop=True)

    report = {
        "source": str(raw_dir),
        "split": split_meta,
        "train_rows": len(train_raw),
        "test_rows": len(test_raw),
        "labels": sorted(known),
        "channels": CHANNEL_COLS,
    }
    save_split_report(report, out_dir / "split_report.json")

    for split, df_split, fname in [
        ("train", train_raw, "windowed_train.parquet"),
        ("test",  test_raw,  "windowed_test.parquet"),
    ]:
        print(f"\nWindowing {split} ({len(df_split):,} rows)...")
        df_win = window_dataframe(
            df_split,
            channel_cols=CHANNEL_COLS,
            label_col="label",
            group_by=["calf_id", "seg_id"],
            subject_col="calf_id",
            time_col="dateTime",
            window_size=WINDOW_SIZE,
            stride=STRIDE,
            purity_threshold=PURITY,
        )
        print(f"  Computing HC features ({len(df_win)} windows, 12 channels)...")
        df_hc = compute_dataframe(df_win, CHANNEL_COLS, FS)
        df_final = pd.concat([df_win, df_hc], axis=1)
        df_final.to_parquet(out_dir / fname, index=False, compression="snappy")
        print(f"  Saved: {out_dir / fname}  shape={df_final.shape}")

    manifest = build_manifest(CHANNEL_COLS, FS, WINDOW_SIZE, STRIDE, PURITY, "WISDM")
    save_manifest(manifest, out_dir / "hc_manifest.json")


def main() -> None:
    p = argparse.ArgumentParser(description="WISDM → windowed parquet + HC features")
    p.add_argument("--raw", type=Path, required=True, help="dataset/raw/WISDM/wisdm-dataset/")
    p.add_argument("--out", type=Path, default=Path("dataset/processed/WISDM"))
    p.add_argument("--test-fraction", type=float, default=0.2)
    args = p.parse_args()
    build(args.raw, args.out, args.test_fraction)


if __name__ == "__main__":
    main()
