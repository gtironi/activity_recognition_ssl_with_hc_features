#!/usr/bin/env python3
"""PAMAP2 human activity dataset → windowed parquet + HC features.

Input:  dataset/raw/PAMAP/PAMAP2_Dataset/Protocol/subject10{1..9}.dat
Output: dataset/processed/PAMAP2/
          windowed_train.parquet
          windowed_test.parquet
          hc_manifest.json
          split_report.json

54-column space-separated .dat files (no header). Layout:
  col 0:  timestamp (s)
  col 1:  activityID  (0 = transient/unlabeled, drop)
  col 2:  heart rate (drop)
  cols 3..19:   hand IMU     (temp, acc16g×3, acc6g×3, gyro×3, mag×3, orientation×4[invalid])
  cols 20..36:  chest IMU    (same layout)
  cols 37..53:  ankle IMU    (same layout)

We keep per-IMU acc16g (3) + gyro (3) + mag (3) = 9 channels × 3 IMUs = 27 channels.
Orientation columns (4 per IMU) are dropped. Heart rate and temp are dropped.

Usage:
    python scripts/datasets/pamap2.py \\
        --raw dataset/raw/PAMAP/PAMAP2_Dataset \\
        --out dataset/processed/PAMAP2
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

FS = 100
WINDOW_SIZE = 300   # 3 s @ 100 Hz
STRIDE = 150        # 50 % overlap
PURITY = 0.9

# Channel layout — acc16g+gyro+mag for each of hand/chest/ankle
CHANNEL_COLS = [
    "hand_acc_x",  "hand_acc_y",  "hand_acc_z",
    "hand_gyr_x",  "hand_gyr_y",  "hand_gyr_z",
    "hand_mag_x",  "hand_mag_y",  "hand_mag_z",
    "chest_acc_x", "chest_acc_y", "chest_acc_z",
    "chest_gyr_x", "chest_gyr_y", "chest_gyr_z",
    "chest_mag_x", "chest_mag_y", "chest_mag_z",
    "ankle_acc_x", "ankle_acc_y", "ankle_acc_z",
    "ankle_gyr_x", "ankle_gyr_y", "ankle_gyr_z",
    "ankle_mag_x", "ankle_mag_y", "ankle_mag_z",
]

# Activity ID → name (from PAMAP2 readme)
ACTIVITY_MAP = {
    1: "lying", 2: "sitting", 3: "standing", 4: "walking", 5: "running",
    6: "cycling", 7: "nordic_walking", 9: "watching_tv", 10: "computer_work",
    11: "car_driving", 12: "ascending_stairs", 13: "descending_stairs",
    14: "vacuum_cleaning", 15: "ironing", 16: "folding_laundry",
    17: "house_cleaning", 18: "playing_soccer", 19: "rope_jumping",
    24: "cricket_bowling", 25: "cricket_batting", 26: "cricket_fielding",
}

# Column indices for each IMU block (relative to block start):
#   0=temp, 1-3=acc16g, 4-6=acc6g, 7-9=gyro, 10-12=mag, 13-16=orient(invalid)
_IMU_BLOCK_STARTS = [3, 20, 37]   # hand, chest, ankle
_ACC16_OFF = slice(1, 4)
_GYR_OFF   = slice(7, 10)
_MAG_OFF   = slice(10, 13)


def _extract_imu_channels(row: np.ndarray, block_start: int) -> list[float]:
    b = block_start
    acc = row[b + 1: b + 4].tolist()
    gyr = row[b + 7: b + 10].tolist()
    mag = row[b + 10: b + 13].tolist()
    return acc + gyr + mag   # 9 values


def _load_subject(dat_path: Path, subject_id: int) -> pd.DataFrame:
    data = np.genfromtxt(dat_path, delimiter=" ", filling_values=np.nan)
    rows = []
    for row in data:
        act_id = int(row[1]) if not np.isnan(row[1]) else 0
        if act_id == 0 or act_id not in ACTIVITY_MAP:
            continue
        channels: list[float] = []
        for bstart in _IMU_BLOCK_STARTS:
            channels.extend(_extract_imu_channels(row, bstart))
        if any(np.isnan(v) for v in channels):
            continue    # drop rows with NaN after strict extraction
        rows.append([row[0], subject_id, ACTIVITY_MAP[act_id]] + channels)

    cols = ["dateTime", "calf_id", "label"] + CHANNEL_COLS
    return pd.DataFrame(rows, columns=cols)


def build(raw_dir: Path, out_dir: Path, test_fraction: float = 0.2) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    protocol_dir = raw_dir / "Protocol"

    frames = []
    for i in range(1, 10):
        dat = protocol_dir / f"subject10{i}.dat"
        if not dat.exists():
            print(f"  Warning: {dat} not found, skipping.")
            continue
        print(f"  Loading {dat.name}...")
        df = _load_subject(dat, i)
        print(f"    {len(df):,} labeled rows, activities: {sorted(df['label'].unique())}")
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No subject*.dat files found in {protocol_dir}")

    df_all = pd.concat(frames, ignore_index=True)
    df_all["seg_id"] = 0
    print(f"\nTotal: {len(df_all):,} rows, {df_all['calf_id'].nunique()} subjects")

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
        print(f"  Computing HC features ({len(df_win)} windows, 27 channels)...")
        df_hc = compute_dataframe(df_win, CHANNEL_COLS, FS)
        df_final = pd.concat([df_win, df_hc], axis=1)
        df_final.to_parquet(out_dir / fname, index=False, compression="snappy")
        print(f"  Saved: {out_dir / fname}  shape={df_final.shape}")

    manifest = build_manifest(CHANNEL_COLS, FS, WINDOW_SIZE, STRIDE, PURITY, "PAMAP2")
    save_manifest(manifest, out_dir / "hc_manifest.json")


def main() -> None:
    p = argparse.ArgumentParser(description="PAMAP2 → windowed parquet + HC features")
    p.add_argument("--raw", type=Path, required=True, help="dataset/raw/PAMAP/PAMAP2_Dataset/")
    p.add_argument("--out", type=Path, default=Path("dataset/processed/PAMAP2"))
    p.add_argument("--test-fraction", type=float, default=0.2)
    args = p.parse_args()
    build(args.raw, args.out, args.test_fraction)


if __name__ == "__main__":
    main()
