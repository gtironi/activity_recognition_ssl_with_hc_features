#!/usr/bin/env python3
"""Marinara dog dataset → windowed parquet + HC features.

Input:  dataset/raw/marinara/{train.csv, test.csv}
Output: dataset/processed/marinara/
          windowed_train.parquet
          windowed_test.parquet
          hc_manifest.json
          split_report.json

27 native channels: Back/Chest/Neck × Acc/Gyr/Mag (XYZ each).
Type and Breed columns are dropped.

Usage:
    python scripts/datasets/marinara.py \\
        --raw dataset/raw/marinara \\
        --out dataset/processed/marinara
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from core.windowing import window_dataframe
from core.hc_features import compute_dataframe
from core.manifest import build_manifest, save_manifest
from core.splits import save_split_report

FS = 100          # 100 Hz (ActiGraph GT9X Link, same hardware as vehkaoja)
WINDOW_SIZE = 300  # 3 s @ 100 Hz
STRIDE = 150       # 50 % overlap
PURITY = 0.9

# All 27 sensor channels — acc_x/y/z must be first three for SMA/ODBA
CHANNEL_COLS = [
    "acc_x",   "acc_y",   "acc_z",
    "gyr_x",   "gyr_y",   "gyr_z",
    "mag_x",   "mag_y",   "mag_z",
    "chest_acc_x", "chest_acc_y", "chest_acc_z",
    "chest_gyr_x", "chest_gyr_y", "chest_gyr_z",
    "chest_mag_x", "chest_mag_y", "chest_mag_z",
    "neck_acc_x",  "neck_acc_y",  "neck_acc_z",
    "neck_gyr_x",  "neck_gyr_y",  "neck_gyr_z",
    "neck_mag_x",  "neck_mag_y",  "neck_mag_z",
]

_COL_MAP = {
    "Back_Acc_X": "acc_x",    "Back_Acc_Y": "acc_y",    "Back_Acc_Z": "acc_z",
    "Back_Gyr_X": "gyr_x",    "Back_Gyr_Y": "gyr_y",    "Back_Gyr_Z": "gyr_z",
    "Back_Mag_X": "mag_x",    "Back_Mag_Y": "mag_y",    "Back_Mag_Z": "mag_z",
    "Chest_Acc_X": "chest_acc_x", "Chest_Acc_Y": "chest_acc_y", "Chest_Acc_Z": "chest_acc_z",
    "Chest_Gyr_X": "chest_gyr_x", "Chest_Gyr_Y": "chest_gyr_y", "Chest_Gyr_Z": "chest_gyr_z",
    "Chest_Mag_X": "chest_mag_x", "Chest_Mag_Y": "chest_mag_y", "Chest_Mag_Z": "chest_mag_z",
    "Neck_Acc_X": "neck_acc_x",   "Neck_Acc_Y": "neck_acc_y",   "Neck_Acc_Z": "neck_acc_z",
    "Neck_Gyr_X": "neck_gyr_x",   "Neck_Gyr_Y": "neck_gyr_y",   "Neck_Gyr_Z": "neck_gyr_z",
    "Neck_Mag_X": "neck_mag_x",   "Neck_Mag_Y": "neck_mag_y",   "Neck_Mag_Z": "neck_mag_z",
    "Label": "label",
    "Subject": "calf_id",
}


def _load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.drop(columns=[c for c in ("Type", "Breed") if c in df.columns])
    df = df.rename(columns=_COL_MAP)
    df["t_idx"] = range(len(df))
    df["seg_id"] = 0
    return df


def build(raw_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    train_raw = _load(raw_dir / "train.csv")
    test_raw  = _load(raw_dir / "test.csv")

    known = set(train_raw["label"].unique())
    test_raw = test_raw[test_raw["label"].isin(known)].reset_index(drop=True)

    report = {
        "source": str(raw_dir),
        "train_subjects": sorted(str(s) for s in train_raw["calf_id"].unique()),
        "test_subjects":  sorted(str(s) for s in test_raw["calf_id"].unique()),
        "train_rows": len(train_raw),
        "test_rows":  len(test_raw),
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
            time_col="t_idx",
            window_size=WINDOW_SIZE,
            stride=STRIDE,
            purity_threshold=PURITY,
        )
        print(f"  Computing HC features ({len(df_win)} windows, 27 channels → 326 features)...")
        df_hc = compute_dataframe(df_win, CHANNEL_COLS, FS)
        df_final = pd.concat([df_win, df_hc], axis=1)
        df_final.to_parquet(out_dir / fname, index=False, compression="snappy")
        print(f"  Saved: {out_dir / fname}  shape={df_final.shape}")

    manifest = build_manifest(CHANNEL_COLS, FS, WINDOW_SIZE, STRIDE, PURITY, "marinara")
    save_manifest(manifest, out_dir / "hc_manifest.json")


def main() -> None:
    p = argparse.ArgumentParser(description="Marinara dog → windowed parquet + HC features")
    p.add_argument("--raw", type=Path, required=True, help="dataset/raw/marinara/")
    p.add_argument("--out", type=Path, default=Path("dataset/processed/marinara"))
    args = p.parse_args()
    build(args.raw, args.out)


if __name__ == "__main__":
    main()
