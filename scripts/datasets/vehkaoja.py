#!/usr/bin/env python3
"""Vehkaoja dog dataset → windowed parquet + HC features.

Input:  dataset/raw/vehkaoja/{train.csv, test.csv}
Output: dataset/processed/vehkaoja/
          windowed_train.parquet   (signal cols match existing dog_raw output)
          windowed_test.parquet
          hc_manifest.json
          split_report.json

Raw columns: t_sec, Subject, TestNum, Back_Acc_{X,Y,Z}, Neck_Acc_{X,Y,Z},
             Back_Gyr_{X,Y,Z}, Neck_Gyr_{X,Y,Z}, Label

Channel mapping (preserves existing dog_raw schema):
  acc_x = Back_Acc_X   acc_y = Back_Acc_Y   acc_z = Back_Acc_Z
  gyr_x = Neck_Acc_X   gyr_y = Neck_Acc_Y   gyr_z = Neck_Acc_Z

Usage:
    python scripts/datasets/vehkaoja.py \\
        --raw dataset/raw/vehkaoja \\
        --out dataset/processed/vehkaoja
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from core.windowing import window_dataframe_strict
from core.hc_features import compute_dataframe
from core.manifest import build_manifest, save_manifest
from core.splits import carve_val_by_subject, save_split_report

FS = 100          # 100 Hz (ActiGraph GT9X Link, confirmed from dataset paper)
WINDOW_SIZE = 300  # 3 s @ 100 Hz
STRIDE = 150       # 50 % overlap
CHANNEL_COLS = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]

_COL_MAP = {
    "Back_Acc_X": "acc_x",
    "Back_Acc_Y": "acc_y",
    "Back_Acc_Z": "acc_z",
    "Neck_Acc_X": "gyr_x",
    "Neck_Acc_Y": "gyr_y",
    "Neck_Acc_Z": "gyr_z",
}


def _load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.rename(columns=_COL_MAP)
    df = df.rename(columns={"Label": "label", "Subject": "calf_id"})
    df["t_idx"] = range(len(df))   # surrogate time column
    return df


def build(raw_dir: Path, out_dir: Path, val_size: float = 0.1) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    train_raw = _load(raw_dir / "train.csv")
    test_raw = _load(raw_dir / "test.csv")

    print(f"Train: {len(train_raw):,} rows | Test: {len(test_raw):,} rows")

    known = set(train_raw["label"].unique())
    test_raw = test_raw[test_raw["label"].isin(known)].reset_index(drop=True)

    report = {
        "source": str(raw_dir),
        "train_subjects": sorted(train_raw["calf_id"].unique().tolist()),
        "test_subjects": sorted(test_raw["calf_id"].unique().tolist()),
        "train_rows": len(train_raw),
        "test_rows": len(test_raw),
        "labels": sorted(known),
    }
    save_split_report(report, out_dir / "split_report.json")

    for split, df_split, fname in [
        ("train", train_raw, "windowed_train.parquet"),
        ("test", test_raw, "windowed_test.parquet"),
    ]:
        print(f"\nWindowing {split} ({len(df_split):,} rows, strict purity)...")
        df_win = window_dataframe_strict(
            df_split,
            channel_cols=CHANNEL_COLS,
            label_col="label",
            group_by=["calf_id", "TestNum"],
            subject_col="calf_id",
            time_col="t_idx",
            window_size=WINDOW_SIZE,
            stride=STRIDE,
        )
        print(f"  Computing HC features ({len(df_win)} windows)...")
        df_hc = compute_dataframe(df_win, CHANNEL_COLS, FS)
        df_final = pd.concat([df_win, df_hc], axis=1)
        df_final.to_parquet(out_dir / fname, index=False, compression="snappy")
        print(f"  Saved: {out_dir / fname}  shape={df_final.shape}")

    manifest = build_manifest(CHANNEL_COLS, FS, WINDOW_SIZE, STRIDE, 1.0, "vehkaoja")
    save_manifest(manifest, out_dir / "hc_manifest.json")


def main() -> None:
    p = argparse.ArgumentParser(description="Vehkaoja dog → windowed parquet + HC features")
    p.add_argument("--raw", type=Path, required=True, help="dataset/raw/vehkaoja/")
    p.add_argument("--out", type=Path, default=Path("dataset/processed/vehkaoja"))
    p.add_argument("--val-size", type=float, default=0.1)
    args = p.parse_args()
    build(args.raw, args.out, args.val_size)


if __name__ == "__main__":
    main()
