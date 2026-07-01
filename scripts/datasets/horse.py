#!/usr/bin/env python3
"""Horse dataset → windowed parquet + HC features.

Input:  dataset/raw/horse/csv/subject_<id>_<name>_part_<n>.csv  (long-format,
        100 Hz Ax..Gz + 12 Hz Mx..Mz, 'label' column, 'segment' = contiguous
        recording-block id, 'subject' = horse id)
Output: dataset/processed/horse/
          windowed_train.parquet
          windowed_test.parquet
          hc_manifest.json
          split_report.json

Each part-file is read and windowed independently (files are ~170 MB each;
the full raw dir is ~22 GB) to keep peak memory low. Magnetometer channels
(Mx,My,Mz @ 12 Hz, mostly NaN at the 100 Hz row rate) and the pre-computed
A3D/G3D/M3D magnitude columns are dropped — acc_x/y/z + gyr_x/y/z (6 native
100 Hz channels) are kept, matching vehkaoja's channel layout.
Rows labeled 'null' or 'unknown' (no behavior annotation) are discarded
before windowing.

Usage:
    python scripts/datasets/horse.py \\
        --raw dataset/raw/horse/csv \\
        --out dataset/processed/horse
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
from core.splits import split_by_gen_split, save_split_report

FS = 100            # Hz — Ax..Gz native rate (Mx..Mz @ 12 Hz dropped)
WINDOW_SIZE = 200    # 2 s @ 100 Hz (dataset's own settings.csv: window_size=200)
STRIDE = 100         # 50 % overlap (settings.csv: overlap=0.5)
PURITY = 0.9
CHANNEL_COLS = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]

_COL_MAP = {
    "Ax": "acc_x", "Ay": "acc_y", "Az": "acc_z",
    "Gx": "gyr_x", "Gy": "gyr_y", "Gz": "gyr_z",
}
_DROP_LABELS = {"null", "unknown", "nan"}


def _iter_part_files(raw_dir: Path):
    files = sorted(raw_dir.glob("subject_*_part_*.csv"))
    if not files:
        raise FileNotFoundError(f"No subject_*_part_*.csv files found in {raw_dir}")
    return files


def _load_and_window_one(csv_path: Path) -> pd.DataFrame:
    usecols = ["Ax", "Ay", "Az", "Gx", "Gy", "Gz", "label", "segment", "subject"]
    df = pd.read_csv(csv_path, usecols=usecols)
    df["label"] = df["label"].astype(str).str.strip().str.lower().str.replace("_", "-")
    df = df[~df["label"].isin(_DROP_LABELS)]
    if df.empty:
        return pd.DataFrame()

    df = df.rename(columns=_COL_MAP)
    df = df.dropna(subset=CHANNEL_COLS)
    if df.empty:
        return pd.DataFrame()

    df["t_idx"] = range(len(df))
    df["calf_id"] = df["subject"].astype(int)

    win = window_dataframe(
        df,
        channel_cols=CHANNEL_COLS,
        label_col="label",
        group_by=["calf_id", "segment"],
        subject_col="calf_id",
        time_col="t_idx",
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        purity_threshold=PURITY,
    )
    return win


def _load_all_windowed(raw_dir: Path) -> pd.DataFrame:
    frames = []
    for csv_path in _iter_part_files(raw_dir):
        print(f"  windowing {csv_path.name} ...")
        win = _load_and_window_one(csv_path)
        if not win.empty:
            frames.append(win)
    if not frames:
        raise RuntimeError("No valid windows produced from any part file.")
    df_win = pd.concat(frames, ignore_index=True)
    print(f"Total windows (all subjects): {len(df_win):,}")
    return df_win


def build(raw_dir: Path, out_dir: Path, test_fraction: float = 0.2) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    df_win = _load_all_windowed(raw_dir)

    train_win, test_win, split_meta = split_by_gen_split(
        df_win, "calf_id", "label", test_fraction,
        existing_split_report=out_dir / "split_report.json",
    )
    known = set(train_win["label"].unique())
    test_win = test_win[test_win["label"].isin(known)].reset_index(drop=True)

    report = {
        "source": str(raw_dir),
        "split": split_meta,
        "train_windows": len(train_win),
        "test_windows": len(test_win),
        "labels": sorted(known),
    }
    save_split_report(report, out_dir / "split_report.json")

    for split, df_split, fname in [
        ("train", train_win, "windowed_train.parquet"),
        ("test",  test_win,  "windowed_test.parquet"),
    ]:
        print(f"\nComputing HC features for {split} ({len(df_split)} windows)...")
        df_hc = compute_dataframe(df_split, CHANNEL_COLS, FS)
        df_final = pd.concat([df_split, df_hc], axis=1)
        df_final.to_parquet(out_dir / fname, index=False, compression="snappy")
        print(f"  Saved: {out_dir / fname}  shape={df_final.shape}")

    manifest = build_manifest(CHANNEL_COLS, FS, WINDOW_SIZE, STRIDE, PURITY, "horse")
    save_manifest(manifest, out_dir / "hc_manifest.json")


def main() -> None:
    p = argparse.ArgumentParser(description="Horse → windowed parquet + HC features")
    p.add_argument("--raw", type=Path, required=True, help="dataset/raw/horse/csv/")
    p.add_argument("--out", type=Path, default=Path("dataset/processed/horse"))
    p.add_argument("--test-fraction", type=float, default=0.2)
    args = p.parse_args()
    build(args.raw, args.out, args.test_fraction)


if __name__ == "__main__":
    main()
