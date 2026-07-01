#!/usr/bin/env python3
"""UCI HAR human activity dataset → windowed parquet + HC features.

Input:  dataset/raw/UCI/UCI HAR Dataset/   (already unzipped)
Output: dataset/processed/HAR_UCI/
          windowed_train.parquet   (signal columns match existing HAR_UCI output)
          windowed_test.parquet
          hc_manifest.json

Uses the body_acc_x/y/z Inertial Signals (128 samples, pre-windowed by UCI).

Usage:
    python scripts/datasets/uci_har.py \\
        --raw "dataset/raw/UCI/UCI HAR Dataset" \\
        --out dataset/processed/HAR_UCI
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

from core.hc_features import compute_dataframe
from core.manifest import build_manifest, save_manifest

FS = 50
WINDOW_SIZE = 128
STRIDE = 64          # not used for windowing (pre-windowed), stored in manifest
PURITY = 1.0
CHANNEL_COLS = ["acc_x", "acc_y", "acc_z"]

_ACTIVITIES = {
    1: "WALKING", 2: "WALKING_UPSTAIRS", 3: "WALKING_DOWNSTAIRS",
    4: "SITTING", 5: "STANDING", 6: "LAYING",
}


def _load_split(root: Path, split: str) -> pd.DataFrame:
    split_dir = root / split
    inertial = split_dir / "Inertial Signals"

    x = np.loadtxt(inertial / f"body_acc_x_{split}.txt")
    y = np.loadtxt(inertial / f"body_acc_y_{split}.txt")
    z = np.loadtxt(inertial / f"body_acc_z_{split}.txt")
    labels_raw = np.loadtxt(split_dir / f"y_{split}.txt", dtype=int)
    subjects = np.loadtxt(split_dir / f"subject_{split}.txt", dtype=int)

    rows = []
    for i in range(len(x)):
        rows.append({
            "dateTime": i,
            "calf_id": int(subjects[i]),
            "acc_x": x[i].tolist(),
            "acc_y": y[i].tolist(),
            "acc_z": z[i].tolist(),
            "label": _ACTIVITIES[int(labels_raw[i])],
        })
    return pd.DataFrame(rows)


def build(raw_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for split, fname in [("train", "windowed_train.parquet"), ("test", "windowed_test.parquet")]:
        print(f"Loading UCI HAR {split}...")
        df_win = _load_split(raw_dir, split)
        print(f"  {len(df_win)} windows, subjects: {sorted(df_win['calf_id'].unique())}")
        df_hc = compute_dataframe(df_win, CHANNEL_COLS, FS)
        df_final = pd.concat([df_win, df_hc], axis=1)
        df_final.to_parquet(out_dir / fname, index=False, compression="snappy")
        print(f"  Saved: {out_dir / fname}  shape={df_final.shape}")

    manifest = build_manifest(CHANNEL_COLS, FS, WINDOW_SIZE, STRIDE, PURITY, "UCI_HAR")
    save_manifest(manifest, out_dir / "hc_manifest.json")


def main() -> None:
    p = argparse.ArgumentParser(description="UCI HAR → windowed parquet + HC features")
    p.add_argument("--raw", type=Path, required=True,
                   help='Path to "UCI HAR Dataset" directory (unzipped)')
    p.add_argument("--out", type=Path, default=Path("dataset/processed/HAR_UCI"))
    args = p.parse_args()
    build(args.raw, args.out)


if __name__ == "__main__":
    main()
