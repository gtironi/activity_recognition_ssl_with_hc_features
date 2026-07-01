#!/usr/bin/env python3
"""AcTBeCalf cattle calf dataset → windowed parquet + HC features.

Input:  dataset/raw/AcTBeCalf/AcTBeCalf.csv
Output: dataset/processed/AcTBeCalf/
          windowed_train.parquet
          windowed_test.parquet
          hc_manifest.json
          split_report.json

Signal columns match the existing processed output exactly (acc_x, acc_y, acc_z).

Usage:
    python scripts/datasets/actbecalf.py \\
        --raw dataset/raw/AcTBeCalf \\
        --out dataset/processed/AcTBeCalf
"""

from __future__ import annotations

import argparse
import json
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
from core.splits import (
    split_by_subject_list,
    split_by_gen_split,
    save_split_report,
)

# ---------------------------------------------------------------------------
# Canonical label map (behaviour strings → class names)
# ---------------------------------------------------------------------------

BEHAVIOUR_LABEL_MAP: dict[str, list[str]] = {
    "Standing": ["standing"],
    "Lying": ["lying", "lying-down"],
    "Drinking": ["drinking", "drinking_milk", "drinking_electrolytes", "drinking|water"],
    "Eating": ["eating", "eating_concentrates", "eating_bedding", "eating_forage"],
    "Walking": ["walking", "backward"],
    "Run": ["running"],
    "Grooming": ["grooming", "grooming_lying", "grooming|None"],
    "Social Interaction": [
        "social", "social_sniff", "social_sniff_lying", "social_groom",
        "social_groom_lying", "social_nudge", "social_nudge_lying",
    ],
    "Play": ["play", "play_object", "headbutt", "jump", "mount"],
    "Rising": ["rising"],
    "Rumination": ["rumination", "rumination_lying"],
    "Defecation": ["defecation"],
    "Urination": ["urination"],
    "Oral manipulation of pen": ["oral_manipulation_of_pen"],
    "Sniff": ["sniff", "sniff_walking", "sniff_lying"],
    "Abnormal": ["abnormal", "cross-suckle_udder", "cross-suckle_other",
                 "tongue_rolling", "tongue_rolling_lying"],
    "SRS": ["SRS", "scratch", "rub", "stretch"],
    "Cough": ["cough"],
    "Fall": ["fall"],
    "Vocalization": ["vocalization"],
}

_RAW_TO_CANONICAL: dict[str, str] = {
    str(raw).lower().strip(): canonical
    for canonical, raw_list in BEHAVIOUR_LABEL_MAP.items()
    for raw in raw_list
}

DEFAULT_TEST_SUBJECTS = (1329, 1343, 1353, 1357, 1372)
FS = 25
WINDOW_SIZE = 75
STRIDE = 37          # ~50 % overlap
PURITY = 0.9
MIN_TRAIN_PROP = 0.01
CHANNEL_COLS = ["acc_x", "acc_y", "acc_z"]


def _apply_label_map(df: pd.DataFrame, col: str) -> pd.DataFrame:
    norm = df[col].astype(str).str.lower().str.strip()
    df[col] = norm.map(_RAW_TO_CANONICAL).fillna("Other")
    return df


def _filter_rare(
    train: pd.DataFrame, test: pd.DataFrame, col: str, min_prop: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = train[col].value_counts()
    total = counts.sum()
    rare = set(counts[counts / total < min_prop].index)
    only_test = set(test[col].unique()) - set(train[col].unique())
    drop = rare | only_test
    if drop:
        print(f"Dropping rare/test-only classes: {sorted(drop)}")
        train = train[~train[col].isin(drop)].reset_index(drop=True)
        test = test[~test[col].isin(drop)].reset_index(drop=True)
    return train, test


def _to_parquet(df_raw: pd.DataFrame, col_map: dict) -> pd.DataFrame:
    """Convert raw CSV columns to the format expected by window_dataframe."""
    return df_raw.rename(columns=col_map)


def build(
    raw_dir: Path,
    out_dir: Path,
    split_by: str = "subject",
    test_subjects=DEFAULT_TEST_SUBJECTS,
    test_fraction: float = 0.2,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv = raw_dir / "AcTBeCalf.csv"
    print(f"Reading {csv} ...")
    df = pd.read_csv(csv)
    print(f"  {len(df):,} rows, columns: {list(df.columns)}")

    # Map raw column names to canonical names
    df = df.rename(columns={"accX": "acc_x", "accY": "acc_y", "accZ": "acc_z"})
    df = _apply_label_map(df, "behaviour")

    if split_by == "subject":
        train_raw, test_raw = split_by_subject_list(df, "calfId", test_subjects)
        split_meta = {
            "method": "fixed_subject_list",
            "test_subject_ids": sorted(int(x) for x in test_subjects),
        }
    else:
        train_raw, test_raw, split_meta = split_by_gen_split(
            df, "calfId", "behaviour", test_fraction,
            existing_split_report=out_dir / "split_report.json",
        )

    train_raw, test_raw = _filter_rare(train_raw, test_raw, "behaviour", MIN_TRAIN_PROP)
    # Ensure test only has labels seen in train
    known = set(train_raw["behaviour"].unique())
    test_raw = test_raw[test_raw["behaviour"].isin(known)].reset_index(drop=True)

    report = {
        "source": str(csv),
        "split": split_meta,
        "train_rows": len(train_raw),
        "test_rows": len(test_raw),
        "train_labels": sorted(train_raw["behaviour"].unique().tolist()),
    }
    save_split_report(report, out_dir / "split_report.json")

    # Save pre-windowed train/test for reference
    train_raw.to_parquet(out_dir / "train.parquet", index=False)
    test_raw.to_parquet(out_dir / "test.parquet", index=False)

    for split, df_split, fname in [
        ("train", train_raw, "windowed_train.parquet"),
        ("test", test_raw, "windowed_test.parquet"),
    ]:
        print(f"\nWindowing {split} ({len(df_split):,} rows)...")
        df_win = window_dataframe(
            df_split,
            channel_cols=CHANNEL_COLS,
            label_col="behaviour",
            group_by=["calfId", "segId"],
            subject_col="calfId",
            time_col="dateTime",
            window_size=WINDOW_SIZE,
            stride=STRIDE,
            purity_threshold=PURITY,
        )
        print(f"  Computing HC features ({len(df_win)} windows)...")
        df_hc = compute_dataframe(df_win, CHANNEL_COLS, FS)
        df_final = pd.concat([df_win, df_hc], axis=1)
        df_final.to_parquet(out_dir / fname, index=False, compression="snappy")
        print(f"  Saved: {out_dir / fname}  shape={df_final.shape}")

    manifest = build_manifest(CHANNEL_COLS, FS, WINDOW_SIZE, STRIDE, PURITY, "AcTBeCalf")
    save_manifest(manifest, out_dir / "hc_manifest.json")


def main() -> None:
    p = argparse.ArgumentParser(description="AcTBeCalf → windowed parquet + HC features")
    p.add_argument("--raw", type=Path, required=True, help="dataset/raw/AcTBeCalf/")
    p.add_argument("--out", type=Path, default=Path("dataset/processed/AcTBeCalf"))
    p.add_argument("--split-by", choices=["subject", "behavior"], default="subject")
    p.add_argument("--test-subjects", nargs="+", type=int, default=list(DEFAULT_TEST_SUBJECTS))
    p.add_argument("--test-fraction", type=float, default=0.2)
    args = p.parse_args()
    build(args.raw, args.out, args.split_by, args.test_subjects, args.test_fraction)


if __name__ == "__main__":
    main()
