#!/usr/bin/env python3
"""Generic windowed-parquet -> canonical .pt exporter for the pretrain_ablations framework.

Reads ``dataset/processed/<NAME>/windowed_{train,test}.parquet`` (uniform list-column
schema produced by scripts/datasets/*.py), auto-detects the signal channels, carves a
group-disjoint validation split off the train set, and writes
``dataset/processed/<dataset_id>/{train,val,test}.pt`` in the canonical format
expected by datasets/loader.py.

Idempotent: if the three .pt files already exist it does nothing (unless --force).

Usage:
    python scripts/datasets/export_windowed.py \\
        --processed-dir dataset/processed/AcTBeCalf \\
        --dataset-id actbecalf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# scripts/core/pretrain_export.py provides the corrected export_to_pt.
_REPO = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from core.pretrain_export import export_to_pt  # noqa: E402

# Metadata columns that are never signal channels.
_META_COLS = frozenset({"dateTime", "calfId", "calf_id", "segId", "seg_id",
                        "subject_id", "subject", "label"})
_SUBJECT_CANDIDATES = ("calf_id", "calfId", "subject_id", "subject")


def detect_channels(df: pd.DataFrame) -> list[str]:
    """List columns whose cells are array-like (the windowed signal channels)."""
    return [c for c in df.columns
            if c not in _META_COLS and hasattr(df[c].iloc[0], "__len__")]


def detect_subject_col(df: pd.DataFrame) -> str:
    for c in _SUBJECT_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(f"No subject column found. Tried {_SUBJECT_CANDIDATES}. "
                     f"Columns: {list(df.columns)}")


def main() -> None:
    p = argparse.ArgumentParser(description="Windowed parquet -> canonical .pt")
    p.add_argument("--processed-dir", type=Path, required=True,
                   help="Dir with windowed_train.parquet / windowed_test.parquet")
    p.add_argument("--dataset-id", type=str, required=True,
                   help="Registry id; output goes to dataset/processed/<id>/")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Override output dir (default: dataset/processed/<id>)")
    p.add_argument("--sampling-hz", type=float, default=None)
    p.add_argument("--val-size", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true", help="Re-export even if .pt exist")
    args = p.parse_args()

    out_dir = args.out_dir or (_REPO / "dataset" / "processed" / args.dataset_id)
    train_pt = out_dir / "train.pt"
    val_pt = out_dir / "val.pt"
    test_pt = out_dir / "test.pt"
    if not args.force and train_pt.exists() and val_pt.exists() and test_pt.exists():
        print(f"[skip] {args.dataset_id}: .pt already present in {out_dir}")
        return

    train_parquet = args.processed_dir / "windowed_train.parquet"
    test_parquet = args.processed_dir / "windowed_test.parquet"
    for f in (train_parquet, test_parquet):
        if not f.exists():
            raise FileNotFoundError(f"Missing windowed parquet: {f}")

    # Read just the first row group to detect channels/subject column cheaply.
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(train_parquet)
    df_head = pf.read_row_group(0).slice(0, 1).to_pandas()
    channels = detect_channels(df_head)
    subject_col = detect_subject_col(df_head)
    if not channels:
        raise ValueError(f"No signal channels detected in {train_parquet}")
    print(f"[{args.dataset_id}] channels={channels} subject_col={subject_col!r}")

    export_to_pt(
        train_parquet=train_parquet,
        test_parquet=test_parquet,
        output_dir=out_dir,
        channel_cols=channels,
        dataset_id=args.dataset_id,
        sampling_hz=args.sampling_hz,
        val_size=args.val_size,
        subject_col=subject_col,
        random_state=args.seed,
    )


if __name__ == "__main__":
    main()
