"""Utilities for exporting windowed data to SSL pretraining formats.

Two functions:
  window_raw_csv  — long raw CSV → unlabeled windowed parquet (for TS2Vec / MAE).
  export_to_pt    — windowed parquet → .pt bundles for the pretrain_ablations framework.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------------------------------
# Raw CSV → unlabeled windowed parquet
# ---------------------------------------------------------------------------

def window_raw_csv(
    input_path: str | Path,
    output_path: str | Path,
    channel_cols: list[str],
    raw_col_map: dict[str, str],
    window_len: int = 75,
    stride: int = 37,
    chunksize: int = 2_000_000,
) -> None:
    """
    Slide a window over a long raw CSV (no labels) and write a parquet.

    Args:
        input_path: path to the raw CSV.
        output_path: output parquet path.
        channel_cols: output column names (e.g. ["acc_x","acc_y","acc_z"]).
        raw_col_map: maps output col names → CSV column names
                     (e.g. {"acc_x": "accX", "acc_y": "accY", "acc_z": "accZ"}).
        window_len: samples per window.
        stride: step in samples.
        chunksize: rows per chunk for large files.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    raw_cols = [raw_col_map[c] for c in channel_cols]

    chunks = []
    leftover = pd.DataFrame(columns=raw_cols)
    total_rows = 0

    for i, chunk in enumerate(pd.read_csv(input_path, chunksize=chunksize, usecols=raw_cols)):
        total_rows += len(chunk)
        chunk = pd.concat([leftover, chunk], ignore_index=True)
        data = chunk[raw_cols].to_numpy(dtype=np.float32)
        N = len(data)
        windows = []
        for s in range(0, N - window_len + 1, stride):
            row = {col: data[s:s + window_len, idx].tolist()
                   for idx, col in enumerate(channel_cols)}
            windows.append(row)
        if windows:
            chunks.append(pd.DataFrame(windows))
        leftover = chunk.iloc[-(window_len - 1):].reset_index(drop=True)
        print(f"  chunk {i}: rows_so_far={total_rows} | windows_in_chunk={len(windows)}")

    df_out = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=channel_cols)
    df_out.to_parquet(out, index=False)
    print(f"Raw windowed parquet: {len(df_out)} windows → {out}")


# ---------------------------------------------------------------------------
# Windowed parquet → .pt for pretrain_ablations framework
# ---------------------------------------------------------------------------

def _stack_signals(df: pd.DataFrame, channels: list[str]) -> np.ndarray:
    return np.stack([np.stack(df[c].values) for c in channels], axis=1).astype(np.float32)


# Metadata columns that are never signal channels nor hand-crafted features.
_META_COLS = frozenset(
    {"dateTime", "calfId", "calf_id", "segId", "seg_id", "subject_id", "subject", "label"}
)


def _feature_columns(df: pd.DataFrame, channel_cols: list[str]) -> list[str]:
    """Scalar hand-crafted feature columns: everything that is neither a signal
    channel, a metadata column, nor the label. Mirrors the hybrid dataloader's
    `_feature_columns` so the .pt features match what the parquet path would produce."""
    sig = frozenset(channel_cols)
    return [c for c in df.columns if c not in _META_COLS and c not in sig]


def _stack_features(df: pd.DataFrame, feat_cols: list[str]) -> np.ndarray:
    if not feat_cols:
        return np.zeros((len(df), 0), dtype=np.float32)
    arr = df[feat_cols].to_numpy(dtype=np.float32)
    return np.nan_to_num(arr, nan=0.0)


def export_to_pt(
    train_parquet: str | Path,
    test_parquet: str | Path,
    output_dir: str | Path,
    channel_cols: list[str],
    dataset_id: str,
    sampling_hz: float | None = None,
    val_parquet: str | Path | None = None,
    val_size: float = 0.1,
    subject_col: str = "calf_id",
    random_state: int = 42,
) -> None:
    """
    Convert windowed parquets to .pt bundles consumed by the pretrain_ablations framework.

    Writes: {output_dir}/{train,val,test}.pt and label2id.json. Each .pt also carries a
    `features` tensor (N, F) of the parquet's scalar hand-crafted feature columns, row-aligned to
    `samples`, plus `feature_names`, so the hybrid model can fuse signals + hand-crafted
    features from the same canonical tensors the ablation uses.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df_tr = pd.read_parquet(train_parquet)
    df_te = pd.read_parquet(test_parquet)

    le = LabelEncoder().fit(df_tr["label"].astype(str))
    known = set(le.classes_)
    df_te = df_te[df_te["label"].astype(str).isin(known)].reset_index(drop=True)

    if val_parquet is not None:
        df_vl = pd.read_parquet(val_parquet)
        df_vl = df_vl[df_vl["label"].astype(str).isin(known)].reset_index(drop=True)
    else:
        gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
        groups = df_tr[subject_col].to_numpy()
        tr_idx, vl_idx = next(gss.split(df_tr, groups=groups))
        df_vl = df_tr.iloc[vl_idx].reset_index(drop=True)
        df_tr = df_tr.iloc[tr_idx].reset_index(drop=True)

    # Hand-crafted feature columns, fixed from the train parquet so val/test align
    # to the same feature set (missing cols in a split → 0 via _align below).
    feat_cols = _feature_columns(df_tr, channel_cols)

    label2id = {cls: int(i) for i, cls in enumerate(le.classes_)}
    id2label = {str(i): cls for i, cls in enumerate(le.classes_)}
    (out / "label2id.json").write_text(json.dumps(label2id, indent=2))

    # Build a stable subject→int mapping from the union of all splits so that
    # `groups` is int64 even when subject ids are strings (e.g. marinara).
    all_subjects = pd.concat(
        [df_tr[subject_col], df_vl[subject_col], df_te[subject_col]]
    ).astype(str)
    subject_codes = {s: i for i, s in enumerate(sorted(all_subjects.unique()))}

    meta = {
        "label2id": label2id,
        "id2label": id2label,
        "channel_names": channel_cols,
        "num_channels": len(channel_cols),
        "dataset_id": dataset_id,
        "sampling_hz": sampling_hz,
        "split_policy": "subject_held_out",
        "feature_names": feat_cols,
        "num_features": len(feat_cols),
    }

    for name, df in (("train", df_tr), ("val", df_vl), ("test", df_te)):
        X = _stack_signals(df, channel_cols)
        # Align feature cols to the train set (cols absent in this split → 0).
        df_feat = df.reindex(columns=feat_cols, fill_value=0.0)
        F = _stack_features(df_feat, feat_cols)
        y = le.transform(df["label"].astype(str)).astype(np.int64)
        g = df[subject_col].astype(str).map(subject_codes).to_numpy(dtype=np.int64)
        dist = Counter(y.tolist())
        print(
            f"  [{name}] sig={X.shape} feat={F.shape} "
            + ", ".join(f"{le.classes_[k]}:{v}" for k, v in sorted(dist.items()))
        )
        torch.save(
            {"samples": torch.tensor(X), "labels": torch.tensor(y),
             "groups": torch.tensor(g), "features": torch.tensor(F),
             "feature_names": feat_cols, "meta": {**meta, "split": name}},
            out / f"{name}.pt",
        )
    print(f"Done: {out}")
