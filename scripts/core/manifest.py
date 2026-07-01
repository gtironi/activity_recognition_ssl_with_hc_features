"""HC manifest: records dataset windowing parameters and feature names.

No training-time fitting — the manifest is fully deterministic from channel_cols + fs.
"""

from __future__ import annotations

import json
from pathlib import Path

from .hc_features import feature_names


def build_manifest(
    channel_cols: list[str],
    fs: int,
    window_size: int,
    stride: int,
    purity_threshold: float,
    dataset_id: str = "",
) -> dict:
    return {
        "schema_version": 2,
        "dataset_id": dataset_id,
        "channel_cols": channel_cols,
        "fs": fs,
        "window_size": window_size,
        "stride": stride,
        "purity_threshold": purity_threshold,
        "feature_names": feature_names(channel_cols),
    }


def save_manifest(manifest: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))
    print(f"Manifest: {path}")


def load_manifest(path: Path) -> dict:
    data = json.loads(path.read_text())
    if data.get("schema_version", 1) != 2:
        raise ValueError(f"Unexpected manifest schema_version: {data.get('schema_version')}")
    return data
