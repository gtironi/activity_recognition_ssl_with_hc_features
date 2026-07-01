"""Dataset registry: load YAML, resolve processed paths."""

from __future__ import annotations

from pathlib import Path

import yaml


def repo_root() -> Path:
    # src/common/registry.py → repo root is 2 up
    return Path(__file__).resolve().parents[2]


def load_registry(path: str | Path | None = None) -> dict:
    if path is None:
        path = repo_root() / "dataset_registry.yaml"
    else:
        path = Path(path)
        if not path.is_absolute():
            path = repo_root() / path
    with open(path) as f:
        return yaml.safe_load(f)["datasets"]


def resolve_processed_root(dataset_id: str, registry: dict) -> Path:
    if dataset_id not in registry:
        raise ValueError(f"Unknown dataset_id={dataset_id!r}; available: {list(registry.keys())}")
    info = registry[dataset_id]
    pr = info.get("processed_root")
    if pr:
        return repo_root() / pr
    return repo_root() / "dataset" / "processed" / dataset_id
