from __future__ import annotations

import hashlib
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def repo_root() -> Path:
    # scripts/core/repro_utils.py -> repo root is 2 up
    return Path(__file__).resolve().parents[2]


def load_registry(path: str | Path | None = None) -> dict:
    if path is None:
        path = repo_root() / "dataset_registry.yaml"
    with open(path) as f:
        return yaml.safe_load(f)["datasets"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=indent)


def load_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


def hash_tensor(t: torch.Tensor) -> str:
    return hashlib.sha256(t.numpy().tobytes()).hexdigest()


def make_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")
