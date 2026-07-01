"""Reproducibility: seed + manifest."""

from __future__ import annotations
import json
import os
import random
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def save_manifest(path: Path, cfg, data_sha256: str = "") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "git_commit": get_git_commit(),
        "seed": getattr(cfg, "seed", None),
        "run_name": getattr(cfg, "run_name", None),
        "data_sha256": data_sha256,
    }
    path.write_text(json.dumps(manifest, indent=2))


def save_environment(path: Path) -> None:
    import subprocess as sp
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = sp.check_output(["pip", "freeze"], stderr=sp.DEVNULL).decode()
    except Exception:
        result = "unavailable"
    path.write_text(result)
