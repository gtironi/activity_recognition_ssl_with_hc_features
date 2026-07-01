"""Console logging setup. The shell-side `tee` in _common.sh writes the file."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(output_dir: str | Path, filename: str = "train.log") -> None:
    """Configure the root logger to write to stdout only.

    File logging is handled by the shell wrapper via `tee`; adding a FileHandler
    here would duplicate every line in the log file. ``output_dir``/``filename``
    are accepted for backwards compatibility but unused.

    Idempotent: removes any handlers previously added by this function before
    attaching new ones.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Remove previous handlers tagged by us
    for h in list(root.handlers):
        if getattr(h, "_hybrid_log", False):
            root.removeHandler(h)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    ch._hybrid_log = True  # type: ignore[attr-defined]
    root.addHandler(ch)
