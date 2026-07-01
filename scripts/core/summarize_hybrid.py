#!/usr/bin/env python3
"""Aggregate runs/hybrid/<dataset>/<combo>/test_stage1_classification_metrics.json
into results/hybrid_summary_all.csv, with the same column layout as
results/<dataset>/summary.csv (deep-only ablation) so the two can be joined on
dataset,encoder,method,finetune.

<combo> = "<method>_<encoder>_<freeze|full>_<self|pool>" (written by
scripts/core/hybrid_grid.py); "self"/"pool" is recorded as checkpoint_source.

Usage:
    python scripts/core/summarize_hybrid.py --runs_dir runs/hybrid --out results/hybrid_summary_all.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


_VARIANTS = {"full", "frozen", "lora"}


def _parse_combo(combo: str) -> tuple[str, str, str, str]:
    """'<method>_<encoder>_<self|pool>' or '<method>_<encoder>_<variant>_<self|pool>'
    -> (method, encoder, variant, source). 3-part combos default variant to 'full'."""
    parts = combo.split("_")
    method, source = parts[0], parts[-1]
    middle = parts[1:-1]
    if middle and middle[-1] in _VARIANTS:
        variant = middle[-1]
        encoder = "_".join(middle[:-1])
    else:
        variant = "full"
        encoder = "_".join(middle)
    return method, encoder, variant, source


def collect_runs(runs_dir: Path) -> list[dict]:
    rows = []
    for ds_dir in sorted(runs_dir.iterdir()):
        if not ds_dir.is_dir():
            continue
        for run_dir in sorted(ds_dir.iterdir()):
            metrics_path = run_dir / "test_stage1_classification_metrics.json"
            if not metrics_path.exists():
                continue
            dataset, combo = ds_dir.name, run_dir.name
            method, encoder, variant, source = _parse_combo(combo)
            with open(metrics_path) as f:
                payload = json.load(f)
            overall = payload["overall"]
            per_behavior = payload.get("per_behavior", {})
            precisions = [b["precision"] for b in per_behavior.values()]
            recalls = [b["recall_true_class_accuracy"] for b in per_behavior.values()]

            row = {
                "run_id": f"{dataset}/{combo}",
                "dataset": dataset,
                "encoder": encoder,
                "method": method,
                "finetune": variant,
                "checkpoint_source": source,
                "acc": overall["accuracy"],
                "bal_acc": overall["balanced_accuracy"],
                "macro_f1": overall["f1_macro"],
                "accuracy": overall["accuracy"],
                "balanced_accuracy": overall["balanced_accuracy"],
                "weighted_f1": overall["f1_weighted"],
                "precision": sum(precisions) / len(precisions) if precisions else "",
                "recall": sum(recalls) / len(recalls) if recalls else "",
            }
            for cls, block in per_behavior.items():
                row[f"f1_{cls}"] = block["f1"]
            rows.append(row)
    return rows


def _write_csv(rows: list[dict], out_path: Path) -> None:
    all_keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} runs to {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", default="runs/hybrid")
    p.add_argument("--out", default="results/hybrid_summary_all.csv")
    args = p.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        print(f"No runs dir: {runs_dir}")
        return

    rows = collect_runs(runs_dir)
    if not rows:
        print("No completed hybrid runs found.")
        return
    _write_csv(rows, Path(args.out))


if __name__ == "__main__":
    main()
