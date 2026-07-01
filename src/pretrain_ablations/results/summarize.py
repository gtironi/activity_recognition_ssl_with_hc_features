"""Aggregate all eval/metrics_test.json from runs/ into a summary CSV.

Usage:
    python -m pretrain_ablations.results.summarize
    python -m pretrain_ablations.results.summarize --runs_dir runs --out results/summary.csv --filter smoke_
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _iter_run_dirs(runs_dir: Path):
    """Yield every directory containing eval/metrics_test.json, recursing one
    level so that both flat (runs/<run>) and nested (runs/<dataset>/<run>)
    layouts are discovered."""
    for entry in sorted(runs_dir.iterdir()):
        if not entry.is_dir():
            continue
        if (entry / "eval" / "metrics_test.json").exists():
            yield entry
        else:
            # descend one level (e.g. runs/<dataset>/<run>)
            for sub in sorted(entry.iterdir()):
                if sub.is_dir() and (sub / "eval" / "metrics_test.json").exists():
                    yield sub


def collect_runs(runs_dir: Path, filter_str: str = "") -> list[dict]:
    rows = []
    for run_dir in _iter_run_dirs(runs_dir):
        # run_id includes the parent (dataset) when nested
        rel = run_dir.relative_to(runs_dir)
        run_id = str(rel)
        if filter_str and filter_str not in run_id:
            continue
        metrics_path = run_dir / "eval" / "metrics_test.json"
        summary_path = run_dir / "eval" / "summary.txt"
        config_path  = run_dir / "artifacts" / "config.yaml"
        with open(metrics_path) as f:
            metrics = json.load(f)

        row: dict = {"run_id": run_id}

        # parse from summary.txt if available
        if summary_path.exists():
            summary = summary_path.read_text().strip()
            for part in summary.split("|"):
                part = part.strip()
                if "=" in part:
                    k, v = part.split("=", 1)
                    row[k.strip()] = v.strip()

        # fill from config.yaml if fields missing
        if config_path.exists():
            import yaml
            with open(config_path) as f:
                cfg = yaml.safe_load(f) or {}
            row.setdefault("dataset", cfg.get("data", {}).get("dataset_id", ""))
            row.setdefault("encoder", cfg.get("encoder", {}).get("name", ""))
            row.setdefault("method", cfg.get("pretext", {}).get("method", ""))
            row.setdefault("finetune", cfg.get("finetune", {}).get("mode", ""))
            row.setdefault("seed", str(cfg.get("seed", "")))

        # add metrics
        row["accuracy"]          = metrics.get("accuracy", "")
        row["balanced_accuracy"] = metrics.get("balanced_accuracy", "")
        row["macro_f1"]     = metrics.get("macro_f1", "")
        row["weighted_f1"]  = metrics.get("weighted_f1", "")
        row["precision"]    = metrics.get("precision_macro", "")
        row["recall"]       = metrics.get("recall_macro", "")
        per_class = metrics.get("per_class_f1", {})
        for cls, val in per_class.items():
            row[f"f1_{cls}"] = val
        rows.append(row)
    return rows


def _write_csv(rows: list[dict], out_path: Path) -> None:
    all_keys = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                all_keys.append(k); seen.add(k)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} runs to {out_path}")


def _print_table(rows: list[dict]) -> None:
    print(f"\n{'run_id':40s} {'encoder':12s} {'method':12s} {'finetune':10s} "
          f"{'acc':6s} {'bal_acc':8s} {'macro_f1':8s}")
    print("-" * 100)
    for row in rows:
        print(f"{str(row.get('run_id',''))[:40]:40s} "
              f"{str(row.get('encoder',''))[:12]:12s} "
              f"{str(row.get('method',''))[:12]:12s} "
              f"{str(row.get('finetune',''))[:10]:10s} "
              f"{str(row.get('accuracy',''))[:6]:6s} "
              f"{str(row.get('balanced_accuracy',''))[:8]:8s} "
              f"{str(row.get('macro_f1',''))[:8]:8s}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", default="runs")
    p.add_argument("--out", default="results/summary.csv")
    p.add_argument("--filter", default="", help="Only include runs whose name contains this string")
    p.add_argument("--per_dataset", action="store_true",
                   help="Also write results/<dataset>/summary.csv grouped by the 'dataset' column.")
    p.add_argument("--results_dir", default="results",
                   help="Base dir for per-dataset CSVs (used with --per_dataset).")
    args = p.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        print(f"No runs dir: {runs_dir}")
        return

    rows = collect_runs(runs_dir, args.filter)
    if not rows:
        print("No completed runs found.")
        return

    _write_csv(rows, Path(args.out))

    if args.per_dataset:
        by_ds: dict[str, list[dict]] = {}
        for row in rows:
            ds = str(row.get("dataset", "") or "unknown")
            by_ds.setdefault(ds, []).append(row)
        for ds, ds_rows in sorted(by_ds.items()):
            _write_csv(ds_rows, Path(args.results_dir) / ds / "summary.csv")

    _print_table(rows)


if __name__ == "__main__":
    main()
