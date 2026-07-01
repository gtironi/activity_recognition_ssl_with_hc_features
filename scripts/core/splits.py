"""Subject-level train/test/val splitting utilities.

Contains the genSplit algorithm (originally genSplit.py) plus higher-level helpers
used by dataset adapters.
"""

from __future__ import annotations

import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


# ---------------------------------------------------------------------------
# genSplit core (verbatim from scripts/genSplit.py)
# ---------------------------------------------------------------------------

def calc_split_subject_amounts(total_subject_count: int, percentages: dict) -> list[int]:
    percentages_ = [percentages["train"], percentages["validation"], percentages["test"]]
    initial_values = [round(total_subject_count * p / 100) for p in percentages_]
    diff = total_subject_count - sum(initial_values)
    for i in range(abs(diff)):
        initial_values[i % 3] += int(diff / abs(diff))
    return initial_values


def generate_sbj_sets(all_subjects, num_to_select):
    return list(combinations(all_subjects, num_to_select))


MAX_COMBINATIONS = 100_000


def find_optimal_subject_combination(
    all_sbj_ids,
    num_to_select: int,
    data_amounts_df: pd.DataFrame,
    split_ratio: float,
    cv: int = 1,
    random_state: int = 2026,
    n_random_samples: int = 100_000,
):
    """
    Find the test-subject combination that best balances label proportions.

    data_amounts_df must have columns: subject_id, <class1>, <class2>, ...
    Returns a tuple of subject ids (cv=1) or a list of tuples (cv>1).

    If the number of combinations exceeds MAX_COMBINATIONS, falls back to
    random sampling of `n_random_samples` combinations instead of exhaustive
    search.
    """
    n_comb = math.comb(len(all_sbj_ids), num_to_select)
    if n_comb > MAX_COMBINATIONS:
        rng = np.random.default_rng(random_state)
        seen = set()
        n_samples = min(n_random_samples, n_comb)
        candidates = []
        all_sbj_arr = np.array(all_sbj_ids, dtype=object)
        while len(candidates) < n_samples:
            combo = tuple(sorted(rng.choice(all_sbj_arr, size=num_to_select, replace=False).tolist()))
            if combo not in seen:
                seen.add(combo)
                candidates.append(combo)
    else:
        candidates = list(combinations(all_sbj_ids, num_to_select))

    total_counts = data_amounts_df.sum().values[1:]
    deviations: dict[float, tuple] = {}

    for combination in candidates:
        comb_counts = (
            data_amounts_df[data_amounts_df["subject_id"].isin(combination)]
            .sum()
            .values[1:]
        )
        train_counts = total_counts - comb_counts
        if np.any(train_counts == 0):
            continue
        label_ratios = comb_counts / train_counts
        mean_deviation = float(np.mean(np.abs(label_ratios - split_ratio)))
        deviations[mean_deviation] = combination

    if not deviations:
        return None if cv == 1 else []
    if cv == 1:
        return deviations[min(deviations)]
    return [comb for _, comb in sorted(deviations.items())[:cv]]


# ---------------------------------------------------------------------------
# Higher-level helpers
# ---------------------------------------------------------------------------

def subject_behavior_wide(
    df: pd.DataFrame, subject_col: str, behavior_col: str
) -> pd.DataFrame:
    g = df.groupby([subject_col, behavior_col], sort=False).size().rename("n").reset_index()
    wide = g.pivot(index=subject_col, columns=behavior_col, values="n").fillna(0).astype(np.int64)
    return wide.reset_index().rename(columns={subject_col: "subject_id"})


def split_by_subject_list(
    df: pd.DataFrame,
    subject_col: str,
    test_subjects,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split df into train/test using a fixed list of test subject IDs."""
    test_ids = set(test_subjects)
    # Coerce dtype to match
    if pd.api.types.is_integer_dtype(df[subject_col]):
        test_ids = {int(x) for x in test_ids}
    elif pd.api.types.is_float_dtype(df[subject_col]):
        test_ids = {float(x) for x in test_ids}
    mask = df[subject_col].isin(test_ids)
    return df.loc[~mask].reset_index(drop=True), df.loc[mask].reset_index(drop=True)


def load_existing_split(
    df: pd.DataFrame,
    subject_col: str,
    split_report_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict] | None:
    """
    If a split_report.json exists at split_report_path and contains 'test_subject_ids',
    use those IDs to recreate the train/test split from df without rerunning genSplit.
    Returns (train_df, test_df, meta) or None if the report doesn't exist or has no IDs.
    """
    if not split_report_path.exists():
        return None
    report = json.loads(split_report_path.read_text())
    split = report.get("split", report)  # handle both flat and nested formats
    test_ids = split.get("test_subject_ids")
    if not test_ids:
        return None
    print(f"Reusing existing split from {split_report_path} (test subjects: {test_ids})")
    train, test = split_by_subject_list(df, subject_col, test_ids)
    return train, test, split


def split_by_gen_split(
    df: pd.DataFrame,
    subject_col: str,
    behavior_col: str,
    test_fraction: float,
    existing_split_report: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Subject-disjoint split using the genSplit label-balance algorithm.

    If existing_split_report is provided and contains a previous split, it is
    reused without rerunning the combination search.
    Raises ValueError if the number of combinations exceeds MAX_COMBINATIONS (100k).
    Returns (train_df, test_df, meta_dict).
    """
    # Reuse existing split if available
    if existing_split_report is not None:
        result = load_existing_split(df, subject_col, existing_split_report)
        if result is not None:
            return result

    train_pct = round(100.0 * (1.0 - test_fraction))
    test_pct = round(100.0 * test_fraction)
    if train_pct + test_pct != 100:
        raise ValueError(
            f"test_fraction={test_fraction} does not produce integer percentages summing to 100."
        )

    wide = subject_behavior_wide(df, subject_col, behavior_col)
    subjects = sorted(wide["subject_id"].unique().tolist())
    n_sub = len(subjects)
    if n_sub < 2:
        raise ValueError("Need at least 2 subjects.")

    n_tr, _n_val, n_te = calc_split_subject_amounts(
        n_sub, {"train": float(train_pct), "validation": 0.0, "test": float(test_pct)}
    )
    if n_te < 1 or n_tr < 1:
        raise ValueError(f"Invalid subject counts: train={n_tr} test={n_te} (n={n_sub}).")

    ratio = test_fraction / (1.0 - test_fraction)
    # find_optimal_subject_combination raises if n_comb > MAX_COMBINATIONS
    test_tuple = find_optimal_subject_combination(tuple(subjects), n_te, wide, ratio, cv=1)
    if test_tuple is None:
        raise ValueError(
            "genSplit found no valid combination (all classes must remain in train). "
            "Try reducing test_fraction or filtering rare classes first."
        )

    test_ids = set(test_tuple)
    mask = df[subject_col].isin(test_ids)
    train = df.loc[~mask].reset_index(drop=True)
    test = df.loc[mask].reset_index(drop=True)
    meta = {
        "method": "gen_split",
        "test_fraction_subjects": test_pct / 100.0,
        "n_subjects_total": n_sub,
        "n_subjects_train": n_sub - len(test_ids),
        "n_subjects_test": len(test_ids),
        "test_subject_ids": sorted(test_ids, key=str),
    }
    return train, test, meta


def carve_val_by_subject(
    df_train: pd.DataFrame,
    subject_col: str,
    val_size: float = 0.1,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carve a subject-disjoint validation set from training data using GroupShuffleSplit.
    Returns (train_remaining, val).
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    groups = df_train[subject_col].to_numpy()
    tr_idx, vl_idx = next(gss.split(df_train, groups=groups))
    return (
        df_train.iloc[tr_idx].reset_index(drop=True),
        df_train.iloc[vl_idx].reset_index(drop=True),
    )


def save_split_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str))
    print(f"Split report: {path}")
