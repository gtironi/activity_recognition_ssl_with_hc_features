"""N-channel sliding-window over a long time-series DataFrame.

Supports any number of signal columns via the channel_cols argument.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def window_dataframe(
    df: pd.DataFrame,
    *,
    channel_cols: list[str],
    label_col: str,
    group_by: list[str],
    subject_col: str,
    time_col: str,
    window_size: int,
    stride: int,
    purity_threshold: float,
) -> pd.DataFrame:
    """
    Slide a window of length ``window_size`` with step ``stride`` over each group.

    Args:
        df: long-format DataFrame (one row = one sample).
        channel_cols: signal columns to window (e.g. ["acc_x","acc_y","acc_z"]).
        label_col: column with the activity label (string).
        group_by: columns defining independent sequences (e.g. ["calfId","segId"]).
        subject_col: which of group_by identifies the subject → stored as calf_id.
        time_col: timestamp column → stored as dateTime.
        window_size: number of samples per window.
        stride: step size in samples.
        purity_threshold: fraction of window that must share the dominant label
                          (1.0 = strict, 0.9 = 90 % majority).

    Returns:
        DataFrame with columns: dateTime, calf_id, <channel_cols...>, label.
        Each cell in channel_cols is a Python list of floats of length window_size.
    """
    df = df.copy()
    df[label_col] = df[label_col].astype(str)

    window_list = []
    for key, group in df.groupby(group_by, sort=False):
        if isinstance(key, tuple):
            subj = key[group_by.index(subject_col)]
        else:
            subj = key

        raw = group[channel_cols].to_numpy(dtype=np.float32)
        times = group[time_col].to_numpy()
        labels = group[label_col].to_numpy()
        n = len(raw)

        if n < window_size:
            continue

        for start in range(0, n - window_size + 1, stride):
            end = start + window_size
            win_labels = labels[start:end]
            most_common, count = Counter(win_labels).most_common(1)[0]
            if count / window_size < purity_threshold:
                continue
            win_signals = raw[start:end]
            row: dict = {
                "dateTime": times[start],
                "calf_id": subj,
                "label": most_common,
            }
            for i, col in enumerate(channel_cols):
                row[col] = win_signals[:, i].tolist()
            window_list.append(row)

    print(f"Windows generated: {len(window_list)}")
    col_order = ["dateTime", "calf_id"] + channel_cols + ["label"]
    if not window_list:
        return pd.DataFrame(columns=col_order)
    df_out = pd.DataFrame(window_list)
    return df_out[col_order].reset_index(drop=True)


def window_dataframe_strict(
    df: pd.DataFrame,
    *,
    channel_cols: list[str],
    label_col: str,
    group_by: list[str],
    subject_col: str,
    time_col: str,
    window_size: int,
    stride: int,
) -> pd.DataFrame:
    """Strict-purity variant: window is kept only if all labels are identical."""
    return window_dataframe(
        df,
        channel_cols=channel_cols,
        label_col=label_col,
        group_by=group_by,
        subject_col=subject_col,
        time_col=time_col,
        window_size=window_size,
        stride=stride,
        purity_threshold=1.0,
    )
