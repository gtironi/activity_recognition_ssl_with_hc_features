"""Fixed hand-crafted feature set for windowed accelerometer/IMU data.

Per-channel (12 features each): mean, std, rms, min, max, range, iqr, zcr, mad,
dom_freq, spec_entropy, spec_energy.
Magnitude (from first 3 channels = acc_x/y/z): sma, odba.

All features are computed with pure numpy + scipy — no training-time fitting.
Column names: hc_{channel}_{feature} for per-channel; hc_sma, hc_odba for magnitude.
"""

from __future__ import annotations

import numpy as np
from scipy.fft import rfft, rfftfreq

HC_PER_CHANNEL_SUFFIXES = [
    "mean", "std", "rms", "min", "max", "range",
    "iqr", "zcr", "mad", "dom_freq", "spec_entropy", "spec_energy",
]


def _per_channel(w: np.ndarray, fs: int) -> dict[str, float]:
    freqs = rfftfreq(len(w), d=1.0 / fs)
    mag = np.abs(rfft(w))
    p = mag / (mag.sum() + 1e-12)
    return {
        "mean": float(np.mean(w)),
        "std": float(np.std(w)),
        "rms": float(np.sqrt(np.mean(w ** 2))),
        "min": float(np.min(w)),
        "max": float(np.max(w)),
        "range": float(np.max(w) - np.min(w)),
        "iqr": float(np.percentile(w, 75) - np.percentile(w, 25)),
        "zcr": float(np.mean(np.diff(np.sign(w)) != 0)),
        "mad": float(np.mean(np.abs(w - np.mean(w)))),
        "dom_freq": float(freqs[np.argmax(mag)]),
        "spec_entropy": float(-np.sum(p * np.log(p + 1e-12))),
        "spec_energy": float(np.sum(mag ** 2) / len(w)),
    }


def feature_names(channel_cols: list[str]) -> list[str]:
    """Return the ordered list of HC column names for a given channel layout."""
    names = []
    for ch in channel_cols:
        for suf in HC_PER_CHANNEL_SUFFIXES:
            names.append(f"hc_{ch}_{suf}")
    names += ["hc_sma", "hc_odba"]
    return names


def compute_window(signals: dict[str, np.ndarray], fs: int) -> dict[str, float]:
    """
    Compute all HC features for one window.

    Args:
        signals: ordered dict {col_name: 1-D float array of length T}.
                 First 3 keys must correspond to acc_x/y/z channels (for SMA/ODBA).
        fs: sampling frequency in Hz.

    Returns:
        Flat dict of {hc_*: float} scalars.
    """
    out: dict[str, float] = {}
    for col, w in signals.items():
        w = np.asarray(w, dtype=np.float64)
        for suf, val in _per_channel(w, fs).items():
            out[f"hc_{col}_{suf}"] = val

    # Magnitude features — first 3 channels are always acc_x/y/z by convention.
    acc_arrays = [np.asarray(w, dtype=np.float64) for w in list(signals.values())[:3]]
    x, y, z = acc_arrays
    out["hc_sma"] = float(np.mean(np.abs(x) + np.abs(y) + np.abs(z)))
    out["hc_odba"] = float(
        np.mean(np.abs(x - x.mean()) + np.abs(y - y.mean()) + np.abs(z - z.mean()))
    )
    return out


def compute_dataframe(
    df_windowed: "pd.DataFrame",
    channel_cols: list[str],
    fs: int,
) -> "pd.DataFrame":
    """
    Compute HC features for every row of a windowed parquet DataFrame.

    Args:
        df_windowed: DataFrame where each cell in channel_cols is a list of floats.
        channel_cols: ordered list of signal column names (first 3 must be acc_x/y/z).
        fs: sampling frequency in Hz.

    Returns:
        DataFrame with hc_* columns aligned to df_windowed's index.
    """
    import pandas as pd

    rows = []
    for tup in df_windowed[channel_cols].itertuples(index=False):
        signals = {col: np.asarray(val, dtype=np.float64)
                   for col, val in zip(channel_cols, tup)}
        rows.append(compute_window(signals, fs))
    return pd.DataFrame(rows, index=df_windowed.index)
