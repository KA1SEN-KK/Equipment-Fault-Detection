"""CMAPSS dataset loading and preprocessing utilities."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# Column layout: unit, cycle, 3 op-settings, 21 sensors
_COL_NAMES = (
    ["unit", "cycle", "op1", "op2", "op3"]
    + [f"s{i}" for i in range(1, 22)]
)

# Sensors with near-zero variance in FD001/FD003 — drop them
_CONST_SENSORS = {"s1", "s5", "s6", "s10", "s16", "s18", "s19"}
USEFUL_SENSORS = [c for c in _COL_NAMES if c.startswith("s") and c not in _CONST_SENSORS]
N_FEATURES = len(USEFUL_SENSORS)  # 14


def load_cmapss(path: Path) -> pd.DataFrame:
    """Load a CMAPSS text file into a DataFrame with named columns."""
    df = pd.read_csv(path, sep=r"\s+", header=None, names=_COL_NAMES)
    df = df[["unit", "cycle"] + USEFUL_SENSORS]
    return df


def add_rul(df: pd.DataFrame) -> pd.DataFrame:
    """Append a 'rul' column (cycles remaining until end of life)."""
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    df = df.copy()
    df["rul"] = max_cycle - df["cycle"]
    return df


def sliding_windows_multivariate(
    arr: np.ndarray,
    win: int,
    step: int,
) -> np.ndarray:
    """Slide over a (T, F) array → (N, win, F) windows."""
    T, F = arr.shape
    n = (T - win) // step + 1
    if n <= 0:
        return np.empty((0, win, F), dtype=np.float32)
    idx = np.arange(win)[None, :] + step * np.arange(n)[:, None]
    return arr[idx].astype(np.float32)


def build_windows_from_df(
    df: pd.DataFrame,
    win: int,
    step: int,
    healthy_rul_min: Optional[int] = None,
    max_windows: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (windows, labels) from a CMAPSS DataFrame.

    healthy_rul_min: if set, only use cycles where rul >= this value as healthy (label=0).
                     Otherwise all cycles are used with label = (rul < 30).
    Returns:
        windows : (N, win, N_FEATURES) float32
        labels  : (N,) int  — 0=healthy, 1=degraded/fault
    """
    has_rul = "rul" in df.columns
    all_windows: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    for unit_id, grp in df.groupby("unit"):
        grp = grp.sort_values("cycle")
        sensor_arr = grp[USEFUL_SENSORS].to_numpy(dtype=np.float32)

        if has_rul:
            rul_arr = grp["rul"].to_numpy(dtype=np.int32)
        else:
            rul_arr = None

        if healthy_rul_min is not None and has_rul:
            mask = rul_arr >= healthy_rul_min
            sensor_arr = sensor_arr[mask]
            if len(sensor_arr) < win:
                continue

        wins = sliding_windows_multivariate(sensor_arr, win, step)
        if len(wins) == 0:
            continue

        if has_rul and healthy_rul_min is None:
            # label by last cycle RUL in each window
            rul_windows = sliding_windows_multivariate(
                rul_arr[:, None].astype(np.float32), win, step
            )
            last_rul = rul_windows[:, -1, 0].astype(np.int32)
            labels = (last_rul < 30).astype(np.int32)
        else:
            labels = np.zeros(len(wins), dtype=np.int32)

        all_windows.append(wins)
        all_labels.append(labels)

        if max_windows is not None:
            total = sum(len(w) for w in all_windows)
            if total >= max_windows:
                break

    if not all_windows:
        return np.empty((0, win, N_FEATURES), dtype=np.float32), np.empty(0, dtype=np.int32)

    windows = np.concatenate(all_windows, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    if max_windows is not None:
        windows = windows[:max_windows]
        labels = labels[:max_windows]

    return windows, labels
