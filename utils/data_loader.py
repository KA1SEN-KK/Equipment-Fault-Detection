"""Data loading utilities for CWRU and generic vibration data."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import scipy.io as sio


@dataclass
class DatasetSpec:
    """Specification for a group of CWRU .mat files."""

    name: str
    folder: Path
    file_ids: List[str]
    channel_key: str


def load_mat_signal(file_path: Path, channel_key: str) -> np.ndarray:
    """Load a channel from CWRU .mat, allowing suffix matching."""
    data = sio.loadmat(file_path)
    if channel_key not in data:
        candidates = [
            k
            for k in data.keys()
            if not k.startswith("__") and (k.endswith(channel_key) or channel_key in k)
        ]
        if not candidates:
            raise KeyError(
                f"Channel {channel_key} not in {file_path.name}; keys={list(data.keys())}"
            )
        channel_key = sorted(candidates)[0]
    sig = np.asarray(data[channel_key]).squeeze()
    return sig.astype(np.float32)


def collect_signals(spec: DatasetSpec) -> List[np.ndarray]:
    """Load all signals described by a DatasetSpec."""
    signals = []
    for fid in spec.file_ids:
        fp = spec.folder / f"{fid}.mat"
        if not fp.exists():
            continue
        sig = load_mat_signal(fp, spec.channel_key)
        signals.append(sig)
    if not signals:
        raise RuntimeError(f"No signals loaded for {spec.name}")
    return signals


def sliding_windows(x: np.ndarray, win: int, step: int) -> np.ndarray:
    """Create sliding windows from a 1D signal."""
    if x.ndim != 1:
        x = x.reshape(-1)
    n = (len(x) - win) // step + 1
    if n <= 0:
        return np.empty((0, win), dtype=np.float32)
    idx = np.arange(win)[None, :] + step * np.arange(n)[:, None]
    return x[idx].astype(np.float32)


def make_dataset(
    signals: List[np.ndarray],
    win: int,
    step: int,
    max_windows: Optional[int] = None,
) -> np.ndarray:
    """Build a windowed dataset from a list of signals."""
    chunks = []
    total = 0
    for sig in signals:
        w = sliding_windows(sig, win, step)
        if max_windows is not None and total + len(w) > max_windows:
            w = w[: max_windows - total]
        chunks.append(w)
        total += len(w)
        if max_windows is not None and total >= max_windows:
            break
    if not chunks:
        return np.empty((0, win), dtype=np.float32)
    return np.concatenate(chunks, axis=0)
