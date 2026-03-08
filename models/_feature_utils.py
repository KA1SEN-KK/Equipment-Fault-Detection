"""Shared per-window feature extraction utilities for sklearn-based model runners.

Provides vectorised feature computation over sliding windows so that
lightweight ML models (IF, OCSVM, PCA, KMeans, RF …) can work on
a compact feature matrix instead of raw waveforms.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from utils.data_loader import sliding_windows as _sliding_windows

# ── Default window parameters (match LSTM-AE defaults) ──────────────
DEFAULT_WIN = 2048
DEFAULT_STEP = 512


def make_windows(signal: np.ndarray, win: int = DEFAULT_WIN, step: int = DEFAULT_STEP) -> np.ndarray:
    """Create sliding windows from a 1-D signal.  Returns (N, win)."""
    if signal.ndim != 1:
        signal = signal.reshape(-1)
    return _sliding_windows(signal, win, step)


# ── Vectorised time-domain features ─────────────────────────────────

def _time_features(windows: np.ndarray) -> np.ndarray:
    """Compute 8 time-domain features per window.  Returns (N, 8)."""
    eps = 1e-12
    mean = np.mean(windows, axis=1)
    std = np.std(windows, axis=1)
    rms = np.sqrt(np.mean(windows ** 2, axis=1))
    peak = np.max(np.abs(windows), axis=1)

    safe_std = np.where(std > eps, std, 1.0)
    centered = windows - mean[:, None]
    normed = centered / safe_std[:, None]
    kurtosis = np.mean(normed ** 4, axis=1) - 3.0
    skewness = np.mean(normed ** 3, axis=1)

    safe_rms = np.where(rms > eps, rms, 1.0)
    crest_factor = peak / safe_rms

    mean_abs = np.mean(np.abs(windows), axis=1)
    safe_mean_abs = np.where(mean_abs > eps, mean_abs, 1.0)
    shape_factor = rms / safe_mean_abs
    impulse_factor = peak / safe_mean_abs

    return np.column_stack([
        rms, std, peak, kurtosis, skewness,
        crest_factor, shape_factor, impulse_factor,
    ]).astype(np.float32)


# ── Vectorised frequency-domain features ────────────────────────────

def _freq_features(windows: np.ndarray, sampling_rate: float = 12000.0) -> np.ndarray:
    """Compute 3 frequency-domain features per window.  Returns (N, 3)."""
    eps = 1e-12
    n_fft = windows.shape[1]
    fft_mag = np.abs(np.fft.rfft(windows, n=n_fft, axis=1))       # (N, n_fft//2+1)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sampling_rate)         # (n_fft//2+1,)
    power = fft_mag ** 2
    total_power = np.sum(power, axis=1, keepdims=True) + eps       # (N, 1)

    # Spectral centroid
    spectral_centroid = np.sum(freqs[None, :] * power, axis=1) / total_power.squeeze()

    # Spectral entropy
    p = power / total_power
    p = np.clip(p, eps, None)
    spectral_entropy = -np.sum(p * np.log2(p), axis=1)

    # Band-energy ratio: low (<1/4 Nyquist) vs total
    nyq_quarter = int(len(freqs) // 4)
    low_energy = np.sum(power[:, :nyq_quarter], axis=1)
    band_ratio = low_energy / total_power.squeeze()

    return np.column_stack([
        spectral_centroid, spectral_entropy, band_ratio,
    ]).astype(np.float32)


# ── Public API ──────────────────────────────────────────────────────

FEATURE_NAMES = [
    "rms", "std", "peak", "kurtosis", "skewness",
    "crest_factor", "shape_factor", "impulse_factor",
    "spectral_centroid", "spectral_entropy", "band_energy_ratio",
]
"""Names of the 11 features returned by *extract_window_features*."""


def extract_window_features(
    windows: np.ndarray,
    sampling_rate: float = 12000.0,
    include_freq: bool = True,
) -> np.ndarray:
    """Compute an (N, 11) feature matrix from (N, win_len) sliding windows.

    Parameters
    ----------
    windows : ndarray, shape (N, win_len)
    sampling_rate : float
        Sensor sampling rate in Hz (used for frequency features).
    include_freq : bool
        If False only the 8 time-domain features are returned.

    Returns
    -------
    ndarray, shape (N, 11) or (N, 8)
    """
    tf = _time_features(windows)
    if not include_freq:
        return tf
    ff = _freq_features(windows, sampling_rate)
    return np.hstack([tf, ff])
