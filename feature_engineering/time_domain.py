"""Time-domain feature extraction for vibration signals."""
from __future__ import annotations

from typing import Dict

import numpy as np


def compute_time_features(signal: np.ndarray) -> Dict[str, float]:
    """Extract common time-domain statistical features from a 1D signal.

    Returns a dict with keys: mean, std, rms, peak, peak_to_peak,
    crest_factor, kurtosis, skewness, shape_factor, impulse_factor,
    clearance_factor.
    """
    if signal.ndim != 1:
        signal = signal.reshape(-1)
    n = len(signal)
    if n == 0:
        return {}

    mean_val = float(np.mean(signal))
    std_val = float(np.std(signal))
    rms = float(np.sqrt(np.mean(signal ** 2)))
    peak = float(np.max(np.abs(signal)))
    peak_to_peak = float(np.max(signal) - np.min(signal))

    # Crest factor = peak / rms
    crest_factor = peak / rms if rms > 1e-12 else 0.0

    # Excess kurtosis
    if std_val > 1e-12:
        kurtosis = float(np.mean(((signal - mean_val) / std_val) ** 4) - 3.0)
    else:
        kurtosis = 0.0

    # Skewness
    if std_val > 1e-12:
        skewness = float(np.mean(((signal - mean_val) / std_val) ** 3))
    else:
        skewness = 0.0

    # Shape factor = rms / mean(|x|)
    mean_abs = float(np.mean(np.abs(signal)))
    shape_factor = rms / mean_abs if mean_abs > 1e-12 else 0.0

    # Impulse factor = peak / mean(|x|)
    impulse_factor = peak / mean_abs if mean_abs > 1e-12 else 0.0

    # Clearance factor = peak / (mean(sqrt(|x|)))^2
    mean_sqrt_abs = float(np.mean(np.sqrt(np.abs(signal))))
    clearance_factor = peak / (mean_sqrt_abs ** 2) if mean_sqrt_abs > 1e-12 else 0.0

    return {
        "mean": mean_val,
        "std": std_val,
        "rms": rms,
        "peak": peak,
        "peak_to_peak": peak_to_peak,
        "crest_factor": crest_factor,
        "kurtosis": kurtosis,
        "skewness": skewness,
        "shape_factor": shape_factor,
        "impulse_factor": impulse_factor,
        "clearance_factor": clearance_factor,
    }
