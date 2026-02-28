"""Frequency-domain feature extraction for vibration signals."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def compute_frequency_features(
    signal: np.ndarray,
    sampling_rate: float,
    n_fft: Optional[int] = None,
) -> Dict[str, float]:
    """Extract frequency-domain features via FFT.

    Returns a dict with keys: spectral_centroid, spectral_spread,
    spectral_peak_freq, spectral_peak_mag, total_spectral_power,
    spectral_entropy.
    """
    if signal.ndim != 1:
        signal = signal.reshape(-1)
    n = len(signal)
    if n == 0:
        return {}

    if n_fft is None:
        n_fft = n

    # Single-sided amplitude spectrum
    fft_vals = np.fft.rfft(signal, n=n_fft)
    fft_mag = np.abs(fft_vals)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sampling_rate)

    # Power spectrum
    power = fft_mag ** 2
    total_power = float(np.sum(power))
    if total_power < 1e-12:
        return {
            "spectral_centroid": 0.0,
            "spectral_spread": 0.0,
            "spectral_peak_freq": 0.0,
            "spectral_peak_mag": 0.0,
            "total_spectral_power": 0.0,
            "spectral_entropy": 0.0,
        }

    # Spectral centroid (weighted mean frequency)
    spectral_centroid = float(np.sum(freqs * power) / total_power)

    # Spectral spread (weighted std of frequency)
    spectral_spread = float(
        np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * power) / total_power)
    )

    # Peak frequency
    peak_idx = int(np.argmax(fft_mag))
    spectral_peak_freq = float(freqs[peak_idx])
    spectral_peak_mag = float(fft_mag[peak_idx])

    # Spectral entropy
    p = power / total_power
    p = p[p > 0]
    spectral_entropy = float(-np.sum(p * np.log2(p)))

    return {
        "spectral_centroid": spectral_centroid,
        "spectral_spread": spectral_spread,
        "spectral_peak_freq": spectral_peak_freq,
        "spectral_peak_mag": spectral_peak_mag,
        "total_spectral_power": total_power,
        "spectral_entropy": spectral_entropy,
    }
