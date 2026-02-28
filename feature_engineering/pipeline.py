"""Unified feature extraction pipeline."""
from __future__ import annotations

from typing import Dict

import numpy as np

from feature_engineering.time_domain import compute_time_features
from feature_engineering.frequency_domain import compute_frequency_features


class FeaturePipeline:
    """Extracts time-domain and frequency-domain features from raw vibration signals.

    Usage::

        pipe = FeaturePipeline(sampling_rate=12000.0)
        feats = pipe.extract(signal)
        summary = pipe.extract_summary(signal)   # compact version for LLM prompts
    """

    def __init__(self, sampling_rate: float = 12000.0):
        self.sampling_rate = sampling_rate

    def extract(
        self,
        signal: np.ndarray,
        include_time: bool = True,
        include_freq: bool = True,
    ) -> Dict[str, float]:
        """Extract all configured features from a 1D signal."""
        features: Dict[str, float] = {}
        if signal.ndim != 1:
            signal = signal.reshape(-1)

        if include_time:
            time_feats = compute_time_features(signal)
            features.update({f"time_{k}": v for k, v in time_feats.items()})

        if include_freq:
            freq_feats = compute_frequency_features(signal, self.sampling_rate)
            features.update({f"freq_{k}": v for k, v in freq_feats.items()})

        return features

    def extract_summary(self, signal: np.ndarray) -> Dict[str, float]:
        """Extract a compact summary suitable for LLM prompts."""
        full = self.extract(signal)
        keys = [
            "time_rms",
            "time_kurtosis",
            "time_crest_factor",
            "time_peak",
            "time_skewness",
            "freq_spectral_centroid",
            "freq_spectral_peak_freq",
            "freq_spectral_entropy",
        ]
        return {k: round(full[k], 6) for k in keys if k in full}
