"""Real-time streaming inference engine with a rolling window buffer."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from models.base import DecisionContext, ModelResult
from models.cmapss_lstm_ae_runner import CMAPSSLSTMAERunner
from models.cmapss_rul_runner import CMAPSSRULRunner
from utils.cmapss_loader import N_FEATURES


@dataclass
class CycleResult:
    """Output produced after each new cycle is fed into the engine."""
    cycle_index: int
    buffer_fill: int          # how many cycles are in the buffer (0..win)
    ready: bool               # True once buffer has seen >= win cycles
    detection: Optional[ModelResult] = None
    rul: Optional[ModelResult] = None
    alert: bool = False       # True if detection is fault/critical
    alert_change: bool = False  # True if alert status flipped vs previous cycle


class StreamingEngine:
    """
    Rolling-window streaming inference engine.

    Usage
    -----
    engine = StreamingEngine(det_runner, rul_runner, win=30)

    for sensor_row in live_data_source:          # shape: (N_FEATURES,)
        result = engine.feed(sensor_row)
        if result.ready:
            print(result.rul.score, result.detection.label)

    Design
    ------
    Internally holds a deque(maxlen=win).  Each call to feed() appends one
    cycle row.  Once the buffer is full, both models are called on the current
    window and a CycleResult is returned.  The deque automatically evicts the
    oldest row when a new one is appended.
    """

    def __init__(
        self,
        det_runner: CMAPSSLSTMAERunner,
        rul_runner: CMAPSSRULRunner,
        win: int = 30,
        sensor_id: str = "stream-engine",
    ):
        self.det_runner = det_runner
        self.rul_runner = rul_runner
        self.win = win
        self.ctx = DecisionContext(
            sensor_id=sensor_id,
            frequency_hz=0.0,
            feature_schema=[f"s{i}" for i in range(N_FEATURES)],
        )
        self._buffer: deque[np.ndarray] = deque(maxlen=win)
        self._cycle_index: int = 0
        self._prev_alert: bool = False

    # ------------------------------------------------------------------ #
    def feed(self, sensor_row: np.ndarray) -> CycleResult:
        """
        Feed one cycle of sensor data (shape: (N_FEATURES,)) and get a result.

        Returns a CycleResult immediately.  If the buffer is not yet full,
        result.ready is False and detection/rul are None.
        """
        row = np.asarray(sensor_row, dtype=np.float32).reshape(-1)
        if len(row) != N_FEATURES:
            raise ValueError(f"Expected {N_FEATURES} sensor values, got {len(row)}")

        self._buffer.append(row)
        self._cycle_index += 1
        fill = len(self._buffer)

        if fill < self.win:
            return CycleResult(
                cycle_index=self._cycle_index,
                buffer_fill=fill,
                ready=False,
            )

        window = np.stack(list(self._buffer), axis=0)  # (win, N_FEATURES)

        det_result = self.det_runner.predict(window, self.ctx)
        rul_result = self.rul_runner.predict(window, self.ctx)

        alert = det_result.label in ("fault", "critical")
        alert_change = alert != self._prev_alert
        self._prev_alert = alert

        return CycleResult(
            cycle_index=self._cycle_index,
            buffer_fill=fill,
            ready=True,
            detection=det_result,
            rul=rul_result,
            alert=alert,
            alert_change=alert_change,
        )

    def reset(self) -> None:
        """Clear the buffer and reset state (start a new engine session)."""
        self._buffer.clear()
        self._cycle_index = 0
        self._prev_alert = False

    @property
    def buffer_fill(self) -> int:
        return len(self._buffer)

    @property
    def is_ready(self) -> bool:
        return len(self._buffer) >= self.win
