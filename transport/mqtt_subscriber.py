"""MQTT subscriber — receives sensor cycles and feeds StreamingEngine.

Each MQTT message payload is a JSON object:
    {"engine_id": "unit-1", "cycle": 42, "sensors": [s1, s2, ..., s14]}

When the StreamingEngine buffer is full (>= win cycles), inference runs
automatically and results are published back to a result topic via
the existing MQTTForwarder.

Usage (standalone)
------------------
python -m transport.mqtt_subscriber \
    --broker 127.0.0.1 --port 1883 \
    --detection_dir artifacts_cmapss_fd001 \
    --rul_dir artifacts_cmapss_rul_fd001

Simulating data (in another terminal)
--------------------------------------
python -m transport.mqtt_subscriber simulate \
    --data_dir CMAPSSData --subset FD001 --unit 1 \
    --broker 127.0.0.1 --interval 0.5
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import threading
import time
from pathlib import Path
from queue import Queue
from typing import Optional

import numpy as np

from stream.engine import StreamingEngine
from models.cmapss_lstm_ae_runner import CMAPSSLSTMAERunner
from models.cmapss_rul_runner import CMAPSSRULRunner
from models import ModelConfig
from transport.mqtt_forwarder import MQTTForwarder, MQTTConfig
from utils.cmapss_loader import N_FEATURES, USEFUL_SENSORS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SENSOR_TOPIC  = "efd/v1/sensors/cycle"     # incoming sensor data
RESULT_TOPIC  = "efd/v1/inference/result"  # outgoing inference results


# ──────────────────────────────────────────────
# Subscriber
# ──────────────────────────────────────────────

class SensorSubscriber:
    """
    Subscribes to SENSOR_TOPIC, feeds each cycle into StreamingEngine,
    and publishes inference results to RESULT_TOPIC.
    """

    def __init__(
        self,
        engine: StreamingEngine,
        forwarder: MQTTForwarder,
        broker_host: str = "127.0.0.1",
        broker_port: int = 1883,
    ):
        self.engine = engine
        self.forwarder = forwarder
        self.broker_host = broker_host
        self.broker_port = broker_port
        self._client = None
        self._running = False

    def start(self) -> None:
        """Connect and start the subscriber loop (blocking)."""
        try:
            mqtt = importlib.import_module("paho.mqtt.client")
        except ImportError as exc:
            raise ImportError("pip install paho-mqtt") from exc

        client = mqtt.Client(client_id="efd-subscriber", protocol=mqtt.MQTTv311)
        client.on_connect    = self._on_connect
        client.on_message    = self._on_message
        client.on_disconnect = self._on_disconnect
        self._client = client
        self._running = True

        logger.info("Connecting to broker %s:%s", self.broker_host, self.broker_port)
        client.connect(self.broker_host, self.broker_port, keepalive=60)
        client.loop_forever()   # blocks until stop() is called

    def stop(self) -> None:
        self._running = False
        if self._client:
            self._client.disconnect()

    # ── Callbacks ──

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            client.subscribe(SENSOR_TOPIC, qos=1)
            logger.info("Subscribed to %s", SENSOR_TOPIC)
        else:
            logger.error("MQTT connect failed rc=%s", rc)

    def _on_disconnect(self, client, userdata, rc):
        if rc != 0:
            logger.warning("Unexpected disconnect rc=%s", rc)

    def _on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            sensors = np.array(payload["sensors"], dtype=np.float32)
            if len(sensors) != N_FEATURES:
                logger.warning("Expected %d sensors, got %d — skipping", N_FEATURES, len(sensors))
                return

            result = self.engine.feed(sensors)

            if result.ready:
                out = {
                    "cycle_index":  result.cycle_index,
                    "engine_id":    payload.get("engine_id", "unknown"),
                    "detection":    result.detection.label,
                    "recon_err":    result.detection.raw.get("recon_err"),
                    "rul_cycles":   round(result.rul.score, 2),
                    "rul_status":   result.rul.label,
                    "alert":        result.alert,
                    "alert_change": result.alert_change,
                }
                self.forwarder.publish_json(RESULT_TOPIC, out)
                logger.info(
                    "cycle=%d  det=%s  RUL=%.1f  alert=%s",
                    result.cycle_index, result.detection.label,
                    result.rul.score, result.alert,
                )
            else:
                logger.debug("Buffer filling: %d/%d", result.buffer_fill, self.engine.win)

        except Exception as exc:
            logger.error("Error processing message: %s", exc)


# ──────────────────────────────────────────────
# Data simulator (publishes test set cycle by cycle)
# ──────────────────────────────────────────────

def simulate(args: argparse.Namespace) -> None:
    """Publish CMAPSS test data cycle-by-cycle to SENSOR_TOPIC."""
    from utils.cmapss_loader import load_cmapss

    df = load_cmapss(Path(args.data_dir) / f"test_{args.subset}.txt")
    grp = df[df["unit"] == args.unit].sort_values("cycle")
    sensor_arr = grp[USEFUL_SENSORS].to_numpy(dtype=np.float32)

    cfg = MQTTConfig(host=args.broker, port=args.port)
    fwd = MQTTForwarder(cfg)
    fwd.connect()

    logger.info("Simulating engine %d  (%d cycles)  interval=%.2fs",
                args.unit, len(sensor_arr), args.interval)

    for i, row in enumerate(sensor_arr):
        payload = {
            "engine_id": f"unit-{args.unit}",
            "cycle": int(grp.iloc[i]["cycle"]),
            "sensors": row.tolist(),
        }
        fwd.publish_json(SENSOR_TOPIC, payload)
        logger.info("Published cycle %d", payload["cycle"])
        time.sleep(args.interval)

    fwd.disconnect()
    logger.info("Simulation complete.")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode")

    # ── subscribe mode ──
    sv = sub.add_parser("subscribe", help="Start MQTT subscriber (default)")
    sv.add_argument("--broker", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=1883)
    sv.add_argument("--detection_dir", default="artifacts_cmapss_fd001")
    sv.add_argument("--rul_dir", default="artifacts_cmapss_rul_fd001")

    # ── simulate mode ──
    sm = sub.add_parser("simulate", help="Publish test data cycle-by-cycle")
    sm.add_argument("--data_dir", default="CMAPSSData")
    sm.add_argument("--subset", default="FD001")
    sm.add_argument("--unit", type=int, default=1)
    sm.add_argument("--broker", default="127.0.0.1")
    sm.add_argument("--port", type=int, default=1883)
    sm.add_argument("--interval", type=float, default=0.5,
                    help="Seconds between cycles")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "simulate":
        simulate(args)
        return

    # Default: subscribe
    det_runner = CMAPSSLSTMAERunner(
        ModelConfig(name="cmapss_lstm_ae", model_path=Path(args.detection_dir))
    )
    rul_runner = CMAPSSRULRunner(
        ModelConfig(name="cmapss_rul", model_path=Path(args.rul_dir))
    )
    engine = StreamingEngine(det_runner, rul_runner)

    cfg = MQTTConfig(host=args.broker, port=args.port)
    forwarder = MQTTForwarder(cfg)
    forwarder.connect()

    subscriber = SensorSubscriber(engine, forwarder, args.broker, args.port)
    try:
        subscriber.start()
    except KeyboardInterrupt:
        subscriber.stop()
        forwarder.disconnect()


if __name__ == "__main__":
    main()
