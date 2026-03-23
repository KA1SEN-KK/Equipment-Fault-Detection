"""MQTT forwarding utilities for diagnosis events.

This module provides a lightweight MQTT publisher with:
- explicit connect/disconnect lifecycle
- retry with exponential backoff
- optional in-memory buffering when publish fails
- JSON payload publishing helper

Dependency:
    pip install paho-mqtt
"""
from __future__ import annotations

import json
import logging
import random
import time
import importlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class MQTTConfig:
    """Configuration for MQTT broker connectivity and publish behavior."""

    host: str = "127.0.0.1"
    port: int = 1883
    client_id: str = "equipment-fault-detection"
    username: Optional[str] = None
    password: Optional[str] = None

    keepalive: int = 60
    topic_prefix: str = "efd/v1"

    qos: int = 1
    retain: bool = False

    connect_timeout_sec: int = 10
    publish_retries: int = 3
    retry_backoff_sec: float = 0.5

    # TLS options
    use_tls: bool = False
    tls_ca_cert: Optional[str] = None
    tls_certfile: Optional[str] = None
    tls_keyfile: Optional[str] = None
    tls_insecure: bool = False


class MQTTForwarder:
    """A simple, resilient MQTT publisher for JSON messages."""

    def __init__(
        self,
        config: MQTTConfig,
        enable_buffer: bool = True,
        max_buffer_size: int = 1000,
    ):
        self.config = config
        self.enable_buffer = enable_buffer
        self.max_buffer_size = max_buffer_size

        self._client = None
        self._connected = False
        self._buffer: List[Tuple[str, str, int, bool]] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def connect(self) -> None:
        """Create and connect MQTT client."""
        if self._connected:
            return

        try:
            mqtt = importlib.import_module("paho.mqtt.client")
        except Exception as exc:  # pragma: no cover
            raise ImportError("paho-mqtt is required for MQTTForwarder") from exc

        client = mqtt.Client(client_id=self.config.client_id, protocol=mqtt.MQTTv311)

        if self.config.username:
            client.username_pw_set(self.config.username, self.config.password)

        if self.config.use_tls:
            client.tls_set(
                ca_certs=self.config.tls_ca_cert,
                certfile=self.config.tls_certfile,
                keyfile=self.config.tls_keyfile,
            )
            client.tls_insecure_set(self.config.tls_insecure)

        client.on_connect = self._on_connect
        client.on_disconnect = self._on_disconnect

        self._client = client

        logger.info("Connecting to MQTT broker %s:%s", self.config.host, self.config.port)
        client.connect(self.config.host, self.config.port, self.config.keepalive)
        client.loop_start()

        # Wait briefly for callback to update connected state
        deadline = time.time() + self.config.connect_timeout_sec
        while time.time() < deadline and not self._connected:
            time.sleep(0.05)

        if not self._connected:
            raise TimeoutError("MQTT connect timeout")

        self.flush_buffer()

    def disconnect(self) -> None:
        """Disconnect MQTT client and stop network loop."""
        if self._client is None:
            return
        try:
            self._client.loop_stop()
            self._client.disconnect()
        finally:
            self._connected = False
            self._client = None

    # ------------------------------------------------------------------
    # Publish
    # ------------------------------------------------------------------
    def publish_json(
        self,
        topic: str,
        payload: Dict[str, Any],
        qos: Optional[int] = None,
        retain: Optional[bool] = None,
    ) -> bool:
        """Publish a JSON payload to an MQTT topic.

        Returns True on success, False on failure.
        """
        json_payload = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        return self.publish_text(topic, json_payload, qos=qos, retain=retain)

    def publish_text(
        self,
        topic: str,
        payload: str,
        qos: Optional[int] = None,
        retain: Optional[bool] = None,
    ) -> bool:
        """Publish a text payload with retry logic."""
        if qos is None:
            qos = self.config.qos
        if retain is None:
            retain = self.config.retain

        if self._client is None or not self._connected:
            try:
                self.connect()
            except Exception as exc:
                logger.warning("MQTT not connected and reconnect failed: %s", exc)
                self._enqueue(topic, payload, qos, retain)
                return False

        for attempt in range(self.config.publish_retries + 1):
            try:
                info = self._client.publish(topic, payload, qos=qos, retain=retain)
                info.wait_for_publish(timeout=self.config.connect_timeout_sec)
                if info.rc == 0:
                    return True
            except Exception as exc:
                logger.warning("MQTT publish attempt %s failed: %s", attempt + 1, exc)

            backoff = self.config.retry_backoff_sec * (2 ** attempt)
            jitter = random.uniform(0.0, 0.1)
            time.sleep(backoff + jitter)

        self._enqueue(topic, payload, qos, retain)
        return False

    def flush_buffer(self) -> int:
        """Try to publish all buffered messages. Returns flushed count."""
        if not self._buffer:
            return 0

        flushed = 0
        pending = list(self._buffer)
        self._buffer.clear()

        for topic, payload, qos, retain in pending:
            ok = self.publish_text(topic, payload, qos=qos, retain=retain)
            if ok:
                flushed += 1
            else:
                # publish_text already re-enqueues on failure
                pass
        return flushed

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def build_topic(
        self,
        site: str,
        line: str,
        asset: str,
        event_type: str = "diagnosis",
    ) -> str:
        """Build topic as: <prefix>/<site>/<line>/<asset>/<event_type>."""
        prefix = self.config.topic_prefix.strip("/")
        return f"{prefix}/{site}/{line}/{asset}/{event_type}"

    def _enqueue(self, topic: str, payload: str, qos: int, retain: bool) -> None:
        if not self.enable_buffer:
            return
        if len(self._buffer) >= self.max_buffer_size:
            self._buffer.pop(0)
        self._buffer.append((topic, payload, qos, retain))

    def _on_connect(self, _client, _userdata, _flags, rc):
        self._connected = (rc == 0)
        if self._connected:
            logger.info("MQTT connected")
        else:
            logger.warning("MQTT connect failed, rc=%s", rc)

    def _on_disconnect(self, _client, _userdata, rc):
        self._connected = False
        if rc != 0:
            logger.warning("MQTT unexpected disconnect, rc=%s", rc)
