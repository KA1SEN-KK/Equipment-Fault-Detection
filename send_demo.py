"""Demo sender — publishes CMAPSS test data cycle-by-cycle over MQTT.

Run this on the SENDING machine:
    python send_demo.py --broker <receiver_ip> --unit 1 --interval 1.0

Each message published to topic  efd/v1/sensors/cycle  looks like:
    {"engine_id": "unit-1", "cycle": 42, "sensors": [s1, s2, ..., s14]}

Requirements on sending machine:
    pip install paho-mqtt pandas numpy
"""
import argparse
import json
import time
import importlib
import numpy as np

SENSOR_COLS = [
    "s2","s3","s4","s7","s8","s9","s11","s12","s13","s14","s15","s17","s20","s21"
]
COL_NAMES = (
    ["unit","cycle","op1","op2","op3"] + [f"s{i}" for i in range(1,22)]
)
TOPIC = "efd/v1/sensors/cycle"


def load_engine(data_path: str, unit: int):
    import pandas as pd
    df = pd.read_csv(data_path, sep=r"\s+", header=None, names=COL_NAMES)
    grp = df[df["unit"] == unit].sort_values("cycle")
    if grp.empty:
        raise ValueError(f"Unit {unit} not found in {data_path}")
    sensors = grp[SENSOR_COLS].to_numpy(dtype=float)
    cycles  = grp["cycle"].tolist()
    return sensors, cycles


def main():
    p = argparse.ArgumentParser(description="MQTT demo sender")
    p.add_argument("--broker",   default="127.0.0.1", help="Receiver broker IP")
    p.add_argument("--port",     type=int, default=1883)
    p.add_argument("--data",     default="CMAPSSData/test_FD001.txt")
    p.add_argument("--unit",     type=int, default=1, help="Engine unit ID to send")
    p.add_argument("--interval", type=float, default=1.0, help="Seconds between cycles")
    p.add_argument("--topic",    default=TOPIC)
    args = p.parse_args()

    mqtt = importlib.import_module("paho.mqtt.client")
    try:
        # paho-mqtt v2
        client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id="efd-sender",
            protocol=mqtt.MQTTv311,
        )
    except AttributeError:
        # paho-mqtt v1 fallback
        client = mqtt.Client(client_id="efd-sender", protocol=mqtt.MQTTv311)
    print(f"Connecting to broker {args.broker}:{args.port} ...")
    client.connect(args.broker, args.port, keepalive=60)
    client.loop_start()
    time.sleep(0.5)

    sensors, cycles = load_engine(args.data, args.unit)
    print(f"Loaded unit {args.unit}: {len(cycles)} cycles  →  topic: {args.topic}")
    print(f"Sending one cycle every {args.interval}s  (Ctrl+C to stop)\n")

    for i, (row, cyc) in enumerate(zip(sensors, cycles)):
        payload = {
            "engine_id": f"unit-{args.unit}",
            "cycle":     int(cyc),
            "sensors":   [round(float(v), 6) for v in row],
        }
        client.publish(args.topic, json.dumps(payload), qos=1)
        bar = "#" * min(i + 1, 30) + "-" * max(0, 30 - i - 1)
        status = "WARMUP" if i < 30 else "INFER "
        print(f"[{status}] cycle {cyc:4d}  [{bar}]  ({i+1}/{len(cycles)})")
        time.sleep(args.interval)

    client.loop_stop()
    client.disconnect()
    print("\nAll cycles sent.")


if __name__ == "__main__":
    main()
