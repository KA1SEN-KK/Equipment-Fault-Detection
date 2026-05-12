"""CMAPSS fault detection training pipeline.

Strategy
--------
1. Load train_FD00X.txt
2. Use only "healthy" cycles (RUL >= healthy_rul_min) to train LSTM AE and IsolationForest.
3. Threshold = 99th percentile of healthy reconstruction error.
4. Save model / scaler / meta to artifacts_cmapss_fd001/ (or chosen out_dir).

Usage
-----
python -m training.cmapss_fault_detection train \
    --data_dir CMAPSSData --subset FD001 --out_dir artifacts_cmapss_fd001

python -m training.cmapss_fault_detection eval \
    --data_dir CMAPSSData --subset FD001 --artifact_dir artifacts_cmapss_fd001
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models

from utils.cmapss_loader import (
    load_cmapss,
    add_rul,
    build_windows_from_df,
    N_FEATURES,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# LSTM Autoencoder  (multivariate)
# ──────────────────────────────────────────────

def build_lstm_ae(win: int, n_feat: int, latent: int = 32) -> tf.keras.Model:
    inp = layers.Input(shape=(win, n_feat))
    x = layers.LSTM(64, return_sequences=True)(inp)
    x = layers.LSTM(latent)(x)
    x = layers.RepeatVector(win)(x)
    x = layers.LSTM(64, return_sequences=True)(x)
    out = layers.TimeDistributed(layers.Dense(n_feat))(x)
    m = models.Model(inp, out)
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mae")
    return m


def reconstruction_error(model: tf.keras.Model, x: np.ndarray, batch: int = 256) -> np.ndarray:
    pred = model.predict(x, batch_size=batch, verbose=0)
    return np.mean(np.abs(pred - x), axis=(1, 2))


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    train_path = data_dir / f"train_{args.subset}.txt"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading %s ...", train_path)
    df = add_rul(load_cmapss(train_path))

    # healthy windows only for training
    healthy_wins, _ = build_windows_from_df(
        df,
        win=args.win,
        step=args.step,
        healthy_rul_min=args.healthy_rul_min,
        max_windows=args.max_windows,
    )
    logger.info("Healthy windows: %d  shape=%s", len(healthy_wins), healthy_wins.shape)

    x_train, x_val = train_test_split(healthy_wins, test_size=0.2, random_state=42)

    # ── Scaler (fit on flattened, apply per-feature across time axis) ──
    T, F = x_train.shape[1], x_train.shape[2]
    flat_train = x_train.reshape(-1, F)
    scaler = StandardScaler()
    scaler.fit(flat_train)

    def scale(x: np.ndarray) -> np.ndarray:
        sh = x.shape
        return scaler.transform(x.reshape(-1, F)).reshape(sh)

    x_train_s = scale(x_train)
    x_val_s = scale(x_val)

    # ── LSTM AE ──
    logger.info("Training LSTM Autoencoder ...")
    ae = build_lstm_ae(args.win, N_FEATURES, latent=args.latent)
    ae.fit(
        x_train_s, x_train_s,
        validation_data=(x_val_s, x_val_s),
        epochs=args.epochs,
        batch_size=args.batch,
        verbose=1,
    )
    val_err = reconstruction_error(ae, x_val_s)
    threshold = float(np.quantile(val_err, args.threshold_q))
    logger.info("LSTM AE threshold (q=%.2f): %.6f", args.threshold_q, threshold)

    ae.save(out_dir / "lstm_ae_model.keras")
    joblib.dump(scaler, out_dir / "lstm_ae_scaler.pkl")
    meta = {
        "subset": args.subset,
        "win": args.win,
        "step": args.step,
        "latent": args.latent,
        "threshold": threshold,
        "threshold_q": args.threshold_q,
        "healthy_rul_min": args.healthy_rul_min,
        "n_features": N_FEATURES,
    }
    (out_dir / "lstm_ae_meta.json").write_text(json.dumps(meta, indent=2))

    # ── IsolationForest on flattened windows ──
    logger.info("Training IsolationForest ...")
    flat_train_s = x_train_s.reshape(len(x_train_s), -1)
    iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    iso.fit(flat_train_s)

    flat_val_s = x_val_s.reshape(len(x_val_s), -1)
    iso_scores = iso.decision_function(flat_val_s)
    iso_threshold = float(np.quantile(iso_scores, 0.05))
    logger.info("IsolationForest threshold (5th pct): %.6f", iso_threshold)

    joblib.dump(iso, out_dir / "isolation_forest.pkl")
    iso_meta = {
        "subset": args.subset,
        "win": args.win,
        "step": args.step,
        "threshold": iso_threshold,
        "n_features": N_FEATURES,
    }
    (out_dir / "isolation_forest_meta.json").write_text(json.dumps(iso_meta, indent=2))

    logger.info("All artifacts saved to %s", out_dir)


# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────

def evaluate(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    test_path = data_dir / f"test_{args.subset}.txt"
    art_dir = Path(args.artifact_dir)

    # test set has no RUL column — we use RUL labels from RUL_FD00X.txt
    rul_path = data_dir / f"RUL_{args.subset}.txt"
    df_test = load_cmapss(test_path)

    logger.info("Loading LSTM AE artifacts ...")
    ae = tf.keras.models.load_model(art_dir / "lstm_ae_model.keras")
    scaler: StandardScaler = joblib.load(art_dir / "lstm_ae_scaler.pkl")
    meta = json.loads((art_dir / "lstm_ae_meta.json").read_text())
    win = int(meta["win"])
    step = int(meta["step"])
    threshold = float(meta["threshold"])
    F = int(meta["n_features"])

    def scale(x: np.ndarray) -> np.ndarray:
        sh = x.shape
        return scaler.transform(x.reshape(-1, F)).reshape(sh)

    # use last window of each engine (most degraded state)
    results = []
    for unit_id, grp in df_test.groupby("unit"):
        grp = grp.sort_values("cycle")
        arr = grp[list(grp.columns[2:])].to_numpy(dtype=np.float32)
        if len(arr) < win:
            continue
        window = arr[-win:][None]  # shape (1, win, F)
        window_s = scale(window)
        err = float(reconstruction_error(ae, window_s)[0])
        alert = err > threshold
        results.append({"unit": unit_id, "recon_err": err, "alert": alert})

    alerts = sum(r["alert"] for r in results)
    logger.info("Test engines: %d  Alerts: %d  (threshold=%.6f)", len(results), alerts, threshold)
    summary = {
        "total_engines": len(results),
        "alerts": alerts,
        "alert_rate": alerts / max(len(results), 1),
        "threshold": threshold,
    }
    print(json.dumps(summary, indent=2))


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CMAPSS fault detection training/eval")
    sub = p.add_subparsers(dest="mode", required=True)

    tr = sub.add_parser("train")
    tr.add_argument("--data_dir", default="CMAPSSData")
    tr.add_argument("--subset", default="FD001", choices=["FD001","FD002","FD003","FD004"])
    tr.add_argument("--out_dir", default="artifacts_cmapss_fd001")
    tr.add_argument("--win", type=int, default=30)
    tr.add_argument("--step", type=int, default=1)
    tr.add_argument("--latent", type=int, default=16)
    tr.add_argument("--epochs", type=int, default=20)
    tr.add_argument("--batch", type=int, default=256)
    tr.add_argument("--threshold_q", type=float, default=0.99)
    tr.add_argument("--healthy_rul_min", type=int, default=50,
                    help="Only train on cycles with RUL >= this value")
    tr.add_argument("--max_windows", type=int, default=50000)

    ev = sub.add_parser("eval")
    ev.add_argument("--data_dir", default="CMAPSSData")
    ev.add_argument("--subset", default="FD001", choices=["FD001","FD002","FD003","FD004"])
    ev.add_argument("--artifact_dir", default="artifacts_cmapss_fd001")

    return p.parse_args()


if __name__ == "__main__":
    tf.get_logger().setLevel("ERROR")
    args = parse_args()
    if args.mode == "train":
        train(args)
    else:
        evaluate(args)
