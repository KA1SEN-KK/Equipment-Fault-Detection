"""CMAPSS RUL (Remaining Useful Life) prediction training pipeline.

Strategy
--------
1. Load train_FD00X.txt, compute piecewise-linear RUL label (capped at rul_cap).
2. Build sliding windows (win cycles × 14 sensors) → label = RUL at last cycle of window.
3. Train LSTM regression model to predict scalar RUL.
4. Evaluate on test set using official RUL_FD00X.txt ground truth.
5. Save model / scaler / meta to artifacts_cmapss_rul_fd001/.

Usage
-----
python -m training.cmapss_rul train \
    --data_dir CMAPSSData --subset FD001 --out_dir artifacts_cmapss_rul_fd001

python -m training.cmapss_rul eval \
    --data_dir CMAPSSData --subset FD001 --artifact_dir artifacts_cmapss_rul_fd001
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models

from utils.cmapss_loader import load_cmapss, add_rul, USEFUL_SENSORS, N_FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Label: piecewise-linear RUL
# ──────────────────────────────────────────────

def clip_rul(df, rul_cap: int = 125):
    """Cap RUL at rul_cap — sensors don't show degradation in very early cycles."""
    df = df.copy()
    df["rul"] = df["rul"].clip(upper=rul_cap)
    return df


# ──────────────────────────────────────────────
# Dataset builder
# ──────────────────────────────────────────────

def build_rul_dataset(df, win: int, step: int, max_windows: int | None = None):
    """
    Returns
    -------
    X : (N, win, N_FEATURES)  float32
    y : (N,)                  float32  — RUL at last cycle of each window
    """
    X_list, y_list = [], []
    for _, grp in df.groupby("unit"):
        grp = grp.sort_values("cycle")
        sensors = grp[USEFUL_SENSORS].to_numpy(dtype=np.float32)
        ruls = grp["rul"].to_numpy(dtype=np.float32)

        T = len(sensors)
        n = (T - win) // step + 1
        if n <= 0:
            continue
        for i in range(n):
            start = i * step
            end = start + win
            X_list.append(sensors[start:end])
            y_list.append(ruls[end - 1])

        if max_windows and sum(len(x) for x in [X_list]) >= max_windows:
            break

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.float32)

    if max_windows:
        X = X[:max_windows]
        y = y[:max_windows]
    return X, y


# ──────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────

def build_rul_lstm(win: int, n_feat: int) -> tf.keras.Model:
    inp = layers.Input(shape=(win, n_feat))
    x = layers.LSTM(64, return_sequences=True)(inp)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(16, activation="relu")(x)
    out = layers.Dense(1, activation="relu")(x)   # RUL >= 0
    m = models.Model(inp, out)
    m.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.Huber(),
        metrics=["mae"],
    )
    return m


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading %s/train_%s.txt ...", data_dir, args.subset)
    df = clip_rul(add_rul(load_cmapss(data_dir / f"train_{args.subset}.txt")), args.rul_cap)

    logger.info("Building windows (win=%d, step=%d) ...", args.win, args.step)
    X, y = build_rul_dataset(df, win=args.win, step=args.step, max_windows=args.max_windows)
    logger.info("Dataset: X=%s  y=%s  RUL range=[%.0f, %.0f]", X.shape, y.shape, y.min(), y.max())

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale per-feature across time axis
    flat = X_train.reshape(-1, N_FEATURES)
    scaler = StandardScaler()
    scaler.fit(flat)

    def scale(x):
        sh = x.shape
        return scaler.transform(x.reshape(-1, N_FEATURES)).reshape(sh)

    X_train_s = scale(X_train)
    X_val_s = scale(X_val)

    logger.info("Training LSTM RUL model ...")
    model = build_rul_lstm(args.win, N_FEATURES)
    cb = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_mae"),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, monitor="val_mae"),
    ]
    model.fit(
        X_train_s, y_train,
        validation_data=(X_val_s, y_val),
        epochs=args.epochs,
        batch_size=args.batch,
        callbacks=cb,
        verbose=1,
    )

    val_pred = model.predict(X_val_s, verbose=0).flatten()
    val_mae = float(np.mean(np.abs(val_pred - y_val)))
    val_rmse = float(np.sqrt(np.mean((val_pred - y_val) ** 2)))
    logger.info("Val MAE=%.2f  RMSE=%.2f", val_mae, val_rmse)

    model.save(out_dir / "rul_lstm_model.keras")
    joblib.dump(scaler, out_dir / "rul_scaler.pkl")
    meta = {
        "subset": args.subset,
        "win": args.win,
        "step": args.step,
        "rul_cap": args.rul_cap,
        "n_features": N_FEATURES,
        "val_mae": val_mae,
        "val_rmse": val_rmse,
    }
    (out_dir / "rul_meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("Saved to %s  (val_mae=%.2f)", out_dir, val_mae)


# ──────────────────────────────────────────────
# Evaluation on official test set
# ──────────────────────────────────────────────

def evaluate(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    art_dir = Path(args.artifact_dir)

    model = tf.keras.models.load_model(art_dir / "rul_lstm_model.keras")
    scaler: StandardScaler = joblib.load(art_dir / "rul_scaler.pkl")
    meta = json.loads((art_dir / "rul_meta.json").read_text())
    win = int(meta["win"])

    df_test = load_cmapss(data_dir / f"test_{args.subset}.txt")
    true_rul = np.loadtxt(data_dir / f"RUL_{args.subset}.txt", dtype=np.float32)

    preds = []
    for unit_id, grp in df_test.groupby("unit"):
        grp = grp.sort_values("cycle")
        arr = grp[USEFUL_SENSORS].to_numpy(dtype=np.float32)
        if len(arr) < win:
            arr = np.pad(arr, ((win - len(arr), 0), (0, 0)), mode="edge")
        window = arr[-win:][None]
        sh = window.shape
        window_s = scaler.transform(window.reshape(-1, N_FEATURES)).reshape(sh)
        pred = float(model.predict(window_s, verbose=0)[0][0])
        preds.append(max(pred, 0.0))

    preds = np.array(preds, dtype=np.float32)
    mae = float(np.mean(np.abs(preds - true_rul)))
    rmse = float(np.sqrt(np.mean((preds - true_rul) ** 2)))

    # NASA scoring function (penalises late predictions more)
    diff = preds - true_rul
    nasa_score = float(np.sum(
        np.where(diff < 0, np.exp(-diff / 13) - 1, np.exp(diff / 10) - 1)
    ))

    result = {"test_mae": mae, "test_rmse": rmse, "nasa_score": nasa_score, "n_engines": len(preds)}
    print(json.dumps(result, indent=2))
    logger.info("Test MAE=%.2f  RMSE=%.2f  NASA_score=%.1f", mae, rmse, nasa_score)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CMAPSS RUL prediction")
    sub = p.add_subparsers(dest="mode", required=True)

    tr = sub.add_parser("train")
    tr.add_argument("--data_dir", default="CMAPSSData")
    tr.add_argument("--subset", default="FD001", choices=["FD001","FD002","FD003","FD004"])
    tr.add_argument("--out_dir", default="artifacts_cmapss_rul_fd001")
    tr.add_argument("--win", type=int, default=30)
    tr.add_argument("--step", type=int, default=1)
    tr.add_argument("--rul_cap", type=int, default=125)
    tr.add_argument("--epochs", type=int, default=50)
    tr.add_argument("--batch", type=int, default=256)
    tr.add_argument("--max_windows", type=int, default=None)

    ev = sub.add_parser("eval")
    ev.add_argument("--data_dir", default="CMAPSSData")
    ev.add_argument("--subset", default="FD001", choices=["FD001","FD002","FD003","FD004"])
    ev.add_argument("--artifact_dir", default="artifacts_cmapss_rul_fd001")

    return p.parse_args()


if __name__ == "__main__":
    tf.get_logger().setLevel("ERROR")
    args = parse_args()
    if args.mode == "train":
        train(args)
    else:
        evaluate(args)
