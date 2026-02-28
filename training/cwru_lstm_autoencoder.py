"""CWRU-based LSTM Autoencoder training pipeline.

Workflow (train):
1) Point to the CWRU root folder (contains 12k/48k subfolders + Normal Baseline Data).
2) Choose sampling group (12k Drive End, 12k Fan End, or 48k Drive End).
3) Train on normal windows, pick an error threshold from validation, save model + scaler + threshold.

Workflow (score):
1) Load model/scaler/threshold, stream sliding windows, output reconstruction error and alert level.

This is a baseline; adjust window length/step, model size, and threshold policy for your use case.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models

from utils.data_loader import (
    DatasetSpec,
    collect_signals,
    load_mat_signal,
    make_dataset,
    sliding_windows,
)


# ------------------------
# Model
# ------------------------

def build_lstm_ae(seq_len: int, latent: int = 32, dropout: float = 0.0) -> tf.keras.Model:
    inp = layers.Input(shape=(seq_len, 1))
    x = layers.LSTM(64, return_sequences=True)(inp)
    x = layers.LSTM(latent)(x)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)
    x = layers.RepeatVector(seq_len)(x)
    x = layers.LSTM(64, return_sequences=True)(x)
    out = layers.TimeDistributed(layers.Dense(1))(x)
    model = models.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mae")
    return model


# ------------------------
# Training pipeline
# ------------------------

def train_pipeline(
    normal_spec: DatasetSpec,
    val_fault_spec: DatasetSpec | None,
    win: int,
    step: int,
    latent: int,
    epochs: int,
    batch: int,
    max_normal_windows: int | None,
    max_val_windows: int | None,
    out_dir: Path,
    threshold_quantile: float,
    dropout: float,
) -> Dict[str, str | float]:
    out_dir.mkdir(parents=True, exist_ok=True)

    normal_signals = collect_signals(normal_spec)
    normal_windows = make_dataset(normal_signals, win, step, max_normal_windows)
    if len(normal_windows) == 0:
        raise RuntimeError("No normal windows available")

    x_train, x_val = train_test_split(normal_windows, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_val_s = scaler.transform(x_val)

    x_train_s = x_train_s.reshape(-1, win, 1)
    x_val_s = x_val_s.reshape(-1, win, 1)

    model = build_lstm_ae(win, latent=latent, dropout=dropout)
    model.fit(
        x_train_s, x_train_s,
        validation_data=(x_val_s, x_val_s),
        epochs=epochs,
        batch_size=batch,
        verbose=1,
    )

    # Threshold from validation reconstruction error
    val_pred = model.predict(x_val_s, batch_size=batch, verbose=0)
    val_err = np.mean(np.abs(val_pred.squeeze(-1) - x_val), axis=1)
    thr = float(np.quantile(val_err, threshold_quantile))

    # Optional: evaluate on fault windows for sanity
    fault_stats = {}
    if val_fault_spec is not None:
        fault_signals = collect_signals(val_fault_spec)
        fault_windows = make_dataset(fault_signals, win, step, max_val_windows)
        if len(fault_windows) > 0:
            fw_s = scaler.transform(fault_windows).reshape(-1, win, 1)
            fp = model.predict(fw_s, batch_size=batch, verbose=0)
            ferr = np.mean(np.abs(fp.squeeze(-1) - fault_windows), axis=1)
            fault_stats = {
                "fault_err_median": float(np.median(ferr)),
                "fault_err_mean": float(np.mean(ferr)),
                "fault_err_p95": float(np.quantile(ferr, 0.95)),
            }

    # Save artifacts
    model_path = out_dir / "lstm_ae_model.h5"
    scaler_path = out_dir / "lstm_ae_scaler.pkl"
    meta_path = out_dir / "lstm_ae_meta.json"
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    meta = {
        "win": win,
        "step": step,
        "latent": latent,
        "threshold": thr,
        "threshold_quantile": threshold_quantile,
        "normal_files": normal_spec.file_ids,
        "fault_files": val_fault_spec.file_ids if val_fault_spec else [],
    }
    meta.update(fault_stats)
    meta_path.write_text(json.dumps(meta, indent=2))

    return {
        "model": str(model_path),
        "scaler": str(scaler_path),
        "meta": str(meta_path),
        "threshold": thr,
    }


# ------------------------
# Scoring (offline or streaming)
# ------------------------

def load_artifacts(model_path: Path, scaler_path: Path, meta_path: Path):
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    meta = json.loads(meta_path.read_text())
    return model, scaler, meta


def score_windows(model, scaler, windows: np.ndarray, win: int) -> np.ndarray:
    xs = scaler.transform(windows).reshape(-1, win, 1)
    pred = model.predict(xs, verbose=0)
    err = np.mean(np.abs(pred.squeeze(-1) - windows), axis=1)
    return err


# ------------------------
# CLI
# ------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train/score LSTM AE on CWRU data")
    p.add_argument("mode", choices=["train", "score"], help="train or score")
    p.add_argument("--cwru_root", type=str, required=True, help="Path to CWRU root folder")
    p.add_argument("--group", type=str, default="12k_drive",
                   choices=["12k_drive", "12k_fan", "48k_drive"], help="Sampling group")
    p.add_argument("--channel", type=str, default="DE_time",
                   help="MAT key for channel (e.g., DE_time/FE_time)")
    p.add_argument("--win", type=int, default=2048, help="Window length")
    p.add_argument("--step", type=int, default=512, help="Window hop")
    p.add_argument("--latent", type=int, default=32, help="Latent size")
    p.add_argument("--epochs", type=int, default=10, help="Training epochs")
    p.add_argument("--batch", type=int, default=128, help="Batch size")
    p.add_argument("--threshold_q", type=float, default=0.99, help="Quantile for threshold")
    p.add_argument("--max_normal_windows", type=int, default=200000,
                   help="Cap normal windows for training")
    p.add_argument("--max_val_windows", type=int, default=20000,
                   help="Cap fault windows for validation stats")
    p.add_argument("--out_dir", type=str, default="artifacts_cwru_lstm_ae",
                   help="Output directory for artifacts")
    p.add_argument("--artifact_dir", type=str,
                   help="For scoring: directory containing model/scaler/meta")
    p.add_argument("--score_file", type=str, help="For scoring: a .mat file to score")
    p.add_argument("--score_limit", type=int, default=5000, help="For scoring: max windows")
    return p.parse_args()


def get_group_files(root: Path, group: str) -> Tuple[Path, list]:
    if group == "12k_drive":
        folder = root / "12k Drive End Bearing Fault Data"
    elif group == "12k_fan":
        folder = root / "12k Fan End Bearing Fault Data"
    else:
        folder = root / "48k Drive End Bearing Fault Data"
    return folder, sorted([f.stem for f in folder.glob("*.mat")])


def main() -> None:
    args = parse_args()
    root = Path(args.cwru_root)
    normal_folder = root / "Normal Baseline Data"
    normal_files = sorted([f.stem for f in normal_folder.glob("*.mat")])
    if args.mode == "train":
        group_folder, group_files = get_group_files(root, args.group)
        val_fault_ids = group_files[:5] if group_files else []
        normal_spec = DatasetSpec("normal", normal_folder, normal_files, args.channel)
        val_fault_spec = (
            DatasetSpec("fault_val", group_folder, val_fault_ids, args.channel)
            if val_fault_ids
            else None
        )
        paths = train_pipeline(
            normal_spec=normal_spec,
            val_fault_spec=val_fault_spec,
            win=args.win,
            step=args.step,
            latent=args.latent,
            epochs=args.epochs,
            batch=args.batch,
            max_normal_windows=args.max_normal_windows,
            max_val_windows=args.max_val_windows,
            out_dir=Path(args.out_dir),
            threshold_quantile=args.threshold_q,
            dropout=0.0,
        )
        print(json.dumps(paths, indent=2))
    else:
        if not args.artifact_dir or not args.score_file:
            raise SystemExit("score mode requires --artifact_dir and --score_file")
        adir = Path(args.artifact_dir)
        model, scaler, meta = load_artifacts(
            adir / "lstm_ae_model.h5",
            adir / "lstm_ae_scaler.pkl",
            adir / "lstm_ae_meta.json",
        )
        win = int(meta["win"])
        step = int(meta["step"])
        thr = float(meta["threshold"])
        sig = load_mat_signal(Path(args.score_file), args.channel)
        windows = make_dataset([sig], win, step, max_windows=args.score_limit)
        if len(windows) == 0:
            raise SystemExit("no windows to score")
        err = score_windows(model, scaler, windows, win)
        summary = {
            "mean_err": float(np.mean(err)),
            "p95_err": float(np.quantile(err, 0.95)),
            "alerts": int(np.sum(err > thr)),
            "total_windows": int(len(err)),
        }
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    tf.get_logger().setLevel("ERROR")
    main()
