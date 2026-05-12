"""CMAPSS fault detection training — 4-model ensemble.

Models
------
1. LSTM Autoencoder   — unsupervised, reconstruction error
2. Isolation Forest   — unsupervised
3. One-Class SVM      — unsupervised
4. Random Forest      — supervised (label: RUL < 30 = fault)

AUC-weighted ensemble scores are saved to ensemble_meta.json.

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
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
import tensorflow as tf
from tensorflow.keras import layers, models

from utils.cmapss_loader import (
    load_cmapss, add_rul, build_windows_from_df, N_FEATURES,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# LSTM Autoencoder
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading %s ...", data_dir / f"train_{args.subset}.txt")
    df = add_rul(load_cmapss(data_dir / f"train_{args.subset}.txt"))

    # ── Healthy windows — used to fit unsupervised models ──
    healthy_wins, _ = build_windows_from_df(
        df, win=args.win, step=args.step,
        healthy_rul_min=args.healthy_rul_min,
        max_windows=args.max_windows,
    )
    logger.info("Healthy windows: %d", len(healthy_wins))
    x_train, x_val = train_test_split(healthy_wins, test_size=0.2, random_state=42)

    # ── Labeled windows (all) — used for RF + AUC eval ──
    all_wins, all_labels = build_windows_from_df(
        df, win=args.win, step=args.step, healthy_rul_min=None,
    )
    fault_rate = 100.0 * all_labels.mean()
    logger.info("Labeled windows: %d  fault_rate=%.1f%%", len(all_wins), fault_rate)
    X_lbl_tr, X_lbl_val, y_lbl_tr, y_lbl_val = train_test_split(
        all_wins, all_labels, test_size=0.2, random_state=42, stratify=all_labels,
    )

    # ── Scaler (fit on healthy training windows only) ──
    F = N_FEATURES
    scaler = StandardScaler()
    scaler.fit(x_train.reshape(-1, F))
    joblib.dump(scaler, out_dir / "lstm_ae_scaler.pkl")

    def scale(x: np.ndarray) -> np.ndarray:
        sh = x.shape
        return scaler.transform(x.reshape(-1, F)).reshape(sh)

    x_train_s   = scale(x_train)
    x_val_s     = scale(x_val)
    X_lbl_tr_s  = scale(X_lbl_tr)
    X_lbl_val_s = scale(X_lbl_val)

    flat_h_tr  = x_train_s.reshape(len(x_train_s), -1)       # healthy flat, for unsup fit
    flat_lbl_tr  = X_lbl_tr_s.reshape(len(X_lbl_tr_s), -1)   # all labeled flat, for RF
    flat_lbl_val = X_lbl_val_s.reshape(len(X_lbl_val_s), -1)  # for AUC eval

    # ── 1. LSTM Autoencoder ──
    logger.info("Training LSTM Autoencoder (%d epochs) ...", args.epochs)
    ae = build_lstm_ae(args.win, N_FEATURES, latent=args.latent)
    ae.fit(
        x_train_s, x_train_s,
        validation_data=(x_val_s, x_val_s),
        epochs=args.epochs, batch_size=args.batch, verbose=1,
    )
    ae_threshold = float(np.quantile(reconstruction_error(ae, x_val_s), args.threshold_q))
    logger.info("LSTM AE threshold (q=%.2f): %.6f", args.threshold_q, ae_threshold)
    ae.save(out_dir / "lstm_ae_model.keras")
    (out_dir / "lstm_ae_meta.json").write_text(json.dumps({
        "subset": args.subset, "win": args.win, "step": args.step,
        "latent": args.latent, "threshold": ae_threshold,
        "threshold_q": args.threshold_q,
        "healthy_rul_min": args.healthy_rul_min,
        "n_features": N_FEATURES,
    }, indent=2))

    # ── 2. Isolation Forest ──
    logger.info("Training Isolation Forest ...")
    iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    iso.fit(flat_h_tr)
    joblib.dump(iso, out_dir / "isolation_forest.pkl")
    (out_dir / "isolation_forest_meta.json").write_text(json.dumps({
        "subset": args.subset, "win": args.win, "n_features": N_FEATURES,
    }, indent=2))

    # ── 3. One-Class SVM (subsample to cap fit time) ──
    logger.info("Training One-Class SVM ...")
    ocsvm_data = flat_h_tr
    if len(ocsvm_data) > 5000:
        rng = np.random.default_rng(42)
        ocsvm_data = ocsvm_data[rng.choice(len(ocsvm_data), 5000, replace=False)]
    ocsvm = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale")
    ocsvm.fit(ocsvm_data)
    joblib.dump(ocsvm, out_dir / "ocsvm.pkl")

    # ── 4. Random Forest (supervised) ──
    logger.info("Training Random Forest ...")
    rf = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1,
    )
    rf.fit(flat_lbl_tr, y_lbl_tr)
    joblib.dump(rf, out_dir / "random_forest_clf.pkl")

    # ── 5. AUC-based weights on labeled val set ──
    logger.info("Computing AUC weights ...")
    ae_scores    = reconstruction_error(ae, X_lbl_val_s)        # high = fault
    iso_scores   = -iso.decision_function(flat_lbl_val)         # high = fault
    ocsvm_scores = -ocsvm.decision_function(flat_lbl_val)       # high = fault
    rf_scores    = rf.predict_proba(flat_lbl_val)[:, 1]         # P(fault)

    auc_ae    = float(roc_auc_score(y_lbl_val, ae_scores))
    auc_iso   = float(roc_auc_score(y_lbl_val, iso_scores))
    auc_ocsvm = float(roc_auc_score(y_lbl_val, ocsvm_scores))
    auc_rf    = float(roc_auc_score(y_lbl_val, rf_scores))
    logger.info("AUC  LSTM-AE=%.3f  IF=%.3f  OC-SVM=%.3f  RF=%.3f",
                auc_ae, auc_iso, auc_ocsvm, auc_rf)

    total = auc_ae + auc_iso + auc_ocsvm + auc_rf
    weights = {
        "lstm_ae":          auc_ae    / total,
        "isolation_forest": auc_iso   / total,
        "ocsvm":            auc_ocsvm / total,
        "random_forest":    auc_rf    / total,
    }

    # Percentile normalization bounds (1st–99th on val scores → [0,1] in runner)
    def bounds(arr):
        return [float(np.percentile(arr, 1)), float(np.percentile(arr, 99))]

    (out_dir / "ensemble_meta.json").write_text(json.dumps({
        "models":   ["lstm_ae", "isolation_forest", "ocsvm", "random_forest"],
        "aucs":     {"lstm_ae": auc_ae, "isolation_forest": auc_iso,
                     "ocsvm": auc_ocsvm, "random_forest": auc_rf},
        "weights":  weights,
        "score_bounds": {
            "lstm_ae":          bounds(ae_scores),
            "isolation_forest": bounds(iso_scores),
            "ocsvm":            bounds(ocsvm_scores),
        },
        "win":               args.win,
        "n_features":        N_FEATURES,
        "fault_threshold":   0.5,
        "critical_threshold": 0.7,
    }, indent=2))
    logger.info("All artifacts saved to %s", out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(args: argparse.Namespace) -> None:
    from models.cmapss_lstm_ae_runner import CMAPSSLSTMAERunner
    from models.base import ModelConfig, DecisionContext
    from utils.cmapss_loader import USEFUL_SENSORS

    data_dir = Path(args.data_dir)
    art_dir  = Path(args.artifact_dir)

    runner = CMAPSSLSTMAERunner(ModelConfig(name="cmapss_lstm_ae", model_path=art_dir))
    df_test = load_cmapss(data_dir / f"test_{args.subset}.txt")
    ctx = DecisionContext(sensor_id="eval", frequency_hz=0, feature_schema=USEFUL_SENSORS)

    results = []
    for unit_id, grp in df_test.groupby("unit"):
        grp = grp.sort_values("cycle")
        arr = grp[USEFUL_SENSORS].to_numpy(dtype=np.float32)
        if len(arr) < runner.win:
            continue
        r = runner.predict(arr, ctx)
        results.append({"unit": str(unit_id), "label": r.label,
                         "score": round(r.score, 4), **r.raw})

    alerts = sum(1 for r in results if r["label"] != "normal")
    summary = {
        "total_engines": len(results),
        "normal": sum(1 for r in results if r["label"] == "normal"),
        "fault": sum(1 for r in results if r["label"] == "fault"),
        "critical": sum(1 for r in results if r["label"] == "critical"),
        "alert_rate": round(alerts / max(len(results), 1), 3),
    }
    print(json.dumps(summary, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CMAPSS fault detection — 4-model ensemble")
    sub = p.add_subparsers(dest="mode", required=True)

    tr = sub.add_parser("train")
    tr.add_argument("--data_dir",        default="CMAPSSData")
    tr.add_argument("--subset",          default="FD001",
                    choices=["FD001", "FD002", "FD003", "FD004"])
    tr.add_argument("--out_dir",         default="artifacts_cmapss_fd001")
    tr.add_argument("--win",             type=int,   default=30)
    tr.add_argument("--step",            type=int,   default=1)
    tr.add_argument("--latent",          type=int,   default=16)
    tr.add_argument("--epochs",          type=int,   default=20)
    tr.add_argument("--batch",           type=int,   default=256)
    tr.add_argument("--threshold_q",     type=float, default=0.99)
    tr.add_argument("--healthy_rul_min", type=int,   default=50)
    tr.add_argument("--max_windows",     type=int,   default=50000)

    ev = sub.add_parser("eval")
    ev.add_argument("--data_dir",      default="CMAPSSData")
    ev.add_argument("--subset",        default="FD001",
                    choices=["FD001", "FD002", "FD003", "FD004"])
    ev.add_argument("--artifact_dir",  default="artifacts_cmapss_fd001")

    return p.parse_args()


if __name__ == "__main__":
    tf.get_logger().setLevel("ERROR")
    args = parse_args()
    if args.mode == "train":
        train(args)
    else:
        evaluate(args)