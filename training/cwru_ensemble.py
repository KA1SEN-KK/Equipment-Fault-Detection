"""CWRU fault detection training — 4-model ensemble.

Models
------
1. LSTM Autoencoder   — unsupervised, raw window (2048×1), reconstruction error
2. Isolation Forest   — unsupervised, 17-dim feature vector per window
3. One-Class SVM      — unsupervised, 17-dim feature vector per window
4. Random Forest      — supervised (normal=0, fault=1), 17-dim feature vector

Feature vector (17 dims) = 11 time-domain + 6 frequency-domain features
extracted by FeaturePipeline from each sliding window.

Usage
-----
python -m training.cwru_ensemble train \
    --cwru_root  "凯斯西储大学数据" \
    --fault_dir  "凯斯西储大学数据/12k Drive End Bearing Fault Data" \
    --out_dir    artifacts_cwru_ensemble

python -m training.cwru_ensemble eval \
    --cwru_root  "凯斯西储大学数据" \
    --fault_dir  "凯斯西储大学数据/12k Drive End Bearing Fault Data" \
    --artifact_dir artifacts_cwru_ensemble
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

from utils.data_loader import DatasetSpec, collect_signals, make_dataset
from feature_engineering.pipeline import FeaturePipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CHANNEL = "DE_time"
WIN     = 2048
STEP    = 512
SR      = 12000.0


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(windows: np.ndarray, sr: float = SR) -> np.ndarray:
    """(N, win) raw windows → (N, 17) feature matrix."""
    pipe = FeaturePipeline(sampling_rate=sr)
    rows = []
    for w in windows:
        f = pipe.extract(w)
        rows.append(list(f.values()))
    return np.array(rows, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# LSTM Autoencoder
# ─────────────────────────────────────────────────────────────────────────────

def build_lstm_ae(win: int, latent: int = 32) -> tf.keras.Model:
    inp = layers.Input(shape=(win, 1))
    x = layers.LSTM(64, return_sequences=True)(inp)
    x = layers.LSTM(latent)(x)
    x = layers.RepeatVector(win)(x)
    x = layers.LSTM(64, return_sequences=True)(x)
    out = layers.TimeDistributed(layers.Dense(1))(x)
    m = models.Model(inp, out)
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mae")
    return m


def ae_reconstruction_error(model, x_scaled: np.ndarray, batch: int = 128) -> np.ndarray:
    pred = model.predict(x_scaled, batch_size=batch, verbose=0)
    return np.mean(np.abs(pred.squeeze(-1) - x_scaled.squeeze(-1)), axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    normal_folder = Path(args.cwru_root) / "Normal Baseline Data"
    fault_folder  = Path(args.fault_dir)

    normal_files = sorted(p.stem for p in normal_folder.glob("*.mat"))
    fault_files  = sorted(p.stem for p in fault_folder.glob("*.mat"))
    logger.info("Normal files: %d  Fault files: %d", len(normal_files), len(fault_files))

    # ── Load windows ──
    normal_spec = DatasetSpec("normal", normal_folder, normal_files, CHANNEL)
    fault_spec  = DatasetSpec("fault",  fault_folder,  fault_files,  CHANNEL)

    normal_wins = make_dataset(collect_signals(normal_spec), WIN, STEP, args.max_normal)
    fault_wins  = make_dataset(collect_signals(fault_spec),  WIN, STEP, args.max_fault)
    logger.info("Normal windows: %d  Fault windows: %d", len(normal_wins), len(fault_wins))

    # ── Labeled dataset for RF + AUC eval ──
    all_wins   = np.concatenate([normal_wins, fault_wins], axis=0)
    all_labels = np.concatenate([
        np.zeros(len(normal_wins), dtype=np.int32),
        np.ones(len(fault_wins),   dtype=np.int32),
    ])
    X_lbl_tr, X_lbl_val, y_lbl_tr, y_lbl_val = train_test_split(
        all_wins, all_labels, test_size=0.2, random_state=42, stratify=all_labels,
    )

    # ── Scaler (fit on normal windows only) ──
    x_tr, x_val = train_test_split(normal_wins, test_size=0.2, random_state=42)
    ae_scaler = StandardScaler()
    ae_scaler.fit(x_tr)

    x_tr_s  = ae_scaler.transform(x_tr).reshape(-1, WIN, 1)
    x_val_s = ae_scaler.transform(x_val).reshape(-1, WIN, 1)

    # ── Feature extraction (for sklearn models) ──
    logger.info("Extracting features ...")
    feat_tr  = extract_features(x_tr)
    feat_lbl_tr  = extract_features(X_lbl_tr)
    feat_lbl_val = extract_features(X_lbl_val)

    feat_scaler = StandardScaler()
    feat_scaler.fit(feat_tr)
    feat_lbl_tr_s  = feat_scaler.transform(feat_lbl_tr)
    feat_lbl_val_s = feat_scaler.transform(feat_lbl_val)
    feat_tr_s      = feat_scaler.transform(feat_tr)

    # ── 1. LSTM Autoencoder ──
    logger.info("Training LSTM Autoencoder (%d epochs) ...", args.epochs)
    ae = build_lstm_ae(WIN, latent=args.latent)
    ae.fit(x_tr_s, x_tr_s,
           validation_data=(x_val_s, x_val_s),
           epochs=args.epochs, batch_size=args.batch, verbose=1)
    ae_threshold = float(np.quantile(ae_reconstruction_error(ae, x_val_s), 0.99))
    logger.info("LSTM AE threshold: %.6f", ae_threshold)
    ae.save(out_dir / "lstm_ae_model.keras")
    joblib.dump(ae_scaler, out_dir / "lstm_ae_scaler.pkl")
    (out_dir / "lstm_ae_meta.json").write_text(json.dumps({
        "win": WIN, "step": STEP, "latent": args.latent,
        "threshold": ae_threshold, "sr": SR,
    }, indent=2))

    # ── 2. Isolation Forest ──
    logger.info("Training Isolation Forest ...")
    iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    iso.fit(feat_tr_s)
    joblib.dump(iso, out_dir / "isolation_forest.pkl")

    # ── 3. One-Class SVM ──
    logger.info("Training One-Class SVM ...")
    ocsvm_data = feat_tr_s
    if len(ocsvm_data) > 5000:
        idx = np.random.default_rng(42).choice(len(ocsvm_data), 5000, replace=False)
        ocsvm_data = ocsvm_data[idx]
    ocsvm = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale")
    ocsvm.fit(ocsvm_data)
    joblib.dump(ocsvm, out_dir / "ocsvm.pkl")

    # ── 4. Random Forest ──
    logger.info("Training Random Forest ...")
    rf = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1,
    )
    rf.fit(feat_lbl_tr_s, y_lbl_tr)
    joblib.dump(rf, out_dir / "random_forest_clf.pkl")
    joblib.dump(feat_scaler, out_dir / "feat_scaler.pkl")

    # ── 5. AUC weights ──
    logger.info("Computing AUC weights ...")
    X_lbl_val_s_ae = ae_scaler.transform(X_lbl_val).reshape(-1, WIN, 1)
    ae_scores    = ae_reconstruction_error(ae, X_lbl_val_s_ae)
    iso_scores   = -iso.decision_function(feat_lbl_val_s)
    ocsvm_scores = -ocsvm.decision_function(feat_lbl_val_s)
    rf_scores    = rf.predict_proba(feat_lbl_val_s)[:, 1]

    auc_ae    = float(roc_auc_score(y_lbl_val, ae_scores))
    auc_iso   = float(roc_auc_score(y_lbl_val, iso_scores))
    auc_ocsvm = float(roc_auc_score(y_lbl_val, ocsvm_scores))
    auc_rf    = float(roc_auc_score(y_lbl_val, rf_scores))
    logger.info("AUC  LSTM-AE=%.3f  IF=%.3f  OC-SVM=%.3f  RF=%.3f",
                auc_ae, auc_iso, auc_ocsvm, auc_rf)

    total = auc_ae + auc_iso + auc_ocsvm + auc_rf

    def bounds(arr):
        return [float(np.percentile(arr, 1)), float(np.percentile(arr, 99))]

    (out_dir / "ensemble_meta.json").write_text(json.dumps({
        "models": ["lstm_ae", "isolation_forest", "ocsvm", "random_forest"],
        "aucs":   {"lstm_ae": auc_ae, "isolation_forest": auc_iso,
                   "ocsvm": auc_ocsvm, "random_forest": auc_rf},
        "weights": {
            "lstm_ae":          auc_ae    / total,
            "isolation_forest": auc_iso   / total,
            "ocsvm":            auc_ocsvm / total,
            "random_forest":    auc_rf    / total,
        },
        "score_bounds": {
            "lstm_ae":          bounds(ae_scores),
            "isolation_forest": bounds(iso_scores),
            "ocsvm":            bounds(ocsvm_scores),
        },
        "win": WIN, "sr": SR,
        "fault_threshold": 0.5,
        "critical_threshold": 0.7,
    }, indent=2))
    logger.info("All artifacts saved to %s", out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(args: argparse.Namespace) -> None:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    normal_folder = Path(args.cwru_root) / "Normal Baseline Data"
    fault_folder  = Path(args.fault_dir)
    art_dir       = Path(args.artifact_dir)

    normal_files = sorted(p.stem for p in normal_folder.glob("*.mat"))
    fault_files  = sorted(p.stem for p in fault_folder.glob("*.mat"))

    normal_wins = make_dataset(
        collect_signals(DatasetSpec("n", normal_folder, normal_files, CHANNEL)),
        WIN, STEP, 2000,
    )
    fault_wins = make_dataset(
        collect_signals(DatasetSpec("f", fault_folder, fault_files, CHANNEL)),
        WIN, STEP, 2000,
    )
    all_wins   = np.concatenate([normal_wins, fault_wins])
    all_labels = np.concatenate([np.zeros(len(normal_wins)), np.ones(len(fault_wins))])

    from models.cwru_ensemble_runner import CWRUEnsembleRunner
    from models.base import ModelConfig, DecisionContext
    runner = CWRUEnsembleRunner(ModelConfig(name="cwru_ensemble", model_path=art_dir))
    ctx = DecisionContext(sensor_id="eval", frequency_hz=SR, feature_schema=[])

    preds = []
    for w in all_wins:
        r = runner.predict(w, ctx)
        preds.append(0 if r.label == "normal" else 1)

    preds = np.array(preds)
    print("\n" + "=" * 55)
    print(f"  CWRU 故障检测评估  (normal={len(normal_wins)}  fault={len(fault_wins)})")
    print("=" * 55)
    for name, pred in [("集成 (AUC加权)", preds)]:
        acc  = accuracy_score(all_labels, pred)
        prec = precision_score(all_labels, pred, zero_division=0)
        rec  = recall_score(all_labels, pred, zero_division=0)
        f1   = f1_score(all_labels, pred, zero_division=0)
        print(f"  {name:<20} Acc={acc:.4f} Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f}")
    print("=" * 55)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="CWRU 4-model ensemble training/eval")
    sub = p.add_subparsers(dest="mode", required=True)

    tr = sub.add_parser("train")
    tr.add_argument("--cwru_root",   default="凯斯西储大学数据")
    tr.add_argument("--fault_dir",   required=True,
                    help="故障数据文件夹路径，如 '凯斯西储大学数据/12k Drive End Bearing Fault Data'")
    tr.add_argument("--out_dir",     default="artifacts_cwru_ensemble")
    tr.add_argument("--epochs",      type=int, default=10)
    tr.add_argument("--batch",       type=int, default=128)
    tr.add_argument("--latent",      type=int, default=32)
    tr.add_argument("--max_normal",  type=int, default=10000)
    tr.add_argument("--max_fault",   type=int, default=5000)

    ev = sub.add_parser("eval")
    ev.add_argument("--cwru_root",    default="凯斯西储大学数据")
    ev.add_argument("--fault_dir",    required=True)
    ev.add_argument("--artifact_dir", default="artifacts_cwru_ensemble")

    return p.parse_args()


if __name__ == "__main__":
    tf.get_logger().setLevel("ERROR")
    args = parse_args()
    if args.mode == "train":
        train(args)
    else:
        evaluate(args)