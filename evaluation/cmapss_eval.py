"""CMAPSS evaluation — fault detection (4 models + ensemble) + RUL regression.

Usage
-----
python -m training.cmapss_eval \
    --data_dir CMAPSSData \
    --fd_dir   artifacts_cmapss_fd001 \
    --rul_dir  artifacts_cmapss_rul_fd001 \
    --subset   FD001
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf

from utils.cmapss_loader import load_cmapss, USEFUL_SENSORS, N_FEATURES

tf.get_logger().setLevel("ERROR")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_last_window(grp, win: int) -> np.ndarray:
    """Return the last `win` cycles of an engine as (win, N_FEATURES)."""
    arr = grp.sort_values("cycle")[USEFUL_SENSORS].to_numpy(dtype=np.float32)
    if len(arr) < win:
        arr = np.pad(arr, ((win - len(arr), 0), (0, 0)), mode="edge")
    return arr[-win:]


def scale(x: np.ndarray, scaler) -> np.ndarray:
    sh = x.shape
    return scaler.transform(x.reshape(-1, N_FEATURES)).reshape(sh)


def percentile_norm(score: float, low: float, high: float) -> float:
    return float(np.clip((score - low) / (high - low + 1e-9), 0.0, 1.0))


def classification_metrics(y_true, y_pred, name: str) -> dict:
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    print(f"  {name:<22} Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}")
    return {"model": name, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


# ─────────────────────────────────────────────────────────────────────────────
# Fault detection evaluation
# ─────────────────────────────────────────────────────────────────────────────

def eval_fault_detection(args) -> None:
    data_dir = Path(args.data_dir)
    fd_dir   = Path(args.fd_dir)

    # ── Load artifacts ──
    ae      = tf.keras.models.load_model(fd_dir / "lstm_ae_model.keras", compile=False)
    scaler  = joblib.load(fd_dir / "lstm_ae_scaler.pkl")
    iso     = joblib.load(fd_dir / "isolation_forest.pkl")
    ocsvm   = joblib.load(fd_dir / "ocsvm.pkl")
    rf      = joblib.load(fd_dir / "random_forest_clf.pkl")

    ae_meta  = json.loads((fd_dir / "lstm_ae_meta.json").read_text())
    ens_meta = json.loads((fd_dir / "ensemble_meta.json").read_text())

    win           = int(ae_meta["win"])
    ae_threshold  = float(ae_meta["threshold"])
    weights       = ens_meta["weights"]
    score_bounds  = ens_meta["score_bounds"]
    fault_thr     = float(ens_meta["fault_threshold"])

    # ── Load test data + ground truth ──
    df_test  = load_cmapss(data_dir / f"test_{args.subset}.txt")
    true_rul = np.loadtxt(data_dir / f"RUL_{args.subset}.txt", dtype=np.float32)
    y_true   = (true_rul < 30).astype(int)   # fault label: RUL < 30

    # ── Per-engine inference ──
    ae_preds, iso_preds, ocsvm_preds, rf_preds, ens_preds = [], [], [], [], []

    for unit_id, grp in df_test.groupby("unit"):
        window   = get_last_window(grp, win)            # (win, N_FEATURES)
        window_s = scale(window[None], scaler)          # (1, win, N_FEATURES)
        flat_s   = window_s.reshape(1, -1)              # (1, win * N_FEATURES)

        # LSTM AE
        recon_err = float(np.mean(np.abs(ae.predict(window_s, verbose=0) - window_s)))
        ae_preds.append(int(recon_err > ae_threshold))

        # Isolation Forest  (decision_function < 0 = anomaly)
        iso_preds.append(int(iso.predict(flat_s)[0] == -1))

        # One-Class SVM  (predict returns -1 for anomaly)
        ocsvm_preds.append(int(ocsvm.predict(flat_s)[0] == -1))

        # Random Forest
        rf_prob = float(rf.predict_proba(flat_s)[0, 1])
        rf_preds.append(int(rf_prob > 0.5))

        # Ensemble
        ae_score    = percentile_norm(recon_err,                          *score_bounds["lstm_ae"])
        iso_score   = percentile_norm(-iso.decision_function(flat_s)[0],  *score_bounds["isolation_forest"])
        ocsvm_score = percentile_norm(-ocsvm.decision_function(flat_s)[0],*score_bounds["ocsvm"])
        ens_score   = (weights["lstm_ae"]          * ae_score +
                       weights["isolation_forest"] * iso_score +
                       weights["ocsvm"]            * ocsvm_score +
                       weights["random_forest"]    * rf_prob)
        ens_preds.append(int(ens_score > fault_thr))

    # ── Print results ──
    print("\n" + "=" * 60)
    print(f"  故障检测评估  ({args.subset}, 测试集 {len(y_true)} 台发动机)")
    print(f"  标签定义: RUL < 30 = 故障  |  故障率: {y_true.mean():.1%}")
    print("=" * 60)

    results = []
    results.append(classification_metrics(y_true, ae_preds,    "LSTM Autoencoder"))
    results.append(classification_metrics(y_true, iso_preds,   "Isolation Forest"))
    results.append(classification_metrics(y_true, ocsvm_preds, "One-Class SVM"))
    results.append(classification_metrics(y_true, rf_preds,    "Random Forest"))
    results.append(classification_metrics(y_true, ens_preds,   "集成 (AUC加权)"))
    print("=" * 60)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# RUL regression evaluation
# ─────────────────────────────────────────────────────────────────────────────

def eval_rul(args) -> None:
    data_dir = Path(args.data_dir)
    rul_dir  = Path(args.rul_dir)

    model   = tf.keras.models.load_model(rul_dir / "rul_lstm_model.keras", compile=False)
    scaler  = joblib.load(rul_dir / "rul_scaler.pkl")
    meta    = json.loads((rul_dir / "rul_meta.json").read_text())
    win     = int(meta["win"])

    df_test  = load_cmapss(data_dir / f"test_{args.subset}.txt")
    true_rul = np.loadtxt(data_dir / f"RUL_{args.subset}.txt", dtype=np.float32)

    preds = []
    for unit_id, grp in df_test.groupby("unit"):
        window   = get_last_window(grp, win)
        window_s = scale(window[None], scaler)
        pred     = float(model.predict(window_s, verbose=0)[0][0])
        preds.append(max(pred, 0.0))

    preds    = np.array(preds, dtype=np.float32)
    mae      = float(np.mean(np.abs(preds - true_rul)))
    rmse     = float(np.sqrt(np.mean((preds - true_rul) ** 2)))

    # NASA scoring function (惩罚偏晚预测)
    diff = preds - true_rul
    nasa = float(np.sum(np.where(diff < 0,
                                  np.exp(-diff / 13) - 1,
                                  np.exp(diff  / 10) - 1)))

    print("\n" + "=" * 60)
    print(f"  RUL 回归评估  ({args.subset}, 测试集 {len(true_rul)} 台发动机)")
    print("=" * 60)
    print(f"  {'MAE':<22} {mae:.2f} 周期")
    print(f"  {'RMSE':<22} {rmse:.2f} 周期")
    print(f"  {'NASA Score':<22} {nasa:.1f}  (越低越好)")
    print(f"  {'val_mae (训练时)':<22} {meta.get('val_mae', 'N/A')}")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="CMAPSS 综合评估")
    p.add_argument("--data_dir", default="CMAPSSData")
    p.add_argument("--fd_dir",   default="artifacts_cmapss_fd001")
    p.add_argument("--rul_dir",  default="artifacts_cmapss_rul_fd001")
    p.add_argument("--subset",   default="FD001",
                   choices=["FD001", "FD002", "FD003", "FD004"])
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    eval_fault_detection(args)
    eval_rul(args)