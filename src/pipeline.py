"""
Railway Track Fault Detection - End-to-End Pipeline
Phase 7: Orchestrate all phases, train models, generate dashboard data, and build HTML dashboard.

Usage:
    python src/pipeline.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Ensure src is on path
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from preprocess import preprocess_pipeline, compute_health_score
from anomaly_model import (
    train_isolation_forest, predict_anomaly,
    evaluate_anomaly_detection,
    save_model as save_anomaly_model,
)
from rul_model import (
    train_rul_model, predict_rul,
    save_model as save_rul_model,
)
from classifier import (
    train_classifier, predict_fault, FAULT_DESCRIPTIONS,
    save_model as save_clf_model,
)
from alerts import (
    generate_alert_log, send_alert_summary, export_alert_log,
)

# Advanced ML modules (TensorFlow required)
try:
    from lstm_rul_model import train_lstm_rul_model, save_lstm_model
    from cnn_lstm_model import train_cnn_lstm_classifier, save_cnn_lstm_model
    ADVANCED_ML_AVAILABLE = True
except ImportError:
    ADVANCED_ML_AVAILABLE = False

ALERT_DOWNSAMPLE = 3  # take every Nth row for dashboard JSON


def run_pipeline():
    """Execute the full ML pipeline — BiLSTM/CNN-LSTM as primary models."""
    import time

    print("=" * 70)
    print("RAILWAY TRACK FAULT DETECTION - FULL PIPELINE (ENHANCED)")
    print("=" * 70)
    if ADVANCED_ML_AVAILABLE:
        print("  [PRI] Primary models : BiLSTM (RUL) + CNN-LSTM (Fault)")
        print("  [CMP] Comparison     : GradientBoosting + RandomForest (baseline)")
    else:
        print("  [TF] TensorFlow not found — falling back to GB + RF as primary")
        print("       Run: pip install tensorflow shap")

    # ── Phase 1-2: Preprocessing ──────────────────────────────────────────
    print("\n[1/8] Preprocessing data...")
    df, vibr_df, encoders = preprocess_pipeline()

    # ── Phase 2: Anomaly Detection (Isolation Forest — shared) ─────────────
    print("\n[2/8] Training Isolation Forest...")
    if_model, if_scaler, if_labels, if_scores = train_isolation_forest(df)
    df["IF_Flag"] = (if_labels == -1).astype(int)
    df["IF_Score"] = if_scores
    evaluate_anomaly_detection(if_labels, if_scores, df)
    save_anomaly_model(if_model, if_scaler)

    # ── Phase 3: BiLSTM RUL (PRIMARY) ───────────────────────────────────
    lstm_metrics   = None
    lstm_y_test    = lstm_y_pred = None
    lstm_model_obj = lstm_scaler_obj = None

    if ADVANCED_ML_AVAILABLE:
        print("\n[3/8] Training BiLSTM RUL model (PRIMARY)...")
        try:
            from lstm_rul_model import predict_rul_lstm, SEQ_LEN as LSTM_SEQ
            lstm_model_obj, lstm_scaler_obj, lstm_metrics, (lstm_y_test, lstm_y_pred) = \
                train_lstm_rul_model(df)
            save_lstm_model(lstm_model_obj, lstm_scaler_obj)

            # Write BiLSTM predictions back to df (primary RUL column)
            # Predictions exist for rows SEQ_LEN..end; pad first SEQ_LEN rows with original
            lstm_preds_full = predict_rul_lstm(lstm_model_obj, lstm_scaler_obj, df)
            lstm_col = np.full(len(df), np.nan)
            lstm_col[LSTM_SEQ:] = lstm_preds_full
            # Fill initial rows with original RUL values
            lstm_col[:LSTM_SEQ] = df["RUL_Predicted_days"].values[:LSTM_SEQ]
            df["LSTM_RUL"] = lstm_col
            print(f"  Wrote LSTM_RUL predictions to df ({len(lstm_preds_full)} rows covered)")
        except Exception as e:
            print(f"  WARNING: BiLSTM training failed: {e}")
            import traceback; traceback.print_exc()
            lstm_metrics = None
    else:
        print("\n[3/8] Skipping BiLSTM (TensorFlow not available) — using original RUL")

    # Fallback: if LSTM not available, copy original RUL column
    if "LSTM_RUL" not in df.columns:
        df["LSTM_RUL"] = df["RUL_Predicted_days"]

    # ── Phase 4: CNN-LSTM Fault Classifier (PRIMARY) ──────────────────────
    cnn_metrics   = None
    cnn_model_obj = cnn_scaler_obj = cnn_le_obj = None

    if ADVANCED_ML_AVAILABLE:
        print("\n[4/8] Training CNN-LSTM fault classifier (PRIMARY)...")
        try:
            from cnn_lstm_model import predict_fault_cnn_lstm, SEQ_LEN as CNN_SEQ
            cnn_model_obj, cnn_scaler_obj, cnn_le_obj, cnn_metrics, _ = \
                train_cnn_lstm_classifier(df)
            save_cnn_lstm_model(cnn_model_obj, cnn_scaler_obj, cnn_le_obj)

            # Write CNN-LSTM fault predictions back to df
            cnn_labels, cnn_conf, _ = predict_fault_cnn_lstm(
                cnn_model_obj, cnn_scaler_obj, cnn_le_obj, df)
            # Predictions exist for rows CNN_SEQ..end
            cnn_fault_col = list(df["Failure_Type"].values.copy())
            for i, lbl in enumerate(cnn_labels):
                cnn_fault_col[CNN_SEQ + i] = lbl
            df["CNN_Fault"] = cnn_fault_col
            print(f"  Wrote CNN_Fault predictions to df ({len(cnn_labels)} rows covered)")
        except Exception as e:
            print(f"  WARNING: CNN-LSTM training failed: {e}")
            import traceback; traceback.print_exc()
            cnn_metrics = None
    else:
        print("\n[4/8] Skipping CNN-LSTM (TensorFlow not available) — using original Failure_Type")

    # Fallback: if CNN-LSTM not available, copy original fault column
    if "CNN_Fault" not in df.columns:
        df["CNN_Fault"] = df["Failure_Type"]

    # ── Phase 5: GradientBoosting RUL (BASELINE / comparison) ─────────────
    print("\n[5/8] Training GradientBoosting RUL model (BASELINE comparison)...")
    t0 = time.time()
    rul_model, rul_scaler, rul_metrics, (rul_y_test, rul_y_pred) = train_rul_model(df)
    rul_metrics["train_time_s"] = round(time.time() - t0, 1)
    save_rul_model(rul_model, rul_scaler)

    # ── Phase 6: Random Forest Classifier (BASELINE / comparison) ─────────
    print("\n[6/8] Training RandomForest fault classifier (BASELINE comparison)...")
    clf_model, clf_scaler, clf_y_test, clf_y_pred, clf_accuracy = train_classifier(df)
    save_clf_model(clf_model, clf_scaler, encoders["Failure_Type"])

    # ── Phase 7: Alert Engine (using LSTM_RUL + CNN_Fault as primary) ─────
    print("\n[7/8] Generating alert log (using BiLSTM + CNN-LSTM predictions)...")
    alert_log = generate_alert_log(df)   # alerts.py reads LSTM_RUL if present
    send_alert_summary(alert_log)
    export_alert_log(alert_log)

    # ── Phase 8: Dashboard Data ───────────────────────────────────────────
    print("\n[8/8] Building dashboard data...")
    dashboard_data = build_dashboard_data(
        df, vibr_df, alert_log,
        rul_metrics, rul_y_test, rul_y_pred,
        clf_model, clf_accuracy, clf_y_test, clf_y_pred,
        encoders,
        lstm_metrics=lstm_metrics,
        lstm_y_test=lstm_y_test, lstm_y_pred=lstm_y_pred,
        cnn_metrics=cnn_metrics,
    )
    save_dashboard_data(dashboard_data)
    build_html_dashboard(dashboard_data)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("[OK] PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Models  : {MODELS_DIR}")
    print(f"  Outputs : {OUTPUTS_DIR}")
    print(f"  Alerts  : outputs/alert_log.csv  ({len(alert_log)} rows)")
    print()
    print("  PRIMARY MODELS:")
    if lstm_metrics:
        print(f"    BiLSTM RUL    MAE={lstm_metrics['mae']:.2f}d  R²={lstm_metrics['r2']:.4f}")
    else:
        print(f"    BiLSTM RUL    [not trained] fallback GB MAE={rul_metrics['mae']:.2f}d")
    if cnn_metrics:
        print(f"    CNN-LSTM Clf  Acc={cnn_metrics['accuracy']*100:.1f}%")
    else:
        print(f"    CNN-LSTM Clf  [not trained] fallback RF Acc={clf_accuracy*100:.1f}%")
    print()
    print("  BASELINE (comparison only):")
    print(f"    GradientBoosting  MAE={rul_metrics['mae']:.2f}d  R²={rul_metrics.get('r2',0):.4f}")
    print(f"    RandomForest      Acc={clf_accuracy*100:.1f}%")
    print("=" * 70)


# ── DASHBOARD DATA BUILDER ────────────────────────────────────────────────────

def build_dashboard_data(df, vibr_df, alert_log,
                         rul_metrics, rul_y_test, rul_y_pred,
                         clf_model, clf_accuracy, clf_y_test, clf_y_pred,
                         encoders,
                         lstm_metrics=None, lstm_y_test=None, lstm_y_pred=None,
                         cnn_metrics=None):
    """Build the complete dashboard_data.json payload (with model comparison)."""

    # Block summary (per Track_Block_ID)
    block_summary = []
    for block_id, g in df.groupby("Track_Block_ID"):
        block_summary.append({
            "block_id": str(block_id),
            "avg_health": round(float(g["Health_Score"].mean()), 1),
            "avg_vibration": round(float(g["Vibration_m_s2"].mean()), 3),
            "avg_rul": round(float(g["RUL_Predicted_days"].mean()), 1),
            "n_critical": int((g["Alert_Level"] == "CRITICAL").sum()),
            "n_warning": int((g["Alert_Level"] == "WARNING").sum()),
            "n_anomalies": int(g["IF_Flag"].sum()),
        })

    # Location summary
    location_summary = []
    for loc, g in df.groupby("Location_ID"):
        location_summary.append({
            "location_id": str(loc),
            "avg_health": round(float(g["Health_Score"].mean()), 1),
            "n_blocks": g["Track_Block_ID"].nunique(),
            "n_alerts": int((g["Alert_Level"] != "HEALTHY").sum()),
        })

    # Time-series (downsampled)
    ts_df = df.iloc[::ALERT_DOWNSAMPLE]
    time_series = {
        "timestamps": ts_df["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist(),
        "vibration": ts_df["Vibration_m_s2"].round(3).tolist(),
        "temperature": ts_df["Temperature_C"].round(1).tolist(),
        "health_score": ts_df["Health_Score"].round(1).tolist(),
        "failure_prob": ts_df["Predicted_Failure_Prob"].round(3).tolist(),
    }

    # Alerts (top 50)
    alerts_top = alert_log.head(50)
    alerts_data = []
    for _, row in alerts_top.iterrows():
        alerts_data.append({
            "block_id": str(row.get("Track_Block_ID", "")),
            "location": str(row.get("Location_ID", "")),
            "level": str(row.get("Alert_Level", "")),
            "score": round(float(row.get("Alert_Score", 0)), 1),
            "fault": str(row.get("Fault_Description", "")),
            "action": str(row.get("Recommended_Action", "")),
            "failure_prob": round(float(row.get("Predicted_Failure_Prob", 0)), 3),
            "rul": round(float(row.get("RUL_Predicted_days", 0)), 1),
        })

    # Vibration accel data (from XLSX) — downsampled
    vibr_sample = vibr_df.iloc[::ALERT_DOWNSAMPLE]
    vibration_accel = {
        "accel_magnitude": vibr_sample["accel_magnitude"].round(4).tolist(),
        "rolling_mag": vibr_sample["rolling_mag"].round(4).tolist(),
        "vibr_anomaly": vibr_sample["vibr_anomaly"].tolist(),
    }
    if "Temp" in vibr_sample.columns:
        vibration_accel["temp"] = vibr_sample["Temp"].round(1).tolist()
    if "Hum" in vibr_sample.columns:
        vibration_accel["hum"] = vibr_sample["Hum"].round(1).tolist()

    # Model metrics
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(clf_y_test, clf_y_pred).tolist()
    clf_report = classification_report(clf_y_test, clf_y_pred, output_dict=True)

    # Feature importance from classifier
    feat_imp = {}
    from classifier import CLF_FEATURES
    available_feats = [f for f in CLF_FEATURES if f in df.columns]
    if hasattr(clf_model, 'feature_importances_'):
        for i, f in enumerate(available_feats):
            if i < len(clf_model.feature_importances_):
                feat_imp[f] = round(float(clf_model.feature_importances_[i]), 4)

    # RUL scatter
    rul_scatter = {
        "actual": [round(float(v), 1) for v in rul_y_test[:200]],
        "predicted": [round(float(v), 1) for v in rul_y_pred[:200]],
    }

    # ── Model comparison table (Proposed models FIRST, baselines second) ──────
    comparison = []
    # Proposed (primary) models go first
    if lstm_metrics:
        comparison.append({
            "model": "BiLSTM (Proposed ★)",
            "task": "RUL Prediction",
            "metric": "MAE",
            "value": round(lstm_metrics["mae"], 2),
            "r2": round(lstm_metrics.get("r2", 0), 4),
            "rmse": round(lstm_metrics.get("rmse", 0), 2),
            "train_time_s": lstm_metrics.get("train_time_s", "-"),
            "epochs": lstm_metrics.get("epochs_trained", "-"),
            "primary": True,
        })
    if cnn_metrics:
        comparison.append({
            "model": "CNN-LSTM (Proposed ★)",
            "task": "Fault Classification",
            "metric": "Accuracy",
            "value": round(cnn_metrics["accuracy"] * 100, 2),
            "r2": "-",
            "rmse": "-",
            "train_time_s": cnn_metrics.get("train_time_s", "-"),
            "epochs": cnn_metrics.get("epochs_trained", "-"),
            "primary": True,
        })
    # Baseline (comparison) models go after
    comparison.append({
        "model": "GradientBoosting (Baseline)",
        "task": "RUL Prediction",
        "metric": "MAE",
        "value": round(rul_metrics["mae"], 2),
        "r2": round(rul_metrics.get("r2", 0), 4),
        "rmse": round(rul_metrics.get("rmse", 0), 2),
        "train_time_s": rul_metrics.get("train_time_s", "-"),
        "primary": False,
    })
    comparison.append({
        "model": "RandomForest (Baseline)",
        "task": "Fault Classification",
        "metric": "Accuracy",
        "value": round(clf_accuracy * 100, 2),
        "r2": "-",
        "rmse": "-",
        "train_time_s": "-",
        "primary": False,
    })

    # BiLSTM scatter (if available)
    lstm_scatter = None
    if lstm_y_test is not None and lstm_y_pred is not None:
        n = min(200, len(lstm_y_test))
        lstm_scatter = {
            "actual":    [round(float(v), 1) for v in lstm_y_test[:n]],
            "predicted": [round(float(v), 1) for v in lstm_y_pred[:n]],
        }

    model_metrics = {
        "rul": rul_metrics,
        "lstm_rul": lstm_metrics,
        "classifier_accuracy": round(clf_accuracy, 4),
        "cnn_lstm_accuracy": round(cnn_metrics["accuracy"], 4) if cnn_metrics else None,
        "confusion_matrix": cm,
        "classification_report": {k: v for k, v in clf_report.items() if isinstance(v, dict)},
        "comparison": comparison,
        # Primary model flags for dashboard highlighting
        "primary_rul_model": "BiLSTM" if lstm_metrics else "GradientBoosting",
        "primary_clf_model": "CNN-LSTM" if cnn_metrics else "RandomForest",
    }

    # Alert distribution
    alert_dist = df["Alert_Level"].value_counts().to_dict()
    failure_dist = df["Failure_Type"].value_counts().to_dict()
    maint_dist = df["Maintenance_Action"].value_counts().to_dict()

    return {
        "block_summary": block_summary,
        "location_summary": location_summary,
        "time_series": time_series,
        "alerts": alerts_data,
        "vibration_accel": vibration_accel,
        "model_metrics": model_metrics,
        "rul_scatter": rul_scatter,
        "lstm_scatter": lstm_scatter,
        "feature_importance": feat_imp,
        "alert_distribution": alert_dist,
        "failure_distribution": failure_dist,
        "maintenance_distribution": maint_dist,
    }


def save_dashboard_data(data, path=None):
    """Save dashboard data as JSON."""
    if path is None:
        path = os.path.join(OUTPUTS_DIR, "dashboard_data.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        size_kb = os.path.getsize(path) / 1024
        print(f"  Saved dashboard_data.json ({size_kb:.0f} KB)")
    except Exception as e:
        raise RuntimeError(f"Failed to save dashboard data: {e}")


# ── HTML DASHBOARD BUILDER ────────────────────────────────────────────────────

def build_html_dashboard(data, path=None):
    """Generate the standalone HTML dashboard with embedded data."""
    if path is None:
        path = os.path.join(OUTPUTS_DIR, "railway_dashboard.html")

    data_json = json.dumps(data, default=str)

    html = _get_dashboard_html(data_json)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        size_kb = os.path.getsize(path) / 1024
        print(f"  Saved railway_dashboard.html ({size_kb:.0f} KB)")
    except Exception as e:
        raise RuntimeError(f"Failed to save dashboard: {e}")


def _get_dashboard_html(data_json: str) -> str:
    """Return the full HTML dashboard string."""
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Railway Track Fault Detection Dashboard</title>
<meta name="description" content="AI-powered Railway Track Fault Detection and Predictive Maintenance Dashboard">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
:root {{
  --bg-primary: #0f1923;
  --bg-secondary: #1a2736;
  --bg-card: #213243;
  --text-primary: #e8edf2;
  --text-secondary: #8899aa;
  --accent-blue: #3b82f6;
  --accent-green: #10b981;
  --accent-yellow: #f59e0b;
  --accent-red: #ef4444;
  --accent-purple: #8b5cf6;
  --border: #2d3f52;
}}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:'Segoe UI',system-ui,sans-serif; background:var(--bg-primary); color:var(--text-primary); display:flex; height:100vh; overflow:hidden; }}

.sidebar {{ width:260px; background:var(--bg-secondary); border-right:1px solid var(--border); display:flex; flex-direction:column; padding:24px 16px; overflow-y:auto; }}
.sidebar h2 {{ font-size:1.2rem; font-weight:700; margin-bottom:24px; padding-left:8px; }}
.sidebar h2 span {{ color:var(--accent-blue); }}
.sidebar .badge {{ background:var(--accent-green); color:#fff; padding:4px 12px; border-radius:12px; font-size:0.75rem; font-weight:600; display:inline-block; margin-left:8px; margin-bottom:24px; }}

.tabs {{ display:flex; flex-direction:column; gap:4px; }}
.tab {{ padding:10px 12px; cursor:pointer; color:var(--text-secondary); border-radius:6px; font-weight:500; transition:all 0.2s; text-align:left; border:none; background:transparent; font-size:1rem; font-family:inherit; }}
.tab:hover {{ color:var(--text-primary); background:rgba(59,130,246,0.05); }}
.tab.active {{ color:var(--text-primary); background:rgba(59,130,246,0.15); font-weight:600; border-left:4px solid var(--accent-blue); padding-left:8px; }}

.main-content {{ flex:1; overflow-y:auto; padding:32px 48px; background:var(--bg-primary); }}
.header {{ margin-bottom:24px; border-bottom:1px solid var(--border); padding-bottom:16px; }}
.header h1 {{ font-size:2rem; font-weight:700; }}

.tab-content {{ display:none; }}
.tab-content.active {{ display:block; }}

.grid {{ display:grid; gap:20px; }}
.grid-6 {{ grid-template-columns:repeat(auto-fill,minmax(160px,1fr)); }}
.grid-2 {{ grid-template-columns:repeat(auto-fill,minmax(400px,1fr)); }}
.card {{ background:var(--bg-card); border:1px solid var(--border); border-radius:12px; padding:20px; }}
.card h3 {{ color:var(--text-secondary); font-size:0.8rem; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:8px; }}
.card .value {{ font-size:1.8rem; font-weight:700; }}
.card .sub {{ color:var(--text-secondary); font-size:0.8rem; margin-top:4px; }}
.chart-card {{ background:var(--bg-card); border:1px solid var(--border); border-radius:12px; padding:20px; }}
.chart-card h3 {{ color:var(--text-primary); font-size:1rem; margin-bottom:16px; font-weight:600; }}
.health-grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(120px,1fr)); gap:8px; margin-top:16px; }}
.health-block {{ padding:12px; border-radius:8px; text-align:center; font-size:0.8rem; font-weight:600; }}
.health-block .score {{ font-size:1.2rem; margin-top:4px; }}
table {{ width:100%; border-collapse:collapse; font-size:0.85rem; }}
th {{ text-align:left; padding:10px 12px; background:var(--bg-secondary); color:var(--text-secondary); font-weight:600; border-bottom:1px solid var(--border); }}
td {{ padding:10px 12px; border-bottom:1px solid var(--border); }}
tr:hover {{ background:rgba(59,130,246,0.05); }}
.badge-critical {{ background:rgba(239,68,68,0.15); color:var(--accent-red); padding:2px 8px; border-radius:4px; font-weight:600; font-size:0.75rem; }}
.badge-warning {{ background:rgba(245,158,11,0.15); color:var(--accent-yellow); padding:2px 8px; border-radius:4px; font-weight:600; font-size:0.75rem; }}
.badge-healthy {{ background:rgba(16,185,129,0.15); color:var(--accent-green); padding:2px 8px; border-radius:4px; font-weight:600; font-size:0.75rem; }}
.health-bar {{ height:6px; background:var(--border); border-radius:3px; overflow:hidden; }}
.health-bar-fill {{ height:100%; border-radius:3px; transition:width 0.5s; }}
.filter-bar {{ display:flex; gap:12px; margin-bottom:16px; flex-wrap:wrap; }}
.filter-bar select {{ padding:8px 12px; background:var(--bg-secondary); color:var(--text-primary); border:1px solid var(--border); border-radius:6px; font-size:0.85rem; }}
.maint-cards {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(300px,1fr)); gap:12px; margin-top:16px; }}
.maint-card {{ background:var(--bg-secondary); border-left:3px solid var(--accent-blue); border-radius:8px; padding:16px; }}
.maint-card.critical {{ border-left-color:var(--accent-red); }}
.maint-card.warning {{ border-left-color:var(--accent-yellow); }}
.maint-card h4 {{ font-size:0.9rem; margin-bottom:6px; }}
.maint-card p {{ font-size:0.8rem; color:var(--text-secondary); }}
canvas {{ max-height: 350px; }}
</style>
</head>
<body>

<div class="sidebar">
  <h2>🚂 <span>Railway</span> AI Dashboard</h2>
  <span class="badge">Static Version</span>
  
  <div class="tabs" id="tabs">
    <div style="font-size:0.75rem; color:var(--text-secondary); margin-bottom:8px; padding-left:12px; text-transform:uppercase; letter-spacing:0.05em;">Navigate</div>
    <div class="tab active" data-tab="overview">Overview</div>
    <div class="tab" data-tab="live">Live Monitoring</div>
    <div class="tab" data-tab="vibration">Vibration Analysis</div>
    <div class="tab" data-tab="blocks">Track Blocks</div>
    <div class="tab" data-tab="models">AI Models</div>
    <div class="tab" data-tab="xai">XAI Explainability</div>
    <div class="tab" data-tab="alerts">Alerts</div>
  </div>
</div>

<div class="main-content">
  <!-- TAB 1: OVERVIEW -->
  <div class="tab-content active" id="tab-overview">
    <div class="header"><h1>Overview</h1></div>
    <div class="grid grid-6" id="kpi-cards"></div>
    <div class="grid grid-2" style="margin-top:20px;">
      <div class="chart-card"><h3>Alert Distribution</h3><canvas id="alertDonut"></canvas></div>
      <div class="chart-card"><h3>Failure Type Distribution</h3><canvas id="failureBar"></canvas></div>
    </div>
    <div class="grid grid-2" style="margin-top:20px;">
      <div class="chart-card"><h3>Location Health Overview</h3><canvas id="locationHealth"></canvas></div>
      <div class="chart-card"><h3>Maintenance Actions</h3><canvas id="maintDonut"></canvas></div>
    </div>
    <div class="chart-card" style="margin-top:20px;">
      <h3>Track Block Health Grid</h3>
      <div class="health-grid" id="healthGrid"></div>
    </div>
  </div>

  <!-- TAB 2: LIVE MONITORING -->
  <div class="tab-content" id="tab-live">
    <div class="header"><h1>🔴 Live Sensor Monitoring</h1></div>
    <div class="chart-card" style="text-align: center; padding: 60px 20px;">
      <h2 style="margin-bottom: 16px;">This feature is only available in Streamlit</h2>
      <p style="color: var(--text-secondary); max-width: 600px; margin: 0 auto; line-height: 1.6;">
        Live streaming telemetry and dynamic background processes require a backend server.<br><br>
        To access the real-time simulation engine, run the following in your terminal:<br><br>
        <code style="background: var(--bg-primary); padding: 8px 16px; border-radius: 6px; color: var(--accent-blue); font-size: 1.1rem; display: inline-block; margin-top: 10px;">streamlit run app.py</code>
      </p>
    </div>
  </div>

  <!-- TAB 3: VIBRATION -->
  <div class="tab-content" id="tab-vibration">
    <div class="header"><h1>Vibration Analysis</h1></div>
    <div class="grid grid-2">
      <div class="chart-card"><h3>Acceleration Magnitude (Time Series)</h3><canvas id="accelTS"></canvas></div>
      <div class="chart-card"><h3>Temperature vs Humidity</h3><canvas id="tempHumScatter"></canvas></div>
    </div>
    <div class="grid grid-2" style="margin-top:20px;">
      <div class="chart-card"><h3>PLC Vibration & Track Resistance</h3><canvas id="vibResLine"></canvas></div>
      <div class="chart-card"><h3>Vibration Histogram</h3><canvas id="vibHist"></canvas></div>
    </div>
  </div>

  <!-- TAB 4: TRACK BLOCKS -->
  <div class="tab-content" id="tab-blocks">
    <div class="header"><h1>Track Blocks</h1></div>
    <div class="grid grid-2">
      <div class="chart-card"><h3>RUL by Block</h3><canvas id="rulBar"></canvas></div>
      <div class="chart-card"><h3>Vibration by Block</h3><canvas id="vibBar"></canvas></div>
    </div>
    <div class="chart-card" style="margin-top:20px;">
      <h3>Block Detail Table</h3>
      <table id="blockTable">
        <thead><tr><th>Block ID</th><th>Avg Health</th><th>Avg RUL</th><th>Vibration</th><th>Critical</th><th>Warning</th><th>Anomalies</th><th>Health Bar</th></tr></thead>
        <tbody></tbody>
      </table>
    </div>
  </div>

  <!-- TAB 5: AI MODELS -->
  <div class="tab-content" id="tab-models">
    <div class="header"><h1>AI Model Performance</h1></div>
    <!-- Primary model badges -->
    <div style="display:flex;gap:12px;margin-bottom:20px;flex-wrap:wrap;">
      <div style="background:rgba(139,92,246,0.15);border:1px solid #8b5cf6;border-radius:8px;padding:12px 20px;">
        <div style="font-size:0.7rem;color:#8b5cf6;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:4px;">★ PRIMARY — RUL Prediction</div>
        <div style="font-size:1.1rem;font-weight:700;">BiLSTM (Bidirectional LSTM)</div>
        <div style="font-size:0.75rem;color:#8899aa;margin-top:2px;">Seq-to-one regression · temporal sequences</div>
      </div>
      <div style="background:rgba(59,130,246,0.15);border:1px solid #3b82f6;border-radius:8px;padding:12px 20px;">
        <div style="font-size:0.7rem;color:#3b82f6;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:4px;">★ PRIMARY — Fault Classification</div>
        <div style="font-size:1.1rem;font-weight:700;">CNN-LSTM Hybrid</div>
        <div style="font-size:0.75rem;color:#8899aa;margin-top:2px;">Spatial CNN + temporal LSTM · 10-class</div>
      </div>
      <div style="background:rgba(17,24,39,0.6);border:1px solid var(--border);border-radius:8px;padding:12px 20px;opacity:0.7;">
        <div style="font-size:0.7rem;color:var(--text-secondary);text-transform:uppercase;letter-spacing:0.1em;margin-bottom:4px;">Baseline — Comparison Only</div>
        <div style="font-size:1rem;font-weight:600;color:var(--text-secondary);">GradientBoosting · RandomForest</div>
        <div style="font-size:0.75rem;color:#8899aa;margin-top:2px;">Classical ML benchmarks</div>
      </div>
    </div>
    <div class="grid grid-2">
      <div class="chart-card"><h3>BiLSTM RUL: Actual vs Predicted</h3><canvas id="rulScatter"></canvas></div>
      <div class="chart-card"><h3>Feature Importance (RF Baseline)</h3><canvas id="featImpBar"></canvas></div>
    </div>
    <div class="grid grid-2" style="margin-top:20px;">
      <div class="chart-card"><h3>Model Comparison</h3><canvas id="modelCompBar"></canvas></div>
      <div class="chart-card"><h3>Confusion Matrix (Heatmap)</h3><canvas id="confMatrix"></canvas></div>
    </div>
    <div class="chart-card" style="margin-top:20px;">
      <h3>Primary Model Metrics</h3>
      <table id="modelTable">
        <thead><tr><th>Model</th><th>Task</th><th>Metric</th><th>Value</th><th>R²</th><th>RMSE</th><th>Epochs</th><th>Train Time</th></tr></thead>
        <tbody></tbody>
      </table>
      <div style="margin-top:20px;">
        <h3 style="font-size:1rem; margin-bottom:12px;">Pipeline Architecture</h3>
        <div style="background:var(--bg-secondary);padding:16px;border-radius:8px;font-family:monospace;font-size:0.8rem;color:var(--accent-blue);">
          CSV + XLSX → <span style="color:var(--accent-green)">Preprocess</span> → <span style="color:var(--accent-yellow)">Isolation Forest</span> → <span style="color:var(--accent-purple)">★ BiLSTM (RUL)</span> + <span style="color:#3b82f6">★ CNN-LSTM (Fault)</span> → Alert Engine → Dashboard<br>
          <span style="color:var(--text-secondary);font-size:0.75rem;">Baselines (comparison): GradientBoosting + RandomForest</span>
        </div>
      </div>
    </div>
  </div>

  <!-- TAB 6: XAI EXPLAINABILITY -->
  <div class="tab-content" id="tab-xai">
    <div class="header"><h1>🧠 XAI — Explainable AI Dashboard</h1></div>
    <div class="chart-card" style="text-align: center; padding: 60px 20px;">
      <h2 style="margin-bottom: 16px;">This feature is only available in Streamlit</h2>
      <p style="color: var(--text-secondary); max-width: 600px; margin: 0 auto; line-height: 1.6;">
        Generating Shapley Additive exPlanations (SHAP) requires compiling the model in Python.<br><br>
        To access interactive SHAP waterfall models and beeswarm plots, run the following:<br><br>
        <code style="background: var(--bg-primary); padding: 8px 16px; border-radius: 6px; color: var(--accent-blue); font-size: 1.1rem; display: inline-block; margin-top: 10px;">streamlit run app.py</code>
      </p>
    </div>
  </div>

  <!-- TAB 7: ALERTS -->
  <div class="tab-content" id="tab-alerts">
    <div class="header"><h1>Alerts & Maintenance</h1></div>
    <div class="filter-bar">
      <select id="alertFilter"><option value="ALL">All Levels</option><option value="CRITICAL">Critical Only</option><option value="WARNING">Warning Only</option></select>
    </div>
    <div class="chart-card">
      <h3>Alert Table</h3>
      <table id="alertTable">
        <thead><tr><th>Block</th><th>Location</th><th>Level</th><th>Score</th><th>Fault</th><th>Action</th><th>Prob</th><th>RUL</th></tr></thead>
        <tbody></tbody>
      </table>
    </div>
    <div class="chart-card" style="margin-top:20px;">
      <h3>Maintenance Recommendations</h3>
      <div class="maint-cards" id="maintCards"></div>
    </div>
    <div class="chart-card" style="margin-top:20px;">
      <h3>Alert Timeline</h3>
      <canvas id="alertTimeline"></canvas>
    </div>
  </div>
</div>

<script>
const DATA = {data_json};

// ── Tab Switching ──────────────────────────────────────────────────────
document.querySelectorAll('.tab').forEach(tab => {{
  tab.addEventListener('click', () => {{
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById('tab-' + tab.dataset.tab).classList.add('active');
  }});
}});

// ── Helpers ────────────────────────────────────────────────────────────
const COLORS = ['#3b82f6','#10b981','#f59e0b','#ef4444','#8b5cf6','#ec4899','#06b6d4','#f97316','#14b8a6','#6366f1'];
function healthColor(s) {{ return s >= 70 ? '#10b981' : s >= 40 ? '#f59e0b' : '#ef4444'; }}

// ── TAB 1: OVERVIEW ───────────────────────────────────────────────────
(function() {{
  const bs = DATA.block_summary;
  const avgHealth = (bs.reduce((a,b)=>a+b.avg_health,0)/bs.length).toFixed(1);
  const totalCrit = bs.reduce((a,b)=>a+b.n_critical,0);
  const totalWarn = bs.reduce((a,b)=>a+b.n_warning,0);
  const totalAnom = bs.reduce((a,b)=>a+b.n_anomalies,0);
  const avgRul = (bs.reduce((a,b)=>a+b.avg_rul,0)/bs.length).toFixed(0);
  const kpis = [
    {{label:'Track Blocks',value:bs.length,color:'var(--accent-blue)',sub:'monitored'}},
    {{label:'Avg Health',value:avgHealth,color:healthColor(avgHealth),sub:'/ 100'}},
    {{label:'Critical Alerts',value:totalCrit,color:'var(--accent-red)',sub:'immediate'}},
    {{label:'Warnings',value:totalWarn,color:'var(--accent-yellow)',sub:'watch list'}},
    {{label:'Anomalies',value:totalAnom,color:'var(--accent-purple)',sub:'IF detected'}},
    {{label:'Avg RUL',value:avgRul+' d',color:'var(--accent-green)',sub:'days remaining'}}
  ];
  const kpiHtml = kpis.map(k => `<div class="card"><h3>${{k.label}}</h3><div class="value" style="color:${{k.color}}">${{k.value}}</div><div class="sub">${{k.sub}}</div></div>`).join('');
  document.getElementById('kpi-cards').innerHTML = kpiHtml;

  // Alert donut
  const ad = DATA.alert_distribution;
  new Chart(document.getElementById('alertDonut'), {{type:'doughnut',data:{{labels:Object.keys(ad),datasets:[{{data:Object.values(ad),backgroundColor:['#ef4444','#f59e0b','#10b981']}}]}},options:{{plugins:{{legend:{{labels:{{color:'#e8edf2'}}}}}}}}}});

  // Failure bar
  const fd = DATA.failure_distribution;
  new Chart(document.getElementById('failureBar'), {{type:'bar',data:{{labels:Object.keys(fd),datasets:[{{label:'Count',data:Object.values(fd),backgroundColor:COLORS}}]}},options:{{plugins:{{legend:{{display:false}}}},scales:{{x:{{ticks:{{color:'#8899aa'}}}},y:{{ticks:{{color:'#8899aa'}}}}}}}}}});

  // Location health
  const ls = DATA.location_summary;
  new Chart(document.getElementById('locationHealth'), {{type:'bar',data:{{labels:ls.map(l=>l.location_id),datasets:[{{label:'Avg Health',data:ls.map(l=>l.avg_health),backgroundColor:ls.map(l=>healthColor(l.avg_health))}},{{label:'Alerts',data:ls.map(l=>l.n_alerts),backgroundColor:'rgba(239,68,68,0.5)',type:'line',borderColor:'#ef4444',fill:false,yAxisID:'y1'}}]}},options:{{scales:{{y:{{ticks:{{color:'#8899aa'}}}},y1:{{position:'right',ticks:{{color:'#8899aa'}},grid:{{display:false}}}},x:{{ticks:{{color:'#8899aa'}}}}}},plugins:{{legend:{{labels:{{color:'#e8edf2'}}}}}}}}}});

  // Maintenance donut
  const md = DATA.maintenance_distribution;
  new Chart(document.getElementById('maintDonut'), {{type:'doughnut',data:{{labels:Object.keys(md),datasets:[{{data:Object.values(md),backgroundColor:COLORS}}]}},options:{{plugins:{{legend:{{labels:{{color:'#e8edf2'}}}}}}}}}});

  // Health grid
  const gridHtml = bs.map(b => {{
    const c = healthColor(b.avg_health);
    return `<div class="health-block" style="background:${{c}}22;border:1px solid ${{c}}44;color:${{c}}"><div>${{b.block_id}}</div><div class="score">${{b.avg_health}}</div></div>`;
  }}).join('');
  document.getElementById('healthGrid').innerHTML = gridHtml;
}})();

// ── TAB 3: VIBRATION ──────────────────────────────────────────────────
(function() {{
  const va = DATA.vibration_accel;
  const labels = Array.from({{length:va.accel_magnitude.length}},(_,i)=>i);

  // Accel time series with anomaly dots
  const normalData = va.accel_magnitude.map((v,i) => va.vibr_anomaly[i]===0 ? v : null);
  const anomalyData = va.accel_magnitude.map((v,i) => va.vibr_anomaly[i]===1 ? v : null);
  new Chart(document.getElementById('accelTS'), {{type:'line',data:{{labels,datasets:[
    {{label:'Normal',data:normalData,borderColor:'#3b82f6',pointRadius:0,borderWidth:1}},
    {{label:'Anomaly',data:anomalyData,borderColor:'#ef4444',backgroundColor:'#ef4444',pointRadius:3,showLine:false}}
  ]}},options:{{scales:{{x:{{display:false}},y:{{ticks:{{color:'#8899aa'}}}}}},plugins:{{legend:{{labels:{{color:'#e8edf2'}}}}}}}}}});

  // Temp vs Hum scatter
  if(va.temp && va.hum) {{
    new Chart(document.getElementById('tempHumScatter'), {{type:'scatter',data:{{datasets:[{{label:'Temp/Hum',data:va.temp.map((t,i)=>({{x:t,y:va.hum[i]}})),backgroundColor:'rgba(139,92,246,0.4)',pointRadius:2}}]}},options:{{scales:{{x:{{title:{{display:true,text:'Temperature',color:'#8899aa'}},ticks:{{color:'#8899aa'}}}},y:{{title:{{display:true,text:'Humidity',color:'#8899aa'}},ticks:{{color:'#8899aa'}}}}}},plugins:{{legend:{{labels:{{color:'#e8edf2'}}}}}}}}}});
  }}

  // PLC Vibration vs Resistance
  const ts = DATA.time_series;
  const tsLabels = ts.timestamps.map(t=>t.substr(11,5));
  new Chart(document.getElementById('vibResLine'), {{type:'line',data:{{labels:tsLabels.slice(0,200),datasets:[
    {{label:'Vibration',data:ts.vibration.slice(0,200),borderColor:'#3b82f6',borderWidth:1,pointRadius:0,yAxisID:'y'}},
    {{label:'FailureProb',data:ts.failure_prob.slice(0,200),borderColor:'#ef4444',borderWidth:1,pointRadius:0,yAxisID:'y1'}}
  ]}},options:{{scales:{{y:{{ticks:{{color:'#8899aa'}}}},y1:{{position:'right',ticks:{{color:'#8899aa'}},grid:{{display:false}}}},x:{{ticks:{{color:'#8899aa',maxTicksLimit:10}}}}}},plugins:{{legend:{{labels:{{color:'#e8edf2'}}}}}}}}}});

  // Vibration histogram
  const bins = Array.from({{length:20}},(_,i)=>i);
  const hist = new Array(20).fill(0);
  const magMin = Math.min(...va.accel_magnitude), magMax = Math.max(...va.accel_magnitude);
  const step = (magMax-magMin)/20;
  va.accel_magnitude.forEach(v => {{ const b = Math.min(19,Math.floor((v-magMin)/step)); hist[b]++; }});
  new Chart(document.getElementById('vibHist'), {{type:'bar',data:{{labels:bins.map(i=>(magMin+i*step).toFixed(2)),datasets:[{{label:'Count',data:hist,backgroundColor:'#10b981'}}]}},options:{{scales:{{x:{{ticks:{{color:'#8899aa',maxTicksLimit:10}}}},y:{{ticks:{{color:'#8899aa'}}}}}},plugins:{{legend:{{display:false}}}}}}}});
}})();

// ── TAB 4: TRACK BLOCKS ──────────────────────────────────────────────
(function() {{
  const bs = DATA.block_summary.sort((a,b) => a.block_id.localeCompare(b.block_id, undefined, {{numeric:true}}));

  new Chart(document.getElementById('rulBar'), {{type:'bar',data:{{labels:bs.map(b=>b.block_id),datasets:[{{label:'Avg RUL (days)',data:bs.map(b=>b.avg_rul),backgroundColor:bs.map(b=>b.avg_rul<60?'#ef4444':b.avg_rul<120?'#f59e0b':'#10b981')}}]}},options:{{indexAxis:'y',scales:{{x:{{ticks:{{color:'#8899aa'}}}},y:{{ticks:{{color:'#8899aa'}}}}}},plugins:{{legend:{{display:false}}}}}}}});

  new Chart(document.getElementById('vibBar'), {{type:'bar',data:{{labels:bs.map(b=>b.block_id),datasets:[{{label:'Avg Vibration',data:bs.map(b=>b.avg_vibration),backgroundColor:'#8b5cf6'}}]}},options:{{indexAxis:'y',scales:{{x:{{ticks:{{color:'#8899aa'}}}},y:{{ticks:{{color:'#8899aa'}}}}}},plugins:{{legend:{{display:false}}}}}}}});

  // Table
  const tbody = document.querySelector('#blockTable tbody');
  tbody.innerHTML = bs.map(b => {{
    const hc = healthColor(b.avg_health);
    return `<tr><td>${{b.block_id}}</td><td style="color:${{hc}}">${{b.avg_health}}</td><td>${{b.avg_rul}} d</td><td>${{b.avg_vibration}}</td><td style="color:var(--accent-red)">${{b.n_critical}}</td><td style="color:var(--accent-yellow)">${{b.n_warning}}</td><td>${{b.n_anomalies}}</td><td><div class="health-bar"><div class="health-bar-fill" style="width:${{b.avg_health}}%;background:${{hc}}"></div></div></td></tr>`;
  }}).join('');
}})();

// ── TAB 5: AI MODELS ──────────────────────────────────────────────────
(function() {{
  const rs = DATA.rul_scatter;
  new Chart(document.getElementById('rulScatter'), {{type:'scatter',data:{{datasets:[
    {{label:'Predictions',data:rs.actual.map((a,i)=>({{x:a,y:rs.predicted[i]}})),backgroundColor:'rgba(59,130,246,0.4)',pointRadius:2}},
    {{label:'Perfect',data:[{{x:0,y:0}},{{x:Math.max(...rs.actual),y:Math.max(...rs.actual)}}],borderColor:'#ef4444',borderDash:[5,5],showLine:true,pointRadius:0,borderWidth:1}}
  ]}},options:{{scales:{{x:{{title:{{display:true,text:'Actual RUL',color:'#8899aa'}},ticks:{{color:'#8899aa'}}}},y:{{title:{{display:true,text:'Predicted RUL',color:'#8899aa'}},ticks:{{color:'#8899aa'}}}}}},plugins:{{legend:{{labels:{{color:'#e8edf2'}}}}}}}}}});

  // Feature importance
  const fi = DATA.feature_importance;
  const fiSorted = Object.entries(fi).sort((a,b) => b[1]-a[1]);
  new Chart(document.getElementById('featImpBar'), {{type:'bar',data:{{labels:fiSorted.map(f=>f[0]),datasets:[{{label:'Importance',data:fiSorted.map(f=>f[1]),backgroundColor:COLORS}}]}},options:{{indexAxis:'y',scales:{{x:{{ticks:{{color:'#8899aa'}}}},y:{{ticks:{{color:'#8899aa',font:{{size:10}}}}}}}},plugins:{{legend:{{display:false}}}}}}}});

  // Confusion matrix
  const cm = DATA.model_metrics.confusion_matrix;
  if(cm && cm.length > 0) {{
    const cmLabels = cm.map((_,i)=>'C'+(i+1));
    const cmData = [];
    cm.forEach((row,i) => row.forEach((val,j) => cmData.push({{x:j,y:i,v:val}})));
    const maxVal = Math.max(...cmData.map(d=>d.v));
    new Chart(document.getElementById('confMatrix'), {{type:'scatter',data:{{datasets:[{{
      data:cmData.map(d=>({{x:d.x,y:d.y}})),
      backgroundColor:cmData.map(d=>`rgba(59,130,246,${{Math.max(0.1,d.v/maxVal)}})`),
      pointRadius:cmData.map(d=>5+15*(d.v/maxVal)),
      pointStyle:'rect'
    }}]}},options:{{scales:{{x:{{min:-0.5,max:cm[0].length-0.5,ticks:{{callback:v=>cmLabels[v]||'',color:'#8899aa'}}}},y:{{min:-0.5,max:cm.length-0.5,reverse:true,ticks:{{callback:v=>cmLabels[v]||'',color:'#8899aa'}}}}}},plugins:{{legend:{{display:false}},tooltip:{{callbacks:{{label:ctx=>{{const d=cmData[ctx.dataIndex];return `${{cmLabels[d.y]}}→${{cmLabels[d.x]}}: ${{d.v}}`}}}}}}}}}}}});
  }}

  // Model metrics table
  const mm = DATA.model_metrics;
  const mtBody = document.querySelector('#modelTable tbody');
  mtBody.innerHTML = `
    <tr><td>RUL MAE</td><td>${{mm.rul.mae.toFixed(2)}} days</td></tr>
    <tr><td>RUL RMSE</td><td>${{mm.rul.rmse.toFixed(2)}} days</td></tr>
    <tr><td>RUL R²</td><td>${{mm.rul.r2.toFixed(4)}}</td></tr>
    <tr><td>Classifier Accuracy</td><td>${{(mm.classifier_accuracy*100).toFixed(1)}}%</td></tr>
    <tr><td>Fault Classes</td><td>${{cm?cm.length:10}}</td></tr>
  `;
}})();

// ── TAB 7: ALERTS ─────────────────────────────────────────────────────
(function() {{
  const alerts = DATA.alerts;
  function renderAlerts(filter) {{
    const filtered = filter === 'ALL' ? alerts : alerts.filter(a=>a.level===filter);
    const tbody = document.querySelector('#alertTable tbody');
    tbody.innerHTML = filtered.map(a => {{
      const bc = a.level==='CRITICAL'?'badge-critical':'badge-warning';
      return `<tr><td>${{a.block_id}}</td><td>${{a.location}}</td><td><span class="${{bc}}">${{a.level}}</span></td><td>${{a.score}}</td><td>${{a.fault}}</td><td style="max-width:200px;font-size:0.75rem">${{a.action}}</td><td>${{a.failure_prob}}</td><td>${{a.rul}}</td></tr>`;
    }}).join('');

    // Maint cards
    const mc = document.getElementById('maintCards');
    mc.innerHTML = filtered.slice(0,8).map(a => {{
      const cls = a.level==='CRITICAL'?'critical':'warning';
      return `<div class="maint-card ${{cls}}"><h4>${{a.block_id}} — ${{a.fault}}</h4><p>${{a.action}}</p><p style="margin-top:6px;color:var(--accent-blue)">Score: ${{a.score}} | RUL: ${{a.rul}}d</p></div>`;
    }}).join('');
  }}
  renderAlerts('ALL');
  document.getElementById('alertFilter').addEventListener('change', e => renderAlerts(e.target.value));

  // Alert timeline
  const levelCounts = {{}};
  alerts.forEach(a => {{ levelCounts[a.level] = (levelCounts[a.level]||0)+1; }});
  new Chart(document.getElementById('alertTimeline'), {{type:'bar',data:{{labels:Object.keys(levelCounts),datasets:[{{label:'Count',data:Object.values(levelCounts),backgroundColor:Object.keys(levelCounts).map(l=>l==='CRITICAL'?'#ef4444':'#f59e0b')}}]}},options:{{scales:{{x:{{ticks:{{color:'#8899aa'}}}},y:{{ticks:{{color:'#8899aa'}}}}}},plugins:{{legend:{{display:false}}}}}}}});
}})();
</script>
</body>
</html>'''


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_pipeline()
