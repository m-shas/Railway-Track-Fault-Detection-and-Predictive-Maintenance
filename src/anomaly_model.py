"""
Railway Track Fault Detection - Anomaly Detection (Isolation Forest)
Phase 3: Train, evaluate, save, and predict anomalies from sensor data.

Public API:
    train_isolation_forest(df, contamination) -> (model, scaler, labels, scores)
    predict_anomaly(model, scaler, new_data)  -> (labels, scores)
    evaluate_anomaly_detection(...)           -> dict of metrics
    save_model(model, scaler, path)
    load_model(path) -> (model, scaler)
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

CONTAMINATION = 0.08
N_ESTIMATORS = 200
RANDOM_STATE = 42

ANOMALY_FEATURES = [
    "Vibration_m_s2",
    "Temperature_C",
    "Track_Resistance_Ohm",
    "PLC_CPU_Load_percent",
    "Edge_Anomaly_Score",
    "Predicted_Failure_Prob",
    "Humidity_percent",
    "Component_Age_days",
]


# ── TRAINING ──────────────────────────────────────────────────────────────────

def train_isolation_forest(
    df: pd.DataFrame,
    contamination: float = CONTAMINATION,
    n_estimators: int = N_ESTIMATORS,
) -> Tuple[IsolationForest, StandardScaler, np.ndarray, np.ndarray]:
    """Train an Isolation Forest anomaly detector.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain all ANOMALY_FEATURES columns.
    contamination : float
        Expected fraction of anomalies (default 0.08 = 8%).
    n_estimators : int
        Number of base estimators.

    Returns
    -------
    (model, scaler, labels, scores)
        labels: -1 = anomaly, 1 = normal.
        scores: anomaly decision scores (lower = more anomalous).
    """
    X = df[ANOMALY_FEATURES].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    labels = model.predict(X_scaled)
    scores = model.decision_function(X_scaled)

    return model, scaler, labels, scores


# ── PREDICTION ────────────────────────────────────────────────────────────────

def predict_anomaly(
    model: IsolationForest,
    scaler: StandardScaler,
    new_data: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict anomalies on unseen data.

    Parameters
    ----------
    model : IsolationForest
    scaler : StandardScaler
    new_data : pd.DataFrame
        Must contain ANOMALY_FEATURES columns.

    Returns
    -------
    (labels, scores)
        labels: -1 = anomaly, 1 = normal.
        scores: anomaly decision scores.
    """
    X = new_data[ANOMALY_FEATURES].values
    X_scaled = scaler.transform(X)

    labels = model.predict(X_scaled)
    scores = model.decision_function(X_scaled)
    return labels, scores


# ── EVALUATION ────────────────────────────────────────────────────────────────

def evaluate_anomaly_detection(
    labels: np.ndarray,
    scores: np.ndarray,
    df: pd.DataFrame,
) -> Dict[str, float]:
    """Evaluate anomaly detection results.

    Parameters
    ----------
    labels : np.ndarray  (-1/1)
    scores : np.ndarray
    df : pd.DataFrame
        Original data with Predicted_Failure_Prob column.

    Returns
    -------
    dict
        Keys: n_anomalies, anomaly_rate, precision_vs_high_prob.
    """
    n_anomalies = int((labels == -1).sum())
    anomaly_rate = n_anomalies / len(labels) if len(labels) > 0 else 0.0

    # Precision: fraction of IF-flagged anomalies that also have high failure prob
    high_prob = df["Predicted_Failure_Prob"].values > 0.5
    if_flagged = labels == -1
    precision = 0.0
    if if_flagged.sum() > 0:
        precision = float((if_flagged & high_prob).sum() / if_flagged.sum())

    metrics = {
        "n_anomalies": n_anomalies,
        "anomaly_rate": anomaly_rate,
        "precision_vs_high_prob": precision,
    }

    print(f"  Anomalies detected : {n_anomalies} / {len(labels)}")
    print(f"  Anomaly rate       : {anomaly_rate:.1%}")
    print(f"  Precision (>0.5 FP): {precision:.3f}")

    return metrics


# ── SAVE / LOAD ───────────────────────────────────────────────────────────────

def save_model(model: IsolationForest, scaler: StandardScaler,
               path: str = os.path.join(MODELS_DIR, "isolation_forest.pkl")) -> None:
    """Pickle the model + scaler bundle.

    Parameters
    ----------
    model : IsolationForest
    scaler : StandardScaler
    path : str
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, "wb") as f:
            pickle.dump({"model": model, "scaler": scaler, "features": ANOMALY_FEATURES}, f)
        print(f"  Saved anomaly model → {path}")
    except Exception as e:
        raise RuntimeError(f"Failed to save model: {e}")


def load_model(path: str = os.path.join(MODELS_DIR, "isolation_forest.pkl")) -> Tuple[IsolationForest, StandardScaler]:
    """Load a previously saved model bundle.

    Parameters
    ----------
    path : str

    Returns
    -------
    (model, scaler)
    """
    try:
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        return bundle["model"], bundle["scaler"]
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found: {path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from preprocess import preprocess_pipeline

    df, vibr_df, encoders = preprocess_pipeline()

    print("\n[Anomaly Detection] Training Isolation Forest...")
    model, scaler, labels, scores = train_isolation_forest(df)

    # Add flag to DataFrame
    df["IF_Flag"] = (labels == -1).astype(int)

    print("\n[Anomaly Detection] Evaluation:")
    evaluate_anomaly_detection(labels, scores, df)

    save_model(model, scaler)
    print("\nDone.")
