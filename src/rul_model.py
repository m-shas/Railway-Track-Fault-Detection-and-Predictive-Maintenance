"""
Railway Track Fault Detection - RUL Prediction Model
Phase 4: Gradient Boosting Regressor for Remaining Useful Life prediction.

Public API:
    train_rul_model(df, n_estimators, test_size) -> (model, scaler, metrics_dict, (y_test, y_pred))
    predict_rul(model, scaler, df)               -> np.ndarray
    rul_alert_level(days)                        -> str
    save_model(bundle, path)
    load_model(path) -> bundle dict
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Tuple, Dict

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

RANDOM_STATE = 42
TEST_SIZE = 0.20

RUL_FEATURES = [
    "Vibration_m_s2",
    "Temperature_C",
    "Humidity_percent",
    "Track_Resistance_Ohm",
    "PLC_CPU_Load_percent",
    "Edge_Anomaly_Score",
    "Predicted_Failure_Prob",
    "Cloud_Health_Index",
    "Component_Age_days",
    "Voltage_V",
    "Current_A",
    "Timer_TON_ms",
    "Timer_TCH_ms",
    "Signal_Transition_Delay_ms",
    "Block_Clearance_Time_s",
    "Train_Headway_s",
    "Ambient_Temp_C",
    "Dust_Index_ppm",
    "IF_Flag",
]

RUL_TARGET = "RUL_Predicted_days"


# ── TRAINING ──────────────────────────────────────────────────────────────────

def train_rul_model(
    df: pd.DataFrame,
    n_estimators: int = 300,
    test_size: float = TEST_SIZE,
) -> Tuple[GradientBoostingRegressor, MinMaxScaler, Dict[str, float], Tuple[np.ndarray, np.ndarray]]:
    """Train a Gradient Boosting Regressor for RUL prediction.

    Uses a time-based split (shuffle=False) to respect temporal ordering.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain RUL_FEATURES columns and RUL_TARGET column.
        Must have IF_Flag column (from Phase 3).
    n_estimators : int
        Number of boosting stages.
    test_size : float
        Fraction of data used as test set.

    Returns
    -------
    (model, scaler, metrics_dict, (y_test, y_pred))
        metrics_dict keys: mae, rmse, r2.
    """
    # Use only available features
    available = [f for f in RUL_FEATURES if f in df.columns]
    if not available:
        raise ValueError("No RUL_FEATURES found in DataFrame")

    X = df[available].values
    y = df[RUL_TARGET].values

    # Time-based split — NO shuffle
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Scale
    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Train
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=0.05,
        max_depth=5,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train_s, y_train)

    # Evaluate
    y_pred = model.predict(X_test_s)
    metrics = {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": float(r2_score(y_test, y_pred)),
    }

    print(f"  MAE  : {metrics['mae']:.2f} days")
    print(f"  RMSE : {metrics['rmse']:.2f} days")
    print(f"  R²   : {metrics['r2']:.4f}")

    # Feature importances
    importances = model.feature_importances_
    top5 = np.argsort(importances)[::-1][:5]
    print("  Top-5 features:")
    for i in top5:
        print(f"    {available[i]}: {importances[i]:.4f}")

    return model, scaler, metrics, (y_test, y_pred)


# ── PREDICTION ────────────────────────────────────────────────────────────────

def predict_rul(
    model: GradientBoostingRegressor,
    scaler: MinMaxScaler,
    df: pd.DataFrame,
) -> np.ndarray:
    """Predict RUL on new data.

    Parameters
    ----------
    model : GradientBoostingRegressor
    scaler : MinMaxScaler
    df : pd.DataFrame
        Must contain RUL_FEATURES columns.

    Returns
    -------
    np.ndarray
        Predicted RUL in days.
    """
    available = [f for f in RUL_FEATURES if f in df.columns]
    X = df[available].values
    X_scaled = scaler.transform(X)
    return model.predict(X_scaled)


def rul_alert_level(days: float) -> str:
    """Map RUL days to alert level string.

    Parameters
    ----------
    days : float

    Returns
    -------
    str
        CRITICAL (< 15), WARNING (< 60), MONITOR (< 120), or HEALTHY.
    """
    if days < 15:
        return "CRITICAL"
    if days < 60:
        return "WARNING"
    if days < 120:
        return "MONITOR"
    return "HEALTHY"


# ── SAVE / LOAD ───────────────────────────────────────────────────────────────

def save_model(model, scaler, features=None,
               path: str = os.path.join(MODELS_DIR, "rul_model.pkl")) -> None:
    """Save the RUL model bundle (model + scaler + feature list).

    Parameters
    ----------
    model : GradientBoostingRegressor
    scaler : MinMaxScaler
    features : list or None
    path : str
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bundle = {"model": model, "scaler": scaler, "features": features or RUL_FEATURES}
    try:
        with open(path, "wb") as f:
            pickle.dump(bundle, f)
        print(f"  Saved RUL model → {path}")
    except Exception as e:
        raise RuntimeError(f"Failed to save RUL model: {e}")


def load_model(path: str = os.path.join(MODELS_DIR, "rul_model.pkl")) -> dict:
    """Load a saved RUL model bundle.

    Returns
    -------
    dict with keys: model, scaler, features.
    """
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"RUL model not found: {path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load RUL model: {e}")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from preprocess import preprocess_pipeline
    from anomaly_model import train_isolation_forest

    df, vibr_df, encoders = preprocess_pipeline()

    # Need IF_Flag from anomaly model
    print("\n[RUL] Running anomaly detection first...")
    _, _, labels, _ = train_isolation_forest(df)
    df["IF_Flag"] = (labels == -1).astype(int)

    print("\n[RUL] Training RUL model...")
    model, scaler, metrics, (y_test, y_pred) = train_rul_model(df)
    save_model(model, scaler)
    print("\nDone.")
