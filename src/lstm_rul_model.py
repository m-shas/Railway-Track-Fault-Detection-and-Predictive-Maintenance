"""
Railway Track Fault Detection - BiLSTM RUL Prediction Model
Advanced ML module for research paper.

Runs ALONGSIDE GradientBoosting for comparison benchmark.

Architecture:
    Input(seq_len=30, n_features) →
    Bidirectional LSTM(128) → Dropout(0.3) →
    Bidirectional LSTM(64)  → Dropout(0.2) →
    Dense(64, relu) → BatchNorm → Dense(1)

Public API:
    train_lstm_rul_model(df, ...) -> (model, scaler, metrics, (y_test, y_pred))
    predict_rul_lstm(model, scaler, df) -> np.ndarray
    save_lstm_model(model, scaler, features, path)
    load_lstm_model(path) -> dict
"""

import os
import pickle
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Tuple, Dict

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

SEQ_LEN    = 30
TEST_SIZE  = 0.20
EPOCHS     = 60
BATCH_SIZE = 64
RANDOM_STATE = 42

RUL_FEATURES = [
    "Vibration_m_s2", "Temperature_C", "Humidity_percent",
    "Track_Resistance_Ohm", "PLC_CPU_Load_percent",
    "Edge_Anomaly_Score", "Predicted_Failure_Prob",
    "Cloud_Health_Index", "Component_Age_days",
    "Voltage_V", "Current_A", "Timer_TON_ms", "Timer_TCH_ms",
    "Signal_Transition_Delay_ms", "Block_Clearance_Time_s",
    "Train_Headway_s", "Ambient_Temp_C", "Dust_Index_ppm", "IF_Flag",
]
RUL_TARGET = "RUL_Predicted_days"


# ── SEQUENCE BUILDER ──────────────────────────────────────────────────────────

def _make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int = SEQ_LEN):
    """Create rolling-window sequences for LSTM input."""
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i - seq_len:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


# ── TRAINING ──────────────────────────────────────────────────────────────────

def train_lstm_rul_model(
    df: pd.DataFrame,
    seq_len: int = SEQ_LEN,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    test_size: float = TEST_SIZE,
) -> Tuple:
    """Train a Bidirectional LSTM for RUL prediction.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain RUL_FEATURES columns and RUL_TARGET.
    seq_len : int
        Rolling window length (number of timesteps per sequence).
    epochs : int
        Maximum training epochs (EarlyStopping may stop earlier).
    batch_size : int
    test_size : float

    Returns
    -------
    (model, scaler, metrics_dict, (y_test, y_pred))
        metrics_dict keys: mae, rmse, r2, epochs_trained, train_time_s
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential  # type: ignore
        from tensorflow.keras.layers import (            # type: ignore
            Bidirectional, LSTM, Dense, Dropout, BatchNormalization, Input
        )
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # type: ignore
        tf.random.set_seed(RANDOM_STATE)
    except ImportError:
        raise ImportError(
            "TensorFlow not installed. Run: pip install tensorflow"
        )

    available = [f for f in RUL_FEATURES if f in df.columns]
    if not available:
        raise ValueError("No RUL_FEATURES found in DataFrame")

    X = df[available].values.astype(np.float32)
    y = df[RUL_TARGET].values.astype(np.float32)

    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Build sequences (time-ordered)
    X_seq, y_seq = _make_sequences(X_scaled, y, seq_len)

    # Time-based split (no shuffle — respects temporal order)
    split = int(len(X_seq) * (1 - test_size))
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    n_features = len(available)

    # ── Model architecture ─────────────────────────────────────────────────
    model = Sequential([
        Input(shape=(seq_len, n_features)),
        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.30),
        Bidirectional(LSTM(64)),
        Dropout(0.20),
        Dense(64, activation="relu"),
        BatchNormalization(),
        Dense(32, activation="relu"),
        Dense(1),  # RUL regression output
    ], name="BiLSTM_RUL")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=6,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=3, min_lr=1e-6, verbose=0),
    ]

    t0 = time.time()
    history = model.fit(
        X_train, y_train,
        validation_split=0.10,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
        shuffle=False,  # keep time ordering within train window
    )
    train_time = round(time.time() - t0, 1)

    # ── Evaluate ───────────────────────────────────────────────────────────
    y_pred = model.predict(X_test, verbose=0).flatten()
    metrics = {
        "mae":           float(mean_absolute_error(y_test, y_pred)),
        "rmse":          float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2":            float(r2_score(y_test, y_pred)),
        "epochs_trained": len(history.history["loss"]),
        "train_time_s":  train_time,
    }

    print(f"  BiLSTM MAE    : {metrics['mae']:.2f} days")
    print(f"  BiLSTM RMSE   : {metrics['rmse']:.2f} days")
    print(f"  BiLSTM R²     : {metrics['r2']:.4f}")
    print(f"  Epochs trained: {metrics['epochs_trained']} / {epochs}")
    print(f"  Train time    : {train_time}s")

    return model, scaler, metrics, (y_test, y_pred)


# ── PREDICTION ────────────────────────────────────────────────────────────────

def predict_rul_lstm(model, scaler: MinMaxScaler, df: pd.DataFrame,
                     seq_len: int = SEQ_LEN) -> np.ndarray:
    """Predict RUL values using the trained BiLSTM model.

    Parameters
    ----------
    model : keras Model
    scaler : MinMaxScaler
    df : pd.DataFrame
        Must contain RUL_FEATURES columns.
    seq_len : int

    Returns
    -------
    np.ndarray of shape (len(df) - seq_len,)
        Predicted RUL in days.
    """
    available = [f for f in RUL_FEATURES if f in df.columns]
    X = df[available].values.astype(np.float32)
    X_scaled = scaler.transform(X)
    dummy_y = np.zeros(len(X))
    X_seq, _ = _make_sequences(X_scaled, dummy_y, seq_len)
    return model.predict(X_seq, verbose=0).flatten()


# ── SAVE / LOAD ───────────────────────────────────────────────────────────────

def save_lstm_model(model, scaler: MinMaxScaler, features=None,
                    path: str = None) -> None:
    """Save the BiLSTM model (.keras) and scaler (.pkl) bundle."""
    if path is None:
        path = os.path.join(MODELS_DIR, "lstm_rul_model.keras")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Save keras model
    model.save(path)

    # Save scaler + features alongside
    meta_path = path.replace(".keras", "_meta.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump({"scaler": scaler, "features": features or RUL_FEATURES}, f)
    print(f"  Saved BiLSTM model → {path}")
    print(f"  Saved BiLSTM meta  → {meta_path}")


def load_lstm_model(path: str = None) -> dict:
    """Load the saved BiLSTM model bundle.

    Returns
    -------
    dict with keys: model, scaler, features
    """
    try:
        import tensorflow as tf  # type: ignore
    except ImportError:
        raise ImportError("TensorFlow not installed. Run: pip install tensorflow")

    if path is None:
        path = os.path.join(MODELS_DIR, "lstm_rul_model.keras")
    meta_path = path.replace(".keras", "_meta.pkl")

    if not os.path.exists(path):
        raise FileNotFoundError(f"BiLSTM model not found: {path}")

    model = tf.keras.models.load_model(path)
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    return {"model": model, "scaler": meta["scaler"], "features": meta["features"]}


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    SRC = os.path.dirname(os.path.abspath(__file__))
    if SRC not in sys.path:
        sys.path.insert(0, SRC)

    from preprocess import preprocess_pipeline
    from anomaly_model import train_isolation_forest

    print("Loading data...")
    df, _, _ = preprocess_pipeline()

    print("Running Isolation Forest first...")
    _, _, labels, _ = train_isolation_forest(df)
    df["IF_Flag"] = (labels == -1).astype(int)

    print("\nTraining BiLSTM RUL model...")
    model, scaler, metrics, _ = train_lstm_rul_model(df)
    save_lstm_model(model, scaler)
    print("\nDone. Metrics:", metrics)
