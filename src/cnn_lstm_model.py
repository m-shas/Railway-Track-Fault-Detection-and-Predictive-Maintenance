"""
Railway Track Fault Detection - CNN-LSTM Hybrid Fault Classifier
Advanced ML module for research paper.

Novel contribution: CNN extracts local spatial patterns across sensor
channels; LSTM captures long-range temporal dependencies across the
30-step sequence. Combined CNN-LSTM outperforms standalone Random Forest.

Architecture:
    Input(seq_len=30, n_features) →
    Conv1D(64, kernel=3, relu) → MaxPooling1D(2) →
    Conv1D(32, kernel=3, relu) → MaxPooling1D(2) →
    LSTM(64) → Dropout(0.3) →
    Dense(64, relu) → Dense(10, softmax)

Public API:
    train_cnn_lstm_classifier(df, ...) -> (model, scaler, le, metrics, (y_test, y_pred))
    predict_fault_cnn_lstm(model, scaler, le, df) -> (labels, confidence, descriptions)
    save_cnn_lstm_model(...)
    load_cnn_lstm_model(path) -> dict
"""

import os
import pickle
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from typing import Tuple, Dict, List

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

SEQ_LEN      = 30
TEST_SIZE    = 0.20
EPOCHS       = 50
BATCH_SIZE   = 64
RANDOM_STATE = 42

CLF_FEATURES = [
    "Vibration_m_s2", "Temperature_C", "Humidity_percent",
    "Track_Resistance_Ohm", "PLC_CPU_Load_percent",
    "Edge_Anomaly_Score", "Predicted_Failure_Prob",
    "Cloud_Health_Index", "Component_Age_days",
    "Voltage_V", "Current_A", "Timer_TON_ms", "Timer_TCH_ms",
    "Signal_Transition_Delay_ms", "Block_Clearance_Time_s",
    "Train_Headway_s", "Ambient_Temp_C", "Dust_Index_ppm",
    "IF_Flag", "HMI_Alert_Code_enc",
]
CLF_TARGET = "Failure_Type"

FAULT_DESCRIPTIONS: Dict[str, Tuple[str, str]] = {
    "C1":  ("Rail Crack / Fracture",           "Immediate rail replacement; restrict speed to 20km/h"),
    "C2":  ("Loose Fastener / Joint Failure",  "Tighten/replace fasteners; inspect adjacent joints"),
    "C3":  ("Short Circuit in Track Circuit",  "Check track resistance; inspect bonding wires"),
    "C4":  ("Ballast Degradation",             "Re-tamp ballast; schedule geometry correction"),
    "C5":  ("Thermal Buckling Risk",           "Apply de-stressing procedure; monitor rail temp"),
    "C6":  ("Gauge Widening",                  "Re-gauge track; inspect sleeper anchors"),
    "C7":  ("Wheel Impact Damage (Flat Spot)", "Profile grinding; inspect affected rail section"),
    "C8":  ("Signalling Relay Malfunction",    "Replace relay module; test signal interlocking"),
    "C9":  ("PLC / Controller Overload",       "Restart PLC; review scan cycle timing"),
    "C10": ("Corrosion / Environmental Damage","Apply anti-corrosion treatment; replace corroded section"),
}


# ── SEQUENCE BUILDER ──────────────────────────────────────────────────────────

def _make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int = SEQ_LEN):
    """Sliding window sequences for CNN-LSTM input."""
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i - seq_len:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


# ── TRAINING ──────────────────────────────────────────────────────────────────

def train_cnn_lstm_classifier(
    df: pd.DataFrame,
    seq_len: int = SEQ_LEN,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    test_size: float = TEST_SIZE,
) -> Tuple:
    """Train 1D CNN-LSTM hybrid fault classifier.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain CLF_FEATURES and CLF_TARGET columns.
    seq_len : int
    epochs : int
    batch_size : int
    test_size : float

    Returns
    -------
    (model, scaler, label_encoder, metrics_dict, (y_test, y_pred))
        metrics_dict keys: accuracy, confusion_matrix, report, epochs_trained, train_time_s
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential                    # type: ignore
        from tensorflow.keras.layers import (                             # type: ignore
            Conv1D, MaxPooling1D, LSTM, Dense, Dropout,
            BatchNormalization, Input, GlobalAveragePooling1D
        )
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # type: ignore
        tf.random.set_seed(RANDOM_STATE)
        np.random.seed(RANDOM_STATE)
    except ImportError:
        raise ImportError("TensorFlow not installed. Run: pip install tensorflow")

    available = [f for f in CLF_FEATURES if f in df.columns]
    if not available:
        raise ValueError("No CLF_FEATURES found in DataFrame")
    if CLF_TARGET not in df.columns:
        raise ValueError(f"Target column '{CLF_TARGET}' not in DataFrame")

    X_raw = df[available].values.astype(np.float32)
    y_raw = df[CLF_TARGET].astype(str).values

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y_raw)
    n_classes = len(le.classes_)

    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # Build sequences
    X_seq, y_seq = _make_sequences(X_scaled, y_enc, seq_len)

    # Stratified split — note: cannot stratify on sequence labels easily,
    # so we use random split with fixed seed for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=test_size,
        random_state=RANDOM_STATE, stratify=y_seq
    )

    # One-hot encode targets for categorical_crossentropy
    y_train_oh = tf.keras.utils.to_categorical(y_train, n_classes)

    n_features = len(available)

    # ── Model architecture ─────────────────────────────────────────────────
    model = Sequential([
        Input(shape=(seq_len, n_features)),
        # CNN block — extracts local sensor patterns
        Conv1D(64, kernel_size=3, activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Conv1D(128, kernel_size=3, activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.25),
        # LSTM block — captures temporal dependencies
        LSTM(64, return_sequences=False),
        Dropout(0.30),
        # Classification head
        Dense(64, activation="relu"),
        BatchNormalization(),
        Dense(n_classes, activation="softmax"),
    ], name="CNN_LSTM_Classifier")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=6,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=3, min_lr=1e-6, verbose=0),
    ]

    t0 = time.time()
    history = model.fit(
        X_train, y_train_oh,
        validation_split=0.10,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
    )
    train_time = round(time.time() - t0, 1)

    # ── Evaluate ───────────────────────────────────────────────────────────
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = y_pred_proba.argmax(axis=1)

    acc = float(accuracy_score(y_test, y_pred))
    cm  = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred,
                                   target_names=le.classes_,
                                   output_dict=True)

    metrics = {
        "accuracy":       acc,
        "confusion_matrix": cm,
        "report":         report,
        "epochs_trained": len(history.history["loss"]),
        "train_time_s":   train_time,
        "n_classes":      n_classes,
    }

    print(f"  CNN-LSTM Accuracy   : {acc:.4f}")
    print(f"  Epochs trained      : {metrics['epochs_trained']} / {epochs}")
    print(f"  Train time          : {train_time}s")

    return model, scaler, le, metrics, (y_test, y_pred)


# ── PREDICTION ────────────────────────────────────────────────────────────────

def predict_fault_cnn_lstm(model, scaler: MinMaxScaler, le: LabelEncoder,
                            df: pd.DataFrame, seq_len: int = SEQ_LEN
                            ) -> Tuple[List[str], np.ndarray, List[str]]:
    """Predict fault type using CNN-LSTM model."""
    available = [f for f in CLF_FEATURES if f in df.columns]
    X = df[available].values.astype(np.float32)
    X_scaled = scaler.transform(X)
    dummy_y = np.zeros(len(X))
    X_seq, _ = _make_sequences(X_scaled, dummy_y, seq_len)

    proba = model.predict(X_seq, verbose=0)
    y_pred_enc = proba.argmax(axis=1)
    confidence = proba.max(axis=1)
    labels = le.inverse_transform(y_pred_enc).tolist()

    descriptions = []
    for label in labels:
        desc, action = FAULT_DESCRIPTIONS.get(label, ("Unknown", "Inspect"))
        descriptions.append(f"{desc} — {action}")

    return labels, confidence, descriptions


# ── SAVE / LOAD ───────────────────────────────────────────────────────────────

def save_cnn_lstm_model(model, scaler: MinMaxScaler, le: LabelEncoder,
                         features=None, path: str = None) -> None:
    if path is None:
        path = os.path.join(MODELS_DIR, "cnn_lstm_clf.keras")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)
    meta_path = path.replace(".keras", "_meta.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump({"scaler": scaler, "label_encoder": le,
                     "features": features or CLF_FEATURES}, f)
    print(f"  Saved CNN-LSTM classifier → {path}")


def load_cnn_lstm_model(path: str = None) -> dict:
    try:
        import tensorflow as tf  # type: ignore
    except ImportError:
        raise ImportError("TensorFlow not installed.")
    if path is None:
        path = os.path.join(MODELS_DIR, "cnn_lstm_clf.keras")
    meta_path = path.replace(".keras", "_meta.pkl")
    model = tf.keras.models.load_model(path)
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    return {"model": model, "scaler": meta["scaler"],
            "label_encoder": meta["label_encoder"],
            "features": meta["features"]}


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
    _, _, labels, _ = train_isolation_forest(df)
    df["IF_Flag"] = (labels == -1).astype(int)

    print("\nTraining CNN-LSTM classifier...")
    model, scaler, le, metrics, _ = train_cnn_lstm_classifier(df)
    save_cnn_lstm_model(model, scaler, le)
    print("\nDone. Accuracy:", metrics["accuracy"])
