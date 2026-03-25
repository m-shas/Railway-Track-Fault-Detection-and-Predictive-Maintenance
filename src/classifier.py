"""
Railway Track Fault Detection - Fault Classification Model
Phase 5: Random Forest 10-class fault classifier (C1-C10).

Public API:
    train_classifier(df, test_size) -> (model, scaler, y_test, y_pred, accuracy)
    predict_fault(model, scaler, le, df) -> (labels, confidence, descriptions)
    save_model(bundle, path)
    load_model(path) -> bundle dict
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import Tuple, Dict, List

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

RANDOM_STATE = 42
TEST_SIZE = 0.20

CLF_FEATURES = [
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
    "HMI_Alert_Code_enc",
]

CLF_TARGET = "Failure_Type_enc"

FAULT_DESCRIPTIONS: Dict[str, Tuple[str, str]] = {
    "C1":  ("Rail Crack / Fracture",             "Immediate rail replacement; restrict speed to 20km/h"),
    "C2":  ("Loose Fastener / Joint Failure",     "Tighten/replace fasteners; inspect adjacent joints"),
    "C3":  ("Short Circuit in Track Circuit",     "Check track resistance; inspect bonding wires"),
    "C4":  ("Ballast Degradation",                "Re-tamp ballast; schedule geometry correction"),
    "C5":  ("Thermal Buckling Risk",              "Apply de-stressing procedure; monitor rail temp"),
    "C6":  ("Gauge Widening",                     "Re-gauge track; inspect sleeper anchors"),
    "C7":  ("Wheel Impact Damage (Flat Spot)",    "Profile grinding; inspect affected rail section"),
    "C8":  ("Signalling Relay Malfunction",       "Replace relay module; test signal interlocking"),
    "C9":  ("PLC / Controller Overload",          "Restart PLC; review scan cycle timing"),
    "C10": ("Corrosion / Environmental Damage",   "Apply anti-corrosion treatment; replace corroded section"),
}

N_FAULT_CLASSES = 10


# ── TRAINING ──────────────────────────────────────────────────────────────────

def train_classifier(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
) -> Tuple[RandomForestClassifier, StandardScaler, np.ndarray, np.ndarray, float]:
    """Train a Random Forest fault classifier.

    Uses stratified split and class_weight=balanced.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain CLF_FEATURES and CLF_TARGET columns.
    test_size : float

    Returns
    -------
    (model, scaler, y_test, y_pred, accuracy)
    """
    available = [f for f in CLF_FEATURES if f in df.columns]
    if not available:
        raise ValueError("No CLF_FEATURES found in DataFrame")

    X = df[available].values
    y = df[CLF_TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    accuracy = float(accuracy_score(y_test, y_pred))

    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  (Expected ~10% on synthetic data — 10 random classes)")

    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    print(f"  Confusion matrix shape: {cm.shape}")

    return model, scaler, y_test, y_pred, accuracy


# ── PREDICTION ────────────────────────────────────────────────────────────────

def predict_fault(
    model: RandomForestClassifier,
    scaler: StandardScaler,
    le,  # LabelEncoder
    df: pd.DataFrame,
) -> Tuple[List[str], np.ndarray, List[str]]:
    """Predict fault type on new data.

    Parameters
    ----------
    model : RandomForestClassifier
    scaler : StandardScaler
    le : LabelEncoder
        Fitted LabelEncoder for Failure_Type.
    df : pd.DataFrame
        Must contain CLF_FEATURES columns.

    Returns
    -------
    (labels, confidence, descriptions)
        labels: list of strings like "C1", "C7".
        confidence: np.ndarray of max class probabilities.
        descriptions: list of human-readable fault descriptions.
    """
    available = [f for f in CLF_FEATURES if f in df.columns]
    X = df[available].values
    X_scaled = scaler.transform(X)

    y_pred_enc = model.predict(X_scaled)
    proba = model.predict_proba(X_scaled)
    confidence = proba.max(axis=1)

    # Decode labels
    labels = le.inverse_transform(y_pred_enc).tolist()

    # Map to descriptions
    descriptions = []
    for label in labels:
        if label in FAULT_DESCRIPTIONS:
            desc, action = FAULT_DESCRIPTIONS[label]
            descriptions.append(f"{desc} — {action}")
        else:
            descriptions.append(f"Unknown fault type: {label}")

    return labels, confidence, descriptions


# ── SAVE / LOAD ───────────────────────────────────────────────────────────────

def save_model(model, scaler, le, features=None,
               path: str = os.path.join(MODELS_DIR, "clf_model.pkl")) -> None:
    """Save classifier bundle (model + scaler + label encoder + features).

    Parameters
    ----------
    model : RandomForestClassifier
    scaler : StandardScaler
    le : LabelEncoder
    features : list or None
    path : str
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bundle = {
        "model": model, "scaler": scaler,
        "label_encoder": le, "features": features or CLF_FEATURES,
    }
    try:
        with open(path, "wb") as f:
            pickle.dump(bundle, f)
        print(f"  Saved classifier → {path}")
    except Exception as e:
        raise RuntimeError(f"Failed to save classifier: {e}")


def load_model(path: str = os.path.join(MODELS_DIR, "clf_model.pkl")) -> dict:
    """Load a saved classifier bundle.

    Returns
    -------
    dict with keys: model, scaler, label_encoder, features.
    """
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Classifier not found: {path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load classifier: {e}")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from preprocess import preprocess_pipeline
    from anomaly_model import train_isolation_forest

    df, vibr_df, encoders = preprocess_pipeline()

    _, _, labels, _ = train_isolation_forest(df)
    df["IF_Flag"] = (labels == -1).astype(int)

    print("\n[Classifier] Training fault classifier...")
    model, scaler, y_test, y_pred, accuracy = train_classifier(df)
    save_model(model, scaler, encoders["Failure_Type"])
    print("\nDone.")
