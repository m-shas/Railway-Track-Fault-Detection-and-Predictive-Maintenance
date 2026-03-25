"""
Railway Track Fault Detection - Data Preprocessing & Feature Engineering
Phase 2: All data loading, cleaning, encoding, and feature engineering functions.

Public API:
    preprocess_pipeline(csv_path, xlsx_path) -> (df, vibr_df, encoders_dict)
    compute_health_score(df)                -> pd.Series (0-100)
    compute_alert_level(row)                -> str: CRITICAL|WARNING|HEALTHY
    make_sequences(data, targets, seq_len)  -> (X_seq, y_seq)
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, Dict, Optional

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

CSV_PATH = os.path.join(DATA_DIR, "RT_PLC_RSFPD.csv")
XLSX_PATH = os.path.join(DATA_DIR, "vibration_analysis_graph.xlsx")

CRITICAL_PROB = 0.75
CRITICAL_RUL = 15
WARNING_PROB = 0.45
WARNING_RUL = 60

CATEGORICAL_COLS = ["Failure_Type", "Occupancy_State", "HMI_Alert_Code", "Maintenance_Action"]


# ── DATA LOADING ──────────────────────────────────────────────────────────────

def load_csv(path: str = CSV_PATH) -> pd.DataFrame:
    """Load the main PLC sensor CSV dataset.

    Parameters
    ----------
    path : str
        Absolute or relative path to RT_PLC_RSFPD.csv.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame sorted by Timestamp ascending.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist.
    """
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found: {path}")
    except Exception as e:
        raise RuntimeError(f"Error loading CSV: {e}")

    # Parse timestamp and sort
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)
    return df


def load_vibration_xlsx(path: str = XLSX_PATH) -> pd.DataFrame:
    """Load a vibration analysis XLSX file (sheet 0).

    Reads the file, renames accelerometer columns to x/y/z if needed,
    computes accel_magnitude = sqrt(x² + y² + z²), and adds rolling
    statistics plus z-score anomaly flag.

    Parameters
    ----------
    path : str
        Path to an .xlsx vibration file.

    Returns
    -------
    pd.DataFrame
        Vibration DataFrame with accel_magnitude, rolling_mag,
        vibr_zscore, and vibr_anomaly columns.
    """
    try:
        df = pd.read_excel(path, sheet_name=0, engine="openpyxl")
    except FileNotFoundError:
        raise FileNotFoundError(f"XLSX file not found: {path}")
    except Exception as e:
        raise RuntimeError(f"Error loading XLSX: {e}")

    # Standardise column names — handle various naming conventions
    col_map = {}
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in ("x", "accel_x", "acc_x"):
            col_map[c] = "x"
        elif cl in ("y", "accel_y", "acc_y"):
            col_map[c] = "y"
        elif cl in ("z", "accel_z", "acc_z"):
            col_map[c] = "z"
        elif cl in ("time", "timestamp", "time_s"):
            col_map[c] = "Time"
        elif cl in ("temp", "temperature"):
            col_map[c] = "Temp"
        elif cl in ("hum", "humidity"):
            col_map[c] = "Hum"
    df = df.rename(columns=col_map)

    # Compute acceleration magnitude
    if {"x", "y", "z"}.issubset(df.columns):
        df["accel_magnitude"] = np.sqrt(df["x"] ** 2 + df["y"] ** 2 + df["z"] ** 2)
    else:
        raise ValueError(f"XLSX missing x/y/z columns. Found: {list(df.columns)}")

    # Rolling magnitude (window=30)
    df["rolling_mag"] = df["accel_magnitude"].rolling(window=30, min_periods=1).mean()

    # Z-score based anomaly flag
    mean_mag = df["accel_magnitude"].mean()
    std_mag = df["accel_magnitude"].std()
    if std_mag > 0:
        df["vibr_zscore"] = (df["accel_magnitude"] - mean_mag) / std_mag
    else:
        df["vibr_zscore"] = 0.0
    df["vibr_anomaly"] = (df["vibr_zscore"].abs() > 2.5).astype(int)

    return df


# ── ENCODING ──────────────────────────────────────────────────────────────────

def encode_categoricals(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """Label-encode categorical columns in-place.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain CATEGORICAL_COLS.

    Returns
    -------
    (pd.DataFrame, dict)
        DataFrame with new *_enc columns and dict of fitted LabelEncoders.
    """
    df = df.copy()
    encoders: Dict[str, LabelEncoder] = {}

    for col in CATEGORICAL_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    return df, encoders


# ── ALERT LEVEL & HEALTH SCORE ────────────────────────────────────────────────

def compute_alert_level(row) -> str:
    """Determine alert level for a single row.

    Logic:
        CRITICAL  if failure_prob > 0.75 OR rul < 15 days
        WARNING   if failure_prob > 0.45 OR rul < 60 days
        else      HEALTHY

    Parameters
    ----------
    row : pd.Series or dict-like
        Must have 'Predicted_Failure_Prob' and 'RUL_Predicted_days'.

    Returns
    -------
    str
        One of CRITICAL, WARNING, HEALTHY.
    """
    prob = row.get("Predicted_Failure_Prob", 0)
    rul = row.get("RUL_Predicted_days", 999)

    if prob > CRITICAL_PROB or rul < CRITICAL_RUL:
        return "CRITICAL"
    if prob > WARNING_PROB or rul < WARNING_RUL:
        return "WARNING"
    return "HEALTHY"


def compute_health_score(df: pd.DataFrame) -> pd.Series:
    """Compute a weighted health score 0-100 (higher = healthier).

    Weights:
        40% (1 - failure_prob)
        25% cloud_health_index
        20% (1 - anomaly_score)
        15% rul_normalised (rul / 365)

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.Series
        Health scores clipped to [0, 100].
    """
    prob = df.get("Predicted_Failure_Prob", pd.Series(0.5, index=df.index))
    cloud = df.get("Cloud_Health_Index", pd.Series(0.5, index=df.index))
    anomaly = df.get("Edge_Anomaly_Score", pd.Series(0.5, index=df.index))
    rul = df.get("RUL_Predicted_days", pd.Series(180, index=df.index))

    rul_norm = (rul / 365).clip(0, 1)

    score = (
        0.40 * (1 - prob) +
        0.25 * cloud +
        0.20 * (1 - anomaly) +
        0.15 * rul_norm
    ) * 100

    return score.clip(0, 100)


# ── LSTM SEQUENCE BUILDER ────────────────────────────────────────────────────

def make_sequences(data: np.ndarray, targets: np.ndarray, seq_len: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """Create sequences for LSTM input.

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_features)
    targets : np.ndarray, shape (n_samples,)
    seq_len : int
        Window length.

    Returns
    -------
    (X_seq, y_seq)
        X_seq shape: (n_samples - seq_len, seq_len, n_features)
        y_seq shape: (n_samples - seq_len,)
    """
    X_seq, y_seq = [], []
    for i in range(seq_len, len(data)):
        X_seq.append(data[i - seq_len:i])
        y_seq.append(targets[i])
    return np.array(X_seq), np.array(y_seq)


# ── MAIN ENTRY POINT ─────────────────────────────────────────────────────────

def preprocess_pipeline(csv_path: str = CSV_PATH,
                        xlsx_path: str = XLSX_PATH) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, LabelEncoder]]:
    """Main preprocessing entry point: load, clean, encode, add features.

    Parameters
    ----------
    csv_path : str
        Path to the PLC CSV dataset.
    xlsx_path : str
        Path to a vibration XLSX file.

    Returns
    -------
    (df, vibr_df, encoders_dict)
        df          – fully featured PLC DataFrame.
        vibr_df     – vibration DataFrame with accel_magnitude.
        encoders_dict – dict of LabelEncoders keyed by column name.
    """
    print("[1/5] Loading CSV data...")
    df = load_csv(csv_path)
    print(f"      Shape: {df.shape}")

    print("[2/5] Loading vibration XLSX...")
    vibr_df = load_vibration_xlsx(xlsx_path)
    print(f"      Shape: {vibr_df.shape}")

    print("[3/5] Encoding categoricals...")
    df, encoders = encode_categoricals(df)

    print("[4/5] Computing health score & alert level...")
    df["Health_Score"] = compute_health_score(df)
    df["Alert_Level"] = df.apply(compute_alert_level, axis=1)

    print("[5/5] Preprocessing complete.")
    print(f"      Final PLC shape: {df.shape}")
    print(f"      Alert distribution: {df['Alert_Level'].value_counts().to_dict()}")

    return df, vibr_df, encoders


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df, vibr_df, encoders = preprocess_pipeline()
    print("\nSample data:")
    print(df[["Timestamp", "Track_Block_ID", "Health_Score", "Alert_Level"]].head(10))
