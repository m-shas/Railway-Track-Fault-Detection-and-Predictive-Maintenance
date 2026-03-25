"""
Railway Track Fault Detection - Alert Engine
Phase 6: Rule-based + ML-score alerts with maintenance action mapping.

Public API:
    compute_alert_level(row)  -> str (CRITICAL | WARNING | HEALTHY)
    compute_alert_score(row)  -> float 0.0-100.0
    generate_alert_log(df)    -> pd.DataFrame (CRITICAL + WARNING only, sorted desc)
    send_alert_summary(alert_log)  -> None (prints console report)
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Tuple

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

THRESHOLDS = {
    "CRITICAL": {
        "failure_prob": 0.75,
        "rul": 15,
        "vibration": 0.70,
        "temp": 55,
    },
    "WARNING": {
        "failure_prob": 0.45,
        "rul": 60,
        "vibration": 0.55,
        "temp": 45,
    },
}

MAINTENANCE_RULES: Dict[str, Tuple[str, str, str]] = {
    "C1":  ("Rail Crack / Fracture",           "Immediate rail replacement; restrict speed to 20km/h",   "CRITICAL"),
    "C2":  ("Loose Fastener / Joint Failure",   "Tighten/replace fasteners; inspect adjacent joints",     "HIGH"),
    "C3":  ("Short Circuit in Track Circuit",   "Check track resistance; inspect bonding wires",          "CRITICAL"),
    "C4":  ("Ballast Degradation",              "Re-tamp ballast; schedule geometry correction",          "MEDIUM"),
    "C5":  ("Thermal Buckling Risk",            "Apply de-stressing procedure; monitor rail temp",        "HIGH"),
    "C6":  ("Gauge Widening",                   "Re-gauge track; inspect sleeper anchors",                "HIGH"),
    "C7":  ("Wheel Impact Damage (Flat Spot)",  "Profile grinding; inspect affected rail section",        "MEDIUM"),
    "C8":  ("Signalling Relay Malfunction",     "Replace relay module; test signal interlocking",         "CRITICAL"),
    "C9":  ("PLC / Controller Overload",        "Restart PLC; review scan cycle timing",                  "MEDIUM"),
    "C10": ("Corrosion / Environmental Damage", "Apply anti-corrosion treatment; replace corroded section","HIGH"),
}


# ── ALERT LEVEL ───────────────────────────────────────────────────────────────

def compute_alert_level(row) -> str:
    """Determine alert level for a single row using threshold rules.

    Checks: failure_prob, RUL, vibration, and temperature thresholds.

    Parameters
    ----------
    row : pd.Series or dict-like

    Returns
    -------
    str
        CRITICAL, WARNING, or HEALTHY.
    """
    prob = row.get("Predicted_Failure_Prob", 0)
    rul = row.get("RUL_Predicted_days", 999)
    vibr = row.get("Vibration_m_s2", 0)
    temp = row.get("Temperature_C", 0)

    ct = THRESHOLDS["CRITICAL"]
    if prob > ct["failure_prob"] or rul < ct["rul"] or vibr > ct["vibration"] or temp > ct["temp"]:
        return "CRITICAL"

    wt = THRESHOLDS["WARNING"]
    if prob > wt["failure_prob"] or rul < wt["rul"] or vibr > wt["vibration"] or temp > wt["temp"]:
        return "WARNING"

    return "HEALTHY"


# ── ALERT SCORE ───────────────────────────────────────────────────────────────

def compute_alert_score(row) -> float:
    """Compute weighted urgency score 0-100 (higher = more urgent).

    Weights:
        40% failure_prob * 100
        30% (1 - rul/365) * 100     (clamped to [0,100])
        15% anomaly_score * 100
        15% IF_Flag * 100

    Parameters
    ----------
    row : pd.Series or dict-like

    Returns
    -------
    float
        Alert score clipped to [0, 100].
    """
    prob = row.get("Predicted_Failure_Prob", 0)
    rul = row.get("RUL_Predicted_days", 365)
    anomaly = row.get("Edge_Anomaly_Score", 0)
    if_flag = row.get("IF_Flag", 0)

    rul_urgency = max(0, min(1, 1 - rul / 365))

    score = (
        0.40 * prob * 100 +
        0.30 * rul_urgency * 100 +
        0.15 * anomaly * 100 +
        0.15 * if_flag * 100
    )
    return float(np.clip(score, 0, 100))


# ── ALERT LOG GENERATION ─────────────────────────────────────────────────────

def generate_alert_log(df: pd.DataFrame) -> pd.DataFrame:
    """Generate alert log with only CRITICAL and WARNING rows.

    Adds Alert_Level, Alert_Score, Fault_Description, Recommended_Action,
    and Base_Priority columns. Sorted by Alert_Score descending.

    Parameters
    ----------
    df : pd.DataFrame
        Fully featured DataFrame (must have IF_Flag, Failure_Type, etc.)

    Returns
    -------
    pd.DataFrame
        Filtered and sorted alert log (no HEALTHY rows).
    """
    alerts = df.copy()

    # Compute alert level and score
    alerts["Alert_Level"] = alerts.apply(compute_alert_level, axis=1)
    alerts["Alert_Score"] = alerts.apply(compute_alert_score, axis=1)

    # Add maintenance info from MAINTENANCE_RULES
    fault_desc = []
    rec_action = []
    base_prio = []
    for _, row in alerts.iterrows():
        ft = row.get("Failure_Type", "")
        if ft in MAINTENANCE_RULES:
            desc, action, prio = MAINTENANCE_RULES[ft]
            fault_desc.append(desc)
            rec_action.append(action)
            base_prio.append(prio)
        else:
            fault_desc.append("Unknown")
            rec_action.append("Inspect and diagnose")
            base_prio.append("LOW")

    alerts["Fault_Description"] = fault_desc
    alerts["Recommended_Action"] = rec_action
    alerts["Base_Priority"] = base_prio

    # Filter: keep only CRITICAL and WARNING
    alerts = alerts[alerts["Alert_Level"].isin(["CRITICAL", "WARNING"])]

    # Sort by Alert_Score descending
    alerts = alerts.sort_values("Alert_Score", ascending=False).reset_index(drop=True)

    return alerts


# ── CONSOLE SUMMARY ──────────────────────────────────────────────────────────

def send_alert_summary(alert_log: pd.DataFrame) -> None:
    """Print a formatted console report of the alert log.

    Shows total counts and top-5 critical items.

    Parameters
    ----------
    alert_log : pd.DataFrame
    """
    n_critical = (alert_log["Alert_Level"] == "CRITICAL").sum()
    n_warning = (alert_log["Alert_Level"] == "WARNING").sum()

    print("\n" + "=" * 70)
    print("ALERT SUMMARY REPORT")
    print("=" * 70)
    print(f"  Total alerts   : {len(alert_log)}")
    print(f"  CRITICAL       : {n_critical}")
    print(f"  WARNING        : {n_warning}")
    print(f"  Avg Alert Score: {alert_log['Alert_Score'].mean():.1f}")

    print("\n  Top-5 Most Urgent:")
    top5 = alert_log.head(5)
    for i, (_, row) in enumerate(top5.iterrows(), 1):
        print(f"    {i}. [{row['Alert_Level']}] Block {row.get('Track_Block_ID', '?')} "
              f"| Score: {row['Alert_Score']:.1f} "
              f"| {row.get('Fault_Description', 'N/A')}")
    print("=" * 70)


# ── EXPORT ────────────────────────────────────────────────────────────────────

def export_alert_log(alert_log: pd.DataFrame,
                     path: str = os.path.join(OUTPUTS_DIR, "alert_log.csv")) -> None:
    """Export alert log to CSV.

    Parameters
    ----------
    alert_log : pd.DataFrame
    path : str
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        alert_log.to_csv(path, index=False)
        print(f"  Exported alert log → {path}  ({len(alert_log)} rows)")
    except Exception as e:
        raise RuntimeError(f"Failed to export alert log: {e}")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from preprocess import preprocess_pipeline
    from anomaly_model import train_isolation_forest

    df, vibr_df, encoders = preprocess_pipeline()
    _, _, labels, _ = train_isolation_forest(df)
    df["IF_Flag"] = (labels == -1).astype(int)

    print("\n[Alerts] Generating alert log...")
    alert_log = generate_alert_log(df)
    send_alert_summary(alert_log)
    export_alert_log(alert_log)
    print("\nDone.")
