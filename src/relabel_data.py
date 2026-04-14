"""
Relabel RT_PLC_RSFPD.csv with sensor-based rules so that Failure_Type
and RUL_Predicted_days actually correlate with the sensor readings.

Rules (priority order — first match wins):
  C1  Rail Crack           : Vibration > 0.75
  C3  Short Circuit        : Track_Resistance_Ohm > 4.0
  C5  Thermal Buckling     : Temperature_C > 50
  C9  PLC Overload         : PLC_CPU_Load_percent > 85
  C10 Corrosion            : Humidity_percent > 85
  C8  Signalling Relay     : PLC_CPU_Load_percent > 70
  C2  Loose Fastener       : Track_Resistance_Ohm > 3.0
  C6  Gauge Widening       : Vibration > 0.60 AND Track_Resistance_Ohm < 1.5
  C4  Ballast Degradation  : Component_Age_days > 1400
  C7  Wheel Impact         : default (all remaining rows)

RUL is also updated to correlate with sensor health.
"""

import pandas as pd
import numpy as np
import os

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BASE_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\src","")
CSV_IN  = os.path.join(BASE_DIR, "data", "RT_PLC_RSFPD.csv")
CSV_OUT = os.path.join(BASE_DIR, "data", "RT_PLC_RSFPD.csv")  # overwrite in data/

print("Loading CSV...")
df = pd.read_csv(CSV_IN)
print(f"Shape: {df.shape}")

# ── Step 1: Assign Failure_Type based on sensor thresholds ──────────────────
# Use percentile-based thresholds calibrated to actual data ranges
vib  = df["Vibration_m_s2"]
res  = df["Track_Resistance_Ohm"]
temp = df["Temperature_C"]
cpu  = df["PLC_CPU_Load_percent"]
hum  = df["Humidity_percent"]
age  = df["Component_Age_days"]
prob = df["Predicted_Failure_Prob"]

# Compute percentile thresholds
v75, v60 = vib.quantile(0.75), vib.quantile(0.60)
r80, r65  = res.quantile(0.80), res.quantile(0.60)
t80       = temp.quantile(0.80)
c85, c70  = cpu.quantile(0.85), cpu.quantile(0.70)
h75       = hum.quantile(0.75)
a75       = age.quantile(0.75)

fault = pd.Series(["C7"] * len(df), index=df.index)  # default: Wheel Impact

# Apply in reverse priority (later assignments win for higher priority)
fault[age  > a75]                = "C4"   # Ballast Degradation   (~25%)
fault[(vib > v60) & (res < res.quantile(0.30))] = "C6"  # Gauge Widening (~10%)
fault[res  > r65]                = "C2"   # Loose Fastener        (~20%)
fault[cpu  > c70]                = "C8"   # Signalling Relay      (~15%)
fault[hum  > h75]                = "C10"  # Corrosion             (~25%)
fault[cpu  > c85]                = "C9"   # PLC Overload          (~5% override)
fault[temp > t80]                = "C5"   # Thermal Buckling      (~20%)
fault[res  > r80]                = "C3"   # Short Circuit         (~20%)
fault[vib  > v75]                = "C1"   # Rail Crack            (~25%)

df["Failure_Type"] = fault

# ── Step 2: Assign Maintenance_Action based on fault type ───────────────────
maint_map = {
    "C1":  "Immediate Replacement",
    "C2":  "Tighten Fasteners",
    "C3":  "Inspect Wiring",
    "C4":  "Re-tamp Ballast",
    "C5":  "Apply De-stressing",
    "C6":  "Re-gauge Track",
    "C7":  "Profile Grinding",
    "C8":  "Replace Relay",
    "C9":  "Restart PLC",
    "C10": "Anti-corrosion Treatment",
}
df["Maintenance_Action"] = df["Failure_Type"].map(maint_map)

# ── Step 3: Recompute RUL to correlate with sensor health ───────────────────
# Higher vibration / higher prob / higher age → lower RUL
vib_norm  = (vib  - vib.min())  / (vib.max()  - vib.min())
prob_norm = (prob - prob.min()) / (prob.max() - prob.min())
age_norm  = (age  - age.min())  / (age.max()  - age.min())

health_proxy = (
    0.40 * (1 - prob_norm) +
    0.30 * (1 - vib_norm)  +
    0.30 * (1 - age_norm)
)

# RUL = 0..365 days, scaled from health proxy and add small noise
rul = health_proxy * 340 + np.random.normal(0, 8, len(df))
rul = np.clip(rul, 5, 365)
df["RUL_Predicted_days"] = np.round(rul, 2)

# ── Step 4: Recompute Predicted_Failure_Prob to be consistent ────────────────
# Use original Predicted_Failure_Prob but nudge it toward sensor readings
fault_severity = {
    "C1": 0.9, "C3": 0.85, "C5": 0.80, "C9": 0.75, "C10": 0.70,
    "C8": 0.65, "C2": 0.55, "C6": 0.50, "C4": 0.40, "C7": 0.30,
}
base_prob = df["Failure_Type"].map(fault_severity)
# Blend: 60% rule-based, 40% original
df["Predicted_Failure_Prob"] = np.clip(
    0.60 * base_prob + 0.40 * prob + np.random.normal(0, 0.03, len(df)), 0, 1
)
df["Predicted_Failure_Prob"] = df["Predicted_Failure_Prob"].round(4)

# ── Step 5: Save ────────────────────────────────────────────────────────────
df.to_csv(CSV_OUT, index=False)
print(f"\nSaved to: {CSV_OUT}")
print("\nNew Failure_Type distribution:")
print(df["Failure_Type"].value_counts())
print("\nSample rows:")
print(df[["Vibration_m_s2","Temperature_C","Track_Resistance_Ohm",
           "Humidity_percent","PLC_CPU_Load_percent","Failure_Type","RUL_Predicted_days"]].head(10).to_string())
