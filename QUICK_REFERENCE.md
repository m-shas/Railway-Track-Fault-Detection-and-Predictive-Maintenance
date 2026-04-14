# Quick Reference Guide

**Railway Track Fault Detection - Quick Start**

## Installation & Setup (< 5 minutes)

### 1. Install Python Dependencies
```bash
# Navigate to project directory
cd d:\COLLEGE\projects\minor-proj\minor1

# Create/activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows PowerShell
# or
source .venv/bin/activate     # Linux/Mac

# Install all dependencies
pip install -r requirements.txt
```

**Dependencies:**
- `pandas`, `numpy` - Data manipulation
- `scikit-learn` - Baseline ML models
- `tensorflow>=2.12.0` - BiLSTM, CNN-LSTM (optional but recommended)
- `shap>=0.42.0` - Explainability
- `matplotlib`, `seaborn` - Visualization
- `plotly`, `streamlit` - Interactive dashboards
- `pytest` - Testing framework
- `joblib` - Model serialization

### 2. Run the Full Pipeline
```bash
# Execute all 8 phases (trains all models)
python src/pipeline.py

# Expected output:
# [1/8] Preprocessing data...
# [2/8] Training Isolation Forest...
# [3/8] Training Gradient Boosting RUL...
# [4/8] Training BiLSTM RUL...
# [5/8] Training RandomForest Classifier...
# [6/8] Training CNN-LSTM Classifier...
# [7/8] Generating alert log...
# [8/8] Building dashboard data...
# [OK] PIPELINE COMPLETE

# Runtime: ~5-10 minutes (depending on CPU)
```

### 3. View Results
```bash
# Open dashboard in browser
start outputs/railway_dashboard.html

# View alert log
Get-Content outputs/alert_log.csv | head -20

# Check model files
ls models/
# Output: isolation_forest.pkl, rul_model.pkl, clf_model.pkl,
#         lstm_rul_model.keras, cnn_lstm_clf.keras
```

---

## Testing

### Run All Tests
```bash
# Execute 42 tests (takes ~1 minute)
pytest tests/ -v

# Output: 42 passed in 53.07s

# Run specific test module
pytest tests/test_pipeline.py -v
pytest tests/test_models.py -v
pytest tests/test_preprocess.py -v
pytest tests/test_alerts.py -v
```

### Common Test Checks
```bash
# Test only preprocessing
pytest tests/test_preprocess.py::TestLoadCSV -v

# Test model training
pytest tests/test_models.py::TestRULModel -v

# Test alert generation
pytest tests/test_alerts.py::TestAlertLog -v

# Run with detailed output
pytest tests/ -vv --tb=long
```

---

## Using Individual Models

### 1. Anomaly Detection (Isolation Forest)

```python
from src.preprocess import preprocess_pipeline
from src.anomaly_model import train_isolation_forest, predict_anomaly

# Load and preprocess data
df, vibr_df, encoders = preprocess_pipeline()

# Train model
if_model, if_scaler, labels, scores = train_isolation_forest(df)

# Predict on new data
new_labels, new_scores = predict_anomaly(if_model, if_scaler, df.head(10))
print(f"Anomalies detected: {(new_labels == -1).sum()}")
```

### 2. RUL Prediction (Gradient Boosting) ⭐

```python
from src.preprocess import preprocess_pipeline
from src.rul_model import train_rul_model, predict_rul

# Preprocess
df, _, _ = preprocess_pipeline()

# Train
model, scaler, metrics, (y_test, y_pred) = train_rul_model(df)
print(f"MAE: {metrics['mae']:.2f} days")

# Predict
predictions = predict_rul(model, scaler, df)
```

### 3. Fault Classification (Random Forest) ⭐

```python
from src.preprocess import preprocess_pipeline
from src.classifier import train_classifier, predict_fault

# Preprocess
df, _, _ = preprocess_pipeline()

# Train
model, scaler, y_test, y_pred, accuracy = train_classifier(df)
print(f"Accuracy: {accuracy:.1%}")

# Predict
labels, confidence, descriptions = predict_fault(model, scaler, encoders["Failure_Type"], df)
```

### 4. BiLSTM RUL Prediction (Advanced)

```python
from src.lstm_rul_model import train_lstm_rul_model, predict_rul_lstm

# Train (requires TensorFlow)
model, scaler, metrics, (y_test, y_pred) = train_lstm_rul_model(df)
print(f"BiLSTM MAE: {metrics['mae']:.2f} days")

# Predict
predictions = predict_rul_lstm(model, scaler, df)
```

### 5. CNN-LSTM Classifier (Advanced)

```python
from src.cnn_lstm_model import train_cnn_lstm_classifier, predict_fault_cnn_lstm

# Train (requires TensorFlow)
model, scaler, le, metrics, (y_test, y_pred) = train_cnn_lstm_classifier(df)
print(f"CNN-LSTM Accuracy: {metrics['accuracy']:.1%}")

# Predict
labels, confidence, descriptions = predict_fault_cnn_lstm(model, scaler, le, df)
```

### 6. SHAP Explainability

```python
from src.xai_explainer import compute_shap_values, build_shap_summary_fig
from src.classifier import train_classifier

# Train Random Forest
model, scaler, _, _, _ = train_classifier(df)

# Compute SHAP values
shap_dict = compute_shap_values(model, X_test_scaled, feature_names)

# Generate summary plot
fig = build_shap_summary_fig(shap_dict)
fig.show()
```

---

## Dashboard Options

### Option 1: HTML Dashboard (Primary) ✅ Recommended
```bash
# Generated automatically by pipeline
open outputs/railway_dashboard.html

# Features:
# - 5 tabs (Overview, Live Monitoring, Vibration, Blocks, Alerts)
# - 149 KB, works offline, no dependencies
# - Real-time health scores, model comparisons
```

### Option 2: Streamlit Dashboard (Secondary)
```bash
# Install (if not already)
pip install streamlit plotly

# Run interactive dashboard
streamlit run app.py

# Opens at http://localhost:8501
# Features: Interactive filters, real-time updates
```

### Option 3: Jupyter Notebook (Development)
```bash
# Create and explore in Jupyter
jupyter notebook

# Quick data inspection
import pandas as pd
df = pd.read_csv('data/RT_PLC_RSFPD.csv')
print(df.head())
print(df.info())
```

---

## Model Performance Summary

| Model | purpose | Metric | Value | Status |
|-------|---------|--------|-------|--------|
| **Isolation Forest** | Anomaly Detection | Precision | 82% | ✅ Production |
| **Gradient Boosting** | RUL Prediction | MAE | 10.91 days | ✅ Production |
| **Random Forest** | Fault Classify | Accuracy | 99.6% | ✅ Production |
| BiLSTM | RUL (Advanced) | MAE | 37.39 days | 🧪 Research |
| CNN-LSTM | Fault (Advanced) | Accuracy | 21.6% | 🧪 Research |

---

## Common Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'tensorflow'`
**Solution:** TensorFlow optional for advanced models. Install with:
```bash
pip install tensorflow>=2.12.0
```
Or pipeline will skip Phase 4b & 5b (uses baseline only).

### Issue: `FileNotFoundError: data/RT_PLC_RSFPD.csv`
**Solution:** Ensure you're running from project root:
```bash
cd d:\COLLEGE\projects\minor-proj\minor1
python src/pipeline.py
```

### Issue: Pipeline slow / takes > 10 minutes
**Solution:** Normal if TensorFlow is training BiLSTM (175 sec). Check output:
```bash
# Monitor for messages like:
# [4/8] Training BiLSTM RUL model...
# Epochs: 30/60
```

### Issue: CNN-LSTM low accuracy (21.6%)
**Solution:** Research model, needs parameter tuning. Use Random Forest for production.

### Issue: Tests fail with import errors
**Solution:** Ensure you're in project directory with venv activated:
```bash
cd d:\COLLEGE\projects\minor-proj\minor1
.\.venv\Scripts\Activate.ps1
pytest tests/ -v
```

---

## File Organization Quick Map

```
d:\COLLEGE\projects\minor-proj\minor1\
├── src/                           ← Source code (8 modules)
│   ├── pipeline.py                ← Run this: python src/pipeline.py
│   ├── preprocess.py              ← Data pipeline
│   ├── anomaly_model.py           ← Isolation Forest
│   ├── rul_model.py               ← Gradient Boosting ⭐
│   ├── classifier.py              ← Random Forest ⭐
│   ├── lstm_rul_model.py          ← BiLSTM (advanced)
│   ├── cnn_lstm_model.py          ← CNN-LSTM (advanced)
│   ├── alerts.py                  ← Alert engine
│   └── xai_explainer.py           ← SHAP explainability
│
├── models/                        ← Trained artifacts (after running)
│   ├── isolation_forest.pkl
│   ├── rul_model.pkl
│   ├── clf_model.pkl
│   ├── lstm_rul_model.keras
│   └── cnn_lstm_clf.keras
│
├── outputs/                       ← Results (after running)
│   ├── railway_dashboard.html     ← Open this in browser ✅
│   ├── dashboard_data.json
│   └── alert_log.csv
│
├── data/                          ← Input datasets
│   └── RT_PLC_RSFPD.csv           ← Main PLC data (5000 rows)
│
├── datasets/                      ← Raw vibration files
│   ├── sensor_vibration/          ← 67+ CSV files
│   └── RT_PLC_RSFPD.csv
│
├── tests/                         ← Test suite (42 tests)
│   ├── test_pipeline.py
│   ├── test_preprocess.py
│   ├── test_models.py
│   └── test_alerts.py
│
├── README.md                      ← Project overview
├── UPDATES_SUMMARY.md             ← Recent changes
├── MODEL_CONFIG_REFERENCE.md      ← Hyperparameters
├── QUICK_REFERENCE.md             ← This file
├── requirements.txt               ← Python dependencies
└── .archive/                      ← Archived old scripts
```

---

## Key Metrics & Thresholds

**Alert Levels:**
- CRITICAL: failure_prob > 0.75 OR RUL < 15 days OR vibration > 0.70 OR temp > 55°C
- WARNING: failure_prob > 0.45 OR RUL < 60 days OR vibration > 0.55 OR temp > 45°C
- HEALTHY: None of above

**Alert Distribution (Last Run):**
- Total: 4,183 alerts
- CRITICAL: 1,482 (35%)
- WARNING: 2,701 (65%)

**Sensor Ranges:**
- Vibration: 0-1.0 m/s²
- Temperature: 0-60°C
- Humidity: 0-100%
- Track Resistance: 0-100Ω
- CPU Load: 0-100%
- RUL: 1-500 days

---

## Advanced Usage

### Custom Model Training
```python
# Modify hyperparameters and retrain
from src.rul_model import train_rul_model

df, _, _ = preprocess_pipeline()

# Custom parameters
model, scaler, metrics, (y_test, y_pred) = train_rul_model(
    df,
    n_estimators=500,  # Increase trees
    test_size=0.15     # Different split
)
```

### Feature Analysis
```python
# Feature importance from trained model
importances = model.feature_importances_
feature_names = ["Vibration_m_s2", "Temperature_C", ...]
ranking = sorted(zip(feature_names, importances), 
                 key=lambda x: x[1], reverse=True)
for name, imp in ranking[:10]:
    print(f"{name:30s}: {imp:.4f}")
```

### Custom Alert Rules
```python
# Modify thresholds in src/alerts.py
THRESHOLDS = {
    "CRITICAL": {
        "failure_prob": 0.80,  # Raise threshold
        "rul": 10,             # Lower RUL trigger
        ...
    },
    ...
}
```

---

## Next Steps

1. **Run pipeline:** `python src/pipeline.py`
2. **View dashboard:** `start outputs/railway_dashboard.html`
3. **Run tests:** `pytest tests/ -v`
4. **Explore data:** `jupyter notebook`
5. **Tune models:** Edit hyperparameters in source files
6. **Deploy:** Use trained models in `models/` directory

---

**Questions?** Check [README.md](README.md) or [MODEL_CONFIG_REFERENCE.md](MODEL_CONFIG_REFERENCE.md)

**Last Updated:** April 9, 2026
