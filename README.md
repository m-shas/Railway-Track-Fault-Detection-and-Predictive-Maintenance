# Railway Track Fault Detection & Predictive Maintenance System

AI-powered system for detecting railway track faults and predicting maintenance needs using multi-sensor fusion.

## Project Structure

```
railway_fault_detection/
├── data/                          # Input data (read-only copies)
│   ├── RT_PLC_RSFPD.csv           # Main PLC sensor dataset (5000 × 32)
│   └── *.xlsx                     # Vibration sensor XLSX files
├── src/                           # Source modules
│   ├── preprocess.py              # Data loading, cleaning, feature engineering
│   ├── anomaly_model.py           # Isolation Forest (IF=200, contamination=0.08)
│   ├── rul_model.py               # Baseline RUL model (Gradient Boosting)
│   ├── classifier.py              # Baseline fault classifier (Random Forest)
│   ├── lstm_rul_model.py          # PRIMARY BiLSTM RUL predictor (Keras)
│   ├── cnn_lstm_model.py          # PRIMARY CNN-LSTM fault classifier (Keras)
│   ├── alerts.py                  # Alert engine (uses Primary DL predictions)
│   ├── xai_explainer.py           # SHAP-based explainability module
│   └── pipeline.py                # End-to-end orchestrator (Primary DL prioritized)
├── models/                        # Trained models (.pkl)
├── outputs/                       # Generated artifacts
│   ├── railway_dashboard.html     # Interactive HTML dashboard (5 tabs)
│   ├── dashboard_data.json        # Dashboard data payload
│   └── alert_log.csv              # Maintenance alert log
├── tests/                         # Pytest test suite
├── app.py                         # Streamlit dashboard (secondary)
├── requirements.txt               # Python dependencies
└── README.md
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Full Pipeline
```bash
python src/pipeline.py
```
This trains the primary models and generates dashboard data:
- **Phase 1-2:** Data preprocessing and feature engineering (5000 rows → 38 features)
- **Phase 3:** Isolation Forest anomaly detection (IF_Flag, IF_Score) — 8% detection rate
- **Phase 4:** **BiLSTM RUL prediction (PRIMARY)** ⭐ — Advanced sequence-to-one regression
- **Phase 5:** **CNN-LSTM fault classifier (PRIMARY)** ⭐ — Hybrid spatiotemporal architecture
- **Phase 6-7:** (Optional) Gradient Boosting & Random Forest baseline comparison
- **Phase 8:** Dashboard generation (HTML/JSON/CSV) and Alert Engine execution


### 3. View Dashboard
Open `outputs/railway_dashboard.html` in any browser.

### 4. Run Tests
```bash
python -m pytest tests/ -v --tb=short
```

### 5. Streamlit Dashboard (Live AI Monitoring)
```bash
streamlit run app.py
```
The Streamlit dashboard now features **Live AI Inference**:
- **Real-time Prediction**: Deep learning models (BiLSTM/CNN-LSTM) compute RUL and Fault types live as data streams.
- **Sequence Buffering**: The system maintains a 30-step sliding window buffer for full spatiotemporal analysis.
- **Dynamic Alerts**: Inference results instantly trigger maintenance recommendations in the "Live Monitoring" tab.


## Models

### Model Overview

| Model | Algorithm | Purpose | Training Time | Memory | Status |
|-------|-----------|---------|----------------|--------|--------|
| **BiLSTM** ⭐ | Bidirectional LSTM | Primary RUL prediction | ~175 sec | ~50MB | ✅ Production |
| **CNN-LSTM** ⭐ | Hybrid CNN-LSTM | Primary fault classification | ~24 sec | ~80MB | ✅ Production |
| Isolation Forest | Unsupervised Ensemble | Real-time anomaly detection | <1 sec | <20MB | ✅ Production |
| Gradient Boosting | Supervised Ensemble | RUL baseline (comparison) | 5-10 sec | <30MB | 📊 Comparison |
| Random Forest | Supervised Ensemble | Fault baseline (comparison) | 5-8 sec | <25MB | 📊 Comparison |


### Model Hyperparameters

#### Isolation Forest (Anomaly Detection)
```python
n_estimators: 200           # Number of isolation trees
contamination: 0.08         # Expected anomaly fraction (8%)
random_state: 42            # Reproducibility seed
n_jobs: -1                  # Parallel processing on all cores
```
**Features:** 8 (Vibration, Temperature, Track Resistance, CPU Load, Anomaly Score, Failure Prob, Humidity, Component Age)

#### Gradient Boosting Regressor (RUL Prediction)
```python
n_estimators: 300           # Boosting iterations
learning_rate: 0.05         # Gradient step size (reduced from 0.1)
max_depth: 5                # Tree depth to prevent overfitting
random_state: 42            # Reproducibility seed
```
**Features:** 19 (Core sensors + derived features + IF_Flag)
**Output:** RUL in days (typically 1-500)

#### Random Forest Classifier (Fault Detection) ⭐ Production
```python
n_estimators: 300           # Number of decision trees
max_depth: 12               # Tree depth (reduced from 15)
class_weight: balanced      # Handle class imbalance
random_state: 42            # Reproducibility seed
n_jobs: -1                  # Parallel processing on all cores
```
**Features:** 20 (All RUL features + HMI_Alert_Code_enc)
**Classes:** 10 (C1-C10 with specific maintenance actions)
**Performance:** Accuracy 99.6% (April 2026 run)

#### BiLSTM RUL Prediction (Advanced Research Model)
```python
seq_len: 30                 # Rolling window for sequences
epochs: 60                  # Maximum training epochs (EarlyStopping)
batch_size: 64              # Training batch size
layers: 2x BiLSTM(128→64)   # Bidirectional layers with dropout(0.3→0.2)
output: Dense(64) → Dense(1)  # 64-dim hidden → RUL scalar
```
**Architecture:** BiLSTM(128) → Dropout(0.3) → BiLSTM(64) → Dropout(0.2) → Dense(64) → Dense(1)
**Features:** 19 RUL features (sequence format)
**Frame Size:** 30-step rolling window
**Performance:** MAE: 37.39 days, RMSE: 45.65 days, R²: -0.1305 (30/60 epochs)
**Note:** Research model; needs hyperparameter tuning. Trained in 174.8s.

#### CNN-LSTM Classifier (Advanced Research Model)
```python
seq_len: 30                 # Rolling window for sequences
epochs: 50                  # Maximum training epochs (EarlyStopping)
batch_size: 64              # Training batch size
layers: Conv1D(64,32) + LSTM(64)  # Spatial + temporal extraction
output: Dense(64) → Dense(10, softmax)  # 10-class softmax
```
**Architecture:** Conv1D(64) → MaxPool → Conv1D(32) → MaxPool → LSTM(64) → Dropout(0.3) → Dense(64) → Dense(10)
**Features:** 20 classifier features (sequence format)
**Frame Size:** 30-step rolling window
**Performance:** Accuracy: 21.6% (14/50 epochs, early stop)
**Note:** Research model with novel CNN-LSTM hybrid; underfitting detected. Trained in 23.6s.

### Actual Performance (April 2026 Pipeline Run)

#### Baseline Models (Production)
| Metric | Isolation Forest | Gradient Boosting | Random Forest |
|--------|------------------|-------------------|---------------|
| **Rate/Accuracy** | 8.0% anomaly rate | MAE: 10.91 days | Accuracy: 99.6% |
| **Precision/RMSE** | Precision: 0.820 | RMSE: 13.71 days | Conf Matrix: 10×10 |
| **Goodness** | R²: - | R²: 0.8978 ⭐ | Top-1 Feature: 50.9% |
| **Training Time** | <1 sec | 8-10 sec | 5-8 sec |
| **Inference Time** | <5ms per record | <1ms per record | <3ms per record |

#### Advanced Models (Research/Experimental)
| Metric | BiLSTM RUL | CNN-LSTM Classifier |
|--------|-----------|---------------------|
| **Accuracy/MAE** | MAE: 37.39 days | Accuracy: 21.6% |
| **Quality** | RMSE: 45.65 days | Training epochs: 14/50 (early stop) |
| **Goodness** | R²: -0.1305 | Note: Under-fitting, needs tuning |
| **Training Time** | 174.8 sec (30/60 epochs) | 23.6 sec (14/50 epochs) |
| **Memory** | ~50MB | ~80MB |
| **Status** | 🧪 Research stage | 🧪 Research stage |

> **⭐ Production Primary:** BiLSTM (RUL) + CNN-LSTM (Classifier).
> **📊 Comparison Baselines:** Gradient Boosting + Random Forest (Training toggled via `TRAIN_BASELINES` in `pipeline.py`).


## Alert Levels

| Level | Trigger Conditions |
|-------|-------------------|
| CRITICAL | `failure_prob > 0.75` OR `RUL < 15 days` OR `vibration > 0.70` OR `temp > 55°C` |
| WARNING | `failure_prob > 0.45` OR `RUL < 60 days` OR `vibration > 0.55` OR `temp > 45°C` |
| HEALTHY | None of the above |

## Configuration Constants

| Module | Constant | Value | Description |
|--------|----------|-------|-------------|
| **anomaly_model.py** | CONTAMINATION | 0.08 | Expected anomaly fraction |
| **anomaly_model.py** | N_ESTIMATORS | 200 | Isolation Forest tree count |
| **rul_model.py** | n_estimators | 300 | Gradient Boosting iterations |
| **rul_model.py** | learning_rate | 0.05 | GB gradient step size |
| **rul_model.py** | max_depth | 5 | GB tree depth |
| **classifier.py** | n_estimators | 300 | RF tree count |
| **classifier.py** | max_depth | 12 | RF tree depth |
| **classifier.py** | class_weight | 'balanced' | Handle class imbalance |
| **All Models** | RANDOM_STATE | 42 | Reproducibility seed |
| **pipeline.py** | TEST_SIZE | 0.20 | Train/test split ratio |
| **classifier.py** | N_FAULT_CLASSES | 10 | Fault types C1-C10 |
| **preprocess.py** | CONTAMINATION | 0.08 | IF anomaly threshold |

### Model Configuration Files

## Explainability (XAI)

The system includes SHAP-based explainability for fault predictions:

**Module:** `src/xai_explainer.py`
- **TreeExplainer:** SHAP values computed from Random Forest classifier
- **Visualizations:** 
  - Summary plot (feature importance by SHAP contribution)
  - Bar plot (mean |SHAP| per feature)
  - Waterfall plot (feature contributions for single predictions)
- **Status:** ✅ Integrated into pipeline
- **Usage:** Convert SHAP values to Plotly figures for dashboard integration

```python
from src.xai_explainer import compute_shap_values, build_shap_summary_fig
shap_dict = compute_shap_values(rf_model, X_test_scaled, feature_names)
fig = build_shap_summary_fig(shap_dict)
fig.show()
```

## Project Organization

### Directory Structure
```
.
├── src/                          # Core ML modules
│   ├── preprocess.py             # Data pipeline
│   ├── anomaly_model.py          # Isolation Forest
│   ├── rul_model.py              # Gradient Boosting (Baseline)
│   ├── classifier.py             # Random Forest (Baseline)
│   ├── lstm_rul_model.py         # BiLSTM (Primary) ⭐
│   ├── cnn_lstm_model.py         # CNN-LSTM (Primary) ⭐
│   ├── alerts.py                 # Alert engine
│   ├── xai_explainer.py          # SHAP explainability
│   └── pipeline.py               # Orchestrator (Deep Learning focus)
├── models/                       # Trained artifacts
│   ├── isolation_forest.pkl
│   ├── rul_model.pkl
│   ├── clf_model.pkl
│   ├── lstm_rul_model.keras
│   └── cnn_lstm_clf.keras
├── outputs/                      # Generated outputs
│   ├── railway_dashboard.html    # Main dashboard
│   ├── dashboard_data.json       # Data payload
│   └── alert_log.csv             # Alert records
├── tests/                        # Pytest suite (42 tests)
├── .archive/                     # Archived experimental scripts
├── app.py                        # Streamlit dashboard (optional)
└── requirements.txt              # Dependencies
```

For comprehensive model parameter details, see:
- **[MODEL_CONFIG_REFERENCE.md](MODEL_CONFIG_REFERENCE.md)** - Complete hyperparameter specifications
- **[RESEARCH_PAPER.md](RESEARCH_PAPER.md)** - Technical methodology and results

## Fault Classes (C1-C10)

The Random Forest classifier can diagnose 10 fault types with specific maintenance actions:

| Code | Fault Type | Recommended Action |
|------|------------|--------------------|
| C1 | Rail Crack / Fracture | Immediate rail replacement; restrict speed to 20km/h |
| C2 | Loose Fastener / Joint Failure | Tighten/replace fasteners; inspect adjacent joints |
| C3 | Short Circuit in Track Circuit | Check track resistance; inspect bonding wires |
| C4 | Ballast Degradation | Re-tamp ballast; schedule geometry correction |
| C5 | Thermal Buckling Risk | Apply de-stressing procedure; monitor rail temp |
| C6 | Gauge Widening | Re-gauge track; inspect sleeper anchors |
| C7 | Wheel Impact Damage (Flat Spot) | Profile grinding; inspect affected rail section |
| C8 | Signalling Relay Malfunction | Replace relay module; test signal interlocking |
| C9 | PLC / Controller Overload | Restart PLC; review scan cycle timing |
| C10 | Corrosion / Environmental Damage | Apply anti-corrosion treatment; replace corroded section |

## Development History & Archived Scripts

Early development scripts have been archived in `.archive/`:
- `data_preprocessing.py` - Initial exploratory analysis
- `fault_detection_models.py` - Earlier model implementations
- `update_html.py` - Manual dashboard update script

These scripts are superseded by the integrated pipeline in `src/pipeline.py`.

## Test Suite

**Status:** ✅ All 42 tests passing

Run tests with:
```bash
pytest tests/ -v --tb=short
```

Test coverage:
- ✅ 6 alert generation tests
- ✅ 12 model training & prediction tests
- ✅ 6 pipeline output artifact tests
- ✅ 18 preprocessing & data handling tests

## Known Limitations & Research Notes

- **Advanced Models**: BiLSTM and CNN-LSTM are research-stage models with performance metrics below baseline. These may improve with:
  - Hyperparameter tuning (learning rate, dropout, batch size)
  - More training epochs (currently early-stopped)
  - Better feature engineering or class balancing
  - Larger dataset or higher data quality

- **TensorFlow Optional**: LSTM and CNN-LSTM models require TensorFlow. If not installed, pipeline gracefully skips Phase 4b & 5b and reports baseline only.

- **Synthetic Data Note**: Model shows expected high performance (99.6% RF accuracy); real-world accuracy will depend on actual fault-labeled historical data.

- **Temporal Data Handling**: Models use time-based split (no shuffle) to prevent temporal leakage; cross-validation not applied.
