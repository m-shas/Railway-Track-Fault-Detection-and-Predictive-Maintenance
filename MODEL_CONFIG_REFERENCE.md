# Model Configuration Reference

**Last Updated:** April 9, 2026

Complete hyperparameter and configuration specification for all models in the Railway Track Fault Detection system.

## Production Models ⭐

### 1. Isolation Forest (Anomaly Detection)

**Module:** `src/anomaly_model.py` (Lines 14-19, 32-45)

**Purpose:** Detect anomalous sensor readings in real-time

**Configuration:**
```python
CONTAMINATION = 0.08              # Expected anomaly fraction (8%)
N_ESTIMATORS = 200                # Number of isolation trees
RANDOM_STATE = 42                 # Reproducibility seed
```

**Input Features (8 total):**
```python
ANOMALY_FEATURES = [
    "Vibration_m_s2",             # Sensor vibration amplitude
    "Temperature_C",              # Track/component temperature
    "Track_Resistance_Ohm",       # Electrical resistance
    "PLC_CPU_Load_percent",       # PLC processor load
    "Edge_Anomaly_Score",         # Edge device anomaly flag
    "Predicted_Failure_Prob",     # ML-predicted failure probability
    "Humidity_percent",           # Environmental humidity
    "Component_Age_days",         # Component service life in days
]
```

**Training Process:**
- Scaling: StandardScaler (fit on full dataset)
- Split: No split (trains on all data for unsupervised learning)
- Parallel: `n_jobs=-1` (all CPU cores)

**Hyperparameters in Code:**
```python
model = IsolationForest(
    n_estimators=200,
    contamination=0.08,
    random_state=42,
    n_jobs=-1,
)
```

**Performance (April 2026):**
- Anomalies detected: 400 / 5000 (8.0%)
- Precision: 0.820 (false positive rate controlled)
- Inference: <5ms per record

---

### 2. Gradient Boosting Regressor (RUL Prediction) ⭐

**Module:** `src/rul_model.py` (Lines 25-45, 63-105)

**Purpose:** Predict Remaining Useful Life (days until maintenance needed)

**Configuration:**
```python
RANDOM_STATE = 42
TEST_SIZE = 0.20                  # 80-20 train-test split
```

**Gradient Boosting Parameters:**
```python
n_estimators: 300                 # Boosting iterations
learning_rate: 0.05               # Gradient step size (eta)
max_depth: 5                       # Tree depth (prevent overfitting)
random_state: 42                  # Reproducibility
```

**Input Features (19 total):**
```python
RUL_FEATURES = [
    "Vibration_m_s2", "Temperature_C", "Humidity_percent",
    "Track_Resistance_Ohm", "PLC_CPU_Load_percent",
    "Edge_Anomaly_Score", "Predicted_Failure_Prob",
    "Cloud_Health_Index", "Component_Age_days",
    "Voltage_V", "Current_A", "Timer_TON_ms", "Timer_TCH_ms",
    "Signal_Transition_Delay_ms", "Block_Clearance_Time_s",
    "Train_Headway_s", "Ambient_Temp_C", "Dust_Index_ppm",
    "IF_Flag",  # From Phase 3 anomaly detection
]
```

**Output:**
```python
RUL_TARGET = "RUL_Predicted_days"  # Continuous value (1-500 days)
```

**Training Process:**
- Scaling: MinMaxScaler (fit on training set)
- Split: Time-based split (shuffle=False, no temporal leakage)
- Train indices: 0 to 4000 (80%)
- Test indices: 4000 to 5000 (20%)

**Hyperparameters in Code:**
```python
model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    random_state=42,
)
```

**Performance (April 2026):**
- MAE: 10.91 days
- RMSE: 13.71 days
- R²: 0.8978 (89.78% variance explained)
- Training time: 8-10 seconds
- Inference: <1ms per record

**Top-5 Feature Importances:**
1. Component_Age_days: 50.90%
2. Predicted_Failure_Prob: 37.05%
3. Vibration_m_s2: 5.31%
4. Track_Resistance_Ohm: 1.74%
5. Temperature_C: 1.20%

---

### 3. Random Forest Classifier (Fault Detection) ⭐

**Module:** `src/classifier.py` (Lines 20-45, 70-110)

**Purpose:** Classify equipment faults into 10 maintenance categories (C1-C10)

**Configuration:**
```python
RANDOM_STATE = 42
TEST_SIZE = 0.20                  # 80-20 train-test split
N_FAULT_CLASSES = 10              # C1 through C10
```

**Random Forest Parameters:**
```python
n_estimators: 300                 # Number of decision trees
max_depth: 12                      # Tree depth
class_weight: 'balanced'           # Handle class imbalance
random_state: 42                   # Reproducibility
n_jobs: -1                         # Parallel on all cores
```

**Input Features (20 total):**
```python
CLF_FEATURES = [
    "Vibration_m_s2", "Temperature_C", "Humidity_percent",
    "Track_Resistance_Ohm", "PLC_CPU_Load_percent",
    "Edge_Anomaly_Score", "Predicted_Failure_Prob",
    "Cloud_Health_Index", "Component_Age_days",
    "Voltage_V", "Current_A", "Timer_TON_ms", "Timer_TCH_ms",
    "Signal_Transition_Delay_ms", "Block_Clearance_Time_s",
    "Train_Headway_s", "Ambient_Temp_C", "Dust_Index_ppm",
    "IF_Flag",  # From Phase 3
    "HMI_Alert_Code_enc",  # Encoded categorical
]
```

**Output:**
```python
CLF_TARGET = "Failure_Type_enc"    # Encoded C1-C10 (0-9)
```

**Training Process:**
- Scaling: StandardScaler (fit on training set)
- Split: Stratified split (preserve class distribution)
- Train indices: 4000 samples
- Test indices: 1000 samples

**Hyperparameters in Code:**
```python
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
)
```

**Performance (April 2026):**
- Accuracy: 99.6%
- Confusion matrix: 10×10 (all classes)
- Training time: 5-8 seconds
- Inference: <3ms per record

**Fault Classes (C1-C10):**
| Code | Type | Maintenance Action |
|------|------|-------------------|
| C1 | Rail Crack / Fracture | Immediate replacement; speed ≤20km/h |
| C2 | Loose Fastener / Joint | Tighten/replace; inspect adjacent |
| C3 | Short Circuit Track | Check resistance; inspect bonding |
| C4 | Ballast Degradation | Re-tamp; schedule geometry fix |
| C5 | Thermal Buckling | Apply de-stressing; monitor temp |
| C6 | Gauge Widening | Re-gauge; inspect sleeper anchors |
| C7 | Wheel Impact Damage | Profile grinding; inspect rail |
| C8 | Signalling Relay Fault | Replace module; test interlocking |
| C9 | PLC / CPU Overload | Restart PLC; review scan timing |
| C10 | Corrosion / Rust | Anti-corrosion treatment; replace |

---

## Advanced/Research Models 🧪

### 4. BiLSTM RUL Prediction (Advanced)

**Module:** `src/lstm_rul_model.py` (Lines 11-30, 56-150)

**Purpose:** Advanced sequence-to-sequence RUL prediction using bidirectional LSTM

**Architecture:**
```
Input (seq_len=30, n_features=19)
    ↓
BiLSTM(128 units) → Dropout(0.3)
    ↓
BiLSTM(64 units) → Dropout(0.2)
    ↓
Dense(64, relu) → BatchNormalization
    ↓
Output Dense(1)  [RUL scalar]
```

**Configuration:**
```python
SEQ_LEN = 30                       # Rolling window size
EPOCHS = 60                        # Max training epochs
BATCH_SIZE = 64                    # Training batch size
RANDOM_STATE = 42
TEST_SIZE = 0.20
```

**Input Features (19, same as GB):**
```python
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
```

**Sequence Construction:**
- Sliding window of 30 timesteps
- Non-overlapping train/test split (time-based)
- Padding: None (sequences exact length)

**Compilation:**
```python
optimizer: Adam(learning_rate=0.001)
loss: mean_absolute_error
metrics: [mae, mse]
```

**Training Callbacks:**
- EarlyStopping: patience=5, monitor='val_loss'
- ReduceLROnPlateau: factor=0.5, patience=3

**Performance (April 2026):**
- MAE: 37.39 days
- RMSE: 45.65 days
- R²: -0.1305 (negative = worse than baseline)
- Epochs trained: 30/60 (early stopped)
- Training time: 174.8 seconds
- Model size: ~50MB
- Inference: ~50ms per sequence
- **Status:** 🧪 Underfitting - needs hyperparameter tuning

**Recommendations for Improvement:**
- Increase training epochs to 100+ with patience=10
- Reduce learning rate (0.0005)
- Add L2 regularization (0.001-0.01)
- Try different sequence lengths (20, 40, 50)
- Implement learning rate scheduling
- Augment training data or increase batch size

---

### 5. CNN-LSTM Hybrid Fault Classifier (Advanced)

**Module:** `src/cnn_lstm_model.py` (Lines 13-32, 56-150)

**Purpose:** Hybrid CNN-LSTM for spatial-temporal fault pattern recognition

**Architecture:**
```
Input (seq_len=30, n_features=20)
    ↓
Conv1D(64, kernel=3, relu) → MaxPooling1D(2)
    ↓
Conv1D(32, kernel=3, relu) → MaxPooling1D(2)
    ↓
LSTM(64) → Dropout(0.3)
    ↓
Dense(64, relu) → BatchNormalization
    ↓
Output Dense(10, softmax)  [10-class softmax]
```

**Configuration:**
```python
SEQ_LEN = 30                       # Rolling window size
EPOCHS = 50                        # Max training epochs
BATCH_SIZE = 64                    # Training batch size
RANDOM_STATE = 42
TEST_SIZE = 0.20
```

**Input Features (20 total):**
```python
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
CLF_TARGET = "Failure_Type"  # C1-C10 (10 classes)
```

**Sequence Construction:**
- Sliding window of 30 timesteps
- Stratified train/test split (preserve class distribution)
- Encoding: LabelEncoder for failure types

**Compilation:**
```python
optimizer: Adam(learning_rate=0.001)
loss: categorical_crossentropy
metrics: [accuracy]
```

**Training Callbacks:**
- EarlyStopping: patience=5, monitor='val_loss'
- ReduceLROnPlateau: factor=0.5, patience=3

**Performance (April 2026):**
- Accuracy: 21.6% (vs Random Forest: 99.6%)
- Epochs trained: 14/50 (early stopped)
- Training time: 23.6 seconds
- Model size: ~80MB
- Inference: ~100ms per sequence
- **Status:** 🧪 Underfitting - needs significant tuning

**Recommendations for Improvement:**
- Increase kernel sizes (5, 7, 9 vs 3)
- Add more conv layers (3-4 vs 2)
- Class weight balancing in loss function
- Ensemble with Random Forest predictions
- Try different sequence lengths
- Increase training data or augment
- Implement dropout in conv layers
- Add batch normalization after conv layers

---

## Global Configurations (All Models)

**`src/preprocess.py` - Preprocessing Config (Lines 30-45):**
```python
CRITICAL_PROB = 0.75              # Failure probability threshold
CRITICAL_RUL = 15                 # RUL days warning level
WARNING_PROB = 0.45               # Warning probability threshold
WARNING_RUL = 60                  # RUL days warning level

CATEGORICAL_COLS = [
    "Failure_Type", "Occupancy_State",
    "HMI_Alert_Code", "Maintenance_Action"
]
```

**`src/alerts.py` - Alert Engine Config (Lines 13-30):**
```python
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
```

**`src/pipeline.py` - Pipeline Config (Lines 72-73):**
```python
ALERT_DOWNSAMPLE = 3              # Sample every 3rd row for dashboard JSON
```

---

## XAI (Explainability) Configuration

**Module:** `src/xai_explainer.py` (Lines 13-60)

**Purpose:** SHAP-based feature importance for Random Forest predictions

**Methods:**
- **TreeExplainer:** Optimized for tree-based models
- **Max Samples:** 200 (for computation speed)
- **API Support:** Both old (<0.40) and new (>=0.42) SHAP versions

**Outputs:**
| Visualization | Purpose | Input |
|---------------|---------|-------|
| Summary Plot | Feature importance beeswarm | SHAP matrix |
| Bar Plot | Mean \|SHAP\| per feature | SHAP values |
| Waterfall | Single-prediction breakdown | SHAP[i] values |

**Configuration in Code:**
```python
max_samples: int = 200             # Rows to compute SHAP for
class_names: list = []             # Optional class labels
```

---

## Model Files & Serialization

**Location:** `models/` directory

**File Format & Names:**
| Model | Extension | Format | Size |
|-------|-----------|--------|------|
| Isolation Forest | `.pkl` | Pickle (joblib) | <20MB |
| Gradient Boosting | `.pkl` | Pickle (joblib) | <30MB |
| Random Forest | `.pkl` | Pickle (joblib) | <25MB |
| BiLSTM | `.keras` | TensorFlow/Keras | ~50MB |
| BiLSTM Meta | `_meta.pkl` | Pickle (joblib) | <1MB |
| CNN-LSTM | `.keras` | TensorFlow/Keras | ~80MB |

**Loading Models:**
```python
import pickle
import joblib
import tensorflow as tf

# Load sklearn models
with open('models/rul_model.pkl', 'rb') as f:
    model = joblib.load(f)

# Load TensorFlow models
lstm_model = tf.keras.models.load_model('models/lstm_rul_model.keras')
```

---

## Summary: Which Model to Use?

**For Production (Use These):**
- **RUL Prediction:** Gradient Boosting (10.91 days MAE) ✅
- **Fault Classification:** Random Forest (99.6% accuracy) ✅
- **Anomaly Detection:** Isolation Forest (82% precision) ✅

**For Research/Benchmarking:**
- **Advanced RUL:** BiLSTM (needs parameter tuning)
- **Advanced Classification:** CNN-LSTM (needs parameter tuning)
- **Explainability:** SHAP with Random Forest ✅

**For Interpretability:**
- Use `src/xai_explainer.py` with Random Forest predictions
- Generate SHAP summary/waterfall plots for decision explanations

---

**Last Verified:** April 9, 2026  
**All Tests Passing:** Yes (42/42)  
**Pipeline Status:** ✅ Fully Functional
