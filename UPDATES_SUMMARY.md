# Railway Track Fault Detection - Updates Summary

**Last Updated:** April 9, 2026

## Recent Changes (Since LSTM/CNN/XAI Integration)

### ✨ New Advanced Models Added

#### 1. BiLSTM RUL Prediction Model
- **Module:** `src/lstm_rul_model.py`
- **Type:** Bidirectional LSTM sequence model
- **Purpose:** Alternative to Gradient Boosting for RUL prediction
- **Architecture:**
  - BiLSTM layer (128 units) → Dropout(0.3)
  - BiLSTM layer (64 units) → Dropout(0.2)
  - Dense(64) → ReLU → Output(1)
  - Sequence length: 30 timesteps
- **Training:**
  - Epochs: 30/60 (early stopped)
  - Training time: 174.8 seconds
  - Batch size: 64
- **Performance (April 2026):**
  - MAE: 37.39 days (vs GB: 10.91 days)
  - RMSE: 45.65 days
  - R²: -0.1305 (underfitting detected)
- **Status:** 🧪 Research stage - needs hyperparameter tuning

#### 2. CNN-LSTM Hybrid Fault Classifier
- **Module:** `src/cnn_lstm_model.py`
- **Type:** Hybrid CNN-LSTM sequence model
- **Purpose:** Extract spatial & temporal patterns for fault classification
- **Architecture:**
  - Conv1D(64, kernel=3) → MaxPooling1D(2)
  - Conv1D(32, kernel=3) → MaxPooling1D(2)
  - LSTM(64) → Dropout(0.3)
  - Dense(64, relu) → Dense(10, softmax)
  - Sequence length: 30 timesteps
- **Training:**
  - Epochs: 14/50 (early stopped)
  - Training time: 23.6 seconds
  - Batch size: 64
- **Performance (April 2026):**
  - Accuracy: 21.6% (vs RF: 99.6%)
  - Trained epochs: 14/50
- **Status:** 🧪 Research stage - underfitting, significant room for improvement

#### 3. SHAP Explainability Module
- **Module:** `src/xai_explainer.py`
- **Type:** TreeExplainer-based feature importance
- **Purpose:** Interpretability for fault predictions
- **Features:**
  - Compute SHAP values from Random Forest
  - Generate summary plots (beeswarm-style)
  - Generate bar plots (mean |SHAP| per feature)
  - Generate waterfall plots (single-prediction contributions)
  - Support both old (<0.40) and new (>=0.42) SHAP APIs
- **Status:** ✅ Integrated - Ready for dashboard use

### 📊 Pipeline Phases (Updated)

| Phase | Name | Status | Output |
|-------|------|--------|--------|
| 1-2 | Preprocessing | ✅ Production | 5000 rows × 38 features |
| 3 | Anomaly Detection | ✅ Production | 400 anomalies (8%), precision=0.82 |
| 4a | RUL (GB) | ✅ Production | MAE=10.91 days, R²=0.8978 |
| 4b | RUL (BiLSTM) | 🧪 Research | MAE=37.39 days, R²=-0.1305 |
| 5a | Classifier (RF) | ✅ Production | Accuracy=99.6% |
| 5b | Classifier (CNN-LSTM) | 🧪 Research | Accuracy=21.6% |
| 6 | Alert Generation | ✅ Production | 4,183 alerts (1,482 CRITICAL) |
| 7 | Dashboard | ✅ Production | JSON (221 KB) + HTML (149 KB) |

### 📁 File Changes

#### New Files Created
- ✅ `src/lstm_rul_model.py` - BiLSTM RUL model
- ✅ `src/cnn_lstm_model.py` - CNN-LSTM classifier
- ✅ `src/xai_explainer.py` - SHAP explainability

#### Updated Documentation
- 📝 `README.md` - Added model performance, XAI section, architecture updates
- 📝 `UPDATES_SUMMARY.md` - This file (new)

#### Archived (Moved to `.archive/`)
- 📦 `data_preprocessing.py` - Exploratory script
- 📦 `fault_detection_models.py` - Older implementation
- 📦 `update_html.py` - Manual dashboard script
- 🗑️ `f fault_detection_models.py` - Deleted (corrupted empty file)

### ✅ Verification Status

**Pipeline Execution:** ✅ PASSED
- All 8 phases completed successfully
- All models trained and saved
- Dashboard artifacts generated

**Test Suite:** ✅ 42/42 PASSED
- 6 alert tests
- 12 model tests
- 6 pipeline tests
- 18 preprocessing tests

**Code Quality:** ✅ NO ERRORS
- 0 syntax errors
- 0 import errors
- All dependencies satisfied

### 📈 Performance Summary

#### Production Models (Recommended)
| Model | Purpose | Accuracy | Speed | Status |
|-------|---------|----------|-------|--------|
| Isolation Forest | Anomaly Detection | 82% Precision | <5ms | ✅ Ready |
| Gradient Boosting | RUL Prediction | MAE: 10.91 days | <1ms | ✅ Ready |
| Random Forest | Fault Classification | 99.6% Accuracy | <3ms | ✅ Ready |

#### Research Models (Experimental)
| Model | Purpose | Accuracy | Speed | Status |
|-------|---------|----------|-------|--------|
| BiLSTM | RUL Prediction | MAE: 37.39 days | ~50ms | 🧪 Tuning |
| CNN-LSTM | Fault Classification | 21.6% Accuracy | ~100ms | 🧪 Tuning |

### 🎯 Next Steps for Advanced Models

#### BiLSTM Improvements
- [ ] Try reduced learning rate (0.001 → 0.0005)
- [ ] Increase epochs to 100+ with patience=10
- [ ] Add regularization (L2, more dropout)
- [ ] Try different seq_len (20, 40 vs 30)
- [ ] Implement custom loss weights for class imbalance

#### CNN-LSTM Improvements
- [ ] Increase kernel sizes (5, 7 vs 3)
- [ ] Try deeper convolution (3-4 layers vs 2)
- [ ] Reduce pooling stride (overlap more)
- [ ] Add batch normalization between layers
- [ ] Class weight balancing in loss function
- [ ] Try different seq_len optimization

#### General
- [ ] Hyperparameter grid search
- [ ] Cross-validation with time-series awareness
- [ ] Learning rate scheduling
- [ ] Ensemble with baseline models

### 📊 Model Comparison Table

```
Metric                    | Isolation Forest | Gradient Boosting* | Random Forest* | BiLSTM | CNN-LSTM
--------------------------|------------------|-------------------|----------------|--------|----------
Training Time             | <1 sec            | 8-10 sec           | 5-8 sec        | 175 s  | 24 s
Inference Time (per rec)  | <5ms              | <1ms               | <3ms           | ~50ms  | ~100ms
Primary Metric            | Precision: 0.82   | MAE: 10.91 days    | Acc: 99.6%     | MAE: 37.39 days | Acc: 21.6%
Model Size                | <20MB             | <30MB              | <25MB          | ~50MB  | ~80MB
Production Ready          | ✅ Yes            | ✅ Yes             | ✅ Yes         | 🧪 No  | 🧪 No
* = Primary models for deployment
```

### 🔧 Configuration Files

All hyperparameters documented in source modules:
- `src/anomaly_model.py` - Lines 14-19
- `src/rul_model.py` - Lines 25-45
- `src/classifier.py` - Lines 20-45
- `src/lstm_rul_model.py` - Lines 11-30
- `src/cnn_lstm_model.py` - Lines 13-32

For production deployment, see `src/pipeline.py` Phase 7-8 for dashboard integration.

### 📝 Notes

- BiLSTM & CNN-LSTM models trained with TensorFlow 2.12.0+
- SHAP values computed with TreeExplainer from sklearn Random Forest
- Pipeline auto-detects TensorFlow and includes/excludes advanced models accordingly
- All models save to `models/` directory with consistent naming:
  - `isolation_forest.pkl`, `rul_model.pkl`, `clf_model.pkl` (sklearn)
  - `lstm_rul_model.keras`, `cnn_lstm_clf.keras` (TensorFlow)

### 📚 References

- **Main Documentation:** `README.md`
- **Research Details:** `RESEARCH_PAPER.md`
- **Model Reference:** `MODEL_CONFIG_REFERENCE.md`
- **Testing:** `tests/` directory (42 tests)

---

**Status:** ✅ Project ready for production (Phase 1-7). Advanced models available for research (Phase 4b, 5b).
