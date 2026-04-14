# Updates Summary - April 14, 2026

## 🚀 Model Architecture Migration
The project has officially migrated its primary inference engine to advanced deep learning architectures.

### Key Changes
1. **Primary Models Redesignated**:
   - **BiLSTM** is now the primary model for Remaining Useful Life (RUL) prediction.
   - **CNN-LSTM** is now the primary model for Fault Classification.
   - **Isolation Forest** remains the primary model for unsupervised Anomaly Detection.

2. **Baseline Deprecation**:
   - Gradient Boosting (RUL) and Random Forest (Fault) have been moved to **Baseline Comparison** status.
   - A new `TRAIN_BASELINES` flag in `src/pipeline.py` allows skipping baseline training to focus on deep learning performance.

3. **Live AI Inference Engine**:
   - The Streamlit dashboard (`app.py`) now performs **Live Inference**.
   - **30-Step Temporal Buffering**: The engine buffers incoming telemetry to provide the spatiotemporal context required by LSTM/CNN architectures.
   - All live alerts and maintenance recommendations are now driven by the runtime model predictions rather than static dataset values.

4. **UI/UX Enhancements**:
   - Updated "AI Models" tab to show direct comparisons between Primary and Baseline models.
   - Updated "XAI Explainability" tab to clarify it provides comparative insights via the Random Forest baseline.
   - Improved "Live Monitoring" status indicators (Buffering vs. Streaming).

## 🛠 Progress Log
- `[x]` Update `pipeline.py` logic to prioritize DL models.
- `[x]` Implement Live Sequence Buffer in `app.py`.
- `[x]` Integrate `.keras` model loading in Streamlit.
- `[x]` Synchronize all documentation (`README.md`, `QUICK_REFERENCE.md`, `MODEL_CONFIG_REFERENCE.md`).
