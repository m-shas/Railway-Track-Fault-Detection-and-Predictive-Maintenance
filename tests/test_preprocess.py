"""Tests for src/preprocess.py — Phase 2 validation."""

import sys
import os
import numpy as np
import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocess import (
    load_csv, load_vibration_xlsx, encode_categoricals,
    compute_alert_level, compute_health_score, make_sequences,
    preprocess_pipeline,
)

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
CSV_PATH = os.path.join(BASE_DIR, 'data', 'RT_PLC_RSFPD.csv')
XLSX_PATH = os.path.join(BASE_DIR, 'data', 'vibration_analysis_graph.xlsx')


class TestLoadCSV:
    def test_shape(self):
        df = load_csv(CSV_PATH)
        assert df.shape[0] == 5000, f"Expected 5000 rows, got {df.shape[0]}"
        assert df.shape[1] == 32, f"Expected 32 columns, got {df.shape[1]}"

    def test_timestamp_dtype(self):
        df = load_csv(CSV_PATH)
        assert pd.api.types.is_datetime64_any_dtype(df['Timestamp'])

    def test_sorted(self):
        df = load_csv(CSV_PATH)
        assert df['Timestamp'].is_monotonic_increasing

    def test_no_null_timestamp(self):
        df = load_csv(CSV_PATH)
        assert df['Timestamp'].isnull().sum() == 0


class TestLoadVibration:
    def test_accel_magnitude_exists(self):
        df = load_vibration_xlsx(XLSX_PATH)
        assert 'accel_magnitude' in df.columns

    def test_accel_magnitude_formula(self):
        df = load_vibration_xlsx(XLSX_PATH)
        # Verify sqrt(x² + y² + z²)
        expected = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)
        np.testing.assert_array_almost_equal(df['accel_magnitude'], expected)

    def test_rolling_mag(self):
        df = load_vibration_xlsx(XLSX_PATH)
        assert 'rolling_mag' in df.columns
        assert df['rolling_mag'].isnull().sum() == 0

    def test_anomaly_flag(self):
        df = load_vibration_xlsx(XLSX_PATH)
        assert 'vibr_anomaly' in df.columns
        assert set(df['vibr_anomaly'].unique()).issubset({0, 1})


class TestAlertLevel:
    def test_critical_high_prob(self):
        row = {'Predicted_Failure_Prob': 0.9, 'RUL_Predicted_days': 100}
        assert compute_alert_level(row) == 'CRITICAL'

    def test_critical_low_rul(self):
        row = {'Predicted_Failure_Prob': 0.1, 'RUL_Predicted_days': 10}
        assert compute_alert_level(row) == 'CRITICAL'

    def test_warning(self):
        row = {'Predicted_Failure_Prob': 0.5, 'RUL_Predicted_days': 100}
        assert compute_alert_level(row) == 'WARNING'

    def test_healthy(self):
        row = {'Predicted_Failure_Prob': 0.1, 'RUL_Predicted_days': 200}
        assert compute_alert_level(row) == 'HEALTHY'

    def test_edge_critical_prob(self):
        row = {'Predicted_Failure_Prob': 0.76, 'RUL_Predicted_days': 200}
        assert compute_alert_level(row) == 'CRITICAL'

    def test_edge_warning_rul(self):
        row = {'Predicted_Failure_Prob': 0.1, 'RUL_Predicted_days': 55}
        assert compute_alert_level(row) == 'WARNING'


class TestHealthScore:
    def test_range(self):
        df = load_csv(CSV_PATH)
        scores = compute_health_score(df)
        assert scores.min() >= 0
        assert scores.max() <= 100

    def test_no_nans(self):
        df = load_csv(CSV_PATH)
        scores = compute_health_score(df)
        assert scores.isnull().sum() == 0


class TestMakeSequences:
    def test_output_shape(self):
        data = np.random.randn(100, 5)
        targets = np.random.randn(100)
        X, y = make_sequences(data, targets, seq_len=30)
        assert X.shape == (70, 30, 5)
        assert y.shape == (70,)


class TestPreprocessPipeline:
    def test_runs(self):
        df, vibr_df, encoders = preprocess_pipeline(CSV_PATH, XLSX_PATH)
        assert df.shape[0] == 5000
        assert 'Health_Score' in df.columns
        assert 'Alert_Level' in df.columns
        assert isinstance(encoders, dict)
        assert len(encoders) > 0
