"""Tests for src/alerts.py — Phase 6 validation."""

import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocess import preprocess_pipeline
from anomaly_model import train_isolation_forest
from alerts import compute_alert_level, compute_alert_score, generate_alert_log

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
CSV_PATH = os.path.join(BASE_DIR, 'data', 'RT_PLC_RSFPD.csv')
XLSX_PATH = os.path.join(BASE_DIR, 'data', 'vibration_analysis_graph.xlsx')


@pytest.fixture(scope="module")
def alert_data():
    """Generate alert log for tests."""
    df, _, _ = preprocess_pipeline(CSV_PATH, XLSX_PATH)
    _, _, labels, _ = train_isolation_forest(df)
    df["IF_Flag"] = (labels == -1).astype(int)
    alert_log = generate_alert_log(df)
    return alert_log, df


class TestAlertLog:
    def test_no_healthy_rows(self, alert_data):
        alert_log, _ = alert_data
        assert 'HEALTHY' not in alert_log['Alert_Level'].values, \
            "Alert log should not contain HEALTHY rows"

    def test_alert_score_positive(self, alert_data):
        alert_log, _ = alert_data
        assert (alert_log['Alert_Score'] > 0).all(), \
            "All alert scores should be > 0"

    def test_sorted_descending(self, alert_data):
        alert_log, _ = alert_data
        scores = alert_log['Alert_Score'].values
        assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1)), \
            "Alert log should be sorted by Alert_Score descending"

    def test_critical_rows_high_prob(self, alert_data):
        alert_log, _ = alert_data
        critical = alert_log[alert_log['Alert_Level'] == 'CRITICAL']
        if len(critical) > 0:
            # At least some CRITICAL rows should have elevated prob or low RUL
            has_trigger = (
                (critical['Predicted_Failure_Prob'] > 0.45) |
                (critical['RUL_Predicted_days'] < 60) |
                (critical['Vibration_m_s2'] > 0.55) |
                (critical['Temperature_C'] > 45)
            )
            assert has_trigger.any(), "CRITICAL rows should have at least one trigger condition met"

    def test_has_fault_description(self, alert_data):
        alert_log, _ = alert_data
        assert 'Fault_Description' in alert_log.columns
        assert 'Recommended_Action' in alert_log.columns

    def test_not_empty(self, alert_data):
        alert_log, _ = alert_data
        assert len(alert_log) > 0, "Alert log should not be empty"


class TestAlertScore:
    def test_range(self):
        row = {'Predicted_Failure_Prob': 0.5, 'RUL_Predicted_days': 100,
               'Edge_Anomaly_Score': 0.3, 'IF_Flag': 0}
        score = compute_alert_score(row)
        assert 0 <= score <= 100

    def test_high_urgency(self):
        row = {'Predicted_Failure_Prob': 1.0, 'RUL_Predicted_days': 0,
               'Edge_Anomaly_Score': 1.0, 'IF_Flag': 1}
        score = compute_alert_score(row)
        assert score >= 80, "Max urgency should give high score"
