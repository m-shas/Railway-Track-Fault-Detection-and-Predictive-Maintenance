"""Tests for anomaly_model, rul_model, and classifier — Phase 3,4,5 validation."""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocess import preprocess_pipeline
from anomaly_model import train_isolation_forest, predict_anomaly, save_model, load_model
from rul_model import train_rul_model, predict_rul
from classifier import train_classifier, predict_fault

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
CSV_PATH = os.path.join(BASE_DIR, 'data', 'RT_PLC_RSFPD.csv')
XLSX_PATH = os.path.join(BASE_DIR, 'data', 'vibration_analysis_graph.xlsx')


@pytest.fixture(scope="module")
def pipeline_data():
    """Run preprocessing once for all tests in this module."""
    df, vibr_df, encoders = preprocess_pipeline(CSV_PATH, XLSX_PATH)
    return df, vibr_df, encoders


@pytest.fixture(scope="module")
def anomaly_results(pipeline_data):
    """Train isolation forest once."""
    df, _, _ = pipeline_data
    model, scaler, labels, scores = train_isolation_forest(df)
    df["IF_Flag"] = (labels == -1).astype(int)
    return model, scaler, labels, scores, df


class TestIsolationForest:
    def test_anomaly_rate(self, anomaly_results):
        _, _, labels, _, _ = anomaly_results
        rate = (labels == -1).sum() / len(labels)
        assert 0.05 <= rate <= 0.15, f"Anomaly rate {rate:.2%} outside 5-15% range"

    def test_labels_shape(self, anomaly_results):
        _, _, labels, _, df = anomaly_results
        assert len(labels) == len(df)

    def test_predict_anomaly(self, anomaly_results):
        model, scaler, _, _, df = anomaly_results
        pred_labels, pred_scores = predict_anomaly(model, scaler, df.head(10))
        assert len(pred_labels) == 10
        assert len(pred_scores) == 10

    def test_save_load(self, anomaly_results, tmp_path):
        model, scaler, _, _, _ = anomaly_results
        path = str(tmp_path / "test_if.pkl")
        save_model(model, scaler, path)
        loaded_model, loaded_scaler = load_model(path)
        assert loaded_model is not None
        assert loaded_scaler is not None


class TestRULModel:
    def test_rul_positive(self, anomaly_results):
        _, _, _, _, df = anomaly_results
        model, scaler, metrics, (y_test, y_pred) = train_rul_model(df)
        # Predictions should mostly be positive
        assert (y_pred > 0).mean() > 0.8, "Most RUL predictions should be positive"

    def test_mae_is_float(self, anomaly_results):
        _, _, _, _, df = anomaly_results
        _, _, metrics, _ = train_rul_model(df)
        assert isinstance(metrics['mae'], float)
        assert isinstance(metrics['rmse'], float)
        assert isinstance(metrics['r2'], float)

    def test_predict_shape(self, anomaly_results):
        _, _, _, _, df = anomaly_results
        model, scaler, _, _ = train_rul_model(df)
        preds = predict_rul(model, scaler, df.head(20))
        assert len(preds) == 20


class TestClassifier:
    def test_output_length(self, anomaly_results):
        _, _, _, _, df = anomaly_results
        model, scaler, y_test, y_pred, accuracy = train_classifier(df)
        assert len(y_test) == len(y_pred)

    def test_accuracy_type(self, anomaly_results):
        _, _, _, _, df = anomaly_results
        _, _, _, _, accuracy = train_classifier(df)
        assert isinstance(accuracy, float)
        assert 0 <= accuracy <= 1

    def test_predict_fault(self, anomaly_results):
        _, _, _, _, df = anomaly_results
        from preprocess import encode_categoricals
        _, encoders = encode_categoricals(df)
        model, scaler, _, _, _ = train_classifier(df)
        labels, conf, descs = predict_fault(model, scaler, encoders['Failure_Type'], df.head(10))
        assert len(labels) == 10
        assert len(conf) == 10
        assert len(descs) == 10
