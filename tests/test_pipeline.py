"""Tests for full pipeline — Phase 7 validation."""

import sys
import os
import subprocess
import pytest

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
PIPELINE_SCRIPT = os.path.join(BASE_DIR, 'src', 'pipeline.py')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')


class TestPipelineExecution:
    """These tests verify that pipeline.py ran successfully.
    
    They check the output files rather than re-running the pipeline,
    which is time-consuming. Run `python src/pipeline.py` first.
    """

    def test_isolation_forest_exists(self):
        path = os.path.join(MODELS_DIR, 'isolation_forest.pkl')
        assert os.path.exists(path), f"Missing: {path}"
        assert os.path.getsize(path) > 1024, "isolation_forest.pkl should be > 1KB"

    def test_rul_model_exists(self):
        path = os.path.join(MODELS_DIR, 'rul_model.pkl')
        assert os.path.exists(path), f"Missing: {path}"
        assert os.path.getsize(path) > 1024, "rul_model.pkl should be > 1KB"

    def test_clf_model_exists(self):
        path = os.path.join(MODELS_DIR, 'clf_model.pkl')
        assert os.path.exists(path), f"Missing: {path}"
        assert os.path.getsize(path) > 1024, "clf_model.pkl should be > 1KB"

    def test_dashboard_exists(self):
        path = os.path.join(OUTPUTS_DIR, 'railway_dashboard.html')
        assert os.path.exists(path), f"Missing: {path}"
        assert os.path.getsize(path) > 1024, "railway_dashboard.html should be > 1KB"

    def test_alert_log_exists(self):
        path = os.path.join(OUTPUTS_DIR, 'alert_log.csv')
        assert os.path.exists(path), f"Missing: {path}"
        assert os.path.getsize(path) > 1024, "alert_log.csv should be > 1KB"

    def test_dashboard_data_exists(self):
        path = os.path.join(OUTPUTS_DIR, 'dashboard_data.json')
        assert os.path.exists(path), f"Missing: {path}"
        assert os.path.getsize(path) > 1024, "dashboard_data.json should be > 1KB"
