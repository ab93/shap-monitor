"""Tests for SHAPMonitor."""

from datetime import datetime, timedelta

import pytest
import shap
from sklearn.ensemble import RandomForestRegressor

from shapmonitor import SHAPMonitor
from shapmonitor.backends import ParquetBackend


class TestSHAPMonitor:
    """Unit tests for SHAPMonitor."""

    @pytest.fixture
    def model(self, regression_data):
        X, y = regression_data
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        return model

    @pytest.fixture
    def explainer(self, model):
        return shap.TreeExplainer(model)

    @pytest.fixture
    def monitor(self, explainer, tmp_path, feature_names):
        return SHAPMonitor(
            explainer=explainer,
            data_dir=tmp_path,
            sample_rate=1.0,
            model_version="test-v1",
            feature_names=feature_names,
        )

    def test_creates_data_directory(self, explainer, tmp_path):
        """Monitor should create data directory if it doesn't exist."""
        data_dir = tmp_path / "new_dir"
        SHAPMonitor(explainer=explainer, data_dir=data_dir)
        assert data_dir.exists()

    def test_compute_returns_explanation(self, monitor, sample_features):
        """Compute should return SHAP explanation."""
        explanation = monitor.compute(sample_features)

        assert hasattr(explanation, "values")
        assert hasattr(explanation, "base_values")

    def test_generate_batch_id_is_unique(self):
        """Each batch ID should be unique."""
        ids = [SHAPMonitor._generate_batch_id() for _ in range(100)]
        assert len(set(ids)) == 100

    def test_sampling_respects_rate(self, explainer, tmp_path, regression_data, feature_names):
        """Sampling should select approximately sample_rate fraction of rows."""
        X, _ = regression_data
        sample_rate = 0.5

        monitor = SHAPMonitor(
            explainer=explainer,
            data_dir=tmp_path,
            sample_rate=sample_rate,
            feature_names=feature_names,
            random_seed=42,
        )
        monitor.log_batch(X)

        # Read back and verify sample count
        backend = ParquetBackend(tmp_path)
        today = datetime.now()
        df = backend.read(today - timedelta(days=1), today + timedelta(days=1))

        expected_samples = int(len(X) * sample_rate)
        assert len(df) == expected_samples


@pytest.mark.integration
class TestSHAPMonitorIntegration:
    """Integration tests for SHAPMonitor end-to-end flow."""

    @pytest.fixture
    def model(self, regression_data):
        X, y = regression_data
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        return model

    @pytest.fixture
    def explainer(self, model):
        return shap.TreeExplainer(model)

    def test_full_logging_flow(self, explainer, tmp_path, regression_data, feature_names):
        """Integration: log_batch should compute SHAP and write to backend."""
        X, _ = regression_data
        monitor = SHAPMonitor(
            explainer=explainer,
            data_dir=tmp_path,
            sample_rate=1.0,
            model_version="integration-v1",
            feature_names=feature_names,
        )

        # Log a batch
        monitor.log_batch(X[:10])

        # Read back from backend
        backend = ParquetBackend(tmp_path)
        today = datetime.now()
        df = backend.read(today - timedelta(days=1), today + timedelta(days=1))

        # Verify data was written
        assert len(df) == 10
        for feat in feature_names:
            assert f"shap_{feat}" in df.columns
