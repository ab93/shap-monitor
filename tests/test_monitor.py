"""Tests for SHAPMonitor."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
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

    def test_sampling_minimum_one_sample(self, explainer, tmp_path, regression_data, feature_names):
        """Sampling should always select at least one sample even with very small sample_rate."""
        X, _ = regression_data
        sample_rate = 0.001  # Very small rate that would give 0 samples for 100 rows

        monitor = SHAPMonitor(
            explainer=explainer,
            data_dir=tmp_path,
            sample_rate=sample_rate,
            feature_names=feature_names,
            random_seed=42,
        )
        monitor.log_batch(X)

        backend = ParquetBackend(tmp_path)
        today = datetime.now()
        df = backend.read(today - timedelta(days=1), today + timedelta(days=1))

        # Should have at least 1 sample
        assert len(df) >= 1

    def test_predictions_sampled_with_features(
        self, explainer, tmp_path, regression_data, feature_names
    ):
        """Predictions (y) should be sampled alongside features (X)."""
        X, y = regression_data
        sample_rate = 0.5

        monitor = SHAPMonitor(
            explainer=explainer,
            data_dir=tmp_path,
            sample_rate=sample_rate,
            feature_names=feature_names,
            random_seed=42,
        )
        monitor.log_batch(X, y=y)

        backend = ParquetBackend(tmp_path)
        today = datetime.now()
        df = backend.read(today - timedelta(days=1), today + timedelta(days=1))

        expected_samples = int(len(X) * sample_rate)
        assert len(df) == expected_samples
        # Check that predictions column exists and has correct length
        assert "prediction" in df.columns
        assert df["prediction"].notna().sum() == expected_samples

    def test_log_shap_writes_to_backend(self, monitor, sample_features, feature_names):
        """log_shap should write pre-computed SHAP values to backend."""
        explanation = monitor.compute(sample_features)
        monitor.log_shap(explanation)

        today = datetime.now()
        df = monitor.backend.read(today - timedelta(days=1), today + timedelta(days=1))

        assert len(df) == 2
        for feat in feature_names:
            assert f"shap_{feat}" in df.columns
        # feature values and predictions should not be present
        assert "prediction" not in df.columns
        assert all(f"feat_{feat}" not in df.columns for feat in feature_names)

    def test_log_shap_auto_generates_feature_names(self, explainer, tmp_path, sample_features):
        """log_shap should generate feature names when none are provided."""
        monitor = SHAPMonitor(
            explainer=explainer,
            data_dir=tmp_path,
            sample_rate=1.0,
        )
        explanation = monitor.compute(sample_features)
        monitor.log_shap(explanation)

        assert monitor.feature_names == [f"feat_{i}" for i in range(5)]

        today = datetime.now()
        df = monitor.backend.read(today - timedelta(days=1), today + timedelta(days=1))
        assert len(df) == 2

    def test_log_shap_numpy_with_base_values(self, monitor, sample_features, feature_names):
        """log_shap should accept a raw numpy array with explicit base_values."""
        explanation = monitor.compute(sample_features)
        shap_array = explanation.values
        base_val = explanation.base_values

        monitor.log_shap(shap_array, base_values=base_val)

        today = datetime.now()
        df = monitor.backend.read(today - timedelta(days=1), today + timedelta(days=1))

        assert len(df) == 2
        for feat in feature_names:
            assert f"shap_{feat}" in df.columns
        assert df["base_value"].notna().all()

    def test_log_shap_numpy_scalar_base_value(self, monitor, sample_features, feature_names):
        """log_shap should broadcast a scalar base_value to all samples."""
        explanation = monitor.compute(sample_features)
        shap_array = explanation.values

        monitor.log_shap(shap_array, base_values=0.5)

        today = datetime.now()
        df = monitor.backend.read(today - timedelta(days=1), today + timedelta(days=1))

        assert len(df) == 2
        np.testing.assert_allclose(df["base_value"].values, 0.5, atol=1e-6)

    def test_log_shap_numpy_without_base_values(self, monitor, sample_features, feature_names):
        """log_shap should fill base_value with NaN when base_values is omitted."""
        explanation = monitor.compute(sample_features)
        shap_array = explanation.values

        monitor.log_shap(shap_array)

        today = datetime.now()
        df = monitor.backend.read(today - timedelta(days=1), today + timedelta(days=1))

        assert len(df) == 2
        for feat in feature_names:
            assert f"shap_{feat}" in df.columns
        assert df["base_value"].isna().all()

    def test_dataframe_input_preserves_categorical(self, explainer, tmp_path, feature_names):
        """DataFrame input with categorical columns should be handled correctly."""
        # Create a DataFrame with mixed types including categorical
        n_samples = 50
        df_input = pd.DataFrame(
            {
                "feat_0": np.random.randn(n_samples),
                "feat_1": np.random.randn(n_samples),
                "feat_2": np.random.randn(n_samples),
                "feat_3": np.random.randn(n_samples),
                "feat_4": np.random.randn(n_samples),
            }
        )

        monitor = SHAPMonitor(
            explainer=explainer,
            data_dir=tmp_path,
            sample_rate=1.0,
            feature_names=feature_names,
            random_seed=42,
        )

        # Should not raise an error
        monitor.log_batch(df_input)

        backend = ParquetBackend(tmp_path)
        today = datetime.now()
        df = backend.read(today - timedelta(days=1), today + timedelta(days=1))

        assert len(df) == n_samples


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
