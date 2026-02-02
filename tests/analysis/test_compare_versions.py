"""Tests for compare_versions functionality."""

from datetime import datetime

import numpy as np
import pytest

from shapmonitor.analysis import SHAPAnalyzer
from shapmonitor.backends import ParquetBackend
from shapmonitor.types import ExplanationBatch


@pytest.fixture
def multi_version_backend(tmp_path):
    """Backend with data from multiple model versions."""
    backend = ParquetBackend(tmp_path, partition_by=["date", "model_version"])

    # v1.0 data
    batch_v1 = ExplanationBatch(
        timestamp=datetime(2025, 12, 20),
        batch_id="batch_v1",
        model_version="v1.0",
        n_samples=100,
        base_values=np.full(100, 0.5),
        shap_values={
            "feature_a": np.random.normal(0.5, 0.1, 100),
            "feature_b": np.random.normal(0.3, 0.1, 100),
        },
    )
    # v2.0 data (different importance)
    batch_v2 = ExplanationBatch(
        timestamp=datetime(2025, 12, 20),
        batch_id="batch_v2",
        model_version="v2.0",
        n_samples=100,
        base_values=np.full(100, 0.5),
        shap_values={
            "feature_a": np.random.normal(0.2, 0.1, 100),  # Less important
            "feature_b": np.random.normal(0.7, 0.1, 100),  # More important
        },
    )
    backend.write(batch_v1)
    backend.write(batch_v2)

    return backend


class TestCompareVersions:
    """Tests for SHAPAnalyzer.compare_versions()."""

    def test_compare_versions_returns_dataframe(self, multi_version_backend):
        """compare_versions should return a comparison DataFrame."""
        analyzer = SHAPAnalyzer(multi_version_backend)
        result = analyzer.compare_versions("v1.0", "v2.0")

        assert not result.empty
        assert "psi" in result.columns
        assert "delta_mean_abs" in result.columns

    def test_compare_versions_detects_importance_change(self, multi_version_backend):
        """compare_versions should detect feature importance changes."""
        analyzer = SHAPAnalyzer(multi_version_backend)
        result = analyzer.compare_versions("v1.0", "v2.0")

        # feature_a should show decreased importance
        assert result.loc["feature_a", "delta_mean_abs"] < 0
        # feature_b should show increased importance
        assert result.loc["feature_b", "delta_mean_abs"] > 0

    def test_compare_versions_missing_version_has_nan(self, multi_version_backend):
        """Comparing with non-existent version should have NaN for missing data."""
        analyzer = SHAPAnalyzer(multi_version_backend)
        result = analyzer.compare_versions("v1.0", "v99.0")

        # Should still return features from v1.0, but with NaN for v99.0
        assert not result.empty
        assert result["mean_abs_1"].notna().all()  # v1.0 data exists
        assert result["mean_abs_2"].isna().all()  # v99.0 data is missing
