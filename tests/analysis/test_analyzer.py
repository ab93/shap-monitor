"""Tests for SHAPAnalyzer."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from freezegun import freeze_time

from shapmonitor.analysis import SHAPAnalyzer
from shapmonitor.backends import ParquetBackend
from shapmonitor.types import ExplanationBatch, Period


@pytest.fixture
def backend(tmp_path):
    """Create a ParquetBackend for testing."""
    return ParquetBackend(tmp_path)


@pytest.fixture
def populated_backend(tmp_path):
    """Backend with 7 days of sample data."""
    backend = ParquetBackend(tmp_path)
    base_date = datetime(2025, 12, 20)

    for day in range(7):
        with freeze_time(base_date + timedelta(days=day)):
            batch = ExplanationBatch(
                timestamp=datetime.now(),
                batch_id=f"batch_day_{day}",
                model_version="v1.0",
                n_samples=100,
                base_values=np.full(100, 0.5),
                shap_values={
                    "feature_a": np.random.normal(0.5, 0.1, 100),  # high importance
                    "feature_b": np.random.normal(-0.3, 0.1, 100),  # medium importance
                    "feature_c": np.random.normal(0.01, 0.005, 100),  # low importance
                },
            )
            backend.write(batch)

    return backend


class TestSummary:
    """Tests for SHAPAnalyzer.summary()."""

    def test_summary_returns_correct_structure(self, populated_backend):
        """Summary should return DataFrame with correct columns and index."""
        analyzer = SHAPAnalyzer(populated_backend)
        result = analyzer.summary(datetime(2025, 12, 20), datetime(2025, 12, 26))

        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == {"mean_abs", "mean", "std", "min", "max"}
        assert result.index.name == "feature"
        assert result.attrs["n_samples"] == 700

    def test_summary_sorted_by_importance(self, populated_backend):
        """Summary should be sorted by mean_abs descending."""
        analyzer = SHAPAnalyzer(populated_backend)
        result = analyzer.summary(datetime(2025, 12, 20), datetime(2025, 12, 26))

        assert result.index[0] == "feature_a"  # highest importance
        assert result.index[-1] == "feature_c"  # lowest importance

    def test_summary_threshold_filters_features(self, populated_backend):
        """Threshold should filter out low importance features."""
        analyzer = SHAPAnalyzer(populated_backend, min_abs_shap=0.1)
        result = analyzer.summary(datetime(2025, 12, 20), datetime(2025, 12, 26))

        assert "feature_c" not in result.index
        assert "feature_a" in result.index

    def test_summary_empty_date_range(self, populated_backend):
        """Should return empty DataFrame for date range with no data."""
        analyzer = SHAPAnalyzer(populated_backend)
        result = analyzer.summary(datetime(2024, 1, 1), datetime(2024, 1, 2))

        assert result.empty


class TestCompareTimePeriods:
    """Tests for SHAPAnalyzer.compare_time_periods()."""

    def test_compare_returns_correct_structure(self, populated_backend):
        """Compare should return DataFrame with correct columns."""
        analyzer = SHAPAnalyzer(populated_backend)
        result = analyzer.compare_time_periods(
            Period(datetime(2025, 12, 20), datetime(2025, 12, 22)),
            Period(datetime(2025, 12, 23), datetime(2025, 12, 26)),
        )

        expected_cols = {
            "psi",
            "mean_abs_1",
            "mean_abs_2",
            "delta_mean_abs",
            "pct_delta_mean_abs",
            "rank_1",
            "rank_2",
            "delta_rank",
            "rank_change",
            "mean_1",
            "mean_2",
            "sign_flip",
        }
        assert set(result.columns) == expected_cols
        assert result.attrs["n_samples_1"] == 300
        assert result.attrs["n_samples_2"] == 400

    def test_compare_delta_calculation(self, populated_backend):
        """Delta should be period_2 - period_1."""
        analyzer = SHAPAnalyzer(populated_backend)
        result = analyzer.compare_time_periods(
            Period(datetime(2025, 12, 20), datetime(2025, 12, 22)),
            Period(datetime(2025, 12, 23), datetime(2025, 12, 26)),
        )

        for feature in result.index:
            expected = result.loc[feature, "mean_abs_2"] - result.loc[feature, "mean_abs_1"]
            np.testing.assert_almost_equal(
                result.loc[feature, "delta_mean_abs"], expected, decimal=5
            )

    def test_compare_accepts_plain_tuple(self, populated_backend):
        """Compare should accept plain tuples as well as Period."""
        analyzer = SHAPAnalyzer(populated_backend)
        # Using plain tuples instead of Period
        result = analyzer.compare_time_periods(
            (datetime(2025, 12, 20), datetime(2025, 12, 22)),
            (datetime(2025, 12, 23), datetime(2025, 12, 26)),
        )

        assert not result.empty
        assert "psi" in result.columns

    def test_compare_sign_flip_detection(self, tmp_path):
        """Sign flip should detect when mean SHAP changes sign."""
        backend = ParquetBackend(tmp_path)

        # Period 1: positive mean
        batch1 = ExplanationBatch(
            timestamp=datetime(2025, 12, 20),
            batch_id="batch_1",
            model_version="v1.0",
            n_samples=100,
            base_values=np.full(100, 0.5),
            shap_values={"feature_x": np.full(100, 0.5)},
        )
        # Period 2: negative mean
        batch2 = ExplanationBatch(
            timestamp=datetime(2025, 12, 25),
            batch_id="batch_2",
            model_version="v1.0",
            n_samples=100,
            base_values=np.full(100, 0.5),
            shap_values={"feature_x": np.full(100, -0.5)},
        )
        backend.write(batch1)
        backend.write(batch2)

        analyzer = SHAPAnalyzer(backend)
        result = analyzer.compare_time_periods(
            Period(datetime(2025, 12, 20), datetime(2025, 12, 21)),
            Period(datetime(2025, 12, 25), datetime(2025, 12, 26)),
        )

        assert result.loc["feature_x", "sign_flip"] == True  # noqa: E712

    def test_compare_rank_change_values(self, populated_backend):
        """Rank change should have valid categorical values."""
        analyzer = SHAPAnalyzer(populated_backend)
        result = analyzer.compare_time_periods(
            Period(datetime(2025, 12, 20), datetime(2025, 12, 22)),
            Period(datetime(2025, 12, 23), datetime(2025, 12, 26)),
        )

        valid_values = {"increased", "decreased", "no_change"}
        assert all(val in valid_values for val in result["rank_change"].unique())

    def test_compare_empty_periods(self, backend):
        """Should return empty DataFrame when both periods have no data."""
        analyzer = SHAPAnalyzer(backend)
        result = analyzer.compare_time_periods(
            Period(datetime(2024, 1, 1), datetime(2024, 1, 2)),
            Period(datetime(2024, 1, 3), datetime(2024, 1, 4)),
        )

        assert result.empty
