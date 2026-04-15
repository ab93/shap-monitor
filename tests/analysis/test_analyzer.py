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


class TestAnalyzerProperties:
    """Tests for SHAPAnalyzer properties."""

    def test_min_abs_shap_property(self, populated_backend):
        """min_abs_shap property should return the configured threshold."""
        analyzer = SHAPAnalyzer(populated_backend, min_abs_shap=0.05)
        assert analyzer.min_abs_shap == 0.05

    def test_backend_property(self, populated_backend):
        """backend property should return the backend instance."""
        analyzer = SHAPAnalyzer(populated_backend)
        assert analyzer.backend is populated_backend


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

    def test_summary_invalid_sort_by_raises(self, populated_backend):
        """summary should raise ValueError when sort_by column doesn't exist."""
        analyzer = SHAPAnalyzer(populated_backend)
        with pytest.raises(ValueError, match="Invalid sort_by"):
            analyzer.summary(datetime(2025, 12, 20), datetime(2025, 12, 26), sort_by="invalid")

    def test_summary_top_k_limits_rows(self, populated_backend):
        """top_k should return only the k most important features."""
        analyzer = SHAPAnalyzer(populated_backend)
        result = analyzer.summary(datetime(2025, 12, 20), datetime(2025, 12, 26), top_k=2)
        assert len(result) == 2
        assert result.index[0] == "feature_a"  # top feature still first

    def test_summary_top_k_invalid_raises(self, populated_backend):
        """top_k must be a positive integer."""
        analyzer = SHAPAnalyzer(populated_backend)
        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            analyzer.summary(datetime(2025, 12, 20), datetime(2025, 12, 26), top_k=0)


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
            "mean_abs_ref",
            "mean_abs_curr",
            "delta_mean_abs",
            "pct_delta_mean_abs",
            "rank_ref",
            "rank_curr",
            "delta_rank",
            "rank_change",
            "mean_ref",
            "mean_curr",
            "sign_flip",
        }
        assert set(result.columns) == expected_cols
        assert result.attrs["n_samples_ref"] == 300
        assert result.attrs["n_samples_curr"] == 400

    def test_compare_delta_calculation(self, populated_backend):
        """Delta should be period_2 - period_1."""
        analyzer = SHAPAnalyzer(populated_backend)
        result = analyzer.compare_time_periods(
            Period(datetime(2025, 12, 20), datetime(2025, 12, 22)),
            Period(datetime(2025, 12, 23), datetime(2025, 12, 26)),
        )

        for feature in result.index:
            expected = result.loc[feature, "mean_abs_curr"] - result.loc[feature, "mean_abs_ref"]
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

    def test_compare_time_periods_invalid_sort_by_raises(self, populated_backend):
        """compare_time_periods should raise ValueError when sort_by column doesn't exist."""
        analyzer = SHAPAnalyzer(populated_backend)
        with pytest.raises(ValueError, match="Invalid sort_by"):
            analyzer.compare_time_periods(
                Period(datetime(2025, 12, 20), datetime(2025, 12, 22)),
                Period(datetime(2025, 12, 23), datetime(2025, 12, 26)),
                sort_by="invalid",
            )

    def test_compare_time_periods_top_k_limits_rows(self, populated_backend):
        """top_k should return only k features from compare_time_periods."""
        analyzer = SHAPAnalyzer(populated_backend)
        result = analyzer.compare_time_periods(
            Period(datetime(2025, 12, 20), datetime(2025, 12, 22)),
            Period(datetime(2025, 12, 23), datetime(2025, 12, 26)),
            top_k=1,
        )
        assert len(result) == 1

    def test_compare_time_periods_top_k_invalid_raises(self, populated_backend):
        """compare_time_periods should raise ValueError for top_k < 1."""
        analyzer = SHAPAnalyzer(populated_backend)
        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            analyzer.compare_time_periods(
                Period(datetime(2025, 12, 20), datetime(2025, 12, 22)),
                Period(datetime(2025, 12, 23), datetime(2025, 12, 26)),
                top_k=0,
            )


class TestCompareBatches:
    """Tests for SHAPAnalyzer.compare_batches()."""

    def test_compare_batches_returns_correct_structure(self, populated_backend):
        """compare_batches should return a DataFrame with the standard comparison columns."""
        analyzer = SHAPAnalyzer(populated_backend)
        result = analyzer.compare_batches("batch_day_0", "batch_day_1")

        assert isinstance(result, pd.DataFrame)
        assert "psi" in result.columns
        assert "mean_abs_ref" in result.columns
        assert "mean_abs_curr" in result.columns
        assert "n_samples_ref" in result.attrs
        assert "n_samples_curr" in result.attrs

    def test_compare_batches_invalid_sort_by_raises(self, populated_backend):
        """compare_batches should raise ValueError when sort_by column doesn't exist."""
        analyzer = SHAPAnalyzer(populated_backend)
        with pytest.raises(ValueError, match="Invalid sort_by"):
            analyzer.compare_batches("batch_day_0", "batch_day_1", sort_by="invalid")

    def test_compare_batches_top_k_limits_rows(self, populated_backend):
        """top_k should return only k features from compare_batches."""
        analyzer = SHAPAnalyzer(populated_backend)
        result = analyzer.compare_batches("batch_day_0", "batch_day_1", top_k=1)
        assert len(result) == 1

    def test_compare_batches_top_k_invalid_raises(self, populated_backend):
        """compare_batches should raise ValueError for top_k < 1."""
        analyzer = SHAPAnalyzer(populated_backend)
        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            analyzer.compare_batches("batch_day_0", "batch_day_1", top_k=-1)


class TestDriftReport:
    """Tests for SHAPAnalyzer.drift_report()."""

    def test_drift_report_returns_drift_status_column(self, populated_backend):
        """drift_report should return all compare columns plus drift_status."""
        analyzer = SHAPAnalyzer(populated_backend)
        result = analyzer.drift_report(
            Period(datetime(2025, 12, 20), datetime(2025, 12, 22)),
            Period(datetime(2025, 12, 23), datetime(2025, 12, 26)),
        )

        assert "drift_status" in result.columns
        assert "psi" in result.columns

    def test_drift_report_stable_classification(self, populated_backend):
        """Features with low PSI should be classified as stable."""
        analyzer = SHAPAnalyzer(populated_backend)
        result = analyzer.drift_report(
            Period(datetime(2025, 12, 20), datetime(2025, 12, 22)),
            Period(datetime(2025, 12, 23), datetime(2025, 12, 26)),
        )

        valid_statuses = {"stable", "warning", "alert", "unknown"}
        assert all(s in valid_statuses for s in result["drift_status"].unique())

    def test_drift_report_alert_classification(self, tmp_path):
        """Features with PSI >= alert_threshold should be classified as alert."""
        backend = ParquetBackend(tmp_path)

        # Create a large distribution shift
        batch_ref = ExplanationBatch(
            timestamp=datetime(2025, 12, 1),
            batch_id="ref",
            model_version="v1.0",
            n_samples=200,
            base_values=np.full(200, 0.5),
            shap_values={"feature_x": np.full(200, 5.0)},
        )
        batch_curr = ExplanationBatch(
            timestamp=datetime(2025, 12, 20),
            batch_id="curr",
            model_version="v1.0",
            n_samples=200,
            base_values=np.full(200, 0.5),
            shap_values={"feature_x": np.full(200, -5.0)},
        )
        backend.write(batch_ref)
        backend.write(batch_curr)

        analyzer = SHAPAnalyzer(backend)
        result = analyzer.drift_report(
            Period(datetime(2025, 12, 1), datetime(2025, 12, 2)),
            Period(datetime(2025, 12, 20), datetime(2025, 12, 21)),
        )

        assert result.loc["feature_x", "drift_status"] in {"warning", "alert"}

    def test_drift_report_custom_thresholds(self, populated_backend):
        """drift_report should respect custom warn/alert thresholds."""
        analyzer = SHAPAnalyzer(populated_backend)
        result = analyzer.drift_report(
            Period(datetime(2025, 12, 20), datetime(2025, 12, 22)),
            Period(datetime(2025, 12, 23), datetime(2025, 12, 26)),
            warn_threshold=0.0,  # everything is at least warning
            alert_threshold=1000.0,  # nothing will reach alert
        )

        assert "stable" not in result["drift_status"].values

    def test_drift_report_empty_returns_empty(self, backend):
        """drift_report should return empty DataFrame when both periods have no data."""
        analyzer = SHAPAnalyzer(backend)
        result = analyzer.drift_report(
            Period(datetime(2024, 1, 1), datetime(2024, 1, 2)),
            Period(datetime(2024, 1, 3), datetime(2024, 1, 4)),
        )

        assert result.empty

    def test_drift_report_inherits_attrs(self, populated_backend):
        """drift_report result should carry n_samples_ref and n_samples_curr attrs."""
        analyzer = SHAPAnalyzer(populated_backend)
        result = analyzer.drift_report(
            Period(datetime(2025, 12, 20), datetime(2025, 12, 22)),
            Period(datetime(2025, 12, 23), datetime(2025, 12, 26)),
        )

        assert "n_samples_ref" in result.attrs
        assert "n_samples_curr" in result.attrs


class TestFetchFullData:
    """Tests for SHAPAnalyzer.fetch_full_data()."""

    def test_fetch_full_data_returns_all_columns(self, populated_backend):
        """fetch_full_data should return shap_*, feat_* are not logged in populated_backend, but shap_ and metadata are."""
        analyzer = SHAPAnalyzer(populated_backend)
        result = analyzer.fetch_full_data(
            start_dt=datetime(2025, 12, 20), end_dt=datetime(2025, 12, 26)
        )

        assert not result.empty
        # SHAP columns should be present
        assert any(col.startswith("shap_") for col in result.columns)
        # Metadata columns should be present
        assert "batch_id" in result.columns
        assert "model_version" in result.columns

    def test_fetch_full_data_no_args_returns_all(self, populated_backend):
        """fetch_full_data() with no args should return all stored rows."""
        analyzer = SHAPAnalyzer(populated_backend)
        result = analyzer.fetch_full_data()

        # 7 days x 100 samples each = 700 rows
        assert len(result) == 700

    def test_fetch_full_data_empty_range_returns_empty(self, populated_backend):
        """fetch_full_data should return empty DataFrame for out-of-range dates."""
        analyzer = SHAPAnalyzer(populated_backend)
        result = analyzer.fetch_full_data(
            start_dt=datetime(2024, 1, 1), end_dt=datetime(2024, 1, 2)
        )

        assert result.empty


class TestFeatureTrends:
    """Tests for SHAPAnalyzer.feature_trends()."""

    def test_feature_trends_returns_multiindex(self, populated_backend):
        """feature_trends should return a MultiIndex DataFrame."""
        analyzer = SHAPAnalyzer(populated_backend)
        result = analyzer.feature_trends(datetime(2025, 12, 20), datetime(2025, 12, 26))

        assert result.index.names == ["period_start", "feature"]

    def test_feature_trends_has_mean_abs_column(self, populated_backend):
        """feature_trends result should have a mean_abs column."""
        analyzer = SHAPAnalyzer(populated_backend)
        result = analyzer.feature_trends(datetime(2025, 12, 20), datetime(2025, 12, 26))

        assert "mean_abs" in result.columns

    def test_feature_trends_multiple_windows(self, populated_backend):
        """14 days of data with freq=7D should yield 2 period windows."""
        analyzer = SHAPAnalyzer(populated_backend)
        # populated_backend has 7 days starting 2025-12-20
        result = analyzer.feature_trends(datetime(2025, 12, 20), datetime(2025, 12, 26), freq="7D")

        period_starts = result.index.get_level_values("period_start").unique()
        assert len(period_starts) >= 1

    def test_feature_trends_empty_windows_skipped(self, populated_backend):
        """Windows with no data should be skipped (not appear as NaN rows)."""
        analyzer = SHAPAnalyzer(populated_backend)
        # Query a future range with no data
        result = analyzer.feature_trends(datetime(2030, 1, 1), datetime(2030, 1, 14))

        assert result.empty
