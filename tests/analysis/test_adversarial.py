"""Tests for SHAPAnalyzer.compare_adversarial()."""

from datetime import datetime

import numpy as np
import pytest

from shapmonitor.analysis import SHAPAnalyzer
from shapmonitor.backends import ParquetBackend
from shapmonitor.types import ExplanationBatch, Period


@pytest.fixture
def shifted_backend(tmp_path):
    """Backend with two clearly different SHAP distributions across two time periods."""
    backend = ParquetBackend(tmp_path)
    rng = np.random.default_rng(42)

    # Period 1: feature_a dominant
    batch_ref = ExplanationBatch(
        timestamp=datetime(2025, 12, 1),
        batch_id="batch_ref",
        model_version="v1.0",
        n_samples=200,
        base_values=np.full(200, 0.5),
        shap_values={
            "feature_a": rng.normal(5.0, 0.1, 200),
            "feature_b": rng.normal(0.0, 0.1, 200),
        },
    )
    # Period 2: feature_b dominant (clearly shifted)
    batch_curr = ExplanationBatch(
        timestamp=datetime(2025, 12, 20),
        batch_id="batch_curr",
        model_version="v1.0",
        n_samples=200,
        base_values=np.full(200, 0.5),
        shap_values={
            "feature_a": rng.normal(0.0, 0.1, 200),
            "feature_b": rng.normal(5.0, 0.1, 200),
        },
    )
    backend.write(batch_ref)
    backend.write(batch_curr)
    return backend


@pytest.fixture
def stable_backend(tmp_path):
    """Backend with two identical SHAP distributions across two time periods."""
    backend = ParquetBackend(tmp_path)
    rng = np.random.default_rng(0)
    data = rng.normal(0, 1, (400, 2))

    for i, (ts, bid) in enumerate(
        [(datetime(2025, 12, 1), "ref"), (datetime(2025, 12, 20), "curr")]
    ):
        batch = ExplanationBatch(
            timestamp=ts,
            batch_id=bid,
            model_version="v1.0",
            n_samples=200,
            base_values=np.full(200, 0.5),
            shap_values={
                "feature_a": data[i * 200 : (i + 1) * 200, 0],
                "feature_b": data[i * 200 : (i + 1) * 200, 1],
            },
        )
        backend.write(batch)
    return backend


class TestCompareAdversarial:
    """Tests for SHAPAnalyzer.compare_adversarial()."""

    def test_returns_correct_columns(self, shifted_backend):
        """compare_adversarial should return the expected columns."""
        analyzer = SHAPAnalyzer(shifted_backend)
        result = analyzer.compare_adversarial(
            Period(datetime(2025, 12, 1), datetime(2025, 12, 2)),
            Period(datetime(2025, 12, 20), datetime(2025, 12, 21)),
            random_state=0,
        )

        assert not result.empty
        expected_cols = {"adv_importance", "mean_abs_ref", "mean_abs_curr", "delta_mean_abs"}
        assert set(result.columns) == expected_cols
        assert result.index.name == "feature"

    def test_adversarial_auc_stored_in_attrs(self, shifted_backend):
        """adversarial_auc should be stored in result.attrs."""
        analyzer = SHAPAnalyzer(shifted_backend)
        result = analyzer.compare_adversarial(
            Period(datetime(2025, 12, 1), datetime(2025, 12, 2)),
            Period(datetime(2025, 12, 20), datetime(2025, 12, 21)),
            random_state=0,
        )

        assert "adversarial_auc" in result.attrs
        assert "n_samples_ref" in result.attrs
        assert "n_samples_curr" in result.attrs

    def test_detects_strong_shift(self, shifted_backend):
        """Clearly different SHAP distributions should produce high adversarial AUC."""
        analyzer = SHAPAnalyzer(shifted_backend)
        result = analyzer.compare_adversarial(
            Period(datetime(2025, 12, 1), datetime(2025, 12, 2)),
            Period(datetime(2025, 12, 20), datetime(2025, 12, 21)),
            random_state=0,
        )

        assert result.attrs["adversarial_auc"] > 0.9

    def test_stable_distributions_low_auc(self, stable_backend):
        """Identical SHAP distributions should produce AUC close to 0.5."""
        analyzer = SHAPAnalyzer(stable_backend)
        result = analyzer.compare_adversarial(
            Period(datetime(2025, 12, 1), datetime(2025, 12, 2)),
            Period(datetime(2025, 12, 20), datetime(2025, 12, 21)),
            random_state=0,
        )

        assert result.attrs["adversarial_auc"] < 0.65

    def test_sorted_by_adv_importance_by_default(self, shifted_backend):
        """Results should be sorted by adv_importance descending by default."""
        analyzer = SHAPAnalyzer(shifted_backend)
        result = analyzer.compare_adversarial(
            Period(datetime(2025, 12, 1), datetime(2025, 12, 2)),
            Period(datetime(2025, 12, 20), datetime(2025, 12, 21)),
            random_state=0,
        )

        importances = result["adv_importance"].tolist()
        assert importances == sorted(importances, reverse=True)

    def test_empty_ref_period_returns_empty(self, shifted_backend):
        """Empty reference period should return empty DataFrame."""
        analyzer = SHAPAnalyzer(shifted_backend)
        result = analyzer.compare_adversarial(
            Period(datetime(2024, 1, 1), datetime(2024, 1, 2)),
            Period(datetime(2025, 12, 20), datetime(2025, 12, 21)),
            random_state=0,
        )
        assert result.empty

    def test_empty_curr_period_returns_empty(self, shifted_backend):
        """Empty current period should return empty DataFrame."""
        analyzer = SHAPAnalyzer(shifted_backend)
        result = analyzer.compare_adversarial(
            Period(datetime(2025, 12, 1), datetime(2025, 12, 2)),
            Period(datetime(2024, 1, 1), datetime(2024, 1, 2)),
            random_state=0,
        )
        assert result.empty

    def test_top_k_limits_rows(self, shifted_backend):
        """top_k should return only k features."""
        analyzer = SHAPAnalyzer(shifted_backend)
        result = analyzer.compare_adversarial(
            Period(datetime(2025, 12, 1), datetime(2025, 12, 2)),
            Period(datetime(2025, 12, 20), datetime(2025, 12, 21)),
            top_k=1,
            random_state=0,
        )
        assert len(result) == 1

    def test_top_k_invalid_raises(self, shifted_backend):
        """top_k < 1 should raise ValueError."""
        analyzer = SHAPAnalyzer(shifted_backend)
        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            analyzer.compare_adversarial(
                Period(datetime(2025, 12, 1), datetime(2025, 12, 2)),
                Period(datetime(2025, 12, 20), datetime(2025, 12, 21)),
                top_k=0,
            )

    def test_invalid_sort_by_raises(self, shifted_backend):
        """Invalid sort_by should raise ValueError."""
        analyzer = SHAPAnalyzer(shifted_backend)
        with pytest.raises(ValueError, match="Invalid sort_by"):
            analyzer.compare_adversarial(
                Period(datetime(2025, 12, 1), datetime(2025, 12, 2)),
                Period(datetime(2025, 12, 20), datetime(2025, 12, 21)),
                sort_by="nonexistent_col",
            )


class TestCompareAdversarialBatches:
    """Tests for SHAPAnalyzer.compare_adversarial_batches()."""

    def test_returns_correct_columns(self, shifted_backend):
        """compare_adversarial_batches should return the expected columns."""
        analyzer = SHAPAnalyzer(shifted_backend)
        result = analyzer.compare_adversarial_batches("batch_ref", "batch_curr", random_state=0)

        assert not result.empty
        expected_cols = {"adv_importance", "mean_abs_ref", "mean_abs_curr", "delta_mean_abs"}
        assert set(result.columns) == expected_cols
        assert result.index.name == "feature"

    def test_adversarial_auc_stored_in_attrs(self, shifted_backend):
        """adversarial_auc should be stored in result.attrs."""
        analyzer = SHAPAnalyzer(shifted_backend)
        result = analyzer.compare_adversarial_batches("batch_ref", "batch_curr", random_state=0)

        assert "adversarial_auc" in result.attrs
        assert "n_samples_ref" in result.attrs
        assert "n_samples_curr" in result.attrs

    def test_detects_strong_shift(self, shifted_backend):
        """Clearly different SHAP batches should produce high adversarial AUC."""
        analyzer = SHAPAnalyzer(shifted_backend)
        result = analyzer.compare_adversarial_batches("batch_ref", "batch_curr", random_state=0)

        assert result.attrs["adversarial_auc"] > 0.9

    def test_stable_distributions_low_auc(self, stable_backend):
        """Identical SHAP distributions across batches should produce AUC close to 0.5."""
        analyzer = SHAPAnalyzer(stable_backend)
        result = analyzer.compare_adversarial_batches("ref", "curr", random_state=0)

        assert result.attrs["adversarial_auc"] < 0.65

    def test_empty_ref_batch_returns_empty(self, shifted_backend):
        """Non-existent reference batch should return empty DataFrame."""
        analyzer = SHAPAnalyzer(shifted_backend)
        result = analyzer.compare_adversarial_batches(
            "nonexistent_batch", "batch_curr", random_state=0
        )
        assert result.empty

    def test_top_k_limits_rows(self, shifted_backend):
        """top_k should return only k features."""
        analyzer = SHAPAnalyzer(shifted_backend)
        result = analyzer.compare_adversarial_batches(
            "batch_ref", "batch_curr", top_k=1, random_state=0
        )
        assert len(result) == 1

    def test_top_k_invalid_raises(self, shifted_backend):
        """top_k < 1 should raise ValueError."""
        analyzer = SHAPAnalyzer(shifted_backend)
        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            analyzer.compare_adversarial_batches("batch_ref", "batch_curr", top_k=0)

    def test_invalid_sort_by_raises(self, shifted_backend):
        """Invalid sort_by should raise ValueError."""
        analyzer = SHAPAnalyzer(shifted_backend)
        with pytest.raises(ValueError, match="Invalid sort_by"):
            analyzer.compare_adversarial_batches(
                "batch_ref", "batch_curr", sort_by="nonexistent_col"
            )
