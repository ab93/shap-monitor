"""Tests for shapmonitor.analysis.metrics."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from shapmonitor.analysis.metrics import adversarial_auc, population_stability_index
from shapmonitor.exceptions import InvalidShapeError


class TestPSI:
    """Tests for population_stability_index."""

    def test_identical_distributions_returns_zero(self):
        """PSI should be 0 for identical distributions."""
        data = np.random.default_rng(0).normal(0, 1, 500)
        psi = population_stability_index(data, data)
        assert psi == pytest.approx(0.0, abs=1e-6)

    def test_buckets_less_than_two_raises(self):
        """PSI should raise ValueError when buckets < 2."""
        data = np.ones(100)
        with pytest.raises(ValueError, match="at least 2"):
            population_stability_index(data, data, buckets=1)

    def test_empty_reference_raises(self):
        """PSI should raise ValueError for empty reference distribution."""
        with pytest.raises(ValueError, match="must not be empty"):
            population_stability_index(np.array([]), np.ones(10))

    def test_empty_current_raises(self):
        """PSI should raise ValueError for empty current distribution."""
        with pytest.raises(ValueError, match="must not be empty"):
            population_stability_index(np.ones(10), np.array([]))

    def test_2d_reference_raises(self):
        """PSI should raise InvalidShapeError for 2-D reference input."""
        with pytest.raises(InvalidShapeError):
            population_stability_index(np.ones((10, 2)), np.ones(10))

    def test_2d_current_raises(self):
        """PSI should raise InvalidShapeError for 2-D current input."""
        with pytest.raises(InvalidShapeError):
            population_stability_index(np.ones(10), np.ones((10, 2)))


class TestAdversarialAUC:
    """Tests for adversarial_auc."""

    @pytest.fixture
    def clearly_different(self):
        """Two DataFrames with clearly separable distributions."""
        rng = np.random.default_rng(0)
        ref = pd.DataFrame({"f0": rng.normal(0, 0.1, 300), "f1": rng.normal(0, 0.1, 300)})
        curr = pd.DataFrame({"f0": rng.normal(5, 0.1, 300), "f1": rng.normal(5, 0.1, 300)})
        return ref, curr

    @pytest.fixture
    def identical(self):
        """Two DataFrames drawn from the same distribution."""
        rng = np.random.default_rng(1)
        data = rng.normal(0, 1, (600, 2))
        ref = pd.DataFrame(data[:300], columns=["f0", "f1"])
        curr = pd.DataFrame(data[300:], columns=["f0", "f1"])
        return ref, curr

    def test_different_distributions_high_auc(self, clearly_different):
        """Clearly different distributions should produce AUC > 0.9."""
        ref, curr = clearly_different
        auc, _ = adversarial_auc(ref, curr, random_state=0)
        assert auc > 0.9

    def test_identical_distributions_low_auc(self, identical):
        """Same distribution should produce AUC close to 0.5."""
        ref, curr = identical
        auc, _ = adversarial_auc(ref, curr, random_state=0)
        assert auc < 0.65

    def test_returns_feature_importances_indexed_correctly(self, clearly_different):
        """feature_importances should be a Series indexed by column names."""
        ref, curr = clearly_different
        _, importances = adversarial_auc(ref, curr, random_state=0)
        assert isinstance(importances, pd.Series)
        assert set(importances.index) == {"f0", "f1"}

    def test_feature_importances_sum_to_one(self, clearly_different):
        """RandomForest feature importances should sum to 1."""
        ref, curr = clearly_different
        _, importances = adversarial_auc(ref, curr, random_state=0)
        assert importances.sum() == pytest.approx(1.0, abs=1e-6)

    def test_empty_reference_raises(self, clearly_different):
        """Empty reference should raise ValueError."""
        _, curr = clearly_different
        with pytest.raises(ValueError, match="must not be empty"):
            adversarial_auc(pd.DataFrame(), curr)

    def test_empty_current_raises(self, clearly_different):
        """Empty current should raise ValueError."""
        ref, _ = clearly_different
        with pytest.raises(ValueError, match="must not be empty"):
            adversarial_auc(ref, pd.DataFrame())

    def test_cv_less_than_two_raises(self, clearly_different):
        """cv < 2 should raise ValueError."""
        ref, curr = clearly_different
        with pytest.raises(ValueError, match="at least 2"):
            adversarial_auc(ref, curr, cv=1)

    def test_no_common_columns_raises(self):
        """DataFrames with no overlapping columns should raise ValueError."""
        ref = pd.DataFrame({"a": np.ones(50)})
        curr = pd.DataFrame({"b": np.ones(50)})
        with pytest.raises(ValueError, match="no common columns"):
            adversarial_auc(ref, curr)

    def test_uses_only_common_columns(self):
        """Only common columns should be used; extra columns are ignored."""
        rng = np.random.default_rng(0)
        ref = pd.DataFrame({"shared": rng.normal(0, 0.1, 100), "extra_ref": rng.normal(0, 1, 100)})
        curr = pd.DataFrame(
            {"shared": rng.normal(5, 0.1, 100), "extra_curr": rng.normal(0, 1, 100)}
        )
        auc, importances = adversarial_auc(ref, curr, random_state=0)
        assert "shared" in importances.index
        assert "extra_ref" not in importances.index
        assert "extra_curr" not in importances.index

    def test_custom_classifier_accepted(self, clearly_different):
        """A custom sklearn classifier should be accepted via classifier=."""
        ref, curr = clearly_different
        # LogisticRegression doesn't have feature_importances_, so expect AttributeError
        with pytest.raises(AttributeError, match="feature_importances_"):
            adversarial_auc(ref, curr, classifier=LogisticRegression())
