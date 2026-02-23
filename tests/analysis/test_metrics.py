"""Tests for shapmonitor.analysis.metrics."""

import numpy as np
import pytest

from shapmonitor.analysis.metrics import population_stability_index
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
