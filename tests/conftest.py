"""Shared pytest fixtures for shap-monitor tests."""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression


@pytest.fixture
def regression_data():
    """Generate regression dataset."""
    X, y = make_regression(n_samples=100, n_features=5, random_state=42)
    return X, y


@pytest.fixture
def classification_data():
    """Generate binary classification dataset."""
    X, y = make_classification(
        n_samples=100, n_features=5, n_informative=3, n_redundant=1, random_state=42
    )
    return X, y


@pytest.fixture
def feature_names():
    """Feature names for test datasets."""
    return ["feat_0", "feat_1", "feat_2", "feat_3", "feat_4"]


@pytest.fixture
def sample_features():
    """Single batch of features for testing."""
    return np.array([[0.5, 0.3, -0.2, 0.1, 0.8], [0.1, -0.5, 0.4, 0.2, -0.3]])
