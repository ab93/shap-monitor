"""Tests for SHAP TreeExplainer with various tree-based models."""

import numpy as np
import pytest
import shap
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)

from shapmonitor import SHAPMonitor


class TestTreeExplainerRandomForestRegressor:
    """Tests for TreeExplainer with RandomForestRegressor."""

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

    def test_explainer_returns_explanation(self, explainer, sample_features):
        """TreeExplainer should return valid explanation object."""
        explanation = explainer(sample_features)

        assert hasattr(explanation, "values")
        assert hasattr(explanation, "base_values")
        assert explanation.values.shape == sample_features.shape

    def test_monitor_compute(self, monitor, sample_features):
        """Monitor.compute should return valid SHAP explanation."""
        explanation = monitor.compute(sample_features)

        assert hasattr(explanation, "values")
        assert hasattr(explanation, "base_values")
        assert explanation.values.shape == sample_features.shape

    def test_shap_values_sum_to_prediction_diff(self, model, explainer, sample_features):
        """SHAP values should sum to (prediction - base_value)."""
        explanation = explainer(sample_features)
        predictions = model.predict(sample_features)

        for i in range(len(sample_features)):
            shap_sum = explanation.values[i].sum() + explanation.base_values[i]
            np.testing.assert_almost_equal(shap_sum, predictions[i], decimal=4)


class TestTreeExplainerRandomForestClassifier:
    """Tests for TreeExplainer with RandomForestClassifier."""

    @pytest.fixture
    def model(self, classification_data):
        X, y = classification_data
        model = RandomForestClassifier(n_estimators=10, random_state=42)
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
            feature_names=feature_names,
        )

    def test_explainer_returns_explanation(self, explainer, sample_features):
        """TreeExplainer should return valid explanation for classifier."""
        explanation = explainer(sample_features)

        assert hasattr(explanation, "values")
        assert hasattr(explanation, "base_values")

    def test_monitor_compute(self, monitor, sample_features):
        """Monitor.compute should work with classifier explainer."""
        explanation = monitor.compute(sample_features)

        assert hasattr(explanation, "values")
        assert hasattr(explanation, "base_values")


class TestTreeExplainerGradientBoosting:
    """Tests for TreeExplainer with GradientBoosting models."""

    @pytest.fixture
    def regressor(self, regression_data):
        X, y = regression_data
        model = GradientBoostingRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        return model

    @pytest.fixture
    def classifier(self, classification_data):
        X, y = classification_data
        model = GradientBoostingClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        return model

    def test_gradient_boosting_regressor(self, regressor, sample_features):
        """TreeExplainer should work with GradientBoostingRegressor."""
        explainer = shap.TreeExplainer(regressor)
        explanation = explainer(sample_features)

        assert hasattr(explanation, "values")
        assert explanation.values.shape == sample_features.shape

    def test_gradient_boosting_classifier(self, classifier, sample_features):
        """TreeExplainer should work with GradientBoostingClassifier."""
        explainer = shap.TreeExplainer(classifier)
        explanation = explainer(sample_features)

        assert hasattr(explanation, "values")
