"""Tests for SHAP LinearExplainer with various linear models."""

import numpy as np
import pytest
import shap
from sklearn.linear_model import (
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)

from shapmonitor import SHAPMonitor


class TestLinearExplainerRegression:
    """Tests for LinearExplainer with linear regression models."""

    @pytest.fixture
    def linear_model(self, regression_data):
        X, y = regression_data
        model = LinearRegression()
        model.fit(X, y)
        return model, X

    @pytest.fixture
    def ridge_model(self, regression_data):
        X, y = regression_data
        model = Ridge(alpha=1.0, random_state=42)
        model.fit(X, y)
        return model, X

    @pytest.fixture
    def lasso_model(self, regression_data):
        X, y = regression_data
        model = Lasso(alpha=0.1, random_state=42)
        model.fit(X, y)
        return model, X

    def test_linear_regression_explainer(self, linear_model, sample_features):
        """LinearExplainer should work with LinearRegression."""
        model, X_train = linear_model
        explainer = shap.LinearExplainer(model, X_train)
        explanation = explainer(sample_features)

        assert hasattr(explanation, "values")
        assert hasattr(explanation, "base_values")
        assert explanation.values.shape == sample_features.shape

    def test_ridge_explainer(self, ridge_model, sample_features):
        """LinearExplainer should work with Ridge regression."""
        model, X_train = ridge_model
        explainer = shap.LinearExplainer(model, X_train)
        explanation = explainer(sample_features)

        assert hasattr(explanation, "values")
        assert explanation.values.shape == sample_features.shape

    def test_lasso_explainer(self, lasso_model, sample_features):
        """LinearExplainer should work with Lasso regression."""
        model, X_train = lasso_model
        explainer = shap.LinearExplainer(model, X_train)
        explanation = explainer(sample_features)

        assert hasattr(explanation, "values")
        assert explanation.values.shape == sample_features.shape

    def test_monitor_with_linear_explainer(self, linear_model, tmp_path, sample_features):
        """SHAPMonitor should work with LinearExplainer."""
        model, X_train = linear_model
        explainer = shap.LinearExplainer(model, X_train)
        monitor = SHAPMonitor(
            explainer=explainer,
            data_dir=tmp_path,
            sample_rate=1.0,
            model_version="linear-v1",
        )

        explanation = monitor.compute(sample_features)
        assert hasattr(explanation, "values")
        assert explanation.values.shape == sample_features.shape

    def test_shap_values_sum_to_prediction_diff(self, linear_model, sample_features):
        """SHAP values should sum to (prediction - base_value) for linear models."""
        model, X_train = linear_model
        explainer = shap.LinearExplainer(model, X_train)
        explanation = explainer(sample_features)
        predictions = model.predict(sample_features)

        for i in range(len(sample_features)):
            shap_sum = explanation.values[i].sum() + explanation.base_values[i]
            np.testing.assert_almost_equal(shap_sum, predictions[i], decimal=4)


class TestLinearExplainerClassification:
    """Tests for LinearExplainer with logistic regression."""

    @pytest.fixture
    def logistic_model(self, classification_data):
        X, y = classification_data
        model = LogisticRegression(random_state=42, max_iter=200)
        model.fit(X, y)
        return model, X

    def test_logistic_regression_explainer(self, logistic_model, sample_features):
        """LinearExplainer should work with LogisticRegression."""
        model, X_train = logistic_model
        explainer = shap.LinearExplainer(model, X_train)
        explanation = explainer(sample_features)

        assert hasattr(explanation, "values")
        assert hasattr(explanation, "base_values")

    def test_monitor_with_logistic_explainer(self, logistic_model, tmp_path, sample_features):
        """SHAPMonitor should work with LogisticRegression explainer."""
        model, X_train = logistic_model
        explainer = shap.LinearExplainer(model, X_train)
        monitor = SHAPMonitor(
            explainer=explainer,
            data_dir=tmp_path,
            sample_rate=1.0,
        )

        explanation = monitor.compute(sample_features)
        assert hasattr(explanation, "values")
