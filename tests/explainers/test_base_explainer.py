"""Tests for SHAP base Explainer (universal/auto explainer)."""

import numpy as np
import pytest
import shap
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

from shapmonitor import SHAPMonitor


class TestBaseExplainerWithTreeModels:
    """Tests for base Explainer auto-detecting tree-based models."""

    @pytest.fixture
    def rf_regressor(self, regression_data):
        X, y = regression_data
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        return model

    @pytest.fixture
    def rf_classifier(self, classification_data):
        X, y = classification_data
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        return model

    def test_auto_explainer_rf_regressor(self, rf_regressor, sample_features):
        """Base Explainer should auto-detect RandomForestRegressor."""
        explainer = shap.Explainer(rf_regressor)
        explanation = explainer(sample_features)

        assert hasattr(explanation, "values")
        assert hasattr(explanation, "base_values")
        assert explanation.values.shape == sample_features.shape

    def test_auto_explainer_rf_classifier(self, rf_classifier, sample_features):
        """Base Explainer should auto-detect RandomForestClassifier."""
        explainer = shap.Explainer(rf_classifier)
        explanation = explainer(sample_features)

        assert hasattr(explanation, "values")
        assert hasattr(explanation, "base_values")

    def test_monitor_with_auto_explainer(self, rf_regressor, tmp_path, sample_features):
        """SHAPMonitor should work with auto Explainer."""
        explainer = shap.Explainer(rf_regressor)
        monitor = SHAPMonitor(
            explainer=explainer,
            data_dir=tmp_path,
            sample_rate=1.0,
            model_version="auto-v1",
        )

        explanation = monitor.compute(sample_features)
        assert hasattr(explanation, "values")
        assert explanation.values.shape == sample_features.shape


class TestBaseExplainerWithLinearModels:
    """Tests for base Explainer with linear models."""

    @pytest.fixture
    def linear_model(self, regression_data):
        X, y = regression_data
        model = LinearRegression()
        model.fit(X, y)
        return model, X

    @pytest.fixture
    def logistic_model(self, classification_data):
        X, y = classification_data
        model = LogisticRegression(random_state=42, max_iter=200)
        model.fit(X, y)
        return model, X

    def test_auto_explainer_linear_regression(self, linear_model, sample_features):
        """Base Explainer should work with LinearRegression."""
        model, X_train = linear_model
        # For linear models, we need to provide masker/background data
        explainer = shap.Explainer(model, X_train)
        explanation = explainer(sample_features)

        assert hasattr(explanation, "values")
        assert hasattr(explanation, "base_values")

    def test_auto_explainer_logistic_regression(self, logistic_model, sample_features):
        """Base Explainer should work with LogisticRegression."""
        model, X_train = logistic_model
        explainer = shap.Explainer(model, X_train)
        explanation = explainer(sample_features)

        assert hasattr(explanation, "values")
        assert hasattr(explanation, "base_values")


class TestExplanationProperties:
    """Tests for SHAP explanation object properties."""

    @pytest.fixture
    def explainer_and_model(self, regression_data):
        X, y = regression_data
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        explainer = shap.Explainer(model)
        return explainer, model

    def test_explanation_has_values(self, explainer_and_model, sample_features):
        """Explanation should have values attribute."""
        explainer, _ = explainer_and_model
        explanation = explainer(sample_features)

        assert hasattr(explanation, "values")
        assert isinstance(explanation.values, np.ndarray)

    def test_explanation_has_base_values(self, explainer_and_model, sample_features):
        """Explanation should have base_values attribute."""
        explainer, _ = explainer_and_model
        explanation = explainer(sample_features)

        assert hasattr(explanation, "base_values")

    def test_explanation_has_data(self, explainer_and_model, sample_features):
        """Explanation should have data attribute (input features)."""
        explainer, _ = explainer_and_model
        explanation = explainer(sample_features)

        assert hasattr(explanation, "data")
        np.testing.assert_array_equal(explanation.data, sample_features)

    def test_values_shape_matches_input(self, explainer_and_model, sample_features):
        """SHAP values shape should match input features shape."""
        explainer, _ = explainer_and_model
        explanation = explainer(sample_features)

        assert explanation.values.shape == sample_features.shape

    def test_shap_additivity(self, explainer_and_model, sample_features):
        """SHAP values + base_value should equal prediction."""
        explainer, model = explainer_and_model
        explanation = explainer(sample_features)
        predictions = model.predict(sample_features)

        for i in range(len(sample_features)):
            shap_sum = explanation.values[i].sum() + explanation.base_values[i]
            np.testing.assert_almost_equal(shap_sum, predictions[i], decimal=4)
