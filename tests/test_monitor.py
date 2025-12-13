import numpy as np
import pytest
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from shapmonitor import SHAPMonitor
from shapmonitor.types import ExplainerLike


@pytest.fixture
def model():
    X, y = make_regression(n_samples=100, n_features=5, random_state=42)
    model = RandomForestRegressor()
    model.fit(X, y)
    return model


@pytest.fixture
def explainer(model) -> ExplainerLike:
    return shap.Explainer(model)


@pytest.fixture
def monitor(explainer, tmp_path):
    return SHAPMonitor(explainer=explainer, data_dir=tmp_path, sample_rate=0.2)


def test_monitor(monitor):
    assert isinstance(monitor, SHAPMonitor)
    assert monitor.sample_rate == 0.2
    assert monitor.data_dir.exists()
    print(monitor.explainer)

    shap_values = monitor.compute_shap(np.array([[0.5] * 5]))
    print(shap_values)
    shap.plots.beeswarm(shap_values)
