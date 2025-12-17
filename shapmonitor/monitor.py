import random
import uuid
from pathlib import Path

import pandas as pd

from shapmonitor.types import PathLike, ExplainerLike, ArrayLike, ExplanationLike


class SHAPMonitor:
    """Monitor SHAP explanations over time.

    Parameters
    ----------
    explainer : ExplainerLike
        A SHAP explainer object that implements the shap_values method.
    data_dir : PathLike
        Directory to store explanation logs.
    sample_rate : float, optional
        Fraction of predictions to log explanations for (default is 0.1).
    model_version : str, optional
        Version identifier for the model (default is "unknown").
    feature_names : list[str], optional
        Names of the features in the input data.
    """

    def __init__(
        self,
        explainer: ExplainerLike,
        data_dir: PathLike,
        sample_rate: float = 0.1,
        model_version: str = "unknown",
        feature_names: list[str] | None = None,
    ) -> None:
        self._explainer = explainer
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._sample_rate = sample_rate
        self._model_version = model_version
        self._feature_names = feature_names

    @property
    def explainer(self) -> ExplainerLike:
        """Get the SHAP explainer object."""
        return self._explainer

    @property
    def data_dir(self) -> Path:
        """Get the data directory path."""
        return self._data_dir

    @property
    def sample_rate(self) -> float:
        """Get the sample rate for logging explanations."""
        return self._sample_rate

    @property
    def model_version(self) -> str:
        """Get the model version identifier."""
        return self._model_version

    @property
    def feature_names(self) -> list[str] | None:
        """Get the feature names."""
        return self._feature_names

    def _should_sample(self) -> bool:
        """Determine if current prediction should be sampled."""
        return random.random() < self._sample_rate

    @staticmethod
    def _generate_batch_id() -> str:
        """Generate a unique batch ID."""
        return str(uuid.uuid4())

    def log(self, X: ArrayLike, y: ArrayLike | None = None) -> None:
        """Log SHAP explanations for a single prediction.

        Parameters
        ----------
        X : ArrayLike
            Input features (1D array: n_features).
        y : ArrayLike, optional
            Model prediction for the input. If not provided, prediction
            will not be stored in the explanation record.

        """
        if self._feature_names and isinstance(X, pd.Series):
            self._feature_names = X.index.tolist()

        if not self._should_sample():
            return

        # Compute SHAP values for the single prediction
        _ = self._explainer.explain_row()

    def log_batch(self, X: ArrayLike, y: ArrayLike | None = None) -> None:
        """Log SHAP explanations for a batch of predictions.

        Parameters
        ----------
        X : ArrayLike
            Input features (2D array: n_samples x n_features).
        y : ArrayLike, optional
            Model predictions for the batch. If not provided, predictions
            will not be stored in the explanation record.

        Note
        ----
        TODO: Add support for single sample (1D array) inputs.
        """
        if not self._feature_names and isinstance(X, pd.DataFrame):
            self._feature_names = X.columns.tolist()

        if not self._should_sample():
            return

        # Compute SHAP values for the batch
        shap_values = self.compute(X)

        # TODO: Write to backend (ParquetBackend)
        _ = shap_values  # Placeholder until backend is implemented

    def compute(self, X: ArrayLike) -> ExplanationLike:
        """
        Compute SHAP values for the given input features.


        Parameters
        ----------
        X : ArrayLike
            Input features for which to compute SHAP values.

        Returns
        -------
        Shap explanation object
            The SHAP explanation object containing SHAP values.
        """
        return self._explainer(X)


if __name__ == "__main__":
    import shap
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.datasets import load_diabetes

    # Load sample data
    data = load_diabetes(as_frame=True)
    X, y = data.data, data.target

    print(type(X), X.shape)

    # Train a sample model
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)

    # Create a SHAP explainer
    explainer_ = shap.TreeExplainer(model)

    # Initialize SHAPMonitor
    monitor = SHAPMonitor(
        explainer=explainer_,
        data_dir="./shap_logs",
        sample_rate=0.5,
        model_version="rf-v1",
    )

    explanations = monitor.compute(X)
    print("SHAP values shape:", explanations.shape)

    # Log SHAP explanations for the training data
    monitor.log_batch(X, y=model.predict(X))

    monitor.log_batch(X.iloc[0])
