import logging
import uuid
from datetime import datetime

import numpy as np
import pandas as pd

from shapmonitor.backends import BackendFactory
from shapmonitor.types import (
    PathLike,
    ExplainerLike,
    ArrayLike,
    ExplanationLike,
    Backend,
    ExplanationBatch,
)

_logger = logging.getLogger(__name__)


# TODO: Add support for single sample (1D array) inputs.
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
    backend : Backend, optional
        Backend for storing explanations (default is None).

    Raises
    ------
    ValueError
        If neither data_dir nor backend is provided or if both are provided.
    """

    def __init__(
        self,
        explainer: ExplainerLike,
        data_dir: PathLike | None = None,
        sample_rate: float = 0.1,
        model_version: str = "unknown",
        feature_names: list[str] | None = None,
        backend: Backend | None = None,
        random_seed: int | None = 42,
    ) -> None:
        self._explainer = explainer

        if data_dir is None and backend is None:
            raise ValueError("Either data_dir or backend must be provided.")

        if data_dir and backend:
            raise ValueError("Provide only one of data_dir or backend, not both.")

        if data_dir:
            self._backend = BackendFactory.get_backend("parquet", file_dir=data_dir)
        else:
            self._backend = backend

        self._sample_rate = sample_rate
        self._model_version = model_version
        self._feature_names = feature_names
        self._rng = np.random.default_rng(random_seed)

    @property
    def explainer(self) -> ExplainerLike:
        """Get the SHAP explainer object."""
        return self._explainer

    @property
    def backend(self) -> Backend:
        """Get the backend for storing explanations."""
        return self._backend

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

    @staticmethod
    def _generate_batch_id() -> str:
        """Generate a unique batch ID."""
        return str(uuid.uuid4())

    def log_shap(self, shap_values: ExplanationLike, batch_id: str | None = None) -> None:
        """Log SHAP explanations for a single batch.

        Parameters
        ----------
        shap_values : ExplanationLike
            SHAP explanation object containing SHAP values and base values.
        batch_id : str, optional
            Unique identifier for the batch. If not provided, a new UUID
            will be generated.
        """
        if not self._feature_names:
            self._feature_names = [f"feat_{i}" for i in range(shap_values.values.shape[1])]

        if not batch_id:
            batch_id = self._generate_batch_id()

        explanation_batch = ExplanationBatch(
            timestamp=datetime.now(),
            batch_id=batch_id,
            model_version=self._model_version,
            n_samples=shap_values.values.shape[0],
            base_values=shap_values.base_values,
            shap_values={
                feat: shap_values.values[:, idx] for idx, feat in enumerate(self._feature_names)
            },
            feature_values=None,  # Feature values are not stored in this method
            predictions=None,  # Predictions are not stored in this method
        )

        path = self._backend.write(explanation_batch)
        _logger.info("Logged SHAP explanations for batch_id: %s in path: %s", batch_id, path)

    def log_batch(
        self, X: ArrayLike, y: ArrayLike | None = None, batch_id: str | None = None
    ) -> None:
        """Log SHAP explanations for a batch of predictions.

        Parameters
        ----------
        X : ArrayLike
            Input features (2D array: n_samples x n_features).
        y : ArrayLike, optional
            Model predictions for the batch. If not provided, predictions
            will not be stored in the explanation record.
        batch_id : str, optional
            Unique identifier for the batch. If not provided, a new UUID
            will be generated.
        """
        if not self._feature_names:
            if isinstance(X, pd.DataFrame):
                self._feature_names = X.columns.tolist()
            else:
                self._feature_names = [f"feat_{i}" for i in range(X.shape[1])]

        # Sample the data
        n_samples = max(1, int(len(X) * self._sample_rate))
        sample_indices = self._rng.choice(len(X), size=n_samples, replace=False)

        if isinstance(X, pd.DataFrame):
            X = X.iloc[sample_indices].reset_index(drop=True)
        else:
            X = np.asarray(X)
            X = X[sample_indices]

        # Sample y to match X if provided
        if y is not None:
            y = np.asarray(y)
            y = y[sample_indices]

        if not batch_id:
            batch_id = self._generate_batch_id()

        # Compute SHAP values for the batch
        explanations = self.compute(X)

        shap_values_dict = {
            feat: explanations.values[:, idx] for idx, feat in enumerate(self._feature_names)
        }
        if isinstance(X, pd.DataFrame):
            feat_values_dict = {
                feat: X.iloc[:, idx].to_numpy() for idx, feat in enumerate(self._feature_names)
            }
        else:
            feat_values_dict = {feat: X[:, idx] for idx, feat in enumerate(self._feature_names)}

        explanation_batch = ExplanationBatch(
            timestamp=datetime.now(),
            batch_id=batch_id,
            model_version=self._model_version,
            n_samples=len(X),
            base_values=explanations.base_values,
            shap_values=shap_values_dict,
            feature_values=feat_values_dict,
            predictions=y,
        )

        path = self._backend.write(explanation_batch)
        _logger.info("Logged SHAP explanations for batch_id: %s in path: %s", batch_id, path)

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
