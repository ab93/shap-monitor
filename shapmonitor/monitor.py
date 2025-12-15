import random
import uuid
from pathlib import Path

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

    def log(self, features: ArrayLike, predictions: ArrayLike | None = None) -> None:
        """Log SHAP explanations for a batch of predictions.

        Parameters
        ----------
        features : ArrayLike
            Input features (2D array: n_samples x n_features).
        predictions : ArrayLike, optional
            Model predictions for the batch. If not provided, predictions
            will not be stored in the explanation record.

        Note
        ----
        TODO: Add support for single sample (1D array) inputs.
        """
        if not self._should_sample():
            return

        # Compute SHAP values for the batch
        shap_values = self.compute(features)

        # TODO: Write to backend (ParquetBackend)
        _ = shap_values  # Placeholder until backend is implemented

    def compute(self, features: ArrayLike) -> ExplanationLike:
        """
        Compute SHAP values for the given input features.


        Parameters
        ----------
        features : ArrayLike
            Input features for which to compute SHAP values.

        Returns
        -------
        Shap explanation object
            The SHAP explanation object containing SHAP values.
        """
        return self._explainer(features)
