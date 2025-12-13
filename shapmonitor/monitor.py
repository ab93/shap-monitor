from pathlib import Path

from shapmonitor.types import PathLike, ExplainerLike, ArrayLike


class SHAPMonitor:
    """ "
    Monitor SHAP explanations over time.

    Parameters
    ----------
    explainer : ExplainerLike
        A SHAP explainer object that implements the shap_values method.
    data_dir : PathLike
        Directory to store explanation logs.
    sample_rate : float, optional
        Fraction of predictions to log explanations for (default is 0.1).
    """

    def __init__(
        self, explainer: ExplainerLike, data_dir: PathLike, sample_rate: float = 0.1
    ) -> None:
        self._explainer = explainer
        self._data_dir = Path(data_dir)
        self._sample_rate = sample_rate

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

    def log(self):
        """Log SHAP explanations for a batch of predictions."""
        pass

    def compute_shap(self, X: ArrayLike) -> ArrayLike:
        """
        Compute SHAP values for the given input features.


        Parameters
        ----------
        X : ArrayLike
            Input features for which to compute SHAP values.

        Returns
        -------
        ArrayLike
            Computed SHAP values.
        """
        return self._explainer(X)
