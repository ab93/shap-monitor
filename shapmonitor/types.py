"""Type definitions for shap-monitor."""

from dataclasses import dataclass
from datetime import datetime
from os import PathLike as OSPathLike
from typing import Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd

# Path types
PathLike = str | OSPathLike

# Array types for features and predictions
ArrayLike = npt.NDArray[np.floating]
DFrameLike = pd.DataFrame
PredictionValue = float | int | str


class ExplanationLike(Protocol):
    """Protocol for SHAP explanation objects.

    Any SHAP explanation object that implements the values and base_values
    attributes can be used.
    """

    @property
    def values(self) -> ArrayLike:
        """Get SHAP values."""
        ...

    @property
    def base_values(self) -> ArrayLike:
        """Get base values."""
        ...

    @property
    def shape(self) -> tuple[int, ...]:
        """Get shape of the explanation values."""
        ...


class ExplainerLike(Protocol):
    """Protocol for SHAP explainer objects.

    Any SHAP explainer (TreeExplainer, KernelExplainer, etc.) that implements
    the shap_values method can be used.
    """

    def __call__(self, X: ArrayLike) -> ExplanationLike:
        """Compute SHAP for input features."""
        ...

    def explain_row(self, X: ArrayLike) -> ExplanationLike:
        """Compute SHAP for a single input feature row."""
        ...


@dataclass(slots=True)
class ExplanationBatch:
    """A batch of explanations to be stored.

    This corresponds to one row in the Parquet backend.
    Feature-specific columns (shap_*, feat_*) are stored as dicts
    mapping feature names to arrays of values.
    """

    timestamp: datetime
    batch_id: str
    model_version: str
    n_samples: int
    base_values: ArrayLike
    shap_values: dict[str, ArrayLike]  # {feature_name: array of shap values}
    feature_values: dict[str, ArrayLike] | None = None  # {feature_name: array of values}
    predictions: ArrayLike | None = None  # optional

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the ExplanationBatch to a pandas DataFrame."""
        data = {
            "timestamp": [self.timestamp] * self.n_samples,
            "batch_id": [self.batch_id] * self.n_samples,
            "model_version": [self.model_version] * self.n_samples,
            "base_value": self.base_values,
        }

        # Add SHAP values and feature values
        for feat_name, shap_vals in self.shap_values.items():
            data[f"shap_{feat_name}"] = shap_vals

        if self.feature_values is not None:
            for feat_name, feat_vals in self.feature_values.items():
                data[f"feat_{feat_name}"] = feat_vals

        # Add predictions if available
        if self.predictions is not None:
            data["prediction"] = self.predictions

        return pd.DataFrame(data)
