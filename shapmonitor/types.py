"""Type definitions for shap-monitor."""

from dataclasses import dataclass
from datetime import datetime
from os import PathLike as OSPathLike
from typing import Protocol

import numpy as np
import numpy.typing as npt

# Path types
PathLike = str | OSPathLike

# Array types for features and predictions
ArrayLike = npt.NDArray[np.floating]
PredictionValue = float | int | npt.NDArray[np.floating]


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


class ExplainerLike(Protocol):
    """Protocol for SHAP explainer objects.

    Any SHAP explainer (TreeExplainer, KernelExplainer, etc.) that implements
    the shap_values method can be used.
    """

    def __call__(self, X: ArrayLike) -> ExplanationLike:
        """Compute SHAP for input features."""
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
    feature_values: dict[str, ArrayLike]  # {feature_name: array of values}
    predictions: ArrayLike | None = None  # optional
