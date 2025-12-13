"""Type definitions for shap-monitor."""

from dataclasses import dataclass
from datetime import datetime
from os import PathLike as OSPathLike
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

# Path types
PathLike = str | OSPathLike

# Array types for features and predictions
ArrayLike = npt.NDArray[np.floating] | list[float]
PredictionValue = float | int | npt.NDArray[np.floating]


class ExplainerLike(Protocol):
    """Protocol for SHAP explainer objects.

    Any SHAP explainer (TreeExplainer, KernelExplainer, etc.) that implements
    the shap_values method can be used.
    """

    def shap_values(self, X: Any, **kwargs) -> Any:
        """Compute SHAP values for input features."""
        ...


@dataclass
class ExplanationRecord:
    """A single explanation record to be stored.

    This corresponds to one row in the Parquet backend.
    """

    timestamp: datetime
    prediction: PredictionValue
    shap_values: npt.NDArray[np.floating]
    feature_values: npt.NDArray[np.floating]
    base_value: float
    model_version: str
    sample_id: str
