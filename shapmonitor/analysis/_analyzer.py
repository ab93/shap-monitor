import logging
from datetime import date, datetime

import pandas as pd

from shapmonitor.types import Backend, DFrameLike

_logger = logging.getLogger(__name__)


class SHAPAnalyzer:
    """Analyzer for SHAP explanations.

    This class provides methods to analyze SHAP explanations stored in a backend.
    It can compute summary statistics, visualize feature importance, and
    generate reports based on the SHAP values.

    Parameters
    ----------
    backend : Backend
        Backend for retrieving SHAP explanations.

    """

    def __init__(self, backend: Backend) -> None:
        self._backend = backend

    @property
    def backend(self) -> Backend:
        """Get the backend for retrieving explanations."""
        return self._backend

    def summary(self, start_dt: datetime | date, end_dt: datetime | date) -> DFrameLike:
        """Compute summary statistics for SHAP values in a date range.

        Parameters
        ----------
        start_dt : datetime | date
            Start of the date range.
        end_dt : datetime | date
            End of the date range.

        Returns
        -------
        DataFrame
            Summary statistics indexed by feature name, sorted by importance (mean_abs).
            Columns: mean_abs, mean, std, min, max
            Attributes: n_samples (number of samples in the date range)
        """
        df = self._backend.read(start_dt, end_dt)

        if df.empty:
            _logger.warning("No data found for date range %s to %s", start_dt, end_dt)
            return pd.DataFrame()

        shap_cols = df.filter(like="shap_")
        feature_names = [col.replace("shap_", "") for col in shap_cols.columns]

        result = pd.DataFrame(
            {
                "feature": feature_names,
                "mean_abs": shap_cols.abs().mean(),
                "mean": shap_cols.mean(),
                "std": shap_cols.std(),
                "min": shap_cols.min(),
                "max": shap_cols.max(),
            },
        ).set_index("feature")
        result.attrs["n_samples"] = len(shap_cols)

        return result.sort_values("mean_abs", ascending=False)

    def compare_versions(self):
        pass

    def compare_time_periods(
        self,
        start1: datetime | date,
        end1: datetime | date,
        start2: datetime | date,
        end2: datetime | date,
    ):
        pass

    def generate_report(self):
        pass
