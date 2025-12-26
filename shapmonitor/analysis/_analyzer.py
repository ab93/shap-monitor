import logging
from datetime import date, datetime

import numpy as np
import pandas as pd

from shapmonitor.types import Backend, DFrameLike

_logger = logging.getLogger(__name__)


class SHAPAnalyzer:
    """Analyze SHAP explanations stored in a backend.

    Provides methods for computing summary statistics, comparing time periods,
    and detecting changes in feature importance over time.

    Parameters
    ----------
    backend : Backend
        Backend for retrieving stored SHAP explanations.
    min_abs_shap : float, optional
        Minimum mean absolute SHAP value threshold (default: 0.0).
        Features below this threshold are excluded from results.
        Useful for filtering out low-impact features and reducing noise.

    Examples
    --------
    >>> backend = ParquetBackend("/path/to/shap_logs")
    >>> analyzer = SHAPAnalyzer(backend, min_abs_shap=0.01)
    >>> summary = analyzer.summary(start_date, end_date)
    """

    def __init__(self, backend: Backend, min_abs_shap: float = 0.0) -> None:
        self._backend = backend
        self._min_abs_shap = min_abs_shap

    @property
    def min_abs_shap(self) -> float:
        """Get the minimum absolute SHAP value threshold."""
        return self._min_abs_shap

    @property
    def backend(self) -> Backend:
        """Get the backend for retrieving explanations."""
        return self._backend

    def summary(
        self, start_dt: datetime | date, end_dt: datetime | date, sort_by: str = "mean_abs"
    ) -> DFrameLike:
        """Compute summary statistics for SHAP values in a date range.

        Parameters
        ----------
        start_dt : datetime | date
            Start of the date range (inclusive).
        end_dt : datetime | date
            End of the date range (inclusive).
        sort_by : str, optional
            Column to sort results by (default: 'mean_abs').
            Options: 'mean_abs', 'mean', 'std', 'min', 'max'.

        Returns
        -------
        DataFrame
            Summary statistics indexed by feature name (dtype: float32).

            Columns:
                - mean_abs: Mean of absolute SHAP values (feature importance)
                - mean: Mean SHAP value (contribution direction)
                - std: Standard deviation of SHAP values
                - min: Minimum SHAP value
                - max: Maximum SHAP value

            Attributes:
                - n_samples: Total number of samples in the date range

        Notes
        -----
        Features with mean_abs below `min_abs_shap` threshold are excluded.
        """
        df = self._backend.read(start_dt, end_dt)

        if df.empty:
            _logger.warning("No data found for date range %s to %s", start_dt, end_dt)
            return pd.DataFrame()

        shap_cols = df.filter(like="shap_")
        feature_names = [col.replace("shap_", "") for col in shap_cols.columns]

        result = (
            pd.DataFrame(
                {
                    "feature": feature_names,
                    "mean_abs": shap_cols.abs().mean(),
                    "mean": shap_cols.mean(),
                    "std": shap_cols.std(),
                    "min": shap_cols.min(),
                    "max": shap_cols.max(),
                },
            )
            .set_index("feature")
            .astype(np.float32)
        )
        result.attrs["n_samples"] = len(shap_cols)

        if self._min_abs_shap > 0.0:
            result = result[result["mean_abs"] >= self._min_abs_shap]

        if sort_by not in result.columns:
            raise ValueError(
                f"Invalid sort_by value: {sort_by}. Must be one of {list(result.columns)}"
            )

        # TODO: Add relationship correlation with target if feature values and predictions are available

        return result.sort_values("mean_abs", ascending=False)

    def compare_versions(self, *model_versions: str):
        """Compare SHAP explanations across different model versions.

        Parameters
        ----------
        model_versions : str
            Model version identifiers to compare.

        Returns
        -------
        DataFrame
            Comparison of SHAP statistics across model versions.
        """
        pass

    def compare_time_periods(
        self,
        start_1: datetime | date,
        end_1: datetime | date,
        start_2: datetime | date,
        end_2: datetime | date,
        sort_by: str = "mean_abs_1",
    ) -> DFrameLike:
        """Compare SHAP explanations between two time periods.

        Useful for detecting feature importance drift, ranking changes,
        and sign flips in model behavior over time.

        Parameters
        ----------
        start_1 : datetime | date
            Start of the first (baseline) time period.
        end_1 : datetime | date
            End of the first time period.
        start_2 : datetime | date
            Start of the second (comparison) time period.
        end_2 : datetime | date
            End of the second time period.
        sort_by : str, optional
            Column to sort results by (default: 'mean_abs_1').

        Returns
        -------
        DataFrame
            Comparison statistics indexed by feature name.

            Columns:
                - mean_abs_1, mean_abs_2: Feature importance per period
                - delta_mean_abs: Absolute change (period_2 - period_1)
                - pct_delta_mean_abs: Percentage change from period_1
                - rank_1, rank_2: Feature importance rank per period
                - delta_rank: Rank change (positive = less important)
                - rank_change: 'increased', 'decreased', or 'no_change'
                - mean_1, mean_2: Mean SHAP value (direction) per period
                - sign_flip: True if contribution direction changed

            Attributes:
                - n_samples_1: Sample count in period 1
                - n_samples_2: Sample count in period 2

        Notes
        -----
        Features with mean_abs below `min_abs_shap` threshold are excluded.
        Uses outer join, so features appearing in only one period will have NaN.
        """
        summary_df_1 = self.summary(start_1, end_1)
        summary_df_2 = self.summary(start_2, end_2)

        if summary_df_1.empty and summary_df_2.empty:
            _logger.warning("No data found for either time period")
            return pd.DataFrame()

        # Capture attrs before suffix (pandas loses attrs on most operations)
        n_samples_1 = summary_df_1.attrs.get("n_samples")
        n_samples_2 = summary_df_2.attrs.get("n_samples")

        # Rename columns with suffixes
        summary_df_1 = summary_df_1.add_suffix("_1")
        summary_df_2 = summary_df_2.add_suffix("_2")

        # Merge on index (feature name)
        comparison_df = pd.merge(
            summary_df_1, summary_df_2, left_index=True, right_index=True, how="outer"
        )
        # Delta calculations
        comparison_df["delta_mean_abs"] = comparison_df["mean_abs_2"] - comparison_df["mean_abs_1"]
        comparison_df["pct_delta_mean_abs"] = (
            comparison_df["delta_mean_abs"] / comparison_df["mean_abs_1"].replace(0, pd.NA)
        ) * 100

        # Rank calculations
        comparison_df["rank_1"] = comparison_df["mean_abs_1"].rank(ascending=False)
        comparison_df["rank_2"] = comparison_df["mean_abs_2"].rank(ascending=False)
        comparison_df["delta_rank"] = comparison_df["rank_2"] - comparison_df["rank_1"]

        conditions = [comparison_df["delta_rank"] < 0, comparison_df["delta_rank"] > 0]
        comparison_df["rank_change"] = np.select(
            conditions, ["increased", "decreased"], default="no_change"
        )

        # Vectorized sign flip calculation (NaN filled with 0 to avoid false positives)
        comparison_df["sign_flip"] = np.sign(comparison_df["mean_1"]).fillna(0) != np.sign(
            comparison_df["mean_2"]
        ).fillna(0)

        # TODO: Add relationship flip calculations

        comparison_df = comparison_df[
            [
                "mean_abs_1",
                "mean_abs_2",
                "delta_mean_abs",
                "pct_delta_mean_abs",
                "rank_1",
                "rank_2",
                "delta_rank",
                "rank_change",
                "mean_1",
                "mean_2",
                "sign_flip",
            ]
        ]
        comparison_df.attrs["n_samples_1"] = n_samples_1
        comparison_df.attrs["n_samples_2"] = n_samples_2

        if sort_by not in comparison_df.columns:
            raise ValueError(
                f"Invalid sort_by value: {sort_by}. Must be one of {list(comparison_df.columns)}"
            )

        return comparison_df.sort_values(sort_by, ascending=False)
