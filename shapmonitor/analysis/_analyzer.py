import logging
from datetime import date, datetime

import numpy as np
import pandas as pd

from shapmonitor.analysis.metrics import population_stability_index
from shapmonitor.types import Backend, DFrameLike, Period, SeriesLike

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
    >>> from shapmonitor.backends import ParquetBackend
    >>> from shapmonitor.analysis import SHAPAnalyzer
    >>> ...
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

    def fetch_shap_values(self, **kwargs) -> DFrameLike:
        """Fetch raw SHAP values from the backend within a date range.

        Parameters
        ----------
        kwargs: Backend read parameters

        Returns
        -------
        DataFrame
            Raw SHAP values indexed by timestamp.
        """
        df = self._backend.read(**kwargs)

        if df.empty:
            _logger.warning("No data found for kwargs: %s", kwargs)
            return pd.DataFrame()

        return df.filter(like="shap_")

    def _construct_summary(self, shap_df: DFrameLike) -> DFrameLike:
        result = (
            pd.DataFrame(
                {
                    "feature": shap_df.columns,
                    "mean_abs": shap_df.abs().mean(),
                    "mean": shap_df.mean(),
                    "std": shap_df.std(),
                    "min": shap_df.min(),
                    "max": shap_df.max(),
                },
            )
            .set_index("feature")
            .astype(np.float32)
        )
        result.attrs["n_samples"] = len(shap_df)
        if self._min_abs_shap > 0.0:
            result = result[result["mean_abs"] >= self._min_abs_shap]
        return result

    def summary(
        self,
        start_dt: datetime | date | None = None,
        end_dt: datetime | date | None = None,
        batch_id: str | None = None,
        model_version: str | None = None,
        sort_by: str = "mean_abs",
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
        shap_df = self.fetch_shap_values(
            start_dt=start_dt, end_dt=end_dt, batch_id=batch_id, model_version=model_version
        ).rename(columns=lambda col: col.replace("shap_", ""))
        result = self._construct_summary(shap_df)

        if sort_by not in result.columns:
            raise ValueError(
                f"Invalid sort_by value: {sort_by}. Must be one of {list(result.columns)}"
            )

        # TODO: Add relationship correlation with target if feature values and predictions are available

        return result.sort_values("mean_abs", ascending=False)

    @staticmethod
    def _calculate_psi(shap_df_ref: DFrameLike, shap_df_curr: DFrameLike) -> SeriesLike:
        common_features = shap_df_ref.columns.intersection(shap_df_curr.columns)

        psi = np.zeros(len(common_features))
        for i, feature in enumerate(common_features):
            psi[i] = population_stability_index(shap_df_ref[feature], shap_df_curr[feature])
        return pd.Series(psi, index=common_features, name="psi")

    def _compare_shap_dataframes(
        self,
        shap_df_ref: DFrameLike,
        shap_df_curr: DFrameLike,
        sort_by: str = "psi",
    ) -> DFrameLike:
        """Compare two SHAP DataFrames and compute comparison statistics.

        Parameters
        ----------
        shap_df_ref : DFrameLike
            Reference SHAP values DataFrame.
        shap_df_curr : DFrameLike
            Current SHAP values DataFrame.
        sort_by : str, optional
            Column to sort results by (default: 'psi').

        Returns
        -------
        DFrameLike
            Comparison statistics indexed by feature name.
        """
        if shap_df_ref.empty and shap_df_curr.empty:
            _logger.warning("No data found for either period")
            return pd.DataFrame()

        # Calculate PSI
        psi = self._calculate_psi(shap_df_ref, shap_df_curr)

        # Calculate summaries
        summary_df_1 = self._construct_summary(shap_df_ref)
        summary_df_2 = self._construct_summary(shap_df_curr)

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

        # Sign flip calculation (NaN filled with 0 to avoid false positives)
        comparison_df["sign_flip"] = np.sign(comparison_df["mean_1"]).fillna(0) != np.sign(
            comparison_df["mean_2"]
        ).fillna(0)

        # Add PSI calculation
        comparison_df = comparison_df.join(psi, how="left").fillna({"psi": np.nan})

        # TODO: Add relationship flip calculations

        comparison_df = comparison_df[
            [
                "psi",
                "mean_abs_1",
                "mean_abs_2",
                "delta_mean_abs",
                "pct_delta_mean_abs",
                "mean_1",
                "mean_2",
                "rank_1",
                "rank_2",
                "delta_rank",
                "rank_change",
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

    def compare_time_periods(
        self,
        period_ref: Period,
        period_curr: Period,
        sort_by: str = "psi",
    ) -> DFrameLike:
        """Compare SHAP explanations between two time periods.

        Useful for detecting feature importance drift, ranking changes,
        and sign flips in model behavior over time.

        Parameters
        ----------
        period_ref : Period
            Tuple of (start_dt, end_dt) defining the reference date range (both inclusive).
        period_curr : Period
            Tuple of (start_dt, end_dt) defining the current date range (both inclusive).
        sort_by : str, optional
            Column to sort results by (default: 'mean_abs_1').

        Returns
        -------
        DataFrame
            Comparison statistics indexed by feature name.

            Columns:
                - psi: Population Stability Index between periods
                - mean_abs_1, mean_abs_2: Feature importance per period
                - delta_mean_abs: Absolute change (period_2 - period_1)
                - pct_delta_mean_abs: Percentage change from period_1
                - mean_1, mean_2: Mean SHAP value (direction) per period
                - rank_1, rank_2: Feature importance rank per period
                - delta_rank: Rank change (positive = less important)
                - rank_change: 'increased', 'decreased', or 'no_change'
                - sign_flip: True if contribution direction changed

            Attributes:
                - n_samples_1: Sample count in period 1
                - n_samples_2: Sample count in period 2

        Notes
        -----
        Features with mean_abs below `min_abs_shap` threshold are excluded.
        Uses outer join, so features appearing in only one period will have NaN.

        Below is a guideline for interpreting PSI values:

          | PSI Value  | Interpretation              |
          |------------|-----------------------------|
          | 0          | Identical distributions     |
          | < 0.1      | No significant shift        |
          | 0.1 - 0.25 | Moderate shift, investigate |
          | 0.25 - 0.5 | Significant shift           |
          | > 0.5      | Severe shift                |


        """
        shap_df_ref = self.fetch_shap_values(start_dt=period_ref[0], end_dt=period_ref[1]).rename(
            columns=lambda col: col.replace("shap_", "")
        )
        shap_df_curr = self.fetch_shap_values(
            start_dt=period_curr[0], end_dt=period_curr[1]
        ).rename(columns=lambda col: col.replace("shap_", ""))

        return self._compare_shap_dataframes(shap_df_ref, shap_df_curr, sort_by)

    def compare_batches(
        self,
        batch_ref: str,
        batch_curr: str,
        sort_by: str = "psi",
    ) -> DFrameLike:
        """Compare SHAP explanations between two batches.

        Parameters
        ----------
        batch_ref : str
            Identifier for the first batch.
        batch_curr : str
            Identifier for the second batch.
        sort_by : str, optional
            Column to sort results by (default: 'psi').

        Returns
        -------
        DataFrame
            Comparison of SHAP statistics between the two batches.
        """
        shap_df_ref = self.fetch_shap_values(batch_id=batch_ref).rename(
            columns=lambda col: col.replace("shap_", "")
        )
        shap_df_curr = self.fetch_shap_values(batch_id=batch_curr).rename(
            columns=lambda col: col.replace("shap_", "")
        )

        return self._compare_shap_dataframes(shap_df_ref, shap_df_curr, sort_by)

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
        raise NotImplementedError("Method not yet implemented.")
