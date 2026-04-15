import logging
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

from shapmonitor.analysis.metrics import adversarial_auc, population_stability_index
from shapmonitor.types import (
    Backend,
    DFrameLike,
    Period,
    SeriesLike,
)  # DFrameLike/SeriesLike kept for internal use

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
    >>> from datetime import datetime
    >>> from shapmonitor.backends import ParquetBackend
    >>> from shapmonitor.analysis import SHAPAnalyzer
    >>> backend = ParquetBackend("/path/to/shap_logs")
    >>> analyzer = SHAPAnalyzer(backend, min_abs_shap=0.01)
    >>> summary = analyzer.summary(datetime(2025, 1, 1), datetime(2025, 1, 31))
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

    @staticmethod
    def _validate_top_k(top_k: int | None) -> None:
        """Raise ValueError if top_k is set but less than 1."""
        if top_k is not None and top_k < 1:
            raise ValueError(f"top_k must be a positive integer, got {top_k}.")

    def _fetch_and_strip_shap_values(self, **kwargs) -> DFrameLike:
        """Fetch SHAP values and strip the 'shap_' column prefix."""
        return self.fetch_shap_values(**kwargs).rename(columns=lambda col: col.replace("shap_", ""))

    def fetch_shap_values(self, **kwargs) -> pd.DataFrame:
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
        top_k: int | None = None,
    ) -> pd.DataFrame:
        """Compute summary statistics for SHAP values in a date range.

        Parameters
        ----------
        start_dt : datetime | date, optional
            Start of the date range (inclusive).
        end_dt : datetime | date, optional
            End of the date range (inclusive).
        batch_id : str, optional
            Batch ID to filter results to a specific batch.
        model_version : str, optional
            Model version to filter results to a specific model version.
        sort_by : str, optional
            Column to sort results by (default: 'mean_abs').
            Options: 'mean_abs', 'mean', 'std', 'min', 'max'.
        top_k : int | None, optional
            If set, return only the top k features after sorting.
            Must be a positive integer. Default is None (return all features).

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
        self._validate_top_k(top_k)

        shap_df = self._fetch_and_strip_shap_values(
            start_dt=start_dt, end_dt=end_dt, batch_id=batch_id, model_version=model_version
        )
        result = self._construct_summary(shap_df)

        if sort_by not in result.columns:
            raise ValueError(
                f"Invalid sort_by value: {sort_by}. Must be one of {list(result.columns)}"
            )

        # TODO: Add relationship correlation with target if feature values and predictions are available

        result = result.sort_values(by=sort_by, ascending=False)
        if top_k is not None:
            result = result.head(top_k)
        return result

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
        top_k: int | None = None,
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
        summary_df_ref = self._construct_summary(shap_df_ref)
        summary_df_curr = self._construct_summary(shap_df_curr)

        # Capture attrs before suffix (pandas loses attrs on most operations)
        n_samples_ref = summary_df_ref.attrs.get("n_samples")
        n_samples_curr = summary_df_curr.attrs.get("n_samples")

        # Rename columns with suffixes
        summary_df_ref = summary_df_ref.add_suffix("_ref")
        summary_df_curr = summary_df_curr.add_suffix("_curr")

        # Merge on index (feature name)
        comparison_df = pd.merge(
            summary_df_ref, summary_df_curr, left_index=True, right_index=True, how="outer"
        )
        # Delta calculations
        comparison_df["delta_mean_abs"] = (
            comparison_df["mean_abs_curr"] - comparison_df["mean_abs_ref"]
        )
        comparison_df["pct_delta_mean_abs"] = (
            comparison_df["delta_mean_abs"] / comparison_df["mean_abs_ref"].replace(0, pd.NA)
        ) * 100

        # Rank calculations
        comparison_df["rank_ref"] = comparison_df["mean_abs_ref"].rank(ascending=False)
        comparison_df["rank_curr"] = comparison_df["mean_abs_curr"].rank(ascending=False)
        comparison_df["delta_rank"] = comparison_df["rank_curr"] - comparison_df["rank_ref"]

        conditions = [comparison_df["delta_rank"] < 0, comparison_df["delta_rank"] > 0]
        comparison_df["rank_change"] = np.select(
            conditions, ["increased", "decreased"], default="no_change"
        )

        # Sign flip calculation (NaN filled with 0 to avoid false positives)
        comparison_df["sign_flip"] = np.sign(comparison_df["mean_ref"].fillna(0)) != np.sign(
            comparison_df["mean_curr"].fillna(0)
        )

        # Add PSI calculation
        comparison_df = comparison_df.join(psi, how="left").fillna({"psi": np.nan})

        # TODO: Add relationship flip calculations

        comparison_df = comparison_df[
            [
                "psi",
                "mean_abs_ref",
                "mean_abs_curr",
                "delta_mean_abs",
                "pct_delta_mean_abs",
                "mean_ref",
                "mean_curr",
                "rank_ref",
                "rank_curr",
                "delta_rank",
                "rank_change",
                "sign_flip",
            ]
        ]
        comparison_df.attrs["n_samples_ref"] = n_samples_ref
        comparison_df.attrs["n_samples_curr"] = n_samples_curr

        if sort_by not in comparison_df.columns:
            raise ValueError(
                f"Invalid sort_by value: {sort_by}. Must be one of {list(comparison_df.columns)}"
            )

        result = comparison_df.sort_values(by=sort_by, ascending=False)
        if top_k is not None:
            result = result.head(top_k)
        return result

    def compare_time_periods(
        self,
        period_ref: Period,
        period_curr: Period,
        sort_by: str = "psi",
        top_k: int | None = None,
    ) -> pd.DataFrame:
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
            Column to sort results by (default: 'psi').
        top_k : int | None, optional
            If set, return only the top k features after sorting.
            Must be a positive integer. Default is None (return all features).

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
        self._validate_top_k(top_k)

        shap_df_ref = self._fetch_and_strip_shap_values(
            start_dt=period_ref[0], end_dt=period_ref[1]
        )
        shap_df_curr = self._fetch_and_strip_shap_values(
            start_dt=period_curr[0], end_dt=period_curr[1]
        )

        return self._compare_shap_dataframes(shap_df_ref, shap_df_curr, sort_by, top_k)

    def compare_batches(
        self,
        batch_ref: str,
        batch_curr: str,
        sort_by: str = "psi",
        top_k: int | None = None,
    ) -> pd.DataFrame:
        """Compare SHAP explanations between two batches.

        Parameters
        ----------
        batch_ref : str
            Identifier for the first batch.
        batch_curr : str
            Identifier for the second batch.
        sort_by : str, optional
            Column to sort results by (default: 'psi').
        top_k : int | None, optional
            If set, return only the top k features after sorting.
            Must be a positive integer. Default is None (return all features).

        Returns
        -------
        DataFrame
            Comparison of SHAP statistics between the two batches.

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
        self._validate_top_k(top_k)

        shap_df_ref = self._fetch_and_strip_shap_values(batch_id=batch_ref)
        shap_df_curr = self._fetch_and_strip_shap_values(batch_id=batch_curr)

        return self._compare_shap_dataframes(shap_df_ref, shap_df_curr, sort_by, top_k)

    def compare_versions(
        self,
        model_version_ref: str,
        model_version_curr: str,
        sort_by: str = "psi",
        top_k: int | None = None,
    ) -> pd.DataFrame:
        """Compare SHAP explanations across different model versions.

        Parameters
        ----------
        model_version_ref : str
            Reference model version identifier.
        model_version_curr : str
            Current model version identifier.
        sort_by : str, optional
            Column to sort results by (default: 'psi').
        top_k : int | None, optional
            If set, return only the top k features after sorting.
            Must be a positive integer. Default is None (return all features).

        Returns
        -------
        DataFrame
            Comparison of SHAP statistics across model versions.

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
        self._validate_top_k(top_k)

        shap_df_ref = self._fetch_and_strip_shap_values(model_version=model_version_ref)
        shap_df_curr = self._fetch_and_strip_shap_values(model_version=model_version_curr)

        return self._compare_shap_dataframes(shap_df_ref, shap_df_curr, sort_by, top_k)

    def compare_adversarial(
        self,
        period_ref: Period,
        period_curr: Period,
        classifier: Any | None = None,
        cv: int = 5,
        sort_by: str = "adv_importance",
        top_k: int | None = None,
        random_state: int | None = None,
    ) -> pd.DataFrame:
        """Compare SHAP distributions between two periods using adversarial validation.

        Trains a binary classifier to distinguish SHAP values from ``period_ref``
        (label 0) vs ``period_curr`` (label 1). The cross-validated AUC measures
        overall distributional shift; per-feature importances reveal which SHAP
        dimensions drive the separability — complementing the univariate PSI score.

        Parameters
        ----------
        period_ref : Period
            Tuple of (start_dt, end_dt) defining the reference date range.
        period_curr : Period
            Tuple of (start_dt, end_dt) defining the current date range.
        classifier : sklearn estimator, optional
            Sklearn-compatible classifier with ``predict_proba`` and
            ``feature_importances_``. Defaults to ``RandomForestClassifier``.
        cv : int, optional
            Number of stratified k-fold splits (default: 5).
        sort_by : str, optional
            Column to sort results by (default: 'adv_importance').
        top_k : int | None, optional
            If set, return only the top k features. Must be a positive integer.
        random_state : int | None, optional
            Random state for the default classifier and CV splitter.

        Returns
        -------
        DataFrame
            Comparison statistics indexed by feature name.

            Columns:
                - adv_importance: Feature's contribution to classifier separability
                - mean_abs_1, mean_abs_2: Mean absolute SHAP value per period
                - delta_mean_abs: Absolute importance change (period_2 - period_1)

            Attributes:
                - adversarial_auc: Cross-validated AUC (0.5 = no shift, 1.0 = max shift)
                - n_samples_ref: Sample count in the reference period
                - n_samples_curr: Sample count in the current period

        Raises
        ------
        ValueError
            If top_k < 1 or sort_by is not a valid column name.

        Notes
        -----
        Returns an empty DataFrame if either period contains no data.

        To run adversarial validation on raw input feature distributions (not SHAP),
        use ``adversarial_auc`` from ``shapmonitor.analysis.metrics`` directly with
        ``backend.read(...).filter(like="feat_")``.

        AUC interpretation guide:

          | AUC        | Interpretation                        |
          |------------|---------------------------------------|
          | 0.50       | Distributions are indistinguishable   |
          | 0.50–0.65  | Minor differences, likely noise       |
          | 0.65–0.80  | Moderate shift — worth investigating  |
          | 0.80–0.90  | Strong shift detected                 |
          | > 0.90     | Severe — clearly different regimes    |
        """
        self._validate_top_k(top_k)

        shap_df_ref = self._fetch_and_strip_shap_values(
            start_dt=period_ref[0], end_dt=period_ref[1]
        )
        shap_df_curr = self._fetch_and_strip_shap_values(
            start_dt=period_curr[0], end_dt=period_curr[1]
        )

        return self._run_adversarial_comparison(
            shap_df_ref, shap_df_curr, classifier, cv, sort_by, top_k, random_state
        )

    def compare_adversarial_batches(
        self,
        batch_ref: str,
        batch_curr: str,
        classifier: Any | None = None,
        cv: int = 5,
        sort_by: str = "adv_importance",
        top_k: int | None = None,
        random_state: int | None = None,
    ) -> pd.DataFrame:
        """Compare SHAP distributions between two batches using adversarial validation.

        Trains a binary classifier to distinguish SHAP values from ``batch_ref``
        (label 0) vs ``batch_curr`` (label 1). The cross-validated AUC measures
        overall distributional shift; per-feature importances reveal which SHAP
        dimensions drive the separability — complementing the univariate PSI score.

        Parameters
        ----------
        batch_ref : str
            Identifier for the reference batch.
        batch_curr : str
            Identifier for the current batch.
        classifier : sklearn estimator, optional
            Sklearn-compatible classifier with ``predict_proba`` and
            ``feature_importances_``. Defaults to ``RandomForestClassifier``.
        cv : int, optional
            Number of stratified k-fold splits (default: 5).
        sort_by : str, optional
            Column to sort results by (default: 'adv_importance').
        top_k : int | None, optional
            If set, return only the top k features. Must be a positive integer.
        random_state : int | None, optional
            Random state for the default classifier and CV splitter.

        Returns
        -------
        DataFrame
            Comparison statistics indexed by feature name.

            Columns:
                - adv_importance: Feature's contribution to classifier separability
                - mean_abs_1, mean_abs_2: Mean absolute SHAP value per batch
                - delta_mean_abs: Absolute importance change (batch_2 - batch_1)

            Attributes:
                - adversarial_auc: Cross-validated AUC (0.5 = no shift, 1.0 = max shift)
                - n_samples_ref: Sample count in the reference batch
                - n_samples_curr: Sample count in the current batch

        Raises
        ------
        ValueError
            If top_k < 1 or sort_by is not a valid column name.

        Notes
        -----
        Returns an empty DataFrame if either batch contains no data.

        Batch sizes sampled via ``sample_rate`` may be small. Ensure each batch
        has enough rows for the chosen ``cv`` splits (at least ``2 * cv`` samples
        total is recommended) for stable AUC estimates.

        AUC interpretation guide:

          | AUC        | Interpretation                        |
          |------------|---------------------------------------|
          | 0.50       | Distributions are indistinguishable   |
          | 0.50–0.65  | Minor differences, likely noise       |
          | 0.65–0.80  | Moderate shift — worth investigating  |
          | 0.80–0.90  | Strong shift detected                 |
          | > 0.90     | Severe — clearly different regimes    |
        """
        self._validate_top_k(top_k)

        shap_df_ref = self._fetch_and_strip_shap_values(batch_id=batch_ref)
        shap_df_curr = self._fetch_and_strip_shap_values(batch_id=batch_curr)

        return self._run_adversarial_comparison(
            shap_df_ref, shap_df_curr, classifier, cv, sort_by, top_k, random_state
        )

    def drift_report(
        self,
        period_ref: Period,
        period_curr: Period,
        warn_threshold: float = 0.1,
        alert_threshold: float = 0.25,
    ) -> pd.DataFrame:
        """Classify features by drift severity using PSI thresholds.

        Wraps ``compare_time_periods()`` and adds a ``drift_status`` column
        classifying each feature as ``"stable"``, ``"warning"``, ``"alert"``,
        or ``"unknown"`` (when PSI is NaN — feature appears in only one period).

        Parameters
        ----------
        period_ref : Period
            Tuple of (start_dt, end_dt) defining the reference date range.
        period_curr : Period
            Tuple of (start_dt, end_dt) defining the current date range.
        warn_threshold : float, optional
            PSI threshold above which status becomes ``"warning"`` (default: 0.1).
        alert_threshold : float, optional
            PSI threshold above which status becomes ``"alert"`` (default: 0.25).

        Returns
        -------
        DataFrame
            All columns from ``compare_time_periods()`` plus ``drift_status``.

            Attributes:
                - n_samples_ref: Sample count in the reference period
                - n_samples_curr: Sample count in the current period
        """
        result = self.compare_time_periods(period_ref, period_curr)

        if result.empty:
            return result

        conditions = [
            result["psi"].isna(),
            result["psi"] < warn_threshold,
            (result["psi"] >= warn_threshold) & (result["psi"] < alert_threshold),
            result["psi"] >= alert_threshold,
        ]
        choices = ["unknown", "stable", "warning", "alert"]
        result["drift_status"] = np.select(conditions, choices, default="unknown")

        return result

    def fetch_full_data(
        self,
        start_dt: datetime | date | None = None,
        end_dt: datetime | date | None = None,
        batch_id: str | None = None,
        model_version: str | None = None,
    ) -> pd.DataFrame:
        """Fetch all columns from the backend (SHAP values, feature values, predictions, metadata).

        Unlike ``fetch_shap_values()``, this returns the full raw DataFrame with
        all stored columns: ``feat_*``, ``shap_*``, ``prediction``, ``timestamp``,
        ``batch_id``, ``model_version``, and ``base_value``.

        Parameters
        ----------
        start_dt : datetime | date, optional
            Start of the date range (inclusive).
        end_dt : datetime | date, optional
            End of the date range (inclusive).
        batch_id : str, optional
            Batch ID to filter to a specific batch.
        model_version : str, optional
            Model version to filter to a specific version.

        Returns
        -------
        DataFrame
            Raw DataFrame with all stored columns, or empty DataFrame if no data.
        """
        return self._backend.read(
            start_dt=start_dt, end_dt=end_dt, batch_id=batch_id, model_version=model_version
        )

    def feature_trends(
        self,
        start_dt: datetime | date,
        end_dt: datetime | date,
        freq: str = "7D",
    ) -> pd.DataFrame:
        """Compute mean absolute SHAP values over rolling time windows.

        Splits ``[start_dt, end_dt]`` into windows of size ``freq`` and computes
        ``summary()`` for each non-empty window. Useful for detecting gradual
        feature importance drift over time.

        Parameters
        ----------
        start_dt : datetime | date
            Start of the overall time range.
        end_dt : datetime | date
            End of the overall time range (inclusive).
        freq : str, optional
            Pandas offset alias defining window size (default: ``"7D"``).
            Examples: ``"1D"``, ``"7D"``, ``"14D"``, ``"30D"``.

        Returns
        -------
        DataFrame
            MultiIndex DataFrame with ``(period_start, feature)`` index and
            a ``mean_abs`` column. Empty windows are skipped.

            Index names: ``["period_start", "feature"]``
        """
        from pandas.tseries.frequencies import to_offset

        _empty = pd.DataFrame(
            columns=["mean_abs"],
            index=pd.MultiIndex.from_tuples([], names=["period_start", "feature"]),
        )

        offset = to_offset(freq)
        window_starts = pd.date_range(start=start_dt, end=end_dt, freq=freq)
        if not len(window_starts):
            return _empty

        # Single backend read for the entire range — partition in memory per window
        full_df = self._backend.read(start_dt=start_dt, end_dt=end_dt)
        if full_df.empty:
            return _empty

        shap_df = full_df.filter(like="shap_").rename(columns=lambda c: c.replace("shap_", ""))
        timestamps = pd.to_datetime(full_df["timestamp"])

        frames = []
        for window_start in window_starts:
            window_end = window_start + offset - pd.Timedelta(seconds=1)
            if window_end > pd.Timestamp(end_dt):
                window_end = pd.Timestamp(end_dt)

            mask = (timestamps >= window_start) & (timestamps <= window_end)
            window_shap = shap_df[mask]
            if window_shap.empty:
                continue

            mean_abs = window_shap.abs().mean().rename("mean_abs")
            mean_abs.index.name = "feature"
            mean_abs_df = mean_abs.to_frame()
            mean_abs_df["period_start"] = window_start.date()
            mean_abs_df = mean_abs_df.reset_index().set_index(["period_start", "feature"])
            frames.append(mean_abs_df)

        return pd.concat(frames) if frames else _empty

    def _run_adversarial_comparison(
        self,
        shap_df_ref: DFrameLike,
        shap_df_curr: DFrameLike,
        classifier: Any | None,
        cv: int,
        sort_by: str,
        top_k: int | None,
        random_state: int | None,
    ) -> DFrameLike:
        """Run adversarial validation on two pre-fetched SHAP DataFrames."""
        if shap_df_ref.empty or shap_df_curr.empty:
            _logger.warning("No data found for one or both inputs.")
            return pd.DataFrame()

        auc, importances = adversarial_auc(shap_df_ref, shap_df_curr, classifier, cv, random_state)

        summary_ref = self._construct_summary(shap_df_ref)
        summary_curr = self._construct_summary(shap_df_curr)

        n_samples_ref = summary_ref.attrs.get("n_samples")
        n_samples_curr = summary_curr.attrs.get("n_samples")

        result = pd.DataFrame(
            {
                "adv_importance": importances,
                "mean_abs_ref": summary_ref["mean_abs"],
                "mean_abs_curr": summary_curr["mean_abs"],
            }
        )
        result["delta_mean_abs"] = result["mean_abs_curr"] - result["mean_abs_ref"]
        result.index.name = "feature"

        if sort_by not in result.columns:
            raise ValueError(
                f"Invalid sort_by value: {sort_by}. Must be one of {list(result.columns)}"
            )

        result = result.sort_values(by=sort_by, ascending=False)
        result.attrs["adversarial_auc"] = auc
        result.attrs["n_samples_ref"] = n_samples_ref
        result.attrs["n_samples_curr"] = n_samples_curr

        if top_k is not None:
            result = result.head(top_k)

        return result
