from typing import Any

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

from shapmonitor.exceptions import InvalidShapeError
from shapmonitor.types import ArrayLike


# TODO: Add support for calculating for multiple features in parallel.
def population_stability_index(
    reference: ArrayLike | pd.Series, current: ArrayLike | pd.Series, buckets: int = 10
) -> float:
    """
    Calculate the Population Stability Index (PSI) between two univariate distributions.

    Parameters
    ----------
    reference : ArrayLike
        The reference distribution (e.g., historical data).
    current : ArrayLike
        The current distribution (e.g., new data).
    buckets : int
        The number of buckets to divide the data into (default is 10).

    Returns
    -------
    float
        The Population Stability Index value.
    """
    if buckets < 2:
        raise ValueError("Number of buckets must be at least 2.")

    if len(reference) == 0 or len(current) == 0:
        raise ValueError("Input distributions must not be empty.")

    if reference.ndim >= 2 or current.ndim >= 2:
        raise InvalidShapeError("Input distributions must be univariate (1D arrays).")

    reference = np.asarray(reference, dtype=np.float32)
    current = np.asarray(current, dtype=np.float32)

    percentiles = np.linspace(0, 100, buckets + 1)
    bin_edges = np.percentile(reference, percentiles)

    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    curr_counts, _ = np.histogram(current, bins=bin_edges)

    ref_props = ref_counts / len(reference)
    curr_props = curr_counts / len(current)

    # Avoid division by zero and log of zero by replacing zeros with a small value
    epsilon = 1e-10
    ref_props = np.where(ref_props == 0, epsilon, ref_props)
    curr_props = np.where(curr_props == 0, epsilon, curr_props)

    psi_values = (curr_props - ref_props) * np.log(curr_props / ref_props)
    return np.sum(psi_values)


def adversarial_auc(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    classifier: Any | None = None,
    cv: int = 5,
    random_state: int | None = None,
) -> tuple[float, pd.Series]:
    """Calculate adversarial AUC between two DataFrames.

    Trains a binary classifier to distinguish samples from ``reference`` (label 0)
    vs ``current`` (label 1) using stratified k-fold cross-validation. AUC ≈ 0.5
    means the two distributions are indistinguishable; AUC → 1.0 indicates strong
    distributional shift. Per-feature importances from the classifier reveal which
    dimensions drive the separability.

    Parameters
    ----------
    reference : DFrameLike
        Reference dataset (e.g., historical SHAP values or feature values).
    current : DFrameLike
        Current dataset to compare against the reference.
    classifier : sklearn estimator, optional
        A fitted or unfitted sklearn-compatible classifier that supports
        ``predict_proba`` and exposes ``feature_importances_`` after fitting.
        Defaults to ``RandomForestClassifier(n_estimators=100)``.
    cv : int, optional
        Number of stratified k-fold splits for cross-validated AUC (default: 5).
        Must be at least 2.
    random_state : int | None, optional
        Random state passed to the default classifier and CV splitter for
        reproducibility. Ignored when a custom ``classifier`` is provided.

    Returns
    -------
    tuple[float, pd.Series]
        A ``(auc, feature_importances)`` tuple where ``auc`` is the
        cross-validated ROC AUC score and ``feature_importances`` is a
        ``pd.Series`` indexed by feature name, derived by fitting the classifier
        on the full combined dataset.

    Raises
    ------
    ValueError
        If either DataFrame is empty, ``cv < 2``, or the DataFrames share no
        common columns.
    AttributeError
        If the provided classifier does not expose ``feature_importances_``
        after fitting.

    Notes
    -----
    Only columns present in both DataFrames are used. Missing values are filled
    with 0 before classification.

    AUC interpretation guide:

      | AUC        | Interpretation                        |
      |------------|---------------------------------------|
      | 0.50       | Distributions are indistinguishable   |
      | 0.50–0.65  | Minor differences, likely noise       |
      | 0.65–0.80  | Moderate shift — worth investigating  |
      | 0.80–0.90  | Strong shift detected                 |
      | > 0.90     | Severe — clearly different regimes    |
    """
    if reference.empty or current.empty:
        raise ValueError("Input DataFrames must not be empty.")

    if cv < 2:
        raise ValueError(f"cv must be at least 2, got {cv}.")

    common_cols = reference.columns.intersection(current.columns).tolist()
    if not common_cols:
        raise ValueError("reference and current DataFrames have no common columns.")

    ref = reference[common_cols].fillna(0.0)
    curr = current[common_cols].fillna(0.0)

    combined = pd.concat([ref, curr], ignore_index=True)
    labels = np.repeat([0, 1], [len(ref), len(curr)])

    if classifier is None:
        classifier = RandomForestClassifier(n_estimators=100, random_state=random_state)

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    cv_results = cross_validate(
        classifier, combined, labels, cv=skf, scoring="roc_auc", return_estimator=True
    )
    auc = float(cv_results["test_score"].mean())

    try:
        importances_per_fold = np.array(
            [est.feature_importances_ for est in cv_results["estimator"]]
        )
    except AttributeError as exc:
        raise AttributeError(
            f"{type(classifier).__name__} does not expose feature_importances_. "
            "Use a tree-based classifier or one that supports feature_importances_."
        ) from exc

    importances = pd.Series(importances_per_fold.mean(axis=0), index=common_cols)
    return auc, importances
