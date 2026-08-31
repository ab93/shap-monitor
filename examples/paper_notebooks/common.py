"""Shared setup for the two reproduction notebooks."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, NamedTuple, TypedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from shap import TreeExplainer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from shapmonitor.analysis.metrics import adversarial_auc, population_stability_index

warnings.filterwarnings("ignore")

SEED = 42
N_ADV = 4000  # rows per window for every adversarial-validation estimate
B_ADV = 20  # independent subsamples behind each adversarial figure
B_PSI = 200  # bootstrap resamples behind each PSI interval
CV = 3
PSI_BUCKETS = 10

CI_PERCENTILES = (2.5, 97.5)

# Empty bins are floored here so the PSI log stays finite.  Section 7 of notebook
# 01 is about how much the reported value depends on this constant.
PSI_EPSILON = 1e-10

PSI_WARN = 0.1
PSI_ALERT = 0.25

# `adv_auc_ci` derives one seed per replicate; the offset keeps that stream clear
# of the seeds handed to the classifier and to `psi_ci`.
ADV_SEED_OFFSET = 1000

AGE_CUTOFF = 45
TEST_SIZE = 0.3

# Guards the shap/raw ratio when a feature's inputs did not move at all.
RATIO_FLOOR = 1e-12

CACHE_DIR = Path(".adult_cache")
FIGURES_DIR = Path("figures")
RESULTS_FILE = Path("study_a_results.json")  # notebook 01 writes it, 02 reads it

# Palette.  BLUE is the reference window, ORANGE the current one, RED the
# attribution view, GREY a null control; GRID is for axis furniture.
BLUE, ORANGE, RED, GREY = "#2f6df6", "#e8944a", "#e5484d", "#8b93a5"
GRID = "#e6e9f0"
ERROR_BAR = "#40465a"

# LightGBM parallelises histogram construction and is not bit-reproducible unless
# asked; `random_state` alone is NOT sufficient.
LGBM_DETERMINISM = {
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": 1,
    "verbose": -1,
}

# The UCI Adult train and test splits label the target differently: the test half
# carries a trailing period.  Both spellings map to the same class.
INCOME_LABELS = {"<=50K": 0, "<=50K.": 0, ">50K": 1, ">50K.": 1}


class Interval(TypedDict):
    """A point estimate with a percentile interval around it.

    A ``TypedDict`` rather than a dataclass because notebook 01 serialises these
    straight to JSON for notebook 02 to read back.
    """

    point: float
    lo: float
    hi: float


class BootstrapInterval(Interval):
    """An :class:`Interval` that also reports the spread of its replicates."""

    sd: float
    n_rep: int


def use_paper_style() -> None:
    """Apply the figure styling used throughout the paper to the global rcParams."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "axes.edgecolor": "#c8cfda",
            "axes.linewidth": 1.1,
            "xtick.color": "#5b6478",
            "ytick.color": "#5b6478",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 200,
        }
    )


def make_model(**kwargs: Any) -> LGBMClassifier:
    """Build an unfitted ``LGBMClassifier`` with determinism pinned.

    Parameters
    ----------
    **kwargs : Any
        Overrides forwarded to ``LGBMClassifier``. Caller values win over both
        ``LGBM_DETERMINISM`` and the default ``random_state``.

    Returns
    -------
    LGBMClassifier
        A classifier that trains bit-reproducibly on a fixed input.
    """
    return LGBMClassifier(**{**LGBM_DETERMINISM, "random_state": SEED, **kwargs})


def load_adult(cache: Path = CACHE_DIR) -> tuple[pd.DataFrame, pd.Series]:
    """Fetch UCI Adult (id=2), cached locally so reruns work offline.

    Parameters
    ----------
    cache : Path
        Directory holding the cached Parquet copy; created if absent.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        Features with string columns as ``category`` dtype, and the binary
        income target.
    """
    cache.mkdir(parents=True, exist_ok=True)
    features_path, target_path = cache / "adult_X.parquet", cache / "adult_y.parquet"

    if features_path.exists() and target_path.exists():
        X, y = pd.read_parquet(features_path), pd.read_parquet(target_path)["income"]
    else:
        X, y = _download_adult()
        X.to_parquet(features_path)
        y.to_frame().to_parquet(target_path)

    for col in X.select_dtypes(include=["object", "string"]).columns:
        X[col] = X[col].astype("category")
    return X, y.astype(int)


def _download_adult() -> tuple[pd.DataFrame, pd.Series]:
    """Fetch Adult from the UCI repository, mapping the target to 0/1."""
    from ucimlrepo import fetch_ucirepo  # only needed on a cold cache

    adult = fetch_ucirepo(id=2)
    X = adult.data.features.copy()
    y = adult.data.targets.copy()
    y["income"] = y["income"].map(INCOME_LABELS)
    return X, y["income"]


def numeric_encode(df: pd.DataFrame) -> pd.DataFrame:
    """Integer-encode categoricals so raw inputs feed the same detectors as SHAP.

    Parameters
    ----------
    df : pd.DataFrame
        Frame that may mix numeric and ``category`` columns.

    Returns
    -------
    pd.DataFrame
        An all-float copy, categoricals replaced by their level codes.
    """
    out = df.copy()
    for col in out.select_dtypes(include=["category"]).columns:
        out[col] = out[col].cat.codes.astype(float)
    return out.astype(float)


def explain(model: LGBMClassifier, X: pd.DataFrame) -> pd.DataFrame:
    """Positive-class SHAP values as a DataFrame aligned to ``X``'s columns.

    Parameters
    ----------
    model : LGBMClassifier
        Fitted model to explain.
    X : pd.DataFrame
        Rows to explain.

    Returns
    -------
    pd.DataFrame
        One SHAP value per cell of ``X``, same index and columns.
    """
    values = np.asarray(TreeExplainer(model).shap_values(X))
    if values.ndim == 3:
        # Binary classifiers return (n_rows, n_features, n_classes); keep the
        # positive class, the same slice `SHAPMonitor.log_batch` stores.
        values = values[:, :, -1]
    return pd.DataFrame(values, columns=X.columns, index=X.index)


def _subsample(df: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Draw ``n`` rows without replacement, or all of them if ``df`` is smaller."""
    return df.iloc[rng.choice(len(df), size=min(n, len(df)), replace=False)]


def adv_auc_ci(ref: pd.DataFrame, curr: pd.DataFrame, seed: int = SEED) -> BootstrapInterval:
    """Adversarial AUC over ``B_ADV`` independent subsamples, with a percentile interval.

    Subsamples WITHOUT replacement on purpose.  A with-replacement bootstrap
    duplicates rows, and because the AUC is cross-validated the same row can land
    in both a training and a validation fold; the classifier memorises it and the
    estimate inflates -- enough to push a true 0.50 control to about 0.61.

    Parameters
    ----------
    ref : pd.DataFrame
        Reference window.
    curr : pd.DataFrame
        Current window, with the same columns as ``ref``.
    seed : int
        Base seed for the subsampling and for the classifier.

    Returns
    -------
    BootstrapInterval
        Mean AUC across replicates, its percentile interval, and their spread.
    """
    aucs = []
    for replicate in range(B_ADV):
        rng = np.random.default_rng(seed + ADV_SEED_OFFSET + replicate)
        auc, _ = adversarial_auc(
            _subsample(ref, N_ADV, rng),
            _subsample(curr, N_ADV, rng),
            cv=CV,
            random_state=seed,
        )
        aucs.append(float(auc))

    lo, hi = np.percentile(aucs, CI_PERCENTILES)
    return BootstrapInterval(
        point=float(np.mean(aucs)),
        lo=float(lo),
        hi=float(hi),
        sd=float(np.std(aucs, ddof=1)),
        n_rep=B_ADV,
    )


def _numeric_psi(ref: pd.Series, curr: pd.Series) -> float:
    """PSI between two numeric columns, binned on the reference quantiles.

    A thin pass-through to the package metric, which accepts a Series and does
    its own float cast.
    """
    return float(population_stability_index(ref, curr, buckets=PSI_BUCKETS))


def _floored_props(props: pd.Series, levels: pd.Index, eps: float) -> np.ndarray:
    """Align ``props`` to ``levels``, flooring absent levels at ``eps``."""
    aligned = props.reindex(levels).fillna(0).to_numpy()
    return np.where(aligned == 0, eps, aligned)


def categorical_psi(ref: pd.Series, curr: pd.Series, eps: float = PSI_EPSILON) -> float:
    """PSI between two categorical columns, using their levels as the bins.

    The package's ``population_stability_index`` bins on reference quantiles,
    which is meaningless for unordered levels, so this applies the same PSI
    formula to level frequencies instead. It is the one metric in this module
    the package does not already provide.

    Parameters
    ----------
    ref : pd.Series
        Reference column.
    curr : pd.Series
        Current column.
    eps : float
        Floor for levels absent from one side, keeping the log finite.

    Returns
    -------
    float
        The population stability index.
    """
    ref_props = ref.astype(str).value_counts(normalize=True)
    curr_props = curr.astype(str).value_counts(normalize=True)
    levels = ref_props.index.union(curr_props.index)

    r = _floored_props(ref_props, levels, eps)
    c = _floored_props(curr_props, levels, eps)
    return float(np.sum((c - r) * np.log(c / r)))


def shap_psi(ref: pd.DataFrame, curr: pd.DataFrame) -> pd.Series:
    """Per-feature PSI between two frames of SHAP values.

    Parameters
    ----------
    ref : pd.DataFrame
        Attributions for the reference window.
    curr : pd.DataFrame
        Attributions for the current window.

    Returns
    -------
    pd.Series
        PSI per feature, indexed by ``ref``'s columns.
    """
    return pd.Series({col: _numeric_psi(ref[col], curr[col]) for col in ref.columns})


def raw_psi(ref: pd.DataFrame, curr: pd.DataFrame) -> pd.Series:
    """Per-feature PSI on raw inputs: quantile bins for numerics, levels for categoricals.

    Parameters
    ----------
    ref : pd.DataFrame
        Reference window, categoricals still as ``category`` dtype.
    curr : pd.DataFrame
        Current window, with the same columns as ``ref``.

    Returns
    -------
    pd.Series
        PSI per feature, indexed by ``ref``'s columns.
    """
    psi = {}
    for col in ref.columns:
        if isinstance(ref[col].dtype, pd.CategoricalDtype):
            psi[col] = categorical_psi(ref[col], curr[col])
        else:
            psi[col] = _numeric_psi(ref[col], curr[col])
    return pd.Series(psi)


def psi_ci(ref: pd.Series, curr: pd.Series, seed: int = SEED) -> Interval:
    """PSI for one feature, with a bootstrap percentile interval.

    Resampling WITH replacement is safe here, unlike in :func:`adv_auc_ci`: PSI
    is not cross-validated, so a duplicated row cannot leak between folds.

    Parameters
    ----------
    ref : pd.Series
        Reference values for the feature.
    curr : pd.Series
        Current values for the feature.
    seed : int
        Seed for the bootstrap resampling.

    Returns
    -------
    Interval
        PSI on the full samples, with the interval from ``B_PSI`` replicates.
    """
    ref_values, curr_values = ref.to_numpy(float), curr.to_numpy(float)
    rng = np.random.default_rng(seed)
    replicates = [
        float(
            population_stability_index(
                rng.choice(ref_values, len(ref_values), replace=True),
                rng.choice(curr_values, len(curr_values), replace=True),
                buckets=PSI_BUCKETS,
            )
        )
        for _ in range(B_PSI)
    ]

    lo, hi = np.percentile(replicates, CI_PERCENTILES)
    return Interval(
        point=float(population_stability_index(ref_values, curr_values, buckets=PSI_BUCKETS)),
        lo=float(lo),
        hi=float(hi),
    )


def interval_yerr(interval: Interval) -> list[list[float]]:
    """Asymmetric ``yerr`` for a matplotlib bar, from an interval.

    Parameters
    ----------
    interval : Interval
        Any mapping with ``point``, ``lo`` and ``hi`` keys, including one read
        back from JSON.

    Returns
    -------
    list[list[float]]
        ``[[lower], [upper]]`` distances from the point estimate, clipped at 0.
    """
    return [
        [max(0.0, interval["point"] - interval["lo"])],
        [max(0.0, interval["hi"] - interval["point"])],
    ]


class StudySplit(NamedTuple):
    """The Adult partition both studies are built on.

    Study A serves ``X_old`` to a model trained on ``X_tr``; Study B ignores
    ``X_old`` and scores every model version on ``X_val``.
    """

    X_tr: pd.DataFrame
    X_val: pd.DataFrame
    y_tr: pd.Series
    y_val: pd.Series
    X_old: pd.DataFrame
    y_old: pd.Series
    categoricals: list[str]

    @property
    def feature_order(self) -> list[str]:
        """Training column order, which LightGBM's per-feature options index into."""
        return list(self.X_tr.columns)


def load_study_split(seed: int = SEED) -> StudySplit:
    """Load Adult and cut it into the training, reference and drifted windows.

    Parameters
    ----------
    seed : int
        Seed for the stratified train/validation split.

    Returns
    -------
    StudySplit
        The younger population split into train and reference, plus the older
        population that stands in for drifted production traffic.
    """
    X, y = load_adult()
    X_young = X[X["age"] <= AGE_CUTOFF].copy()
    X_old = X[X["age"] > AGE_CUTOFF].copy()
    y_young, y_old = y.loc[X_young.index], y.loc[X_old.index]

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_young, y_young, test_size=TEST_SIZE, random_state=seed, stratify=y_young
    )
    return StudySplit(
        X_tr=X_tr,
        X_val=X_val,
        y_tr=y_tr,
        y_val=y_val,
        X_old=X_old,
        y_old=y_old,
        categoricals=X_tr.select_dtypes(include=["category"]).columns.tolist(),
    )


def fit_model(split: StudySplit, **kwargs: Any) -> LGBMClassifier:
    """Fit a deterministic ``LGBMClassifier`` on the split's training rows.

    Parameters
    ----------
    split : StudySplit
        Supplies the training rows and the categorical column names.
    **kwargs : Any
        Overrides forwarded to :func:`make_model`.

    Returns
    -------
    LGBMClassifier
        The fitted model.
    """
    model = make_model(**kwargs)
    model.fit(split.X_tr, split.y_tr, categorical_feature=split.categoricals)
    return model


def split_half(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cut a frame in two, giving the no-drift control both studies rely on.

    Parameters
    ----------
    frame : pd.DataFrame
        Rows to halve; an odd count leaves the extra row in the second half.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        The leading and trailing halves.
    """
    half = len(frame) // 2
    return frame.iloc[:half], frame.iloc[half:]


def psi_comparison(attribution_psi: pd.Series, input_psi: pd.Series) -> pd.DataFrame:
    """Attribution PSI against input PSI per feature, ranked by their ratio.

    Parameters
    ----------
    attribution_psi : pd.Series
        Per-feature PSI on SHAP values.
    input_psi : pd.Series
        Per-feature PSI on raw inputs.

    Returns
    -------
    pd.DataFrame
        Columns ``shap_psi``, ``raw_psi`` and ``ratio``, largest ratio first.
        A ratio above 1 means the attributions moved more than the inputs.
    """
    out = pd.DataFrame({"shap_psi": attribution_psi, "raw_psi": input_psi})
    out["ratio"] = out["shap_psi"] / out["raw_psi"].clip(lower=RATIO_FLOOR)
    return out.sort_values("ratio", ascending=False)


class PsiBins(NamedTuple):
    """The reference quantile bins behind one PSI figure, and the mass in them."""

    ref_values: np.ndarray
    curr_values: np.ndarray
    edges: np.ndarray
    ref_props: np.ndarray
    curr_props: np.ndarray
    empty: np.ndarray

    def terms(self, eps: float = PSI_EPSILON) -> np.ndarray:
        """Per-bin PSI contributions, with bins the current window misses floored at ``eps``.

        Parameters
        ----------
        eps : float
            Floor applied to empty current bins.

        Returns
        -------
        np.ndarray
            One contribution per bin; they sum to the reported PSI.
        """
        floored = np.where(self.empty, eps, self.curr_props)
        return (floored - self.ref_props) * np.log(floored / self.ref_props)


def psi_bins(ref: pd.Series, curr: pd.Series, buckets: int = PSI_BUCKETS) -> PsiBins:
    """Recreate the binning ``population_stability_index`` performs internally.

    Exposing it is the point of Study A's saturation section: it shows how much
    of a large PSI comes from bins the current window never reaches.

    Parameters
    ----------
    ref : pd.Series
        Reference values, whose quantiles define the bin edges.
    curr : pd.Series
        Current values, binned against those same edges.
    buckets : int
        Number of quantile bins.

    Returns
    -------
    PsiBins
        The edges, the proportion each window puts in each bin, and a mask of
        the bins the current window leaves empty.
    """
    ref_values, curr_values = ref.to_numpy(float), curr.to_numpy(float)
    edges = np.percentile(ref_values, np.linspace(0, 100, buckets + 1))
    ref_props = np.histogram(ref_values, bins=edges)[0] / len(ref_values)
    curr_props = np.histogram(curr_values, bins=edges)[0] / len(curr_values)
    return PsiBins(ref_values, curr_values, edges, ref_props, curr_props, curr_props == 0)


def importance_change(ref: pd.DataFrame, curr: pd.DataFrame) -> pd.Series:
    """Percent change in mean |SHAP| per feature, largest gain first.

    Parameters
    ----------
    ref : pd.DataFrame
        Attributions for the reference window.
    curr : pd.DataFrame
        Attributions for the current window.

    Returns
    -------
    pd.Series
        Percent change per feature, sorted descending.
    """
    ref_imp, curr_imp = ref.abs().mean(), curr.abs().mean()
    return ((curr_imp - ref_imp) / ref_imp * 100).sort_values(ascending=False)


def sign_flips(ref: pd.DataFrame, curr: pd.DataFrame) -> list[str]:
    """Features whose mean SHAP value changed sign between the two windows.

    Parameters
    ----------
    ref : pd.DataFrame
        Attributions for the reference window.
    curr : pd.DataFrame
        Attributions for the current window.

    Returns
    -------
    list[str]
        Feature names, in the column order of ``ref``.
    """
    return [col for col in ref.columns if np.sign(ref[col].mean()) != np.sign(curr[col].mean())]


def save_study_a(
    *,
    f1_ref: float,
    f1_curr: float,
    adversarial: dict[str, BootstrapInterval],
    attribution_psi: pd.Series,
    input_psi: pd.Series,
    age_psi_ci: Interval,
    n_ref: int,
    n_curr: int,
) -> None:
    """Cache Study A's numbers to ``RESULTS_FILE`` for notebook 02's summary figure."""
    RESULTS_FILE.write_text(
        json.dumps(
            {
                "f1_ref": float(f1_ref),
                "f1_curr": float(f1_curr),
                "adversarial": adversarial,
                "psi_shap": attribution_psi.to_dict(),
                "psi_raw": input_psi.to_dict(),
                "age_psi_ci": age_psi_ci,
                "n_ref": int(n_ref),
                "n_curr": int(n_curr),
            },
            indent=2,
        )
    )


def load_study_a() -> dict:
    """Read back what :func:`save_study_a` wrote."""
    return json.loads(RESULTS_FILE.read_text())


class Construction(NamedTuple):
    """One recipe for a version 2: how it differs from v1, and where we expect movement."""

    name: str
    kwargs: dict
    affected: list[str]


class Baseline(NamedTuple):
    """Model v1, the fixed point every construction is measured against."""

    model: LGBMClassifier
    pred: np.ndarray
    f1: float
    shap: pd.DataFrame
    mean_abs: pd.Series
    rank: pd.Series


class VariantResult(NamedTuple):
    """What one construction produced, measured against v1 on the same rows."""

    f1: float
    delta_f1: float
    agreement: float
    identical: bool
    adv: BootstrapInterval | None
    psi: pd.Series
    mean_abs: pd.Series
    affected: list[str]


def split_gain_penalty(feature: str, weight: float, feature_order: list[str]) -> list[float]:
    """LightGBM per-feature split-gain multiplier; 0.0 denies the feature entirely.

    Parameters
    ----------
    feature : str
        Feature to reweight.
    weight : float
        Multiplier applied to that feature's split gain.
    feature_order : list[str]
        Training column order the returned list is aligned to.

    Returns
    -------
    list[float]
        One multiplier per column, 1.0 everywhere but ``feature``.
    """
    return [weight if col == feature else 1.0 for col in feature_order]


def build_constructions(feature_order: list[str]) -> list[Construction]:
    """The four version-2 recipes the paper reports.

    A single constructed regression would invite the objection that it is the one
    setting where the signal happens to work, so this spans a true null control,
    an un-engineered configuration change, and two deliberate deprioritisations.

    Parameters
    ----------
    feature_order : list[str]
        Training column order, needed by the split-gain penalties.

    Returns
    -------
    list[Construction]
        The recipes, null control first.
    """
    return [
        Construction("identical retrain (null control)", {}, []),
        Construction(
            "hyperparameter change",
            dict(
                random_state=7,
                num_leaves=63,
                colsample_bytree=0.6,
                learning_rate=0.05,
                subsample=0.8,
                subsample_freq=1,
            ),
            [],
        ),
        Construction(
            "`relationship` deprioritised",
            dict(feature_contri=split_gain_penalty("relationship", 0.0, feature_order)),
            ["relationship", "marital-status", "sex"],
        ),
        Construction(
            "`education-num` dropped",
            dict(feature_contri=split_gain_penalty("education-num", 0.0, feature_order)),
            ["education-num", "education"],
        ),
    ]


def fit_baseline(split: StudySplit) -> Baseline:
    """Fit model v1 and cache the quantities every comparison needs.

    Parameters
    ----------
    split : StudySplit
        Training and scoring rows.

    Returns
    -------
    Baseline
        The fitted model with its predictions, F1, attributions and rankings.
    """
    model = fit_model(split)
    pred = model.predict(split.X_val)
    shap = explain(model, split.X_val)
    mean_abs = shap.abs().mean()
    return Baseline(
        model=model,
        pred=pred,
        f1=f1_score(split.y_val, pred),
        shap=shap,
        mean_abs=mean_abs,
        rank=mean_abs.rank(ascending=False),
    )


def fit_variants(
    split: StudySplit, baseline: Baseline, constructions: list[Construction]
) -> dict[str, VariantResult]:
    """Fit every construction and measure each against ``baseline``.

    Prints one line per fit, since the fits take a noticeable moment each.

    Parameters
    ----------
    split : StudySplit
        Training and scoring rows, identical for every version.
    baseline : Baseline
        Model v1 to compare against.
    constructions : list[Construction]
        Recipes to fit.

    Returns
    -------
    dict[str, VariantResult]
        One result per construction, keyed by its name.
    """
    variants: dict[str, VariantResult] = {}
    for construction in constructions:
        model = fit_model(split, **construction.kwargs)
        pred = model.predict(split.X_val)
        f1 = f1_score(split.y_val, pred)
        shap = explain(model, split.X_val)
        identical = bool(np.allclose(baseline.shap.to_numpy(), shap.to_numpy()))

        variants[construction.name] = VariantResult(
            f1=f1,
            delta_f1=f1 - baseline.f1,
            agreement=float((baseline.pred == pred).mean()),
            identical=identical,
            # A v2 identical to v1 has nothing to separate; running the adversarial
            # classifier on a sample against itself duplicates every row under opposite
            # labels and scores BELOW chance -- an artefact, not a measurement.
            adv=None if identical else adv_auc_ci(baseline.shap, shap),
            psi=shap_psi(baseline.shap, shap),
            mean_abs=shap.abs().mean(),
            affected=construction.affected,
        )
        print(f"  fitted: {construction.name}")
    return variants


def variants_table(variants: dict[str, VariantResult]) -> pd.DataFrame:
    """Study B's table: accuracy, prediction agreement, and what the monitor flagged.

    Parameters
    ----------
    variants : dict[str, VariantResult]
        Results from :func:`fit_variants`.

    Returns
    -------
    pd.DataFrame
        One row per construction, indexed by name.
    """
    rows = []
    for name, variant in variants.items():
        alerting = variant.psi[variant.psi >= PSI_ALERT]
        flagged = list(alerting.sort_values(ascending=False, kind="stable").index)
        rows.append(
            {
                "construction": name,
                "F1": round(variant.f1, 4),
                "dF1": round(variant.delta_f1, 4),
                "agreement": f"{variant.agreement:.1%}",
                "raw input PSI": 0.0,  # zero by construction: both versions score identical rows
                "SHAP AUC": "--" if variant.adv is None else round(variant.adv["point"], 3),
                "flagged": len(flagged),
                "which": ", ".join(flagged[:3]) or "none",
            }
        )
    return pd.DataFrame(rows).set_index("construction")


def variant_detail(baseline: Baseline, variant: VariantResult) -> pd.DataFrame:
    """Per-feature view of one construction: importance, rank and PSI on both sides.

    Parameters
    ----------
    baseline : Baseline
        Model v1.
    variant : VariantResult
        The construction to detail.

    Returns
    -------
    pd.DataFrame
        Indexed by feature, sorted by PSI descending.
    """
    return pd.DataFrame(
        {
            "mean_abs_v1": baseline.mean_abs,
            "mean_abs_v2": variant.mean_abs,
            "rank_v1": baseline.rank.astype(int),
            "rank_v2": variant.mean_abs.rank(ascending=False).astype(int),
            "psi": variant.psi,
        }
    ).sort_values("psi", ascending=False)
