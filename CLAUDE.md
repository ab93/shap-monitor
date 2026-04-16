# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development Commands

```bash
# Install all dependencies (dev, docs, examples)
make setup

# Run tests
make test

# Run a single test file
poetry run pytest tests/test_monitor.py -v

# Run a single test by name
poetry run pytest tests/test_monitor.py::TestSHAPMonitor::test_log_batch -v

# Format and lint code
make lint

# Run tests with coverage
make coverage

# Serve documentation locally
make docs-serve
```

## Architecture Overview

shap-monitor is a production ML explainability toolkit for monitoring SHAP values over time. It has three main layers:

### Core Components

**SHAPMonitor** (`shapmonitor/monitor.py`)
- Main entry point for logging SHAP explanations
- Takes a SHAP explainer, applies sampling (`sample_rate`), computes SHAP values, and writes to backend
- `log_batch(X, y)` - logs explanations for a batch of predictions
- `compute(X)` - returns raw SHAP explanation without logging

**Backends** (`shapmonitor/backends/`)
- `Backend` protocol in `types.py` defines the interface: `read()`, `write()`, `delete()`
- `ParquetBackend` (`_parquet.py`) - Hive-style partitioned Parquet storage (default partition: `date=YYYY-MM-DD/`)
- `BackendFactory` provides backend instantiation

**Analysis** (`shapmonitor/analysis/`)
- `SHAPAnalyzer` - computes summary statistics, compares time periods, detects drift
- Uses Population Stability Index (PSI) for drift detection
- `compare_time_periods()`, `compare_batches()`, `summary()` are key methods

### Data Flow

```
SHAPMonitor.log_batch(X, y)
    → apply sample_rate
    → compute SHAP values via explainer(X)
    → create ExplanationBatch
    → backend.write() → Parquet file
```

### Type System (`shapmonitor/types.py`)

Protocol-based typing for flexibility:
- `ExplainerLike` - any SHAP explainer (Tree, Linear, Kernel)
- `Backend` - storage interface protocol
- `ExplanationBatch` - dataclass wrapping batch metadata and SHAP results
- `Period` - NamedTuple for time range queries

## Testing Patterns

Tests use pytest fixtures from `tests/conftest.py`:
- `regression_data`, `classification_data` - sklearn generated datasets
- `feature_names` - standard 5-feature names
- `sample_features` - small 2-sample array for unit tests

Integration tests are marked with `@pytest.mark.integration`.

## Analysis Roadmap

Planned additions to `shapmonitor/analysis/`. In-code TODOs mark specific spots.

### New metrics (`metrics.py`)
- **KS statistic + p-value** — complements PSI with statistical significance; PSI alone can't tell you if a shift is meaningful on small samples (`scipy.stats.ks_2samp`)
- **Wasserstein distance** — bin-free alternative to PSI; geometrically intuitive ("how far does the distribution need to move?") (`scipy.stats.wasserstein_distance`)
- **Jensen-Shannon divergence** — symmetric, bounded [0, 1]; more stable than PSI on sparse distributions

### Enhancements to `summary()`
- **Percentile columns** (`p5`, `p25`, `p75`, `p95`) — `min`/`max` are outlier-prone; percentiles give the actual distribution shape
- **Coefficient of variation** (`cv = std / mean_abs`) — normalized volatility; distinguishes a stable high-importance feature from a noisy one with similar mean
- **Correlation with target** — Pearson/Spearman of SHAP values vs. prediction/target column; requires `log_batch(X, y)` data

### New `SHAPAnalyzer` methods
- **`drift_report(period_ref, period_curr)`** — wraps `compare_time_periods()` and classifies each feature as `"stable"` / `"warning"` / `"alert"` using PSI thresholds; returns an alert-ready DataFrame
- **`feature_trends(start_dt, end_dt, window)`** — fits a linear slope to mean_abs over rolling time windows; distinguishes gradual drift from sudden shifts
- **`anomaly_scores(period)`** — z-score or IQR-based outlier detection per batch; answers "which batch had abnormal SHAP distributions?"
- **`feature_coverage(start_dt, end_dt)`** — reports feature presence across batches; detects schema/pipeline drift early

### Adversarial validation
- **`adversarial_auc(reference_df, current_df, classifier=None, cv=5)`** in `metrics.py` — general-purpose function that accepts any two DataFrames (SHAP values, feature values, or any numeric data); trains a binary classifier to distinguish the two datasets and returns AUC + per-feature importances. AUC ≈ 0.5 = indistinguishable, AUC → 1.0 = strong distributional shift.
- **`SHAPAnalyzer.compare_adversarial(period_ref, period_curr)`** — high-level method that operates **only on SHAP value distributions**; fetches SHAP columns and calls `adversarial_auc` internally. Columns: `adv_importance`, `mean_abs_1`, `mean_abs_2`, `delta_mean_abs`. Attrs: `adversarial_auc`, `n_samples_ref`, `n_samples_curr`. PSI is intentionally excluded — use `compare_time_periods()` for PSI and join if both are needed.
- **`SHAPAnalyzer.compare_adversarial_batches(batch_ref, batch_curr)`** — same as above but compares two specific batch IDs instead of time periods. Both share `_run_adversarial_comparison()` internally.
- Default classifier: `sklearn.ensemble.RandomForestClassifier` (sklearn is a transitive dependency via shap); accept any sklearn-compatible classifier via `classifier=` param for users who want LightGBM/XGBoost sensitivity.
- **Adversarial analysis on input feature distributions is intentionally outside `SHAPAnalyzer`** — users who want this call `adversarial_auc` directly with `backend.read(...).filter(like="feat_")`. Keeping feature-level analysis out of `SHAPAnalyzer` preserves separation of concerns and avoids leaking backend column naming (`feat_*`) into the analysis API.

### Enhancements to comparison methods
- Add `ks_stat` + `ks_pvalue` columns alongside `psi` in `_compare_shap_dataframes()`; lets users use significance thresholds instead of (or in addition to) PSI thresholds

## Plotting Roadmap

Planned `shapmonitor.plotting` module — accepts DataFrames from `SHAPAnalyzer` methods directly.
matplotlib is already a dev dependency but **not a core dependency**; add it as an optional extra (`pip install shap-monitor[plot]`) with a soft import that raises a helpful `ImportError` if missing.

### API conventions
- All plot functions return `(fig, ax)` for composability
- Accept optional `ax=None` parameter so users can embed into subplot layouts
- `top_k=20` default to keep large feature sets readable
- Consistent severity color palette across all plots: green (stable) / amber (warning) / red (alert)

### Plots for `summary()`
- **`plot_importance(summary_df)`** — horizontal bar chart; bar length = `mean_abs`, error bars = `±std`; features ranked top-to-bottom

### Plots for `compare_*()`
- **`plot_importance_delta(compare_df)`** — diverging bar chart centered at 0; features that gained importance go right (green), lost go left (red); most scannable view of what changed between two periods/versions
- **`plot_rank_change(compare_df)`** — bump chart; features as lines connecting rank position in period 1 to period 2; crossing lines = rank inversions; label movers

### Plots for time-series monitoring
- **`plot_psi_heatmap(frames)`** — features on y-axis, rolling time windows on x-axis, PSI as color intensity (white → amber → red); best single-glance view of "what drifted when"
- **`plot_drift_report(report_df)`** — traffic light grid (dot matrix); features on rows, time windows on columns, green/amber/red cells by severity; extremely scannable for ops teams

## Code Style

- Python 3.10+, type hints required
- Black formatting (line length: 100)
- Ruff for linting
