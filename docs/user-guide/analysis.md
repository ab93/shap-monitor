# Analysis with SHAPAnalyzer

`SHAPAnalyzer` provides tools for analyzing logged SHAP explanations.

## Basic Usage

```python
from datetime import datetime, timedelta
from shapmonitor.analysis import SHAPAnalyzer
from shapmonitor.backends import ParquetBackend

# Create backend and analyzer
backend = ParquetBackend("./shap_logs")
analyzer = SHAPAnalyzer(backend)

# Analyze data from the past week
today = datetime.now()
week_ago = today - timedelta(days=7)
summary = analyzer.summary(week_ago, today)
```

## Configuration Options

### backend (required)

The storage backend to read data from:

```python
from shapmonitor.backends import ParquetBackend

backend = ParquetBackend("/path/to/logs")
analyzer = SHAPAnalyzer(backend)
```

### min_abs_shap (optional, default=0.0)

Minimum mean absolute SHAP value threshold. Features below this threshold are excluded from results:

```python
# Include all features
analyzer = SHAPAnalyzer(backend, min_abs_shap=0.0)

# Exclude low-impact features
analyzer = SHAPAnalyzer(backend, min_abs_shap=0.01)
```

Useful for:

- Filtering noise in high-dimensional data
- Focusing on important features
- Reducing output size

## Analysis Methods

### summary()

Compute summary statistics for SHAP values over a time period:

```python
summary = analyzer.summary(
    start_dt=datetime(2025, 12, 20),
    end_dt=datetime(2025, 12, 27),
    sort_by="mean_abs"  # Optional, default
)
```

**Parameters:**

- `start_dt`: Start of date range (inclusive)
- `end_dt`: End of date range (inclusive)
- `batch_id`: Filter to a specific batch ID
- `model_version`: Filter to a specific model version
- `sort_by`: Column to sort by (default: `"mean_abs"`)
- `top_k`: If set, return only the top k features after sorting

**Returns:**

DataFrame with columns:

- `mean_abs`: Mean of absolute SHAP values (feature importance)
- `mean`: Mean SHAP value (contribution direction)
- `std`: Standard deviation
- `min`: Minimum SHAP value
- `max`: Maximum SHAP value

**Attributes:**

- `n_samples`: Total number of samples analyzed

**Example output:**

```python
print(summary)
#                 mean_abs      mean       std       min       max
# feature
# MedInc          0.345     0.312     0.156    -0.234     0.891
# AveRooms        0.234    -0.198     0.145    -0.567     0.234
# Latitude        0.189     0.167     0.123    -0.345     0.456

print(summary.attrs['n_samples'])
# 5420
```

**Use cases:**

- Identify most important features
- Understand feature contribution direction
- Detect high-variance features
- Monitor feature importance over time

### compare_time_periods()

Compare SHAP explanations between two time periods:

```python
# Using plain tuples
comparison = analyzer.compare_time_periods(
    (datetime(2025, 12, 1), datetime(2025, 12, 15)),   # reference period
    (datetime(2025, 12, 16), datetime(2025, 12, 30)),  # current period
)

# Or using Period for named fields
from shapmonitor.types import Period

comparison = analyzer.compare_time_periods(
    Period(datetime(2025, 12, 1), datetime(2025, 12, 15)),
    Period(datetime(2025, 12, 16), datetime(2025, 12, 30)),
)
```

**Parameters:**

- `period_ref`: Reference time period as `Period(start, end)` or tuple
- `period_curr`: Current time period as `Period(start, end)` or tuple
- `sort_by`: Column to sort by (default: `"psi"`)
- `top_k`: If set, return only the top k features after sorting

**Returns:**

DataFrame with columns:

**Drift Detection:**

- `psi`: Population Stability Index between periods (primary drift metric)

**Feature Importance:**

- `mean_abs_1`, `mean_abs_2`: Feature importance per period
- `delta_mean_abs`: Absolute change (period_2 - period_1)
- `pct_delta_mean_abs`: Percentage change from period_1

**Rankings:**

- `rank_1`, `rank_2`: Feature importance rank per period
- `delta_rank`: Rank change (positive = became less important)
- `rank_change`: "increased", "decreased", or "no_change"

**Direction:**

- `mean_1`, `mean_2`: Mean SHAP value per period
- `sign_flip`: True if contribution direction changed

**PSI Interpretation:**

| PSI Value  | Interpretation              |
|------------|-----------------------------|
| 0          | Identical distributions     |
| < 0.1      | No significant shift        |
| 0.1 - 0.25 | Moderate shift, investigate |
| 0.25 - 0.5 | Significant shift           |
| > 0.5      | Severe shift                |

**Attributes:**

- `n_samples_1`: Sample count in period 1
- `n_samples_2`: Sample count in period 2

**Example output:**

```python
print(comparison)
#              psi  mean_abs_1  mean_abs_2  delta_mean_abs  ...  rank_change  sign_flip
# MedInc      0.08       0.345       0.289          -0.056  ...    no_change      False
# AveRooms    0.15       0.234       0.312           0.078  ...    no_change      False
# Latitude    0.32       0.189       0.145          -0.044  ...    decreased      False

print(comparison.attrs['n_samples_1'])  # 2430
print(comparison.attrs['n_samples_2'])  # 2990
```

**Use cases:**

- Detect feature importance drift
- Identify ranking changes
- Spot sign flips (feature contributions reversing)
- Compare model behavior across deployments

### compare_adversarial()

Detect distributional shift between two periods using adversarial validation — a model-based approach that complements PSI. A binary classifier is trained to distinguish SHAP values from `period_ref` vs `period_curr`. If it can, the distributions differ.

```python
from shapmonitor.types import Period

result = analyzer.compare_adversarial(
    Period(datetime(2025, 12, 1), datetime(2025, 12, 15)),   # reference
    Period(datetime(2025, 12, 16), datetime(2025, 12, 30)),  # current
    random_state=42,
)

# Overall shift score
print(result.attrs["adversarial_auc"])  # 0.5 = no shift, 1.0 = maximum shift

# Per-feature breakdown
print(result)
#              adv_importance  mean_abs_1  mean_abs_2  delta_mean_abs
# feature_a          0.41          0.31        0.62            0.31
# feature_b          0.34          0.52        0.22           -0.30
```

**Parameters:**

- `period_ref`: Reference time period
- `period_curr`: Current time period
- `classifier`: Any sklearn-compatible classifier with `feature_importances_` (default: `RandomForestClassifier`)
- `cv`: Number of cross-validation folds (default: `5`)
- `sort_by`: Column to sort by (default: `"adv_importance"`)
- `top_k`: If set, return only the top k features
- `random_state`: Seed for reproducibility

**Returns:**

DataFrame columns:

- `adv_importance`: How much each SHAP dimension contributes to making the two periods separable
- `mean_abs_1`, `mean_abs_2`: Mean absolute SHAP value per period
-
- `delta_mean_abs`: Absolute change in importance (period 2 − period 1)

Attributes:

- `adversarial_auc`: Cross-validated AUC — the headline distributional shift score
- `n_samples_ref`, `n_samples_curr`: Sample counts

**Adversarial AUC interpretation:**

| AUC | Interpretation |
|---|---|
| 0.50 | Distributions are indistinguishable |
| 0.50–0.65 | Minor differences, likely noise |
| 0.65–0.80 | Moderate shift — worth investigating |
| 0.80–0.90 | Strong shift detected |
| > 0.90 | Severe — clearly different regimes |

**Why use this alongside `compare_time_periods()`?**

`compare_time_periods` tests each feature independently via PSI (univariate). Adversarial validation tests all SHAP dimensions at once (multivariate) and can detect subtle joint distributional shifts that no single feature's PSI would flag. Use both together — join on the feature index if you want PSI and adversarial importance side by side:

```python
adv = analyzer.compare_adversarial(period_ref, period_curr)
psi = analyzer.compare_time_periods(period_ref, period_curr)[["psi"]]
combined = adv.join(psi)
```

### adversarial_auc() — standalone function

For adversarial validation on **raw input feature distributions** (rather than SHAP values), use `adversarial_auc` directly from `shapmonitor.analysis.metrics`. It accepts any two DataFrames:

```python
from shapmonitor.analysis.metrics import adversarial_auc

# Read and filter to feature columns (logged via log_batch)
feat_ref = backend.read(start_dt=datetime(2025, 12, 1), end_dt=datetime(2025, 12, 15)).filter(like="feat_")
feat_curr = backend.read(start_dt=datetime(2025, 12, 16), end_dt=datetime(2025, 12, 30)).filter(like="feat_")

auc, importances = adversarial_auc(feat_ref, feat_curr, random_state=42)

print(f"Feature AUC: {auc:.3f}")    # How separable are the input distributions?
print(importances.sort_values(ascending=False))  # Which features drive the separation?
```

This is intentionally kept separate from `SHAPAnalyzer` — the analyzer focuses on SHAP value distributions. Feature-level adversarial analysis is a data quality concern that any user can invoke directly.

## Practical Examples

### Weekly Monitoring Report

```python
from datetime import datetime, timedelta

# Get current week and previous week
today = datetime.now()
this_week_start = today - timedelta(days=7)
last_week_start = today - timedelta(days=14)

# Compare weeks
comparison = analyzer.compare_time_periods(
    (last_week_start, this_week_start),
    (this_week_start, today),
)

# Find features with significant PSI (distribution shift)
drifted_features = comparison[comparison['psi'] > 0.1]
print("Features with significant drift (PSI > 0.1):")
print(drifted_features[['psi', 'mean_abs_1', 'mean_abs_2']])

# Find significant importance changes
significant_changes = comparison[
    abs(comparison['pct_delta_mean_abs']) > 20
]
print("Features with >20% importance change:")
print(significant_changes[['psi', 'mean_abs_1', 'mean_abs_2', 'pct_delta_mean_abs']])
```

### Ranking Changes

```python
# Find features that changed rank
rank_changes = comparison[comparison['rank_change'] != 'no_change']

print("Features with ranking changes:")
print(rank_changes[['rank_1', 'rank_2', 'delta_rank', 'rank_change']])

# Features that became more important
more_important = comparison[comparison['rank_change'] == 'increased']

# Features that became less important
less_important = comparison[comparison['rank_change'] == 'decreased']
```

### Sign Flip Detection

```python
# Find features that flipped contribution direction
sign_flips = comparison[comparison['sign_flip'] == True]

if not sign_flips.empty:
    print("WARNING: Features changed contribution direction!")
    print(sign_flips[['mean_1', 'mean_2', 'sign_flip']])
```

### Monthly Summary

```python
# Get summary for entire month
month_start = datetime(2025, 12, 1)
month_end = datetime(2025, 12, 31)

monthly_summary = analyzer.summary(month_start, month_end)

# Top 10 most important features
top_features = monthly_summary.head(10)

print("Top 10 features for December:")
print(top_features[['mean_abs', 'mean', 'std']])
```

### Filter Low-Impact Features

```python
# Only analyze features with significant impact
analyzer = SHAPAnalyzer(backend, min_abs_shap=0.05)

# Summary automatically excludes low-impact features
summary = analyzer.summary(start_date, end_date)
print(f"Analyzing {len(summary)} significant features")
```

### Adversarial Validation Workflow

```python
from shapmonitor.types import Period

ref_period = Period(datetime(2025, 11, 1), datetime(2025, 11, 30))
curr_period = Period(datetime(2025, 12, 1), datetime(2025, 12, 31))

# Step 1: Quick PSI-based comparison
comparison = analyzer.compare_time_periods(ref_period, curr_period)
drifted = comparison[comparison["psi"] > 0.1]
print(f"{len(drifted)} features show PSI drift")

# Step 2: Adversarial validation for a multivariate view
adv = analyzer.compare_adversarial(ref_period, curr_period, random_state=42)
print(f"Adversarial AUC: {adv.attrs['adversarial_auc']:.3f}")

# Step 3: Features flagged by both metrics are highest priority
high_priority = adv[
    (adv["adv_importance"] > 0.15) & (adv["psi"] > 0.1)
]
print("High-priority features (flagged by both PSI and adversarial):")
print(high_priority[["adv_importance", "psi", "delta_mean_abs"]])
```

### Handle Empty Results

```python
summary = analyzer.summary(start_date, end_date)

if summary.empty:
    print("No data found for date range")
else:
    print(f"Analyzed {summary.attrs['n_samples']} samples")
    print(summary)
```

## Properties

Access analyzer configuration:

```python
analyzer.backend        # Storage backend
analyzer.min_abs_shap   # Minimum threshold
```

## Understanding the Metrics

### mean_abs (Feature Importance)

Average absolute SHAP value across all samples. Higher values indicate more important features.

- Measures magnitude of feature impact
- Always positive
- Use for ranking features by importance

### mean (Contribution Direction)

Average SHAP value (with sign). Indicates typical contribution direction.

- Positive: Feature typically increases prediction
- Negative: Feature typically decreases prediction
- Close to zero: Contributions cancel out or vary

### std (Variability)

Standard deviation of SHAP values. Indicates consistency of feature impact.

- Low std: Feature has consistent impact
- High std: Feature impact varies significantly
- Compare to mean_abs to assess stability

### Rank Changes

- **rank_1, rank_2**: Feature position in importance ranking (1 = most important)
- **delta_rank**: Change in ranking (positive = dropped in importance)
- **rank_change**: Categorical change indicator

### Sign Flips

Indicates when a feature changes from positive to negative contribution (or vice versa).

- Can signal data distribution changes
- May indicate model instability
- Worth investigating for important features

### Adversarial AUC

A model-based distributional shift score. Closer to 0.5 means the two periods are indistinguishable; closer to 1.0 means strong separation.

- Multivariate — captures joint distributional shifts that univariate metrics miss
- `adv_importance` ranks which SHAP dimensions drive the separation
- Complements PSI: use PSI for per-feature significance, adversarial AUC for an overall summary signal

## Performance Tips

### Date Range Selection

- Smaller date ranges → faster queries
- Avoid querying very large time ranges unless needed
- Use appropriate granularity for your analysis

### Filtering with min_abs_shap

- Reduces computation for high-dimensional data
- Focuses analysis on actionable features
- Set threshold based on domain knowledge

### Memory Considerations

```python
# For very large datasets, process in smaller chunks
import pandas as pd

summaries = []
for week in date_ranges:
    summary = analyzer.summary(week.start, week.end)
    summaries.append(summary)

# Combine results
combined = pd.concat(summaries)
```
