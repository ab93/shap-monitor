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
- `sort_by`: Column to sort by (default: `"mean_abs"`)

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
