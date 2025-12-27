# Quick Start

Get started with shap-monitor in just a few minutes.

## Basic Example

Here's a minimal example using a RandomForest classifier:

```python
from shapmonitor import SHAPMonitor
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Train a simple model
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Create SHAP explainer
explainer = shap.TreeExplainer(model)

# Initialize the monitor
monitor = SHAPMonitor(
    explainer=explainer,
    data_dir="./shap_logs",
    sample_rate=0.1,  # Log 10% of predictions
    model_version="v1.0",
    feature_names=[f"feature_{i}" for i in range(10)]
)

# Log a batch of predictions
monitor.log_batch(X[:100])
```

## What Just Happened?

1. **Created a SHAP explainer** - TreeExplainer for tree-based models
2. **Initialized SHAPMonitor** - Configured to log 10% of predictions to `./shap_logs`
3. **Logged explanations** - Computed and stored SHAP values for 100 predictions

The monitor automatically:

- Samples predictions based on `sample_rate` (10% in this example)
- Computes SHAP values using the explainer
- Stores explanations to Parquet files organized by date

## Computing SHAP Values Directly

You can also compute SHAP values without logging:

```python
# Compute SHAP values for analysis
explanation = monitor.compute(X[:10])

# Access the values
shap_values = explanation.values  # SHAP values
base_values = explanation.base_values  # Base values (expected value)
```

## Analyzing Logged Data

After logging data, you can analyze it using `SHAPAnalyzer`:

```python
from datetime import datetime, timedelta
from shapmonitor.analysis import SHAPAnalyzer
from shapmonitor.backends import ParquetBackend

# Create backend and analyzer
backend = ParquetBackend("./shap_logs")
analyzer = SHAPAnalyzer(backend)

# Get summary statistics for the past 7 days
today = datetime.now()
week_ago = today - timedelta(days=7)
summary = analyzer.summary(week_ago, today)

print(summary)
# Output shows feature importance (mean_abs), direction (mean), and variability (std)
```

## Comparing Time Periods

Detect changes in feature importance over time:

```python
# Compare this week vs last week
last_week = today - timedelta(days=14)
two_weeks_ago = today - timedelta(days=21)

comparison = analyzer.compare_time_periods(
    start_1=two_weeks_ago,
    end_1=last_week,
    start_2=last_week,
    end_2=today
)

print(comparison)
# Shows changes in feature importance rankings and values
```

## Complete Working Example

See the [LightGBM example](../user-guide/examples.md#lightgbm-example) for a complete working example with the California Housing dataset.

## Next Steps

- Learn about [monitoring in production](../user-guide/monitoring.md)
- Explore [analysis capabilities](../user-guide/analysis.md)
- Understand [storage backends](../user-guide/backends.md)
- Check out the [API Reference](../api/index.md)
