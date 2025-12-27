# Monitoring with SHAPMonitor

`SHAPMonitor` is the core class for logging SHAP explanations in production.

## Basic Usage

```python
from shapmonitor import SHAPMonitor
import shap

# Create your SHAP explainer
explainer = shap.TreeExplainer(model)

# Initialize monitor
monitor = SHAPMonitor(
    explainer=explainer,
    data_dir="./shap_logs",
    sample_rate=0.1,
    model_version="v1.0",
    feature_names=["feature_1", "feature_2", "feature_3"]
)

# Log a batch of predictions
monitor.log_batch(X_batch)
```

## Configuration Options

### explainer (required)

The SHAP explainer to use for computing explanations. Must be a SHAP explainer object (TreeExplainer, LinearExplainer, etc.).

```python
# Tree-based models
explainer = shap.TreeExplainer(model)

# Linear models
explainer = shap.LinearExplainer(model, X_train)

# Any model (slower)
explainer = shap.KernelExplainer(model.predict, X_train)
```

### data_dir (required if no backend)

Directory where explanation logs will be stored. Files are organized by date:

```
shap_logs/
  2025-12-26/
    uuid1.parquet
    uuid2.parquet
  2025-12-27/
    uuid3.parquet
```

### sample_rate (optional, default=0.1)

Fraction of predictions to log (0.0 to 1.0). Use sampling to reduce storage and computation:

```python
# Log all predictions (expensive!)
monitor = SHAPMonitor(explainer=explainer, data_dir="./logs", sample_rate=1.0)

# Log 10% of predictions (recommended)
monitor = SHAPMonitor(explainer=explainer, data_dir="./logs", sample_rate=0.1)

# Log 1% of predictions (high-volume systems)
monitor = SHAPMonitor(explainer=explainer, data_dir="./logs", sample_rate=0.01)
```

### model_version (optional, default="unknown")

Version identifier for the model. Useful for tracking multiple model versions:

```python
monitor_v1 = SHAPMonitor(
    explainer=explainer_v1,
    data_dir="./logs",
    model_version="v1.0"
)

monitor_v2 = SHAPMonitor(
    explainer=explainer_v2,
    data_dir="./logs",
    model_version="v2.0"
)
```

### feature_names (optional)

List of feature names. If not provided and X is a DataFrame, column names are used. Otherwise, features are named `feat_0`, `feat_1`, etc.

```python
# Explicit feature names
monitor = SHAPMonitor(
    explainer=explainer,
    data_dir="./logs",
    feature_names=["age", "income", "score"]
)

# Automatically inferred from DataFrame
X_df = pd.DataFrame(X, columns=["age", "income", "score"])
monitor.log_batch(X_df)  # Uses DataFrame column names
```

### backend (optional)

Custom storage backend. If not provided, ParquetBackend is used with `data_dir`.

```python
from shapmonitor.backends import ParquetBackend

# Use custom backend
backend = ParquetBackend("/custom/path")
monitor = SHAPMonitor(explainer=explainer, backend=backend)
```

## Logging Methods

### log_batch()

Log SHAP explanations for a batch of predictions:

```python
# Basic usage
monitor.log_batch(X_batch)

# With predictions (optional)
monitor.log_batch(X_batch, y_pred)
```

Parameters:

- `X`: Input features (2D array or DataFrame)
- `y`: Optional predictions (1D array)

The method:

1. Checks if the batch should be sampled (based on `sample_rate`)
2. If sampled, computes SHAP values
3. Stores results to backend with timestamp and batch ID

### compute()

Compute SHAP values without logging:

```python
explanation = monitor.compute(X)

# Access components
shap_values = explanation.values        # SHAP values array
base_values = explanation.base_values   # Expected value(s)
```

Useful for:

- Interactive analysis
- Testing explainer configuration
- One-off explanations

## Production Integration

### Typical Production Pattern

```python
import logging
from shapmonitor import SHAPMonitor

logger = logging.getLogger(__name__)

# Initialize once (e.g., at service startup)
monitor = SHAPMonitor(
    explainer=explainer,
    data_dir="/var/log/shap",
    sample_rate=0.05,
    model_version="prod-v2.1"
)

# In prediction endpoint
def predict(request):
    X = preprocess(request.data)
    predictions = model.predict(X)

    # Log explanations (async recommended for production)
    try:
        monitor.log_batch(X, predictions)
    except Exception as e:
        logger.error(f"Failed to log explanations: {e}")

    return predictions
```

### Batch Processing Pattern

```python
# Process data in batches
for batch in data_loader:
    X_batch = batch.features
    y_pred = model.predict(X_batch)

    # Log each batch
    monitor.log_batch(X_batch, y_pred)
```

## Performance Considerations

### Sampling Strategy

Choose appropriate `sample_rate` based on:

- **Prediction volume**: Higher volume → lower sample rate
- **Storage capacity**: Limited storage → lower sample rate
- **Analysis needs**: Need more data → higher sample rate

Recommended rates:

- Low volume (<1000/day): 0.5 - 1.0
- Medium volume (1000-100k/day): 0.1 - 0.5
- High volume (>100k/day): 0.01 - 0.1

### Explainer Performance

Tree explainers are fast, but other explainers can be slow:

```python
# Fast (milliseconds)
shap.TreeExplainer(lightgbm_model)
shap.TreeExplainer(xgboost_model)
shap.LinearExplainer(linear_model, X_train)

# Slow (seconds to minutes)
shap.KernelExplainer(model.predict, X_train)
```

### Asynchronous Logging

For production systems, consider async logging (coming in v0.2):

```python
# Current (synchronous)
monitor.log_batch(X)  # Blocks until complete

# Future (asynchronous)
await monitor.log_batch_async(X)  # Non-blocking
```

## Data Storage

Logged data is stored with:

- **Timestamp**: When the batch was logged
- **Batch ID**: Unique identifier (UUID)
- **Model version**: From configuration
- **SHAP values**: One column per feature (`shap_feature_name`)
- **Feature values**: Original feature values (`feature_name`)
- **Base values**: Expected values from explainer
- **Predictions**: If provided to `log_batch()`

Example Parquet structure:

| timestamp | batch_id | model_version | n_samples | base_value | shap_age | shap_income | age | income |
|-----------|----------|---------------|-----------|------------|----------|-------------|-----|--------|
| 2025-... | uuid1 | v1.0 | 100 | 0.5 | 0.2 | -0.1 | 35 | 50000 |

## Properties

Access monitor configuration:

```python
monitor.explainer       # The SHAP explainer
monitor.backend         # Storage backend
monitor.sample_rate     # Configured sample rate
monitor.model_version   # Model version identifier
monitor.feature_names   # Feature names
```

## Error Handling

Handle common errors:

```python
# Missing both data_dir and backend
try:
    monitor = SHAPMonitor(explainer=explainer)
except ValueError as e:
    print("Must provide data_dir or backend")

# Invalid sample_rate
monitor = SHAPMonitor(
    explainer=explainer,
    data_dir="./logs",
    sample_rate=1.5  # Should be 0.0-1.0
)
```
