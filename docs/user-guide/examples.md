# Examples

Complete working examples demonstrating shap-monitor usage.

## LightGBM Example

This example demonstrates real-world usage with LightGBM on the California Housing dataset.

The complete code is available in [`examples/demo_lightgbm.py`](https://github.com/ab93/shap-monitor/blob/main/examples/demo_lightgbm.py).

### Overview

The example shows:

1. Training a LightGBM model
2. Creating a SHAP explainer
3. Logging explanations with SHAPMonitor
4. Simulating production batches over time
5. Analyzing results with SHAPAnalyzer

### Running the Example

```bash
# From project root
poetry run python examples/demo_lightgbm.py
```

### Code Walkthrough

#### 1. Load Data and Train Model

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import lightgbm as lgb

# Load California Housing dataset
data = fetch_california_housing(as_frame=True)
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train LightGBM model
model = lgb.LGBMRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    verbosity=-1
)
model.fit(X_train, y_train)
```

#### 2. Initialize Monitor

```python
import shap
from shapmonitor import SHAPMonitor

# Create SHAP explainer
explainer = shap.TreeExplainer(model)

# Initialize monitor with 100% sampling for demo
monitor = SHAPMonitor(
    explainer=explainer,
    data_dir=".demo_shap_logs",
    sample_rate=1.0,  # Log everything for demo
    model_version="lgbm-v1",
    feature_names=list(X.columns)
)
```

#### 3. Simulate Production Batches

```python
from datetime import datetime, timedelta
from freezegun import freeze_time

# Simulate logging batches across multiple days
def simulate_production_batches(monitor, X, batch_size=100, days_to_simulate=7):
    n_batches = len(X) // batch_size
    batches_per_day = n_batches // days_to_simulate

    base_date = datetime.now() - timedelta(days=days_to_simulate)
    batch_idx = 0

    for day in range(days_to_simulate):
        frozen_date = base_date + timedelta(days=day)

        with freeze_time(frozen_date):
            for _ in range(batches_per_day):
                if batch_idx >= n_batches:
                    break

                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                batch = X.iloc[start_idx:end_idx]

                monitor.log_batch(batch)
                batch_idx += 1

# Run simulation
simulate_production_batches(monitor, X_test, batch_size=100, days_to_simulate=7)
```

#### 4. Analyze Results

```python
from shapmonitor.analysis import SHAPAnalyzer
from shapmonitor.backends import ParquetBackend

# Create analyzer
backend = ParquetBackend(".demo_shap_logs")
analyzer = SHAPAnalyzer(backend)

# Get summary statistics
today = datetime.now()
week_ago = today - timedelta(days=7)
summary = analyzer.summary(week_ago, today)

print(summary)
print(f"Total samples: {summary.attrs['n_samples']}")
```

Example output:

```
                mean_abs      mean       std       min       max
feature
MedInc          0.456123  0.412456  0.187234 -0.234567  0.891234
AveRooms        0.234567 -0.198765  0.145678 -0.567890  0.234567
HouseAge        0.198765  0.176543  0.134567 -0.156789  0.456789
AveBedrms       0.176543 -0.165432  0.123456 -0.445678  0.345678
Population      0.145678  0.134567  0.112345 -0.334567  0.434567
AveOccup        0.134567  0.123456  0.101234 -0.223456  0.323456
Latitude        0.123456  0.112345  0.098765 -0.212345  0.312345
Longitude       0.112345  0.101234  0.087654 -0.201234  0.301234

Total samples: 6180
```

#### 5. Compare Time Periods

```python
# Compare first half vs second half of week
period_1_start = today - timedelta(days=7)
period_1_end = today - timedelta(days=4)
period_2_start = today - timedelta(days=3)
period_2_end = today

comparison = analyzer.compare_time_periods(
    period_1_start, period_1_end,
    period_2_start, period_2_end
)

print(comparison)
```

Example output:

```
              mean_abs_1  mean_abs_2  delta_mean_abs  pct_delta_mean_abs  rank_1  rank_2  delta_rank rank_change    mean_1    mean_2  sign_flip
MedInc             0.456       0.445          -0.011                -2.4     1.0     1.0         0.0  no_change      0.412     0.398      False
AveRooms           0.235       0.289           0.054                23.0     2.0     2.0         0.0  no_change     -0.199    -0.267      False
HouseAge           0.199       0.187          -0.012                -6.0     3.0     3.0         0.0  no_change      0.177     0.165      False
```

### Key Takeaways

1. **Setup is simple**: Just create explainer and monitor
2. **Logging is automatic**: Monitor handles SHAP computation and storage
3. **Analysis is powerful**: SHAPAnalyzer provides rich insights
4. **Time-based analysis**: Compare periods to detect drift

### Full Example Code

See the complete implementation in the repository:

[`examples/demo_lightgbm.py`](https://github.com/ab93/shap-monitor/blob/main/examples/demo_lightgbm.py)

## Minimal Example

A minimal example to get started quickly:

```python
from shapmonitor import SHAPMonitor
from shapmonitor.analysis import SHAPAnalyzer
from shapmonitor.backends import ParquetBackend
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from datetime import datetime, timedelta

# Train model
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Setup monitoring
explainer = shap.TreeExplainer(model)
monitor = SHAPMonitor(
    explainer=explainer,
    data_dir="./shap_logs",
    sample_rate=0.1,
    model_version="v1.0"
)

# Log some data
monitor.log_batch(X[:100])
monitor.log_batch(X[100:200])
monitor.log_batch(X[200:300])

# Analyze
backend = ParquetBackend("./shap_logs")
analyzer = SHAPAnalyzer(backend)

today = datetime.now()
summary = analyzer.summary(today, today)
print(summary)
```

## Additional Examples

### XGBoost Example

```python
import xgboost as xgb
import shap
from shapmonitor import SHAPMonitor

# Train XGBoost model
dtrain = xgb.DMatrix(X_train, label=y_train)
params = {'max_depth': 3, 'eta': 0.1, 'objective': 'reg:squarederror'}
model = xgb.train(params, dtrain, num_boost_round=100)

# Create explainer and monitor
explainer = shap.TreeExplainer(model)
monitor = SHAPMonitor(
    explainer=explainer,
    data_dir="./xgb_logs",
    sample_rate=0.1,
    model_version="xgb-v1"
)

# Log predictions
dtest = xgb.DMatrix(X_test)
monitor.log_batch(X_test)
```

### Linear Model Example

```python
from sklearn.linear_model import LogisticRegression
import shap
from shapmonitor import SHAPMonitor

# Train logistic regression
model = LogisticRegression()
model.fit(X_train, y_train)

# Create explainer (needs background data)
explainer = shap.LinearExplainer(model, X_train)

# Monitor
monitor = SHAPMonitor(
    explainer=explainer,
    data_dir="./linear_logs",
    sample_rate=0.2,
    model_version="logreg-v1"
)

monitor.log_batch(X_test)
```

### Production Deployment Pattern

```python
import logging
from shapmonitor import SHAPMonitor

logger = logging.getLogger(__name__)

class PredictionService:
    def __init__(self, model, explainer):
        self.model = model
        self.monitor = SHAPMonitor(
            explainer=explainer,
            data_dir="/var/log/shap",
            sample_rate=0.05,  # 5% sampling
            model_version="prod-v2.1"
        )

    def predict(self, X):
        predictions = self.model.predict(X)

        # Log asynchronously in production
        try:
            self.monitor.log_batch(X, predictions)
        except Exception as e:
            logger.error(f"Failed to log SHAP: {e}")

        return predictions
```

## Next Steps

- Explore the [API Reference](../api/index.md) for detailed documentation
- Read about [Monitoring](monitoring.md) best practices
- Learn advanced [Analysis](analysis.md) techniques
- Understand [Storage Backends](backends.md) options
