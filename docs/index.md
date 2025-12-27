# shap-monitor

Production ML explainability toolkit for monitoring SHAP values over time. Track how your model's explanations evolve, detect explanation drift, and maintain interpretability at scale.

![Work in Progress](https://img.shields.io/badge/status-work%20in%20progress-yellow)
![Python Version](https://img.shields.io/badge/python-3.11+-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-green)
[![codecov](https://codecov.io/gh/ab93/shap-monitor/branch/main/graph/badge.svg)](https://codecov.io/gh/ab93/shap-monitor)

## Overview

Most SHAP tooling focuses on development and analysis. **shap-monitor** bridges the gap for production monitoring, helping ML teams understand when and how their models' explanations change over time, compare reasoning across model versions, and detect shifts in feature importance patterns.

## Key Features

- **Explanation Logging**: Automatically log SHAP values for production predictions with configurable sampling
- **Flexible Storage**: Parquet-based storage with pluggable backend support for efficient storage and retrieval
- **Multiple Explainers**: Works with TreeExplainer, LinearExplainer, and other SHAP explainers
- **Time-Series Analysis**: Compare SHAP explanations across different time periods to detect drift
- **Feature Importance Tracking**: Monitor how feature importance evolves over time
- **Model Version Comparison**: Compare explanations across different model versions

## Quick Example

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

# In your prediction loop
predictions = model.predict(X[:100])
monitor.log_batch(X[:100])
```

## Why shap-monitor?

### For ML Engineers

- Monitor model explanations in production alongside performance metrics
- Detect when your model's reasoning changes unexpectedly
- Debug model behavior by analyzing SHAP values over time
- Set up alerts for explanation drift

### For Data Scientists

- Understand how feature importance evolves as data distributions shift
- Compare model versions based on their explanations, not just accuracy
- Identify features that become more or less important over time
- Validate that deployed models reason as expected

### For ML Platform Teams

- Provide teams with explanation monitoring infrastructure
- Enable reproducible analysis of model behavior
- Support model debugging and incident response
- Integrate with existing MLOps tooling

## Current Status

This project is in early development (v0.1). The core functionality is being actively developed.

### Roadmap

- **v0.1 (Current)**: Core synchronous monitoring, Parquet storage
- **v0.2 (Planned)**: Drift detection, asynchronous processing, MLflow integration
- **v0.3+ (Future)**: Dashboard/visualization, additional framework integrations, advanced alerting

See the [Roadmap](development/roadmap.md) for detailed information.

## Getting Started

Ready to start monitoring your model's explanations?

<div class="grid cards" markdown>

-   :material-clock-fast:{ .lg .middle } **Quick Start**

    ---

    Get started with shap-monitor in minutes

    [:octicons-arrow-right-24: Quick Start Guide](getting-started/quickstart.md)

-   :material-book-open-variant:{ .lg .middle } **User Guide**

    ---

    Learn how to use shap-monitor effectively

    [:octicons-arrow-right-24: Read the User Guide](user-guide/index.md)

-   :material-api:{ .lg .middle } **API Reference**

    ---

    Detailed API documentation for all components

    [:octicons-arrow-right-24: Browse API Docs](api/index.md)

-   :material-github:{ .lg .middle } **Contributing**

    ---

    Help build the future of ML explainability monitoring

    [:octicons-arrow-right-24: Contribution Guide](development/contributing.md)

</div>

## Acknowledgments

Built on top of the excellent [SHAP](https://github.com/slundberg/shap) library by Scott Lundberg and the SHAP community.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](https://github.com/ab93/shap-monitor/blob/main/LICENSE) file for details.
