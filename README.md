# shap-monitor

Production ML explainability toolkit for monitoring SHAP values over time. Track how your model's explanations evolve, detect explanation drift, and maintain interpretability at scale.

![Work in Progress](https://img.shields.io/badge/status-work%20in%20progress-yellow)
![Python Version](https://img.shields.io/badge/python-3.11+-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-green)
[![codecov](https://codecov.io/gh/ab93/shap-monitor/branch/main/graph/badge.svg)](https://codecov.io/gh/ab93/shap-monitor)

## Overview

Most SHAP tooling focuses on development and analysis. shap-monitor bridges the gap for production monitoring, helping ML teams understand when and how their models' explanations change over time, compare reasoning across model versions, and detect shifts in feature importance patterns.

## Key Features

- **Explanation Logging**: Automatically log SHAP values for production predictions with configurable sampling
- **Flexible Storage**: Parquet-based storage with pluggable backend support
- **Multiple Explainers**: Works with TreeExplainer, LinearExplainer, and other SHAP explainers

## Installation

You can install shap-monitor via pip:

```bash
pip install shap-monitor
```

### Installation from Source

This project uses Poetry for dependency management. Ensure you have Python 3.11 or higher installed.

```bash
# Clone the repository
git clone https://github.com/yourusername/shap-monitor.git
cd shap-monitor

# Install with Poetry
poetry install

# Or install for development with all dev dependencies
poetry install --with dev
```



## Quick Start

Here's a minimal example to get started with shap-monitor:

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
monitor.log(X[:100], predictions)

# Or compute SHAP values directly
explanation = monitor.compute(X[:10])
```

The monitor will automatically:
1. Sample predictions based on the configured `sample_rate`
2. Compute SHAP values for sampled predictions
3. Store explanations to Parquet files in the specified `data_dir`

## Current Status

This project is in early development (v0.1). The core functionality is being actively developed.

### Roadmap

- **v0.1 (Current)**: Core synchronous monitoring, Parquet storage
- **v0.2 (Planned)**: Drift detection, asynchronous processing, MLflow integration
- **v0.3+ (Future)**: Dashboard/visualization, additional framework integrations, advanced alerting

## Development

### Prerequisites

- Python 3.11 or higher
- Poetry for dependency management

### Setup

```bash
# Install development dependencies
make setup

# Install pre-commit hooks
poetry run pre-commit install

# Run tests
make test

# Format code
make lint
```

### Code Quality

This project uses:
- **black** for code formatting (line length: 100)
- **ruff** for linting
- **pytest** for testing
- **pre-commit** for automated checks

## Contributing

Contributions are welcome! This project is in early development, and we're building the foundation for production ML explainability monitoring.

### How to Contribute

1. **Report Issues**: Found a bug or have a feature request? Open an issue on GitHub
2. **Submit Pull Requests**:
   - Fork the repository
   - Create a feature branch (`git checkout -b feature/your-feature`)
   - Make your changes with tests
   - Ensure code passes all checks (`poetry run pytest && poetry run black . && poetry run ruff check .`)
   - Submit a pull request

### Development Guidelines

- Write tests for new features
- Follow existing code style (enforced by black and ruff)
- Use type hints for all function signatures
- Add docstrings for public APIs
- Keep commits focused and write clear commit messages

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

Built on top of the excellent [SHAP](https://github.com/slundberg/shap) library by Scott Lundberg and the SHAP community.
