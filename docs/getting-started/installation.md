# Installation

## Requirements

- Python 3.11 or higher
- Poetry (for development)

## Via PyPI
You can install shap-monitor via pip:

```bash
pip install shap-monitor
```

## From Source (Development)

To install from source:

```bash
# Clone the repository
git clone https://github.com/ab93/shap-monitor.git
cd shap-monitor

# Install with Poetry
poetry install

# Or install for development with all dev dependencies
poetry install --with dev

# Or install with documentation dependencies
poetry install --with docs
```

## Verify Installation

To verify your installation, run:

```bash
poetry run python -c "from shapmonitor import SHAPMonitor; print('Installation successful!')"
```

## Dependencies

shap-monitor has the following core dependencies:

- **shap** (>=0.50.0): Core SHAP library
- **numpy** (>=2.0.0): Numerical computing
- **pandas** (>=2.0.0): Data manipulation
- **pyarrow** (>=22.0.0): Parquet file support

## Optional Dependencies

For running the examples:

- **lightgbm** (>=4.6.0): LightGBM examples
- **xgboost** (>=3.1.2): XGBoost examples
- **matplotlib** (>=3.0.0): Visualization

Install with:

```bash
poetry install --with dev
```

## Next Steps

Once installed, proceed to the [Quick Start](quickstart.md) guide to begin using shap-monitor.
