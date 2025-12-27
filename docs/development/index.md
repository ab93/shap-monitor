# Development

Information for contributors and developers.

## Contents

- **[Contributing](contributing.md)** - How to contribute to shap-monitor
- **[Roadmap](roadmap.md)** - Project roadmap and planned features

## Quick Start for Contributors

```bash
# Clone the repository
git clone https://github.com/ab93/shap-monitor.git
cd shap-monitor

# Install dependencies
poetry install --with dev --with docs

# Install pre-commit hooks
poetry run pre-commit install

# Run tests
poetry run pytest

# Build documentation
poetry run mkdocs serve
```

## Project Structure

```
shap-monitor/
├── shapmonitor/           # Source code
│   ├── __init__.py
│   ├── monitor.py         # SHAPMonitor class
│   ├── types.py           # Type definitions
│   ├── analysis/          # Analysis module
│   │   └── _analyzer.py   # SHAPAnalyzer class
│   ├── backends/          # Storage backends
│   │   ├── _base.py       # Base backend
│   │   └── _parquet.py    # Parquet backend
│   └── integrations/      # Framework integrations
├── tests/                 # Test suite
├── examples/              # Example scripts
├── docs/                  # Documentation
└── pyproject.toml         # Project configuration
```
