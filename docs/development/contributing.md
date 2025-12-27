# Contributing

Contributions are welcome! This project is in early development, and we're building the foundation for production ML explainability monitoring.

## How to Contribute

### Report Issues

Found a bug or have a feature request? [Open an issue on GitHub](https://github.com/ab93/shap-monitor/issues).

### Submit Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes with tests
4. Ensure code passes all checks
5. Submit a pull request

## Development Setup

### Prerequisites

- Python 3.11 or higher
- Poetry for dependency management
- Git

### Setup Steps

```bash
# Clone your fork
git clone https://github.com/YOUR-USERNAME/shap-monitor.git
cd shap-monitor

# Install dependencies
poetry install --with dev --with docs

# Install pre-commit hooks
poetry run pre-commit install
```

## Development Workflow

### Running Tests

```bash
# Run all tests
make test

# Run with coverage
make coverage

# Run specific test file
poetry run pytest tests/test_monitor.py

# Run specific test
poetry run pytest tests/test_monitor.py::test_monitor_initialization
```

### Code Formatting & Linting

The project uses Black and Ruff for code formatting and linting.

```bash
# Format all code
make lint
```

### Pre-commit Hooks

Pre-commit hooks automatically run checks before commits:

```bash
# Manually run all hooks
poetry run pre-commit run --all-files
```

## Code Guidelines

### Style Guide

- Follow [PEP 8](https://pep8.org/) style guide
- Use [Black](https://black.readthedocs.io/) for formatting (line length: 100)
- Use type hints for all function signatures
- Write docstrings for public APIs (NumPy style)

### Documentation

- Add docstrings to all public classes and methods
- Use NumPy-style docstrings
- Include examples in docstrings where helpful
- Update user guide for new features

Example docstring:

```python
def summary(self, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    """Compute summary statistics for SHAP values in a date range.

    Parameters
    ----------
    start_dt : datetime
        Start of the date range (inclusive).
    end_dt : datetime
        End of the date range (inclusive).

    Returns
    -------
    DataFrame
        Summary statistics indexed by feature name.

    Examples
    --------
    >>> analyzer = SHAPAnalyzer(backend)
    >>> summary = analyzer.summary(start_date, end_date)
    >>> print(summary['mean_abs'].head())
    """
```

### Testing

- Write tests for new features
- Maintain or improve code coverage
- Use pytest fixtures for common setups
- Test edge cases and error conditions

Example test:

```python
def test_monitor_log_batch(tmp_path):
    """Test SHAPMonitor.log_batch() logs data correctly."""
    # Setup
    explainer = create_test_explainer()
    monitor = SHAPMonitor(
        explainer=explainer,
        data_dir=tmp_path,
        sample_rate=1.0
    )

    # Execute
    X = create_test_data()
    monitor.log_batch(X)

    # Verify
    backend = ParquetBackend(tmp_path)
    df = backend.read(datetime.now(), datetime.now())
    assert len(df) > 0
```

## Pull Request Guidelines

### Before Submitting

- [ ] Tests pass (`make test`)
- [ ] Code is formatted (`make lint`)
- [ ] Documentation is updated
- [ ] Commit messages are clear and descriptive

### PR Description

Include:

- Summary of changes
- Motivation and context
- Related issues (if any)
- Testing done
- Screenshots (if applicable)

### Review Process

1. Automated checks must pass
2. At least one maintainer review required
3. Address feedback
4. Maintain clean commit history

## Documentation

### Building Documentation

```bash
# Serve documentation locally
make docs-serve

# Build documentation
make docs-build

# Deploy to GitHub Pages (maintainers only)
poetry run mkdocs gh-deploy
```

### Writing Documentation

- Write clear, concise documentation
- Include code examples
- Update relevant sections when adding features
- Check for broken links

## Release Process

(For maintainers)

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Create release tag
4. Build and publish to PyPI
5. Update documentation

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help newcomers get started
- Follow GitHub's Community Guidelines

## Questions?

- Ask in an [Issue](https://github.com/ab93/shap-monitor/issues)
- Check existing documentation

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
