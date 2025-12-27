# Roadmap

Project roadmap and planned features for shap-monitor.

## Current Status

**Version: 0.0.1** (Work in Progress)

shap-monitor is in early development. The core functionality is being actively developed.

## Version History

### v0.0.1 (Current)

Core functionality:

- SHAPMonitor for logging SHAP explanations
- ParquetBackend for efficient storage
- SHAPAnalyzer for basic analysis
- Summary statistics over time periods
- Time period comparison
- Support for tree-based explainers

## Planned Features

### v0.2 (Planned)

**Drift Detection**

- Automatic drift detection in feature importance
- Configurable drift thresholds
- Alerting capabilities

**Asynchronous Processing**

- Async logging for production systems
- Background workers for SHAP computation
- Queue-based architecture

**MLflow Integration**

- Log SHAP values to MLflow experiments
- Compare explanations across MLflow runs
- Integration with MLflow Model Registry

**Model Version Comparison**

- Compare explanations across model versions
- A/B testing support
- Champion/challenger analysis

### v0.3+ (Future)

**Dashboard & Visualization**

- Web-based dashboard for monitoring
- Interactive visualizations
- Real-time monitoring

**Additional Integrations**

- Weights & Biases integration
- Neptune.ai integration
- Custom integration framework

**Advanced Features**

- Explanation clustering
- Anomaly detection in explanations
- Explanation stability metrics
- Custom drift metrics

**Additional Backends**

- S3/Cloud storage support
- Database backends (PostgreSQL, DuckDB)
- Time-series database integration

## Design Principles

1. **Production-First**: Built for production ML systems
2. **Minimal Overhead**: Low latency impact on predictions
3. **Flexible Storage**: Pluggable backend architecture
4. **Easy Integration**: Works with existing ML pipelines
5. **Comprehensive Analysis**: Rich analysis capabilities

## Contributing

Want to help build these features? See the [Contributing Guide](contributing.md) for how to get involved.

## Feedback

Have suggestions for the roadmap? [Open a discussion](https://github.com/ab93/shap-monitor/discussions) or [file an issue](https://github.com/ab93/shap-monitor/issues).
