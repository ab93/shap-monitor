# API Reference

Complete API documentation for shap-monitor.

## Modules

- **[Monitor](monitor.md)** - SHAPMonitor class for logging explanations
- **[Analysis](analysis.md)** - SHAPAnalyzer class for analyzing logged data
- **[Backends](backends.md)** - Storage backend implementations
- **[Types](types.md)** - Type definitions and protocols

## Quick Links

### Main Classes

- [`SHAPMonitor`](monitor.md#shapmonitor.monitor.SHAPMonitor) - Monitor SHAP explanations over time
- [`SHAPAnalyzer`](analysis.md#shapmonitor.analysis.SHAPAnalyzer) - Analyze logged SHAP data
- [`ParquetBackend`](backends.md#shapmonitor.backends.ParquetBackend) - Parquet storage backend

### Key Methods

**SHAPMonitor:**

- [`log_batch()`](monitor.md#shapmonitor.monitor.SHAPMonitor.log_batch) - Log explanations for a batch
- [`compute()`](monitor.md#shapmonitor.monitor.SHAPMonitor.compute) - Compute SHAP values

**SHAPAnalyzer:**

- [`summary()`](analysis.md#shapmonitor.analysis.SHAPAnalyzer.summary) - Summary statistics
- [`compare_time_periods()`](analysis.md#shapmonitor.analysis.SHAPAnalyzer.compare_time_periods) - Compare periods

**ParquetBackend:**

- [`read()`](backends.md#shapmonitor.backends.ParquetBackend.read) - Read data from date range
- [`write()`](backends.md#shapmonitor.backends.ParquetBackend.write) - Write batch to storage
- [`delete()`](backends.md#shapmonitor.backends.ParquetBackend.delete) - Delete old data
