# User Guide

This guide covers how to use shap-monitor effectively in your ML pipelines.

## Contents

- **[Monitoring](monitoring.md)** - Using SHAPMonitor to log explanations
- **[Analysis](analysis.md)** - Analyzing logged data with SHAPAnalyzer
- **[Storage Backends](backends.md)** - Understanding storage options
- **[Examples](examples.md)** - Complete working examples

## Overview

shap-monitor provides two main components:

### SHAPMonitor

Logs SHAP explanations in production with:

- Configurable sampling rate
- Automatic SHAP value computation
- Efficient Parquet storage
- Model version tracking

### SHAPAnalyzer

Analyzes logged explanations with:

- Summary statistics over time periods
- Time period comparisons
- Feature importance rankings
- Change detection
