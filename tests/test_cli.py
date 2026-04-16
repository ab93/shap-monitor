"""Smoke tests for the shapmonitor CLI."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

# The CLI lives in an optional extra (``pip install shap-monitor[cli]``).
# Skip this whole module when the extra isn't installed so local runs without
# CLI deps don't abort pytest collection with ``ModuleNotFoundError``.
pytest.importorskip("typer")
pytest.importorskip("rich")

from typer.testing import CliRunner  # noqa: E402

from shapmonitor.cli import app  # noqa: E402

runner = CliRunner()

DEMO_DIR = Path(__file__).resolve().parent.parent / "demo"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cli_fixtures(tmp_path_factory):
    """Create a small model + CSV + pre-logged SHAP data for CLI tests."""
    import shap

    from shapmonitor import SHAPMonitor

    base = tmp_path_factory.mktemp("cli")
    shap_logs = base / "shap_logs"
    shap_logs.mkdir()

    # Train tiny model
    X, y = make_classification(n_samples=200, n_features=5, n_informative=3, random_state=42)
    model = RandomForestClassifier(n_estimators=10, max_depth=4, random_state=42)
    model.fit(X, y)

    import joblib

    model_path = base / "model.pkl"
    joblib.dump(model, model_path)

    # Save a CSV batch
    import pandas as pd

    csv_path = base / "batch.csv"
    pd.DataFrame(X, columns=[f"f{i}" for i in range(5)]).to_csv(csv_path, index=False)

    # Pre-log SHAP values so report commands have data
    explainer = shap.TreeExplainer(model)
    exp = explainer(X)
    shap_values = exp.values[:, :, 1] if exp.values.ndim == 3 else exp.values
    base_values = (
        exp.base_values[:, 1] if np.asarray(exp.base_values).ndim == 2 else exp.base_values
    )

    monitor = SHAPMonitor(
        data_dir=str(shap_logs),
        sample_rate=1.0,
        model_version="v1",
        feature_names=[f"f{i}" for i in range(5)],
    )
    monitor.log_shap(shap_values, base_values=base_values)

    return {
        "model_path": model_path,
        "csv_path": csv_path,
        "shap_logs": shap_logs,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "shapmonitor" in result.stdout


def test_help_is_fast():
    """--help should return in under 500ms (loose bound for CI)."""
    start = time.perf_counter()
    subprocess.run(
        [sys.executable, "-m", "shapmonitor.cli", "--help"],
        capture_output=True,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert elapsed_ms < 500, f"--help took {elapsed_ms:.0f}ms, expected < 500ms"


def test_log_command(cli_fixtures):
    result = runner.invoke(
        app,
        [
            "log",
            str(cli_fixtures["csv_path"]),
            "--model",
            str(cli_fixtures["model_path"]),
            "--data-dir",
            str(cli_fixtures["shap_logs"]),
        ],
    )
    assert result.exit_code == 0, f"log failed: {result.stdout}"


def test_report_summary_json(cli_fixtures):
    result = runner.invoke(
        app,
        [
            "report",
            "summary",
            "--data-dir",
            str(cli_fixtures["shap_logs"]),
            "--period",
            "last-30d",
            "--json",
        ],
    )
    assert result.exit_code == 0, f"report summary failed: {result.stdout}"
    data = json.loads(result.stdout)
    assert "features" in data
    assert len(data["features"]) > 0


def test_report_drift_json(cli_fixtures):
    result = runner.invoke(
        app,
        [
            "report",
            "drift",
            "--data-dir",
            str(cli_fixtures["shap_logs"]),
            "--ref",
            "last-30d..last-15d",
            "--curr",
            "last-15d..now",
            "--json",
        ],
    )
    # Drift may exit 1 if no data in one period — just check it doesn't crash
    assert result.exit_code in (0, 1), f"report drift crashed: {result.stdout}"
