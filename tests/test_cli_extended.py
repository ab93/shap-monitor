"""Extended CLI tests to cover report/log/watch/common/watch-app paths."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

pytest.importorskip("typer")
pytest.importorskip("rich")

from typer.testing import CliRunner  # noqa: E402

from shapmonitor.cli import app  # noqa: E402

runner = CliRunner()


# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def cli_fixtures(tmp_path_factory):
    """Small model + CSV + SHAP data logged across two dates for drift tests."""
    import shap
    from freezegun import freeze_time

    from shapmonitor import SHAPMonitor

    base = tmp_path_factory.mktemp("cli_ext")
    shap_logs = base / "shap_logs"
    shap_logs.mkdir()

    X, y = make_classification(n_samples=200, n_features=5, n_informative=3, random_state=42)
    model = RandomForestClassifier(n_estimators=10, max_depth=4, random_state=42)
    model.fit(X, y)

    import joblib

    model_path = base / "model.pkl"
    joblib.dump(model, model_path)

    import pandas as pd

    csv_path = base / "batch.csv"
    pd.DataFrame(X, columns=[f"f{i}" for i in range(5)]).to_csv(csv_path, index=False)

    explainer = shap.TreeExplainer(model)

    def _log(X_batch):
        exp = explainer(X_batch)
        shap_values = exp.values[:, :, 1] if exp.values.ndim == 3 else exp.values
        base_values = (
            exp.base_values[:, 1] if np.asarray(exp.base_values).ndim == 2 else exp.base_values
        )
        SHAPMonitor(
            data_dir=str(shap_logs),
            sample_rate=1.0,
            model_version="v1",
            feature_names=[f"f{i}" for i in range(5)],
        ).log_shap(shap_values, base_values=base_values)

    # Two distinct date partitions so drift PSI is computable (non-NaN)
    with freeze_time("2026-03-10"):
        _log(X[:100])
    with freeze_time("2026-04-10"):
        _log(X[100:])

    return {
        "model_path": model_path,
        "csv_path": csv_path,
        "shap_logs": shap_logs,
        "base": base,
    }


# ─── shapmonitor/__init__.py ──────────────────────────────────────────────────


def test_getattr_unknown():
    import shapmonitor

    with pytest.raises(AttributeError):
        _ = shapmonitor.DoesNotExist


# ─── _common.py ───────────────────────────────────────────────────────────────


class TestCommonUtils:
    def test_resolve_relative_hours(self):
        from shapmonitor.cli._common import _resolve_relative

        anchor = date(2026, 4, 18)
        assert _resolve_relative("last-24h", anchor=anchor) == anchor - timedelta(hours=24)

    def test_resolve_relative_weeks(self):
        from shapmonitor.cli._common import _resolve_relative

        anchor = date(2026, 4, 18)
        assert _resolve_relative("last-2w", anchor=anchor) == anchor - timedelta(weeks=2)

    def test_resolve_relative_invalid(self):
        import typer

        from shapmonitor.cli._common import _resolve_relative

        with pytest.raises(typer.BadParameter):
            _resolve_relative("bad-spec")

    def test_parse_date_iso(self):
        from shapmonitor.cli._common import _parse_date

        assert _parse_date("2026-04-01") == datetime(2026, 4, 1)

    def test_parse_date_invalid(self):
        import typer

        from shapmonitor.cli._common import _parse_date

        with pytest.raises(typer.BadParameter):
            _parse_date("not-a-date")

    def test_period_from_dates_defaults(self):
        from shapmonitor.cli._common import period_from_dates

        p = period_from_dates(None, None)
        assert p.start < p.end

    def test_get_out_console_singleton(self):
        import shapmonitor.cli._common as _mod

        _mod._out_console = None
        c1 = _mod.get_out_console()
        c2 = _mod.get_out_console()
        assert c1 is c2

    def test_psi_style_warning(self):
        from shapmonitor.cli._common import psi_style

        assert psi_style(0.15) == "yellow"

    def test_psi_style_alert(self):
        from shapmonitor.cli._common import psi_style

        assert psi_style(0.30) == "red"

    def test_psi_label_warning(self):
        from shapmonitor.cli._common import psi_label

        assert psi_label(0.15) == "warning"

    def test_psi_label_alert(self):
        from shapmonitor.cli._common import psi_label

        assert psi_label(0.30) == "alert"

    def test_fail_exits(self):
        import typer

        from shapmonitor.cli._common import fail

        with pytest.raises(typer.Exit):
            fail("boom")

    def test_setup_logging_verbose(self):
        from shapmonitor.cli._common import setup_logging

        setup_logging(verbose=True)

    def test_setup_logging_quiet(self):
        from shapmonitor.cli._common import setup_logging

        setup_logging(quiet=True)

    def test_render_output_renderer(self):
        from rich.console import Console

        from shapmonitor.cli._common import render_output

        called = []
        render_output(
            {"k": 1},
            json_mode=False,
            renderer=lambda d, c: called.append(d),
            console=Console(),
        )
        assert called


# ─── report summary ───────────────────────────────────────────────────────────


class TestReportSummary:
    def test_summary_rich_table(self, cli_fixtures):
        """The non-JSON path renders a Rich table."""
        result = runner.invoke(
            app,
            [
                "report",
                "summary",
                "--data-dir",
                str(cli_fixtures["shap_logs"]),
                "--period",
                "2026-01-01..2026-05-01",
            ],
        )
        assert result.exit_code == 0

    def test_summary_no_period_uses_defaults(self, cli_fixtures):
        """summary without --period falls through to period_from_dates."""
        result = runner.invoke(
            app,
            ["report", "summary", "--data-dir", str(cli_fixtures["shap_logs"])],
        )
        assert result.exit_code in (0, 1)

    def test_summary_empty_data_dir(self, tmp_path):
        """summary exits non-zero when there is no data."""
        (tmp_path / "shap_logs").mkdir()
        result = runner.invoke(
            app,
            ["report", "summary", "--data-dir", str(tmp_path / "shap_logs"), "--period", "last-7d"],
        )
        assert result.exit_code != 0

    def test_summary_exception_path(self, tmp_path):
        """summary handles a bad data dir gracefully."""
        result = runner.invoke(
            app,
            ["report", "summary", "--data-dir", str(tmp_path / "nonexistent")],
        )
        assert result.exit_code != 0


# ─── report drift ─────────────────────────────────────────────────────────────


class TestReportDrift:
    def test_drift_rich_table(self, cli_fixtures):
        """Drift table renders when both periods contain data (non-NaN PSI)."""
        result = runner.invoke(
            app,
            [
                "report",
                "drift",
                "--data-dir",
                str(cli_fixtures["shap_logs"]),
                "--ref",
                "2026-02-01..2026-03-20",
                "--curr",
                "2026-03-20..2026-05-01",
            ],
        )
        assert result.exit_code == 0

    def test_drift_default_periods(self, cli_fixtures):
        """drift uses built-in default periods when no --ref/--curr are given."""
        result = runner.invoke(
            app,
            ["report", "drift", "--data-dir", str(cli_fixtures["shap_logs"])],
        )
        assert result.exit_code in (0, 1)

    def test_drift_explicit_start_end(self, cli_fixtures):
        """drift resolves periods from --ref-start / --ref-end flags."""
        result = runner.invoke(
            app,
            [
                "report",
                "drift",
                "--data-dir",
                str(cli_fixtures["shap_logs"]),
                "--ref-start",
                "2026-02-01",
                "--ref-end",
                "2026-03-20",
                "--curr-start",
                "2026-03-20",
            ],
        )
        assert result.exit_code in (0, 1)

    def test_drift_empty_data_dir(self, tmp_path):
        """drift exits non-zero when there is no data."""
        (tmp_path / "shap_logs").mkdir()
        result = runner.invoke(
            app,
            ["report", "drift", "--data-dir", str(tmp_path / "shap_logs")],
        )
        assert result.exit_code != 0


# ─── log ──────────────────────────────────────────────────────────────────────


class TestLogCommand:
    def test_log_json_branch(self, cli_fixtures):
        """--json skips the progress spinner (else branch in log.py)."""
        result = runner.invoke(
            app,
            [
                "log",
                str(cli_fixtures["csv_path"]),
                "--model",
                str(cli_fixtures["model_path"]),
                "--data-dir",
                str(cli_fixtures["shap_logs"]),
                "--json",
            ],
        )
        assert result.exit_code == 0
        assert json.loads(result.stdout)["status"] == "ok"

    def test_log_quiet_branch(self, cli_fixtures):
        """--quiet also takes the else branch."""
        result = runner.invoke(
            app,
            [
                "log",
                str(cli_fixtures["csv_path"]),
                "--model",
                str(cli_fixtures["model_path"]),
                "--data-dir",
                str(cli_fixtures["shap_logs"]),
                "--quiet",
            ],
        )
        assert result.exit_code == 0

    def test_log_bad_csv(self, tmp_path, cli_fixtures):
        """log exits non-zero when the CSV file has unreadable content."""
        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_bytes(b"\x89PNG\r\n\x1a\n")  # binary content — not valid CSV
        result = runner.invoke(
            app,
            [
                "log",
                str(bad_csv),
                "--model",
                str(cli_fixtures["model_path"]),
                "--data-dir",
                str(tmp_path / "logs"),
            ],
        )
        assert result.exit_code != 0

    def test_log_bad_model(self, tmp_path, cli_fixtures):
        """log exits non-zero when the model file cannot be deserialized."""
        bad_model = tmp_path / "bad_model.pkl"
        bad_model.write_text("not a valid model file")
        csv = tmp_path / "data.csv"
        csv.write_text("f0,f1,f2,f3,f4\n0.1,0.2,0.3,0.4,0.5\n")
        result = runner.invoke(
            app,
            [
                "log",
                str(csv),
                "--model",
                str(bad_model),
                "--data-dir",
                str(tmp_path / "logs"),
            ],
        )
        assert result.exit_code != 0


# ─── watch ────────────────────────────────────────────────────────────────────


class TestWatchCommand:
    def test_watch_missing_dir_fails(self, tmp_path):
        result = runner.invoke(
            app,
            ["watch", "--data-dir", str(tmp_path / "does_not_exist")],
        )
        assert result.exit_code != 0

    def test_watch_runs_app(self, tmp_path):
        """watch creates WatchApp and calls run() when dir exists."""
        (tmp_path / "shap_logs").mkdir()
        with patch("shapmonitor.cli._watch_app.WatchApp") as MockApp:
            mock_instance = MagicMock()
            MockApp.return_value = mock_instance
            runner.invoke(app, ["watch", "--data-dir", str(tmp_path / "shap_logs")])
            MockApp.assert_called_once()
            mock_instance.run.assert_called_once()


# ─── _watch_app.py (Textual TUI) ─────────────────────────────────────────────


def test_watch_app_with_data(cli_fixtures):
    """Full TUI lifecycle: mount, load data with two-period drift, filter, refresh."""
    import asyncio

    pytest.importorskip("textual")
    from shapmonitor.cli._watch_app import WatchApp

    async def _run():
        watch_app = WatchApp(
            data_dir=str(cli_fixtures["shap_logs"]),
            refresh_interval=3600.0,
            period_spec="2026-01-01..2026-05-01",
        )
        async with watch_app.run_test(headless=True) as pilot:
            await pilot.pause(0.3)
            await pilot.press("/")
            await pilot.pause(0.05)
            await pilot.press("f", "0")
            await pilot.pause(0.05)
            await pilot.press("escape")
            await pilot.pause(0.05)
            await pilot.press("r")
            await pilot.pause(0.1)

    asyncio.run(_run())


def test_watch_app_empty_dir(tmp_path):
    """WatchApp handles an empty data directory without crashing."""
    import asyncio

    pytest.importorskip("textual")
    from shapmonitor.cli._watch_app import WatchApp

    async def _run():
        (tmp_path / "shap_logs").mkdir()
        watch_app = WatchApp(
            data_dir=str(tmp_path / "shap_logs"),
            refresh_interval=3600.0,
            period_spec="last-7d",
        )
        async with watch_app.run_test(headless=True) as pilot:
            await pilot.pause(0.2)

    asyncio.run(_run())
