"""shapmonitor watch — live SHAP monitoring dashboard (Textual TUI)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

from shapmonitor.cli._common import fail, setup_logging


def watch_command(
    data_dir: Annotated[
        Path,
        typer.Option(
            "--data-dir",
            "-d",
            help="Directory containing SHAP logs.",
            envvar="SHAPMONITOR_DATA_DIR",
        ),
    ] = Path("./shap_logs"),
    refresh: Annotated[
        float,
        typer.Option("--refresh", help="Refresh interval in seconds."),
    ] = 5.0,
    period: Annotated[
        Optional[str],
        typer.Option("--period", "-p", help="Lookback period, e.g. 'last-7d'."),
    ] = "last-7d",
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
    quiet: Annotated[bool, typer.Option("--quiet", "-q")] = False,
) -> None:
    """Launch a live SHAP monitoring dashboard."""
    setup_logging(verbose=verbose, quiet=quiet)

    if not Path(data_dir).exists():
        fail(f"Data directory does not exist: {data_dir}")

    from shapmonitor.cli._watch_app import WatchApp

    app = WatchApp(
        data_dir=str(data_dir), refresh_interval=refresh, period_spec=period or "last-7d"
    )
    app.run()
