"""shapmonitor CLI — monitor SHAP explanations from the terminal."""

from __future__ import annotations

import sys

try:
    import typer
except ImportError:
    sys.stderr.write(
        "The shapmonitor CLI requires optional extras.\n"
        "Install with:  pip install shap-monitor[cli]\n"
    )
    sys.exit(1)

from importlib.metadata import version as _pkg_version

from shapmonitor.cli.log import log_command
from shapmonitor.cli.report import report_app
from shapmonitor.cli.watch import watch_command


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"shapmonitor {_pkg_version('shap-monitor')}")
        raise typer.Exit()


app = typer.Typer(
    name="shapmonitor",
    no_args_is_help=True,
    help="Monitor SHAP explanations over time.",
    rich_markup_mode="rich",
)


@app.callback(invoke_without_command=True)
def _main(
    version: bool = typer.Option(  # noqa: ARG001
        False,
        "--version",
        help="Show version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """Monitor SHAP explanations over time."""


app.command(name="log", help="Log SHAP values for a batch of predictions.")(log_command)
app.command(name="watch", help="Launch a live SHAP monitoring dashboard.")(watch_command)
app.add_typer(report_app, name="report", help="Generate reports from logged SHAP values.")
