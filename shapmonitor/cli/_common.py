"""Shared CLI utilities — period parsing, output helpers, logging setup."""

from __future__ import annotations

import json
import logging
import re
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Any

import typer

if TYPE_CHECKING:
    from rich.console import Console
    from shapmonitor.types import Period

# ---------------------------------------------------------------------------
# Period parsing
# ---------------------------------------------------------------------------

_RELATIVE_RE = re.compile(r"^last-(\d+)([dhw])$")
_RANGE_RE = re.compile(r"^(.+?)\.\.(.+)$")


def _resolve_relative(spec: str, anchor: date | None = None) -> date:
    """Resolve a relative spec like ``last-7d`` to a concrete date."""
    anchor = anchor or date.today()
    m = _RELATIVE_RE.match(spec)
    if not m:
        raise typer.BadParameter(f"Invalid relative period: {spec!r}")
    amount, unit = int(m.group(1)), m.group(2)
    if unit == "d":
        return anchor - timedelta(days=amount)
    if unit == "h":
        return anchor - timedelta(hours=amount)
    if unit == "w":
        return anchor - timedelta(weeks=amount)
    raise typer.BadParameter(f"Unknown unit in period spec: {spec!r}")


def _parse_date(spec: str) -> datetime:
    """Parse an ISO date string *or* a relative spec into a datetime."""
    if spec == "now" or spec == "today":
        return datetime.combine(date.today(), datetime.min.time())
    if spec.startswith("last-"):
        resolved = _resolve_relative(spec)
        return datetime.combine(resolved, datetime.min.time())
    try:
        return datetime.combine(date.fromisoformat(spec), datetime.min.time())
    except ValueError:
        raise typer.BadParameter(f"Cannot parse date: {spec!r}") from None


def parse_period(spec: str) -> Period:
    """Parse a period string into a ``Period`` NamedTuple.

    Accepted formats::

        last-7d            → (today - 7 days, today)
        last-2w            → (today - 2 weeks, today)
        last-14d..last-7d  → (today - 14d, today - 7d)
        2026-04-01..2026-04-08 → literal range
    """
    from shapmonitor.types import Period

    m = _RANGE_RE.match(spec)
    if m:
        return Period(start=_parse_date(m.group(1)), end=_parse_date(m.group(2)))
    # Single relative spec → (resolved, today)
    start = _parse_date(spec)
    return Period(start=start, end=datetime.combine(date.today(), datetime.min.time()))


def period_from_dates(
    start: datetime | date | None,
    end: datetime | date | None,
) -> Period:
    """Build a Period from explicit start/end dates (CLI --start/--end path)."""
    from shapmonitor.types import Period

    _today = datetime.combine(date.today(), datetime.min.time())
    return Period(
        start=start or _today - timedelta(days=7),
        end=end or _today,
    )


# ---------------------------------------------------------------------------
# Backend helper
# ---------------------------------------------------------------------------


def get_backend(data_dir: str):
    """Construct a ParquetBackend from a directory path (lazy import)."""
    from shapmonitor.backends import ParquetBackend

    return ParquetBackend(file_dir=data_dir)


# ---------------------------------------------------------------------------
# Console & output
# ---------------------------------------------------------------------------

_err_console: Console | None = None
_out_console: Console | None = None


def get_err_console() -> Console:
    """Return a Rich console that writes to stderr."""
    global _err_console
    if _err_console is None:
        from rich.console import Console

        _err_console = Console(stderr=True)
    return _err_console


def get_out_console() -> Console:
    """Return a Rich console that writes to stdout."""
    global _out_console
    if _out_console is None:
        from rich.console import Console

        _out_console = Console()
    return _out_console


def render_output(
    data: Any,
    json_mode: bool,
    renderer: Any | None = None,
    console: Console | None = None,
) -> None:
    """Print *data* as JSON or pass it through *renderer* for Rich output."""
    if json_mode:
        typer.echo(json.dumps(data, default=str, indent=2))
        return
    if renderer and console:
        renderer(data, console)


# ---------------------------------------------------------------------------
# PSI severity
# ---------------------------------------------------------------------------

PSI_THRESHOLDS = {"stable": 0.1, "warning": 0.25}


def psi_style(value: float) -> str:
    """Return a Rich style string for a PSI value."""
    if value < PSI_THRESHOLDS["stable"]:
        return "green"
    if value < PSI_THRESHOLDS["warning"]:
        return "yellow"
    return "red"


def psi_label(value: float) -> str:
    """Return a human-readable label for a PSI value."""
    if value < PSI_THRESHOLDS["stable"]:
        return "stable"
    if value < PSI_THRESHOLDS["warning"]:
        return "warning"
    return "alert"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def fail(message: str, code: int = 1) -> None:
    """Print an error to stderr and exit."""
    console = get_err_console()
    console.print(f"[bold red]Error:[/bold red] {message}")
    raise typer.Exit(code=code)


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def setup_logging(*, verbose: bool = False, quiet: bool = False) -> None:
    """Configure Python logging with RichHandler."""
    if quiet:
        logging.disable(logging.CRITICAL)
        return

    level = logging.DEBUG if verbose else logging.WARNING

    from rich.logging import RichHandler

    handler = RichHandler(
        console=get_err_console(),
        show_path=verbose,
        rich_tracebacks=True,
    )
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[handler],
        force=True,
    )
