"""shapmonitor report — generate summary and drift reports."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Annotated, Optional

import typer

from shapmonitor.cli._common import (
    fail,
    get_backend,
    get_out_console,
    parse_period,
    period_from_dates,
    psi_label,
    psi_style,
    render_output,
    setup_logging,
)

report_app = typer.Typer(no_args_is_help=True)


# ---------------------------------------------------------------------------
# shapmonitor report summary
# ---------------------------------------------------------------------------


@report_app.command("summary")
def summary_command(
    data_dir: Annotated[
        Path,
        typer.Option(help="Directory containing SHAP logs.", envvar="SHAPMONITOR_DATA_DIR"),
    ] = Path("./shap_logs"),
    period: Annotated[
        Optional[str],
        typer.Option(help="Period spec, e.g. 'last-7d' or '2026-04-01..2026-04-08'."),
    ] = None,
    start: Annotated[
        Optional[datetime],
        typer.Option(
            help="Start date (ISO format). Alternative to --period.", formats=["%Y-%m-%d"]
        ),
    ] = None,
    end: Annotated[
        Optional[datetime],
        typer.Option(help="End date (ISO format). Alternative to --period.", formats=["%Y-%m-%d"]),
    ] = None,
    top_k: Annotated[
        Optional[int],
        typer.Option("--top-k", help="Show only top K features."),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output result as JSON."),
    ] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
    quiet: Annotated[bool, typer.Option("--quiet", "-q")] = False,
) -> None:
    """Show summary statistics for SHAP values in a time period."""
    setup_logging(verbose=verbose, quiet=quiet)

    if period:
        p = parse_period(period)
    else:
        p = period_from_dates(start, end)

    from shapmonitor.analysis import SHAPAnalyzer

    backend = get_backend(str(data_dir))
    analyzer = SHAPAnalyzer(backend)

    try:
        summary_df = analyzer.summary(start_dt=p.start, end_dt=p.end, top_k=top_k)
    except Exception as exc:
        fail(f"Cannot generate summary: {exc}")

    if summary_df.empty:
        fail("No data found for the specified period.")

    # JSON path
    if json_output:
        render_output(
            {
                "period": {"start": str(p.start), "end": str(p.end)},
                "n_samples": summary_df.attrs.get("n_samples"),
                "features": summary_df.reset_index().to_dict(orient="records"),
            },
            json_mode=True,
        )
        return

    # Rich table
    from rich.table import Table

    console = get_out_console()

    table = Table(title=f"SHAP Summary  ({p.start} → {p.end})", show_lines=False)
    table.add_column("Feature", style="bold")
    table.add_column("Mean |SHAP|", justify="right")
    table.add_column("", justify="left")  # inline bar
    table.add_column("Mean", justify="right")
    table.add_column("Std", justify="right")

    max_mean_abs = summary_df["mean_abs"].max()
    for feat, row in summary_df.iterrows():
        bar_len = int(20 * row["mean_abs"] / max_mean_abs) if max_mean_abs > 0 else 0
        bar = "[cyan]" + "\u2588" * bar_len + "[/cyan]"
        table.add_row(
            str(feat),
            f"{row['mean_abs']:.4f}",
            bar,
            f"{row['mean']:.4f}",
            f"{row['std']:.4f}",
        )

    n_samples = summary_df.attrs.get("n_samples", "?")
    console.print(f"\n[dim]Samples: {n_samples}[/dim]")
    console.print(table)


# ---------------------------------------------------------------------------
# shapmonitor report drift
# ---------------------------------------------------------------------------


@report_app.command("drift")
def drift_command(
    data_dir: Annotated[
        Path,
        typer.Option(help="Directory containing SHAP logs.", envvar="SHAPMONITOR_DATA_DIR"),
    ] = Path("./shap_logs"),
    ref: Annotated[
        Optional[str],
        typer.Option("--ref", help="Reference period, e.g. 'last-14d..last-7d'."),
    ] = None,
    curr: Annotated[
        Optional[str],
        typer.Option("--curr", help="Current period, e.g. 'last-7d..now'."),
    ] = None,
    ref_start: Annotated[
        Optional[datetime],
        typer.Option(help="Reference start date.", formats=["%Y-%m-%d"]),
    ] = None,
    ref_end: Annotated[
        Optional[datetime],
        typer.Option(help="Reference end date.", formats=["%Y-%m-%d"]),
    ] = None,
    curr_start: Annotated[
        Optional[datetime],
        typer.Option(help="Current start date.", formats=["%Y-%m-%d"]),
    ] = None,
    curr_end: Annotated[
        Optional[datetime],
        typer.Option(help="Current end date.", formats=["%Y-%m-%d"]),
    ] = None,
    top_k: Annotated[
        Optional[int],
        typer.Option("--top-k", help="Show only top K features."),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output result as JSON."),
    ] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
    quiet: Annotated[bool, typer.Option("--quiet", "-q")] = False,
) -> None:
    """Compare SHAP value drift between two time periods."""
    setup_logging(verbose=verbose, quiet=quiet)

    # Resolve reference period
    if ref:
        period_ref = parse_period(ref)
    elif ref_start or ref_end:
        period_ref = period_from_dates(ref_start, ref_end)
    else:
        # Default: 14d ago → 7d ago
        today = date.today()
        from datetime import timedelta

        period_ref = period_from_dates(
            datetime.combine(today - timedelta(days=14), datetime.min.time()),
            datetime.combine(today - timedelta(days=7), datetime.min.time()),
        )

    # Resolve current period
    if curr:
        period_curr = parse_period(curr)
    elif curr_start or curr_end:
        period_curr = period_from_dates(curr_start, curr_end)
    else:
        # Default: last 7d → now
        today = date.today()
        from datetime import timedelta

        period_curr = period_from_dates(
            datetime.combine(today - timedelta(days=7), datetime.min.time()),
            datetime.combine(today, datetime.min.time()),
        )

    from shapmonitor.analysis import SHAPAnalyzer

    backend = get_backend(str(data_dir))
    analyzer = SHAPAnalyzer(backend)

    try:
        drift_df = analyzer.compare_time_periods(period_ref, period_curr, top_k=top_k)
    except Exception as exc:
        fail(f"Cannot generate drift report: {exc}")

    if drift_df.empty:
        fail("No data found for the specified periods.")

    # JSON path
    if json_output:
        render_output(
            {
                "ref_period": {"start": str(period_ref.start), "end": str(period_ref.end)},
                "curr_period": {"start": str(period_curr.start), "end": str(period_curr.end)},
                "n_samples_ref": drift_df.attrs.get("n_samples_1"),
                "n_samples_curr": drift_df.attrs.get("n_samples_2"),
                "features": drift_df.reset_index().to_dict(orient="records"),
            },
            json_mode=True,
        )
        return

    # Rich output
    from rich.panel import Panel
    from rich.table import Table

    console = get_out_console()

    # Headline panel — count by severity
    import numpy as np

    alerts = int((drift_df["psi"] >= 0.25).sum())
    warnings = int(((drift_df["psi"] >= 0.1) & (drift_df["psi"] < 0.25)).sum())
    stable = int((drift_df["psi"] < 0.1).sum())
    nan_count = int(drift_df["psi"].isna().sum())

    headline_parts = [
        f"[red]{alerts} alert{'s' if alerts != 1 else ''}[/red]",
        f"[yellow]{warnings} warning{'s' if warnings != 1 else ''}[/yellow]",
        f"[green]{stable} stable[/green]",
    ]
    if nan_count:
        headline_parts.append(f"[dim]{nan_count} N/A[/dim]")

    console.print(
        Panel(
            " \u00b7 ".join(headline_parts),
            title=f"Drift Report  (ref: {period_ref.start}\u2192{period_ref.end}  vs  curr: {period_curr.start}\u2192{period_curr.end})",
            border_style="blue",
        )
    )

    # Drift table
    table = Table(show_lines=False)
    table.add_column("Feature", style="bold")
    table.add_column("PSI", justify="right")
    table.add_column("Status", justify="center")
    table.add_column("|SHAP| ref", justify="right")
    table.add_column("|SHAP| curr", justify="right")
    table.add_column("\u0394 |SHAP|", justify="right")
    table.add_column("Rank \u0394", justify="right")

    for feat, row in drift_df.iterrows():
        psi_val = row["psi"]
        if np.isnan(psi_val):
            style = "dim"
            label = "n/a"
            psi_str = "—"
        else:
            style = psi_style(psi_val)
            label = psi_label(psi_val)
            psi_str = f"{psi_val:.4f}"

        delta_rank = row.get("delta_rank", 0)
        if np.isnan(delta_rank):
            delta_rank_str = "[dim]—[/dim]"
        elif int(delta_rank) == 0:
            # "No change" renders dim so real rank moves pop against it.
            delta_rank_str = "[dim]—[/dim]"
        else:
            delta_rank_str = f"{int(delta_rank):+d}"

        table.add_row(
            str(feat),
            f"[{style}]{psi_str}[/{style}]",
            f"[{style}]{label}[/{style}]",
            f"{row['mean_abs_1']:.4f}",
            f"{row['mean_abs_2']:.4f}",
            f"{row['delta_mean_abs']:.4f}",
            delta_rank_str,
        )

    console.print(table)
