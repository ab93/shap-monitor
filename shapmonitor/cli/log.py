"""shapmonitor log — log SHAP values for a batch of predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

from shapmonitor.cli._common import fail, get_err_console, render_output, setup_logging


def log_command(
    data: Annotated[
        Path,
        typer.Argument(help="Path to input data CSV file.", exists=True, readable=True),
    ],
    model: Annotated[
        Path,
        typer.Option(
            "--model", "-m", help="Path to pickled model file.", exists=True, readable=True
        ),
    ],
    data_dir: Annotated[
        Path,
        typer.Option(
            "--data-dir",
            "-d",
            help="Directory for SHAP log storage.",
            envvar="SHAPMONITOR_DATA_DIR",
        ),
    ] = Path("./shap_logs"),
    sample_rate: Annotated[
        float,
        typer.Option(
            "--sample-rate", "-s", help="Fraction of rows to sample (0.0–1.0).", min=0.0, max=1.0
        ),
    ] = 1.0,
    model_version: Annotated[
        str,
        typer.Option(help="Version identifier for the model."),
    ] = "v0",
    feature_names: Annotated[
        Optional[list[str]],
        typer.Option("--feature-name", "-f", help="Feature name (repeat for multiple)."),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", "-j", help="Output result as JSON."),
    ] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable debug logging.")] = False,
    quiet: Annotated[
        bool, typer.Option("--quiet", "-q", help="Suppress non-error output.")
    ] = False,
) -> None:
    """Log SHAP values for a batch of predictions."""
    setup_logging(verbose=verbose, quiet=quiet)

    # All heavy imports are lazy — keeps `shapmonitor --help` fast.
    import joblib  # noqa: S403 — user-provided model path, standard ML workflow
    import shap
    import numpy as np
    import pandas as pd
    from shapmonitor import SHAPMonitor

    console = get_err_console()

    # Load data
    try:
        df = pd.read_csv(data)
    except Exception as exc:
        fail(f"Cannot read data file: {exc}")

    X = df.to_numpy(dtype=np.float64)

    # Load model and create SHAP explainer
    try:
        loaded_model = joblib.load(model)  # noqa: S301
    except Exception as exc:
        fail(f"Cannot load model: {exc}")

    try:
        explainer = shap.TreeExplainer(loaded_model)
    except Exception:
        try:
            explainer = shap.Explainer(loaded_model)
        except Exception as exc:
            fail(f"Cannot create SHAP explainer: {exc}")

    monitor = SHAPMonitor(
        explainer=explainer,
        data_dir=str(data_dir),
        sample_rate=sample_rate,
        model_version=model_version,
        feature_names=feature_names or df.columns.tolist(),
    )

    # Log batch with progress indication
    if not quiet and not json_output:
        from rich.progress import Progress, SpinnerColumn, TextColumn

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("Computing SHAP values...", total=None)
            monitor.log_batch(X)
    else:
        monitor.log_batch(X)

    result = {
        "status": "ok",
        "n_samples": len(X),
        "sample_rate": sample_rate,
        "n_logged": max(1, int(len(X) * sample_rate)),
        "data_dir": str(data_dir),
        "model_version": model_version,
    }

    def _render_success(data, cons):
        from rich.panel import Panel

        cons.print(
            Panel(
                f"[green]Logged SHAP values[/green]\n"
                f"  Samples: {data['n_logged']} / {data['n_samples']}\n"
                f"  Model:   {data['model_version']}\n"
                f"  Dir:     {data['data_dir']}",
                title="shapmonitor log",
                border_style="green",
            )
        )

    render_output(result, json_output, renderer=_render_success, console=console)
