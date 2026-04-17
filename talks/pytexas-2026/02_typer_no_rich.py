#!/usr/bin/env python3
"""Log SHAP values for a batch — step 2 of the PyTexas talk arc.

Diff from ``01_before_argparse.py``: argparse is gone. That's it.

What vanished:
    - ``argparse.ArgumentParser(...)`` + ``parser.add_argument(...)`` boilerplate
    - Manual ``os.path.exists(...)`` checks   → ``exists=True`` on the Path param
    - Manual ``float(args.sample_rate)``      → ``float`` type hint auto-coerces
    - Blanket ``try/except Exception``        → Typer owns exit codes + tracebacks
    - The ``sys`` import entirely

What stayed the same (so step 3's Rich diff is uncontaminated):
    - Hand-padded f-string "table" output, no color
    - Single command (no subcommand split yet — that's step 3)

Run it the same way as step 1::

    python talks/pytexas-2026/02_typer_no_rich.py demo/batch_current.csv \\
        --model demo/model.pkl --data-dir /tmp/typer_demo_logs

Note: model loading uses joblib (pickle under the hood) to match the real CLI
path in ``shapmonitor/cli/log.py`` — only load models you trust.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import joblib
import numpy as np
import pandas as pd
import shap
import typer

from shapmonitor import SHAPMonitor


def log(
    data: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, help="Path to CSV file with feature data"),
    ],
    model: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, help="Path to pickled model"),
    ],
    data_dir: Annotated[
        Path,
        typer.Option(help="Output directory for SHAP logs"),
    ] = Path("./shap_logs"),
    sample_rate: Annotated[
        float,
        typer.Option(help="Sampling rate (0.0-1.0)"),
    ] = 1.0,
    model_version: Annotated[
        str,
        typer.Option(help="Model version tag"),
    ] = "v1",
) -> None:
    """Log SHAP values for a batch of predictions."""
    df = pd.read_csv(data)
    mdl = joblib.load(model)  # noqa: S301 — mirrors real CLI path

    explainer = shap.TreeExplainer(mdl)
    exp = explainer(df.values)

    # Binary-classifier SHAP comes back 3-D; take the positive class.
    values = exp.values[:, :, 1] if exp.values.ndim == 3 else exp.values
    base_values = (
        exp.base_values[:, 1] if np.asarray(exp.base_values).ndim == 2 else exp.base_values
    )

    monitor = SHAPMonitor(
        data_dir=str(data_dir),
        sample_rate=sample_rate,
        model_version=model_version,
        feature_names=list(df.columns),
    )
    monitor.log_shap(values, base_values=base_values)

    # Same hand-rolled table as step 1 — Rich replaces this in step 3.
    mean_abs = np.abs(values).mean(axis=0)
    print()
    print(f"{'Feature':<20} {'Mean |SHAP|':>12}")
    print("-" * 33)
    for col, val in zip(df.columns, mean_abs):
        print(f"{col:<20} {val:>12.4f}")
    print()
    print(f"Logged {len(df)} samples to {data_dir}")


if __name__ == "__main__":
    typer.run(log)
