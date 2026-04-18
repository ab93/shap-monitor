"""
Log SHAP values for a batch — the "before" script used in the PyTexas talk.

This is intentionally painful. Every wart below maps to a slide in the talk and
to a feature the final ``shapmonitor log`` command (in ``shapmonitor/cli/log.py``)
uses to eliminate it.

    - argparse boilerplate              → Typer decorators + type hints
    - manual `os.path.exists` checks    → Typer's `exists=True` file param
    - manual `float(...)` coercion      → Typer infers from the type hint
    - hand-padded f-string "table"      → Rich Table + inline bars
    - blanket `except Exception`        → proper exit codes + stderr discipline
    - no `--json`, no colors, no spinner → all one-liners with Rich/Typer

Run it with::

    python talks/pytexas-2026/01_before_argparse.py demo/batch_current.csv \\
        --model demo/model.pkl --data-dir /tmp/argparse_demo_logs

Note: model loading uses joblib (pickle under the hood) to match the real CLI
path in ``shapmonitor/cli/log.py`` — only load models you trust.
"""

from __future__ import annotations

import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd
import shap

from shapmonitor import SHAPMonitor


def main() -> None:
    parser = argparse.ArgumentParser(description="Log SHAP values for a batch of predictions.")
    parser.add_argument("data", help="Path to CSV file with feature data")
    parser.add_argument("--model", required=True, help="Path to pickled model")
    parser.add_argument("--data-dir", default="./shap_logs", help="Output directory")
    parser.add_argument("--sample-rate", default="1.0", help="Sampling rate (0.0-1.0)")
    parser.add_argument("--model-version", default="v1", help="Model version tag")
    args = parser.parse_args()

    # Manual file-exists checks — Typer does this with `exists=True`.
    if not os.path.exists(args.data):
        print(f"Error: data file not found: {args.data}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.model):
        print(f"Error: model file not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    # Manual type coercion — argparse hands us strings.
    try:
        sample_rate = float(args.sample_rate)
    except ValueError:
        print(
            f"Error: --sample-rate must be a number, got {args.sample_rate!r}",
            file=sys.stderr,
        )
        sys.exit(1)

    # One giant try/except that swallows every failure mode into exit 1.
    try:
        df = pd.read_csv(args.data)
        model = joblib.load(args.model)  # noqa: S301 — mirrors real CLI path

        explainer = shap.TreeExplainer(model)
        exp = explainer(df.values)

        # Binary-classifier SHAP comes back 3-D; take the positive class.
        values = exp.values[:, :, 1] if exp.values.ndim == 3 else exp.values
        base_values = (
            exp.base_values[:, 1] if np.asarray(exp.base_values).ndim == 2 else exp.base_values
        )

        monitor = SHAPMonitor(
            data_dir=args.data_dir,
            sample_rate=sample_rate,
            model_version=args.model_version,
            feature_names=list(df.columns),
        )
        monitor.log_shap(values, base_values=base_values)

        # Hand-rolled summary table with hardcoded column widths.
        mean_abs = np.abs(values).mean(axis=0)
        print()
        print(f"{'Feature':<20} {'Mean |SHAP|':>12}")
        print("-" * 33)
        for col, val in zip(df.columns, mean_abs):
            print(f"{col:<20} {val:>12.4f}")
        print()
        print(f"Logged {len(df)} samples to {args.data_dir}")

    except Exception as e:  # noqa: BLE001 — intentionally broad for the "before"
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
