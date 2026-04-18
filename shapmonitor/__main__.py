"""``python -m shapmonitor`` — log SHAP values for a batch of predictions."""

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
    parser = argparse.ArgumentParser(
        prog="shapmonitor",
        description="Log SHAP values for a batch of predictions.",
    )
    parser.add_argument("data", help="Path to CSV file with feature data")
    parser.add_argument("--model", required=True, help="Path to pickled model")
    parser.add_argument("--data-dir", default="./shap_logs", help="Output directory")
    parser.add_argument("--sample-rate", type=float, default=1.0, help="Sampling rate (0.0-1.0)")
    parser.add_argument("--model-version", default="v1", help="Model version tag")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"Error: data file not found: {args.data}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.model):
        print(f"Error: model file not found: {args.model}", file=sys.stderr)
        sys.exit(1)
    if not 0.0 <= args.sample_rate <= 1.0:
        print("Error: --sample-rate must be between 0.0 and 1.0", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.data)
    model = joblib.load(args.model)  # noqa: S301

    explainer = shap.TreeExplainer(model)
    exp = explainer(df.values)

    # Binary-classifier SHAP returns 3-D values; take the positive class.
    values = exp.values[:, :, 1] if exp.values.ndim == 3 else exp.values
    base_values = (
        exp.base_values[:, 1] if np.asarray(exp.base_values).ndim == 2 else exp.base_values
    )

    monitor = SHAPMonitor(
        data_dir=args.data_dir,
        sample_rate=args.sample_rate,
        model_version=args.model_version,
        feature_names=list(df.columns),
    )
    monitor.log_shap(values, base_values=base_values)

    mean_abs = np.abs(values).mean(axis=0)
    print()
    print(f"{'Feature':<20} {'Mean |SHAP|':>12}")
    print("-" * 33)
    for col, val in zip(df.columns, mean_abs):
        print(f"{col:<20} {val:>12.4f}")
    print()
    print(f"Logged {len(df)} samples to {args.data_dir}")


if __name__ == "__main__":
    main()
