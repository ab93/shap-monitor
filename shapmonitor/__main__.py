"""``python -m shapmonitor`` — log and report on SHAP values."""

from __future__ import annotations

import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd
import shap

from datetime import datetime, timedelta
from shapmonitor import SHAPMonitor

from shapmonitor.analysis import SHAPAnalyzer
from shapmonitor.backends import BackendFactory


def main() -> None:
    parser = argparse.ArgumentParser(prog="shapmonitor")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ----- log -----
    log_p = subparsers.add_parser("log", help="Log SHAP values for a batch")
    log_p.add_argument("data", help="Path to CSV file with feature data")
    log_p.add_argument("--model", required=True, help="Path to pickled model")
    log_p.add_argument("--data-dir", default="./shap_logs", help="Output directory")
    log_p.add_argument("--sample-rate", type=float, default=1.0, help="Sampling rate (0.0-1.0)")
    log_p.add_argument("--model-version", default="v1", help="Model version tag")

    # ----- report -----
    report_p = subparsers.add_parser("report", help="Generate reports")
    report_subs = report_p.add_subparsers(dest="report_command", required=True)

    summary_p = report_subs.add_parser("summary", help="Show summary table")
    summary_p.add_argument("--data-dir", default="./shap_logs", help="SHAP log directory")
    summary_p.add_argument("--days", type=int, default=30, help="Lookback window (days)")

    args = parser.parse_args()

    if args.command == "log":
        cmd_log(args)
    elif args.command == "report" and args.report_command == "summary":
        cmd_summary(args)


def cmd_log(args: argparse.Namespace) -> None:
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
    print(f"Logged {len(df)} samples to {args.data_dir}")


def cmd_summary(args: argparse.Namespace) -> None:
    if not os.path.isdir(args.data_dir):
        print(f"Error: data dir not found: {args.data_dir}", file=sys.stderr)
        sys.exit(1)

    backend = BackendFactory.get_backend("parquet", file_dir=args.data_dir)
    analyzer = SHAPAnalyzer(backend)

    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=args.days)
    summary_df = analyzer.summary(start_dt=start_dt, end_dt=end_dt)

    print()
    print(f"{'Feature':<20} {'Mean |SHAP|':>12} {'Mean':>10} {'Std':>10}")
    print("-" * 54)
    for feat, row in summary_df.iterrows():
        print(f"{feat:<20} {row['mean_abs']:>12.4f} {row['mean']:>10.4f} {row['std']:>10.4f}")
    print()


if __name__ == "__main__":
    main()
