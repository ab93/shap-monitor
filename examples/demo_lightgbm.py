"""
Demo: Real-world usage of SHAPMonitor with LightGBM.

Uses California Housing dataset to demonstrate:
1. Training a LightGBM model
2. Logging SHAP explanations with SHAPMonitor
3. Simulating multiple batches over time
4. Reading back and analyzing the data

Run from project root:
    poetry run python examples/demo_lightgbm.py
"""

import logging
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

import lightgbm as lgb
import shap
from freezegun import freeze_time
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from shapmonitor import SHAPMonitor
from shapmonitor.analysis import SHAPAnalyzer
from shapmonitor.backends import ParquetBackend

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Output directory for this demo
DATA_DIR = Path(".demo_shap_logs")

pd.set_option("display.max_columns", 10)


def load_data():
    """Load California Housing dataset."""
    data = fetch_california_housing(as_frame=True)
    X, y = data.data, data.target
    logger.info("Loaded California Housing: %d samples, %d features", len(X), X.shape[1])
    logger.info("Features: %s", list(X.columns))
    logger.info("Target: %s", data.target_names)
    return X, y


def train_model(X_train, y_train):
    """Train a LightGBM regressor."""
    model = lgb.LGBMRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        verbosity=-1,
    )
    model.fit(X_train, y_train)
    logger.info("Trained LightGBM model")
    return model


def simulate_production_batches(
    monitor: SHAPMonitor, X, batch_size: int = 100, days_to_simulate: int = 7
):
    """
    Simulate production inference by logging multiple batches across multiple days.

    Uses freezegun to mock timestamps so data is spread across different dates.
    This helps test the time-based analysis features.
    """
    n_batches = len(X) // batch_size
    batches_per_day = n_batches // days_to_simulate

    logger.info(
        "Simulating %d production batches across %d days (batch_size=%d)",
        n_batches,
        days_to_simulate,
        batch_size,
    )

    base_date = datetime.now() - timedelta(days=days_to_simulate)
    batch_idx = 0

    for day in range(days_to_simulate):
        frozen_date = base_date + timedelta(days=day)

        with freeze_time(frozen_date):
            for _ in range(batches_per_day):
                if batch_idx >= n_batches:
                    break

                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                batch = X.iloc[start_idx:end_idx]

                monitor.log_batch(batch)
                batch_idx += 1

            logger.info(
                "Day %d (%s): logged %d batches", day + 1, frozen_date.date(), batches_per_day
            )

    logger.info("Finished logging %d batches across %d days", batch_idx, days_to_simulate)


def analyze_results(data_dir: Path):
    """Read back and analyze the logged data."""
    backend = ParquetBackend(data_dir)
    analyzer = SHAPAnalyzer(backend)

    # Read all data from the past week
    today = datetime.now()
    return analyzer.summary(today - timedelta(days=8), today + timedelta(days=1))


def compare_periods(data_dir: Path):
    """Compare SHAP explanations between two time periods."""
    backend = ParquetBackend(data_dir)
    analyzer = SHAPAnalyzer(backend)

    today = datetime.now()
    # Period 1: first half of the week
    period_1_start = today - timedelta(days=7)
    period_1_end = today - timedelta(days=4)
    # Period 2: second half of the week
    period_2_start = today - timedelta(days=3)
    period_2_end = today + timedelta(days=1)

    logger.info("Comparing periods:")
    logger.info("  Period 1: %s to %s", period_1_start.date(), period_1_end.date())
    logger.info("  Period 2: %s to %s", period_2_start.date(), period_2_end.date())

    comparison_df = analyzer.compare_time_periods(
        period_1_start, period_1_end, period_2_start, period_2_end
    )
    return comparison_df


def main():
    # Clean up previous run
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
        logger.info("Cleaned up previous demo data")

    # Load data
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    test_score = model.score(X_test, y_test)
    logger.info("Test R^2 Score: %.4f", test_score)

    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)

    # Initialize monitor with 100% sampling for demo
    monitor = SHAPMonitor(
        explainer=explainer,
        data_dir=DATA_DIR,
        sample_rate=1.0,  # Log everything for demo
        model_version="lgbm-v1",
        feature_names=list(X.columns),
    )

    # Simulate production batches
    simulate_production_batches(monitor, X_test, batch_size=100)

    # Analyze results - summary
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    summary_df = analyze_results(DATA_DIR)
    print(summary_df)
    print(f"n_samples: {summary_df.attrs.get('n_samples', 'N/A')}")

    # Compare time periods
    print("\n" + "=" * 60)
    print("TIME PERIOD COMPARISON")
    print("=" * 60)
    comparison_df = compare_periods(DATA_DIR)
    print(comparison_df)

    print(f"\nDemo data saved to: {DATA_DIR.absolute()}")
    print("You can now use this data to test analysis features.")


if __name__ == "__main__":
    df = main()
