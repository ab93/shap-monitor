"""Generate real demo data for the shapmonitor CLI.

Creates:
  demo/model.pkl           — small RandomForest classifier
  demo/batch_current.csv   — fresh batch for `shapmonitor log` demo
  demo/shap_logs/          — 14 days of logged SHAP values (with drift on days 8-14)
"""

from __future__ import annotations

import shutil
import sys
from datetime import date, timedelta
from pathlib import Path

import joblib
import numpy as np
from freezegun import freeze_time
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

DEMO_DIR = Path(__file__).resolve().parent.parent / "demo"
SHAP_LOGS_DIR = DEMO_DIR / "shap_logs"
N_FEATURES = 8
N_SAMPLES_PER_DAY = 200
N_DAYS = 14
DRIFT_START_DAY = 8  # 0-indexed: drift begins on day 8
DRIFT_FEATURE_IDX = 2  # inject drift into feature 2
DRIFT_MAGNITUDE = 1.5
RANDOM_SEED = 42

FEATURE_NAMES = [
    "age",
    "income",
    "credit_score",
    "debt_ratio",
    "employment_years",
    "num_accounts",
    "recent_inquiries",
    "payment_history",
]


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)

    # Clean slate
    if DEMO_DIR.exists():
        shutil.rmtree(DEMO_DIR)
    DEMO_DIR.mkdir(parents=True)
    SHAP_LOGS_DIR.mkdir()

    print(f"Generating demo data in {DEMO_DIR} ...")

    # 1. Train a small model
    X_train, y_train = make_classification(
        n_samples=2000,
        n_features=N_FEATURES,
        n_informative=5,
        n_redundant=1,
        n_classes=2,
        random_state=RANDOM_SEED,
    )
    model = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=RANDOM_SEED)
    model.fit(X_train, y_train)
    model_path = DEMO_DIR / "model.pkl"
    joblib.dump(model, model_path)
    print(f"  Model saved to {model_path}")

    # 2. Create a SHAP explainer
    import shap

    explainer = shap.TreeExplainer(model)

    # 3. Simulate 14 days of logged batches
    #    Use log_shap with pre-extracted positive-class SHAP values
    #    (TreeExplainer on binary classifiers returns 3D values: n_samples x n_features x 2)
    from shapmonitor import SHAPMonitor

    start_date = date.today() - timedelta(days=N_DAYS)

    for day_offset in range(N_DAYS):
        current_date = start_date + timedelta(days=day_offset)

        # Generate a batch of data
        X_batch, y_batch = make_classification(
            n_samples=N_SAMPLES_PER_DAY,
            n_features=N_FEATURES,
            n_informative=5,
            n_redundant=1,
            n_classes=2,
            random_state=RANDOM_SEED + day_offset,
        )

        # Inject drift on later days
        if day_offset >= DRIFT_START_DAY:
            drift_amount = (
                DRIFT_MAGNITUDE * (day_offset - DRIFT_START_DAY + 1) / (N_DAYS - DRIFT_START_DAY)
            )
            X_batch[:, DRIFT_FEATURE_IDX] += drift_amount + rng.normal(0, 0.3, N_SAMPLES_PER_DAY)

        # Compute SHAP values and extract the positive class (index 1)
        explanation = explainer(X_batch)
        shap_values = explanation.values[:, :, 1]  # (n_samples, n_features) for class 1
        base_values = explanation.base_values[:, 1]  # (n_samples,) for class 1

        # Freeze time so the backend writes to the correct date partition
        fake_dt = f"{current_date.isoformat()} 12:00:00"
        with freeze_time(fake_dt):
            monitor = SHAPMonitor(
                data_dir=str(SHAP_LOGS_DIR),
                sample_rate=1.0,
                model_version="v1",
                feature_names=FEATURE_NAMES,
            )
            monitor.log_shap(shap_values, base_values=base_values)

        status = " (drift injected)" if day_offset >= DRIFT_START_DAY else ""
        print(
            f"  Day {day_offset + 1:2d} ({current_date}): logged {N_SAMPLES_PER_DAY} samples{status}"
        )

    # 4. Save a current batch CSV for the `shapmonitor log` demo
    import pandas as pd

    X_current, _ = make_classification(
        n_samples=5000,
        n_features=N_FEATURES,
        n_informative=5,
        n_redundant=1,
        n_classes=2,
        random_state=RANDOM_SEED + 999,
    )
    batch_df = pd.DataFrame(X_current, columns=FEATURE_NAMES)
    batch_path = DEMO_DIR / "batch_current.csv"
    batch_df.to_csv(batch_path, index=False)
    print(f"  Current batch saved to {batch_path}")

    print("\nDone! Try these commands:")
    print(f"  shapmonitor log {batch_path} --model {model_path} --data-dir {SHAP_LOGS_DIR}")
    print(f"  shapmonitor report summary --data-dir {SHAP_LOGS_DIR} --period last-7d")
    print(
        f"  shapmonitor report drift --data-dir {SHAP_LOGS_DIR} --ref last-14d..last-7d --curr last-7d..now"
    )
    print(f"  shapmonitor watch --data-dir {SHAP_LOGS_DIR}")


if __name__ == "__main__":
    sys.exit(main() or 0)
