"""Tests for ParquetBackend."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from shapmonitor.backends import ParquetBackend
from shapmonitor.types import ExplanationBatch


@pytest.fixture
def sample_batch() -> ExplanationBatch:
    """Create a sample ExplanationBatch for testing."""
    return ExplanationBatch(
        timestamp=datetime(2025, 12, 17, 10, 30, 0),
        batch_id="test_batch_001",
        model_version="v1.0",
        n_samples=3,
        base_values=np.array([0.5, 0.5, 0.5]),
        shap_values={
            "feature_a": np.array([0.1, 0.2, 0.3]),
            "feature_b": np.array([-0.1, -0.2, -0.3]),
        },
        predictions=np.array([0.6, 0.7, 0.8]),
    )


class TestParquetBackend:
    """Tests for ParquetBackend."""

    def test_creates_directory(self, tmp_path):
        """Backend should create the directory if it doesn't exist."""
        data_dir = tmp_path / "new_dir"
        backend = ParquetBackend(data_dir)
        assert backend.file_dir.exists()

    def test_write_creates_partitioned_file(self, tmp_path, sample_batch):
        """Write should create file in date-partitioned directory."""
        backend = ParquetBackend(tmp_path)
        backend.write(sample_batch)

        expected_path = tmp_path / "2025-12-17" / "test_batch_001.parquet"
        assert expected_path.exists()

    def test_write_creates_valid_parquet(self, tmp_path, sample_batch):
        """Written file should be readable with correct columns."""
        backend = ParquetBackend(tmp_path)
        backend.write(sample_batch)

        df = pd.read_parquet(tmp_path / "2025-12-17" / "test_batch_001.parquet")

        assert len(df) == 3
        assert "shap_feature_a" in df.columns
        assert "shap_feature_b" in df.columns
        assert "prediction" in df.columns
        np.testing.assert_array_almost_equal(df["shap_feature_a"].values, [0.1, 0.2, 0.3])

    def test_create_multiple_writes(self, tmp_path, sample_batch):
        """Multiple writes should create separate files."""
        backend = ParquetBackend(tmp_path)
        backend.write(sample_batch)

        # Modify batch_id for second write
        sample_batch.batch_id = "test_batch_002"
        backend.write(sample_batch)

        path1 = tmp_path / "2025-12-17" / "test_batch_001.parquet"
        path2 = tmp_path / "2025-12-17" / "test_batch_002.parquet"

        assert path1.exists()
        assert path2.exists()

    def test_read_single_day(self, tmp_path, sample_batch):
        """Read should return DataFrame for a single day."""
        backend = ParquetBackend(tmp_path)
        backend.write(sample_batch)

        df = backend.read(datetime(2025, 12, 17))

        assert len(df) == 3
        assert "shap_feature_a" in df.columns

    def test_read_date_range(self, tmp_path):
        """Read should return data across multiple days."""
        backend = ParquetBackend(tmp_path)

        batch1 = ExplanationBatch(
            timestamp=datetime(2025, 12, 17, 10, 0, 0),
            batch_id="batch_day1",
            model_version="v1.0",
            n_samples=2,
            base_values=np.array([0.5, 0.5]),
            shap_values={"feat": np.array([0.1, 0.2])},
        )
        batch2 = ExplanationBatch(
            timestamp=datetime(2025, 12, 18, 10, 0, 0),
            batch_id="batch_day2",
            model_version="v1.0",
            n_samples=2,
            base_values=np.array([0.5, 0.5]),
            shap_values={"feat": np.array([0.3, 0.4])},
        )
        backend.write(batch1)
        backend.write(batch2)

        df = backend.read(datetime(2025, 12, 17), datetime(2025, 12, 18))
        print(df)

        assert len(df) == 4

    def test_read_empty_returns_empty_dataframe(self, tmp_path):
        """Read with no data should return empty DataFrame."""
        backend = ParquetBackend(tmp_path)

        df = backend.read(datetime(2025, 12, 17))

        assert len(df) == 0
        assert isinstance(df, pd.DataFrame)

    def test_delete_removes_old_partitions(self, tmp_path):
        """Delete should remove partitions before cutoff and keep newer ones."""
        backend = ParquetBackend(tmp_path)

        batch_old = ExplanationBatch(
            timestamp=datetime(2025, 12, 15, 10, 0, 0),
            batch_id="old_batch",
            model_version="v1.0",
            n_samples=2,
            base_values=np.array([0.5, 0.5]),
            shap_values={"feat": np.array([0.1, 0.2])},
        )
        batch_new = ExplanationBatch(
            timestamp=datetime(2025, 12, 18, 10, 0, 0),
            batch_id="new_batch",
            model_version="v1.0",
            n_samples=2,
            base_values=np.array([0.5, 0.5]),
            shap_values={"feat": np.array([0.3, 0.4])},
        )
        backend.write(batch_old)
        backend.write(batch_new)

        deleted_count = backend.delete(datetime(2025, 12, 17))

        assert deleted_count == 1
        assert not (tmp_path / "2025-12-15").exists()
        assert (tmp_path / "2025-12-18").exists()
