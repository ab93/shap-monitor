"""Tests for ParquetBackend error handling."""

import shutil
import sys
from datetime import datetime
from unittest.mock import patch

import numpy as np
import pyarrow
import pytest

from shapmonitor.backends import BackendFactory, ParquetBackend
from shapmonitor.types import ExplanationBatch


class TestParquetBackendErrors:
    """Tests for error handling in ParquetBackend."""

    def test_read_nonexistent_directory_returns_empty(self, tmp_path):
        """Reading from empty backend should return empty DataFrame."""
        backend = ParquetBackend(tmp_path)
        df = backend.read(datetime(2025, 1, 1))
        assert df.empty

    @pytest.mark.skipif(sys.platform == "win32", reason="Permission tests not reliable on Windows")
    def test_read_permission_error_propagates(self, tmp_path):
        """Permission errors should propagate, not be swallowed."""
        backend = ParquetBackend(tmp_path)
        # Make directory unreadable
        original_mode = tmp_path.stat().st_mode
        tmp_path.chmod(0o000)
        try:
            with pytest.raises(PermissionError):
                backend.read(datetime(2025, 1, 1))
        finally:
            tmp_path.chmod(original_mode)  # Restore permissions for cleanup

    def test_read_no_filters_raises(self, tmp_path):
        """read() with no arguments should raise ValueError."""
        backend = ParquetBackend(tmp_path)
        with pytest.raises(ValueError, match="At least one filter"):
            backend.read()

    def test_read_batch_id_filter(self, tmp_path):
        """read() with only a batch_id filter should return matching rows."""
        backend = ParquetBackend(tmp_path)
        batch = ExplanationBatch(
            timestamp=datetime(2025, 6, 1),
            batch_id="my-batch",
            model_version="v1",
            n_samples=3,
            base_values=np.zeros(3),
            shap_values={"f0": np.ones(3)},
        )
        backend.write(batch)

        df = backend.read(batch_id="my-batch")
        assert len(df) == 3
        assert (df["batch_id"] == "my-batch").all()

    def test_read_filenotfound_returns_empty(self, tmp_path):
        """read() should return empty DataFrame when the data directory was removed."""
        backend = ParquetBackend(tmp_path)
        shutil.rmtree(tmp_path)
        df = backend.read(datetime(2025, 1, 1))
        assert df.empty

    def test_read_arrowinvalid_reraises(self, tmp_path):
        """read() should re-raise ArrowInvalid errors that aren't 'no match' cases."""
        backend = ParquetBackend(tmp_path)
        exc = pyarrow.lib.ArrowInvalid("something unexpected went wrong")
        with patch("pandas.read_parquet", side_effect=exc):
            with pytest.raises(pyarrow.lib.ArrowInvalid):
                backend.read(datetime(2025, 1, 1))

    def test_unsupported_backend_raises(self):
        """BackendFactory should raise ValueError for unknown backend names."""
        with pytest.raises(ValueError, match="Unsupported backend"):
            BackendFactory.get_backend("nonexistent")

    def test_delete_skips_non_directory_date_entries(self, tmp_path):
        """delete() should skip files named date=* that are not directories."""
        backend = ParquetBackend(tmp_path)
        # Create a file (not a dir) that matches the date=* glob pattern
        fake_file = tmp_path / "date=not-a-dir"
        fake_file.write_text("oops")
        # Should not raise and should skip the file
        deleted = backend.delete(datetime(2099, 1, 1))
        assert deleted == 0

    def test_delete_skips_invalid_date_format(self, tmp_path):
        """delete() should skip date= directories with unparseable date strings."""
        backend = ParquetBackend(tmp_path)
        bad_dir = tmp_path / "date=not-a-date"
        bad_dir.mkdir()
        deleted = backend.delete(datetime(2099, 1, 1))
        assert deleted == 0

    def test_unsupported_partition_by_at_init_raises(self, tmp_path):
        """ParquetBackend should raise ValueError at init for unsupported partition keys."""
        with pytest.raises(ValueError, match="Invalid partition_by value"):
            ParquetBackend(tmp_path, partition_by=["unsupported_key"])

    def test_partition_path_batch_id_key(self, tmp_path):
        """_get_partition_path should handle batch_id as a partition key."""
        backend = ParquetBackend(tmp_path, partition_by=["batch_id"])
        batch = ExplanationBatch(
            timestamp=datetime(2025, 6, 1),
            batch_id="b-xyz",
            model_version="v1",
            n_samples=2,
            base_values=np.zeros(2),
            shap_values={"f0": np.ones(2)},
        )
        path = backend.write(batch)
        assert "batch_id=b-xyz" in str(path)
