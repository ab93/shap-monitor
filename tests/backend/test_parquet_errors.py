"""Tests for ParquetBackend error handling."""

import sys
from datetime import datetime

import pytest

from shapmonitor.backends import ParquetBackend


class TestParquetBackendErrors:
    """Tests for error handling in ParquetBackend."""

    def test_read_nonexistent_directory_returns_empty(self, tmp_path):
        """Reading from empty backend should return empty DataFrame."""
        backend = ParquetBackend(tmp_path)
        df = backend.read(datetime(2025, 1, 1))
        assert df.empty

    @pytest.mark.skipif(
        sys.platform == "win32", reason="Permission tests not reliable on Windows"
    )
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
