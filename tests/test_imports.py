"""Tests for top-level shapmonitor imports."""


def test_shap_analyzer_importable_from_top_level():
    """SHAPAnalyzer should be importable from shapmonitor directly."""
    from shapmonitor import SHAPAnalyzer

    assert SHAPAnalyzer is not None


def test_period_importable_from_top_level():
    """Period should be importable from shapmonitor directly."""
    from shapmonitor import Period

    assert Period is not None


def test_parquet_backend_importable_from_top_level():
    """ParquetBackend should be importable from shapmonitor directly."""
    from shapmonitor import ParquetBackend

    assert ParquetBackend is not None


def test_shap_monitor_importable_from_top_level():
    """SHAPMonitor should remain importable from shapmonitor."""
    from shapmonitor import SHAPMonitor

    assert SHAPMonitor is not None


def test_all_exports_present():
    """__all__ should contain the four main exports."""
    import shapmonitor

    assert "SHAPMonitor" in shapmonitor.__all__
    assert "SHAPAnalyzer" in shapmonitor.__all__
    assert "ParquetBackend" in shapmonitor.__all__
    assert "Period" in shapmonitor.__all__
