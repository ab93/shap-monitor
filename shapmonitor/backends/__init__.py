__all__ = ["ParquetBackend", "BaseBackend", "BackendFactory"]


def __getattr__(name: str):
    if name == "ParquetBackend":
        from shapmonitor.backends._parquet import ParquetBackend

        return ParquetBackend
    if name == "BaseBackend":
        from shapmonitor.backends._base import BaseBackend

        return BaseBackend
    if name == "BackendFactory":
        from shapmonitor.backends._factory import BackendFactory

        return BackendFactory

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
