import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())


__all__ = ["SHAPMonitor"]


def __getattr__(name: str):
    if name == "SHAPMonitor":
        from shapmonitor.monitor import SHAPMonitor

        return SHAPMonitor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
