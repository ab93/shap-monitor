__all__ = ["SHAPAnalyzer"]


def __getattr__(name: str):
    if name == "SHAPAnalyzer":
        from shapmonitor.analysis._analyzer import SHAPAnalyzer

        return SHAPAnalyzer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
