from shapmonitor.backends._parquet import ParquetBackend
from shapmonitor.backends._base import BaseBackend


class BackendFactory:
    """Factory class for creating backend instances."""

    _backends = {
        "parquet": ParquetBackend,
    }

    @classmethod
    def get_backend(cls, backend_name: str, *args, **kwargs) -> BaseBackend:
        """Get an instance of the specified backend.

        Parameters
        ----------
        backend_name : str
            Name of the backend to instantiate.
        *args
            Positional arguments to pass to the backend constructor.
        **kwargs
            Keyword arguments to pass to the backend constructor.

        Returns
        -------
        BaseBackend
            An instance of the requested backend.

        Raises
        ------
        ValueError
            If the specified backend is not supported.
        """
        if backend_name not in cls._backends:
            raise ValueError(f"Unsupported backend: {backend_name}")

        backend_class = cls._backends[backend_name]
        return backend_class(*args, **kwargs)


__all__ = ["ParquetBackend", "BaseBackend"]
