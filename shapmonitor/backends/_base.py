from abc import ABCMeta, abstractmethod
from datetime import datetime


from shapmonitor.types import ExplanationBatch, DFrameLike


class BaseBackend(metaclass=ABCMeta):
    """Abstract base class for all backends."""

    @abstractmethod
    def read(
        self,
        start_dt: datetime,
        end_dt: datetime,
    ) -> DFrameLike:
        """Read data from the backend."""
        pass  # pragma: no cover

    @abstractmethod
    def write(self, batch: ExplanationBatch) -> None:
        """Write data to the backend."""
        pass  # pragma: no cover

    @abstractmethod
    def delete(self, cutoff_dt: datetime) -> None:
        """Delete data before a certain datetime."""
        pass  # pragma: no cover
