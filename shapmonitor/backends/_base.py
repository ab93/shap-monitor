from abc import ABCMeta, abstractmethod
from datetime import datetime


from shapmonitor.types import ExplanationBatch


class BaseBackend(metaclass=ABCMeta):
    """Abstract base class for all backends."""

    @abstractmethod
    def read(
        self, start_dt: datetime, end_dt: datetime, batch_id: str | None = None
    ) -> ExplanationBatch:
        """Read data from the backend."""
        pass

    @abstractmethod
    def write(self, batch: ExplanationBatch) -> None:
        """Write data to the backend."""
        pass

    @abstractmethod
    def read_all(self) -> list[ExplanationBatch]:
        """Read all data from the backend."""
        pass

    @abstractmethod
    def delete(self, before_dt: datetime) -> None:
        """Delete data before a certain datetime."""
        pass
