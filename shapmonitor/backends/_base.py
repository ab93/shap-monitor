from abc import ABCMeta, abstractmethod
from datetime import datetime, date

import pandas as pd

from shapmonitor.types import ExplanationBatch


class BaseBackend(metaclass=ABCMeta):
    """Abstract base class for all backends."""

    @abstractmethod
    def read(
        self,
        start_dt: datetime | date | None = None,
        end_dt: datetime | date | None = None,
        batch_id: str | None = None,
        model_version: str | None = None,
    ) -> pd.DataFrame:
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
