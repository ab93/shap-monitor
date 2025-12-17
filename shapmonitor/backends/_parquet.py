import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from shapmonitor.backends._base import BaseBackend
from shapmonitor.types import PathLike, ExplanationBatch


_logger = logging.getLogger(__name__)


class ParquetBackend(BaseBackend):
    """
    Backend for storing and retrieving SHAP explanations using Parquet files.


    Parameters
    ----------
    file_dir : PathLike
        Directory where Parquet files will be stored.

    Raises
    ------
    NotADirectoryError
        If the provided file_dir is not a valid directory.

    """

    def __init__(self, file_dir: PathLike):
        self._file_dir = Path(file_dir)
        self._file_dir.mkdir(parents=True, exist_ok=True)

        if not self._file_dir.is_dir():
            raise NotADirectoryError(f"{self._file_dir} is not a valid directory.")

    @property
    def file_dir(self) -> Path:
        return self._file_dir

    def read(self, start_dt: datetime, end_dt: datetime, batch_id: str | None = None):
        pass

    def write(self, batch: ExplanationBatch) -> None:
        """
        Write a batch of explanations to a Parquet file.

        Parameters
        ----------
        batch : ExplanationBatch
            The batch of explanations to write.
        """
        partition_date = batch.timestamp.strftime("%Y-%m-%d")
        file_path = self.file_dir / partition_date / f"{batch.batch_id}.parquet"
        file_path.parent.mkdir(parents=True, exist_ok=True)

        df = batch.to_dataframe()
        df.to_parquet(file_path, index=False)
        _logger.info("Wrote batch %s to %s", batch.batch_id, file_path)

    def read_all(self) -> list[ExplanationBatch]:
        pass

    def delete(self, before_dt: datetime) -> None:
        pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    backend = ParquetBackend("./shap_data/")
    print("ParquetBackend initialized at:", backend.file_dir)

    # Example usage of write method
    batch = ExplanationBatch(
        timestamp=datetime.now(),
        batch_id="batch_001",
        model_version="v1.0",
        n_samples=5,
        base_values=[0.5] * 5,
        shap_values={"feature1": [0.1, 0.2, 0.3, 0.4, 0.5]},
        predictions=[0.6, 0.7, 0.8, 0.9, 1.0],
    )
    backend.write(batch)

    df = pd.read_parquet("shap_data/2025-12-17/batch_001.parquet")
    print(df)
