import logging
import shutil
from datetime import datetime, date
from pathlib import Path

import pandas as pd

from shapmonitor.backends._base import BaseBackend
from shapmonitor.types import PathLike, ExplanationBatch, DFrameLike

_logger = logging.getLogger(__name__)


class ParquetBackend(BaseBackend):
    """
    Backend for storing and retrieving SHAP explanations using Parquet files.


    Parameters
    ----------
    file_dir : PathLike
        Directory where Parquet files will be stored.
    purge_existing : bool, optional
        If True, existing files in the directory will be deleted (default is False).

    Raises
    ------
    NotADirectoryError
        If the provided file_dir is not a valid directory.

    """

    def __init__(self, file_dir: PathLike, purge_existing: bool = False) -> None:
        self._file_dir = Path(file_dir)
        _logger.info("ParquetBackend initialized at: %s", file_dir)

        if purge_existing and self._file_dir.exists():
            shutil.rmtree(self._file_dir)
            _logger.warning("Purged existing files in directory: %s", file_dir)

        self._file_dir.mkdir(parents=True, exist_ok=True)

        if not self._file_dir.is_dir():
            raise NotADirectoryError(f"{self._file_dir} is not a valid directory.")

    @property
    def file_dir(self) -> Path:
        """Get the directory where Parquet files are stored."""
        return self._file_dir

    def read(self, start_dt: datetime | date, end_dt: datetime | date | None = None) -> DFrameLike:
        """
        Read explanations from Parquet files within a specified date range.


        Parameters
        ----------
        start_dt: datetime
            Start datetime for filtering explanations.
        end_dt: datetime | None
            End datetime for filtering explanations.

        Returns
        -------
        DataFrame
            A DataFrame containing the explanations within the specified range.
        """
        start_date = start_dt.date().strftime("%Y-%m-%d")
        if end_dt is None:
            end_date = start_date
        else:
            end_date = end_dt.date().strftime("%Y-%m-%d")

        _logger.debug("Reading data from %s to %s", start_date, end_date)

        date_range = pd.date_range(start_date, end_date, freq="D")

        dfs = []
        for _date in date_range:
            file_dir = self._file_dir / _date.strftime("%Y-%m-%d")

            for file_path in file_dir.glob("*.parquet"):
                _df = pd.read_parquet(file_path)
                dfs.append(_df)

        if dfs:
            return pd.concat(dfs, ignore_index=True)
        return pd.DataFrame()

    def write(self, batch: ExplanationBatch) -> Path:
        """
        Write a batch of explanations to a Parquet file.

        Parameters
        ----------
        batch : ExplanationBatch
            The batch of explanations to write.

        Returns
        -------
        Path
            The path to the written Parquet file.
        """
        partition_date = batch.timestamp.strftime("%Y-%m-%d")
        file_path = self.file_dir / partition_date / f"{batch.batch_id}.parquet"
        file_path.parent.mkdir(parents=True, exist_ok=True)

        df = batch.to_dataframe()
        df.to_parquet(file_path, index=False)

        _logger.info("Wrote batch %s to %s", batch.batch_id, file_path)
        return file_path

    def delete(self, cutoff_dt: datetime | date) -> int:
        """
        Delete Parquet files containing explanations before a specified datetime.

        Parameters
        ----------
        cutoff_dt : datetime
            Datetime before which files will be deleted.

        Returns
        -------
        int
            Number of partitions deleted.
        """
        cutoff_date = cutoff_dt.date()
        deleted_count = 0

        for partition_dir in self.file_dir.iterdir():
            if not partition_dir.is_dir():
                continue

            try:
                partition_date = datetime.strptime(partition_dir.name, "%Y-%m-%d").date()
            except ValueError:
                _logger.debug("Skipping non-date directory: %s", partition_dir.name)
                continue

            if partition_date < cutoff_date:
                shutil.rmtree(partition_dir)
                _logger.info("Deleted partition: %s", partition_dir.name)
                deleted_count += 1

        return deleted_count
