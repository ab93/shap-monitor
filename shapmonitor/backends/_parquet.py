import logging
import shutil
from collections.abc import Sequence
from datetime import datetime, date
from pathlib import Path

import pandas as pd

from shapmonitor.backends._base import BaseBackend
from shapmonitor.types import PathLike, ExplanationBatch, DFrameLike

_logger = logging.getLogger("shapmonitor.backends")


_SUPPORTED_PARTITION_BY_VALUES = {"date", "batch_id", "model_version"}


class ParquetBackend(BaseBackend):
    """
    Backend for storing and retrieving SHAP explanations using Parquet files.


    Parameters
    ----------
    file_dir : PathLike
        Directory where Parquet files will be stored.
    partition_by : Sequence[str], optional
        List of columns to partition the data by (default is ("date",)).
        Other Parameters include: "batch_id", "model_version".
    purge_existing : bool, optional
        If True, existing files in the directory will be deleted (default is False).

    Raises
    ------
    ValueError
        If invalid partition_by values are provided.
    NotADirectoryError
        If the provided file_dir is not a valid directory.

    """

    def __init__(
        self,
        file_dir: PathLike,
        partition_by: Sequence[str] | None = None,
        purge_existing: bool = False,
    ) -> None:
        self._file_dir = Path(file_dir)
        _logger.info("ParquetBackend initialized at: %s", self._file_dir)

        self.partition_by = partition_by or ["date"]

        if set(self.partition_by) - _SUPPORTED_PARTITION_BY_VALUES:
            raise ValueError(
                "Invalid partition_by value: %s; Supported values include: %s",
                self.partition_by,
                _SUPPORTED_PARTITION_BY_VALUES,
            )

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

    def read(
        self,
        start_dt: datetime | date,
        end_dt: datetime | date | None = None,
        batch_id: str | None = None,
        model_version: str | None = None,
    ) -> DFrameLike:
        """
        Read explanations from Parquet files within a specified date range.


        Parameters
        ----------
        start_dt: datetime
            Start datetime for filtering explanations.
        end_dt: datetime | None
            End datetime for filtering explanations.
        batch_id : str, optional
            Batch ID to filter explanations.
        model_version : str, optional
            Model version to filter explanations.

        Returns
        -------
        DataFrame
            A DataFrame containing the explanations within the specified range.
        """
        filters = []

        start_date = start_dt.date().strftime("%Y-%m-%d")
        end_date = end_dt.date().strftime("%Y-%m-%d") if end_dt else start_date

        filters.append(("date", ">=", start_date))
        filters.append(("date", "<=", end_date))

        if batch_id:
            filters.append(("batch_id", "==", batch_id))
        if model_version:
            filters.append(("model_version", "==", model_version))

        _logger.debug("Reading data with filters: %s", filters)

        try:
            return pd.read_parquet(self._file_dir, filters=filters)
        except Exception as e:
            _logger.error("Error reading Parquet files: %s", e)
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
        file_path = self._get_partition_path(batch)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        df = batch.to_dataframe()

        # Drop partition columns to avoid type conflicts with Hive-style partitioning
        # (partition columns are reconstructed from directory names when reading)
        cols_to_drop = [col for col in self.partition_by if col in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

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

        # Find all date= partition directories (Hive-style)
        for date_dir in self.file_dir.rglob("date=*"):
            if not date_dir.is_dir():
                continue

            try:
                date_str = date_dir.name.split("=", 1)[1]
                partition_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            except (ValueError, IndexError):
                _logger.debug("Skipping invalid date directory: %s", date_dir.name)
                continue

            if partition_date < cutoff_date:
                shutil.rmtree(date_dir)
                _logger.info("Deleted partition: %s", date_dir)
                deleted_count += 1

        return deleted_count

    def _get_partition_path(self, batch: ExplanationBatch) -> Path:
        """
        Get the partition path for a given batch based on partitioning columns.

        Parameters
        ----------
        batch : ExplanationBatch
            The batch of explanations.

        Returns
        -------
        Path
            The partition path.

        Raises:
        ------
        ValueError
            If an unsupported partition key is encountered.
        """
        partition_dirs = []
        for key in self.partition_by:
            if key == "date":
                value = batch.timestamp.strftime("%Y-%m-%d")
            elif key == "batch_id":
                value = batch.batch_id
            elif key == "model_version":
                value = batch.model_version
            else:
                raise ValueError(f"Unsupported partition key: {key}")
            partition_dirs.append(f"{key}={value}")

        return self._file_dir.joinpath(*partition_dirs, f"{batch.batch_id}.parquet")
