# Storage Backends

Backends determine how and where SHAP explanations are stored and retrieved.

## ParquetBackend

The default backend using Apache Parquet format for efficient columnar storage.

### Basic Usage

```python
from shapmonitor.backends import ParquetBackend

# Create backend
backend = ParquetBackend("/path/to/shap_logs")

# Use with SHAPMonitor
from shapmonitor import SHAPMonitor

monitor = SHAPMonitor(
    explainer=explainer,
    backend=backend  # Or use data_dir instead
)
```

### Directory Structure

Parquet files are organized by date for efficient querying:

```
shap_logs/
├── 2025-12-26/
│   ├── uuid-1234.parquet
│   ├── uuid-5678.parquet
│   └── uuid-9abc.parquet
├── 2025-12-27/
│   ├── uuid-def0.parquet
│   └── uuid-1234.parquet
└── 2025-12-28/
    └── uuid-5678.parquet
```

Each file contains one batch of explanations.

### Data Schema

Each Parquet file contains:

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | datetime | When batch was logged |
| `batch_id` | string | Unique batch identifier (UUID) |
| `model_version` | string | Model version identifier |
| `n_samples` | int | Number of samples in batch |
| `base_value` | float | Expected value from explainer |
| `shap_{feature}` | float | SHAP value for each feature |
| `{feature}` | float | Original feature value |
| `prediction` | float | Model prediction (if provided) |

Example:

```python
import pandas as pd

# Read a Parquet file directly
df = pd.read_parquet("shap_logs/2025-12-26/uuid-1234.parquet")

print(df.columns)
# Index(['timestamp', 'batch_id', 'model_version', 'n_samples', 'base_value',
#        'shap_MedInc', 'shap_HouseAge', 'MedInc', 'HouseAge', 'prediction'], dtype='object')
```

### Configuration

```python
backend = ParquetBackend(file_dir="/path/to/logs")

# Access configuration
backend.file_dir  # Path object
```

## Backend Operations

### Write

Write a batch of explanations:

```python
from shapmonitor.types import ExplanationBatch
from datetime import datetime

# Create batch (normally done by SHAPMonitor)
batch = ExplanationBatch(
    timestamp=datetime.now(),
    batch_id="uuid-1234",
    model_version="v1.0",
    n_samples=100,
    base_values=[0.5] * 100,
    shap_values={"feature_1": [0.1, 0.2, ...], ...},
    feature_values={"feature_1": [1.0, 2.0, ...], ...},
    predictions=[0.6, 0.7, ...]
)

# Write to backend
path = backend.write(batch)
print(f"Wrote to {path}")
```

### Read

Read explanations from a date range:

```python
from datetime import datetime, timedelta

# Read single day
today = datetime.now()
df = backend.read(today, today)

# Read date range
week_ago = today - timedelta(days=7)
df = backend.read(week_ago, today)

print(f"Read {len(df)} samples")
```

**Parameters:**

- `start_dt`: Start date (inclusive)
- `end_dt`: End date (inclusive), defaults to `start_dt`

**Returns:**

- DataFrame with all samples in the date range
- Empty DataFrame if no data found

### Delete

Delete old data to manage storage:

```python
from datetime import datetime, timedelta

# Delete data older than 30 days
cutoff = datetime.now() - timedelta(days=30)
deleted_count = backend.delete(cutoff)

print(f"Deleted {deleted_count} partitions")
```

**Parameters:**

- `cutoff_dt`: Delete data before this date

**Returns:**

- Number of date partitions deleted

## Parquet Benefits

### Storage Efficiency

- **Columnar format**: Only read columns you need
- **Compression**: Efficient compression algorithms
- **Type optimization**: Stores data types efficiently

Typical compression ratios: 5-10x compared to CSV.

### Query Performance

```python
# Fast: Only reads required dates
df = backend.read(
    datetime(2025, 12, 26),
    datetime(2025, 12, 27)
)  # Only reads 2 days

# Efficient: Columnar access
shap_values = df[['shap_feature_1', 'shap_feature_2']]  # Only reads 2 columns
```

### Compatibility

- Readable by pandas, pyarrow, DuckDB, Spark
- Standard format for data science workflows
- Easy integration with data lakes

## Storage Management

### Monitor Storage Size

```bash
# Check storage usage
du -sh /path/to/shap_logs

# List partitions
ls -lh /path/to/shap_logs/
```

### Retention Policy

Implement a retention policy to manage storage:

```python
from datetime import datetime, timedelta

def cleanup_old_data(backend, retention_days=30):
    """Delete data older than retention_days."""
    cutoff = datetime.now() - timedelta(days=retention_days)
    deleted = backend.delete(cutoff)
    print(f"Deleted {deleted} partitions older than {retention_days} days")

# Run periodically (e.g., daily cron job)
cleanup_old_data(backend, retention_days=90)
```

### Archival Strategy

For long-term storage:

```python
import shutil
from datetime import datetime, timedelta

# Archive data older than 1 year to cold storage
cutoff = datetime.now() - timedelta(days=365)
archive_dir = "/archive/shap_logs"

# Copy then delete
shutil.copytree(
    "/path/to/shap_logs",
    archive_dir,
    ignore=lambda dir, files: [f for f in files if is_newer_than(f, cutoff)]
)
backend.delete(cutoff)
```

## Custom Backends

The backend interface is pluggable. Future versions may support:

- S3/Cloud storage
- Database backends (PostgreSQL, DuckDB)
- Time-series databases

To implement a custom backend, extend `BaseBackend`:

```python
from shapmonitor.backends._base import BaseBackend

class CustomBackend(BaseBackend):
    def read(self, start_dt, end_dt):
        # Implement read logic
        pass

    def write(self, batch):
        # Implement write logic
        pass

    def delete(self, cutoff_dt):
        # Implement delete logic
        pass
```

## Performance Tips

### Batch Size

- Larger batches → fewer files → better read performance
- Smaller batches → more granular timestamps
- Recommended: 100-1000 samples per batch

### Date Partitioning

- Daily partitions work well for most use cases
- Automatic in ParquetBackend
- Enables efficient date range queries

### Compression

Parquet automatically compresses data. For custom compression:

```python
# When writing directly to Parquet
df.to_parquet(
    path,
    compression='snappy',  # Fast compression
    # compression='gzip',   # Better compression ratio
    index=False
)
```

### Concurrent Access

ParquetBackend is thread-safe for:

- Multiple writers (different batch IDs)
- Concurrent reads
- Read while writing

## Troubleshooting

### File Not Found

```python
# Check if directory exists
if backend.file_dir.exists():
    print("Backend directory exists")
else:
    print("Backend directory not found")
```

### Empty Results

```python
# Verify data exists for date range
df = backend.read(start_date, end_date)

if df.empty:
    print("No data in date range")
    # Check what dates have data
    for date_dir in sorted(backend.file_dir.iterdir()):
        if date_dir.is_dir():
            print(f"Data available: {date_dir.name}")
```

### Corrupt Files

```python
# List all files in partition
from pathlib import Path

partition = backend.file_dir / "2025-12-26"
for file in partition.glob("*.parquet"):
    try:
        df = pd.read_parquet(file)
        print(f"✓ {file.name}: {len(df)} rows")
    except Exception as e:
        print(f"✗ {file.name}: {e}")
```
