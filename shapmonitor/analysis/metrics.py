import numpy as np

from shapmonitor.exceptions import InvalidShapeError
from shapmonitor.types import ArrayLike, SeriesLike


# TODO: Add support for calculating for multiple features in parallel.
def population_stability_index(
    reference: ArrayLike | SeriesLike, current: ArrayLike | SeriesLike, buckets: int = 10
) -> float:
    """
    Calculate the Population Stability Index (PSI) between two univariate distributions.

    Parameters
    ----------
    reference : ArrayLike
        The reference distribution (e.g., historical data).
    current : ArrayLike
        The current distribution (e.g., new data).
    buckets : int
        The number of buckets to divide the data into (default is 10).

    Returns
    -------
    float
        The Population Stability Index value.
    """
    if buckets < 2:
        raise ValueError("Number of buckets must be at least 2.")

    if len(reference) == 0 or len(current) == 0:
        raise ValueError("Input distributions must not be empty.")

    if reference.ndim >= 2 or current.ndim >= 2:
        raise InvalidShapeError("Input distributions must be univariate (1D arrays).")

    reference = np.asarray(reference, dtype=np.float32)
    current = np.asarray(current, dtype=np.float32)

    percentiles = np.linspace(0, 100, buckets + 1)
    bin_edges = np.percentile(reference, percentiles)

    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    curr_counts, _ = np.histogram(current, bins=bin_edges)

    ref_props = ref_counts / len(reference)
    curr_props = curr_counts / len(current)

    # Avoid division by zero and log of zero by replacing zeros with a small value
    epsilon = 1e-10
    ref_props = np.where(ref_props == 0, epsilon, ref_props)
    curr_props = np.where(curr_props == 0, epsilon, curr_props)

    psi_values = (curr_props - ref_props) * np.log(curr_props / ref_props)
    return np.sum(psi_values)
