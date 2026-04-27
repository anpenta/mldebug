import numpy as np
from numpy.typing import NDArray


def compute_safe_histogram(
    values: NDArray[np.floating],
    bins: int = 10,
) -> tuple[NDArray[np.integer], NDArray[np.floating]]:
    """Compute a safe histogram for numerical stability in ML monitoring.

    Handles edge cases such as:
    - empty arrays
    - constant arrays (zero variance)

    Parameters
    ----------
    values : NDArray[np.floating]
        Input numeric array.
    bins : int = 10, optional
        Number of histogram bins.

    Returns
    -------
    tuple[NDArray[np.integer], NDArray[np.floating]]:
        Bin counts.
        Bin edges used for histogram.

    """
    values = np.asarray(values)

    # Empty input.
    if values.size == 0:
        edges = np.linspace(0.0, 1.0, bins + 1)
        return np.zeros(bins, dtype=int), edges

    # Constant input (zero variance).
    if np.all(values == values[0]):
        center = float(values[0])
        edges = np.linspace(center - 1.0, center + 1.0, bins + 1)
        hist = np.zeros(bins, dtype=int)
        idx = np.digitize(values[0], edges) - 1
        idx = np.clip(idx, 0, bins - 1)
        hist[idx] = values.size
        return hist, edges

    # Standard case.
    hist, edges = np.histogram(values, bins=bins, density=False)
    return hist, edges


def normalize_distribution(
    hist: NDArray[np.integer],
) -> NDArray[np.floating]:
    """Convert histogram counts into a probability distribution.

    Parameters
    ----------
    hist :  NDArray[np.integer]
        Histogram counts.

    Returns
    -------
    NDArray[np.floating]
        Normalized probability distribution.

    """
    hist = np.asarray(hist, dtype=float)

    total = hist.sum()

    if total <= 0:
        return np.zeros_like(hist, dtype=float)

    return hist / total
