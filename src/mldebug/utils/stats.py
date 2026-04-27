import numpy as np


def safe_histogram(values, bins=10):
    """Returns histogram counts + bin edges safely.
    Handles empty / constant arrays.
    """
    values = np.asarray(values)

    if len(values) == 0:
        return np.zeros(bins), np.linspace(0, 1, bins + 1)

    if np.all(values == values[0]):
        # constant distribution edge case
        hist = np.zeros(bins)
        hist[0] = len(values)
        return hist, np.linspace(values[0] - 1, values[0] + 1, bins + 1)

    hist, edges = np.histogram(values, bins=bins, density=False)
    return hist, edges


def normalize_distribution(hist):
    """Convert counts to probabilities."""
    total = np.sum(hist)
    if total == 0:
        return np.zeros_like(hist, dtype=float)
    return hist / total
