import numpy as np
from numpy.typing import ArrayLike

from .shared import compute_valid_mask, is_numeric_vector, normalize_str_array


def compute_numeric_ratio(values: ArrayLike) -> float:
    """Compute the proportion of numeric values.

    Empty and missing-like values are ignored.
    """
    arr = normalize_str_array(values)

    valid = compute_valid_mask(arr)

    if not valid.any():
        return 0.0

    numeric_mask = is_numeric_vector(arr[valid])

    return float(numeric_mask.mean())


def compute_unique_ratio(values: ArrayLike) -> float:
    """Compute the proportion of unique values.

    Empty and missing-like values are ignored.
    """
    arr = normalize_str_array(values)

    valid = compute_valid_mask(arr)

    if not valid.any():
        return 0.0

    filtered = arr[valid]

    return float(len(np.unique(filtered)) / len(filtered))
