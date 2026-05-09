import numpy as np
from numpy.typing import ArrayLike

from .shared import compute_present_mask, is_numeric_vector, normalize_str_array


def compute_numeric_ratio(values: ArrayLike) -> float:
    """Compute the proportion of numeric values.

    Any missing-like values are ignored.
    """
    arr = normalize_str_array(values)

    present = compute_present_mask(arr)

    if not present.any():
        return 0.0

    numeric_mask = is_numeric_vector(arr[present])

    return float(numeric_mask.mean())


def compute_unique_ratio(values: ArrayLike) -> float:
    """Compute the proportion of unique values.

    Any missing-like values are ignored.
    """
    arr = normalize_str_array(values)

    present = compute_present_mask(arr)

    if not present.any():
        return 0.0

    filtered = arr[present]

    return float(len(np.unique(filtered)) / len(filtered))
