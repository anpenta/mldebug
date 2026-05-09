
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .shared import compute_valid_mask, is_numeric_vector, normalize_str_array

_MISSING_VALUES = ("", "nan", "none", "null")


def normalize_numeric(values: ArrayLike) -> NDArray[np.floating]:
    """Normalize values into a numeric NumPy array.

    Non-numeric values are converted to NaN.
    """
    arr = normalize_str_array(values)

    out = np.full(arr.shape, np.nan, dtype=float)

    valid = is_numeric_vector(arr)

    if valid.any():
        out[valid] = arr[valid].astype(float)

    return out


def normalize_categorical(values: ArrayLike) -> NDArray[np.str_]:
    """Normalize values into a categorical NumPy array.

    Numeric and missing-like values are converted to empty strings.
    """
    arr = normalize_str_array(values)

    valid = compute_valid_mask(arr)

    arr[~valid] = ""
    return arr
