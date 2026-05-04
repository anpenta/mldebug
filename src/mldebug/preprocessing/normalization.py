from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

_MISSING_VALUES = ("", "nan", "none", "null")


def normalize_numeric(data: Sequence[Any]) -> NDArray[np.floating]:
    """Normalize a sequence into a numeric NumPy array.

    Non-numeric values are converted to NaN.
    """
    arr = np.asarray(data, dtype=str)
    arr = np.char.strip(arr)

    out = np.full(arr.shape, np.nan, dtype=float)

    valid = _is_numeric_vector(arr)

    if valid.any():
        out[valid] = arr[valid].astype(float)

    return out


def normalize_categorical(data: Sequence[Any]) -> NDArray[np.str_]:
    """Normalize a sequence into a categorical NumPy array.

    Numeric and missing-like values are converted to empty strings.
    """
    arr = np.asarray(data, dtype=str)
    arr = np.char.strip(arr)

    lower = np.char.lower(arr)

    missing = (arr == "") | np.isin(lower, _MISSING_VALUES)

    arr[missing] = ""

    return arr


def compute_numeric_ratio(values: Sequence[Any]) -> float:
    """Compute the proportion of numeric values in a sequence.

    Empty and missing-like values are ignored.
    """
    arr = np.asarray(values, dtype=str)
    arr = np.char.strip(arr)

    valid = ~np.isin(np.char.lower(arr), _MISSING_VALUES)

    if not valid.any():
        return 0.0

    candidates = arr[valid]

    numeric_mask = _is_numeric_vector(candidates)

    return float(numeric_mask.mean())


def _is_numeric_vector(arr: NDArray[np.str_]) -> NDArray[np.bool_]:
    try:
        arr.astype(float)
        return np.ones(arr.shape, dtype=bool)
    except ValueError:
        return np.fromiter(
            (_is_floatable_scalar(x) for x in arr),
            dtype=bool,
            count=len(arr),
        )


def _is_floatable_scalar(x: Any) -> bool:
    try:
        float(x)
    except (TypeError, ValueError):
        return False
    else:
        return True
