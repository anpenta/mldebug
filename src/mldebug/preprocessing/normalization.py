from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

_MISSING_VALUES = ("", "nan", "none", "null")


def compute_numeric_score(values: ArrayLike) -> float:
    """Compute confidence score for numeric feature likelihood in [0, 1]."""
    numeric_ratio = compute_numeric_ratio(values)
    unique_ratio = compute_unique_ratio(values)

    structure_penalty = 1.0 - unique_ratio
    return numeric_ratio * (1.0 - 0.3 * structure_penalty)


def compute_categorical_score(values: ArrayLike) -> float:
    """Compute confidence score for categorical feature likelihood in [0, 1]."""
    numeric_ratio = compute_numeric_ratio(values)
    unique_ratio = compute_unique_ratio(values)

    structure_bonus = 1.0 - unique_ratio
    return structure_bonus * (1.0 - 0.3 * numeric_ratio)


def normalize_numeric(values: ArrayLike) -> NDArray[np.floating]:
    """Normalize values into a numeric NumPy array.

    Non-numeric values are converted to NaN.
    """
    arr = np.asarray(values, dtype=str)
    arr = np.char.strip(arr)

    out = np.full(arr.shape, np.nan, dtype=float)

    valid = _is_numeric_vector(arr)

    if valid.any():
        out[valid] = arr[valid].astype(float)

    return out


def normalize_categorical(values: ArrayLike) -> NDArray[np.str_]:
    """Normalize values into a categorical NumPy array.

    Numeric and missing-like values are converted to empty strings.
    """
    arr = np.asarray(values, dtype=str)
    arr = np.char.strip(arr)

    lower = np.char.lower(arr)
    missing = np.isin(lower, _MISSING_VALUES)

    arr[missing] = ""
    return arr


def compute_numeric_ratio(values: ArrayLike) -> float:
    """Compute the proportion of numeric values.

    Empty and missing-like values are ignored.
    """
    arr = np.asarray(values, dtype=str)
    arr = np.char.strip(arr)

    lower = np.char.lower(arr)
    valid = ~np.isin(lower, _MISSING_VALUES)

    if not valid.any():
        return 0.0

    numeric_mask = _is_numeric_vector(arr[valid])

    return float(numeric_mask.mean())


def compute_unique_ratio(values: ArrayLike) -> float:
    """Compute the proportion of unique values.

    Empty and missing-like values are ignored.
    """
    arr = np.asarray(values, dtype=str)
    arr = np.char.strip(arr)

    lower = np.char.lower(arr)
    valid = ~np.isin(lower, _MISSING_VALUES)

    if not valid.any():
        return 0.0

    filtered = arr[valid]

    return float(len(np.unique(filtered)) / len(filtered))


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


def _is_floatable_scalar(x: Any) -> bool:  # noqa: ANN401 # Need to keep this broad to catch everything.
    try:
        float(x)
    except (TypeError, ValueError):
        return False
    else:
        return True
