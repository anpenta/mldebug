from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

_MISSING_VALUES = ("", "nan", "none", "null")


def normalize_str_array(values: ArrayLike) -> NDArray[np.str_]:
    arr = np.asarray(values, dtype=str)
    return np.char.strip(arr)


def compute_present_mask(arr: NDArray[np.str_]) -> NDArray[np.bool_]:
    lower = np.char.lower(arr)
    return ~np.isin(lower, _MISSING_VALUES)


def is_numeric_vector(arr: NDArray[np.str_]) -> NDArray[np.bool_]:
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
