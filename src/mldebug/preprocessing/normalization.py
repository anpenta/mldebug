from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray


def normalize_numeric(data: Sequence[Any]) -> NDArray[np.floating]:
    """Normalize numeric input into a float NumPy array.

    Values that cannot be parsed as floats are represented as NaN in the output array
    to ensure compatibility with downstream checks.
    """
    arr = np.asarray(data, dtype=object)

    out = np.full(arr.shape, np.nan, dtype=float)

    mask = np.fromiter(
        (_is_floatable_scalar(x) for x in arr),
        dtype=bool,
        count=len(arr),
    )

    if mask.any():
        out[mask] = np.asarray(arr[mask], dtype=float)

    return out


def _is_floatable_scalar(x: Any) -> bool:
    try:
        float(x)
    except (TypeError, ValueError):
        return False
    else:
        return True


def normalize_categorical(data: Sequence[Any]) -> NDArray[np.str_]:
    """Normalize categorical input into a string NumPy array.

    Missing or invalid values are mapped to empty strings to ensure compatibility with downstream checks.
    """
    arr = np.asarray(data, dtype=str)
    arr = np.char.strip(arr)
    missing_set = np.array(["", "nan", "none"], dtype=object)
    mask = np.isin(np.char.lower(arr), missing_set)
    arr[mask] = ""
    return arr
