from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray


def normalize_numeric(data: Sequence[Any]) -> NDArray[np.floating]:
    """Normalize numeric input into a float NumPy array.

    Invalid or non-convertible values are coerced to NaN to ensure compatibility with downstream checks.
    """
    return np.asarray(data, dtype=float)


def normalize_categorical(data: Sequence[Any]) -> NDArray[np.str_]:
    """Normalize categorical input into a string NumPy array.

    Missing or invalid values are mapped to empty strings to ensure compatibility with downstream checks.
    """
    arr = np.asarray(data, dtype=str)
    missing_set = np.array(["", "nan", "none"], dtype=object)
    mask = np.isin(np.char.lower(arr), missing_set)
    arr[mask] = ""
    return arr
