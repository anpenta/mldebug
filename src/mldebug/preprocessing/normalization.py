from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

_MISSING_STRINGS = {"", "nan", "NaN", "none", "None"}


def normalize_data(feature_type: Literal["numeric", "categorical"], data: Sequence[Any]) -> NDArray[Any]:
    """Normalize raw feature data into a NumPy array suitable for downstream checks.

    This function routes input data to the appropriate normalization strategy based on the declared feature
    type. It ensures consistent representation of missing values and type safety across the pipeline.

    Parameters
    ----------
    feature_type : Literal["numeric", "categorical"]
        Declares how the input data should be interpreted and normalized.

    data : Sequence[Any]
        Raw feature values.

    Returns
    -------
    NDArray[Any]
        Normalized NumPy array:
        - numeric: float array (invalid values coerced to NaN)
        - categorical: string array (missing values mapped to "")

    Raises
    ------
    ValueError
        If feature_type is not supported.

    """
    if feature_type == "numeric":
        return _normalize_numeric(data)

    if feature_type == "categorical":
        return _normalize_categorical(data)

    error_msg = f"Unsupported feature type: {feature_type}"
    raise ValueError(error_msg)


def _normalize_numeric(data: Sequence[Any]) -> NDArray[np.floating]:
    """Convert numeric input into a float array with NaNs for invalid values."""
    return np.asarray(data, dtype=float)


def _normalize_categorical(data: Sequence[Any]) -> NDArray[np.str_]:
    """Normalize categorical input into string array with empty string as missing."""
    arr = np.asarray(data, dtype=object)

    # Convert everything to string, map missing  to "".
    mask = np.array(
        [
            v is None
            or (isinstance(v, float) and np.isnan(v))
            or (isinstance(v, str) and v.strip() in _MISSING_STRINGS)
            for v in arr
        ],
        dtype=bool,
    )
    arr_str = arr.astype(str)
    arr_str[mask] = ""

    return arr_str
