from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from mldebug.preprocessing.normalization import normalize_categorical, normalize_numeric


def normalize_feature(
    reference: Sequence[Any], current: Sequence[Any], ftype: Literal["numeric", "categorical"]
) -> tuple[NDArray[np.generic], NDArray[np.generic]]:
    match ftype:
        case "numeric":
            return normalize_numeric(reference), normalize_numeric(current)
        case "categorical":
            return normalize_categorical(reference), normalize_categorical(current)
        case _:
            raise NotImplementedError()
