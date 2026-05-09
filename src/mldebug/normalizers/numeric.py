from dataclasses import dataclass

from numpy.typing import ArrayLike

from mldebug.preprocessing.normalization import normalize_numeric
from mldebug.types import Array


@dataclass(frozen=True, slots=True)
class NumericNormalizer:
    def __call__(self, values: ArrayLike) -> Array:
        return normalize_numeric(values)
