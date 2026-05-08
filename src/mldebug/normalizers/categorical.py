from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from mldebug.preprocessing.normalization import normalize_categorical
from mldebug.types import Array


@dataclass(frozen=True, slots=True)
class CategoricalNormalizer:
    def __call__(self, values: Sequence[Any]) -> Array:
        return normalize_categorical(values)
