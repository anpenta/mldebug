from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from mldebug.preprocessing.normalization import (
    compute_numeric_ratio,
)


@dataclass(frozen=True, slots=True)
class NumericFeatureDetector:
    threshold: float = 0.9

    def __call__(self, values: Sequence[Any]) -> bool:
        return compute_numeric_ratio(values) >= self.threshold
