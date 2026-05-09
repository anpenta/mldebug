from dataclasses import dataclass

from numpy.typing import ArrayLike

from mldebug.preprocessing.normalization import compute_numeric_ratio


@dataclass(frozen=True, slots=True)
class CategoricalFeatureDetector:
    threshold: float = 0.9

    def __call__(self, values: ArrayLike) -> bool:
        return compute_numeric_ratio(values) <= self.threshold
