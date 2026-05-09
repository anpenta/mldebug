from dataclasses import dataclass

from numpy.typing import ArrayLike

from mldebug.preprocessing.normalization import compute_categorical_score


@dataclass(frozen=True, slots=True)
class CategoricalFeatureDetector:
    threshold: float = 0.8

    def __call__(self, values: ArrayLike) -> bool:
        score = compute_categorical_score(values)
        return score >= self.threshold
