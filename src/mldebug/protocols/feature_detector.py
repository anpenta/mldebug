from typing import Protocol

from numpy.typing import ArrayLike


class FeatureDetector(Protocol):
    def __call__(self, values: ArrayLike) -> bool: ...
