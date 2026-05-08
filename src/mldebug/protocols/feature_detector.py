from collections.abc import Sequence
from typing import Any, Protocol


class FeatureDetector(Protocol):
    def __call__(self, values: Sequence[Any]) -> bool: ...
