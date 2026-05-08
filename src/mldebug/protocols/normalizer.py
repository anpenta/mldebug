from typing import Protocol

from numpy.typing import ArrayLike

from mldebug.types import Array


class Normalizer(Protocol):
    def __call__(self, values: ArrayLike) -> Array: ...
