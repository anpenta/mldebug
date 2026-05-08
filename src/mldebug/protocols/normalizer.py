from collections.abc import Sequence
from typing import Any, Protocol

from mldebug.types import Array


class Normalizer(Protocol):
    def __call__(self, values: Sequence[Any]) -> Array: ...
