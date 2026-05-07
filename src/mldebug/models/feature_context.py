from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np
from numpy.typing import NDArray

C = TypeVar("C")
T = TypeVar("T", bound=np.generic)


@dataclass(frozen=True, slots=True)
class FeatureContext(Generic[C, T]):
    feature: str
    reference: NDArray[T]
    current: NDArray[T]
    config: C
