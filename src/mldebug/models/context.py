from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class FeatureContext:
    feature: str
    reference: NDArray[np.generic]
    current: NDArray[np.generic]
    config: Any
