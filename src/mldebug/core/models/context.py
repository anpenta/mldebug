from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from mldebug.core.config import CheckConfig


@dataclass(frozen=True, slots=True)
class FeatureContext:
    """Execution context for a single feature check.

    Encapsulates all data and configuration required to run feature-level validation checks.
    """

    feature: str
    ftype: Literal["numeric", "categorical"]
    reference: NDArray[np.generic]
    current: NDArray[np.generic]
    config: CheckConfig
