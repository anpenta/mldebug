from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from mldebug.config import CategoricalCheckConfig, NumericCheckConfig

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class NumericFeatureContext:
    """Execution context for a single numeric feature check.

    Encapsulates all data and configuration required to run feature-level validation checks.
    """

    feature: str
    reference: NDArray[np.floating]
    current: NDArray[np.floating]
    config: NumericCheckConfig


@dataclass(frozen=True, slots=True)
class CategoricalFeatureContext:
    """Execution context for a single categorical feature check.

    Encapsulates all data and configuration required to run feature-level validation checks.
    """

    feature: str
    reference: NDArray[np.str_]
    current: NDArray[np.str_]
    config: CategoricalCheckConfig
