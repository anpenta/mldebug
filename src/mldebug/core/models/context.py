from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from mldebug.core.config import CategoricalCheckConfig, NumericCheckConfig
from mldebug.preprocessing.normalization import normalize_data

if TYPE_CHECKING:
    from collections.abc import Sequence

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

    @classmethod
    def from_raw(cls, feature: str, ref: Sequence[Any], cur: Sequence[Any]) -> NumericFeatureContext:
        """Create a numeric context from raw feature data.

        Applies numeric normalization and attaches default config.
        """
        return cls(
            feature=feature,
            reference=normalize_data(feature_type="numeric", data=ref),
            current=normalize_data(feature_type="numeric", data=cur),
            config=NumericCheckConfig(),
        )


@dataclass(frozen=True, slots=True)
class CategoricalFeatureContext:
    """Execution context for a single categorical feature check.

    Encapsulates all data and configuration required to run feature-level validation checks.
    """

    feature: str
    reference: NDArray[np.str_]
    current: NDArray[np.str_]
    config: CategoricalCheckConfig

    @classmethod
    def from_raw(cls, feature: str, ref: Sequence[Any], cur: Sequence[Any]) -> CategoricalFeatureContext:
        """Create a categorical context from raw feature data.

        Applies categorical normalization and attaches default config.
        """
        return cls(
            feature=feature,
            reference=normalize_data(feature_type="categorical", data=ref),
            current=normalize_data(feature_type="categorical", data=cur),
            config=CategoricalCheckConfig(),
        )
