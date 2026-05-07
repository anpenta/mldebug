from collections.abc import Callable
from enum import Enum

from .context import FeatureContext
from .issue import Issue


class FeatureType(Enum):
    """Supported feature types in mldebug.

    Defines the canonical feature categories used across schema validation, normalization, and feature-level checks.

    NUMERIC:
        Numeric features validated using numeric-based validation checks.

    CATEGORICAL:
        Categorical features validated using category-based validation checks.

    """

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"


CheckType = Callable[[FeatureContext], Issue | None]
