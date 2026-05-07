from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import numpy as np
from numpy.typing import NDArray

from mldebug.checks.categorical.missing_values import run_categorical_missing_value_check
from mldebug.checks.categorical.psi import run_categorical_psi_drift_check
from mldebug.checks.categorical.unseen_values import run_categorical_unseen_category_check
from mldebug.checks.numeric.ks_test import run_numeric_ks_test_check
from mldebug.checks.numeric.missing_values import run_numeric_missing_value_check
from mldebug.checks.numeric.range_anomaly import run_numeric_range_anomaly_check
from mldebug.checks.numeric.variance_drift import run_numeric_variance_drift_check
from mldebug.config import CategoricalCheckConfig, NumericCheckConfig
from mldebug.models.feature_context import FeatureContext
from mldebug.models.feature_type import FeatureType
from mldebug.models.issue import Issue
from mldebug.preprocessing.normalization import compute_numeric_ratio, normalize_categorical, normalize_numeric

C = TypeVar("C")
T = TypeVar("T", bound=np.generic)


@dataclass(frozen=True, slots=True)
class FeatureSpec(Generic[C, T]):
    type_checker: Callable[[Sequence[Any]], bool]
    normalizer: Callable[[Sequence[Any]], NDArray[T]]
    checks: list[Callable[[FeatureContext[C, T]], Issue | None]]
    config: C


FEATURE_SPECS: dict[FeatureType, FeatureSpec[Any, Any]] = {
    FeatureType.NUMERIC: FeatureSpec[NumericCheckConfig, np.floating](
        type_checker=lambda values: compute_numeric_ratio(values) < 0.9,
        normalizer=normalize_numeric,
        checks=[
            run_numeric_missing_value_check,
            run_numeric_ks_test_check,
            run_numeric_variance_drift_check,
            run_numeric_range_anomaly_check,
        ],
        config=NumericCheckConfig(),
    ),
    FeatureType.CATEGORICAL: FeatureSpec[CategoricalCheckConfig, np.str_](
        type_checker=lambda values: compute_numeric_ratio(values) > 0.9,
        normalizer=normalize_categorical,
        checks=[
            run_categorical_missing_value_check,
            run_categorical_psi_drift_check,
            run_categorical_unseen_category_check,
        ],
        config=CategoricalCheckConfig(),
    ),
}
