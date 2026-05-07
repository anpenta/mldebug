from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

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
from mldebug.models.context import FeatureContext
from mldebug.models.issue import Issue
from mldebug.models.types import FeatureType
from mldebug.preprocessing.normalization import normalize_categorical, normalize_numeric


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    normalizer: Callable[[Sequence[Any]], NDArray[np.generic]]
    checks: list[Callable[[FeatureContext], Issue | None]]
    config: NumericCheckConfig | CategoricalCheckConfig


FEATURE_SPECS: dict[FeatureType, FeatureSpec] = {
    "numeric": FeatureSpec(
        normalizer=normalize_numeric,
        checks=[
            run_numeric_missing_value_check,
            run_numeric_ks_test_check,
            run_numeric_variance_drift_check,
            run_numeric_range_anomaly_check,
        ],
        config=NumericCheckConfig(),
    ),
    "categorical": FeatureSpec(
        normalizer=normalize_categorical,
        checks=[
            run_categorical_missing_value_check,
            run_categorical_psi_drift_check,
            run_categorical_unseen_category_check,
        ],
        config=CategoricalCheckConfig(),
    ),
}
