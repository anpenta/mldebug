from collections.abc import Callable
from dataclasses import dataclass

from mldebug.checks.categorical.missing_values import run_categorical_missing_value_check
from mldebug.checks.categorical.psi import run_categorical_psi_drift_check
from mldebug.checks.categorical.unseen_values import run_categorical_unseen_category_check
from mldebug.checks.numeric.ks_test import run_numeric_ks_test_check
from mldebug.checks.numeric.missing_values import run_numeric_missing_value_check
from mldebug.checks.numeric.range_anomaly import run_numeric_range_anomaly_check
from mldebug.checks.numeric.variance_drift import run_numeric_variance_drift_check
from mldebug.models.issue import Issue


@dataclass(frozen=True, slots=True)
class CheckGroup:
    """Feature-type check registry entry.

    Attributes
    ----------
    checks : list[Callable[..., Issue | None]]
        Validation checks for this feature type.

    """

    checks: list[Callable[..., Issue | None]]


CHECKS: dict[str, CheckGroup] = {
    "numeric": CheckGroup(
        checks=[
            run_numeric_missing_value_check,
            run_numeric_ks_test_check,
            run_numeric_variance_drift_check,
            run_numeric_range_anomaly_check,
        ],
    ),
    "categorical": CheckGroup(
        checks=[
            run_categorical_missing_value_check,
            run_categorical_psi_drift_check,
            run_categorical_unseen_category_check,
        ],
    ),
}
