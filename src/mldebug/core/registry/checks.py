from collections.abc import Callable
from dataclasses import dataclass

from mldebug.checks.categorical.missing_values import run_categorical_missing_value_check
from mldebug.checks.categorical.psi import run_categorical_psi_drift_check
from mldebug.checks.categorical.unseen import run_categorical_unseen_category_check
from mldebug.checks.numeric.ks_test import run_numeric_ks_test_check
from mldebug.checks.numeric.missing_values import run_numeric_missing_value_check
from mldebug.checks.numeric.variance_drift import run_numeric_variance_drift_check
from mldebug.core.models.context import CategoricalFeatureContext, NumericFeatureContext
from mldebug.core.models.issue import Issue


@dataclass(frozen=True, slots=True)
class CheckGroup:
    """Feature-type check registry entry.

    Attributes
    ----------
    context : type
        Context class used to build feature data.

    checks : list[Callable[..., Issue | None]]
        Validation checks for this feature type.

    """

    context: type[NumericFeatureContext | CategoricalFeatureContext]
    checks: list[Callable[..., Issue | None]]


CHECKS: dict[str, CheckGroup] = {
    "numeric": CheckGroup(
        context=NumericFeatureContext,
        checks=[run_numeric_missing_value_check, run_numeric_ks_test_check, run_numeric_variance_drift_check],
    ),
    "categorical": CheckGroup(
        context=CategoricalFeatureContext,
        checks=[
            run_categorical_missing_value_check,
            run_categorical_psi_drift_check,
            run_categorical_unseen_category_check,
        ],
    ),
}


def list_checks() -> dict[str, list[str]]:
    """List all available checks grouped by feature type.

    Returns
    -------
    dict[str, list[str]]
        A dictionary mapping each feature type (e.g. "numeric", "categorical")
        to a list of available check function names for that type.

    """
    return {k: [fn.__name__ for fn in v.checks] for k, v in CHECKS.items()}
