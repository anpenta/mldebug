from mldebug.checks.categorical.missing_values import run_categorical_missing_value_check
from mldebug.checks.categorical.psi import run_categorical_psi_drift_check
from mldebug.checks.categorical.unseen import run_categorical_unseen_category_check
from mldebug.checks.numeric.ks_test import run_numeric_ks_test_check
from mldebug.checks.numeric.missing_values import run_numeric_missing_value_check

CHECKS = {
    "numeric": [
        run_numeric_missing_value_check,
        run_numeric_ks_test_check,
    ],
    "categorical": [
        run_categorical_missing_value_check,
        run_categorical_psi_drift_check,
        run_categorical_unseen_category_check,
    ],
}


def list_checks() -> dict[str, list[str]]:
    """List all available checks grouped by feature type.

    Returns
    -------
    dict[str, list[str]]
        A dictionary mapping each feature type (e.g. "numeric", "categorical")
        to a list of available check function names for that type.

    """
    return {group: [fn.__name__ for fn in checks] for group, checks in CHECKS.items()}
