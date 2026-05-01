from mldebug.checks.categorical.missing_values import run_categorical_missing_value_check
from mldebug.checks.categorical.psi import run_categorical_psi_drift_check
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
    ],
}
