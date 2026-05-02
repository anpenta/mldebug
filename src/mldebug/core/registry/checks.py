from mldebug.checks.categorical.missing_values import run_categorical_missing_value_check
from mldebug.checks.categorical.psi import run_categorical_psi_drift_check
from mldebug.checks.categorical.unseen import run_categorical_unseen_category_check
from mldebug.checks.numeric.ks_test import run_numeric_ks_test_check
from mldebug.checks.numeric.missing_values import run_numeric_missing_value_check
from mldebug.core.config import CategoricalCheckConfig, NumericCheckConfig
from mldebug.core.models.context import CategoricalFeatureContext, NumericFeatureContext
from mldebug.preprocessing.normalization import normalize_data

CHECKS = {
    "numeric": {
        "context": NumericFeatureContext,
        "checks": [
            run_numeric_missing_value_check,
            run_numeric_ks_test_check,
        ],
        "builder": lambda feature, ref, cur: NumericFeatureContext(
            feature=feature,
            reference=normalize_data("numeric", ref),
            current=normalize_data("numeric", cur),
            config=NumericCheckConfig(),
        ),
    },
    "categorical": {
        "context": CategoricalFeatureContext,
        "checks": [
            run_categorical_missing_value_check,
            run_categorical_psi_drift_check,
            run_categorical_unseen_category_check,
        ],
        "builder": lambda feature, ref, cur: CategoricalFeatureContext(
            feature=feature,
            reference=normalize_data("categorical", ref),
            current=normalize_data("categorical", cur),
            config=CategoricalCheckConfig(),
        ),
    },
}


def list_checks() -> dict[str, list[str]]:
    """List all available checks grouped by feature type.

    Returns
    -------
    dict[str, list[str]]
        A dictionary mapping each feature type (e.g. "numeric", "categorical")
        to a list of available check function names for that type.

    """
    return {feature_type: [check_fn.__name__ for check_fn in entry["checks"]] for feature_type, entry in CHECKS.items()}
