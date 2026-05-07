import numpy as np

from mldebug.config import CategoricalCheckConfig
from mldebug.models.feature_context import FeatureContext
from mldebug.models.issue import Issue, Severity


def run_categorical_missing_value_check(context: FeatureContext[CategoricalCheckConfig, np.str_]) -> Issue | None:
    """Detect increase in missing values for a categorical feature.

    This check compares the proportion of missing values between reference and current data.
    An issue is reported when the increase in missing rate exceeds the configured threshold.

    Parameters
    ----------
    context : FeatureContext[CategoricalCheckConfig, np.str_]
        Execution context for the feature check.

    Returns
    -------
    Issue | None
        Issue if the increase in missing rate exceeds the configured threshold, otherwise None.

    """
    reference = context.reference
    current = context.current
    feature = context.feature
    threshold = context.config.missing_threshold

    ref_missing = (reference == "").mean()
    cur_missing = (current == "").mean()

    delta = cur_missing - ref_missing

    if delta > threshold:
        return Issue(
            name="missing_values",
            metric="missing_rate_increase",
            severity=Severity.WARNING,
            message=f"{feature}: missing rate drift detected ({delta:.4f})",
            feature=feature,
            value=float(delta),
            threshold=threshold,
        )

    return None
