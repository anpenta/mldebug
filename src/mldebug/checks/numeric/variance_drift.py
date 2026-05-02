import numpy as np

from mldebug.core.models.context import NumericFeatureContext
from mldebug.core.models.issue import Issue, Severity


def run_numeric_variance_drift_check(
    context: NumericFeatureContext,
) -> Issue | None:
    """Detect variance drift for a numeric feature.

    This check compares the variance of the reference and current data and flags an issue when the relative change
    exceeds a configured threshold. The ratio is computed as current variance divided by reference variance.

    Parameters
    ----------
    context : NumericFeatureContext
        Execution context for the feature check.

    Returns
    -------
    Issue | None
        Issue if variance drift exceeds the configured threshold, otherwise None.

    """
    ref = context.reference
    cur = context.current
    threshold = context.config.variance_drift_threshold

    ref_var = np.var(ref)
    cur_var = np.var(cur)

    # Edge case: no variance in reference.
    if ref_var == 0:
        return None

    ratio = cur_var / ref_var

    if ratio > threshold or ratio < 1 / threshold:
        return Issue(
            name="variance_drift",
            metric="variance_ratio",
            severity=Severity.WARNING,
            message=(f"{context.feature}: variance drift detected (ratio={ratio:.4f}, threshold={threshold})"),
            feature=context.feature,
            value=float(ratio),
            threshold=threshold,
        )

    return None
