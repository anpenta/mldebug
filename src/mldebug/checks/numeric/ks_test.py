from scipy.stats import ks_2samp

from mldebug.core.models.context import NumericFeatureContext
from mldebug.core.models.issue import Issue, Severity


def run_numeric_ks_test_check(context: NumericFeatureContext) -> Issue | None:
    """Detect numeric distribution shift using the Kolmogorov-Smirnov (KS) test.

    This check compares the empirical distributions of reference and current data for a numeric feature using
    the two-sample KS test. An issue is reported when the p-value falls below the configured significance level.

    The significance level (alpha) is obtained from the check configuration.

    Parameters
    ----------
    context : NumericFeatureContext
        Execution context for the feature check.

    Returns
    -------
    Issue | None
        Issue if a statistically significant distribution shift is detected, otherwise None.

    """
    reference = context.reference
    current = context.current
    feature = context.feature
    alpha = context.config.ks_alpha

    p_value = ks_2samp(reference, current).pvalue

    if p_value < alpha:
        return Issue(
            name="ks_test",
            metric="distribution_shift_score",
            severity=Severity.WARNING,
            message=f"{feature}: distribution shift detected (p={p_value:.6f})",
            feature=feature,
            value=float(p_value),
            threshold=alpha,
        )

    return None
