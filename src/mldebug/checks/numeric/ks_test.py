from dataclasses import dataclass

from scipy.stats import ks_2samp

from mldebug.domain.issue import Issue, Severity
from mldebug.runtime.feature_context import FeatureContext


@dataclass(frozen=True, slots=True)
class NumericKSTestCheck:
    """Detect numeric distribution shift using the Kolmogorov-Smirnov (KS) test.

    This check compares the empirical distributions of reference and
    current data using the two-sample KS test. An issue is reported
    when the p-value falls below the configured significance level.

    Parameters
    ----------
    alpha : float, default=0.05
        Significance level for detecting distribution shift.

    """

    alpha: float = 0.05

    def __call__(self, context: FeatureContext) -> Issue | None:
        """Run KS test for numeric distribution shift detection.

        Parameters
        ----------
        context : FeatureContext
            Execution context for the feature check.

        Returns
        -------
        Issue | None
            Issue if a statistically significant distribution shift
            is detected, otherwise None.

        """
        reference = context.reference
        current = context.current
        feature = context.feature

        p_value = ks_2samp(reference, current).pvalue

        if p_value < self.alpha:
            return Issue(
                name="ks_test",
                metric="distribution_shift_score",
                severity=Severity.WARNING,
                message=f"{feature}: distribution shift detected (p={p_value:.6f})",
                feature=feature,
                value=float(p_value),
                threshold=self.alpha,
            )

        return None
