from scipy.stats import ks_2samp

from mldebug.checks.base import BaseCheck
from mldebug.core.issue import Issue, Severity


class KSTestCheck(BaseCheck):
    """Kolmogorov-Smirnov test for distribution shift."""

    name = "ks_test"

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def run(self, reference, current):
        stat, p_value = ks_2samp(reference, current)

        if p_value < self.alpha:
            return Issue(
                name=self.name,
                metric="ks_pvalue",
                severity=Severity.WARNING,
                message=f"KS test rejected null hypothesis (p={p_value:.6f})",
                value=p_value,
                threshold=self.alpha,
            )

        return None
