from dataclasses import dataclass

from mldebug.domain.issue import Issue, Severity
from mldebug.runtime.feature_context import FeatureContext


@dataclass(frozen=True, slots=True)
class NumericVarianceDriftCheck:
    """Detect variance drift for a numeric feature.

    This check compares the variance of the reference and current data and flags an
    issue when the relative change exceeds a configured threshold. The ratio is computed
    as current variance divided by reference variance.

    Parameters
    ----------
    threshold : float, default=2.0
        Allowed multiplicative deviation in variance between current and reference
        distributions.

    """

    threshold: float = 2.0

    def __call__(self, context: FeatureContext) -> Issue | None:
        """Run variance drift detection for numeric features.

        Parameters
        ----------
        context : FeatureContext
            Execution context for the feature check.

        Returns
        -------
        Issue | None
            Issue if variance drift exceeds threshold bounds, otherwise None.

        """
        ref = context.reference
        cur = context.current
        feature = context.feature

        ref_var = ref.var()
        cur_var = cur.var()

        # Edge case: no variance in reference.
        if ref_var == 0:
            return None

        ratio = cur_var / ref_var

        if ratio > self.threshold or ratio < 1 / self.threshold:
            return Issue(
                name="variance_drift",
                metric="variance_ratio",
                severity=Severity.WARNING,
                message=(
                    f"{feature}: variance drift detected (ratio={ratio:.4f}, "
                    f"threshold={self.threshold})"
                ),
                feature=feature,
                value=float(ratio),
                threshold=self.threshold,
            )

        return None
