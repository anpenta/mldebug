from dataclasses import dataclass

from mldebug.domain.issue import Issue, Severity
from mldebug.runtime.feature_context import FeatureContext


@dataclass(frozen=True, slots=True)
class CategoricalMissingValueCheck:
    """Detect increase in missing values for a categorical feature.

    This check compares the proportion of missing values between reference and current
    data. An issue is reported when the increase in missing rate exceeds the configured
    threshold.

    Parameters
    ----------
    threshold : float, default=0.1
        Maximum allowed increase in missing value ratio before reporting an issue.

    """

    threshold: float = 0.1

    def __call__(self, context: FeatureContext) -> Issue | None:
        """Run the categorical missing value drift check.

        Parameters
        ----------
        context : FeatureContext
            Execution context for the feature check.

        Returns
        -------
        Issue | None
            Issue if the increase in missing rate exceeds the configured threshold,
            otherwise None.

        """
        reference = context.reference
        current = context.current
        feature = context.feature

        ref_missing = (reference == "").mean()
        cur_missing = (current == "").mean()

        delta = cur_missing - ref_missing

        if delta > self.threshold:
            return Issue(
                name="missing_values",
                metric="missing_rate_increase",
                severity=Severity.WARNING,
                message=(f"{feature}: missing rate drift detected ({delta:.4f})"),
                feature=feature,
                value=float(delta),
                threshold=self.threshold,
            )

        return None
