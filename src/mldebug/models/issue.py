from dataclasses import dataclass
from enum import Enum


class Severity(Enum):
    """Severity level of a detected issue.

    INFO:
        Informational issue with no immediate impact.

    WARNING:
        Potential problem that should be reviewed.

    CRITICAL:
        Serious issue likely to affect model performance or reliability.

    """

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True, slots=True)
class Issue:
    """Represents a detected issue from a validation or monitoring check.

    This is the atomic output of all checks and is intended to be consumed by downstream reporting,
    alerting, or debugging components.

    Parameters
    ----------
    name : str
        Identifier of the issue type (e.g., "ks_test", "missing_values").

    metric : str
        Name of the metric used to detect the issue (e.g., "distribution_shift_score", "missing_rate_increase").

    severity : Severity
        Importance level of the issue.

    message : str
        Human-readable explanation of the issue.

    feature : str | None
        Feature associated with the issue. None for global issues.

    value : float | None
        Observed metric value that triggered the issue (if applicable).

    threshold : float | None
        Threshold used for comparison. Interpretation depends on the metric.

    """

    name: str
    metric: str
    severity: Severity

    message: str

    feature: str | None = None
    value: float | None = None
    threshold: float | None = None

    def __str__(self) -> str:
        return f"[{self.severity.value.upper()}] {self.name} - {self.message}"

    def __repr__(self) -> str:
        return (
            f"Issue("
            f"name={self.name!r}, "
            f"metric={self.metric!r}, "
            f"severity={self.severity.value!r}, "
            f"message={self.message!r}, "
            f"feature={self.feature!r}, "
            f"value={self.value}, "
            f"threshold={self.threshold}"
            f")"
        )
