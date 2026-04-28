from dataclasses import dataclass
from typing import Any

from .issue import Issue, Severity


@dataclass(frozen=True, slots=True)
class Report:
    """Aggregated output of a full ML debugging run.

    Parameters
    ----------
    issues : list of Issue
        Collection of detected issues.

    """

    issues: list[Issue]

    def summary(self) -> dict[str, int]:
        """Count issues grouped by severity.

        Returns
        -------
        dict[str, int]
            Mapping from severity level ("info", "warning", "critical") to the number of issues.

        """
        counts = {
            Severity.INFO.value: 0,
            Severity.WARNING.value: 0,
            Severity.CRITICAL.value: 0,
        }

        for issue in self.issues:
            counts[issue.severity.value] += 1

        return counts

    def to_dict(self) -> dict[str, Any]:
        """Serialize report for logging / APIs."""
        return {
            "issues": [
                {
                    "name": i.name,
                    "metric": i.metric,
                    "severity": i.severity.value,
                    "message": i.message,
                    "feature": i.feature,
                    "value": i.value,
                    "threshold": i.threshold,
                }
                for i in self.issues
            ],
        }
