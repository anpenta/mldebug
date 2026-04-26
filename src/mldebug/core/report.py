"""Core reporting object for ML debugging results.

The Report class aggregates issues detected during a debugging run, such as data drift, missing values,
or distribution anomalies.

It provides a structured view of all detected issues, along with utility methods for summarization and
serialization for logging, debugging, or downstream systems.

This is the primary output artifact of mldebug pipelines.
"""

from dataclasses import dataclass
from typing import Any

from .issue import Issue, Severity


@dataclass(frozen=True, slots=True)
class Report:
    """Aggregated output of a full ML debugging run."""

    issues: list[Issue]

    def summary(self) -> dict[str, int]:
        """Count issues by severity."""
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
            ]
        }
