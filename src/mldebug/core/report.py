from __future__ import annotations

from dataclasses import dataclass
from typing import dict, list

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

    def filter(
        self,
        severity: Severity | None = None,
        feature: str | None = None,
    ) -> Report:
        """Returns a filtered view of the report."""
        filtered = self.issues

        if severity is not None:
            filtered = [i for i in filtered if i.severity == severity]

        if feature is not None:
            filtered = [i for i in filtered if i.feature == feature]

        return Report(filtered)

    def to_dict(self) -> dict:
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
