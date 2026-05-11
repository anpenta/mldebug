from dataclasses import dataclass
from typing import Any

from mldebug.scoring import score_issues

from .issue import Issue
from .severity import Severity


@dataclass(frozen=True, slots=True)
class Report:
    """Aggregated output of a full ML debugging run.

    Parameters
    ----------
    issues : list[Issue]
        Collection of detected issues.

    """

    issues: list[Issue]

    def summary(self) -> dict[str, Any]:
        """Summarize issues by severity and total count."""
        counts = {
            Severity.INFO.value: 0,
            Severity.WARNING.value: 0,
            Severity.CRITICAL.value: 0,
        }

        for issue in self.issues:
            counts[issue.severity.value] += 1

        return {
            "total": len(self.issues),
            "by_severity": counts,
            "status": "clean" if len(self.issues) == 0 else "issues_detected",
        }

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

    def score(self) -> dict[str, Any]:
        """Compute dataset quality score from detected issues.

        Returns
        -------
        dict[str, Any]
            A dictionary containing:
            - overall_score: float in range [0, 100] representing dataset quality
            - feature_scores: per-feature quality scores
            - status: dataset health status ("pass", "warning", or "fail")
            - schema_issue_count: number of schema-level issues detected
        """
        return score_issues(self.issues)
