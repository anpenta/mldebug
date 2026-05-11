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
        """Return a dataset quality score.

        The score represents data quality based only on feature-level issues.
        System-level issues (e.g. schema errors, invalid inputs) are not included
        in the score but are available in the report.

        Returns
        -------
        dict[str, Any]
            Dictionary containing:

            overall_score : float
                Dataset quality score in [0, 100]. Higher is better.

            feature_scores : dict[str, float]
                Per-feature scores.

            status : str
                pass / warning / fail.

            system_issue_count : int
                Number of system-level issues.
        """
        return score_issues(self.issues)
