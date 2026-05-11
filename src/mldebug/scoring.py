from collections import defaultdict
from typing import Any

from mldebug.domain.issue import Issue, Severity

_SEVERITY_PENALTY = {
    Severity.INFO: 5,
    Severity.WARNING: 15,
    Severity.CRITICAL: 40,
}


def score_issues(issues: list[Issue]) -> dict[str, Any]:
    """Compute dataset quality score from feature-level issues.

    Internal scoring function used by Report.score(). System-level issues
    are excluded before scoring.

    Implements:
    - feature grouping
    - severity-based penalties
    - mean aggregation
    """

    feature_issues = [i for i in issues if i.feature is not None]

    feature_buckets: dict[str, list[Issue]] = defaultdict(list)

    for issue in feature_issues:
        assert issue.feature
        feature_buckets[issue.feature].append(issue)

    feature_scores = {
        feature: _score_feature_issues(feature_issues)
        for feature, feature_issues in feature_buckets.items()
    }

    overall = (
        sum(feature_scores.values()) / len(feature_scores)
        if feature_scores
        else 100.0
    )

    status = "fail" if overall < 50 else "warning" if overall < 80 else "pass"

    return {
        "overall_score": overall,
        "feature_scores": feature_scores,
        "status": status,
        "system_issue_count": sum(1 for i in issues if i.feature is None),
    }


def _score_feature_issues(issues: list[Issue]) -> float:
    score = 100.0

    for issue in issues:
        score -= _SEVERITY_PENALTY[issue.severity]

    return max(0.0, score)
