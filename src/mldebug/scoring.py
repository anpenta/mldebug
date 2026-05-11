from collections import defaultdict
from typing import Any

from mldebug.domain.issue import Issue, Severity

_SEVERITY_PENALTY = {
    Severity.INFO: 5,
    Severity.WARNING: 15,
    Severity.CRITICAL: 40,
}


def score_issues(issues: list[Issue]) -> dict[str, Any]:
    """Compute a deterministic quality score for a dataset based on detected issues.

    The scoring is split into two components:
    - Feature-level issues: aggregated per feature and converted into feature scores.
    - Schema-level issues: applied as a global penalty on the overall score.

    The final output provides:
    - an overall dataset score (0 - 100)
    - per-feature scores
    - a status indicating dataset quality (pass / warning / fail)
    """

    feature_buckets: dict[str, list[Issue]] = defaultdict(list)
    schema_issues: list[Issue] = []

    for issue in issues:
        if issue.feature is None:
            schema_issues.append(issue)
        else:
            feature_buckets[issue.feature].append(issue)

    feature_scores = {
        feature: _score_feature_issues(feature_issues)
        for feature, feature_issues in feature_buckets.items()
    }

    # Base overall is the average of feature scores.
    overall = (
        sum(feature_scores.values()) / len(feature_scores) if feature_scores else 100.0
    )

    schema_penalty = sum(_SEVERITY_PENALTY[i.severity] for i in schema_issues)

    overall = max(0.0, overall - schema_penalty)

    status = "fail" if overall < 50 else "warning" if overall < 80 else "pass"

    return {
        "overall_score": overall,
        "feature_scores": feature_scores,
        "status": status,
        "schema_issue_count": len(schema_issues),
    }


def _score_feature_issues(issues: list[Issue]) -> float:
    score = 100.0

    for issue in issues:
        score -= _SEVERITY_PENALTY[issue.severity]

    return max(0.0, score)
