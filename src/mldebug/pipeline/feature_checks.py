from collections.abc import Mapping, Sequence
from typing import Any

from mldebug.domain.feature_type import FeatureType
from mldebug.domain.issue import Issue, Severity
from mldebug.registry import FEATURE_SPECS
from mldebug.runtime.feature_context import FeatureContext


def run_feature_checks(
    feature: str,
    ftype: FeatureType,
    reference: Mapping[str, Sequence[Any]],
    current: Mapping[str, Sequence[Any]],
) -> list[Issue]:
    """Run all checks for a single feature.

    Parameters
    ----------
    feature : str
        Feature name to evaluate.

    ftype : FeatureType
        Type of the feature determining which checks are executed.

    reference : Mapping[str, Sequence[Any]]
        Reference dataset keyed by feature name.

    current : Mapping[str, Sequence[Any]]
        Current dataset keyed by feature name.

    Returns
    -------
    list[Issue]
        Detected issues for the feature.

    """

    ref = reference[feature]
    cur = current[feature]

    empty_issues = _collect_empty_feature_issues(feature, ref, cur)
    if empty_issues:
        return empty_issues

    spec = FEATURE_SPECS[ftype]

    ref = spec.normalizer(ref)
    cur = spec.normalizer(cur)

    context = FeatureContext(feature=feature, reference=ref, current=cur)

    return [issue for check in spec.checks if (issue := check(context)) is not None]


def _collect_empty_feature_issues(feature: str, reference: Sequence[Any], current: Sequence[Any]) -> list[Issue]:
    issues: list[Issue] = []

    if _is_empty(reference):
        issues.append(
            Issue(
                name="empty_feature_reference",
                metric="data_quality",
                severity=Severity.CRITICAL,
                message=f"{feature}: empty data in reference",
                feature=feature,
            )
        )

    if _is_empty(current):
        issues.append(
            Issue(
                name="empty_feature_current",
                metric="data_quality",
                severity=Severity.CRITICAL,
                message=f"{feature}: empty data in current",
                feature=feature,
            )
        )

    return issues


def _is_empty(data: Sequence[Any]) -> bool:
    return len(data) == 0
