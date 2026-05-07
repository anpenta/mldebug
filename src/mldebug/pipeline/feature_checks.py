from collections.abc import Callable, Mapping, Sequence
from typing import Any

from mldebug.models.context import FeatureContext
from mldebug.models.issue import Issue, Severity
from mldebug.models.types import FeatureType
from mldebug.registry.specs import FEATURE_SPECS


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

    normalized_ref = spec.normalizer(ref)
    normalized_cur = spec.normalizer(cur)

    context = FeatureContext(feature=feature, reference=normalized_ref, current=normalized_cur, config=spec.config)

    return _run_check_group(checks=spec.checks, context=context)


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


def _run_check_group(checks: list[Callable[[FeatureContext], Issue | None]], context: FeatureContext) -> list[Issue]:
    issues: list[Issue] = []

    for check_fn in checks:
        issue = check_fn(context)

        if issue:
            issues.append(issue)

    return issues
