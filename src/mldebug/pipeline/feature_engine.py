from collections.abc import Mapping, Sequence
from typing import Any, Literal

from mldebug.models.issue import Issue, Severity
from mldebug.registry.checks import CHECKS

from .checks import run_check_group
from .context import build_feature_context
from .normalization import normalize_feature


def run_feature_checks(
    feature: str,
    ftype: Literal["numeric", "categorical"],
    reference: Mapping[str, Sequence[Any]],
    current: Mapping[str, Sequence[Any]],
) -> list[Issue]:
    """Run all checks for a single feature.

    Parameters
    ----------
    feature : str
        Feature name to evaluate.

    ftype : Literal["numeric", "categorical"]
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

    normalized_ref, normalized_cur = normalize_feature(reference=ref, current=cur, ftype=ftype)

    context = build_feature_context(
        feature=feature, ftype=ftype, normalized_reference=normalized_ref, normalized_current=normalized_cur
    )

    return run_check_group(checks=CHECKS[ftype].checks, context=context)


def _collect_empty_feature_issues(
    feature: str,
    reference: Sequence[Any],
    current: Sequence[Any],
) -> list[Issue]:
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
