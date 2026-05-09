from collections.abc import Mapping

from numpy.typing import ArrayLike

from mldebug.domain.feature_type import FeatureType
from mldebug.domain.issue import Issue, Severity
from mldebug.registry import FEATURE_SPECS
from mldebug.runtime.feature_context import FeatureContext
from mldebug.types import Array


def run_feature_checks(
    feature: str,
    ftype: FeatureType,
    reference: Mapping[str, ArrayLike],
    current: Mapping[str, ArrayLike],
) -> list[Issue]:
    """Run all checks for a single feature.

    Parameters
    ----------
    feature : str
        Feature name to evaluate.

    ftype : FeatureType
        Type of the feature determining which checks are executed.

    reference : Mapping[str, ArrayLike]
        Reference dataset keyed by feature name.

    current : Mapping[str, ArrayLike]
        Current dataset keyed by feature name.

    Returns
    -------
    list[Issue]
        Detected issues for the feature.

    """
    spec = FEATURE_SPECS[ftype]

    ref = spec.normalizer(reference[feature])
    cur = spec.normalizer(current[feature])

    empty_issues = _collect_empty_feature_issues(feature, ref, cur)
    if empty_issues:
        return empty_issues

    context = FeatureContext(feature=feature, reference=ref, current=cur)

    return [issue for check in spec.checks if (issue := check(context)) is not None]


def _collect_empty_feature_issues(feature: str, reference: Array, current: Array) -> list[Issue]:
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


def _is_empty(data: Array) -> bool:
    return len(data) == 0
