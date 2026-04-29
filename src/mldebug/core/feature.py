from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from mldebug.checks.ks import run_ks_test_check
from mldebug.checks.missing_values import run_missing_value_check
from mldebug.checks.psi import run_psi_drift_check_categorical
from mldebug.core.issue import Issue, Severity

_CHECKS = {
    "numeric": [
        run_missing_value_check,
        run_ks_test_check,
    ],
    "categorical": [
        run_psi_drift_check_categorical,
    ],
}


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
    issues: list[Issue] = []

    ref = reference.get(feature)
    cur = current.get(feature)

    issues.extend(_check_missing_features(feature, ref, cur))
    if ref is None or cur is None:
        return issues

    issues.extend(_check_empty_feature(feature, ref, cur))
    if _is_empty(ref) or _is_empty(cur):
        return issues

    ref_arr = _normalize(ftype, ref)
    cur_arr = _normalize(ftype, cur)

    for check_fn in _CHECKS[ftype]:
        issue: Issue | None = check_fn(feature=feature, reference=ref_arr, current=cur_arr)
        if issue:
            issues.append(issue)

    return issues


def _check_missing_features(
    feature: str,
    ref: Sequence[Any] | None,
    cur: Sequence[Any] | None,
) -> list[Issue]:
    issues: list[Issue] = []

    if ref is None:
        issues.append(
            Issue(
                name="missing_feature_reference",
                metric="schema",
                severity=Severity.CRITICAL,
                message=f"'{feature}' missing in reference data",
                feature=feature,
            )
        )

    if cur is None:
        issues.append(
            Issue(
                name="missing_feature_current",
                metric="schema",
                severity=Severity.CRITICAL,
                message=f"'{feature}' missing in current data",
                feature=feature,
            )
        )

    return issues


def _check_empty_feature(
    feature: str,
    ref: Sequence[Any],
    cur: Sequence[Any],
) -> list[Issue]:
    if _is_empty(ref) or _is_empty(cur):
        return [
            Issue(
                name="empty_feature_data",
                metric="data_quality",
                severity=Severity.WARNING,
                message=f"'{feature}' has empty data",
                feature=feature,
            )
        ]
    return []


def _is_empty(data: Sequence[Any]) -> bool:
    return len(data) == 0


def _normalize(
    feature_type: Literal["numeric", "categorical"],
    data: Sequence[Any],
) -> NDArray[Any]:
    if feature_type == "categorical":
        return np.asarray(data, dtype=object)

    if feature_type == "numeric":
        return np.asarray(data, dtype=float)

    error_msg = f"Unsupported feature type: {feature_type}"
    raise ValueError(error_msg)
