from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from mldebug.checks.ks import run_numeric_ks_test_check
from mldebug.checks.missing_values import run_numeric_missing_value_check
from mldebug.checks.psi import run_categorical_psi_drift_check
from mldebug.core.issue import Issue, Severity

CHECKS = {
    "numeric": [
        run_numeric_missing_value_check,
        run_numeric_ks_test_check,
    ],
    "categorical": [
        run_categorical_psi_drift_check,
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

    ref = reference[feature]
    cur = current[feature]

    is_empty_ref = _is_empty(ref)
    if is_empty_ref:
        issues.append(
            Issue(
                name="empty_feature_reference",
                metric="data_quality",
                severity=Severity.CRITICAL,
                message=f"{feature}: empty data in reference",
                feature=feature,
            )
        )

    is_empty_cur = _is_empty(cur)
    if is_empty_cur:
        issues.append(
            Issue(
                name="empty_feature_current",
                metric="data_quality",
                severity=Severity.CRITICAL,
                message=f"{feature}: empty data in current",
                feature=feature,
            )
        )

    if is_empty_ref or is_empty_cur:
        return issues

    ref_arr = _normalize(ftype, ref)
    cur_arr = _normalize(ftype, cur)

    for check_fn in CHECKS[ftype]:
        issue: Issue | None = check_fn(feature=feature, reference=ref_arr, current=cur_arr)
        if issue:
            issues.append(issue)

    return issues


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
