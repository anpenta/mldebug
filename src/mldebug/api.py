from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from .checks.data_quality.missing_values import run_missing_value_check
from .checks.drift.ks import run_ks_test_check
from .checks.drift.psi import run_psi_drift_check_categorical
from .core.issue import Issue, Severity
from .core.report import Report

_CHECKS = {
    "numeric": [
        run_missing_value_check,
        run_ks_test_check,
    ],
    "categorical": [
        run_psi_drift_check_categorical,
    ],
}


def run_checks(
    reference: Mapping[str, Sequence[Any]],
    current: Mapping[str, Sequence[Any]],
    schema: Mapping[str, Literal["numeric", "categorical"]],
) -> Report:
    """Run data quality and drift checks on reference and current datasets.

    This is the main entrypoint of the library. It executes a set of
    feature-type-specific checks (e.g., missing values, statistical drift)
    and returns a structured report of detected issues.

    Parameters
    ----------
    reference : Mapping[str, Sequence[Any]]
        Reference dataset keyed by feature name (e.g. training data).

    current : Mapping[str, Sequence[Any]]
        Current dataset keyed by feature name (e.g. production data).

    schema : Mapping[str, Literal["numeric", "categorical"]]
        Mapping of feature names to their expected type.

    Returns
    -------
    Report
        Aggregated report containing all detected issues.

    """
    if not schema:
        error_msg = "Schema cannot be empty"
        raise ValueError(error_msg)

    issues: list[Issue] = []

    for feature, ftype in schema.items():
        ref = reference.get(feature)
        cur = current.get(feature)

        if ref is None:
            issues.append(
                Issue(
                    name="missing_feature_reference",
                    metric="schema",
                    severity=Severity.CRITICAL,
                    message=f"{feature} missing in reference data",
                    feature=feature,
                )
            )

        if cur is None:
            issues.append(
                Issue(
                    name="missing_feature_current",
                    metric="schema",
                    severity=Severity.CRITICAL,
                    message=f"{feature} missing in current data",
                    feature=feature,
                )
            )

        if ref is None or cur is None:
            continue

        if ftype not in _CHECKS:
            error_msg = f"Unknown feature type '{ftype}' for feature '{feature}'. Expected one of: {list(_CHECKS)}"
            raise ValueError(error_msg)

        ref = _normalize(feature_type=ftype, data=ref)
        cur = _normalize(feature_type=ftype, data=cur)

        for check_fn in _CHECKS[ftype]:
            issue = check_fn(feature=feature, reference=ref, current=cur)
            if issue:
                issues.append(issue)

    return Report(issues=issues)


def _normalize(feature_type: Literal["numeric", "categorical"], data: Sequence[Any]) -> NDArray[Any]:
    if feature_type == "categorical":
        return np.asarray(data, dtype=object)
    if feature_type == "numeric":
        return np.asarray(data, dtype=float)
    error_msg = f"Unsupported feature type: {feature_type}"
    raise ValueError(error_msg)
