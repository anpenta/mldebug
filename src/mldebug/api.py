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

    schema_keys = set(schema)
    ref_keys = set(reference)
    cur_keys = set(current)

    # Schema validation.
    extra_ref = ref_keys - schema_keys
    extra_cur = cur_keys - schema_keys
    missing_schema = (ref_keys | cur_keys) - schema_keys

    for f in extra_ref:
        issues.append(  # noqa: PERF401 # Hurts readability.
            Issue(
                name="unexpected_feature_reference",
                metric="schema",
                severity=Severity.WARNING,
                message=f"'{f}' present in reference but not in schema",
                feature=f,
            )
        )

    for f in extra_cur:
        issues.append(  # noqa: PERF401 # Hurts readability.
            Issue(
                name="unexpected_feature_current",
                metric="schema",
                severity=Severity.WARNING,
                message=f"'{f}' present in current but not in schema",
                feature=f,
            )
        )

    if missing_schema:
        error_msg = f"Missing schema definitions for features: {sorted(missing_schema)}"
        raise ValueError(error_msg)

    # Feature loop.
    for feature, ftype in schema.items():
        ref = reference.get(feature)
        cur = current.get(feature)

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

        if ref is None or cur is None:
            continue

        if len(ref) == 0 or len(cur) == 0:
            issues.append(
                Issue(
                    name="empty_feature_data",
                    metric="data_quality",
                    severity=Severity.WARNING,
                    message=f"'{feature}' has empty data",
                    feature=feature,
                )
            )
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
