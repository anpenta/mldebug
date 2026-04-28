from typing import Any, Literal

from numpy.typing import NDArray

from .checks.data_quality.missing_values import run_missing_value_check
from .checks.drift.ks import run_ks_test_check
from .checks.drift.psi import run_psi_drift_check_categorical
from .core.issue import Issue, Severity
from .core.report import Report


def run_checks(
    reference: dict[str, NDArray[Any]],
    current: dict[str, NDArray[Any]],
    schema: dict[str, Literal["numeric", "categorical"]],
) -> Report:
    """Run all validation checks on input data.

    This is the main public entrypoint of the library. It executes a suite of checks (e.g.,
    missing values, data drift) on the provided datasets and returns a structured report of
    detected issues.

    Parameters
    ----------
    reference : dict[str, NDArray[Any]]
        Reference dataset keyed by feature name (e.g., training dataset).

    current : dict[str, NDArray[Any]]
        Current dataset keyed by feature name (e.g., production dataset).

    schema: dict[str, Literal["numeric", "categorical"]],
        Feature schema mapping feature name to type.

    Returns
    -------
    Report
        Aggregated report containing all detected issues.

    """
    checks = {
        "numeric": [
            run_missing_value_check,
            run_ks_test_check,
        ],
        "categorical": [
            run_psi_drift_check_categorical,
        ],
    }

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

        for check_fn in checks.get(ftype, []):
            issue = check_fn(feature=feature, reference=ref, current=cur)
            if issue is not None:
                issues.append(issue)

    return Report(issues=issues)
