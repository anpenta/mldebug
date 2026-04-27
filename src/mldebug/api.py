from typing import Any, Literal

from numpy.typing import NDArray

from .checks import run_drift_check, run_missing_values_check
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

        if ref is not None and cur is not None:
            issues.extend(
                run_missing_values_check(
                    feature=feature,
                    reference=ref,
                    current=cur,
                    feature_type=ftype,
                )
            )

            issues.extend(
                run_drift_check(
                    feature=feature,
                    reference=ref,
                    current=cur,
                    feature_type=ftype,
                )
            )

    return Report(issues=issues)
