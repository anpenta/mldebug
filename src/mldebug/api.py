from collections.abc import Mapping, Sequence
from typing import Any, Literal

from .core.feature import run_feature_checks
from .core.issue import Issue, Severity
from .core.report import Report
from .core.schema import validate_schema


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
    issues: list[Issue] = []

    # 1. Schema validation (structural issues)
    issues.extend(
        validate_schema(
            schema=schema,
            reference=reference,
            current=current,
        )
    )

    # 2. Explicit mismatch detection (IMPORTANT FIX)
    ref_keys = set(reference)
    cur_keys = set(current)
    schema_keys = set(schema)

    for f in ref_keys - schema_keys:
        issues.append(  # noqa: PERF401 # Hurts readability.
            Issue(
                name="unexpected_feature_reference",
                metric="schema",
                severity=Severity.WARNING,
                message=f"'{f}' present in reference but not in schema",
                feature=f,
            )
        )

    for f in cur_keys - schema_keys:
        issues.append(  # noqa: PERF401 # Hurts readability.
            Issue(
                name="unexpected_feature_current",
                metric="schema",
                severity=Severity.WARNING,
                message=f"'{f}' present in current but not in schema",
                feature=f,
            )
        )

    # 3. Feature-level checks (schema-driven execution)
    for feature, ftype in schema.items():
        issues.extend(
            run_feature_checks(
                feature=feature,
                ftype=ftype,
                reference=reference,
                current=current,
            )
        )

    return Report(issues=issues)
