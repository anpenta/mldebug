from collections.abc import Mapping, Sequence
from typing import Any

from ..core.issue import Issue, Severity


def validate_schema(
    schema: Mapping[str, str],
    reference: Mapping[str, Sequence[Any]],
    current: Mapping[str, Sequence[Any]],
) -> None:
    """Validate schema completeness against datasets.

    Parameters
    ----------
    schema : Mapping[str, str]
        Feature-to-type mapping.

    reference : Mapping[str, Sequence[Any]]
        Reference dataset keyed by feature name.

    current : Mapping[str, Sequence[Any]]
        Current dataset keyed by feature name.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If schema is empty or missing features present in data.

    """
    if not schema:
        error_msg = "Schema cannot be empty"
        raise ValueError(error_msg)

    schema_keys = set(schema)
    ref_keys = set(reference)
    cur_keys = set(current)

    missing = (ref_keys | cur_keys) - schema_keys
    if missing:
        error_msg = f"Missing schema definitions for features: {sorted(missing)}"
        raise ValueError(error_msg)


def check_schema_mismatches(
    schema: Mapping[str, str],
    reference: Mapping[str, Sequence[Any]],
    current: Mapping[str, Sequence[Any]],
) -> list[Issue]:
    """Detect schema mismatches between declared schema and datasets.

    Parameters
    ----------
    schema : Mapping[str, str]
        Feature-to-type mapping.

    reference : Mapping[str, Sequence[Any]]
        Reference dataset keyed by feature name.

    current : Mapping[str, Sequence[Any]]
        Current dataset keyed by feature name.

    Returns
    -------
    list[Issue]
        List of schema mismatch issues.

    """
    issues: list[Issue] = []

    schema_keys = set(schema)
    ref_keys = set(reference)
    cur_keys = set(current)

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

    return issues
