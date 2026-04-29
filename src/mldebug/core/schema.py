from collections.abc import Mapping, Sequence
from typing import Any

from ..core.issue import Issue, Severity


def analyze_schema(
    schema: Mapping[str, str],
    reference: Mapping[str, Sequence[Any]],
    current: Mapping[str, Sequence[Any]],
) -> list[Issue]:
    """Analyze schema consistency against reference and current datasets.

    Performs schema validation and detects mismatches between the provided
    schema and the observed features in the datasets, including missing
    schema definitions and unexpected features.

    Parameters
    ----------
    schema : Mapping[str, str]
        Feature-to-type mapping defining expected dataset structure.

    reference : Mapping[str, Sequence[Any]]
        Reference dataset keyed by feature name.

    current : Mapping[str, Sequence[Any]]
        Current dataset keyed by feature name.

    Returns
    -------
    list[Issue]
        Schema-related issues detected during validation and comparison.

    """
    issues: list[Issue] = []

    if not schema:
        issues.append(
            Issue(
                name="empty_schema",
                metric="schema",
                severity=Severity.CRITICAL,
                message="Schema cannot be empty",
                feature=None,
            )
        )
        return issues

    schema_keys = set(schema)
    ref_keys = set(reference)
    cur_keys = set(current)

    # Missing schema definitions
    missing = (ref_keys | cur_keys) - schema_keys
    if missing:
        issues.append(
            Issue(
                name="missing_schema_definitions",
                metric="schema",
                severity=Severity.CRITICAL,
                message=f"Missing schema definitions for features: {sorted(missing)}",
                feature=None,
            )
        )

    # Unexpected features
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
