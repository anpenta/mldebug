from collections.abc import Mapping, Sequence
from typing import Any

from ..core.issue import Issue, Severity


def validate_schema(
    schema: Mapping[str, str],
    reference: Mapping[str, Sequence[Any]],
    current: Mapping[str, Sequence[Any]],
) -> list[Issue]:
    """Validate schema completeness against reference and current datasets.

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
        Schema-related issues including missing schema definitions
        and empty schema configuration.

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

    return issues
