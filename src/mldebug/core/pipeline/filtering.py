from collections.abc import Mapping, Sequence
from typing import Any

from mldebug.core.models import Issue, Severity


def get_valid_features(
    reference: Mapping[str, Sequence[Any]],
    current: Mapping[str, Sequence[Any]],
    schema: Mapping[str, str],
    schema_issues: list[Issue],
) -> list[str]:
    """Get features eligible for feature-level checks.

    A feature is included if it exists in the schema, reference dataset, and current dataset,
    and has no critical schema validation issues.

    Parameters
    ----------
    reference : Mapping[str, Sequence[Any]]
        Reference dataset keyed by feature name.

    current : Mapping[str, Sequence[Any]]
        Current dataset keyed by feature name.

    schema : Mapping[str, str]
        Feature-to-type mapping defining expected dataset structure.

    schema_issues : list[Issue]
        Issues produced by schema validation used to exclude invalid features.

    Returns
    -------
    list[str]
        Features safe to use in feature-level checks.

    """
    critical_features = {
        issue.feature for issue in schema_issues if issue.severity == Severity.CRITICAL and issue.feature is not None
    }

    return [
        feature
        for feature in schema
        if feature in reference and feature in current and feature not in critical_features
    ]
