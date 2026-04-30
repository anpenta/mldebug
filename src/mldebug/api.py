from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

from mldebug.core.feature import CHECKS, run_feature_checks
from mldebug.core.filtering import get_valid_features
from mldebug.core.report import Report
from mldebug.core.schema import analyze_schema

if TYPE_CHECKING:
    from .core.issue import Issue


def list_checks() -> dict[str, list[str]]:
    """List all available checks grouped by feature type.

    Returns
    -------
    dict[str, list[str]]
        A dictionary mapping each feature type (e.g. "numeric", "categorical")
        to a list of available check function names for that type.

    """
    return {group: [fn.__name__ for fn in checks] for group, checks in CHECKS.items()}


def run_checks(
    reference: Mapping[str, Sequence[Any]],
    current: Mapping[str, Sequence[Any]],
    schema: Mapping[str, Literal["numeric", "categorical"]],
) -> Report:
    """Run checks on reference and current datasets.

    This is the main entrypoint of the library. It performs schema analysis (validation and mismatch detection)
    followed by feature-level checks based on the provided schema, and returns a structured report of issues.

    Parameters
    ----------
    reference : Mapping[str, Sequence[Any]]
        Reference dataset keyed by feature name (e.g. training data).

    current : Mapping[str, Sequence[Any]]
        Current dataset keyed by feature name (e.g. production data).

    schema : Mapping[str, Literal["numeric", "categorical"]]
        Mapping of feature names to their expected types.

    Returns
    -------
    Report
        Aggregated report containing all detected issues.

    """
    issues: list[Issue] = []

    # Schema analysis (validation and mismatch detection).
    issues.extend(analyze_schema(schema=schema, reference=reference, current=current))

    valid_features = get_valid_features(reference=reference, current=current, schema=schema)

    # Feature execution (schema-driven).
    for feature in valid_features:
        ftype = schema[feature]
        issues.extend(
            run_feature_checks(
                feature=feature,
                ftype=ftype,
                reference=reference,
                current=current,
            )
        )

    return Report(issues=issues)
