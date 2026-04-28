from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

from .core.feature import run_feature_checks
from .core.report import Report
from .core.schema import check_schema_mismatches, validate_schema

if TYPE_CHECKING:
    from .core.issue import Issue


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
    validate_schema(schema=schema, reference=reference, current=current)

    issues: list[Issue] = []

    issues.extend(check_schema_mismatches(schema=schema, reference=reference, current=current))

    for feature, ftype in schema.items():
        issues.extend(run_feature_checks(feature=feature, ftype=ftype, reference=reference, current=current))

    return Report(issues=issues)
