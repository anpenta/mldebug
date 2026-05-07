from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from mldebug.models.feature_type import FeatureType
from mldebug.models.issue import Severity
from mldebug.models.report import Report

from .feature_checks import run_feature_checks
from .schema_analysis import analyze_schema

if TYPE_CHECKING:
    from mldebug.models.issue import Issue


def run_checks(
    reference: Mapping[str, Sequence[Any]],
    current: Mapping[str, Sequence[Any]],
    schema: Mapping[str, FeatureType],
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

    schema : Mapping[str, FeatureType]
        Mapping of feature names to their expected types.

    Returns
    -------
    Report
        Aggregated report containing all detected issues.

    """
    schema_issues = analyze_schema(schema=schema, reference=reference, current=current)

    valid_features = _get_valid_features(
        reference=reference, current=current, schema=schema, schema_issues=schema_issues
    )

    feature_issues: list[Issue] = []
    for feature in valid_features:
        feature_issues.extend(
            run_feature_checks(feature=feature, ftype=schema[feature], reference=reference, current=current)
        )

    return Report(issues=schema_issues + feature_issues)


def _get_valid_features(
    reference: Mapping[str, Sequence[Any]],
    current: Mapping[str, Sequence[Any]],
    schema: Mapping[str, FeatureType],
    schema_issues: list[Issue],
) -> list[str]:
    critical_features = {i.feature for i in schema_issues if i.feature and i.severity == Severity.CRITICAL}
    return [
        feature
        for feature in schema
        if feature in reference and feature in current and feature not in critical_features
    ]
