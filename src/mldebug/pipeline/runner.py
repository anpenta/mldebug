from collections.abc import Mapping

from numpy.typing import ArrayLike

from mldebug.domain.feature_type import FeatureType
from mldebug.domain.issue import Issue, Severity
from mldebug.domain.report import Report

from .feature_checks import run_feature_checks
from .input_validation import validate_inputs
from .schema_analysis import analyze_schema


def validate(
    reference: Mapping[str, ArrayLike],
    current: Mapping[str, ArrayLike],
    schema: Mapping[str, FeatureType],
) -> Report:
    """Run validation checks on reference and current datasets.

    This is the main entrypoint of the library. It performs schema analysis (validation
    and mismatch detection) followed by feature-level checks based on the provided
    schema, and returns a structured report of issues.

    Parameters
    ----------
    reference : Mapping[str, ArrayLike]
        Reference dataset keyed by feature name (e.g. training data).

    current : Mapping[str, ArrayLike]
        Current dataset keyed by feature name (e.g. production data).

    schema : Mapping[str, FeatureType]
        Mapping of feature names to their expected types.

    Returns
    -------
    Report
        Aggregated report containing all detected issues.

    """
    validate_inputs(reference=reference, current=current, schema=schema)

    schema_issues = analyze_schema(schema=schema, reference=reference, current=current)

    valid_features = _get_valid_features(
        reference=reference, current=current, schema=schema, schema_issues=schema_issues
    )

    feature_issues: list[Issue] = []
    for feature in valid_features:
        feature_issues.extend(
            run_feature_checks(
                feature=feature,
                ftype=schema[feature],
                reference=reference,
                current=current,
            )
        )

    return Report(issues=schema_issues + feature_issues)


def _get_valid_features(
    reference: Mapping[str, ArrayLike],
    current: Mapping[str, ArrayLike],
    schema: Mapping[str, FeatureType],
    schema_issues: list[Issue],
) -> list[str]:
    critical_features = {
        i.feature
        for i in schema_issues
        if i.feature and i.severity == Severity.CRITICAL
    }
    return [
        feature
        for feature in schema
        if feature in reference
        and feature in current
        and feature not in critical_features
    ]
