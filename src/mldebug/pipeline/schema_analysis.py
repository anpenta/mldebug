from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np

from mldebug.domain.feature_type import FeatureType
from mldebug.domain.issue import Issue, Severity
from mldebug.registry import FEATURE_SPECS


def analyze_schema(
    schema: Mapping[str, FeatureType], reference: Mapping[str, Sequence[Any]], current: Mapping[str, Sequence[Any]]
) -> list[Issue]:
    """Analyze schema consistency against reference and current datasets.

    Performs schema validation and detects mismatches between the provided schema and observed features in the
    datasets, including missing or unexpected features and feature type inconsistencies.

    Parameters
    ----------
    schema : Mapping[str, FeatureType]
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
    if not schema:
        return [_create_empty_schema_issue()]

    schema_keys = set(schema)
    reference_keys = set(reference)
    current_keys = set(current)

    issues: list[Issue] = []

    issues.extend(_detect_missing_features(schema_keys=schema_keys, data_keys=reference_keys, side="reference"))
    issues.extend(_detect_missing_features(schema_keys=schema_keys, data_keys=current_keys, side="current"))

    issues.extend(_detect_unexpected_features(schema_keys=schema_keys, data_keys=reference_keys, side="reference"))
    issues.extend(_detect_unexpected_features(schema_keys=schema_keys, data_keys=current_keys, side="current"))

    issues.extend(_detect_type_mismatches(schema=schema, reference=reference, current=current))

    return issues


def _create_empty_schema_issue() -> Issue:
    return Issue(
        name="empty_schema",
        metric="schema",
        severity=Severity.CRITICAL,
        message="schema is empty",
    )


def _detect_missing_features(schema_keys: set[str], data_keys: set[str], side: str) -> list[Issue]:
    return [
        Issue(
            name=f"missing_feature_{side}",
            metric="schema",
            severity=Severity.CRITICAL,
            message=f"{feature}: missing in {side} data",
            feature=feature,
        )
        for feature in schema_keys - data_keys
    ]


def _detect_unexpected_features(schema_keys: set[str], data_keys: set[str], side: str) -> list[Issue]:
    return [
        Issue(
            name=f"unexpected_feature_{side}",
            metric="schema",
            severity=Severity.CRITICAL,
            message=f"{feature}: present in {side} but not in schema",
            feature=feature,
        )
        for feature in data_keys - schema_keys
    ]


def _detect_type_mismatches(
    schema: Mapping[str, FeatureType], reference: Mapping[str, Sequence[Any]], current: Mapping[str, Sequence[Any]]
) -> list[Issue]:
    issues: list[Issue] = []

    for feature, declared_type in schema.items():
        ref = reference.get(feature)
        cur = current.get(feature)

        if ref is None or cur is None:
            continue

        values = cast("Sequence[Any]", np.concatenate([np.asarray(ref, dtype=object), np.asarray(cur, dtype=object)]))

        if not FEATURE_SPECS[declared_type].detector(values):
            issues.append(
                Issue(
                    name="feature_type_mismatch",
                    metric="schema",
                    severity=Severity.WARNING,
                    message=(
                        f"{feature}: declared {declared_type.value.lower()} "
                        f"but observed values do not conform to this type"
                    ),
                    feature=feature,
                )
            )

    return issues
