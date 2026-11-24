from collections.abc import Mapping

from numpy.typing import ArrayLike

from mldebug.domain.feature_type import FeatureType
from mldebug.domain.issue import Issue, Severity
from mldebug.preprocessing.shared import (
    compute_present_mask,
    is_numeric_vector,
    normalize_str_array,
)


def analyze_schema(
    schema: Mapping[str, FeatureType],
    reference: Mapping[str, ArrayLike],
    current: Mapping[str, ArrayLike],
) -> list[Issue]:
    """Analyze schema consistency against reference and current datasets.

    Performs schema validation and detects mismatches between the provided schema and
    observed features in the datasets, including missing or unexpected features and
    feature type inconsistencies.

    Parameters
    ----------
    schema : Mapping[str, FeatureType]
        Feature-to-type mapping defining expected dataset structure.

    reference : Mapping[str, ArrayLike]
        Reference dataset keyed by feature name.

    current : Mapping[str, ArrayLike]
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

    issues.extend(
        _detect_missing_features(
            schema_keys=schema_keys, data_keys=reference_keys, side="reference"
        )
    )
    issues.extend(
        _detect_missing_features(
            schema_keys=schema_keys, data_keys=current_keys, side="current"
        )
    )

    issues.extend(
        _detect_unexpected_features(
            schema_keys=schema_keys, data_keys=reference_keys, side="reference"
        )
    )
    issues.extend(
        _detect_unexpected_features(
            schema_keys=schema_keys, data_keys=current_keys, side="current"
        )
    )

    issues.extend(
        _detect_type_mismatches(schema=schema, reference=reference, current=current)
    )

    return issues


def _create_empty_schema_issue() -> Issue:
    return Issue(
        name="empty_schema",
        metric="schema",
        severity=Severity.CRITICAL,
        message="schema is empty",
    )


def _detect_missing_features(
    schema_keys: set[str], data_keys: set[str], side: str
) -> list[Issue]:
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


def _detect_unexpected_features(
    schema_keys: set[str], data_keys: set[str], side: str
) -> list[Issue]:
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
    schema: Mapping[str, FeatureType],
    reference: Mapping[str, ArrayLike],
    current: Mapping[str, ArrayLike],
) -> list[Issue]:
    issues: list[Issue] = []

    for feature, declared_type in schema.items():
        if declared_type != FeatureType.NUMERIC:
            continue

        ref = reference.get(feature)
        cur = current.get(feature)

        if ref is None or cur is None:
            continue

        if _contains_non_numeric_values(ref) or _contains_non_numeric_values(cur):
            issues.append(
                Issue(
                    name="feature_type_mismatch",
                    metric="schema",
                    severity=Severity.CRITICAL,
                    message=(
                        f"{feature}: declared {declared_type.value.lower()} "
                        f"but observed values do not conform to this type"
                    ),
                    feature=feature,
                )
            )

    return issues


def _contains_non_numeric_values(values: ArrayLike) -> bool:
    normalized = normalize_str_array(values)
    present = normalized[compute_present_mask(normalized)]
    return not is_numeric_vector(present).all()
