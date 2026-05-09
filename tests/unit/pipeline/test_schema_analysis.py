from mldebug.domain.feature_type import FeatureType
from mldebug.pipeline.schema_analysis import analyze_schema


def test_empty_schema_is_reported() -> None:
    ref = {"a": [1, 2, 3]}
    cur = {"a": [1, 2, 3]}
    schema = {}

    issues = analyze_schema(schema=schema, reference=ref, current=cur)

    assert any(i.name == "empty_schema" for i in issues)


def test_missing_features_in_reference_are_detected() -> None:
    ref = {}
    cur = {"a": [1, 2, 3]}
    schema = {"a": FeatureType.NUMERIC}

    issues = analyze_schema(schema=schema, reference=ref, current=cur)

    assert any(i.name == "missing_feature_reference" for i in issues)


def test_missing_features_in_current_are_detected() -> None:
    ref = {"a": [1, 2, 3]}
    cur = {}
    schema = {"a": FeatureType.NUMERIC}

    issues = analyze_schema(schema=schema, reference=ref, current=cur)

    assert any(i.name == "missing_feature_current" for i in issues)


def test_unexpected_features_in_reference_are_detected() -> None:
    ref = {"a": [1, 2, 3], "b": [1, 2, 3]}
    cur = {"a": [1, 2, 3]}
    schema = {"a": FeatureType.NUMERIC}

    issues = analyze_schema(schema=schema, reference=ref, current=cur)

    assert any(i.name == "unexpected_feature_reference" for i in issues)


def test_unexpected_features_in_current_are_detected() -> None:
    ref = {"a": [1, 2, 3]}
    cur = {"a": [1, 2, 3], "b": [1, 2, 3]}
    schema = {"a": FeatureType.NUMERIC}

    issues = analyze_schema(schema=schema, reference=ref, current=cur)

    assert any(i.name == "unexpected_feature_current" for i in issues)

def test_type_mismatch_detected_when_current_breaks_schema() -> None:
    ref = {"a": [1, 2, 3]}
    cur = {"a": ["x", "y", "z"]}
    schema = {"a": FeatureType.NUMERIC}

    issues = analyze_schema(
        schema=schema, reference=ref, current=cur
    )

    assert any(i.name == "feature_type_mismatch" for i in issues)


def test_no_type_mismatch_when_values_match_schema() -> None:
    ref = {"a": [1, 2, 3]}
    cur = {"a": [4, 5, 6]}
    schema = {"a": FeatureType.NUMERIC}

    issues = analyze_schema(
        schema=schema, reference=ref, current=cur
    )

    assert not any(i.name == "feature_type_mismatch" for i in issues)


def test_type_mismatch_detected_on_mixed_current_values() -> None:
    ref = {"a": [1, 2, 3]}
    cur = {"a": [1, "x", 3]}
    schema = {"a": FeatureType.NUMERIC}

    issues = analyze_schema(
        schema=schema, reference=ref, current=cur
    )

    assert any(i.name == "feature_type_mismatch" for i in issues)


def test_type_check_skipped_when_feature_missing() -> None:
    ref = {}
    cur = {}
    schema = {"a": FeatureType.NUMERIC}

    issues = analyze_schema(
        schema=schema, reference=ref, current=cur
    )

    assert not any(i.name == "feature_type_mismatch" for i in issues)
