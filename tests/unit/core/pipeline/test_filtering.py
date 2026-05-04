from mldebug.core.models import Issue, Severity
from mldebug.core.pipeline.filtering import get_valid_features


def test_filtering_returns_all_valid_features_when_no_issues() -> None:
    reference = {"a": [1], "b": [2]}
    current = {"a": [1], "b": [2]}
    schema = {"a": "numeric", "b": "numeric"}

    result = get_valid_features(reference, current, schema, schema_issues=[])

    assert set(result) == {"a", "b"}


def test_filtering_excludes_feature_missing_in_reference() -> None:
    reference = {"a": [1]}
    current = {"a": [1], "b": [2]}
    schema = {"a": "numeric", "b": "numeric"}

    result = get_valid_features(reference, current, schema, schema_issues=[])

    assert result == ["a"]


def test_filtering_excludes_feature_missing_in_current() -> None:
    reference = {"a": [1], "b": [2]}
    current = {"a": [1]}
    schema = {"a": "numeric", "b": "numeric"}

    result = get_valid_features(reference, current, schema, schema_issues=[])

    assert result == ["a"]


def test_filtering_excludes_feature_with_critical_issue() -> None:
    reference = {"a": [1], "b": [2]}
    current = {"a": [1], "b": [2]}
    schema = {"a": "numeric", "b": "numeric"}

    issues = [
        Issue(
            name="some_issue",
            metric="schema",
            severity=Severity.CRITICAL,
            message="bad feature",
            feature="b",
        )
    ]

    result = get_valid_features(reference, current, schema, schema_issues=issues)

    assert result == ["a"]


def test_filtering_keeps_feature_with_non_critical_issue() -> None:
    reference = {"a": [1], "b": [2]}
    current = {"a": [1], "b": [2]}
    schema = {"a": "numeric", "b": "numeric"}

    issues = [
        Issue(
            name="warning_issue",
            metric="schema",
            severity=Severity.WARNING,
            message="not critical",
            feature="b",
        )
    ]

    result = get_valid_features(reference, current, schema, schema_issues=issues)

    assert set(result) == {"a", "b"}


def test_filtering_ignores_issues_without_feature() -> None:
    reference = {"a": [1]}
    current = {"a": [1]}
    schema = {"a": "numeric"}

    issues = [
        Issue(
            name="global_issue",
            metric="schema",
            severity=Severity.CRITICAL,
            message="no feature",
            feature=None,
        )
    ]

    result = get_valid_features(reference, current, schema, schema_issues=issues)

    assert result == ["a"]
