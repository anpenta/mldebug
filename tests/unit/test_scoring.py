import pytest

from mldebug.domain.issue import Issue, Severity
from mldebug.scoring import score_issues


def test_empty_issue_list_returns_perfect_result() -> None:
    result = score_issues([])

    assert result["overall_score"] == pytest.approx(100.0)
    assert result["feature_scores"] == {}
    assert result["status"] == "pass"
    assert result["system_issue_count"] == 0


def test_single_feature_warning_issue_reduces_feature_score() -> None:
    issues = [
        Issue(
            name="missing_values",
            metric="missing_rate",
            severity=Severity.WARNING,
            message="missing",
            feature="age",
        )
    ]

    result = score_issues(issues)

    assert result["overall_score"] < 100
    assert result["feature_scores"]["age"] == pytest.approx(85.0)
    assert result["system_issue_count"] == 0


def test_multiple_issues_on_same_feature_accumulate_penalty() -> None:
    issues = [
        Issue(
            name="a", metric="m1", severity=Severity.INFO, message="x", feature="age"
        ),
        Issue(
            name="b", metric="m2", severity=Severity.WARNING, message="x", feature="age"
        ),
    ]

    result = score_issues(issues)

    assert result["feature_scores"]["age"] == pytest.approx(80.0)


def test_feature_scores_are_aggregated_across_features() -> None:
    issues = [
        Issue(
            name="a", metric="m1", severity=Severity.WARNING, message="x", feature="age"
        ),
        Issue(
            name="b",
            metric="m2",
            severity=Severity.WARNING,
            message="x",
            feature="income",
        ),
    ]

    result = score_issues(issues)

    assert result["overall_score"] == pytest.approx(85.0)


def test_system_issues_are_reported_but_do_not_affect_score() -> None:
    issues = [
        Issue(
            name="schema1",
            metric="schema",
            severity=Severity.CRITICAL,
            message="broken",
            feature=None,
        ),
    ]

    result = score_issues(issues)

    assert result["overall_score"] == pytest.approx(100.0)
    assert result["system_issue_count"] == 1


def test_system_issues_do_not_affect_overall_score() -> None:
    issues = [
        Issue(
            name="feat1",
            metric="m1",
            severity=Severity.WARNING,
            message="x",
            feature="age",
        ),
        Issue(
            name="schema1",
            metric="schema",
            severity=Severity.WARNING,
            message="x",
            feature=None,
        ),
    ]

    result = score_issues(issues)

    assert result["overall_score"] == pytest.approx(85.0)
    assert result["status"] == "pass"


@pytest.mark.parametrize(
    "score,expected_status",
    [
        (90, "pass"),
        (75, "warning"),
        (40, "fail"),
    ],
)
def test_dataset_quality_status_thresholds(
    monkeypatch: pytest.MonkeyPatch, score: int, expected_status: str
) -> None:
    from mldebug import scoring

    monkeypatch.setattr(scoring, "_score_feature_issues", lambda _: score)

    issues = [
        Issue(
            name="a", metric="m1", severity=Severity.INFO, message="x", feature="age"
        ),
    ]

    result = scoring.score_issues(issues)

    assert result["status"] == expected_status
