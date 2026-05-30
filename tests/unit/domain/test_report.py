import pytest

from mldebug.domain.issue import Issue, Severity
from mldebug.domain.report import Report


def test_report_sorting_is_deterministic() -> None:
    issues = [
        Issue(
            name="b", metric="m", severity=Severity.WARNING, message="msg", feature="b"
        ),
        Issue(
            name="a", metric="m", severity=Severity.WARNING, message="msg", feature="a"
        ),
        Issue(
            name="c", metric="m", severity=Severity.WARNING, message="msg", feature="c"
        ),
    ]

    r1 = Report(issues)
    r2 = Report(list(reversed(issues)))

    assert r1.issues == r2.issues


def test_report_sorting_order_is_feature_then_name() -> None:
    issues = [
        Issue(
            name="z", metric="m", severity=Severity.WARNING, message="msg", feature="b"
        ),
        Issue(
            name="a", metric="m", severity=Severity.WARNING, message="msg", feature="a"
        ),
        Issue(
            name="b", metric="m", severity=Severity.WARNING, message="msg", feature="a"
        ),
    ]

    report = Report(issues)

    assert [(i.feature, i.name) for i in report.issues] == [
        ("a", "a"),
        ("a", "b"),
        ("b", "z"),
    ]


def test_report_summary_groups_issues_by_severity() -> None:
    report = Report(
        [
            Issue(
                name="test", metric="metric", severity=Severity.WARNING, message="msg"
            ),
            Issue(name="test", metric="metric", severity=Severity.INFO, message="msg"),
            Issue(name="test", metric="metric", severity=Severity.INFO, message="msg"),
        ],
    )

    summary = report.summary()

    assert summary["by_severity"]["warning"] == 1
    assert summary["by_severity"]["info"] == 2
    assert summary["by_severity"]["critical"] == 0

    assert summary["total"] == 3
    assert summary["status"] == "issues_detected"


def test_report_to_dict_serializes_all_issues_with_full_schema() -> None:
    report = Report(
        [
            Issue(name="test", metric="metric", severity=Severity.INFO, message="msg"),
            Issue(
                name="test", metric="metric", severity=Severity.WARNING, message="msg"
            ),
        ],
    )

    report_as_dict = report.to_dict()

    assert report_as_dict == {
        "issues": [
            {
                "name": "test",
                "metric": "metric",
                "severity": "info",
                "message": "msg",
                "feature": None,
                "value": None,
                "threshold": None,
            },
            {
                "name": "test",
                "metric": "metric",
                "severity": "warning",
                "message": "msg",
                "feature": None,
                "value": None,
                "threshold": None,
            },
        ],
    }


def test_report_score_returns_valid_score_structure() -> None:
    report = Report(
        [
            Issue(
                name="missing_values",
                metric="missing_rate",
                severity=Severity.WARNING,
                message="msg",
                feature="age",
            )
        ]
    )

    result = report.score()

    assert "overall_score" in result
    assert "feature_scores" in result
    assert "status" in result
    assert "system_issue_count" in result


def test_report_score_empty_report_returns_perfect_score() -> None:
    report = Report([])

    result = report.score()

    assert result["overall_score"] == 100.0
    assert result["feature_scores"] == {}
    assert result["system_issue_count"] == 0
    assert result["status"] == "pass"


def test_report_score_reflects_issue_severity_impact() -> None:
    report = Report(
        [
            Issue(
                name="drift",
                metric="ks_test",
                severity=Severity.WARNING,
                message="msg",
                feature="age",
            )
        ]
    )

    result = report.score()

    assert result["overall_score"] < 100
    assert result["feature_scores"]["age"] < 100


def test_report_score_counts_schema_level_issues() -> None:
    report = Report(
        [
            Issue(
                name="schema_error",
                metric="schema",
                severity=Severity.CRITICAL,
                message="broken",
                feature=None,
            )
        ]
    )

    result = report.score()

    assert result["system_issue_count"] == 1


@pytest.mark.parametrize(
    "issues, expected",
    [
        ([], True),
        (
            [
                Issue(
                    name="test",
                    metric="metric",
                    severity=Severity.INFO,
                    message="msg",
                )
            ],
            False,
        ),
    ],
)
def test_report_is_clean_expected_behavior(issues: list[Issue], expected: bool) -> None:
    report = Report(issues)

    assert report.is_clean() is expected


@pytest.mark.parametrize(
    "issues, expected",
    [
        (
            [
                Issue(
                    name="schema_error",
                    metric="schema",
                    severity=Severity.CRITICAL,
                    message="broken",
                ),
                Issue(
                    name="warning",
                    metric="metric",
                    severity=Severity.WARNING,
                    message="msg",
                ),
            ],
            True,
        ),
        (
            [
                Issue(
                    name="warning",
                    metric="metric",
                    severity=Severity.WARNING,
                    message="msg",
                ),
                Issue(
                    name="info",
                    metric="metric",
                    severity=Severity.INFO,
                    message="msg",
                ),
            ],
            False,
        ),
        ([], False),
    ],
)
def test_report_has_critical_expected_behavior(
    issues: list[Issue], expected: bool
) -> None:
    report = Report(issues)

    assert report.has_critical() is expected


@pytest.mark.parametrize(
    "issues, expected",
    [
        (
            [
                Issue(
                    name="info",
                    metric="metric",
                    severity=Severity.INFO,
                    message="msg",
                ),
                Issue(
                    name="critical",
                    metric="metric",
                    severity=Severity.CRITICAL,
                    message="msg",
                ),
                Issue(
                    name="warning",
                    metric="metric",
                    severity=Severity.WARNING,
                    message="msg",
                ),
            ],
            Severity.CRITICAL,
        ),
        (
            [
                Issue(
                    name="info",
                    metric="metric",
                    severity=Severity.INFO,
                    message="msg",
                ),
                Issue(
                    name="warning",
                    metric="metric",
                    severity=Severity.WARNING,
                    message="msg",
                ),
            ],
            Severity.WARNING,
        ),
        (
            [
                Issue(
                    name="info",
                    metric="metric",
                    severity=Severity.INFO,
                    message="msg",
                )
            ],
            Severity.INFO,
        ),
        ([], None),
    ],
)
def test_report_highest_severity_expected_behavior(
    issues: list[Issue], expected: Severity | None
) -> None:
    report = Report(issues)

    assert report.highest_severity() == expected
