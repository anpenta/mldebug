from mldebug.core.issue import Issue, Severity
from mldebug.core.report import Report


def test_report_summary_counts_issues_by_severity() -> None:
    report = Report(
        [
            Issue(name="test", metric="metric", severity=Severity.WARNING, message="msg"),
            Issue(name="test", metric="metric", severity=Severity.INFO, message="msg"),
            Issue(name="test", metric="metric", severity=Severity.INFO, message="msg"),
        ],
    )

    summary = report.summary()

    assert summary["warning"] == 1
    assert summary["info"] == 2
    assert summary["critical"] == 0


def test_report_to_dict_returns_dict_with_correct_structure() -> None:
    report = Report(
        [
            Issue(name="test", metric="metric", severity=Severity.WARNING, message="msg"),
            Issue(name="test", metric="metric", severity=Severity.INFO, message="msg"),
        ],
    )

    report_as_dict = report.to_dict()

    assert report_as_dict == {
        "issues": [
            {
                "name": "test",
                "metric": "metric",
                "severity": "warning",
                "message": "msg",
                "feature": None,
                "value": None,
                "threshold": None,
            },
            {
                "name": "test",
                "metric": "metric",
                "severity": "info",
                "message": "msg",
                "feature": None,
                "value": None,
                "threshold": None,
            },
        ],
    }
