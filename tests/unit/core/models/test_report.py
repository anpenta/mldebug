from mldebug.core.models.issue import Issue, Severity
from mldebug.core.models.report import Report


def test_report_summary_groups_issues_by_severity() -> None:
    report = Report(
        [
            Issue(name="test", metric="metric", severity=Severity.WARNING, message="msg"),
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
