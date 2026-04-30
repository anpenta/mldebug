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

    assert summary["by_severity"]["warning"] == 1
    assert summary["by_severity"]["info"] == 2
    assert summary["by_severity"]["critical"] == 0

    assert summary["total"] == 3
    assert summary["status"] == "issues_detected"


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


def test_report_to_logs_formats_output_correctly() -> None:
    report = Report(
        [
            Issue(
                name="psi_drift",
                metric="psi",
                severity=Severity.WARNING,
                message="country: drift detected",
                feature="country",
                value=0.32,
                threshold=0.2,
            )
        ]
    )

    logs = report.to_logs()

    assert isinstance(logs, list)
    assert len(logs) == 1

    assert logs[0] == "[WARNING] psi_drift - country: drift detected"
