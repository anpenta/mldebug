from mldebug.core.issue import Issue, Severity
from mldebug.core.report import Report


def test_report_summary():
    report = Report([Issue("test", "metric", Severity.WARNING, "msg")])

    summary = report.summary()

    assert summary["warning"] == 1
