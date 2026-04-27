from mldebug import run_checks
from mldebug.core.issue import Severity
from tests.fixtures.data import generate_normal_data, inject_missing_values


def test_run_checks_detects_drift():
    ref = {
        "feature_1": generate_normal_data(mean=0),
    }

    cur = {
        "feature_1": generate_normal_data(mean=5),  # strong drift
    }

    schema = {
        "feature_1": "numeric",
    }

    report = run_checks(reference=ref, current=cur, schema=schema)

    assert report is not None
    assert isinstance(report.issues, list)
    assert len(report.issues) > 0

    drift_issues = [i for i in report.issues if "drift" in i.name or "shift" in i.name]

    assert len(drift_issues) > 0
    assert all(i.severity in (Severity.WARNING, Severity.CRITICAL) for i in drift_issues)


def test_run_checks_detects_missing_values():
    ref = {
        "feature_1": generate_normal_data(mean=0),
    }

    cur = {
        "feature_1": inject_missing_values(
            generate_normal_data(mean=0),
            rate=0.3,
        ),
    }

    schema = {
        "feature_1": "numeric",
    }

    report = run_checks(ref, cur, schema)

    assert report is not None
    assert isinstance(report.issues, list)
    assert len(report.issues) > 0

    missing_issues = [i for i in report.issues if "missing" in i.name]

    assert len(missing_issues) > 0
    assert all(i.severity in (Severity.WARNING, Severity.CRITICAL) for i in missing_issues)
