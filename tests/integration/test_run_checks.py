from mldebug import run_checks
from mldebug.core.models.issue import Severity
from tests.fixtures.generators import generate_normal_data, inject_numeric_missing_values


def test_run_checks_detects_numeric_drift() -> None:
    ref = {"feature_1": generate_normal_data(mean=0)}
    cur = {"feature_1": generate_normal_data(mean=5)}

    schema = {"feature_1": "numeric"}

    report = run_checks(reference=ref, current=cur, schema=schema)

    assert isinstance(report.issues, list)
    assert len(report.issues) > 0

    ks_issues = [i for i in report.issues if i.name == "ks_test"]

    assert len(ks_issues) > 0
    assert all(i.severity == Severity.WARNING for i in ks_issues)


def test_run_checks_detects_categorical_drift() -> None:
    ref = {
        "feature_1": ["A"] * 80 + ["B"] * 20,
    }

    cur = {
        "feature_1": ["A"] * 40 + ["B"] * 40 + ["C"] * 20,
    }

    schema = {"feature_1": "categorical"}

    report = run_checks(reference=ref, current=cur, schema=schema)

    assert isinstance(report.issues, list)
    assert len(report.issues) > 0

    psi_issues = [i for i in report.issues if i.name == "psi_drift"]

    assert len(psi_issues) > 0
    assert all(i.severity == Severity.WARNING for i in psi_issues)


def test_run_checks_detects_missing_values() -> None:
    ref = {"feature_1": generate_normal_data(mean=0)}
    cur = {
        "feature_1": inject_numeric_missing_values(generate_normal_data(mean=0), rate=0.3),
    }

    schema = {"feature_1": "numeric"}

    report = run_checks(reference=ref, current=cur, schema=schema)

    assert isinstance(report.issues, list)
    assert len(report.issues) > 0

    missing_issues = [i for i in report.issues if i.name == "missing_values"]

    assert len(missing_issues) > 0
    assert all(i.severity == Severity.WARNING for i in missing_issues)


def test_run_checks_handles_empty_schema() -> None:
    ref = {"a": [1, 2, 3]}
    cur = {"a": [1, 2, 3]}
    schema = {}

    report = run_checks(ref, cur, schema)

    assert any(i.name == "empty_schema" for i in report.issues)


def test_run_checks_detects_missing_reference_feature() -> None:
    ref = {}
    cur = {"a": [1, 2, 3]}
    schema = {"a": "numeric"}

    report = run_checks(ref, cur, schema)

    assert any(i.name == "missing_feature_reference" for i in report.issues)


def test_run_checks_detects_missing_current_feature() -> None:
    ref = {"a": [1, 2, 3]}
    cur = {}
    schema = {"a": "numeric"}

    report = run_checks(ref, cur, schema)

    assert any(i.name == "missing_feature_current" for i in report.issues)


def test_run_checks_detects_unexpected_reference_feature() -> None:
    ref = {"a": [1, 2, 3], "b": [1, 2, 3]}
    cur = {"a": [1, 2, 3]}
    schema = {"a": "numeric"}

    report = run_checks(ref, cur, schema)

    assert any(i.name == "unexpected_feature_reference" for i in report.issues)


def test_run_checks_detects_unexpected_current_feature() -> None:
    ref = {"a": [1, 2, 3]}
    cur = {"a": [1, 2, 3], "b": [1, 2, 3]}
    schema = {"a": "numeric"}

    report = run_checks(ref, cur, schema)

    assert any(i.name == "unexpected_feature_current" for i in report.issues)


def test_run_checks_detects_empty_reference_feature() -> None:
    ref = {"feature_1": []}
    cur = {"feature_1": [1, 2, 3]}

    schema = {"feature_1": "numeric"}

    report = run_checks(ref, cur, schema)

    assert any(i.name == "empty_feature_reference" for i in report.issues)


def test_run_checks_detects_empty_current_feature() -> None:
    ref = {"feature_1": [1, 2, 3]}
    cur = {"feature_1": []}

    schema = {"feature_1": "numeric"}

    report = run_checks(ref, cur, schema)

    assert any(i.name == "empty_feature_current" for i in report.issues)


def test_run_checks_returns_no_issues_for_clean_data() -> None:
    ref = {"a": [1, 2, 3]}
    cur = {"a": [1, 2, 3]}
    schema = {"a": "numeric"}

    report = run_checks(ref, cur, schema)

    assert report is not None
    assert isinstance(report.issues, list)
    assert len(report.issues) == 0


def test_run_checks_handles_empty_inputs() -> None:
    ref = {}
    cur = {}
    schema = {}

    report = run_checks(ref, cur, schema)

    assert any(i.name == "empty_schema" for i in report.issues)
