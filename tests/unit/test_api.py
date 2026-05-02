from mldebug import list_checks, run_checks
from mldebug.core.models.issue import Severity
from tests.fixtures.data.generators import generate_normal_data
from tests.fixtures.data.missing_values import inject_numeric_missing_values


def test_checks_are_grouped_by_type() -> None:
    result = list_checks()

    assert isinstance(result, dict)
    assert len(result) > 0

    for group, checks in result.items():
        assert isinstance(group, str)
        assert isinstance(checks, list)
        assert all(isinstance(name, str) for name in checks)


def test_core_check_groups_exist() -> None:
    result = list_checks()

    assert "numeric" in result
    assert "categorical" in result


def test_expected_checks_are_registered() -> None:
    result = list_checks()

    numeric_checks = result["numeric"]
    categorical_checks = result["categorical"]

    assert any("missing" in name for name in numeric_checks)
    assert any("ks" in name for name in numeric_checks)
    assert any("psi" in name for name in categorical_checks)


def test_numeric_drift_is_detected_by_ks_test() -> None:
    ref = {"feature_1": generate_normal_data(mean=0)}
    cur = {"feature_1": generate_normal_data(mean=5)}

    schema = {"feature_1": "numeric"}

    report = run_checks(reference=ref, current=cur, schema=schema)

    assert isinstance(report.issues, list)
    assert len(report.issues) > 0

    ks_issues = [i for i in report.issues if i.name == "ks_test"]

    assert len(ks_issues) > 0
    assert all(i.severity == Severity.WARNING for i in ks_issues)


def test_categorical_drift_is_detected_by_psi() -> None:
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


def test_missing_values_are_flagged() -> None:
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


def test_empty_schema_is_reported() -> None:
    ref = {"a": [1, 2, 3]}
    cur = {"a": [1, 2, 3]}
    schema = {}

    report = run_checks(ref, cur, schema)

    assert any(i.name == "empty_schema" for i in report.issues)


def test_missing_features_in_reference_are_detected() -> None:
    ref = {}
    cur = {"a": [1, 2, 3]}
    schema = {"a": "numeric"}

    report = run_checks(ref, cur, schema)

    assert any(i.name == "missing_feature_reference" for i in report.issues)


def test_missing_features_in_current_are_detected() -> None:
    ref = {"a": [1, 2, 3]}
    cur = {}
    schema = {"a": "numeric"}

    report = run_checks(ref, cur, schema)

    assert any(i.name == "missing_feature_current" for i in report.issues)


def test_unexpected_features_in_reference_are_detected() -> None:
    ref = {"a": [1, 2, 3], "b": [1, 2, 3]}
    cur = {"a": [1, 2, 3]}
    schema = {"a": "numeric"}

    report = run_checks(ref, cur, schema)

    assert any(i.name == "unexpected_feature_reference" for i in report.issues)


def test_unexpected_features_in_current_are_detected() -> None:
    ref = {"a": [1, 2, 3]}
    cur = {"a": [1, 2, 3], "b": [1, 2, 3]}
    schema = {"a": "numeric"}

    report = run_checks(ref, cur, schema)

    assert any(i.name == "unexpected_feature_current" for i in report.issues)


def test_empty_reference_features_are_detected() -> None:
    ref = {"feature_1": []}
    cur = {"feature_1": [1, 2, 3]}

    schema = {"feature_1": "numeric"}

    report = run_checks(ref, cur, schema)

    assert any(i.name == "empty_feature_reference" for i in report.issues)


def test_empty_current_features_are_detected() -> None:
    ref = {"feature_1": [1, 2, 3]}
    cur = {"feature_1": []}

    schema = {"feature_1": "numeric"}

    report = run_checks(ref, cur, schema)

    assert any(i.name == "empty_feature_current" for i in report.issues)


def test_clean_data_produces_no_issues() -> None:
    ref = {"a": [1, 2, 3]}
    cur = {"a": [1, 2, 3]}
    schema = {"a": "numeric"}

    report = run_checks(ref, cur, schema)

    assert report is not None
    assert isinstance(report.issues, list)
    assert len(report.issues) == 0


def test_empty_inputs_are_handled() -> None:
    ref = {}
    cur = {}
    schema = {}

    report = run_checks(ref, cur, schema)

    assert any(i.name == "empty_schema" for i in report.issues)
