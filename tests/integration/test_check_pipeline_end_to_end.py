import pytest

from mldebug import FeatureType, Report, run_checks


def test_check_pipeline_runs_and_returns_report() -> None:
    ref = {"a": [1, 2, 3]}
    cur = {"a": [4, 5, 6]}
    schema = {"a": FeatureType.NUMERIC}

    report = run_checks(reference=ref, current=cur, schema=schema)

    assert isinstance(report, Report)
    assert isinstance(report.issues, list)


def test_check_pipeline_detects_schema_and_feature_issues() -> None:
    ref = {"a": [1, 2, 3], "b": [1, 2, 3]}
    cur = {"a": ["x", "y", "z"], "b": []}
    schema = {
        "a": FeatureType.NUMERIC,
        "b": FeatureType.NUMERIC,
    }

    report = run_checks(reference=ref, current=cur, schema=schema)

    issue_names = {i.name for i in report.issues}

    assert "feature_type_mismatch" in issue_names
    assert "empty_feature_current" in issue_names


def test_check_pipeline_handles_empty_inputs() -> None:
    ref = {}
    cur = {}
    schema = {}

    report = run_checks(reference=ref, current=cur, schema=schema)

    assert any(i.name == "empty_schema" for i in report.issues)


def test_check_pipeline_rejects_invalid_inputs() -> None:
    with pytest.raises(TypeError):
        run_checks(reference=None, current=None, schema=None)
