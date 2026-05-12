import pytest

from mldebug import FeatureType, Report, validate


def test_validation_pipeline_runs_and_produces_report() -> None:
    ref = {"a": [1, 2, 3]}
    cur = {"a": [4, 5, 6]}
    schema = {"a": FeatureType.NUMERIC}

    report = validate(reference=ref, current=cur, schema=schema)

    assert isinstance(report, Report)
    assert isinstance(report.issues, list)
    assert report.summary()["total"] >= 0

    score = report.score()
    assert "overall_score" in score
    assert "feature_scores" in score
    assert "status" in score


def test_validation_pipeline_detects_schema_and_feature_issues() -> None:
    ref = {"a": [1, 2, 3], "b": [1, 2, 3]}
    cur = {"a": ["x", "y", "z"], "b": []}
    schema = {
        "a": FeatureType.NUMERIC,
        "b": FeatureType.NUMERIC,
    }

    report = validate(reference=ref, current=cur, schema=schema)

    issue_names = {i.name for i in report.issues}

    assert "feature_type_mismatch" in issue_names
    assert "empty_feature_current" in issue_names

    score = report.score()
    assert score["overall_score"] <= 100


def test_validation_pipeline_handles_empty_inputs() -> None:
    ref = {}
    cur = {}
    schema = {}

    report = validate(reference=ref, current=cur, schema=schema)

    assert any(i.name == "empty_schema" for i in report.issues)

    score = report.score()
    assert score["overall_score"] == pytest.approx(100.0)


def test_validation_pipeline_rejects_invalid_inputs() -> None:
    with pytest.raises(TypeError):
        validate(reference=None, current=None, schema=None)
