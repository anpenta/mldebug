from mldebug import run_checks
from mldebug.domain.feature_type import FeatureType


def test_clean_data_produces_no_issues() -> None:
    ref = {"a": [1, 2, 3]}
    cur = {"a": [1, 2, 3]}
    schema = {"a": FeatureType.NUMERIC}

    report = run_checks(ref, cur, schema)

    assert report is not None
    assert isinstance(report.issues, list)
    assert len(report.issues) == 0
