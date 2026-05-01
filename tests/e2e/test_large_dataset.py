from mldebug import run_checks
from tests.fixtures.tabular_dataset import generate_mixed_tabular_dataset


def test_run_checks_stress_large_dataset() -> None:
    reference, current, schema = generate_mixed_tabular_dataset(n=50_000, n_features=100)

    report = run_checks(reference=reference, current=current, schema=schema)

    valid_issue_names = {
        "ks_test",
        "psi_drift",
        "missing_values",
        "empty_schema",
        "missing_feature_reference",
        "missing_feature_current",
        "unexpected_feature_reference",
        "unexpected_feature_current",
        "empty_feature_reference",
        "empty_feature_current",
    }

    assert report is not None
    assert isinstance(report.issues, list)

    assert len(report.issues) > 0
    assert len(report.issues) < 10_000

    assert all(i.name in valid_issue_names and i.severity is not None for i in report.issues)
