from mldebug import run_checks
from tests.datasets.tabular import (
    generate_mixed_tabular_dataset,
)


def test_run_checks_stress_large_dataset() -> None:
    reference, current, schema = generate_mixed_tabular_dataset(n=50_000, n_features=100)

    report = run_checks(reference=reference, current=current, schema=schema)

    assert report is not None
    assert isinstance(report.issues, list)

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

    assert all(i.name in valid_issue_names for i in report.issues)
    assert len(report.issues) < 10_000
