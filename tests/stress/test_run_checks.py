

from mldebug import run_checks
from tests.datasets.tabular import (
    generate_mixed_tabular_dataset,
)


def test_run_checks_stress_large_dataset() -> None:
    reference, current, schema = generate_mixed_tabular_dataset(n=50_000, n_features=100)

    report = run_checks(reference=reference, current=current, schema=schema)

    assert report is not None
    assert isinstance(report.issues, list)
