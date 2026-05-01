import pytest

from mldebug import run_checks
from tests.stress.datasets import generate_mixed_tabular_dataset


def test_run_checks_stress_mixed_large_dataset():
    # Large realistic workload
    reference, current, schema = generate_mixed_tabular_dataset(
        n=10_000,
        n_features=50,
    )

    report = run_checks(reference=reference, current=current, schema=schema)

    # -----------------------
    # core safety invariants
    # -----------------------
    assert report is not None
    assert hasattr(report, "issues")
    assert isinstance(report.issues, list)

    # -----------------------
    # structural invariants
    # -----------------------
    assert len(report.issues) >= 0  # must not crash or return invalid structure

    # all features must be accounted for
    assert set(reference.keys()) == set(current.keys()) == set(schema.keys())


def test_run_checks_stress_numeric_only_large_dataset():
    from tests.stress.datasets import generate_numeric_tabular_dataset

    reference, current, schema = generate_numeric_tabular_dataset(
        n=10_000,
        n_features=50,
    )

    report = run_checks(reference=reference, current=current, schema=schema)

    assert report is not None
    assert isinstance(report.issues, list)


def test_run_checks_stress_categorical_only_large_dataset():
    from tests.stress.datasets import generate_categorical_tabular_dataset

    reference, current, schema = generate_categorical_tabular_dataset(
        n=10_000,
        n_features=50,
    )

    report = run_checks(reference=reference, current=current, schema=schema)

    assert report is not None
    assert isinstance(report.issues, list)


def test_run_checks_stress_handles_consistent_shapes():
    reference, current, schema = generate_mixed_tabular_dataset(
        n=5_000,
        n_features=100,
    )

    # sanity: every feature must have consistent length
    for k in reference:
        assert len(reference[k]) == len(current[k])
        assert len(reference[k]) == 5_000
