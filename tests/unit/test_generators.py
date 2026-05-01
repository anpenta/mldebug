import numpy as np

from ..fixtures.generators import (
    generate_categorical_data,
    generate_normal_data,
    inject_categorical_missing_values,
    inject_numeric_missing_values,
)


def test_inject_numeric_missing_values_preserves_shape_and_copy() -> None:
    data = generate_normal_data(n=1000)

    corrupted = inject_numeric_missing_values(data, rate=0.2)

    assert len(corrupted) == len(data)
    assert not np.shares_memory(data, corrupted)


def test_inject_numeric_missing_values_approx_rate() -> None:
    data = generate_normal_data(n=1000)
    expected_rate = 0.2

    corrupted = inject_numeric_missing_values(data, rate=expected_rate, seed=42)
    actual_rate = np.mean(np.isnan(corrupted))

    assert abs(actual_rate - expected_rate) < 0.05


def test_inject_categorical_missing_values_preserves_shape_and_copy() -> None:
    data = generate_categorical_data(n=1000)

    corrupted = inject_categorical_missing_values(data, rate=0.2)

    assert len(corrupted) == len(data)
    assert not np.shares_memory(data, corrupted)


def test_inject_categorical_missing_values_approx_rate() -> None:
    data = generate_categorical_data(n=1000)
    expected_rate = 0.2

    corrupted = inject_categorical_missing_values(data, rate=expected_rate, seed=42)
    actual_rate = np.mean(corrupted == "")

    assert abs(actual_rate - expected_rate) < 0.05


def test_generate_categorical_data_values_as_expected() -> None:
    data = generate_categorical_data(n=1000)

    assert set(data).issubset({"A", "B", "C", "D", "E", "F", "G", "H"})
