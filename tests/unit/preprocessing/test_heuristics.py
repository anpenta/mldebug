from typing import Any

import pytest

from mldebug.preprocessing.heuristics import compute_numeric_ratio, compute_unique_ratio


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (["1", "2.5", "3"], 1.0),
        (["1", "a", "3"], 2 / 3),
        (["1", "", "  ", "2"], 1.0),  # Missing-like values should be ignored.
        (["a", "b", ""], 0.0),
        ([], 0.0),  # Empty input. Ratio defaults to 0.0.
    ],
)
def test_numeric_ratio_computes_expected_ratio(
    values: list[Any], expected: float
) -> None:
    ratio = compute_numeric_ratio(values)

    assert ratio == expected


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (["1", "2", "3"], 1.0),
        (["1", "1", "2", "2"], 0.5),
        (["1", "", "  ", "2"], 1.0),  # Missing-like values should be ignored.
        (["", " ", "nan"], 0.0),  # No valid values. Ratio defaults to 0.0.
        ([], 0.0),  # Empty input. Ratio defaults to 0.0.
    ],
)
def test_unique_ratio_computes_expected_ratio(
    values: list[Any], expected: float
) -> None:
    ratio = compute_unique_ratio(values)

    assert ratio == expected
