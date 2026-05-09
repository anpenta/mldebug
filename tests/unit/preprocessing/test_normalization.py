from typing import Any

import numpy as np
import pytest

from mldebug.preprocessing.normalization import (
    compute_numeric_ratio,
    compute_unique_ratio,
    normalize_categorical,
    normalize_numeric,
)


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ([1, 2, 3], [1.0, 2.0, 3.0]),
        ([1.5, 2.5], [1.5, 2.5]),
        ([1, 2.5, 3], [1.0, 2.5, 3.0]),
        (["1", "2", "3"], [1.0, 2.0, 3.0]),
        ([1, "2", 3.5], [1.0, 2.0, 3.5]),
        ([1, None, 3], [1.0, np.nan, 3.0]),  # None should become NaN.
        ([1, float("nan"), 3], [1.0, np.nan, 3.0]),
        ([], []),
    ],
)
def test_numeric_values_are_normalized_to_floats(
    data: list[Any], expected: list[str]
) -> None:
    out = normalize_numeric(data)

    assert np.allclose(out, expected, equal_nan=True)
    assert np.issubdtype(out.dtype, np.floating)


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (["a", "b", "c"], ["a", "b", "c"]),
        (["a", None, "b"], ["a", "", "b"]),
        (["a", float("nan"), "b"], ["a", "", "b"]),
        (["a", 1, 2.5], ["a", "1", "2.5"]),
        ([None, float("nan"), None], ["", "", ""]),
        # String-encoded missing values.
        (["a", "nan", "b"], ["a", "", "b"]),
        (["a", "NaN", "b"], ["a", "", "b"]),
        (["a", "None", "b"], ["a", "", "b"]),
        (["a", "none", "b"], ["a", "", "b"]),
        (
            # Mixed type and missing value normalization.
            ["a", "NaN", None, 3],
            ["a", "", "", "3"],
        ),
        ([], []),
    ],
)
def test_categorical_values_are_normalized_and_missing_values_are_filled(
    data: list[Any], expected: list[str]
) -> None:
    out = normalize_categorical(data)

    assert np.array_equal(out, np.array(expected))
    assert out.dtype.type is np.str_


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
