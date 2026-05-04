import numpy as np

from mldebug.core.pipeline.schema import _compute_numeric_ratio


def test_numeric_ratio_computes_ratio_of_numeric_values() -> None:
    values = np.array(["1", "2.5", "3"], dtype=object)

    ratio = _compute_numeric_ratio(values)

    assert ratio == 1.0


def test_numeric_ratio_ignores_non_numeric_values() -> None:
    values = np.array(["1", "a", "3"], dtype=object)

    ratio = _compute_numeric_ratio(values)

    assert ratio == 2 / 3


def test_numeric_ratio_ignores_missing_values() -> None:
    values = np.array(["1", "", "  ", "2"], dtype=object)

    ratio = _compute_numeric_ratio(values)

    assert ratio == 1.0


def test_numeric_ratio_returns_zero_for_all_non_numeric_values() -> None:
    values = np.array(["a", "b", ""], dtype=object)

    ratio = _compute_numeric_ratio(values)

    assert ratio == 0.0


def test_numeric_ratio_returns_zero_for_empty_input() -> None:
    values = np.array([], dtype=object)

    ratio = _compute_numeric_ratio(values)

    assert ratio == 0.0
