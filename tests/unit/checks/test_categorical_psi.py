import numpy as np
import pytest
from numpy.typing import NDArray

from mldebug.checks.categorical.psi import CategoricalPSICheck
from mldebug.runtime.feature_context import FeatureContext


def test_categorical_psi_check_triggers_on_distribution_shift() -> None:
    feature = "feature_1"

    reference = np.array(["A"] * 80 + ["B"] * 20, dtype="object")
    current = np.array(["A"] * 40 + ["B"] * 40 + ["C"] * 20, dtype="object")

    context = FeatureContext(feature=feature, reference=reference, current=current)

    issue = CategoricalPSICheck(threshold=0.1)(context)

    assert issue is not None
    assert issue.metric == "psi"
    assert issue.feature == feature
    assert issue.value > 0.1


def test_categorical_psi_check_does_not_trigger_for_stable_distribution() -> None:
    reference = np.array(["A"] * 50 + ["B"] * 50, dtype="object")
    current = np.array(["A"] * 52 + ["B"] * 48, dtype="object")

    context = FeatureContext(feature="feature_1", reference=reference, current=current)

    issue = CategoricalPSICheck(threshold=0.1)(context)

    assert issue is None


@pytest.mark.parametrize(
    ("reference", "current"),
    [
        # Simple shift.
        (
            np.array(["A"] * 80 + ["B"] * 20, dtype=str),
            np.array(["A"] * 50 + ["B"] * 50, dtype=str),
        ),
        # Missing category in current.
        (
            np.array(["A"] * 40 + ["B"] * 30 + ["C"] * 30, dtype=str),
            np.array(["A"] * 60 + ["B"] * 40, dtype=str),
        ),
        # New category in current.
        (
            np.array(["A"] * 100, dtype=str),
            np.array(["A"] * 80 + ["B"] * 20, dtype=str),
        ),
    ],
)
def test_categorical_psi_returns_positive_values_for_distribution_shift(
    reference: NDArray[np.str_], current: NDArray[np.str_]
) -> None:
    psi = CategoricalPSICheck()._compute_psi(reference=reference, current=current)

    assert np.isfinite(psi)
    assert psi > 0


def test_categorical_psi_is_zero_for_identical_distributions() -> None:
    data = np.array(["A", "A", "B", "C", "C"], dtype=str)

    psi = CategoricalPSICheck()._compute_psi(reference=data, current=data)

    assert np.isclose(psi, 0.0)


def test_categorical_psi_is_invariant_to_category_order() -> None:
    ref1 = np.array(["A", "B", "B", "C"], dtype=str)
    ref2 = np.array(["C", "B", "A", "B"], dtype=str)

    cur1 = np.array(["A", "A", "B", "C"], dtype=str)
    cur2 = np.array(["C", "B", "A", "A"], dtype=str)

    psi1 = CategoricalPSICheck()._compute_psi(reference=ref1, current=cur1)
    psi2 = CategoricalPSICheck()._compute_psi(reference=ref2, current=cur2)

    assert np.isfinite(psi1)
    assert np.isclose(psi1, psi2)
