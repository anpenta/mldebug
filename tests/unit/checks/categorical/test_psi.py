import numpy as np
import pytest
from numpy.typing import NDArray

from mldebug.checks.categorical.psi import _compute_categorical_psi, run_categorical_psi_drift_check
from mldebug.core.config import CheckConfig
from mldebug.core.models.context import FeatureContext


def test_run_categorical_psi_drift_check_detects_shift() -> None:
    feature = "feature_1"

    reference = np.array(["A"] * 80 + ["B"] * 20, dtype="object")
    current = np.array(["A"] * 40 + ["B"] * 40 + ["C"] * 20, dtype="object")

    context = FeatureContext(
        feature=feature,
        ftype="categorical",
        reference=reference,
        current=current,
        config=CheckConfig(psi_threshold=0.1),
    )

    issue = run_categorical_psi_drift_check(context)

    assert issue is not None
    assert issue.metric == "psi"
    assert issue.feature == feature
    assert issue.value > 0.1


def test_run_categorical_psi_drift_check_no_detection_when_stable() -> None:
    reference = np.array(["A"] * 50 + ["B"] * 50, dtype="object")
    current = np.array(["A"] * 52 + ["B"] * 48, dtype="object")

    context = FeatureContext(
        feature="feature_1",
        ftype="categorical",
        reference=reference,
        current=current,
        config=CheckConfig(psi_threshold=0.1),
    )

    issue = run_categorical_psi_drift_check(context)

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
def test_psi_detects_distribution_shift(reference: NDArray[np.str_], current: NDArray[np.str_]) -> None:
    psi = _compute_categorical_psi(reference=reference, current=current)

    assert np.isfinite(psi)
    assert psi > 0


def test_psi_is_zero_for_identical_distributions() -> None:
    data = np.array(["A", "A", "B", "C", "C"], dtype=str)

    psi = _compute_categorical_psi(reference=data, current=data)

    assert np.isclose(psi, 0.0)


def test_psi_is_order_invariant() -> None:
    ref1 = np.array(["A", "B", "B", "C"], dtype=str)
    ref2 = np.array(["C", "B", "A", "B"], dtype=str)

    cur1 = np.array(["A", "A", "B", "C"], dtype=str)
    cur2 = np.array(["C", "B", "A", "A"], dtype=str)

    psi1 = _compute_categorical_psi(reference=ref1, current=cur1)
    psi2 = _compute_categorical_psi(reference=ref2, current=cur2)

    assert np.isfinite(psi1)
    assert np.isclose(psi1, psi2)
