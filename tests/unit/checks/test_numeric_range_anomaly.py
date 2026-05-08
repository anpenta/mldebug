import numpy as np
import pytest
from numpy.typing import NDArray

from mldebug.checks.numeric.range_anomaly import NumericRangeAnomalyCheck
from mldebug.runtime.feature_context import FeatureContext


@pytest.mark.parametrize(
    "reference, current, expected_outliers",
    [
        # single upper outlier
        (
            np.array([0.0, 1.0, 2.0]),
            np.array([0.0, 1.0, 2.0, 10.0]),
            1,
        ),
        # single lower outlier
        (
            np.array([0.0, 1.0, 2.0]),
            np.array([0.0, 1.0, 2.0, -10.0]),
            1,
        ),
        # both sides
        (
            np.array([0.0, 1.0, 2.0]),
            np.array([0.0, 1.0, 2.0, 10.0, -10.0]),
            2,
        ),
        # multiple upper outliers
        (
            np.array([0.0, 1.0, 2.0]),
            np.array([0.0, 1.0, 2.0, 10.0, 11.0]),
            2,
        ),
        # constant reference edge case
        (
            np.array([5.0, 5.0, 5.0]),
            np.array([5.0, 5.0, 6.0]),
            1,
        ),
    ],
)
def test_numeric_range_anomaly_check_detects_out_of_range_values(
    reference: NDArray[np.floating],
    current: NDArray[np.floating],
    expected_outliers: int,
) -> None:
    context = FeatureContext(feature="feature_1", reference=reference, current=current)

    issue = NumericRangeAnomalyCheck()(context)

    assert issue is not None
    assert issue.name == "range_anomaly"
    assert issue.metric == "out_of_range_count"
    assert issue.value == expected_outliers


@pytest.mark.parametrize(
    "reference, current",
    [
        # values strictly inside a small positive range
        (
            np.array([0.0, 1.0, 2.0]),
            np.array([0.1, 1.5, 1.9]),
        ),
        # symmetric range around zero, all values remain inside bounds
        (
            np.array([-5.0, 0.0, 5.0]),
            np.array([-4.0, 0.0, 4.0]),
        ),
        # larger scale range with safe interior values
        (
            np.array([10.0, 20.0, 30.0]),
            np.array([11.0, 25.0, 29.0]),
        ),
        # edge-safe case where current touches but does not exceed bounds
        (
            np.array([0.0, 5.0, 10.0]),
            np.array([0.0, 5.0, 10.0]),
        ),
    ],
)
def test_numeric_range_anomaly_check_does_not_trigger_when_within_range(
    reference: NDArray[np.floating],
    current: NDArray[np.floating],
) -> None:
    context = FeatureContext(feature="feature_1", reference=reference, current=current)

    issue = NumericRangeAnomalyCheck()(context)

    assert issue is None
