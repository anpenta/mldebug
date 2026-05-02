import numpy as np

from mldebug.checks.numeric.variance_drift import run_numeric_variance_drift_check
from mldebug.core.config import NumericCheckConfig
from mldebug.core.models.context import NumericFeatureContext
from tests.fixtures.data.generators import generate_normal_data


def test_numeric_variance_drift_check_detects_increase_in_variance_ratio() -> None:
    feature = "feature_1"

    ref = generate_normal_data(mean=0, std=1)
    cur = generate_normal_data(mean=0, std=3)

    context = NumericFeatureContext(
        feature=feature,
        reference=ref,
        current=cur,
        config=NumericCheckConfig(variance_drift_threshold=2.0),
    )

    issue = run_numeric_variance_drift_check(context)

    assert issue is not None
    assert issue.name == "variance_drift"
    assert issue.metric == "variance_ratio"
    assert issue.feature == feature


def test_numeric_variance_drift_check_detects_decrease_in_variance_ratio() -> None:
    feature = "feature_1"

    ref = generate_normal_data(mean=0, std=3)
    cur = generate_normal_data(mean=0, std=1)

    context = NumericFeatureContext(
        feature=feature,
        reference=ref,
        current=cur,
        config=NumericCheckConfig(variance_drift_threshold=2.0),
    )

    issue = run_numeric_variance_drift_check(context)

    assert issue is not None
    assert issue.name == "variance_drift"
    assert issue.metric == "variance_ratio"
    assert issue.feature == feature


def test_numeric_variance_drift_check_does_not_trigger_when_variance_is_stable() -> None:
    feature = "feature_1"

    ref = generate_normal_data(mean=0, std=1)
    cur = generate_normal_data(mean=0, std=1)

    context = NumericFeatureContext(
        feature=feature,
        reference=ref,
        current=cur,
        config=NumericCheckConfig(variance_drift_threshold=2.0),
    )

    issue = run_numeric_variance_drift_check(context)

    assert issue is None


def test_numeric_variance_drift_check_returns_none_when_reference_variance_is_zero() -> None:
    feature = "feature_1"

    ref = np.array([5, 5, 5], dtype=float)
    cur = generate_normal_data()

    context = NumericFeatureContext(
        feature=feature,
        reference=ref,
        current=cur,
        config=NumericCheckConfig(variance_drift_threshold=2.0),
    )

    issue = run_numeric_variance_drift_check(context)

    assert issue is None
