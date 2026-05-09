import numpy as np

from mldebug.checks.numeric.variance_drift import NumericVarianceDriftCheck
from mldebug.runtime.feature_context import FeatureContext
from tests.fixtures.generators import generate_normal_data


def test_numeric_variance_drift_check_detects_increase_in_variance_ratio() -> None:
    feature = "feature_1"

    ref = generate_normal_data(mean=0, std=1)
    cur = generate_normal_data(mean=0, std=3)

    context = FeatureContext(feature=feature, reference=ref, current=cur)

    issue = NumericVarianceDriftCheck(threshold=2.0)(context)

    assert issue is not None
    assert issue.name == "variance_drift"
    assert issue.metric == "variance_ratio"
    assert issue.feature == feature


def test_numeric_variance_drift_check_detects_decrease_in_variance_ratio() -> None:
    feature = "feature_1"

    ref = generate_normal_data(mean=0, std=3)
    cur = generate_normal_data(mean=0, std=1)

    context = FeatureContext(feature=feature, reference=ref, current=cur)

    issue = NumericVarianceDriftCheck(threshold=2.0)(context)

    assert issue is not None
    assert issue.name == "variance_drift"
    assert issue.metric == "variance_ratio"
    assert issue.feature == feature


def test_numeric_variance_drift_check_does_not_trigger_when_variance_is_stable() -> (
    None
):
    feature = "feature_1"

    ref = generate_normal_data(mean=0, std=1)
    cur = generate_normal_data(mean=0, std=1)

    context = FeatureContext(feature=feature, reference=ref, current=cur)

    issue = NumericVarianceDriftCheck(threshold=2.0)(context)

    assert issue is None


def test_numeric_variance_drift_check_returns_none_when_reference_variance_zero() -> (
    None
):
    feature = "feature_1"

    ref = np.array([5, 5, 5], dtype=float)
    cur = generate_normal_data()

    context = FeatureContext(feature=feature, reference=ref, current=cur)

    issue = NumericVarianceDriftCheck(threshold=2.0)(context)

    assert issue is None
