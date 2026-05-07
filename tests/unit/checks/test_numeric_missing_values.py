import numpy as np

from mldebug.checks.numeric.variance_drift import run_numeric_variance_drift_check
from mldebug.config import NumericCheckConfig
from mldebug.models.feature_context import FeatureContext


def test_numeric_variance_drift_check_detects_increase_in_variance_ratio() -> None:
    feature = "feature_1"

    context = FeatureContext(
        feature=feature,
        reference=np.array([1, 2, 3], dtype=float),
        current=np.array([1, 2, 10], dtype=float),
        config=NumericCheckConfig(variance_drift_threshold=2.0),
    )

    issue = run_numeric_variance_drift_check(context)

    assert issue is not None
    assert issue.name == "variance_drift"
    assert issue.metric == "variance_ratio"
    assert issue.feature == feature
    assert issue.value is not None


def test_numeric_variance_drift_check_detects_decrease_in_variance_ratio() -> None:
    feature = "feature_1"

    context = FeatureContext(
        feature=feature,
        reference=np.array([1, 10, 100], dtype=float),
        current=np.array([5, 5, 5], dtype=float),
        config=NumericCheckConfig(variance_drift_threshold=2.0),
    )

    issue = run_numeric_variance_drift_check(context)

    assert issue is not None
    assert issue.name == "variance_drift"


def test_numeric_variance_drift_check_does_not_trigger_when_variance_is_stable() -> None:
    feature = "feature_1"

    context = FeatureContext(
        feature=feature,
        reference=np.array([1, 2, 3], dtype=float),
        current=np.array([1, 2, 3], dtype=float),
        config=NumericCheckConfig(variance_drift_threshold=2.0),
    )

    issue = run_numeric_variance_drift_check(context)

    assert issue is None


def test_numeric_variance_drift_check_returns_none_when_reference_variance_is_zero() -> None:
    feature = "feature_1"

    context = FeatureContext(
        feature=feature,
        reference=np.array([5, 5, 5], dtype=float),
        current=np.array([1, 2, 3], dtype=float),
        config=NumericCheckConfig(variance_drift_threshold=2.0),
    )

    issue = run_numeric_variance_drift_check(context)

    assert issue is None
