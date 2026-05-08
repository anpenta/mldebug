from mldebug.checks.numeric.ks_test import NumericKSCheck
from mldebug.runtime.feature_context import FeatureContext
from tests.fixtures.generators import generate_normal_data


def test_numeric_ks_test_detects_distribution_shift() -> None:
    feature = "feature_1"

    reference = generate_normal_data()
    current = generate_normal_data(mean=1, std=1)

    context = FeatureContext(feature=feature, reference=reference, current=current)

    issue = NumericKSCheck(alpha=0.05)(context)

    assert issue is not None
    assert issue.metric == "distribution_shift_score"
    assert issue.feature == feature


def test_numeric_ks_test_does_not_trigger_when_distribution_is_stable() -> None:
    feature = "feature_1"

    reference = generate_normal_data()
    current = generate_normal_data(mean=0.05)

    context = FeatureContext(feature=feature, reference=reference, current=current)

    issue = NumericKSCheck(alpha=0.05)(context)

    assert issue is None
