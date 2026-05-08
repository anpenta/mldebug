from mldebug.checks.numeric.missing_values import NumericMissingValueCheck
from mldebug.runtime.feature_context import FeatureContext
from tests.fixtures.generators import generate_normal_data
from tests.fixtures.missing_values import inject_numeric_missing_values


def test_numeric_missing_value_check_triggers_when_missing_rate_increases() -> None:
    feature = "feature_1"

    ref = inject_numeric_missing_values(generate_normal_data(), rate=0.01)

    cur = inject_numeric_missing_values(generate_normal_data(), rate=0.2)

    context = FeatureContext(feature=feature, reference=ref, current=cur)

    issue = NumericMissingValueCheck(threshold=0.05)(context)

    assert issue is not None
    assert issue.metric == "missing_rate_increase"
    assert issue.feature == feature
    assert issue.value is not None
    assert issue.value > 0


def test_numeric_missing_value_check_does_not_trigger_when_missing_rate_is_stable() -> None:
    feature = "feature_1"

    ref = inject_numeric_missing_values(generate_normal_data(), rate=0.05)

    cur = inject_numeric_missing_values(generate_normal_data(), rate=0.05)

    context = FeatureContext(
        feature=feature,
        reference=ref,
        current=cur,
    )

    issue = NumericMissingValueCheck(threshold=0.05)(context)

    assert issue is None


def test_numeric_missing_value_check_does_not_trigger_when_missing_rate_decreases() -> None:
    feature = "feature_1"

    ref = inject_numeric_missing_values(
        generate_normal_data(),
        rate=0.25,
    )

    cur = inject_numeric_missing_values(generate_normal_data(), rate=0.05)

    context = FeatureContext(
        feature=feature,
        reference=ref,
        current=cur,
    )

    issue = NumericMissingValueCheck(threshold=0.05)(context)

    assert issue is None
