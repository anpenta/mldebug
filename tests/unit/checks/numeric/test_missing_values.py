from mldebug.checks.numeric.missing_values import run_numeric_missing_value_check
from mldebug.core.config import NumericCheckConfig
from mldebug.core.models.context import NumericFeatureContext
from tests.fixtures.data.generators import generate_normal_data
from tests.fixtures.data.missing_values import inject_numeric_missing_values


def test_numeric_missing_value_check_detects_increase_in_missing_rate() -> None:
    feature = "feature_1"

    ref = inject_numeric_missing_values(generate_normal_data(), rate=0.01)
    cur = inject_numeric_missing_values(generate_normal_data(), rate=0.2)

    context = NumericFeatureContext(
        feature=feature, reference=ref, current=cur, config=NumericCheckConfig(missing_threshold=0.05)
    )

    issue = run_numeric_missing_value_check(context)

    assert issue is not None
    assert issue.metric == "missing_rate_increase"
    assert issue.feature == feature
    assert issue.value is not None
    assert issue.value > 0


def test_numeric_missing_value_check_does_not_trigger_when_missing_rate_is_stable() -> None:
    feature = "feature_1"

    ref = inject_numeric_missing_values(generate_normal_data(), rate=0.05)
    cur = inject_numeric_missing_values(generate_normal_data(), rate=0.05)

    context = NumericFeatureContext(
        feature=feature, reference=ref, current=cur, config=NumericCheckConfig(missing_threshold=0.05)
    )

    issue = run_numeric_missing_value_check(context)

    assert issue is None


def test_numeric_missing_value_check_does_not_trigger_when_missing_rate_decreases() -> None:
    feature = "feature_1"

    ref = inject_numeric_missing_values(generate_normal_data(), rate=0.25)
    cur = inject_numeric_missing_values(generate_normal_data(), rate=0.05)

    context = NumericFeatureContext(
        feature=feature, reference=ref, current=cur, config=NumericCheckConfig(missing_threshold=0.05)
    )

    issue = run_numeric_missing_value_check(context)

    assert issue is None
