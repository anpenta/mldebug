from mldebug.checks.categorical.missing_values import CategoricalMissingValueCheck
from mldebug.runtime.feature_context import FeatureContext
from tests.fixtures.generators import generate_categorical_data
from tests.fixtures.missing_values import inject_categorical_missing_values


def test_categorical_missing_value_check_triggers_when_missing_rate_increases() -> None:
    feature = "feature_1"

    ref = inject_categorical_missing_values(generate_categorical_data(), rate=0.01)

    cur = inject_categorical_missing_values(generate_categorical_data(), rate=0.2)

    context = FeatureContext(feature=feature, reference=ref, current=cur)

    issue = CategoricalMissingValueCheck(threshold=0.05)(context)

    assert issue is not None
    assert issue.metric == "missing_rate_increase"
    assert issue.feature == feature
    assert issue.value is not None
    assert issue.value > 0


def test_categorical_missing_value_check_no_trigger_when_missing_rate_is_stable() -> (
    None
):
    feature = "feature_1"

    ref = inject_categorical_missing_values(
        generate_categorical_data(),
        rate=0.05,
    )

    cur = inject_categorical_missing_values(
        generate_categorical_data(),
        rate=0.05,
    )

    context = FeatureContext(feature=feature, reference=ref, current=cur)

    issue = CategoricalMissingValueCheck(threshold=0.05)(context)

    assert issue is None


def test_categorical_missing_value_check_no_trigger_when_missing_rate_decreases() -> (
    None
):
    feature = "feature_1"

    ref = inject_categorical_missing_values(
        generate_categorical_data(),
        rate=0.25,
    )

    cur = inject_categorical_missing_values(
        generate_categorical_data(),
        rate=0.05,
    )

    context = FeatureContext(feature=feature, reference=ref, current=cur)

    issue = CategoricalMissingValueCheck(threshold=0.05)(context)

    assert issue is None
