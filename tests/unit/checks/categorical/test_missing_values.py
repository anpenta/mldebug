from mldebug.checks.categorical.missing_values import run_categorical_missing_value_check
from mldebug.core.config import CheckConfig
from mldebug.core.models.context import FeatureContext
from tests.fixtures.data.generators import generate_categorical_data
from tests.fixtures.data.missing_values import inject_categorical_missing_values


def test_run_categorical_missing_value_check_detects_increase() -> None:
    feature = "feature_1"

    ref = inject_categorical_missing_values(generate_categorical_data(), rate=0.01)

    cur = inject_categorical_missing_values(generate_categorical_data(), rate=0.2)

    context = FeatureContext(
        feature=feature, ftype="categorical", reference=ref, current=cur, config=CheckConfig(missing_threshold=0.05)
    )

    issue = run_categorical_missing_value_check(context)

    assert issue is not None
    assert issue.metric == "missing_rate_increase"
    assert issue.feature == feature
    assert issue.value is not None
    assert issue.value > 0


def test_run_categorical_missing_value_check_no_detection_when_stable() -> None:
    feature = "feature_1"

    ref = inject_categorical_missing_values(
        generate_categorical_data(),
        rate=0.05,
    )

    cur = inject_categorical_missing_values(
        generate_categorical_data(),
        rate=0.05,
    )

    context = FeatureContext(
        feature=feature, ftype="categorical", reference=ref, current=cur, config=CheckConfig(missing_threshold=0.05)
    )

    issue = run_categorical_missing_value_check(context)

    assert issue is None


def test_run_categorical_missing_value_check_no_detection_when_decrease() -> None:
    feature = "feature_1"

    ref = inject_categorical_missing_values(
        generate_categorical_data(),
        rate=0.25,
    )

    cur = inject_categorical_missing_values(
        generate_categorical_data(),
        rate=0.05,
    )

    context = FeatureContext(
        feature=feature, ftype="categorical", reference=ref, current=cur, config=CheckConfig(missing_threshold=0.05)
    )

    issue = run_categorical_missing_value_check(context)

    assert issue is None
