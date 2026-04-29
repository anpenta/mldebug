from mldebug.checks.missing_values import run_missing_value_check
from tests.factories.data import generate_normal_data, inject_missing_values


def test_missing_values_detects_increase() -> None:
    feature = "feature_1"

    ref = inject_missing_values(generate_normal_data(), rate=0.01)
    cur = inject_missing_values(generate_normal_data(), rate=0.2)

    issue = run_missing_value_check(
        feature=feature,
        reference=ref,
        current=cur,
        threshold=0.05,
    )

    assert issue is not None
    assert issue.metric == "missing_rate_change"
    assert issue.feature == feature
    assert issue.value is not None
    assert issue.value > 0


def test_missing_values_no_detection_when_stable() -> None:
    feature = "feature_1"

    ref = inject_missing_values(generate_normal_data(), rate=0.05)
    cur = inject_missing_values(generate_normal_data(), rate=0.05)

    issue = run_missing_value_check(
        feature=feature,
        reference=ref,
        current=cur,
        threshold=0.05,
    )

    assert issue is None


def test_missing_values_no_detection_when_decrease() -> None:
    feature = "feature_1"

    ref = inject_missing_values(generate_normal_data(), rate=0.25)
    cur = inject_missing_values(generate_normal_data(), rate=0.05)

    issue = run_missing_value_check(
        feature=feature,
        reference=ref,
        current=cur,
        threshold=0.05,
    )

    assert issue is None
