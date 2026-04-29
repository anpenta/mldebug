from mldebug.checks.ks import run_ks_test_check
from tests.factories.data import generate_normal_data


def test_ks_detects_shift() -> None:
    feature = "feature_1"

    reference = generate_normal_data()
    current = generate_normal_data(mean=1, std=1)

    issue = run_ks_test_check(
        feature=feature,
        reference=reference,
        current=current,
        alpha=0.05,
    )

    assert issue is not None
    assert issue.metric == "distribution_shift_score"
    assert issue.feature == feature


def test_ks_no_detection_when_stable() -> None:
    feature = "feature_1"

    reference = generate_normal_data()
    current = generate_normal_data(mean=0.05)

    issue = run_ks_test_check(
        feature=feature,
        reference=reference,
        current=current,
        alpha=0.05,
    )

    assert issue is None
