from mldebug.checks.numeric.ks_test import run_numeric_ks_test_check
from mldebug.core.config import CheckConfig
from mldebug.core.models.context import FeatureContext
from tests.fixtures.data.generators import generate_normal_data


def test_run_numeric_ks_test_check_detects_shift() -> None:
    feature = "feature_1"

    reference = generate_normal_data()
    current = generate_normal_data(mean=1, std=1)

    context = FeatureContext(
        feature=feature, ftype="numeric", reference=reference, current=current, config=CheckConfig(ks_alpha=0.05)
    )

    issue = run_numeric_ks_test_check(context)

    assert issue is not None
    assert issue.metric == "distribution_shift_score"
    assert issue.feature == feature


def test_run_numeric_ks_test_check_no_detection_when_stable() -> None:
    feature = "feature_1"

    reference = generate_normal_data()
    current = generate_normal_data(mean=0.05)

    context = FeatureContext(
        feature=feature, ftype="numeric", reference=reference, current=current, config=CheckConfig(ks_alpha=0.05)
    )

    issue = run_numeric_ks_test_check(context)

    assert issue is None
