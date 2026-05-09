from mldebug.domain.feature_type import FeatureType
from mldebug.pipeline.feature_checks import run_feature_checks


def test_empty_reference_features_are_detected() -> None:
    feature = "feature"

    ref = {feature: []}
    cur = {feature: [1, 2, 3]}

    issues = run_feature_checks(
        feature=feature, ftype=FeatureType.NUMERIC, reference=ref, current=cur
    )

    assert any(i.name == "empty_feature_reference" for i in issues)


def test_empty_current_features_are_detected() -> None:
    feature = "feature"

    ref = {feature: [1, 2, 3]}
    cur = {feature: []}

    issues = run_feature_checks(
        feature=feature, ftype=FeatureType.NUMERIC, reference=ref, current=cur
    )

    assert any(i.name == "empty_feature_current" for i in issues)
