import pytest

from mldebug.domain.feature_type import FeatureType
from mldebug.inference.feature_type import infer_feature_type


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (["1", "2", "3"], FeatureType.NUMERIC),
        (["1.5", "2.5", "3.5"], FeatureType.NUMERIC),
        (["a", "b", "c"], FeatureType.CATEGORICAL),
        (["x", "y", "z"], FeatureType.CATEGORICAL),
        (["1", "a", "2", "b"], FeatureType.CATEGORICAL),
        (["1"] * 100, FeatureType.CATEGORICAL),
    ],
)
def test_infer_feature_type_for_various_cases(
    values: list[str], expected: FeatureType
) -> None:
    assert infer_feature_type(values) == expected


def test_infer_feature_type_does_not_fail_on_empty_input() -> None:
    values: list[str] = []

    result = infer_feature_type(values)

    assert result in {FeatureType.NUMERIC, FeatureType.CATEGORICAL}
