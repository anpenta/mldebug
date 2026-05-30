import numpy as np
import pytest

from mldebug.domain.feature_type import FeatureType
from mldebug.inference.schema import infer_schema


def test_infer_schema_returns_expected_feature_types() -> None:
    dataset = {
        "age": [20, 21, 22],
        "country": ["US", "CA", "US"],
        "score": np.array([0.1, 0.2, 0.3]),
    }

    result = infer_schema(dataset)

    assert result == {
        "age": FeatureType.NUMERIC,
        "country": FeatureType.CATEGORICAL,
        "score": FeatureType.NUMERIC,
    }


def test_infer_schema_rejects_non_mapping_inputs() -> None:
    with pytest.raises(TypeError, match="Expected a mapping"):
        infer_schema([1, 2, 3])


def test_infer_schema_rejects_non_string_feature_name() -> None:
    with pytest.raises(TypeError, match="Feature names must be strings"):
        infer_schema({1: [1, 2, 3]})


def test_infer_schema_rejects_empty_feature_name() -> None:
    with pytest.raises(ValueError, match="Feature names cannot be empty"):
        infer_schema({"": [1, 2, 3]})


def test_infer_schema_rejects_none_feature_values() -> None:
    with pytest.raises(TypeError, match="Values cannot be None"):
        infer_schema({"age": None})


class BadArrayLike:
    def __array__(self) -> None:
        raise ValueError("Cannot convert to array")


def test_infer_schema_rejects_non_array_like_values() -> None:
    with pytest.raises(TypeError, match="array-like"):
        infer_schema({"age": BadArrayLike()})


def test_infer_schema_rejects_non_1d_feature_values() -> None:
    with pytest.raises(TypeError, match="one-dimensional"):
        infer_schema({"age": [[1, 2], [3, 4]]})
