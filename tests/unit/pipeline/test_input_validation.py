# We are actually doing input validation in here so disabling below check.
# pyright: reportArgumentType=false
from typing import Any

import pytest

from mldebug.domain.feature_type import FeatureType
from mldebug.pipeline.input_validation import validate_inputs


def _get_valid_reference() -> dict[str, list[int]]:
    return {"age": [1, 2, 3], "income": [100, 200, 300]}


def _get_valid_current() -> dict[str, list[int]]:
    return {"age": [2, 3, 4], "income": [110, 210, 310]}


def _get_valid_schema() -> dict[str, FeatureType]:
    return {"age": FeatureType.NUMERIC, "income": FeatureType.NUMERIC}


def test_input_validation_does_not_raise_when_input_valid() -> None:
    validate_inputs(
        reference=_get_valid_reference(),
        current=_get_valid_current(),
        schema=_get_valid_schema(),
    )


@pytest.mark.parametrize(
    "arg_name, value",
    [
        ("reference", [1, 2, 3]),
        ("current", [1, 2, 3]),
        ("schema", [("age", FeatureType.NUMERIC)]),
    ],
)
def test_input_validation_rejects_non_mapping_inputs(
    arg_name: str, value: list[Any]
) -> None:
    kwargs = {
        "reference": _get_valid_reference(),
        "current": _get_valid_current(),
        "schema": _get_valid_schema(),
    }
    kwargs[arg_name] = value

    with pytest.raises(TypeError, match=arg_name):
        validate_inputs(**kwargs)


def test_input_validation_rejects_non_string_feature_name_in_schema() -> None:
    schema = {123: FeatureType.NUMERIC}

    with pytest.raises(TypeError, match="Feature names must be strings"):
        validate_inputs(
            reference=_get_valid_reference(),
            current=_get_valid_current(),
            schema=schema,
        )


def test_input_validation_rejects_empty_feature_name_in_schema() -> None:
    schema = {"": FeatureType.NUMERIC}

    with pytest.raises(ValueError, match="Feature names cannot be empty"):
        validate_inputs(
            reference=_get_valid_reference(),
            current=_get_valid_current(),
            schema=schema,
        )


def test_input_validation_rejects_invalid_feature_type_in_schema() -> None:
    schema = {"age": "numeric"}

    with pytest.raises(TypeError, match="Invalid feature type"):
        validate_inputs(
            reference=_get_valid_reference(),
            current=_get_valid_current(),
            schema=schema,
        )


@pytest.mark.parametrize(
    "dataset_name, dataset",
    [
        ("reference", {123: [1, 2, 3]}),
        ("current", {123: [1, 2, 3]}),
    ],
)
def test_input_validation_rejects_non_string_feature_keys_in_dataset(
    dataset_name: str, dataset: dict[object, list[int]]
) -> None:
    kwargs = {
        "reference": _get_valid_reference(),
        "current": _get_valid_current(),
        "schema": _get_valid_schema(),
    }
    kwargs[dataset_name] = dataset

    with pytest.raises(TypeError, match="Feature names must be strings"):
        validate_inputs(**kwargs)


@pytest.mark.parametrize(
    "dataset_name, dataset",
    [
        ("reference", {"": [1, 2, 3]}),
        ("current", {"": [1, 2, 3]}),
    ],
)
def test_input_validation_rejects_empty_string_feature_keys_in_dataset(
    dataset_name: str, dataset: dict[str, list[int]]
) -> None:
    kwargs = {
        "reference": _get_valid_reference(),
        "current": _get_valid_current(),
        "schema": _get_valid_schema(),
    }
    kwargs[dataset_name] = dataset

    with pytest.raises(ValueError, match="Feature names cannot be empty"):
        validate_inputs(**kwargs)


@pytest.mark.parametrize(
    "dataset_name, dataset",
    [
        ("reference", {"age": None}),
        ("current", {"age": None}),
    ],
)
def test_input_validation_rejects_none_values_in_dataset(
    dataset_name: str, dataset: dict[str, None]
) -> None:
    kwargs = {
        "reference": _get_valid_reference(),
        "current": _get_valid_current(),
        "schema": _get_valid_schema(),
    }
    kwargs[dataset_name] = dataset

    with pytest.raises(TypeError, match="Values cannot be None"):
        validate_inputs(**kwargs)


@pytest.mark.parametrize(
    "dataset_name, dataset",
    [
        ("reference", {"age": object()}),
        ("current", {"age": object()}),
    ],
)
def test_input_validation_rejects_non_sequence_values_in_dataset(
    dataset_name: str, dataset: dict[str, object]
) -> None:
    kwargs = {
        "reference": _get_valid_reference(),
        "current": _get_valid_current(),
        "schema": _get_valid_schema(),
    }
    kwargs[dataset_name] = dataset

    with pytest.raises(TypeError, match="Expected sequence-like data"):
        validate_inputs(**kwargs)
