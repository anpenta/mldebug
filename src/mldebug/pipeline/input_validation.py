# We are actually doing input validation in this module so we disable the below checks.
# pyright: reportUnnecessaryIsInstance=false
# pyright: reportUnnecessaryComparison=false
from collections.abc import Mapping, Sequence
from typing import Any

from mldebug.domain.feature_type import FeatureType


def validate_inputs(
    reference: Mapping[str, Sequence[Any]],
    current: Mapping[str, Sequence[Any]],
    schema: Mapping[str, FeatureType],
) -> None:
    _validate_mapping(
        name="reference", value=reference, expected_desc="array-like sequences"
    )
    _validate_mapping(
        name="current", value=current, expected_desc="array-like sequences"
    )
    _validate_mapping(
        name="schema", value=schema, expected_desc=_get_enum_description(FeatureType)
    )

    _validate_schema(schema=schema)

    _validate_dataset(name="reference", dataset=reference)
    _validate_dataset(name="current", dataset=current)


def _validate_mapping(name: str, value: object, expected_desc: str) -> None:
    if not isinstance(value, Mapping):
        raise TypeError(
            f"Invalid '{name}' argument. "
            f"Expected a mapping of feature names to {expected_desc}."
        )


def _validate_schema(schema: Mapping[str, FeatureType]) -> None:
    for feature, ftype in schema.items():
        if not isinstance(feature, str):
            raise TypeError("Invalid schema. Feature names must be strings.")

        if not feature.strip():
            raise ValueError("Invalid schema. Feature names cannot be empty.")

        if not isinstance(ftype, FeatureType):
            raise TypeError(
                f"Invalid feature type for '{feature}'. "
                f"Expected {_get_enum_description(FeatureType)}."
            )


def _get_enum_description(enum_type: type[FeatureType]) -> str:
    values = ", ".join(e.name for e in enum_type)
    return f"{enum_type.__name__} ({values})"


def _validate_dataset(name: str, dataset: Mapping[str, Sequence[Any]]) -> None:
    for feature, values in dataset.items():
        if not isinstance(feature, str):
            raise TypeError(f"Invalid '{name}' dataset. Feature names must be strings.")

        if values is None:
            raise TypeError(
                f"Invalid values for feature '{feature}' in '{name}'. "
                "Values cannot be None."
            )

        if not isinstance(values, Sequence):
            raise TypeError(
                f"Invalid values for feature '{feature}' in '{name}'. "
                "Expected sequence-like data."
            )
