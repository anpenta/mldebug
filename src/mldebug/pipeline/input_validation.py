# We are actually doing input validation in this module so we disable the below checks.
# pyright: reportUnnecessaryIsInstance=false
# pyright: reportUnnecessaryComparison=false
from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike

from mldebug.domain.feature_type import FeatureType


def validate_inputs(
    reference: Mapping[str, ArrayLike],
    current: Mapping[str, ArrayLike],
    schema: Mapping[str, FeatureType],
) -> None:
    """Validate input datasets and schema before running feature checks.

    This function ensures that the inputs to the pipeline are structurally valid and
    type-consistent. It acts as a strict precondition guard before schema analysis and
    feature-level checks are executed.

    It validates that:
    - reference and current are mappings of feature names to array-like values
    - schema is a mapping of feature names to FeatureType enums
    - feature names are non-empty strings
    - dataset values are non-null and array-like

    Parameters
    ----------
    reference : Mapping[str, ArrayLike]
        Reference dataset keyed by feature name (e.g. training data).

    current : Mapping[str, ArrayLike]
        Current dataset keyed by feature name (e.g. production data).

    schema : Mapping[str, FeatureType]
        Mapping of feature names to their expected feature types.

    Raises
    ------
    TypeError
        If inputs are not mappings, contain invalid feature types, or invalid dataset
        values.

    ValueError
        If feature names are empty strings.
    """
    _validate_mapping(
        name="reference", value=reference, expected_desc="array-like values"
    )
    _validate_mapping(name="current", value=current, expected_desc="array-like values")
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


def _validate_dataset(name: str, dataset: Mapping[str, ArrayLike]) -> None:
    for feature, values in dataset.items():
        if not isinstance(feature, str):
            raise TypeError(f"Invalid '{name}' dataset. Feature names must be strings.")

        if not feature.strip():
            raise ValueError(
                f"Invalid '{name}' dataset. Feature names cannot be empty."
            )

        if values is None:
            raise TypeError(
                f"Invalid values for feature '{feature}' in '{name}'. "
                "Values cannot be None."
            )

        try:
            np.asarray(values)
        except Exception as e:
            raise TypeError(
                f"Invalid values for feature '{feature}' in '{name}'. "
                "Expected array-like input compatible with numpy."
            ) from e
