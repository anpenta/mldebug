# We are actually doing input validation in this module so we disable the below checks.
# pyright: reportUnnecessaryIsInstance=false
# pyright: reportUnnecessaryComparison=false
from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike

from mldebug.domain.feature_type import FeatureType


def validate_mapping(name: str, value: object, expected_desc: str) -> None:
    """Validate that a value is a mapping for feature data or schema."""
    if not isinstance(value, Mapping):
        raise TypeError(
            f"Invalid '{name}' argument. "
            f"Expected a mapping of feature names to {expected_desc}."
        )


def validate_schema(schema: Mapping[str, FeatureType]) -> None:
    """Validate a schema mapping from feature names to feature type values."""
    for feature, ftype in schema.items():
        if not isinstance(feature, str):
            raise TypeError("Invalid schema. Feature names must be strings.")

        if not feature.strip():
            raise ValueError("Invalid schema. Feature names cannot be empty.")

        if not isinstance(ftype, FeatureType):
            raise TypeError(
                f"Invalid feature type for '{feature}'. "
                f"Expected '{FeatureType.__name__}'."
            )


def validate_feature_dataset(name: str, dataset: Mapping[str, ArrayLike]) -> None:
    """Validate a dataset mapping for feature-level values."""
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
            array = np.asarray(values)
        except Exception as exc:
            raise TypeError(
                f"Invalid values for feature '{feature}' in '{name}'. "
                "Expected array-like input compatible with numpy."
            ) from exc

        if array.ndim != 1:
            raise TypeError(
                f"Invalid values for feature '{feature}' in '{name}'. "
                "Feature values must be one-dimensional."
            )
