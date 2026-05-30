# We are actually doing input validation in this module so we disable the below checks.
# pyright: reportUnnecessaryIsInstance=false
# pyright: reportUnnecessaryComparison=false
from collections.abc import Mapping

from numpy.typing import ArrayLike

from mldebug.domain.feature_type import FeatureType
from mldebug.validation import (
    validate_feature_dataset,
    validate_mapping,
    validate_schema,
)


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
    - dataset values are non-null, one-dimensional, and array-like

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
    validate_mapping(
        name="reference", value=reference, expected_desc="array-like values"
    )
    validate_mapping(name="current", value=current, expected_desc="array-like values")
    validate_mapping(
        name="schema", value=schema, expected_desc=f"'{FeatureType.__name__}' values"
    )

    validate_schema(schema=schema)

    validate_feature_dataset(name="reference", dataset=reference)
    validate_feature_dataset(name="current", dataset=current)
