# pyright: reportUnnecessaryIsInstance=false
from collections.abc import Mapping

from numpy.typing import ArrayLike

from mldebug.domain.feature_type import FeatureType
from mldebug.validation import validate_feature_dataset, validate_mapping

from .feature_type import infer_feature_type


def infer_schema(dataset: Mapping[str, ArrayLike]) -> dict[str, FeatureType]:
    """Infer a feature schema from raw dataset values.

    Parameters
    ----------
    dataset : Mapping[str, ArrayLike]
        Mapping from feature names to array-like values.

    Returns
    -------
    dict[str, FeatureType]
        Inferred schema mapping feature names to inferred feature types.
    """
    validate_mapping(
        name="dataset",
        value=dataset,
        expected_desc="array-like values",
    )

    validate_feature_dataset(name="dataset", dataset=dataset)

    inferred_schema: dict[str, FeatureType] = {}
    for feature, values in dataset.items():
        inferred_schema[feature] = infer_feature_type(values)

    return inferred_schema
