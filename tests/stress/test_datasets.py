from collections.abc import Callable
from typing import Literal

import numpy as np
import pytest
from numpy.typing import NDArray

from tests.stress.datasets import (
    generate_categorical_tabular_dataset,
    generate_mixed_tabular_dataset,
    generate_numeric_tabular_dataset,
)


@pytest.mark.parametrize(
    ("dataset_generator_function", "expected_schema_values"),
    [
        (generate_mixed_tabular_dataset, {"numeric", "categorical"}),
        (generate_numeric_tabular_dataset, {"numeric"}),
        (generate_categorical_tabular_dataset, {"categorical"}),
    ],
)
def test_dataset_structure(
    dataset_generator_function: Callable[
        ...,
        tuple[
            dict[str, NDArray[np.floating | np.str_]],
            dict[str, NDArray[np.floating | np.str_]],
            dict[str, Literal["numeric", "categorical"]],
        ],
    ],
    expected_schema_values: set[Literal["numeric", "categorical"]],
) -> None:
    n = 100
    n_features = 10

    reference, current, schema = dataset_generator_function(n=n, n_features=n_features)

    assert len(reference) == n_features
    assert len(current) == n_features
    assert len(schema) == n_features

    assert reference.keys() == current.keys() == schema.keys()

    assert set(schema.values()) == expected_schema_values

    for k in reference:
        assert len(reference[k]) == n
        assert len(current[k]) == n
