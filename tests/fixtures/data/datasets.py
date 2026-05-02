from typing import Literal

import numpy as np
from numpy.typing import NDArray

from tests.fixtures.data.generators import (
    generate_categorical_data,
    generate_normal_data,
)


def generate_mixed_tabular_dataset(
    n: int = 100_000,
    n_features: int = 100,
    seed: int = 42,
) -> tuple[
    dict[str, NDArray[np.floating | np.str_]],
    dict[str, NDArray[np.floating | np.str_]],
    dict[str, Literal["numeric", "categorical"]],
]:
    n_numeric = n_features // 2
    n_categorical = n_features - n_numeric

    reference = {}
    current = {}
    schema = {}

    for i in range(n_numeric):
        name = f"num_{i}"
        reference[name] = generate_normal_data(n=n, seed=seed + i)
        current[name] = generate_normal_data(n=n, seed=seed + i + n_features)
        schema[name] = "numeric"

    for i in range(n_categorical):
        name = f"cat_{i}"
        reference[name] = generate_categorical_data(n=n, seed=seed + i)
        current[name] = generate_categorical_data(n=n, seed=seed + i + n_features)
        schema[name] = "categorical"

    return reference, current, schema
