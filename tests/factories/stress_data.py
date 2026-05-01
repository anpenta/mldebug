from typing import Literal

import numpy as np
from numpy.typing import NDArray

from tests.factories.data import (
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


def generate_numeric_tabular_dataset(
    n: int = 100_000,
    n_features: int = 100,
    seed: int = 42,
) -> tuple[
    dict[str, NDArray[np.floating]],
    dict[str, NDArray[np.floating]],
    dict[str, Literal["numeric"]],
]:
    reference = {f"num_{i}": generate_normal_data(n=n, seed=seed + i) for i in range(n_features)}
    current = {f"num_{i}": generate_normal_data(n=n, seed=seed + i + n_features) for i in range(n_features)}
    schema = {f"num_{i}": "numeric" for i in range(n_features)}
    return reference, current, schema


def generate_categorical_tabular_dataset(
    n: int = 100_000,
    n_features: int = 100,
    seed: int = 42,
) -> tuple[
    dict[str, NDArray[np.str_]],
    dict[str, NDArray[np.str_]],
    dict[str, Literal["categorical"]],
]:
    reference = {f"cat_{i}": generate_categorical_data(n=n, seed=seed + i) for i in range(n_features)}
    current = {f"cat_{i}": generate_categorical_data(n=n, seed=seed + i + n_features) for i in range(n_features)}
    schema = {f"cat_{i}": "categorical" for i in range(n_features)}
    return reference, current, schema
