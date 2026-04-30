import numpy as np
from numpy.typing import NDArray


def generate_normal_data(n: int = 1000, mean: float = 0, std: float = 1, seed: int = 42) -> NDArray[np.floating]:
    rng = np.random.default_rng(seed)
    return rng.normal(loc=mean, scale=std, size=n)


def inject_numeric_missing_values(data: NDArray[np.float64], rate: float = 0.1, seed: int = 42) -> NDArray[np.floating]:
    data = data.copy()
    rng = np.random.default_rng(seed)
    mask = rng.random(len(data)) < rate
    data[mask] = np.nan
    return data


def generate_categorical_data(n: int = 1000, seed: int = 42) -> NDArray[np.str_]:
    rng = np.random.default_rng(seed)
    return rng.choice(["A", "B", "C", "D"], size=n)


def inject_categorical_missing_values(data: NDArray[np.str_], rate: float = 0.1, seed: int = 42) -> NDArray[np.str_]:
    data = data.copy()
    rng = np.random.default_rng(seed)
    mask = rng.random(len(data)) < rate
    data[mask] = ""
    return data
