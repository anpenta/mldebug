import numpy as np
from numpy.typing import NDArray


def generate_normal_data(n: int = 1000, mean: float = 0, std: float = 1, seed: int = 42) -> NDArray[np.floating]:
    rng = np.random.default_rng(seed)
    return rng.normal(loc=mean, scale=std, size=n)


def generate_categorical_data(n: int = 1000, seed: int = 42) -> NDArray[np.str_]:
    rng = np.random.default_rng(seed)
    return rng.choice(["A", "B", "C", "D", "E", "F", "G", "H"], size=n)
