import numpy as np
from numpy.typing import NDArray


def generate_normal_data(n: int = 1000, mean: float = 0, std: float = 1, seed: int = 42):
    rng = np.random.default_rng(seed)
    return rng.normal(loc=mean, scale=std, size=n)


def inject_missing_values(base: NDArray[np.float64], rate: float = 0.1, seed: int = 42) -> NDArray[np.float64]:
    data = base.copy()
    rng = np.random.default_rng(seed)
    mask = rng.random(len(data)) < rate
    data[mask] = np.nan
    return data
