import numpy as np


def make_normal_data(n=1000, mean=0, std=1, seed=42):
    rng = np.random.default_rng(seed)
    return rng.normal(loc=mean, scale=std, size=n)


def shifted_normal_data(n=1000, mean=2, std=1):
    return np.random.normal(mean, std, n)


def missing_data(base, rate=0.1):
    data = base.copy()
    mask = np.random.rand(len(data)) < rate
    data[mask] = np.nan
    return data
