import numpy as np

from mldebug.utils.stats import compute_safe_histogram, normalize_distribution


def test_compute_safe_histogram_normal_case():
    bins = 3
    values = np.array([1, 2, 2, 3, 3, 3])

    hist, edges = compute_safe_histogram(values, bins=bins)

    assert len(hist) == bins
    assert len(edges) == bins + 1
    assert np.sum(hist) == len(values)


def test_compute_safe_histogram_empty():
    bins = 5
    values = np.array([])

    hist, edges = compute_safe_histogram(values, bins=bins)

    assert np.allclose(hist, 0)
    assert len(hist) == bins
    assert len(edges) == bins + 1


def test_compute_safe_histogram_constant():
    bins = 4
    values = np.array([5, 5, 5, 5])

    hist, edges = compute_safe_histogram(values, bins=bins)

    assert np.sum(hist) == len(values)
    assert np.count_nonzero(hist) == 1
    assert np.max(hist) == len(values)
    assert len(edges) == bins + 1


def test_normalize_distribution_basic():
    hist = np.array([1, 1, 2])

    dist = normalize_distribution(hist)

    assert np.isclose(np.sum(dist), 1.0)
    assert np.isclose(dist[0], 0.25)
    assert np.isclose(dist[1], 0.25)
    assert np.isclose(dist[2], 0.5)


def test_normalize_distribution_stability():
    hist = np.array([0, 0, 1])

    dist = normalize_distribution(hist)

    assert np.all(np.isfinite(dist))
    assert np.isclose(np.sum(dist), 1.0)


def test_normalize_distribution_all_zeroes():
    hist = np.array([0, 0, 0])

    dist = normalize_distribution(hist)

    assert np.allclose(dist, 0)
