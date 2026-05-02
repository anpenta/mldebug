from tests.fixtures.data.datasets import (
    generate_mixed_tabular_dataset,
)


def test_mixed_tabular_dataset_structure() -> None:
    n = 100
    n_features = 10

    reference, current, schema = generate_mixed_tabular_dataset(n=n, n_features=n_features)

    assert len(reference) == n_features
    assert len(current) == n_features
    assert len(schema) == n_features

    assert reference.keys() == current.keys() == schema.keys()

    assert set(schema.values()) == {"numeric", "categorical"}

    for k in reference:
        assert len(reference[k]) == n
        assert len(current[k]) == n
