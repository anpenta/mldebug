from mldebug.domain.feature_type import FeatureType
from mldebug.types import Array
from tests.fixtures.generators import generate_categorical_data, generate_normal_data


def generate_mixed_tabular_dataset(
    n: int = 100_000, n_features: int = 100, seed: int = 42
) -> tuple[dict[str, Array], dict[str, Array], dict[str, FeatureType]]:
    n_numeric = n_features // 2
    n_categorical = n_features - n_numeric

    reference = {}
    current = {}
    schema = {}

    for i in range(n_numeric):
        name = f"num_{i}"
        reference[name] = generate_normal_data(n=n, seed=seed + i)
        current[name] = generate_normal_data(n=n, seed=seed + i + n_features)
        schema[name] = FeatureType.NUMERIC

    for i in range(n_categorical):
        name = f"cat_{i}"
        reference[name] = generate_categorical_data(n=n, seed=seed + i)
        current[name] = generate_categorical_data(n=n, seed=seed + i + n_features)
        schema[name] = FeatureType.CATEGORICAL

    return reference, current, schema
