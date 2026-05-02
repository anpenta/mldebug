from tests.fixtures.data.generators import generate_categorical_data


def test_generated_categorical_data_contains_only_valid_categories() -> None:
    data = generate_categorical_data(n=1000)

    assert set(data).issubset({"A", "B", "C", "D", "E", "F", "G", "H"})
