from pytest_benchmark.fixture import BenchmarkFixture

from mldebug import validate
from tests.fixtures.datasets import generate_mixed_tabular_dataset


def test_validation_pipeline_latency_is_bounded(benchmark: BenchmarkFixture) -> None:
    reference, current, schema = generate_mixed_tabular_dataset(n=1_000, n_features=10)

    benchmark(validate, reference, current, schema)

    assert benchmark.stats.stats.median < 0.15
    assert benchmark.stats.stats.mean < 0.2
