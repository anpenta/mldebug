from pytest_benchmark.fixture import BenchmarkFixture

from mldebug import run_checks
from tests.fixtures.data.datasets import generate_mixed_tabular_dataset


def test_run_checks_benchmark(benchmark: BenchmarkFixture) -> None:
    reference, current, schema = generate_mixed_tabular_dataset(n=50_000, n_features=100)

    benchmark(run_checks, reference, current, schema)

    assert benchmark.stats.stats.median < 5.0
    assert benchmark.stats.stats.max < 10.0
