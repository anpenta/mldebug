from pytest_benchmark.fixture import BenchmarkFixture

from mldebug import run_checks
from tests.datasets.tabular import generate_mixed_tabular_dataset


def test_run_checks_benchmark(benchmark: BenchmarkFixture) -> None:
    reference, current, schema = generate_mixed_tabular_dataset(n=50_000, n_features=100)

    benchmark(run_checks, reference, current, schema)
