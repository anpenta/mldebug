import gc
import tracemalloc

from mldebug import run_checks
from tests.fixtures.data.datasets import generate_mixed_tabular_dataset


def test_run_checks_memory() -> None:
    reference, current, schema = generate_mixed_tabular_dataset(n=10_000, n_features=50)

    gc.collect()

    tracemalloc.start()
    try:
        for _ in range(2):
            run_checks(reference=reference, current=current, schema=schema)
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    peak_mib = peak_bytes / (1024 * 1024)

    assert peak_mib < 150, f"Memory usage too high: {peak_mib:.2f} MiB"
