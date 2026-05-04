import gc
import tracemalloc

from mldebug import run_checks
from tests.fixtures.data.datasets import generate_mixed_tabular_dataset


def test_check_pipeline_memory_is_bounded() -> None:
    reference, current, schema = generate_mixed_tabular_dataset(n=1_000, n_features=10)

    gc.collect()

    tracemalloc.start()
    try:
        for _ in range(2):
            run_checks(reference=reference, current=current, schema=schema)
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    peak_mib = peak_bytes / (1024 * 1024)

    assert peak_mib < 10, f"Memory usage too high: {peak_mib:.2f} MiB"
