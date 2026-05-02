import gc
import tracemalloc

from mldebug import run_checks
from tests.fixtures.data.datasets import generate_mixed_tabular_dataset


def test_run_checks_memory() -> None:
    reference, current, schema = generate_mixed_tabular_dataset(
        n=50_000,
        n_features=100,
    )

    gc.collect()

    # Warmup run to reduce cold-start noise.
    run_checks(reference=reference, current=current, schema=schema)

    tracemalloc.start()
    try:
        run_checks(reference=reference, current=current, schema=schema)
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    peak_mib = peak_bytes / (1024 * 1024)

    assert peak_mib < 400, f"Memory usage too high: {peak_mib:.2f} MiB"
