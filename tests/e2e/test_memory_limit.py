import tracemalloc

from mldebug import run_checks
from tests.fixtures.tabular_dataset import generate_mixed_tabular_dataset


def test_run_checks_memory_usage() -> None:
    reference, current, schema = generate_mixed_tabular_dataset(n=50_000, n_features=100)

    tracemalloc.start()

    run_checks(reference=reference, current=current, schema=schema)

    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Convert to MiB (binary units).
    peak_mib = peak_bytes / (1024 * 1024)

    # Loose sanity threshold.
    assert peak_mib < 400, f"Memory usage too high: {peak_mib:.2f} MiB"
