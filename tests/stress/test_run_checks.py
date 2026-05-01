from collections.abc import Callable
from typing import Literal

import numpy as np
import pytest
from numpy.typing import NDArray

from mldebug import run_checks
from tests.datasets.tabular import (
    generate_categorical_tabular_dataset,
    generate_mixed_tabular_dataset,
    generate_numeric_tabular_dataset,
)


@pytest.mark.parametrize(
    "dataset_generation_function",
    [
        generate_mixed_tabular_dataset,
        generate_numeric_tabular_dataset,
        generate_categorical_tabular_dataset,
    ],
)
def test_run_checks_stress_large_datasets(
    dataset_generation_function: Callable[
        ...,
        tuple[
            dict[str, NDArray[np.floating | np.str_]],
            dict[str, NDArray[np.floating | np.str_]],
            dict[str, Literal["numeric", "categorical"]],
        ],
    ],
) -> None:
    reference, current, schema = dataset_generation_function(n=50_000, n_features=100)

    report = run_checks(reference=reference, current=current, schema=schema)

    assert report is not None
    assert isinstance(report.issues, list)
