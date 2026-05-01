import numpy as np
from numpy.typing import NDArray

from mldebug.core.models.issue import Issue, Severity


def run_categorical_missing_value_check(
    feature: str,
    reference: NDArray[np.str_],
    current: NDArray[np.str_],
    threshold: float = 0.1,
) -> Issue | None:
    """Detect increase in missing values for a categorical feature.

    Missing values are assumed to have been normalized during preprocessing
    and are represented as the empty string "".

    Parameters
    ----------
    feature : str
        Name of the feature being checked.

    reference : NDArray[np.str_]
        Reference data (baseline categorical values).

    current : NDArray[np.str_]
        Current data to evaluate.

    threshold : float
        Maximum allowed increase in missing rate.

    Returns
    -------
    Issue | None
        Issue if missing rate increase exceeds threshold.

    """
    ref_missing = (reference == "").mean()
    cur_missing = (current == "").mean()

    delta = cur_missing - ref_missing

    if delta > threshold:
        return Issue(
            name="missing_values",
            metric="missing_rate_increase",
            severity=Severity.WARNING,
            message=f"{feature}: missing rate drift detected ({delta:.4f})",
            feature=feature,
            value=float(delta),
            threshold=threshold,
        )

    return None
