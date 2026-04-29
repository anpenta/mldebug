import numpy as np
from numpy.typing import NDArray

from mldebug.core.issue import Issue, Severity


def run_missing_value_check(
    feature: str,
    reference: NDArray[np.floating],
    current: NDArray[np.floating],
    threshold: float = 0.1,
) -> Issue | None:
    """Detect increase in missing values for a single numeric feature.

    Parameters
    ----------
    feature : str
        Name of the feature being checked.

    reference : NDArray[np.floating]
        Reference data (baseline).

    current : NDArray[np.floating]
        Current data to evaluate.

    threshold : float
        Maximum allowed increase in missing rate.

    Returns
    -------
    Issue | None
        Issue if missing rate increase exceeds threshold.

    """
    ref_missing = np.mean(np.isnan(reference))
    cur_missing = np.mean(np.isnan(current))

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
