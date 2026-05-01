import numpy as np
from numpy.typing import NDArray
from scipy.stats import ks_2samp

from mldebug.core.models.issue import Issue, Severity


def run_numeric_ks_test_check(
    feature: str,
    reference: NDArray[np.floating],
    current: NDArray[np.floating],
    alpha: float = 0.05,
) -> Issue | None:
    """Detect numeric distribution shift using the Kolmogorov-Smirnov test.

    The KS test compares the empirical distributions of reference and current data for a given feature.

    Parameters
    ----------
    feature : str
        Name of the feature being checked.

    reference : NDArray[np.floating]
        Reference (baseline) data.

    current : NDArray[np.floating]
        Current data to evaluate.

    alpha : float = 0.05, optional
        Significance level for rejecting the null hypothesis.

    Returns
    -------
    Issue | None
        Issue if distribution shift is detected, otherwise None.

    """
    p_value = ks_2samp(reference, current).pvalue

    if p_value < alpha:
        return Issue(
            name="ks_test",
            metric="distribution_shift_score",
            severity=Severity.WARNING,
            message=f"{feature}: distribution shift detected (p={p_value:.6f})",
            feature=feature,
            value=float(p_value),
            threshold=alpha,
        )

    return None
