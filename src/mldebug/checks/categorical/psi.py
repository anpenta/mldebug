import numpy as np
from numpy.typing import NDArray

from mldebug.core.models.issue import Issue, Severity


def run_categorical_psi_drift_check(
    feature: str,
    reference: NDArray[np.str_],
    current: NDArray[np.str_],
    threshold: float = 0.2,
) -> Issue | None:
    """Detect categorical distribution drift using PSI.

    Parameters
    ----------
    feature : str
        Name of the feature being checked.

    reference : NDArray[np.str_]
        Reference data (baseline categorical values).

    current : NDArray[np.str_]
        Current data to evaluate.

    threshold : float
        PSI threshold for detecting drift.

    Returns
    -------
    Issue | None
        Issue if PSI exceeds threshold.

    """
    psi = _compute_categorical_psi(reference, current)

    if psi > threshold:
        return Issue(
            name="psi_drift",
            metric="psi",
            severity=Severity.WARNING,
            message=f"{feature}: PSI drift detected ({psi:.4f})",
            feature=feature,
            value=psi,
            threshold=threshold,
        )

    return None


def _compute_categorical_psi(
    reference: NDArray[np.str_],
    current: NDArray[np.str_],
    eps: float = 1e-8,
) -> float:
    # Build shared category space (union of all categories).
    all_values = np.concatenate([reference, current])
    _, encoded = np.unique(all_values, return_inverse=True)

    ref_encoded = encoded[: len(reference)]
    cur_encoded = encoded[len(reference) :]

    # Histogram counts aligned to the same category space.
    n_categories = encoded.max() + 1
    ref_counts = np.bincount(ref_encoded, minlength=n_categories)
    cur_counts = np.bincount(cur_encoded, minlength=n_categories)

    # Convert to probabilities.
    ref_total = ref_counts.sum()
    cur_total = cur_counts.sum()

    p = ref_counts / ref_total
    q = cur_counts / cur_total

    # Numerical stability.
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)

    # PSI computation (vectorized).
    psi = np.sum((p - q) * np.log(p / q))

    return float(psi)
