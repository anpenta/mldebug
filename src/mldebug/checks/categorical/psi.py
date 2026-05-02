import numpy as np
from numpy.typing import NDArray

from mldebug.core.models.context import CategoricalFeatureContext
from mldebug.core.models.issue import Issue, Severity


def run_categorical_psi_drift_check(context: CategoricalFeatureContext) -> Issue | None:
    """Detect categorical distribution drift using Population Stability Index (PSI).

    This check compares the distribution of categorical values between reference and current data using PSI.
    An issue is reported when the PSI value exceeds the configured threshold.

    Parameters
    ----------
    context : CategoricalFeatureContext
        Execution context for the feature check.

    Returns
    -------
    Issue | None
        Issue if the PSI value exceeds the configured threshold, otherwise None.

    """
    reference = context.reference
    current = context.current
    feature = context.feature
    threshold = context.config.psi_threshold

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
