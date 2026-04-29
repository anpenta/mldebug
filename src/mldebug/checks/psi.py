from collections import Counter

import numpy as np
from numpy.typing import NDArray

from mldebug.core.issue import Issue, Severity


def run_psi_drift_check_categorical(
    feature: str,
    reference: NDArray[np.object_],
    current: NDArray[np.object_],
    threshold: float = 0.2,
) -> Issue | None:
    """Detect categorical distribution drift using PSI.

    Parameters
    ----------
    feature : str
        Name of the feature being checked.

    reference : NDArray[np.object_]
        Reference data (baseline categorical values).

    current : NDArray[np.object_]
        Current data to evaluate.

    threshold : float
        PSI threshold for detecting drift.

    Returns
    -------
    Issue | None
        Issue if PSI exceeds threshold.

    """
    psi = _compute_psi_categorical(reference, current)

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


def _compute_psi_categorical(
    reference: NDArray[np.object_],
    current: NDArray[np.object_],
    eps: float = 1e-8,
) -> float:
    ref_counts = Counter(reference)
    cur_counts = Counter(current)

    all_categories = set(ref_counts) | set(cur_counts)

    ref_total = sum(ref_counts.values())
    cur_total = sum(cur_counts.values())

    psi = 0.0

    for cat in all_categories:
        p = ref_counts.get(cat, 0) / ref_total
        q = cur_counts.get(cat, 0) / cur_total

        p = max(p, eps)
        q = max(q, eps)

        psi += (p - q) * np.log(p / q)

    return float(psi)
