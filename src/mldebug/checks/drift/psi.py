from __future__ import annotations

import numpy as np

from mldebug.checks.base import BaseCheck
from mldebug.core.issue import Issue, Severity
from mldebug.utils.stats import normalize_distribution, safe_histogram


def psi_score(expected, actual, bins=10) -> float:
    """Population Stability Index (PSI)
    Measures distribution shift between two samples.
    """
    exp_hist, edges = safe_histogram(expected, bins=bins)
    act_hist, _ = safe_histogram(actual, bins=edges)

    exp_dist = normalize_distribution(exp_hist)
    act_dist = normalize_distribution(act_hist)

    eps = 1e-8
    exp_dist = np.clip(exp_dist, eps, 1)
    act_dist = np.clip(act_dist, eps, 1)

    psi = np.sum((act_dist - exp_dist) * np.log(act_dist / exp_dist))
    return float(psi)


class PSIDriftCheck(BaseCheck):
    """Detects distribution shift using PSI."""

    name = "psi_drift"

    def __init__(self, threshold: float = 0.2, bins: int = 10):
        self.threshold = threshold
        self.bins = bins

    def run(self, reference, current):
        # assume 1D numeric arrays for v1
        psi = psi_score(reference, current, bins=self.bins)

        if psi > self.threshold:
            return Issue(
                name=self.name,
                metric="psi",
                severity=Severity.WARNING,
                message=f"PSI drift detected (psi={psi:.4f})",
                value=psi,
                threshold=self.threshold,
            )

        return None
