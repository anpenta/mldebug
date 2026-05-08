from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from mldebug.domain.issue import Issue, Severity
from mldebug.runtime.feature_context import FeatureContext


@dataclass(frozen=True, slots=True)
class CategoricalPSICheck:
    """Detect categorical distribution drift using Population Stability Index (PSI).

    This check compares the distribution of categorical values between reference and current data using PSI.
    An issue is reported when the PSI value exceeds the configured threshold.

    Parameters
    ----------
    threshold : float, default=0.2
        Maximum allowed PSI value before reporting drift.

    eps : float, default=1e-8
        Small value added for numerical stability in probability computation.

    """

    threshold: float = 0.2
    eps: float = 1e-8

    def __call__(self, context: FeatureContext) -> Issue | None:
        """Run PSI drift detection for categorical features.

        Parameters
        ----------
        context : FeatureContext
            Execution context for the feature check.

        Returns
        -------
        Issue | None
            Issue if PSI exceeds threshold, otherwise None.

        """
        reference = context.reference
        current = context.current
        feature = context.feature

        psi = self._compute_psi(reference, current)

        if psi > self.threshold:
            return Issue(
                name="psi_drift",
                metric="psi",
                severity=Severity.WARNING,
                message=f"{feature}: PSI drift detected ({psi:.4f})",
                feature=feature,
                value=psi,
                threshold=self.threshold,
            )

        return None

    def _compute_psi(self, reference: NDArray[np.str_], current: NDArray[np.str_]) -> float:
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
        p = np.clip(p, self.eps, None)
        q = np.clip(q, self.eps, None)

        # PSI computation (vectorized).
        psi = np.sum((p - q) * np.log(p / q))

        return float(psi)
