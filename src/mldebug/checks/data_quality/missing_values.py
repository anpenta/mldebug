from __future__ import annotations

import numpy as np

from mldebug.checks.base import BaseCheck
from mldebug.core.issue import Issue, Severity


class MissingValueCheck(BaseCheck):
    """Detects increase in missing values between datasets."""

    name = "missing_values"

    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold

    def run(self, reference, current):
        ref_missing = np.mean(np.isnan(reference))
        cur_missing = np.mean(np.isnan(current))

        delta = cur_missing - ref_missing

        if delta > self.threshold:
            return Issue(
                name=self.name,
                metric="missing_delta",
                severity=Severity.CRITICAL,
                message=f"Missing values increased by {delta:.4f}",
                value=delta,
                threshold=self.threshold,
            )

        return None
