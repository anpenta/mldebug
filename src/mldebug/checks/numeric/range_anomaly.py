from dataclasses import dataclass

import numpy as np

from mldebug.domain.issue import Issue, Severity
from mldebug.runtime.feature_context import FeatureContext


@dataclass(frozen=True, slots=True)
class NumericRangeAnomalyCheck:
    """Detect values outside the observed reference range for a numeric feature.

    This check compares current data against the minimum and maximum values observed in the reference data.
    An issue is reported when one or more values fall outside this range.

    Parameters
    ----------
    """

    def __call__(self, context: FeatureContext) -> Issue | None:
        """Run numeric range anomaly detection.

        Parameters
        ----------
        context : FeatureContext
            Execution context for the feature check.

        Returns
        -------
        Issue | None
            Issue if any values are outside the reference range, otherwise None.

        """
        ref = context.reference
        cur = context.current
        feature = context.feature

        ref_min = ref.min()
        ref_max = ref.max()

        out_of_range = (cur < ref_min) | (cur > ref_max)
        count = int(np.sum(out_of_range))

        if count == 0:
            return None

        return Issue(
            name="range_anomaly",
            metric="out_of_range_count",
            severity=Severity.WARNING,
            message=(f"{feature}: {count} values outside [{ref_min:.4f}, {ref_max:.4f}]"),
            feature=feature,
            value=float(count),
            threshold=0.0,
        )
