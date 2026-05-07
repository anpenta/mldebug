import numpy as np

from mldebug.models.context import FeatureContext
from mldebug.models.issue import Issue, Severity


def run_numeric_range_anomaly_check(context: FeatureContext) -> Issue | None:
    """Detect values outside the observed reference range for a numeric feature.

    This check compares current data against the minimum and maximum values observed in the reference
    data. An issue is reported when one or more values fall outside this range.

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
        message=(f"{context.feature}: {count} values outside [{ref_min:.4f}, {ref_max:.4f}]"),
        feature=context.feature,
        value=float(count),
        threshold=0.0,
    )
