from typing import TYPE_CHECKING, cast

import numpy as np

from mldebug.core.models.context import FeatureContext
from mldebug.core.models.issue import Issue, Severity

if TYPE_CHECKING:
    from numpy.typing import NDArray


def run_numeric_missing_value_check(context: FeatureContext) -> Issue | None:
    """Detect increase in missing values for a numeric feature.

    This check compares the proportion of missing values (NaNs) between the reference and current
    data. An issue is reported when the increase in missing rate exceeds the configured threshold.

    Parameters
    ----------
    context : FeatureContext
        Execution context for the feature check.

    Returns
    -------
    Issue | None
        Issue if the increase in missing rate exceeds the configured threshold, otherwise None.

    """
    reference = cast("NDArray[np.floating]", context.reference)
    current = cast("NDArray[np.floating]", context.current)
    feature = context.feature
    threshold = context.config.missing_threshold

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
