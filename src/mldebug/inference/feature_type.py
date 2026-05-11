from numpy.typing import ArrayLike

from mldebug.domain.feature_type import FeatureType
from mldebug.preprocessing.heuristics import compute_numeric_ratio, compute_unique_ratio


def infer_feature_type(values: ArrayLike) -> FeatureType:
    """Infer feature type from raw values."""
    numeric_ratio = compute_numeric_ratio(values)
    unique_ratio = compute_unique_ratio(values)

    # Empty / invalid fallback.
    if numeric_ratio == 0.0 and unique_ratio == 0.0:
        return FeatureType.CATEGORICAL

    # Constant / near-constant columns.
    if unique_ratio <= 0.01:
        return FeatureType.CATEGORICAL

    # Strong numeric signal.
    if numeric_ratio >= 0.95:
        return FeatureType.NUMERIC

    # Strong categorical signal.
    if numeric_ratio <= 0.2:
        return FeatureType.CATEGORICAL

    # Ambiguous region. Bias by structure.
    if numeric_ratio > unique_ratio:
        return FeatureType.NUMERIC

    return FeatureType.CATEGORICAL
