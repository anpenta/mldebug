from mldebug.checks.categorical.missing_values import (
    CategoricalMissingValueCheck,
)
from mldebug.checks.categorical.psi import (
    CategoricalPSICheck,
)
from mldebug.checks.categorical.unseen_categories import (
    CategoricalUnseenCategoryCheck,
)
from mldebug.checks.numeric.ks_test import (
    NumericKSTestCheck,
)
from mldebug.checks.numeric.missing_values import (
    NumericMissingValueCheck,
)
from mldebug.checks.numeric.range_anomaly import (
    NumericRangeAnomalyCheck,
)
from mldebug.checks.numeric.variance_drift import (
    NumericVarianceDriftCheck,
)
from mldebug.detectors.categorical import (
    CategoricalFeatureDetector,
)
from mldebug.detectors.numeric import (
    NumericFeatureDetector,
)
from mldebug.domain.feature_type import FeatureType
from mldebug.normalizers.categorical import (
    CategoricalNormalizer,
)
from mldebug.normalizers.numeric import (
    NumericNormalizer,
)
from mldebug.runtime.feature_spec import FeatureSpec

FEATURE_SPECS: dict[FeatureType, FeatureSpec] = {
    FeatureType.NUMERIC: FeatureSpec(
        detector=NumericFeatureDetector(),
        normalizer=NumericNormalizer(),
        checks=[
            NumericMissingValueCheck(),
            NumericKSTestCheck(),
            NumericVarianceDriftCheck(),
            NumericRangeAnomalyCheck(),
        ],
    ),
    FeatureType.CATEGORICAL: FeatureSpec(
        detector=CategoricalFeatureDetector(),
        normalizer=CategoricalNormalizer(),
        checks=[
            CategoricalMissingValueCheck(),
            CategoricalPSICheck(),
            CategoricalUnseenCategoryCheck(),
        ],
    ),
}
