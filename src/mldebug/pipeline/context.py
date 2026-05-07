from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from mldebug.config import CategoricalCheckConfig, NumericCheckConfig
from mldebug.models.context import CategoricalFeatureContext, NumericFeatureContext


def build_feature_context(
    feature: str,
    ftype: Literal["numeric", "categorical"],
    normalized_reference: NDArray[np.generic],
    normalized_current: NDArray[np.generic],
) -> NumericFeatureContext | CategoricalFeatureContext:

    match ftype:
        case "numeric":
            return NumericFeatureContext(
                feature=feature,
                reference=cast(NDArray[np.floating], normalized_reference),
                current=cast(NDArray[np.floating], normalized_current),
                config=NumericCheckConfig(),
            )

        case "categorical":
            return CategoricalFeatureContext(
                feature=feature,
                reference=cast(NDArray[np.str_], normalized_reference),
                current=cast(NDArray[np.str_], normalized_current),
                config=CategoricalCheckConfig(),
            )

        case _:
            raise NotImplementedError()
