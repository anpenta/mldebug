from dataclasses import dataclass

from mldebug.protocols.check import Check
from mldebug.protocols.feature_detector import FeatureDetector
from mldebug.protocols.normalizer import Normalizer


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    detector: FeatureDetector
    normalizer: Normalizer
    checks: list[Check]
