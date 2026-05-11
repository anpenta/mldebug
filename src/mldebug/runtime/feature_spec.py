from dataclasses import dataclass

from mldebug.protocols.check import Check
from mldebug.protocols.normalizer import Normalizer


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    normalizer: Normalizer
    checks: list[Check]
