from dataclasses import dataclass

from mldebug.types import Array


@dataclass(frozen=True, slots=True)
class FeatureContext:
    feature: str
    reference: Array
    current: Array
