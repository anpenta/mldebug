from typing import Protocol

from mldebug.domain.issue import Issue
from mldebug.runtime.feature_context import FeatureContext


class Check(Protocol):
    def __call__(self, context: FeatureContext) -> Issue | None: ...
