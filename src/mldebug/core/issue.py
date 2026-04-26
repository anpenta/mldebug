from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Severity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True, slots=True)
class Issue:
    """Core unit of any detected ML problem.

    This is the atomic output of all checks.
    """

    name: str
    metric: str
    severity: Severity

    message: str

    feature: str | None = None
    value: float | None = None
    threshold: float | None = None
