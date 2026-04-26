"""mldebug v1

Lightweight ML debugging toolkit for:
- data drift detection
- data quality checks
- statistical monitoring
"""

from mldebug.api import (
    detect_drift,
    generate_report,
    run_checks,
)
from mldebug.checks.base import BaseCheck
from mldebug.core.issue import Issue, Severity
from mldebug.core.report import Report

__all__ = [
    # main API
    "detect_drift",
    "run_checks",
    "generate_report",
    # core types
    "Issue",
    "Report",
    "Severity",
    # extension point
    "BaseCheck",
]
