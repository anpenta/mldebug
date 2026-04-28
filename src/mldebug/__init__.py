"""mldebug: A lightweight Python package for comparing datasets and detecting unexpected changes or issues.

The library runs a suite of checks on a reference and current dataset, producing structured reports of detected issues.

Outputs are standardized as `Issue` objects aggregated into a `Report`.
"""

from mldebug.api import run_checks
from mldebug.core.issue import Issue, Severity
from mldebug.core.report import Report

__all__ = [
    # Main API.
    "run_checks",
    # Core types.
    "Issue",
    "Report",
    "Severity",
]
