"""mldebug

A lightweight Python package for comparing datasets and detecting unexpected changes in machine learning systems.

Provides tools to run validation checks on reference and current datasets and return structured reports of detected
issues.
"""

from .api import list_checks, run_checks
from .core.models import Issue, Report, Severity

__all__ = [
    "Issue",
    "Report",
    "Severity",
    "list_checks",
    "run_checks",
]
