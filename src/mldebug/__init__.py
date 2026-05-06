"""mldebug

A lightweight Python package for comparing datasets and detecting unexpected changes in machine learning systems.

Provides tools to run validation checks on reference and current datasets and return structured reports of detected
issues.
"""

from .models import Issue, Report, Severity
from .pipeline.runner import run_checks

__all__ = [
    "Issue",
    "Report",
    "Severity",
    "run_checks",
]
