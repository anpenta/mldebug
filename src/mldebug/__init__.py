"""mldebug

A lightweight Python package for comparing datasets and detecting unexpected changes in machine learning systems.

Provides tools to run validation checks on reference and current datasets and return reports of detected issues.
"""

from .models.feature_type import FeatureType
from .models.issue import Issue, Severity
from .models.report import Report
from .pipeline.runner import run_checks

__all__ = [
    "Issue",
    "Report",
    "Severity",
    "FeatureType",
    "run_checks",
]
