"""mldebug

A lightweight Python package for comparing datasets and detecting unexpected changes in machine learning systems.

Provides tools to run validation checks on reference and current datasets and return reports of detected issues.
"""

from .domain.feature_type import FeatureType
from .domain.issue import Issue, Severity
from .domain.report import Report
from .pipeline.runner import run_checks

__all__ = [
    "Issue",
    "Report",
    "Severity",
    "FeatureType",
    "run_checks",
]
