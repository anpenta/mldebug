"""mldebug

A lightweight Python package for validating and comparing datasets in machine learning
pipelines.

Provides tools to run validation checks on reference and current datasets and return
reports of detected issues.
"""

from .domain.feature_type import FeatureType
from .domain.issue import Issue, Severity
from .domain.report import Report
from .pipeline.runner import validate

__all__ = [
    "Issue",
    "Report",
    "Severity",
    "FeatureType",
    "validate",
]
