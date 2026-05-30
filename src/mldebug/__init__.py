"""mldebug

A lightweight Python package for validating and comparing datasets in machine learning
pipelines.

Provides tools to run validation checks on reference and current datasets and return
reports of detected issues.
"""

import importlib.metadata as _importlib_metadata

from .domain.feature_type import FeatureType
from .domain.issue import Issue, Severity
from .domain.report import Report
from .pipeline.runner import validate

try:
    __version__ = _importlib_metadata.version("mldebug")
except _importlib_metadata.PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "Issue",
    "Report",
    "Severity",
    "FeatureType",
    "validate",
    "__version__",
]
