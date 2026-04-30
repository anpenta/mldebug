"""mldebug: A lightweight Python package for comparing datasets and detecting unexpected changes in machine learning systems.

It compares a reference and current dataset, runs validation checks, and produces a structured report of issues.

Results are standardized as `Issue` objects aggregated into a `Report`.
"""  # noqa: E501 # First line is the package's headline.

from mldebug.api import list_checks, run_checks
from mldebug.core.issue import Issue, Severity
from mldebug.core.report import Report

__all__ = ["Issue", "Report", "Severity", "list_checks", "run_checks"]
