"""mldebug: Lightweight ML debugging toolkit for detecting issues in tabular ML data.

The library runs a suite of statistical and data quality checks on a reference and current dataset,
producing structured reports of detected issues.

Core capabilities include:
- data drift detection
- missing value detection
- schema consistency checks

Outputs are standardized as `Issue` objects aggregated into a `Report`.

Notes
-----
- Designed for batch validation of tabular datasets.
- Operates on dictionary-of-arrays inputs (no pandas dependency).

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
