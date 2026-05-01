"""
Core data models for mldebug.

Defines the fundamental data structures used to represent issues, reports, and severity
levels produced by validation checks.
"""

from .issue import Issue, Severity
from .report import Report

__all__ = [
    "Issue",
    "Report",
    "Severity",
]
