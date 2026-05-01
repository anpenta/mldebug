"""Core data models for mldebug.

Defines the fundamental data structures used across the system, including issues, reports, and severity levels.
"""

from .issue import Issue, Severity
from .report import Report

__all__ = [
    "Issue",
    "Report",
    "Severity",
]
