"""Public API layer for mldebug.

Exposes user-facing functions for running validation checks.
"""

from mldebug.pipeline.runner import run_checks

__all__ = [
    "run_checks",
]
