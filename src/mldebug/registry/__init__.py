"""
Registry of available data validation checks.

Defines and exposes the mapping between feature types and their associated checks.
"""

from .checks import CHECKS

__all__ = ["CHECKS"]
