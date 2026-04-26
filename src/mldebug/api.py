from __future__ import annotations

from mldebug.checks.base import BaseCheck
from mldebug.checks.data_quality.missing_values import MissingValueCheck
from mldebug.checks.drift.detector import DriftDetector
from mldebug.checks.drift.ks import KSTestCheck
from mldebug.checks.drift.psi import PSIDriftCheck
from mldebug.core.report import Report


def default_checks() -> list[BaseCheck]:
    """Default ML debugging configuration for v1.

    Designed to cover:
    - distribution shift (PSI)
    - statistical shift (KS test)
    - data quality degradation (missing values)
    """
    return [
        PSIDriftCheck(threshold=0.2),
        KSTestCheck(alpha=0.05),
        MissingValueCheck(threshold=0.1),
    ]


def detect_drift(
    reference,
    current,
    checks: list[BaseCheck] | None = None,
) -> Report:
    """Run ML drift detection between reference and current datasets.

    This is the main entry point for mldebug v1.

    Parameters
    ----------
    reference :
        Baseline dataset (training / historical data)
    current :
        New dataset (production / inference data)
    checks :
        Optional custom list of checks. If None, uses defaults.

    Returns
    -------
    Report
        Aggregated drift report with all detected issues.

    """
    detector = DriftDetector(checks=checks or default_checks())

    return detector.run(reference, current)


def run_checks(
    reference,
    current,
    checks: list[BaseCheck],
) -> Report:
    """Run a custom set of checks explicitly.

    This bypasses default configuration.
    """
    detector = DriftDetector(checks=checks)
    return detector.run(reference, current)


def generate_report(issues) -> Report:
    """Build a Report object from raw issues.

    Useful for testing or custom pipelines.
    """
    return Report(issues)
