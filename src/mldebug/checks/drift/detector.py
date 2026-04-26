from __future__ import annotations

from mldebug.checks.base import BaseCheck
from mldebug.core.issue import Issue
from mldebug.core.report import Report


class DriftDetector:
    """Orchestrates execution of multiple ML checks
    and aggregates results into a Report.
    """

    def __init__(self, checks: list[BaseCheck] | None = None):
        self._checks: list[BaseCheck] = checks or []

    # -----------------------------
    # Check management
    # -----------------------------

    def add_check(self, check: BaseCheck) -> None:
        """Add a new check to the pipeline."""
        self._checks.append(check)

    def remove_check(self, name: str) -> None:
        """Remove check by name."""
        self._checks = [c for c in self._checks if c.name != name]

    def list_checks(self) -> list[str]:
        """Return active check names."""
        return [c.name for c in self._checks]

    def run(self, reference, current) -> Report:
        """Run all checks and return a structured Report."""
        issues: list[Issue] = []

        for check in self._checks:
            try:
                result = check.run(reference, current)

                if result is not None:
                    issues.append(result)

            except Exception as e:
                # IMPORTANT: fail-safe design (never break pipeline)
                issues.append(
                    Issue(
                        name=check.name,
                        metric="execution_error",
                        severity="critical",
                        message=f"Check failed: {e!s}",
                        value=None,
                        threshold=None,
                    )
                )

        return Report(issues)
