from __future__ import annotations

from abc import ABC, abstractmethod

from mldebug.core.issue import Issue


class BaseCheck(ABC):
    """Abstract base class for all ML debug checks.

    Every check must:
    - accept (reference, current)
    - return Issue or None
    """

    name: str

    def __init__(self, name: str | None = None):
        # allows override but keeps default stable identity
        if name is not None:
            self.name = name

    @abstractmethod
    def run(self, reference, current) -> Issue | None:
        """Execute check.

        Returns:
            Issue if problem detected, else None

        """
        raise NotImplementedError
