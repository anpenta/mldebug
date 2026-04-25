"""Example module demonstrating public and private function patterns.

This module shows how public APIs can use private helper functions within the mldebug package.
"""


def run_public_func() -> None:
    _run_private_func()


def _run_private_func() -> None:
    pass
