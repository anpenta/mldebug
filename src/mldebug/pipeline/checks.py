from collections.abc import Callable

from mldebug.models.context import CategoricalFeatureContext, NumericFeatureContext
from mldebug.models.issue import Issue


def run_check_group(
    checks: list[Callable[..., Issue | None]], context: NumericFeatureContext | CategoricalFeatureContext
) -> list[Issue]:
    issues: list[Issue] = []

    for check_fn in checks:
        issue = check_fn(context)

        if issue:
            issues.append(issue)

    return issues
