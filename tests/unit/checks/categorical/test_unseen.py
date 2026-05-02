import numpy as np

from mldebug.checks.categorical.unseen import run_categorical_unseen_category_check
from mldebug.core.config import CheckConfig
from mldebug.core.models.context import FeatureContext


def test_unseen_category_check_triggers_when_new_categories_appear() -> None:
    context = FeatureContext(
        feature="color",
        ftype="categorical",
        reference=np.array(["red", "blue"]),
        current=np.array(["red", "green"]),
        config=CheckConfig(),
    )

    issue = run_categorical_unseen_category_check(context)

    assert issue is not None
    assert issue.name == "unseen_categories"
    assert issue.metric == "unseen_category_count"
    assert issue.value == 1


def test_unseen_category_check_does_not_trigger_when_categories_are_known() -> None:
    context = FeatureContext(
        feature="color",
        ftype="categorical",
        reference=np.array(["red", "blue"]),
        current=np.array(["red", "blue"]),
        config=CheckConfig(),
    )
    issue = run_categorical_unseen_category_check(context)

    assert issue is None


def test_unseen_category_check_counts_multiple_new_categories_correctly() -> None:
    context = FeatureContext(
        feature="color",
        ftype="categorical",
        reference=np.array(["red"]),
        current=np.array(["blue", "green"]),
        config=CheckConfig(),
    )

    issue = run_categorical_unseen_category_check(context)

    assert issue is not None
    assert issue.value == 2


def test_unseen_category_check_counts_unique_categories_not_occurrences() -> None:
    context = FeatureContext(
        feature="color",
        ftype="categorical",
        reference=np.array(["red"]),
        current=np.array(["green", "green", "green"]),
        config=CheckConfig(),
    )

    issue = run_categorical_unseen_category_check(context)

    assert issue is not None
    # Should count unique unseen categories, not occurrences.
    assert issue.value == 1


def test_unseen_category_check_returns_none_for_empty_current() -> None:
    context = FeatureContext(
        feature="color", ftype="categorical", reference=np.array(["red"]), current=np.array([]), config=CheckConfig()
    )

    issue = run_categorical_unseen_category_check(context)

    assert issue is None
