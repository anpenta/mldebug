import numpy as np

from mldebug.checks.categorical.unseen_values import CategoricalUnseenCategoryCheck
from mldebug.runtime.feature_context import FeatureContext


def test_unseen_category_check_triggers_when_new_categories_appear() -> None:
    context = FeatureContext(feature="color", reference=np.array(["red", "blue"]), current=np.array(["red", "green"]))

    issue = CategoricalUnseenCategoryCheck()(context)

    assert issue is not None
    assert issue.name == "unseen_categories"
    assert issue.metric == "unseen_category_count"
    assert issue.value == 1


def test_unseen_category_check_does_not_trigger_when_categories_are_known() -> None:
    context = FeatureContext(feature="color", reference=np.array(["red", "blue"]), current=np.array(["red", "blue"]))
    issue = CategoricalUnseenCategoryCheck()(context)

    assert issue is None


def test_unseen_category_check_counts_multiple_new_categories_correctly() -> None:
    context = FeatureContext(feature="color", reference=np.array(["red"]), current=np.array(["blue", "green"]))

    issue = CategoricalUnseenCategoryCheck()(context)

    assert issue is not None
    assert issue.value == 2


def test_unseen_category_check_counts_unique_categories_not_occurrences() -> None:
    context = FeatureContext(
        feature="color", reference=np.array(["red"]), current=np.array(["green", "green", "green"])
    )

    issue = CategoricalUnseenCategoryCheck()(context)

    assert issue is not None
    # Should count unique unseen categories, not occurrences.
    assert issue.value == 1


def test_unseen_category_check_returns_none_for_empty_current() -> None:
    context = FeatureContext(feature="color", reference=np.array(["red"]), current=np.array([]))

    issue = CategoricalUnseenCategoryCheck()(context)

    assert issue is None
