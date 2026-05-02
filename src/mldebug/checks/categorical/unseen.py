from mldebug.core.models.context import CategoricalFeatureContext
from mldebug.core.models.issue import Issue, Severity


def run_categorical_unseen_category_check(context: CategoricalFeatureContext) -> Issue | None:
    """Detect unseen categories in a categorical feature.

    This check identifies values that appear in the current data but were not observed in the reference data.
    An issue is reported when at least one unseen category is detected.

    Parameters
    ----------
    context : CategoricalFeatureContext
        Execution context for the feature check.

    Returns
    -------
    Issue | None
        Issue if unseen categories are detected, otherwise None.

    """
    reference = context.reference
    current = context.current
    feature = context.feature

    ref_set = set(reference)
    cur_set = set(current)

    unseen = cur_set - ref_set
    unseen_count = len(unseen)

    if unseen_count == 0:
        return None

    return Issue(
        name="unseen_categories",
        metric="unseen_category_count",
        severity=Severity.WARNING,
        message=(f"{feature}: {unseen_count} unseen categories detected (e.g. {list(unseen)[:3]})"),
        feature=feature,
        value=float(unseen_count),
        threshold=0.0,
    )
