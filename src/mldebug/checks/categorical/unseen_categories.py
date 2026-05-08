from dataclasses import dataclass

from mldebug.domain.issue import Issue, Severity
from mldebug.runtime.feature_context import FeatureContext


@dataclass(frozen=True, slots=True)
class CategoricalUnseenCategoryCheck:
    """Detect unseen categories in a categorical feature.

    This check identifies values that appear in the current data
    but were not observed in the reference data. An issue is
    reported when at least one unseen category is detected.

    Parameters
    ----------
    max_examples : int, default=3
        Number of unseen categories to show in the message.

    """

    max_examples: int = 3

    def __call__(self, context: FeatureContext) -> Issue | None:
        """Run unseen category detection.

        Parameters
        ----------
        context : FeatureContext
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

        examples = list(map(str, list(unseen)[: self.max_examples]))

        return Issue(
            name="unseen_categories",
            metric="unseen_category_count",
            severity=Severity.WARNING,
            message=(f"{feature}: {unseen_count} unseen categories detected (e.g. {examples})"),
            feature=feature,
            value=float(unseen_count),
            threshold=0.0,
        )
