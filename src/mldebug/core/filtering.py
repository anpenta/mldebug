from collections.abc import Mapping, Sequence
from typing import Any


def get_valid_features(
    reference: Mapping[str, Sequence[Any]],
    current: Mapping[str, Sequence[Any]],
    schema: Mapping[str, str],
) -> list[str]:
    """Get features eligible for feature-level checks.

    A feature is included only if it exists in the schema, reference dataset, and current dataset. This defines the
    execution scope for feature checks after schema validation.

    Parameters
    ----------
    reference : Mapping[str, Sequence[Any]]
        Reference dataset keyed by feature name.

    current : Mapping[str, Sequence[Any]]
        Current dataset keyed by feature name.

    schema : Mapping[str, str]
        Feature-to-type mapping defining expected dataset structure.

    Returns
    -------
    list[str]
        Features safe to use in feature-level checks.

    """
    return [f for f in schema if f in reference and f in current]
