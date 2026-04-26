from mldebug.checks.data_quality.missing_values import MissingValueCheck
from tests.fixtures.data import inject_missing_values, normal_data


def test_missing_values_detects_increase():
    ref = inject_missing_values(normal_data(), rate=0.01)
    cur = inject_missing_values(normal_data(), rate=0.2)

    check = MissingValueCheck(threshold=0.05)

    issue = check.run(ref, cur)

    assert issue is not None
    assert issue.metric == "missing_delta"
