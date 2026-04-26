from mldebug.checks.drift.ks import KSTestCheck
from tests.fixtures.data import generate_normal_data


def test_ks_detects_shift():
    check = KSTestCheck(alpha=0.05)

    issue = check.run(generate_normal_data(), generate_normal_data(mean=1, std=1))

    assert issue is not None
