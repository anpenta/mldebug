from mldebug.checks.drift.ks_test import KSTestCheck

from tests.fixtures.data import make_shifted_normal_data, normal_data


def test_ks_detects_shift():
    check = KSTestCheck(alpha=0.05)

    issue = check.run(normal_data(), make_shifted_normal_data())

    assert issue is not None
