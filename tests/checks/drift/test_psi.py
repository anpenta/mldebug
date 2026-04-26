from mldebug.checks.drift.psi import PSIDriftCheck
from tests.fixtures.data import normal_data, shifted_normal_data


def test_psi_detects_shift():
    check = PSIDriftCheck(threshold=0.1)

    issue = check.run(normal_data(), shifted_normal_data())

    assert issue is not None
    assert issue.metric == "psi"
