from mldebug.checks.drift.psi import PSIDriftCheck
from tests.fixtures.data import generate_normal_data


def test_psi_detects_shift():
    check = PSIDriftCheck(threshold=0.1)

    issue = check.run(generate_normal_data(), generate_normal_data(mean=1))

    assert issue is not None
    assert issue.metric == "psi"
