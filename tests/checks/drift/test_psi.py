from mldebug.checks.drift.psi import run_psi_drift_check
from tests.fixtures.data import generate_normal_data


def test_psi_detects_shift():
    check = run_psi_drift_check(threshold=0.1)

    issue = check.run(generate_normal_data(), generate_normal_data(mean=1))

    assert issue is not None
    assert issue.metric == "psi"


def test_psi_no_detection_when_stable():
    check = run_psi_drift_check(threshold=0.1)

    issue = check.run(generate_normal_data(), generate_normal_data(mean=0.05))

    assert issue is None
