from mldebug.checks.drift.detector import DriftDetector
from mldebug.checks.drift.psi import PSIDriftCheck
from tests.fixtures.data import generate_normal_data


def test_detector_runs_all_checks():
    detector = DriftDetector([PSIDriftCheck()])

    report = detector.run(generate_normal_data(), generate_normal_data(mean=1))

    assert report is not None
    assert isinstance(report.issues, list)
