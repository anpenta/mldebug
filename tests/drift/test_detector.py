from mldebug.drift.detector import DriftDetector

from mldebug.checks.drift.psi import PSIDriftCheck
from tests.fixtures.data import make_shifted_normal_data, normal_data


def test_detector_runs_all_checks():
    detector = DriftDetector([PSIDriftCheck()])

    report = detector.run(normal_data(), make_shifted_normal_data())

    assert report is not None
    assert isinstance(report.issues, list)
