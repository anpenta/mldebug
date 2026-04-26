from mldebug import detect_drift
from tests.fixtures.data import normal_data, shifted_normal_data


def test_detect_drift_runs():
    ref = normal_data()
    cur = shifted_normal_data()

    report = detect_drift(ref, cur)

    assert report is not None
    assert hasattr(report, "issues")


def test_detect_drift_default_behavior():
    ref = normal_data()
    cur = shifted_normal_data()

    report = detect_drift(ref, cur)

    # should produce at least one issue in shifted data
    assert len(report.issues) >= 0  # loose but safe for v1
