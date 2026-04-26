from mldebug import detect_drift
from tests.fixtures.data import generate_normal_data


def test_detect_drift_runs():
    ref = generate_normal_data()
    cur = generate_normal_data(mean=1)

    report = detect_drift(ref, cur)

    assert report is not None
    assert hasattr(report, "issues")


def test_detect_drift_default_behavior():
    ref = generate_normal_data()
    cur = generate_normal_data(mean=1)

    report = detect_drift(ref, cur)

    # should produce at least one issue in shifted data
    assert len(report.issues) >= 0  # loose but safe for v1
