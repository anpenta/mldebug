import numpy as np

from mldebug.checks.psi import run_psi_drift_check_categorical


def test_run_psi_drift_check_categorical_detects_shift() -> None:
    reference = np.array(["A"] * 80 + ["B"] * 20, dtype="object")
    current = np.array(["A"] * 40 + ["B"] * 40 + ["C"] * 20, dtype="object")

    issue = run_psi_drift_check_categorical(
        feature="feature_1",
        reference=reference,
        current=current,
        threshold=0.1,
    )

    assert issue is not None
    assert issue.metric == "psi"
    assert issue.feature == "feature_1"
    assert issue.value > 0.1


def test_run_psi_drift_check_categorical_no_detection_when_stable() -> None:
    reference = np.array(["A"] * 50 + ["B"] * 50, dtype="object")
    current = np.array(["A"] * 52 + ["B"] * 48, dtype="object")

    issue = run_psi_drift_check_categorical(
        feature="feature_1",
        reference=reference,
        current=current,
        threshold=0.1,
    )

    assert issue is None
