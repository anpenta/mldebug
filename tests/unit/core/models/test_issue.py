from mldebug.core.models.issue import Issue, Severity


def test_issue_str_format_correct() -> None:
    issue = Issue(
        name="psi_drift",
        metric="psi",
        severity=Severity.WARNING,
        message="country: drift detected",
        feature="country",
        value=0.32,
        threshold=0.2,
    )

    assert str(issue) == "[WARNING] psi_drift - country: drift detected"


def test_issue_repr_contains_all_fields() -> None:
    issue = Issue(
        name="psi_drift",
        metric="psi",
        severity=Severity.WARNING,
        message="country: drift detected",
        feature="country",
        value=0.32,
        threshold=0.2,
    )

    r = repr(issue)

    assert r.startswith("Issue(")
    assert "name='psi_drift'" in r
    assert "metric='psi'" in r
    assert "severity='warning'" in r
    assert "message='country: drift detected'" in r
    assert "feature='country'" in r
    assert "value=0.32" in r
    assert "threshold=0.2" in r
    assert r.endswith(")")
