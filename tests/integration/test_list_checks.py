from mldebug import list_checks


def test_list_checks_returns_valid_structure() -> None:
    result = list_checks()

    assert isinstance(result, dict)
    assert len(result) > 0

    for group, checks in result.items():
        assert isinstance(group, str)
        assert isinstance(checks, list)
        assert all(isinstance(name, str) for name in checks)


def test_list_checks_contains_core_groups() -> None:
    result = list_checks()

    assert "numeric" in result
    assert "categorical" in result


def test_list_checks_exposes_expected_check_kinds() -> None:
    result = list_checks()

    numeric_checks = result["numeric"]
    categorical_checks = result["categorical"]

    assert any("missing" in name for name in numeric_checks)
    assert any("ks" in name for name in numeric_checks)
    assert any("psi" in name for name in categorical_checks)
