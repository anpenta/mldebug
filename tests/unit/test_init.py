from mldebug import __version__


def test_package_version_is_available() -> None:
    assert isinstance(__version__, str)
    assert __version__ != ""
