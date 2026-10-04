"""What an installed copy needs is in the wheel (#167, #177)."""

import tomllib
from fnmatch import fnmatch

from repyability.tests.repository import ROOT, source


def test_every_data_file_in_the_package_ships():
    # setuptools puts only modules in the wheel, and data files listed as
    # package data: a test's recorded results, left out, failed 13 modules
    # of the installed tests.
    project = tomllib.loads(source("pyproject.toml").read_text())
    listed = project["tool"]["setuptools"]["package-data"]
    files = [
        path
        for path in (ROOT / "repyability").rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in (".py", ".pyc")
    ]
    assert files
    for path in files:
        package = ".".join(path.parent.relative_to(ROOT).parts)
        globs = listed.get(package, [])
        assert any(
            fnmatch(path.name, glob) for glob in globs
        ), f"{path.relative_to(ROOT)} is not package data in pyproject.toml"


def test_the_package_is_marked_typed():
    assert (ROOT / "repyability" / "py.typed").is_file()


def test_the_licence_is_an_spdx_expression():
    project = tomllib.loads(source("pyproject.toml").read_text())["project"]
    assert project["license"] == "MIT"
    assert project["license-files"] == ["LICENSE"]
    # PEP 639: the expression replaces the licence classifiers.
    assert not any(c.startswith("License ::") for c in project["classifiers"])
    assert "Python Packaging Authority" not in source("LICENSE").read_text()
