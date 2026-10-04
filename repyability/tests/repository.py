"""The repository's own files that some tests check against: the docs, the
README and ``pyproject.toml``. They are there in a checkout, but not in an
installed copy, whose tests sit in site-packages (which may hold other
packages' ``docs`` or ``README.md``): there the tests that need them skip
(#167)."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _is_checkout(root: Path) -> bool:
    """Whether ``root`` is RePyability's repository."""
    project = root / "pyproject.toml"
    return (
        (root / "mkdocs.yml").is_file()
        and project.is_file()
        and 'name = "repyability"' in project.read_text()
    )


#: Whether the tests run from a checkout of the repository.
CHECKOUT = _is_checkout(ROOT)


def source(relative: str) -> Path:
    """The repository's file at ``relative``, or skip the test that asks
    for it when the tests run from an installed copy."""
    if not CHECKOUT:
        pytest.skip("the repository's files are not available")
    return ROOT / relative
