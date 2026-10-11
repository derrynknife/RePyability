"""The README's capability matrices are the code's (see
``repyability.tests.capabilities``, which makes them): a method added,
removed or made exact, or a refusal lifted, shows there."""

import pathlib

from repyability.tests import capabilities


def test_the_readme_s_capability_matrices_are_current():
    readme = pathlib.Path(__file__).resolve().parents[2] / "README.md"
    text = readme.read_text()
    start = text.index(capabilities.START)
    end = text.index(capabilities.END) + len(capabilities.END)
    assert text[start:end] == capabilities.matrix(), (
        "The README's capability matrices are out of date: run "
        "`python -m repyability.tests.capabilities` to write them."
    )


def test_every_public_method_has_a_row_or_is_listed_as_none():
    # Each model type's public analyses appear in the matrix, so that a new
    # one is not left out of it.
    named = {
        method
        for _, rows in capabilities.CAPABILITIES
        for _, methods in rows
        for method in methods
    }
    for name, cls in capabilities.MODELS.items():
        missing = sorted(
            method
            for method in dir(cls)
            if not method.startswith("_")
            and callable(getattr(cls, method))
            and method not in named
            and method not in capabilities.NOT_ANALYSES
        )
        assert not missing, (name, missing)
