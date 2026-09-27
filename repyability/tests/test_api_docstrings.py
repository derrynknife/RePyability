"""Every part of the public API is documented.

Walks everything exported from ``repyability`` and the public members its
classes define, and requires a docstring. Where a callable takes parameters,
each must be documented by name: a function's or method's in its own
``Parameters`` section, a constructor's in the class docstring's
``Parameters`` (or, for the result dataclasses, ``Attributes``) section.
``self``, ``cls``, ``*args`` and ``**kwargs`` are exempt. The API reference is
generated from these docstrings, so a gap here is a gap in the docs.
"""

import inspect
import re

import pytest

import repyability


def _heading(title: str) -> str:
    """A regex for a numpydoc section heading: ``title`` underlined with
    dashes."""
    return rf"^\s*{title}\s*\n\s*-{{3,}}"


def _section(doc: str, *titles: str) -> str:
    """The text of the first numpydoc section with one of ``titles``."""
    for title in titles:
        body = r"\s*\n(.*?)(?=" + _heading(r"\w[\w ]*") + r"|\Z)"
        match = re.search(_heading(title) + body, doc, re.M | re.S)
        if match:
            return match.group(1)
    return ""


def _named_parameters(func) -> list[str]:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return []
    return [
        name
        for name, p in signature.parameters.items()
        if name not in ("self", "cls")
        and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    ]


def _documents(section: str, name: str) -> bool:
    return re.search(rf"^\s*\**{re.escape(name)}\b", section, re.M) is not None


def _members():
    """(qualified name, object, kind) for every public API member, each
    checked on the class that defines it."""
    seen = set()
    for export in repyability.__all__:
        if export == "__version__":
            continue
        obj = getattr(repyability, export)
        yield export, obj, "class" if inspect.isclass(obj) else "function"
        if not inspect.isclass(obj):
            continue
        for cls in obj.__mro__:
            if not cls.__module__.startswith("repyability"):
                continue
            for name, member in vars(cls).items():
                if name.startswith("_") or (cls, name) in seen:
                    continue
                seen.add((cls, name))
                if isinstance(member, property):
                    yield f"{cls.__name__}.{name}", member, "property"
                elif isinstance(member, (staticmethod, classmethod)):
                    yield f"{cls.__name__}.{name}", member.__func__, "method"
                elif inspect.isfunction(member):
                    yield f"{cls.__name__}.{name}", member, "method"


MEMBERS = list(_members())


@pytest.mark.parametrize(
    "qualname, obj, kind", MEMBERS, ids=[m[0] for m in MEMBERS]
)
def test_documented(qualname, obj, kind):
    doc = inspect.cleandoc(obj.__doc__ or "")
    assert doc, f"{qualname} has no docstring"
    if kind == "property":
        return
    if kind == "class":
        init = obj.__dict__.get("__init__")
        if init is None:
            return  # inherits its constructor (and its documentation)
        section = _section(doc, "Parameters", "Attributes")
        if not section and init.__doc__:
            section = _section(inspect.cleandoc(init.__doc__), "Parameters")
        params = _named_parameters(init)
    else:
        section = _section(doc, "Parameters")
        params = _named_parameters(obj)
    missing = [p for p in params if not _documents(section, p)]
    assert not missing, f"{qualname} does not document {missing}"


def test_the_walk_finds_the_api():
    # Guard against the walk silently finding nothing.
    assert len(MEMBERS) > 150
