"""Docstrings that read well in ``help()`` (#238).

The API reference's cross-references, ``[`point_availability`][repyability.
RepairableRBD.point_availability]``, link the web pages, which mkdocstrings
builds from the source; at run time, where ``help()`` reads them, they are
the names alone, ````point_availability````.
"""

import re

#: A mkdocs cross-reference: its text, then its target.
LINK = re.compile(r"\[`([^`\]]+)`\]\[[\w.]*\]")


def readable(text):
    """``text`` with its cross-references as their names (``None`` as is)."""
    if not text or "][" not in text:
        return text
    return LINK.sub(r"``\1``", text)


def _rewrite(obj) -> None:
    try:
        obj.__doc__ = readable(obj.__doc__)
    except (AttributeError, TypeError):  # a builtin's, or read-only
        pass


def readable_docstrings(objects) -> None:
    """Rewrite the docstrings of ``objects`` (classes and functions) and of
    the classes' own methods and properties, in place, once each."""
    seen = set()
    for obj in objects:
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        _rewrite(obj)
        if not isinstance(obj, type):
            continue
        for member in vars(obj).values():
            if isinstance(member, (staticmethod, classmethod)):
                member = member.__func__
            if isinstance(member, property):
                _rewrite(member)
                member = member.fget
            if callable(member) and id(member) not in seen:
                seen.add(id(member))
                _rewrite(member)
