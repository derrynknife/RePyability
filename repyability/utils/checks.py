"""Checks of the arguments the public API shares."""

import numpy as np


def is_whole(value) -> bool:
    """Whether ``value`` is a whole number: ``3`` or ``3.0``, but not
    ``True``."""
    if isinstance(value, (bool, np.bool_)):
        return False
    if isinstance(value, (int, np.integer)):
        return True
    return isinstance(value, (float, np.floating)) and bool(
        float(value).is_integer()
    )


def whole_number(value, name: str, minimum: int = 1) -> int:
    """``value`` as an ``int``, if it is a whole number (see ``is_whole``)
    of at least ``minimum``; else a ``ValueError`` that names it."""
    if not is_whole(value) or value < minimum:
        raise ValueError(
            f"{name} must be a whole number, at least {minimum}, got "
            f"{value!r}."
        )
    return int(value)


#: The ways the structure is evaluated: through the minimal path sets or the
#: minimal cut sets (both give the same values).
_STRUCTURE_METHODS = {"p": "p", "paths": "p", "c": "c", "cuts": "c"}


def structure_method(method) -> str:
    """``"p"`` or ``"c"``, from a structure ``method``: ``"p"`` or
    ``"paths"`` for the minimal path sets, ``"c"`` or ``"cuts"`` for the
    minimal cut sets (#179); else a ``ValueError``."""
    try:
        return _STRUCTURE_METHODS[method]
    except (KeyError, TypeError):
        raise ValueError(
            "`method` must be 'p' (or 'paths') or 'c' (or 'cuts'), got "
            f"{method!r}."
        ) from None
