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
