"""Old argument names and ignored arguments, kept working until 1.0 (#105).

Each warns with a ``DeprecationWarning`` pointing at the caller's line.
"""

import warnings
from typing import Any, Dict


def renamed(
    new: str, value: Any, old: str, old_value: Any, stacklevel: int = 3
) -> Any:
    """The value of an argument renamed from ``old`` to ``new``.

    ``value`` if the new name was used, else ``old_value`` with a
    ``DeprecationWarning``. Both default to None in the signature, so an
    unused name is None.

    Parameters
    ----------
    new, old : str
        The argument's new and old names.
    value, old_value : Any
        What was passed under each name (None if nothing).
    stacklevel : int, optional
        Passed to ``warnings.warn``: 3 points at the caller of the function
        that calls ``renamed``.

    Returns
    -------
    Any
        The value to use.

    Raises
    ------
    TypeError
        If both names were used.
    """
    if old_value is None:
        return value
    if value is not None:
        raise TypeError(f"Give {new} or its old name {old}, not both.")
    warnings.warn(
        f"{old} is deprecated: use {new}. {old} will be removed in 1.0.",
        DeprecationWarning,
        stacklevel=stacklevel,
    )
    return old_value


def ignored(method: str, why: str, given: Dict[str, Any]) -> None:
    """Warn that the arguments in ``given`` that were passed (not None or
    False) are ignored by ``method``, for the reason ``why``."""
    passed = [
        name
        for name, value in given.items()
        if value is not None and value is not False
    ]
    if passed:
        warnings.warn(
            f"{method} ignores {', '.join(passed)}: {why} Passing "
            f"{'it' if len(passed) == 1 else 'them'} is deprecated.",
            DeprecationWarning,
            stacklevel=3,
        )
