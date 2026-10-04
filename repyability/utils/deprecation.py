"""Deprecated names, arguments and calls.

A deprecation gives one minor release's notice: what 0.12 deprecates goes
in 0.13 (``NEXT_REMOVAL``). Each warns with a ``FutureWarning``, which
Python always shows, as the notice is short:

- calling a result's value that is now a property, such as
  ``SparesDemand.mean()`` (``called``, #184);
- ``StandbyModel``'s and ``LoadSharingModel``'s ``mc_samples``, ``lower``
  and ``seed``, which set a fit to simulated lifetimes that 0.12 removed
  (``ignored``, #149).

What 0.13 deprecates goes in 0.14 (``REMOVAL_AFTER_NEXT``): the
search's ``offsets`` of ``optimal_inspection_intervals``, renamed
``offset_shares`` (``renamed``, #222).

What 0.11 deprecated went in 0.12 (#149).
"""

import warnings
from typing import Any, Dict

#: The release that removes what 0.12 deprecates.
NEXT_REMOVAL = "0.13"
#: The release that removes what 0.13 deprecates.
REMOVAL_AFTER_NEXT = "0.14"


class CalledValue(float):
    """A number that a result gave by a method and now gives as a property:
    it is the number, and calling it (the old way) gives the number too,
    with a ``FutureWarning`` (see ``called``)."""

    _name: str

    def __new__(cls, value: float, name: str):
        number = super().__new__(cls, value)
        number._name = name
        return number

    def __call__(self) -> float:
        warnings.warn(
            f"{self._name} is a property now, as the other results' values "
            f"are: write {self._name.split('.')[-1]}, not "
            f"{self._name.split('.')[-1]}(). Calling it is deprecated and "
            f"will be refused in {NEXT_REMOVAL}.",
            FutureWarning,
            stacklevel=2,
        )
        return float(self)

    def __reduce__(self):
        return (float, (float(self),))


def called(value: float, name: str) -> CalledValue:
    """``value``, a result's property ``name`` (``"Class.attribute"``) that
    used to be a method: still callable, with a ``FutureWarning``, until
    ``NEXT_REMOVAL``."""
    return CalledValue(value, name)


def ignored(method: str, why: str, given: Dict[str, Any]) -> None:
    """Warn that the arguments in ``given`` that were passed (not None or
    False) are ignored by ``method``, for the reason ``why``: deprecated in
    0.12, they are refused in ``NEXT_REMOVAL``."""
    passed = [
        name
        for name, value in given.items()
        if value is not None and value is not False
    ]
    if passed:
        warnings.warn(
            f"{method} ignores {', '.join(passed)}: {why} Passing "
            f"{'it' if len(passed) == 1 else 'them'} is deprecated, and "
            f"{NEXT_REMOVAL} will refuse "
            f"{'it' if len(passed) == 1 else 'them'}.",
            FutureWarning,
            stacklevel=3,
        )


def renamed(method: str, old: str, new: str, why: str) -> None:
    """Warn that ``method``'s argument ``old`` is renamed ``new``, for the
    reason ``why``: deprecated in 0.13, it is refused in
    ``REMOVAL_AFTER_NEXT``."""
    warnings.warn(
        f"{method}'s {old}= is renamed {new}=, as {why}. Passing {old}= is "
        f"deprecated, and {REMOVAL_AFTER_NEXT} will refuse it.",
        FutureWarning,
        stacklevel=3,
    )
