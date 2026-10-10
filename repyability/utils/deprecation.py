"""Deprecated names, arguments and calls.

A deprecation gives one minor release's notice: what 0.13 deprecates goes
in 0.14 (``NEXT_REMOVAL``). Each warns with a ``FutureWarning``, which
Python always shows, as the notice is short:

- the search's ``offsets`` of ``optimal_inspection_intervals``, renamed
  ``offset_shares`` (``renamed``, #222);
- calling ``CapacityDistribution.mean()``, now a property (``called``,
  #235);
- the simulation options of a ``StandbyModel``'s, ``LoadSharingModel``'s
  or ``DegradingNode``'s exact ``mean``, which it ignores (``ignored``,
  #233).

What 0.12 deprecated went in 0.13: calling ``SparesDemand.mean()`` and
``std()``, properties since 0.12 (#184), and ``StandbyModel``'s and
``LoadSharingModel``'s ``mc_samples``, ``lower`` and ``seed``, which set a
fit to simulated lifetimes that 0.12 removed (#149). What 0.11 deprecated
went in 0.12 (#149): its old names are refused with the names that took
their place (``refuses_removed_names``, #232).
"""

import functools
import inspect
import warnings
from typing import Any, Dict

import numpy as np

from repyability.utils.wrappers import outside_level

#: The release that removes what 0.13 deprecates.
NEXT_REMOVAL = "0.14"


def _called_warning(name: str, removal: str) -> None:
    """Warn that the property ``name`` (``"Class.attribute"``) was called,
    as the method it was: refused in ``removal``."""
    short = name.split(".")[-1]
    warnings.warn(
        f"{name} is a property now, as the other results' values are: "
        f"write {short}, not {short}(). Calling it is deprecated and will "
        f"be refused in {removal}.",
        FutureWarning,
        stacklevel=outside_level(),
    )


class CalledValue(float):
    """A number that a result gave by a method and now gives as a property:
    it is the number, and calling it (the old way) gives the number too,
    with a ``FutureWarning`` (see ``called``)."""

    _name: str
    _removal: str

    def __new__(cls, value: float, name: str, removal: str = NEXT_REMOVAL):
        number = super().__new__(cls, value)
        number._name = name
        number._removal = removal
        return number

    def __call__(self) -> float:
        _called_warning(self._name, self._removal)
        return float(self)

    def __reduce__(self):
        return (float, (float(self),))


class CalledArray(np.ndarray):
    """An array that a result gave by a method and now gives as a
    property, as ``CalledValue`` is a number: calling it gives a plain copy
    of the array, with a ``FutureWarning``."""

    _name: str = ""
    _removal: str = NEXT_REMOVAL

    def __call__(self) -> np.ndarray:
        _called_warning(self._name, self._removal)
        return np.array(self)

    def __reduce__(self):
        return (np.array, (np.array(self),))


def called(value: Any, name: str, removal: str = NEXT_REMOVAL) -> Any:
    """``value``, a result's property ``name`` (``"Class.attribute"``) that
    used to be a method: still callable, with a ``FutureWarning``, until
    ``removal`` (``NEXT_REMOVAL`` by default). A number
    becomes a ``CalledValue``, an array a ``CalledArray``."""
    if np.ndim(value) == 0:
        return CalledValue(float(value), name, removal)
    array = np.asarray(value).view(CalledArray)
    array._name, array._removal = name, removal
    return array


def ignored(
    method: str,
    why: str,
    given: Dict[str, Any],
    removal: str = NEXT_REMOVAL,
) -> None:
    """Warn that the arguments in ``given`` that were passed (not None or
    False) are ignored by ``method``, for the reason ``why``: refused in
    ``removal`` (``NEXT_REMOVAL`` by default)."""
    passed = [
        name
        for name, value in given.items()
        if value is not None and value is not False
    ]
    if passed:
        warnings.warn(
            f"{method} ignores {', '.join(passed)}: {why} Passing "
            f"{'it' if len(passed) == 1 else 'them'} is deprecated, and "
            f"{removal} will refuse "
            f"{'it' if len(passed) == 1 else 'them'}.",
            FutureWarning,
            stacklevel=outside_level(),
        )


def renamed(method: str, old: str, new: str, why: str) -> None:
    """Warn that ``method``'s argument ``old`` is renamed ``new``, for the
    reason ``why``: deprecated in 0.13, it is refused in ``NEXT_REMOVAL``."""
    warnings.warn(
        f"{method}'s {old}= is renamed {new}=, as {why}. Passing {old}= is "
        f"deprecated, and {NEXT_REMOVAL} will refuse it.",
        FutureWarning,
        stacklevel=outside_level(),
    )


#: The simulation-count names 0.12 removed (#105, #149), and the names
#: that took their place.
REMOVED_NAMES = {
    "N": "mc_samples",
    "max_N": "max_samples",
    "n_sims": "mc_samples",
    "n_simulations": "mc_samples",
}
_REMOVED = frozenset(REMOVED_NAMES)
#: What a function takes that the removed names stood for.
_COUNTS = frozenset({"mc_samples", "max_samples"})


def refuses_removed_names(function):
    """``function``, refusing the simulation-count names 0.12 removed with
    the names that took their place, where Python would say only that the
    keyword was unexpected; ``function`` itself where it takes
    neither ``mc_samples`` nor ``max_samples``."""
    if getattr(function, "refuses_removed_names", False):
        return function
    try:
        taken = set(inspect.signature(function).parameters)
    except (TypeError, ValueError):
        return function
    if not taken & _COUNTS:
        return function
    name = function.__qualname__.removesuffix(".__init__")

    @functools.wraps(function)
    def checked(*args, **kwargs):
        if kwargs and not _REMOVED.isdisjoint(kwargs):
            for old in kwargs:
                if old in _REMOVED and old not in taken:
                    raise TypeError(
                        f"{name}() got an unexpected keyword argument "
                        f"{old!r}: 0.12 removed it (#149); give "
                        f"{REMOVED_NAMES[old]}= instead."
                    )
        return function(*args, **kwargs)

    checked.refuses_removed_names = True  # type: ignore[attr-defined]
    return checked


def refuse_removed_names(cls):
    """``cls``, its constructor and public methods that take
    ``mc_samples`` or ``max_samples`` refusing the names 0.12 removed (see
    ``refuses_removed_names``)."""
    for name, value in list(vars(cls).items()):
        if (name == "__init__" or not name.startswith("_")) and (
            inspect.isfunction(value)
        ):
            wrapped = refuses_removed_names(value)
            if wrapped is not value:
                setattr(cls, name, wrapped)
    return cls
