"""Deprecated names, arguments and calls.

A deprecation gives one minor release's notice: what 0.20 deprecates goes
in 0.21 (``NEXT_REMOVAL``), and what 0.21 deprecates in 0.22
(``REMOVED_IN_0_22``). Each warns with a ``FutureWarning``, which
Python always shows, as the notice is short: an argument renamed
(``renamed``; an attribute, ``renamed_attribute``), a result's value
called that is now a property (``called``), arguments a method ignores
(``ignored``), or a function that moved to surpyval (``moved``).

0.20 deprecates, for removal in 0.21 (``NEXT_REMOVAL``), a price on an
opportunity (``price_on_part``: ``Opportunity(name, cost, ...)``,
``Shutdown(replace_cost=)``, ``Shutdowns(any_day=<price>)``,
``Calendar(failure=, anytime=)``; the prices come with the part), and
the names ``Calendar``, ``Shutdown``, ``Shutdowns``,
``NonRepairable.optimal_calendar_policy`` and ``calendar_policy_cost``
(``replaced``), for ``Opportunities``, ``Opportunity``,
``optimal_opportunity_policy`` and ``opportunity_policy_cost``; and a
choice made for the caller that it should make itself (``refuses_from``:
``method="auto"`` sampling a calendar too long to work out exactly, which
0.21 refuses, naming a ready ``horizon=`` and ``method="sampling"``).

0.19 deprecated nothing, so 0.20 removes nothing. 0.18 deprecated
nothing, so 0.19 removed nothing.
What 0.17 deprecated went in 0.18: ``repyability.demonstration``'s nine
functions and ``DemonstrationPlan``, for ``surpyval.test_planning``'s.
0.16 deprecated nothing, nor did 0.15, so 0.16 and 0.17 removed nothing.

What 0.14 deprecated went in 0.15: ``NodeState``'s ``phase``, renamed
``since_scheduled`` (#350). What 0.13 deprecated went in 0.14:
``optimal_inspection_intervals``' ``offsets``, renamed ``offset_shares``
(#222); calling ``CapacityDistribution.mean()``, a property since 0.13 (#235);
the simulation options given to the exact ``mean`` of a ``StandbyModel``,
``LoadSharingModel`` or ``DegradingNode``, which it would ignore (#233); and
``rbd._sensitivity``'s ``levers`` and ``Lever``, the names callers used before
the levers were public (#244). What 0.12 deprecated went in 0.13: calling
``SparesDemand.mean()`` and ``std()``, properties since 0.12 (#184), and
``StandbyModel``'s and ``LoadSharingModel``'s ``mc_samples``, ``lower`` and
``seed``, which set a fit to simulated lifetimes that 0.12 removed (#149). What
0.11 deprecated went in 0.12 (#149): its old names are refused with the names
that took their place (``refuses_removed_names``, #232).
"""

import functools
import inspect
import warnings
from typing import Any, Dict

import numpy as np

from repyability.utils.wrappers import outside_level

#: The release that removes what 0.20 deprecates: one minor release's
#: notice from the release that first warns of it.
NEXT_REMOVAL = "0.21"
#: The release that removes what 0.21 deprecates.
REMOVED_IN_0_22 = "0.22"


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


def renamed(
    method: str, old: str, new: str, why: str, removal: str = NEXT_REMOVAL
) -> None:
    """Warn that ``method``'s argument ``old`` is renamed ``new``, for the
    reason ``why``: deprecated, it is refused in ``removal``
    (``NEXT_REMOVAL`` by default)."""
    warnings.warn(
        f"{method}'s {old}= is renamed {new}=, as {why}. Passing {old}= is "
        f"deprecated, and {removal} will refuse it.",
        FutureWarning,
        stacklevel=outside_level(),
    )


def moved(old: str, new: str, why: str, removal: str = NEXT_REMOVAL) -> None:
    """Warn that the function ``old`` (``"repyability.name"``) is
    deprecated for ``new``, another package's, for the reason ``why``: it
    is removed in ``removal`` (``NEXT_REMOVAL`` by default)."""
    warnings.warn(
        f"{old} is deprecated: use {new}, as {why}. {old} will be removed "
        f"in {removal}.",
        FutureWarning,
        stacklevel=outside_level(),
    )


def replaced(
    old: str, new: str, why: str, removal: str = NEXT_REMOVAL
) -> None:
    """Warn that ``old`` (a class, or a method as ``"Class.method"``) is
    deprecated for ``new``, for the reason ``why``: removed in
    ``removal`` (``NEXT_REMOVAL`` by default)."""
    warnings.warn(
        f"{old} is deprecated: use {new}, as {why}. {old} will be removed "
        f"in {removal}.",
        FutureWarning,
        stacklevel=outside_level(),
    )


def price_on_part(how: str, removal: str = NEXT_REMOVAL) -> None:
    """Warn that ``how`` (``"Opportunity(name, cost, ...)"``) puts a price
    on an opportunity or a collection of them, which the part now
    carries: deprecated, refused in ``removal`` (``NEXT_REMOVAL`` by
    default)."""
    warnings.warn(
        f"{how} puts a price on the opportunities, which is deprecated: an "
        "opportunity applies to every part, so the price comes with the "
        "part (NonRepairable's planned_cost on any day, unplanned_cost at a "
        "failure, opportunity_cost at an opportunity, and "
        "opportunity_costs={name: price} for one opportunity). It is used "
        f"as the part's price for now; {removal} will refuse it.",
        FutureWarning,
        stacklevel=outside_level(),
    )


def refuses_from(what: str, instead: str, removal: str = NEXT_REMOVAL):
    """Warn that ``what`` (a choice the call made for the caller, as it
    has until now) is deprecated: from ``removal`` (``NEXT_REMOVAL`` by
    default) the call refuses, and the caller gives ``instead``."""
    warnings.warn(
        f"{what} From {removal} this refuses: {instead}",
        FutureWarning,
        stacklevel=outside_level(),
    )


def renamed_attribute(
    owner: str, old: str, new: str, why: str, removal: str = NEXT_REMOVAL
) -> None:
    """Warn that ``owner``'s attribute ``old`` (read as ``owner.old``) is
    renamed ``new``, for the reason ``why``: deprecated, reading it is
    refused in ``removal`` (``NEXT_REMOVAL`` by default)."""
    warnings.warn(
        f"{owner}.{old} is renamed {new}, as {why}. Reading {old} is "
        f"deprecated, and {removal} will refuse it.",
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


#: The names of a simulation's count that a method drawing parameters
#: (``n_draws``) is given by mistake.
_SIMULATION_COUNTS = _REMOVED | _COUNTS


def refuses_removed_names(function):
    """``function``, refusing the simulation-count names 0.12 removed with
    the names that took their place, where Python would say only that the
    keyword was unexpected; ``function`` itself where it takes
    neither ``mc_samples`` nor ``max_samples``.

    A method that takes ``n_draws`` but not ``mc_samples`` (the
    ``*_uncertainty`` methods, which draw the uncertain models'
    parameters and work each draw out without simulating it) refuses
    ``mc_samples`` and the removed names, naming ``n_draws``: the two
    names count different things, so ``n_draws`` is kept (#105's
    ``mc_samples`` counts simulations)."""
    if getattr(function, "refuses_removed_names", False):
        return function
    try:
        taken = set(inspect.signature(function).parameters)
    except (TypeError, ValueError):
        return function
    draws = "n_draws" in taken and not taken & _COUNTS
    if not taken & _COUNTS and not draws:
        return function
    name = function.__qualname__.removesuffix(".__init__")

    @functools.wraps(function)
    def checked(*args, **kwargs):
        if draws and not _SIMULATION_COUNTS.isdisjoint(kwargs):
            old = next(k for k in kwargs if k in _SIMULATION_COUNTS)
            raise TypeError(
                f"{name}() got an unexpected keyword argument {old!r}: it "
                "takes n_draws=, the number of draws of the uncertain "
                "models' parameters, each draw worked out without "
                "simulating it (mc_samples counts simulations)."
            )
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
