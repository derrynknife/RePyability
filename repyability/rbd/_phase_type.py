"""A time (a life, or a repair) as phases a Markov chain can carry.

A chain's states hold no clock, so a time that is not exponential is
written as phases: the time passes through them in turn, leaves each at
its rate, and on leaving one either ends (with that phase's probability)
or goes on to the next. An exponential time is one phase that always
ends, and an Erlang of ``k`` is ``k`` phases of one rate, ending from the
last: both are exact.

Any other time is approximated by a mixture of Erlangs (Tijms's): on a
grid of ages ``0 = a_0 < a_1 < ...``, phase ``j`` stands for the ages in
``(a_{j-1}, a_j]``, is left at rate ``1 / (a_j - a_{j-1})`` and ends with
the probability that the time, having reached ``a_{j-1}``, ends by
``a_j``. The number of phases passed when it ends then has the
distribution of the grid's cell the time falls in, and the time is the
sum of that many exponential steps, which is within a step or so of the
cell: as the grid is refined the phases' time converges to the time
itself, with an error in a probability that is a series in the step's
size. That is what lets a few levels be extrapolated (``converged``).

The grid's steps are ``scale / level`` up to the time's ``scale`` (a
typical value of it) and that share of the age beyond, so that a long
tail takes a number of phases that grows with its logarithm. It stops
where the time has ended but for ``_TAIL`` of its probability, or, when
only the ages up to a ``horizon`` matter, a margin of steps past it
(a time older than the last phase's ages stays in it, ending at its
rate, which the margin keeps from mattering).
"""

import math
from typing import Callable, List, NamedTuple, Optional

import numpy as np

from repyability.rbd._common import _cdf, _constant_rate
from repyability.rbd._model_utils import (
    MODEL_ERRORS,
    distribution_name,
    failure_time_scale,
    never_fails,
)

#: The levels a value is worked out at: each halves the grid's steps.
LEVELS = (4, 8, 16, 32, 64, 128, 256, 512)
#: The fewest levels a value that is not exact is extrapolated from.
FEWEST = 3
#: The probability of a time's tail that the phases leave out.
_TAIL = 1e-13
#: The steps past a horizon: as many standard deviations of the number of
#: steps taken by then, and a few more (as uniformization's margin).
_SPREAD, _MARGIN = 12.0, 20
#: The most phases an Erlang is written out in.
_ERLANG = 64


class Phases(NamedTuple):
    """A time's phases: the rate each is left at, and the probability
    that the time ends on leaving it (otherwise the next phase follows; a
    time in the last phase of a run stays until it ends).

    ``fresh`` is the phase a time that starts in the chain's course
    starts in. It is 0 unless the time going on at the chain's start is
    another (a life at an age, a repair going on for a while): that one's
    phases then come first, from phase 0, and end where ``fresh`` starts.
    ``exact`` says whether the phases are the time's own distribution.
    """

    rates: tuple
    ends: tuple
    fresh: int = 0
    exact: bool = True

    def last(self, phase: int) -> bool:
        """Whether ``phase`` is the last of its run."""
        return phase + 1 == self.fresh or phase + 1 == len(self.rates)


def exponential(rate: float) -> Phases:
    """An exponential time's one phase."""
    return Phases((rate,), (1.0,))


def _erlang(model) -> Optional[Phases]:
    """The phases of a gamma time of a whole shape, or None."""
    if distribution_name(model) != "Gamma" or getattr(model, "extras", None):
        return None
    try:
        shape, rate = (float(value) for value in np.ravel(model.params))
    except MODEL_ERRORS:
        return None
    if shape != round(shape) or not 1 <= shape <= _ERLANG or rate <= 0.0:
        return None
    k = int(shape)
    return Phases((rate,) * k, (0.0,) * (k - 1) + (1.0,))


class Aged:
    """A time that has gone on for ``age``: what is left of it."""

    def __init__(self, model, age: float):
        self.model, self.age = model, float(age)
        self._cdf = _cdf(model)
        self._passed = float(self._cdf(np.array([self.age]))[0])

    def ff(self, x):
        x = np.asarray(x, dtype=float)
        left = 1.0 - self._passed
        if left <= 0.0:
            return np.ones(x.shape)
        return (self._cdf(self.age + x) - self._passed) / left


def scale(model) -> float:
    """A typical value of the time, to size its grid by: its mean, or its
    median where it has none."""
    size = failure_time_scale(model)
    if not (math.isfinite(size) and size > 0.0):
        try:
            size = float(np.ravel(model.qf(np.array([0.5])))[0])
        except MODEL_ERRORS:
            size = math.nan
    if not (math.isfinite(size) and size > 0.0):
        raise NotImplementedError(
            "A time with neither a mean nor a median has no scale to "
            "size its phases by."
        )
    return size


def exact(model) -> Optional[Phases]:
    """The phases of a time that has them exactly, or None."""
    rate = _constant_rate(model)
    if rate is not None:
        return exponential(rate)
    return _erlang(model)


def _grid(size: float, level: int, horizon: float, ended: Callable) -> list:
    """The grid's edges (see the module's notes), to the first past
    ``horizon`` and a margin of steps more, or to where ``ended`` says the
    time is over, whichever comes first."""
    edges = [0.0]
    step = size / level
    beyond: Optional[int] = None
    while beyond is None or len(edges) <= beyond:
        age = edges[-1]
        edges.append(age + max(step, age / level))
        if beyond is None and edges[-1] >= horizon:
            count = len(edges)
            beyond = count + int(_SPREAD * math.sqrt(count)) + _MARGIN
        if len(edges) % 64 == 0 and ended(edges[-1]):
            break
    return edges


def approximate(
    model, level: int, horizon: float = math.inf, age: float = 0.0
) -> Phases:
    """The phases of the time's mixture of Erlangs at ``level`` (see the
    module's notes), of what is left of it at ``age``."""
    size = scale(model)
    time = Aged(model, age) if age > 0.0 else model
    cdf = _cdf(time)
    whole = 1.0 - (never_fails(model) if age == 0.0 else 0.0)

    def ended(at: float) -> bool:
        passed = float(cdf(np.array([at]))[0])
        return passed >= whole - _TAIL or (age > 0.0 and passed >= 1.0)

    edges = np.array(_grid(size, level, horizon, ended))
    passed = np.clip(cdf(edges), 0.0, 1.0)
    passed[0] = 0.0
    passed = np.maximum.accumulate(passed)
    # Up to the first edge by which the time is over.
    over = int(np.searchsorted(passed, min(whole, 1.0) - _TAIL))
    count = max(min(over, len(edges) - 1), 1)
    left = 1.0 - passed[:count]
    with np.errstate(all="ignore"):
        ends = np.where(
            left > 0.0, (passed[1 : count + 1] - passed[:count]) / left, 1.0
        )
    rates = 1.0 / np.diff(edges[: count + 1])
    return Phases(
        tuple(float(r) for r in rates),
        tuple(float(e) for e in np.clip(ends, 0.0, 1.0)),
        exact=False,
    )


def _started(time, level: int, horizon: float, age: float) -> Phases:
    own = exact(time)
    if own is not None and (age <= 0.0 or len(own.rates) == 1):
        return own
    return approximate(time, level, horizon, age)


def phases(
    model, level: int, horizon: float = math.inf, age: float = 0.0
) -> Phases:
    """The time's phases: its own where it has them, its mixture of
    Erlangs at ``level`` otherwise. At an ``age``, what is left of it
    comes first and a fresh time's phases after (``Phases.fresh``); an
    exponential time is the same at any age."""
    after = _started(model, level, horizon, 0.0)
    if age <= 0.0 or (after.exact and len(after.rates) == 1):
        return after
    first = _started(model, level, horizon, age)
    return Phases(
        first.rates + after.rates,
        first.ends + after.ends,
        fresh=len(first.rates),
        exact=False,
    )


class Unaffordable(Exception):
    """A level of phases whose chain is too large to work out."""


class NotSettled(NotImplementedError):
    """The levels of phases that could be worked out do not agree to the
    accuracy asked for."""


def converged(
    value: Callable[[int], np.ndarray],
    rtol: float,
    size: Callable[[np.ndarray], np.ndarray],
    exact: bool,
    instead: str,
) -> np.ndarray:
    """``value(level)`` as the levels grow, extrapolated.

    The value's error at a level is a series in the step's size, so the
    levels' values are extrapolated as Romberg's are (each column takes
    out a power of the step). The levels are taken until two successive
    extrapolations agree to within ``rtol * size(value)`` (element by
    element; ``size`` is positive), from at least
    ``FEWEST`` of them. ``value`` raises ``Unaffordable`` at a level that
    is too large; ``NotSettled`` is then raised, saying how far the
    values agreed and what to do ``instead``. An ``exact`` value is its
    first level's.
    """
    if exact:
        try:
            return np.asarray(value(LEVELS[0]), dtype=float)
        except Unaffordable as error:
            raise NotImplementedError(f"{error}.{instead}") from None
    table: List[List[np.ndarray]] = []
    agreed = math.inf
    short: Exception = Unaffordable(
        "Its values have not settled at the finest phases taken"
    )
    for j, level in enumerate(LEVELS):
        try:
            row = [np.asarray(value(level), dtype=float)]
        except Unaffordable as error:
            short = error
            break
        if not np.all(np.isfinite(row[0])):
            return row[0]
        for m in range(1, j + 1):
            row.append(
                row[m - 1] + (row[m - 1] - table[j - 1][m - 1]) / (2**m - 1)
            )
        table.append(row)
        if j:
            agreed = float(
                np.max(np.abs(row[-1] - table[j - 1][-1]) / size(row[-1]))
            )
            if j + 1 >= FEWEST and agreed <= rtol:
                return row[-1]
    reached = (
        "before two levels of them could be compared"
        if not math.isfinite(agreed)
        else f"the last two levels it could take agree to {agreed:.1e} of "
        f"their size, short of rtol={rtol:g} (a looser rtol takes them)"
    )
    raise NotSettled(
        f"{short}: the lives and repairs that are not exponential are "
        "followed as phases, on finer and finer grids of ages, and "
        f"{reached}.{instead}"
    )
