"""The long-run optimal replacement of a part over a calendar of
opportunities (#319, UC-13 of this package's ``README.md``): opportunity-
based age replacement (Dekker et al., early 1990s), with deterministic
periodic opportunities at prices of their own.

A planned replacement's price depends on the time (a cheap shut, a dearer
day, or none), so the state is the time and the part's age, and the best
policy is a threshold for each time: replace if the age is at least
``a*(t)``. It is worked out on a grid of ``step`` by dynamic programming
(``_sweep``): from each step, keep the part (it survives the step with a
chance ``p_j`` at age index ``j``, or fails and is replaced, as new by the
next step, at the failure price) or replace it now at the step's price.

- **A finite horizon** (``_sweep`` from zero values): backward over the
  steps. It gives the thresholds over any calendar, and the cost rate by
  differencing the cost to go of a new part at two times.
- **A periodic calendar** (``_relative_values``): relative value iteration
  over one cycle, the cost rate exact for the grid, its bounds the span of
  a sweep's change.
- **A long cycle or none** (``_sampled_rate``): the cost rate is the mean,
  over uniformly sampled times of the cycle, of the rate differenced over
  a window from each; each window an exact solve. The policy at a time is
  a solve over a window from it (deterministic, cached).
- **A given rule** (``_evaluate``): its cost rate, by the same iteration
  with the decisions fixed, to price naive policies.

A failure is replaced by the next step, so the grid's cost rate is low by
about ``step / 2`` of each failure's time over the mean cycle: O(step).
Phase 2 (``_integration``) works between opportunities in continuous time.
"""

import math
import warnings
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Callable, ClassVar, Dict, Optional, cast

import numpy as np

from repyability.maintenance._calendar import (
    ANYTIME,
    Calendar,
    _fraction,
    _Grid,
)
from repyability.rbd.results import _ResultMapping
from repyability.utils import checks
from repyability.utils.wrappers import outside_level

#: Default grid: about this many steps to the mean life, put on the
#: opportunities' own grid where it can be.
STEPS_PER_LIFE = 1000
#: The chance of a unit outliving the oldest age on the grid.
FAR = 1e-9
#: The chance of a unit outliving the age taken as a life's span, for the
#: windows the policy at a time is worked out over.
SPAN = 1e-3
#: The most steps a cycle may have for relative value iteration; a longer
#: one is sampled.
MAX_CYCLE = 20_000
#: Relative value iteration stops when the span of a sweep's change is
#: within this share of it.
TOLERANCE = 1e-10
#: The most steps relative value iteration may take, in lives of the age
#: grid (sweeps times the cycle).
MAX_LIVES = 400
#: A window's settling length: this many of the longest gap between
#: planned options, and of a life's span.
SETTLE_GAPS = 2
SETTLE_SPANS = 3

METHODS = ("auto", "dp", "integration", "sampling")
TAILS = ("repeat", "anytime")


class _Ages:
    """A part's life on the age grid: ``p[j]``, the chance a unit of age
    ``j * step`` survives the next step (the oldest age kept as the
    oldest)."""

    def __init__(self, part, step: float, max_age: Optional[float]):
        self.step = step
        self.sf = _survival(part)
        self.mean = _mean(part, self.sf)
        if max_age is None:
            max_age = _age_where(self.sf, FAR, self.mean)
        self.span = _age_where(self.sf, SPAN, self.mean)
        size = int(math.ceil(max_age / step))
        ages = np.arange(size + 1) * step
        held = self.sf(ages)
        later = self.sf(ages + step)
        with np.errstate(divide="ignore", invalid="ignore"):
            p = np.where(held > 0.0, later / held, 0.0)
        self.p = np.clip(np.nan_to_num(p), 0.0, 1.0)
        self.size = size


def _survival(part) -> Callable[[np.ndarray], np.ndarray]:
    """The part's survival function, over an array of ages."""
    sf = part.reliability_function

    def survival(ages):
        ages = np.asarray(ages, float)
        with np.errstate(all="ignore"):
            try:
                out = np.asarray(sf(ages), float)
                if out.shape != ages.shape:
                    raise ValueError
            except (TypeError, ValueError):
                out = np.array([float(np.ravel(sf(a))[0]) for a in ages])
        return np.clip(np.nan_to_num(out, nan=0.0), 0.0, 1.0)

    return survival


def _mean(part, sf) -> float:
    from repyability.rbd._model_utils import MODEL_ERRORS, model_mean

    try:
        mean = float(model_mean(part.reliability))
    except MODEL_ERRORS:
        mean = math.nan
    if not 0.0 < mean < math.inf:
        mean = _age_where(sf, 0.5, 1.0)
    return mean


def _age_where(sf, chance: float, scale: float) -> float:
    """The age a unit outlives with ``chance``: doubled from ``scale``
    past it, then halved down to it."""
    low, high = 0.0, scale
    for _ in range(200):
        if float(sf(np.array([high]))[0]) <= chance:
            break
        low, high = high, 2.0 * high
    else:
        raise ValueError(
            "The life never falls to a chance of working of "
            f"{chance:g} (some units never fail): give max_age."
        )
    for _ in range(60):
        middle = 0.5 * (low + high)
        if float(sf(np.array([middle]))[0]) > chance:
            low = middle
        else:
            high = middle
    return high


def _default_step(calendar: Calendar, mean: float) -> float:
    """About ``mean / STEPS_PER_LIFE``, but a whole fraction of the
    largest step every opportunity's period, offset and dates lie on, so
    that they lie on the grid; with none such (or one far finer), the
    first."""
    base = mean / STEPS_PER_LIFE
    times = []
    for o in calendar.opportunities:
        if o.periodic:
            times += [o.period, o.offset]
        else:
            times += list(o.dates)
    fractions = [_fraction(abs(t)) for t in times if t != 0.0]
    if not fractions:
        return base
    common = float(
        Fraction(
            math.gcd(*(f.numerator for f in fractions)),
            math.lcm(*(f.denominator for f in fractions)),
        )
    )
    if common < base / 10.0:
        return base
    return common / math.ceil(common / base)


def _sweep(
    values: np.ndarray,
    costs: np.ndarray,
    p: np.ndarray,
    failure: float,
    beta: float = 1.0,
    fixed: Optional[np.ndarray] = None,
    record: Optional[np.ndarray] = None,
):
    """Backward over the steps of ``costs``, from ``values`` (the cost to
    go, by age index, after the last): each step, keep the part or
    replace it at the step's price (``inf``: no replacement).

    ``beta`` discounts a step (1: none). ``fixed`` gives, for a fixed
    rule, the age index from which each step replaces (-1: never),
    instead of the best. ``record`` gets the cost to go of a new part at
    each step.

    Returns the values at the first step, each step's first replacing age
    index (-1: none), and whether every step's replacing ages were all
    those from its first (a threshold).
    """
    first = np.full(costs.size, -1, dtype=np.int64)
    upper = True
    v = values
    shifted = np.empty_like(values)
    for n in range(costs.size - 1, -1, -1):
        shifted[:-1] = v[1:]
        shifted[-1] = v[-1]
        keep = p * shifted
        keep += (1.0 - p) * (failure + 0.5 * (v[0] + v[1]))
        if beta != 1.0:
            keep *= beta
        cost = costs[n]
        if cost < math.inf:
            replace = cost + keep[0]
            if fixed is None:
                act = replace < keep
                if act.any():
                    i = int(np.argmax(act))
                    first[n] = i
                    if upper and not act[i:].all():
                        upper = False
                v = np.minimum(keep, replace)
            else:
                i = int(fixed[n])
                first[n] = i
                if i >= 0:
                    keep[i:] = replace
                v = keep
        else:
            v = keep
        if record is not None:
            record[n] = v[0]
    return v, first, upper


@dataclass
class _Cycle:
    """Relative value iteration's result over one cycle."""

    rate: float
    low: float
    high: float
    first: np.ndarray
    upper: bool
    sweeps: int
    converged: bool


def _relative_values(
    costs: np.ndarray,
    ages: _Ages,
    failure: float,
    fixed: Optional[np.ndarray] = None,
) -> _Cycle:
    """The average cost rate over a periodic calendar of ``costs`` (one
    cycle of steps), by relative value iteration: sweep the cycle from the
    last relative values until the span of the change is within
    ``TOLERANCE`` of it. The rate is exact for the grid; ``low`` and
    ``high`` bound it."""
    h = np.zeros(ages.size + 1)
    length = costs.size * ages.step
    limit = max(1, int(MAX_LIVES * (ages.size + 1) / costs.size))
    low = high = math.nan
    for sweep in range(1, limit + 1):
        v, first, upper = _sweep(h, costs, ages.p, failure, fixed=fixed)
        change = v - h
        low, high = float(change.min()), float(change.max())
        h = v - v[0]
        if high - low <= TOLERANCE * max(abs(high), abs(low), 1e-300):
            return _Cycle(
                0.5 * (low + high) / length,
                low / length,
                high / length,
                first,
                upper,
                sweep,
                True,
            )
    warnings.warn(
        f"The cost rate did not settle in {limit} sweeps of the cycle: it "
        f"lies between {low / length:.6g} and {high / length:.6g}.",
        stacklevel=outside_level(),
    )
    return _Cycle(
        0.5 * (low + high) / length,
        low / length,
        high / length,
        first,
        upper,
        limit,
        False,
    )


def _window(grid: _Grid, start: int, n: int, ages: _Ages, failure: float):
    """A finite-horizon solve over ``n`` steps from grid index ``start``:
    the cost to go of a new part at each step, each step's first
    replacing age index and the kinds, and whether they are
    thresholds."""
    costs, kinds = grid.prices(start, n)
    record = np.empty(n)
    _, first, upper = _sweep(
        np.zeros(ages.size + 1), costs, ages.p, failure, record=record
    )
    return record, first, kinds, upper


def _propagate(share: np.ndarray, first: np.ndarray, p: np.ndarray):
    """Forward over the steps of ``first`` (each step's first replacing
    age index, -1 for none) from ``share``, the chance of the unit in
    place being of each age index: the share at the end, the chance of a
    planned replacement at each step, and the expected failures. A failed
    unit is renewed as the backward sweep has it: by the next step, as
    new or a step old, half each."""
    share = share.copy()
    replaced = np.zeros(first.size)
    failures = 0.0
    later = np.empty_like(share)
    for n in range(first.size):
        i = first[n]
        if i >= 0:
            moved = float(share[i:].sum())
            share[i:] = 0.0
            share[0] += moved
            replaced[n] = moved
        failed = share * (1.0 - p)
        lost = float(failed.sum())
        failures += lost
        later[0] = 0.0
        later[1:] = share[:-1] * p[:-1]
        later[-1] += share[-1] * p[-1]
        later[0] += 0.5 * lost
        later[1] += 0.5 * lost
        share, later = later, share
    return share, replaced, failures


def _stationary(first: np.ndarray, ages: _Ages):
    """The long-run chance of a planned replacement at each step of a
    cycle, and the expected failures over it: forward over the cycle from
    a new unit until the ages at its start settle (within
    ``TOLERANCE``)."""
    share = np.zeros(ages.size + 1)
    share[0] = 1.0
    limit = max(1, int(MAX_LIVES * (ages.size + 1) / first.size))
    for _ in range(limit):
        later, _, _ = _propagate(share, first, ages.p)
        settled = float(np.abs(later - share).sum()) <= TOLERANCE
        share = later
        if settled:
            break
    _, replaced, failures = _propagate(share, first, ages.p)
    return replaced, failures


def _summary(names, anytime, kinds, costs, replaced, failures, failure, time):
    """Each kind of replacement's rate and cost a unit of time, failures
    as ``"failure"``: from the chance of a planned replacement at each
    step (``replaced``), the steps' kinds and prices, and the failures,
    over ``time``."""
    shown = [n for n in names if n != ANYTIME or anytime is not None]
    rates = dict.fromkeys(shown, 0.0)
    spent = dict.fromkeys(shown, 0.0)
    for k, name in enumerate(names):
        if name not in rates:
            continue
        here = kinds == k
        rates[name] = float(replaced[here].sum() / time)
        spent[name] = float((replaced[here] * costs[here]).sum() / time)
    rates["failure"] = float(failures / time)
    spent["failure"] = float(failures * failure / time)
    return rates, spent


class _Thresholds:
    """Where a policy's thresholds come from: ``at(index)`` gives the
    first replacing age index and the kind of replacement at grid
    indices."""

    def at(self, index: np.ndarray):  # pragma: no cover - interface
        raise NotImplementedError


class _CycleTable(_Thresholds):
    """Thresholds repeating every cycle, and the long-run chance of a
    planned replacement at each step."""

    def __init__(self, first, kinds, chance):
        self.first, self.kinds, self.chance = first, kinds, chance

    def chances(self, n: int) -> np.ndarray:
        return self.chance[np.mod(np.arange(n), self.first.size)]

    def at(self, index):
        where = np.mod(index, self.first.size)
        return self.first[where], self.kinds[where]


class _WindowTable(_Thresholds):
    """Thresholds from finite-horizon solves: a table from 0, and past it
    blocks of steps, each solved over a window long enough for its end
    not to reach the block's decisions (cached)."""

    def __init__(self, grid, ages, failure, settle, first, kinds):
        self.grid, self.ages, self.failure = grid, ages, failure
        self.settle = settle
        self.first, self.kinds = first, kinds
        self.block = max(settle, 1024)
        self.blocks: Dict[int, tuple] = {}

    def _block(self, b: int):
        if b not in self.blocks:
            _, f, k, _ = _window(
                self.grid,
                b * self.block,
                self.block + self.settle,
                self.ages,
                self.failure,
            )
            self.blocks[b] = (f[: self.block], k[: self.block])
        return self.blocks[b]

    def chances(self, n: int) -> np.ndarray:
        """The chance of a planned replacement at each of the first ``n``
        steps, from a new unit at 0."""
        first, _ = self.at(np.arange(n))
        start = np.zeros(self.ages.size + 1)
        start[0] = 1.0
        return _propagate(start, first, self.ages.p)[1]

    def at(self, index):
        index = np.asarray(index, np.int64)
        first = np.empty(index.shape, np.int64)
        kinds = np.empty(index.shape, np.int64)
        table = self.first.size
        inside = (index >= 0) & (index < table)
        first[inside] = self.first[index[inside]]
        kinds[inside] = self.kinds[index[inside]]
        outside = ~inside
        if outside.any():
            blocks = np.floor_divide(index, self.block)
            for b in np.unique(blocks[outside]):
                f, k = self._block(int(b))
                here = outside & (blocks == b)
                first[here] = f[index[here] - b * self.block]
                kinds[here] = k[index[here] - b * self.block]
        return first, kinds


#: Steps a policy looks ahead at once for a unit's next planned
#: replacement.
_CHUNK = 1024


@dataclass(frozen=True)
class PlannedReplacement(_ResultMapping):
    """A unit's next planned replacement on a calendar policy: returned by
    [`CalendarPolicy.next_replacement`][repyability.CalendarPolicy.next_replacement].

    Attributes
    ----------
    time : float
        When, in the calendar's time (``inf``: none, the unit all but
        sure to fail first).
    kind : str or None
        The opportunity it falls at (``"anytime"`` for a day with no
        shut); None if none.
    price : float or None
        What the replacement costs there; None if none.
    """

    time: float
    kind: Optional[str]
    price: Optional[float]


@dataclass(frozen=True)
class CalendarPolicy(_ResultMapping):
    """The long-run optimal replacement policy of a part over a calendar
    of opportunities. Returned by
    [`NonRepairable.optimal_calendar_policy`][repyability.NonRepairable.optimal_calendar_policy].

    At each opportunity the policy replaces the part if it is at least a
    threshold age, which depends on the time: a shut shortly before a
    cheaper one waits for an older part than one long before it.
    ``schedule()`` lists the thresholds with the long-run chance of a
    replacement at each, ``next_replacement(age)`` says when a unit in
    service will be replaced, and ``replacement_rates`` and
    ``cost_rates`` say what the policy does and costs in the long run,
    by kind.

    Attributes
    ----------
    cost_rate : float
        The long-run average cost a unit of time, ``g``, whatever the
        start.
    cost_rate_se : float or None
        Its standard error, where it is estimated by sampling (None where
        it is worked out).
    replacement_rates : dict
        Replacements a unit of time in the long run, by kind: each
        opportunity's name, ``"anytime"`` (a day with no shut, where the
        calendar allows one) and ``"failure"``. A kind the policy never
        uses has 0.
    cost_rates : dict
        What each kind costs a unit of time in the long run (its rate
        times its price); they sum to ``cost_rate``.
    method : str
        ``"dp"`` (dynamic programming on the grid), ``"integration"``
        (between opportunities in continuous time) or ``"sampling"``
        (sampled windows of a long cycle).
    basis : str
        ``"exact"`` (exact for the grid, or for the quadrature),
        ``"differenced"`` (from a long finite horizon) or ``"sampled"``;
        the rates by kind are estimated from a new unit run in, where the
        cost rate is not exact.
    step : float or None
        The grid's step, sized from the life and put on the opportunities'
        times (None for integration, which has none).
    period : float or None
        The time after which the thresholds repeat (None if they do not,
        or too long to keep).
    determined_until : float or None
        For a calendar with given dates: the first time at which the
        policy depends on how the calendar goes on after its last date
        (repeating it, or leaving only replacement on any day). None for a
        periodic calendar.
    threshold_only : bool
        Whether, at every time worked out, the ages that replace are all
        those from the first (else the threshold is the youngest that
        replaces, and some older ones do not: a life whose hazard falls;
        the rates by kind then take every age from the youngest).
    settings : dict
        How it was worked out: the calendar's tail, sweeps, bounds,
        samples, windows.

    Examples
    --------
    A bearing with a Weibull life (scale 1000 days, shape 2.5), shuts
    every 91 days at 3 and every 364 days at 1, no replacement between
    them, and 20 at a failure:

    >>> import surpyval as surv
    >>> from repyability import Calendar, NonRepairable, Opportunity
    >>> bearing = NonRepairable(surv.Weibull.from_params([1000.0, 2.5]))
    >>> calendar = Calendar(
    ...     [
    ...         Opportunity("quarterly", 3.0, period=91.0),
    ...         Opportunity("annual", 1.0, period=364.0),
    ...     ],
    ...     failure=20.0,
    ...     anytime=None,
    ... )
    >>> policy = bearing.optimal_calendar_policy(calendar)
    >>> policy.method, policy.basis, policy.period
    ('dp', 'exact', 364.0)
    >>> round(policy.cost_rate * 364, 3)
    2.527
    >>> {k: round(v * 364, 3) for k, v in policy.replacement_rates.items()}
    {'quarterly': 0.0, 'annual': 0.962, 'failure': 0.078}
    >>> policy.next_replacement(age=300.0, now=200.0)
    PlannedReplacement(time=364.0, kind='annual', price=1.0)
    """

    # Bound after construction (``_bind``), not fields: where the
    # thresholds come from, the kinds' names, the calendar's grid and the
    # oldest age worked out.
    _table: ClassVar[Any] = None
    _names: ClassVar[Any] = None
    _grid: ClassVar[Any] = None
    _oldest: ClassVar[Any] = None

    cost_rate: float
    cost_rate_se: Optional[float]
    replacement_rates: Dict[str, float]
    cost_rates: Dict[str, float]
    method: str
    basis: str
    step: Optional[float]
    period: Optional[float]
    determined_until: Optional[float]
    threshold_only: bool
    settings: Dict[str, Any] = field(default_factory=dict)

    def _bind(self, table, names, grid=None, oldest=None) -> "CalendarPolicy":
        object.__setattr__(self, "_table", table)
        object.__setattr__(self, "_names", names)
        object.__setattr__(self, "_grid", grid)
        object.__setattr__(self, "_oldest", oldest)
        return self

    def _next_planned(self, installed: float, after: float):
        """When a unit put in at ``installed`` is next replaced on the
        policy, at or after ``after`` (strictly after, if put in then):
        its time, price and kind, ``(inf, 0.0, None)`` if it would outlive
        the oldest age the policy was worked out to first. What a
        ``RepairableRBD`` component on a ``"calendar"`` preventive policy
        asks at each of its renewals."""
        if self._grid is None:
            return self._table.next_planned(installed, after, self._oldest)
        step = cast(float, self.step)
        if after <= installed:
            n = int(math.floor(after / step + 1e-9)) + 1
        else:
            n = int(math.ceil(after / step - 1e-9))
        limit = int(math.ceil((installed + self._oldest) / step)) + 1
        while n <= limit:
            index = np.arange(n, min(n + _CHUNK, limit + 1))
            first, kinds = self._table.at(index)
            ages = index * step - installed
            due = (first >= 0) & (ages >= first * step - 1e-9 * step)
            if due.any():
                at = int(np.argmax(due))
                i = int(index[at])
                price, _ = self._grid.prices(i, 1)
                return i * step, float(price[0]), self._names[kinds[at]]
            n += _CHUNK
        return math.inf, 0.0, None

    def _planned_price(self, t: float) -> float:
        """The price of a planned replacement at time ``t``, whatever the
        unit's age: the opportunity's if one falls there (on a grid, the
        one laid at the grid time nearest ``t``), else the calendar's
        price on any day; ``inf`` where the calendar allows none. What a
        ``RepairableRBD`` component on a ``"calendar"`` preventive policy
        is charged when a stop of its maintenance group renews it early
        (#349)."""
        if self._grid is None:
            return self._table.price_at(t)
        step = cast(float, self.step)
        price, _ = self._grid.prices(int(np.rint(t / step)), 1)
        return float(price[0])

    def _threshold(self, t):
        """The threshold age at time (or times) ``t``: ``inf`` where no
        replacement is open, or none pays (for the tests and the
        diagrams)."""
        times = checks.real_array(t, "t")
        index = (
            times
            if self.step is None
            else np.rint(times / self.step).astype(np.int64)
        )
        values, _ = self._table.at(np.ravel(index))
        if self.step is not None:
            values = np.where(values >= 0, values * self.step, math.inf)
        out = np.asarray(values, float).reshape(np.shape(index))
        return float(out) if out.ndim == 0 else out

    def next_replacement(self, age, now=0.0) -> PlannedReplacement:
        """When the unit in service, ``age`` old at time ``now``, will be
        replaced on the policy, if it does not fail first.

        Parameters
        ----------
        age : float
            The unit's age now, at least 0 (0: just put in, so not
            replaced again now).
        now : float, optional
            The time now, in the calendar's time, by default 0.

        Returns
        -------
        PlannedReplacement
            When, at which opportunity, and at what price.
        """
        if not (checks.is_number(age) and 0.0 <= float(age) < math.inf):
            raise ValueError(f"age must be a number, at least 0; got {age!r}.")
        if not (checks.is_number(now) and math.isfinite(float(now))):
            raise ValueError(f"now must be a finite number; got {now!r}.")
        now = float(now)
        time, price, kind = self._next_planned(now - float(age), now)
        if kind is None:
            return PlannedReplacement(math.inf, None, None)
        return PlannedReplacement(float(time), kind, float(price))

    def schedule(self, kinds=(), until=None) -> list:
        """The planned replacements open from time 0 up to ``until``, with
        each one's threshold and the chance the policy replaces the unit
        there.

        Parameters
        ----------
        kinds : tuple of str, optional
            The kinds to list, by name; by default every opportunity.
            ``"anytime"`` among them lists the days with only a replacement
            on any day open too.
        until : float, optional
            The time to list them up to (not including). By default the
            period over which the thresholds repeat, or the end of a
            calendar's given dates; give it where there is neither.

        Returns
        -------
        list of (float, str, float, float)
            Each as ``(time, kind, threshold, chance)``: replace the unit
            there if it is at least ``threshold`` old, which happens with
            ``chance`` in the long run (a periodic calendar's), or from a
            new unit at 0 (given dates, or a cycle too long to keep). An
            opportunity the policy never uses has a chance of 0.
        """
        if until is None:
            until = self.period or self.settings.get("table_until")
            if until is None:
                raise ValueError(
                    "The policy does not repeat: give until, the time to "
                    "list the opportunities up to."
                )
        if not (checks.is_number(until) and 0.0 < float(until) < math.inf):
            raise ValueError(
                f"until must be a number, more than 0; got {until!r}."
            )
        wanted = set(kinds) if kinds else None
        unknown = (wanted or set()) - set(self._names)
        if unknown:
            raise ValueError(
                f"Unknown kinds {sorted(unknown)}: the calendar's are "
                f"{self._names}."
            )
        if self.step is None:
            return self._table.schedule(float(until), wanted)
        n = int(math.ceil(float(until) / self.step - 1e-9))
        first, kind = self._table.at(np.arange(n))
        chance = self._table.chances(n)
        out = []
        for i in np.flatnonzero(kind >= 0):
            name = self._names[kind[i]]
            if wanted is None and name == ANYTIME:
                continue
            if wanted is not None and name not in wanted:
                continue
            age = first[i] * self.step if first[i] >= 0 else math.inf
            out.append(
                (float(i * self.step), name, float(age), float(chance[i]))
            )
        return out


def _resolve_step(step, calendar, part) -> float:
    if step is None:
        sf = _survival(part)
        return _default_step(calendar, _mean(part, sf))
    if not (checks.is_number(step) and 0.0 < float(step) < math.inf):
        raise ValueError(f"step must be a number, more than 0; got {step!r}.")
    return float(step)


def _settle(grid: _Grid, ages: _Ages) -> int:
    """A window's settling length, in steps: the policy at a time depends
    on the calendar this far ahead of it, no further."""
    calendar = grid.calendar
    gaps = [o.period for o in calendar.opportunities if o.periodic]
    if grid.repeat is not None:
        gaps.append(grid.repeat)
    if grid.plan_end is not None:
        gaps.append(abs(grid.plan_end))
    longest = max(gaps, default=0.0)
    length = SETTLE_GAPS * longest + SETTLE_SPANS * ages.span
    return int(math.ceil(length / grid.step))


def optimal_calendar_policy(
    part,
    calendar,
    *,
    step=None,
    max_age=None,
    method="auto",
    tail="repeat",
    mc_samples=64,
    seed=None,
) -> CalendarPolicy:
    """The work of ``NonRepairable.optimal_calendar_policy`` (see
    there)."""
    if not isinstance(calendar, Calendar):
        raise TypeError(f"calendar must be a Calendar; got {calendar!r}.")
    checks.one_of("method", method, METHODS)
    checks.one_of("tail", tail, TAILS)
    if max_age is not None and not (
        checks.is_number(max_age) and 0.0 < float(max_age) < math.inf
    ):
        raise ValueError(
            f"max_age must be a number, more than 0; got {max_age!r}."
        )
    failure, anytime = calendar._prices(part)
    dated = not calendar.periodic
    if method == "integration":
        from repyability.maintenance._integration import integrated_policy

        if anytime is not None:
            raise NotImplementedError(
                "method='integration' works between opportunities, with no "
                "replacement on any day; this calendar has one (anytime "
                f"{anytime:g}): use method='dp', or anytime=None."
            )
        return integrated_policy(
            part, calendar, failure, max_age=max_age, tail=tail
        )
    step = _resolve_step(step, calendar, part)
    ages = _Ages(part, step, None if max_age is None else float(max_age))
    grid = _Grid(calendar, anytime, step, tail=tail if dated else None)
    names = grid.names
    settle = _settle(grid, ages)
    short = grid.cycle is not None and grid.cycle <= MAX_CYCLE
    if method == "auto":
        method = "sampling" if (not dated and not short) else "dp"
    if method == "sampling":
        if dated:
            raise NotImplementedError(
                "method='sampling' samples times of a periodic calendar; "
                "this one has given dates: use method='dp'."
            )
        if (
            mc_samples is None
            or not checks.is_whole(mc_samples)
            or int(mc_samples) < 2
        ):
            raise ValueError(
                f"mc_samples must be a whole number, at least 2; got "
                f"{mc_samples!r}."
            )
        return _sampled_policy(
            grid, ages, failure, settle, int(mc_samples), seed, names
        )
    if not dated and short:
        return _cycle_policy(grid, ages, failure, names, calendar)
    if not dated:
        return _differenced_policy(grid, ages, failure, settle, names)
    return _dated_policy(
        part, calendar, anytime, ages, failure, settle, tail, names
    )


def _cycle_policy(grid, ages, failure, names, calendar) -> CalendarPolicy:
    costs, kinds = grid.prices(0, grid.cycle)
    cycle = _relative_values(costs, ages, failure)
    _warn_not_threshold(cycle.upper)
    replaced, failures = _stationary(cycle.first, ages)
    length = grid.cycle * grid.step
    rates, spent = _summary(
        names, grid.anytime, kinds, costs, replaced, failures, failure, length
    )
    policy = CalendarPolicy(
        cost_rate=cycle.rate,
        cost_rate_se=None,
        replacement_rates=rates,
        cost_rates=spent,
        method="dp",
        basis="exact",
        step=grid.step,
        period=length,
        determined_until=None,
        threshold_only=cycle.upper,
        settings={
            "sweeps": cycle.sweeps,
            "converged": cycle.converged,
            "cost_rate_bounds": (cycle.low, cycle.high),
            "cycle_steps": grid.cycle,
            "max_age": ages.size * ages.step,
        },
    )
    table = _CycleTable(cycle.first, kinds, replaced)
    return policy._bind(table, names, grid, ages.size * ages.step)


def _differenced(record, n1: int, n2: int, step: float) -> float:
    return float((record[n1] - record[n2]) / ((n2 - n1) * step))


def _run_in(grid, ages, failure, start, settle, span):
    """A window from grid index ``start``: a new unit run in over
    ``settle`` steps, then ``span`` steps, and a settling length for the
    end. The cost rate differenced over the ``span`` steps, and each kind's
    replacements and failures over them (from the new unit run in)."""
    record, first, kinds, upper = _window(
        grid, start, settle + span + settle, ages, failure
    )
    rate = _differenced(record, settle, settle + span, grid.step)
    new = np.zeros(ages.size + 1)
    new[0] = 1.0
    settled, _, _ = _propagate(new, first[:settle], ages.p)
    window = slice(settle, settle + span)
    _, replaced, failures = _propagate(settled, first[window], ages.p)
    costs, _ = grid.prices(start + settle, span)
    return rate, replaced, failures, kinds[window], costs, upper


def _differenced_policy(grid, ages, failure, settle, names) -> CalendarPolicy:
    """A periodic calendar too long to sweep, by a long finite horizon:
    the cost rate differenced over ``2 * settle`` steps, after a settling
    length."""
    span = 2 * max(settle, 1)
    rate, replaced, failures, kinds, costs, upper = _run_in(
        grid, ages, failure, 0, settle, span
    )
    _warn_not_threshold(upper)
    rates, spent = _summary(
        names,
        grid.anytime,
        kinds,
        costs,
        replaced,
        failures,
        failure,
        span * grid.step,
    )
    policy = CalendarPolicy(
        cost_rate=rate,
        cost_rate_se=None,
        replacement_rates=rates,
        cost_rates=spent,
        method="dp",
        basis="differenced",
        step=grid.step,
        period=None,
        determined_until=None,
        threshold_only=upper,
        settings={
            "differenced_over": span * grid.step,
            "settle": settle * grid.step,
            "table_until": span * grid.step,
            "max_age": ages.size * ages.step,
        },
    )
    _, first, kinds, _ = _window(grid, 0, span + settle, ages, failure)
    table = _WindowTable(
        grid, ages, failure, settle, first[:span], kinds[:span]
    )
    return policy._bind(table, names, grid, ages.size * ages.step)


def _sampled_rate(grid, ages, failure, settle, samples, rng):
    """The cost rate as the mean, over uniformly sampled times of the
    cycle, of the rate differenced over a window from each (the cost to
    go of a new part there, less at ``span`` steps on, over the time
    between), and its standard error; and each kind's rate, from a new
    unit run in before each window."""
    calendar = grid.calendar
    longest = max(
        [o.period for o in calendar.opportunities if o.periodic] + [ages.mean]
    )
    span = int(math.ceil(longest / grid.step))
    cycle = grid.cycle if grid.cycle is not None else 2**62
    rates = np.empty(samples)
    by_kind = np.zeros(len(grid.names))
    spent = np.zeros(len(grid.names))
    failures = 0.0
    for i in range(samples):
        start = int(rng.integers(0, cycle)) - settle
        rate, replaced, failed, kinds, costs, _ = _run_in(
            grid, ages, failure, start, settle, span
        )
        rates[i] = rate
        open_ = kinds >= 0
        np.add.at(by_kind, kinds[open_], replaced[open_])
        np.add.at(spent, kinds[open_], replaced[open_] * costs[open_])
        failures += failed
    time = samples * span * grid.step
    shown = [n for n in grid.names if n != ANYTIME or grid.anytime is not None]
    kinds_rates = {
        n: float(by_kind[k] / time)
        for k, n in enumerate(grid.names)
        if n in shown
    }
    kinds_costs = {
        n: float(spent[k] / time)
        for k, n in enumerate(grid.names)
        if n in shown
    }
    kinds_rates["failure"] = float(failures / time)
    kinds_costs["failure"] = float(failures * failure / time)
    return (
        float(rates.mean()),
        float(rates.std(ddof=1) / math.sqrt(samples)),
        kinds_rates,
        kinds_costs,
    )


def _sampled_policy(
    grid, ages, failure, settle, samples, seed, names
) -> CalendarPolicy:
    rng = np.random.default_rng(checks.seed(seed))
    rate, error, rates, spent = _sampled_rate(
        grid, ages, failure, settle, samples, rng
    )
    _, first, kinds, upper = _window(grid, 0, 1 + settle, ages, failure)
    policy = CalendarPolicy(
        cost_rate=rate,
        cost_rate_se=error,
        replacement_rates=rates,
        cost_rates=spent,
        method="sampling",
        basis="sampled",
        step=grid.step,
        period=None,
        determined_until=None,
        threshold_only=upper,
        settings={
            "mc_samples": samples,
            "settle": settle * grid.step,
            "cycle": None if grid.cycle is None else grid.cycle * grid.step,
            "max_age": ages.size * ages.step,
        },
    )
    table = _WindowTable(grid, ages, failure, settle, first[:1], kinds[:1])
    return policy._bind(table, names, grid, ages.size * ages.step)


def _dated_policy(
    part, calendar, anytime, ages, failure, settle, tail, names
) -> CalendarPolicy:
    """A calendar with given dates: the policy over the plan and a
    settling length past it, solved with the chosen tail; with the other
    tail too, for the first time the two differ. The cost rate, and each
    kind's, are the repeated plan's."""
    step = ages.step
    end = max(calendar.plan_end, 0.0)
    table = int(math.ceil(end / step)) + settle
    tables = {}
    for name in TAILS:
        grid = _Grid(calendar, anytime, step, tail=name)
        _, first, kinds, upper = _window(
            grid, 0, table + settle, ages, failure
        )
        tables[name] = (grid, first[:table], kinds[:table], upper)
    grid, first, kinds, upper = tables[tail]
    other = tables["anytime" if tail == "repeat" else "repeat"]
    differ = np.flatnonzero((first != other[1]) | (kinds != other[2]))
    until = (differ[0] if differ.size else table) * step
    repeated = tables["repeat"][0]
    if repeated.cycle is not None and repeated.cycle <= MAX_CYCLE:
        costs, cycle_kinds = repeated.prices(
            repeated.cycle_start, repeated.cycle
        )
        cycle = _relative_values(costs, ages, failure)
        replaced, failures = _stationary(cycle.first, ages)
        rate, basis = cycle.rate, "exact"
        time = repeated.cycle * step
    else:
        span = 2 * max(settle, 1)
        rate, replaced, failures, cycle_kinds, costs, _ = _run_in(
            repeated, ages, failure, 0, settle, span
        )
        basis = "differenced"
        time = span * step
    rates, spent = _summary(
        names, anytime, cycle_kinds, costs, replaced, failures, failure, time
    )
    _warn_not_threshold(upper)
    policy = CalendarPolicy(
        cost_rate=rate,
        cost_rate_se=None,
        replacement_rates=rates,
        cost_rates=spent,
        method="dp",
        basis=basis,
        step=step,
        period=None,
        determined_until=float(until),
        threshold_only=upper,
        settings={
            "tail": tail,
            "repeat": repeated.repeat,
            "table_until": end + step,
            "settle": settle * step,
            "max_age": ages.size * ages.step,
        },
    )
    window = _WindowTable(grid, ages, failure, settle, first, kinds)
    return policy._bind(window, names, grid, ages.size * ages.step)


def _warn_not_threshold(upper: bool) -> None:
    if not upper:
        warnings.warn(
            "At some times the ages that replace are not all those from "
            "the youngest (a life whose hazard falls): the threshold given "
            "is the youngest age that replaces, and some older ones are "
            "kept.",
            stacklevel=outside_level(),
        )


def _rule_prices(grid: _Grid, rule: dict, start: int, n: int):
    """For a fixed rule (the age from which each kind replaces), each
    step's price and first replacing age index, taking at each step the
    kind with the youngest age in the rule (the cheapest at a tie)."""
    step = grid.step
    costs = np.full(n, math.inf)
    first = np.full(n, -1, dtype=np.int64)
    best = np.full(n, math.inf)
    calendar = grid.calendar
    options = []
    if grid.anytime is not None and ANYTIME in rule:
        options.append((ANYTIME, grid.anytime, np.arange(n)))
    low, high = (start - 0.5) * step, (start + n - 0.5) * step
    for o in calendar.opportunities:
        if o.name not in rule:
            continue
        times = grid._dates(o, low - step, high + step)
        index = np.rint(times / step).astype(np.int64) - start
        options.append((o.name, o.cost, index[(index >= 0) & (index < n)]))
    for name, price, index in options:
        age = float(rule[name])
        i = int(math.ceil(age / step - 1e-9))
        better = (age < best[index]) | (
            (age == best[index]) & (price < costs[index])
        )
        chosen = index[better]
        best[chosen] = age
        costs[chosen] = price
        first[chosen] = i
    return costs, first


def calendar_policy_cost(
    part, calendar, rule, *, step=None, max_age=None, tail="repeat"
) -> float:
    """The work of ``NonRepairable.calendar_policy_cost`` (see there)."""
    if not isinstance(calendar, Calendar):
        raise TypeError(f"calendar must be a Calendar; got {calendar!r}.")
    checks.one_of("tail", tail, TAILS)
    if not isinstance(rule, dict):
        raise TypeError(
            "rule must be a dict of the age from which each kind of "
            f"replacement replaces the part; got {rule!r}."
        )
    names = [o.name for o in calendar.opportunities] + [ANYTIME]
    for name, age in rule.items():
        if name not in names:
            raise ValueError(
                f"Unknown kind {name!r} in the rule: the calendar's are "
                f"{names}."
            )
        if not (checks.is_number(age) and 0.0 <= float(age)):
            raise ValueError(
                f"The age for {name!r} must be a number, at least 0 (inf "
                f"for never); got {age!r}."
            )
    rule = {k: float(v) for k, v in rule.items() if float(v) < math.inf}
    failure, anytime = calendar._prices(part)
    if ANYTIME in rule and anytime is None:
        raise ValueError(
            "The rule replaces on any day, which the calendar does not "
            "allow (its anytime is None)."
        )
    step = _resolve_step(step, calendar, part)
    ages = _Ages(part, step, None if max_age is None else float(max_age))
    dated = not calendar.periodic
    grid = _Grid(calendar, anytime, step, tail="repeat" if dated else None)
    if grid.cycle is not None and grid.cycle <= MAX_CYCLE:
        costs, first = _rule_prices(grid, rule, grid.cycle_start, grid.cycle)
        return _relative_values(costs, ages, failure, fixed=first).rate
    settle = _settle(grid, ages)
    span = max(settle, 1)
    n = 2 * span + settle
    costs, first = _rule_prices(grid, rule, 0, n)
    record = np.empty(n)
    _sweep(
        np.zeros(ages.size + 1),
        costs,
        ages.p,
        failure,
        fixed=first,
        record=record,
    )
    return _differenced(record, 0, 2 * span, step)
