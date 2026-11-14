"""A part's cost over a window from its state now, with its spread (UC-8
of this package's ``README.md``).

One part, renewed as new at every failure (at its ``unplanned_cost``) and
by a policy (at its ``planned_cost``), in no time either way. From its
state now (its age, and for block replacement how far it is into the
interval), the cost over a window ``[0, W)`` is ``cu N_f + cp N_p``, with
``N_f`` the failures and ``N_p`` the planned replacements in the window.
Everything follows from the joint distribution of the two counts: the
mean, the standard deviation, the percentiles, the chance of exceeding a
budget. An event at exactly ``W`` falls in the next window, as for
[`RepairableRBD.expected_events`][repyability.RepairableRBD.expected_events].

**Cycles.** The policies renew the part, so its history is a run of
cycles, each ending in a failure or a planned replacement. Under age
replacement at ``T`` a new part's cycle ends in a failure before ``T``
(sub-distribution ``G(x) = F(min(x, T))``, mass ``q = F(T)``) or in a
planned replacement at exactly ``T`` (chance ``p = R(T)``). The part in
now, ``a`` old, has a first cycle of its own: a failure before ``tau =
T - a``, ``G_1(x) = 1 - S(min(x, tau))`` with ``S(u) = R(a + u) /
R(a)``, or a planned replacement at ``tau`` (chance ``p_1 = S(tau)``; a
part at or past ``T`` is replaced at once, ``tau = 0``). Run to failure is
``T`` infinite: no planned replacement.

**The counts.** A planned cycle has a fixed length, so the time taken by
``i`` failure cycles and ``j`` planned ones is the failure cycles' total
plus ``j T``, whatever their order: the ``C(i + j - 1, j)`` orders of the
later cycles share one distribution. With ``A_i`` the distribution
function of the first failure cycle plus ``i - 1`` new ones, and ``B_i``
of ``i`` new failure cycles (both sub-distributions), the chance that a
total ``U`` of cycles ends before ``y`` and the next cycle runs past it is
``Q[K](y) = K(y) - K'(y) - p K(y - T)``, ``K'`` the next of the sequence,
and::

    P(0, 0)     = S(W)                                   (W <= tau)
    P(0, j)     = p_1 p^(j-1) R(z)                       (0 < z <= T)
    P(i, j)    += C(i - 1 + j, j) p^j Q[A_i](W - j T)     (i >= 1)
    P(i, j)    += C(i + j - 1, i) p_1 p^(j-1) Q[B_i](z)   (i, j >= 1)

with ``z = W - tau - (j - 1) T``. Under block replacement every ``T``, the
planned replacements fall at fixed times, so ``N_p`` is known, and the
part runs to failure between them, from new after each: ``N_f`` is the
sum of independent counts, one for each stretch between planned
replacements (the first from the part's age), each from run to failure,
and its distribution their convolution.

**Numerically.** ``A_i`` and ``B_i`` are worked out on a grid of step
``h`` sized from the lives (``_STEPS_PER_SPREAD`` steps per interquartile
range of the life new and of the part's life left, at least
``_MIN_STEPS`` over the span and at most ``_MAX_STEPS``; never in
absolute time): ``K_(i+1)(x) = integral of K_i(x - u) dG(u)``, with the
exact increments of ``G`` over each cell and ``K_i`` averaged over the
cell (the trapezoidal rule in the measure), whose error falls as ``h **
2``; each step is one convolution by FFT. ``A_1`` and ``B_1`` are taken
exactly, and the others between grid points linearly. The sequences stop
when what is left is below ``_TAIL``. The same is worked out with a step
of ``2 h``, and ``precision`` is the largest difference of a probability
between the two (about three times the error at ``h``), plus the mass
the counts leave out.

**A calendar policy** (a ``CalendarPolicy``: at each opportunity, replace
the part if it is at least that time's threshold age) prices its planned
replacements by kind of opportunity and starts each cycle at a time of
its own: ``_window_calendar`` works it out.

**Simulated** (``method="simulate"``): the part's cycles drawn, seeded,
from its life (the part's life left from its age first), as a check, with
each probability's share of the simulations.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
from scipy.special import gammaln

from repyability.rbd.results import _ResultMapping
from repyability.utils import checks

#: Grid steps per interquartile range of a life, at least over the span,
#: and at most (see the module's docstring).
_STEPS_PER_SPREAD = 200
_MIN_STEPS = 2_000
_MAX_STEPS = 2**16
#: The sequences of counts stop when the chance of more is below this.
_TAIL = 1e-13
#: The most failures in a window the numerical route follows.
_MAX_COUNT = 5_000
#: The simulations by default.
_SIMULATIONS = 10_000
#: The percentiles reported by default.
LEVELS = (10.0, 50.0, 90.0)

RUN_TO_FAILURE = "run to failure"
AGE = "age replacement"
BLOCK = "block replacement"
CALENDAR = "calendar"


@dataclass
class CostDistribution(_ResultMapping):
    """What a part's maintenance costs over a window from its state now,
    and how widely that can vary: returned by
    [`NonRepairable.cost_distribution`][repyability.NonRepairable.cost_distribution].

    The cost over the window is the failures in it at the unplanned cost
    plus the planned replacements at the planned cost, so it comes in
    steps: ``costs`` lists every amount it can be and
    ``cost_probabilities`` the chance of each, from which the mean, the
    spread, the percentiles and the chance of exceeding a budget follow.
    ``probabilities[i, j]`` is the chance of exactly ``i`` failures and
    ``j`` planned replacements.

    Attributes
    ----------
    start : float
        When the window starts, from now (0 for
        [`cost_distribution`][repyability.NonRepairable.cost_distribution]).
    window : float
        The window's length.
    policy : str
        ``"run to failure"``, ``"age replacement"``, ``"block
        replacement"`` or ``"calendar"`` (a
        [`CalendarPolicy`][repyability.CalendarPolicy]).
    every : float or None
        The policy's interval (None for run to failure and a calendar
        policy).
    age : float
        The part's age now.
    since_renewal : float or None
        For block replacement, how far into the interval the part is now.
    unplanned_cost : float
        What a failure costs.
    planned_cost : float or None
        What a planned replacement costs (None for run to failure, and for
        a calendar policy, whose price depends on the opportunity:
        ``kind_prices``).
    method : str
        ``"numerical"`` (worked out on a grid, to ``precision``) or
        ``"simulated"``.
    probabilities : numpy.ndarray
        The chance of each number of failures (rows, from 0) and of
        planned replacements (columns, from 0).
    failure_probabilities : numpy.ndarray
        The chance of each number of failures, from 0.
    planned_probabilities : numpy.ndarray
        The chance of each number of planned replacements, from 0.
    costs : numpy.ndarray
        Every cost the window can have, cheapest first.
    cost_probabilities : numpy.ndarray
        The chance of each of ``costs``.
    mean : float
        The expected cost.
    standard_deviation : float
        The cost's standard deviation.
    expected_failures : float
        The expected number of failures.
    expected_planned : float
        The expected number of planned replacements.
    levels : tuple of float
        The percentiles of ``percentiles``, from 0 to 100.
    percentiles : numpy.ndarray
        The cost at each of ``levels``: the least cost not exceeded with
        at least that chance.
    budget : float or None
        The budget given, if any.
    probability_above_budget : float or None
        The chance the cost exceeds ``budget``.
    precision : float or None
        For the numerical route, the largest error of any probability
        (estimated, see the guide); None when simulated.
    standard_error : float or None
        For a simulation, the standard error of ``mean``; None otherwise.
    n_simulations : int or None
        The number of simulations, or None.
    kinds : tuple of str or None
        For a calendar policy, the kinds of planned replacement it makes
        in the window (its opportunities' names, ``"anytime"`` for a day
        with none), in the order of ``kind_prices``; None otherwise.
    kind_prices : tuple of float or None
        For a calendar policy, the price of a planned replacement of each
        of ``kinds``.
    kind_outcomes : list of dict or None
        For a calendar policy, each outcome with a chance of at least
        1e-12, by kind: ``{"failures": i, "planned": {kind: count},
        "cost": ..., "probability": ...}``, most likely first (what
        ``outcomes()`` returns).

    Examples
    --------
    See
    [`NonRepairable.cost_distribution`][repyability.NonRepairable.cost_distribution].
    """

    start: float
    window: float
    policy: str
    every: Optional[float]
    age: float
    since_renewal: Optional[float]
    unplanned_cost: float
    planned_cost: Optional[float]
    method: str
    probabilities: np.ndarray
    failure_probabilities: np.ndarray
    planned_probabilities: np.ndarray
    costs: np.ndarray
    cost_probabilities: np.ndarray
    mean: float
    standard_deviation: float
    expected_failures: float
    expected_planned: float
    levels: Tuple[float, ...]
    percentiles: np.ndarray
    budget: Optional[float]
    probability_above_budget: Optional[float]
    precision: Optional[float]
    standard_error: Optional[float]
    n_simulations: Optional[int]
    kinds: Optional[Tuple[str, ...]] = None
    kind_prices: Optional[Tuple[float, ...]] = None
    kind_outcomes: Optional[List[Dict[str, Any]]] = None

    def percentile(self, q: float) -> float:
        """The cost at the ``q``-th percentile: the least cost not
        exceeded with a chance of at least ``q`` / 100.

        Parameters
        ----------
        q : float
            The percentile, from 0 to 100.
        """
        return _percentile(self.costs, self.cost_probabilities, _level(q))

    def probability_above(self, budget: float) -> float:
        """The chance the cost exceeds ``budget``.

        Parameters
        ----------
        budget : float
            The cost to exceed.
        """
        return _above(self.costs, self.cost_probabilities, _budget(budget))

    def outcomes(self) -> list:
        """Each outcome with a chance of at least 1e-12, most likely
        first: ``[{"failures": i, "planned": j, "cost": ..., "probability":
        ...}]``; for a calendar policy, ``"planned"`` is a count by kind,
        ``{kind: count}``."""
        if self.kind_outcomes is not None:
            return list(self.kind_outcomes)
        rows, cols = np.nonzero(self.probabilities >= 1e-12)
        out = [
            {
                "failures": int(i),
                "planned": int(j),
                "cost": float(self._cost(i, j)),
                "probability": float(self.probabilities[i, j]),
            }
            for i, j in zip(rows, cols)
        ]
        return sorted(out, key=lambda o: -o["probability"])

    def _cost(self, i, j):
        return self.unplanned_cost * i + (self.planned_cost or 0.0) * j


# -- inputs -------------------------------------------------------------------


def _number(value, name: str, strict: bool = False) -> float:
    ok = (
        checks.is_number(value)
        and math.isfinite(float(value))
        and (float(value) > 0.0 if strict else float(value) >= 0.0)
    )
    if not ok:
        bound = "more than 0" if strict else "at least 0"
        raise ValueError(f"{name} must be a number, {bound}; got {value!r}.")
    return float(value)


def _level(q) -> float:
    if not (checks.is_number(q) and 0.0 <= float(q) <= 100.0):
        raise ValueError(
            f"A percentile must be a number from 0 to 100; got {q!r}."
        )
    return float(q)


def _budget(budget) -> float:
    if not (checks.is_number(budget) and not math.isnan(float(budget))):
        raise ValueError(f"budget must be a number; got {budget!r}.")
    return float(budget)


class _Plan(NamedTuple):
    """The policy and the part's state, checked."""

    policy: str
    every: float  # inf for run to failure and a calendar policy
    age: float
    since: Optional[float]  # block replacement only
    calendar: Any = None  # a calendar policy's CalendarPolicy


def _plan(part, every, resets, age, since_renewal, calendar=None) -> _Plan:
    """The policy from ``every`` and ``resets`` (as ``renewal_forecast``
    takes them) or ``calendar``, and the part's state, checked."""
    age = _number(age, "age")
    if calendar is not None:
        from repyability.maintenance._window_calendar import check_policy

        if not (every is None and resets is None and since_renewal is None):
            raise ValueError(
                "A calendar policy says when to replace by its thresholds "
                "at its opportunities: give no every, resets or "
                "since_renewal with calendar."
            )
        check_policy(part, calendar)
        return _Plan(CALENDAR, math.inf, age, None, calendar)
    if every is None:
        if resets is not None or since_renewal is not None:
            raise ValueError(
                "Run to failure (no every given) has no interval: give "
                "resets and since_renewal only with every."
            )
        return _Plan(RUN_TO_FAILURE, math.inf, age, None)
    every = _number(every, "every", strict=True)
    if not isinstance(resets, bool):
        raise TypeError(
            "resets must be True (age replacement: each interval starts at "
            "the last replacement, planned or not) or False (block "
            "replacement: the planned replacements keep to fixed times); "
            f"got {resets!r}."
        )
    if not hasattr(part, "cp"):
        raise ValueError(
            "A policy with planned replacements needs the part's "
            "planned_cost: give NonRepairable(..., planned_cost=...)."
        )
    if resets:
        if since_renewal is not None:
            raise ValueError(
                "Age replacement counts its interval from the last "
                "replacement, which is the part's age: give no "
                "since_renewal."
            )
        return _Plan(AGE, every, age, None)
    since = age if since_renewal is None else since_renewal
    since = _number(since, "since_renewal")
    if since >= every:
        raise ValueError(
            f"The part is {since:g} into an interval of {every:g}: a "
            "planned replacement is due now or past. Give a time into the "
            "interval less than every."
        )
    return _Plan(BLOCK, every, age, since)


def _check_part(part) -> None:
    """The part's life must have a survival function in closed form and
    no failures at age 0, and it must be replaced in no time."""
    if not hasattr(part, "cu"):
        raise ValueError(
            "The cost over a window needs the part's unplanned_cost: give "
            "NonRepairable(..., unplanned_cost=...)."
        )
    if part.model_parameterization not in ("parametric", "degradation"):
        raise NotImplementedError(
            "cost_distribution needs a life with a survival function in "
            "closed form (a surpyval parametric distribution, mixture or "
            "degradation process); this part's is "
            f"{part.model_parameterization}."
        )
    if float(np.ravel(part.reliability_function(0.0))[0]) < 1.0:
        raise ValueError(
            "The part's life fails at age 0 with a chance above 0 (a "
            "zero-inflated life): a new part could fail at once, again and "
            "again, so the counts have no distribution. Give a life with "
            "no failures at age 0."
        )
    replace = float(np.ravel(part.time_to_replace.mean())[0])
    if replace != 0.0:
        raise ValueError(
            "cost_distribution takes a replacement as done in no time, and "
            f"this part takes {replace:g} on average: give a part replaced "
            "in no time (time_to_replace None or 'instant')."
        )


# -- the lives ----------------------------------------------------------------


class _Life:
    """The part's survival function, new and from its age now."""

    def __init__(self, part, age: float):
        self.part = part
        self.sf = part.reliability_function
        self.age = age
        self.base = float(np.ravel(self.sf(age))[0])
        if not self.base > 0.0:
            raise ValueError(
                f"The life gives no chance of a part working at age {age:g}."
            )

    def new(self) -> "_Life":
        """A new part's life."""
        return _Life(self.part, 0.0)

    def R(self, x) -> np.ndarray:
        """The new part's survival function, ``R(x)``."""
        x = np.asarray(x, dtype=float)
        return np.asarray(self.sf(x), dtype=float).reshape(x.shape)

    def S(self, u) -> np.ndarray:
        """The part's in now: ``S(u) = R(age + u) / R(age)``."""
        u = np.asarray(u, dtype=float)
        return self.R(self.age + u) / self.base


def _quantile(survival, level: float, start: float) -> Optional[float]:
    """The time at which ``survival`` falls to ``level``, by bisection
    (None if it never does)."""
    high = start
    for _ in range(400):
        if float(survival(high)) <= level:
            break
        high *= 2.0
    else:
        return None
    low = 0.0
    for _ in range(100):
        middle = 0.5 * (low + high)
        if float(survival(middle)) <= level:
            high = middle
        else:
            low = middle
    return high


def _spread(survival, start: float) -> Optional[float]:
    """The interquartile range of a life given by its survival function."""
    lower = _quantile(survival, 0.75, start)
    upper = _quantile(survival, 0.25, start)
    if lower is None or upper is None or not upper > lower:
        return None
    return upper - lower


def _step(life: _Life, part, span: float) -> float:
    """The grid step over ``span``: ``_STEPS_PER_SPREAD`` steps per
    interquartile range of the life new and of the part's life left, from
    ``_MIN_STEPS`` to ``_MAX_STEPS`` steps over the span."""
    from repyability.rbd._model_utils import failure_time_scale

    scale = failure_time_scale(part.reliability)
    if not (math.isfinite(scale) and scale > 0.0):
        scale = span
    spreads = [
        s
        for s in (
            _spread(lambda x: life.R(x), scale),
            _spread(lambda u: life.S(u), scale),
        )
        if s is not None
    ]
    step = min(spreads + [span]) / _STEPS_PER_SPREAD
    return min(max(step, span / _MAX_STEPS), span / _MIN_STEPS)


class _Grid:
    """A grid ``0, h, ..., n h`` covering a span, and the convolutions of
    distribution functions on it."""

    def __init__(self, h: float, span: float):
        self.h = h
        self.n = int(math.ceil(span / h - 1e-9))
        self.x = h * np.arange(self.n + 1)
        self.size = 1 << int(math.ceil(math.log2(2 * (self.n + 1))))

    def kernel(self, G: np.ndarray) -> np.ndarray:
        """The transform of the cell weights for ``K -> integral of
        K(x - u) dG(u)``: each cell's exact increment of ``G``, ``K``
        averaged over the cell (``c_k = (dG_k + dG_(k-1)) / 2``)."""
        dG = np.diff(G)
        c = np.zeros(self.n + 1)
        c[:-1] += 0.5 * dG
        c[1:] += 0.5 * dG
        return np.fft.rfft(c, self.size)

    def convolve(self, K: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """``integral of K(x - u) dG(u)`` at the grid points, for a ``K``
        that is 0 at 0."""
        out = np.fft.irfft(np.fft.rfft(K, self.size) * kernel, self.size)
        return np.clip(out[: self.n + 1], 0.0, None)

    def at(self, K: np.ndarray, y) -> np.ndarray:
        """``K`` at the times ``y``, linearly between grid points, 0 before
        0."""
        return np.interp(y, self.x, K, left=0.0, right=K[-1])


def _powers(grid: _Grid, first: np.ndarray, kernel: np.ndarray) -> list:
    """``first``, then each convolved once more, until the last is below
    ``_TAIL``; the list ends with one below it."""
    out = [first]
    while out[-1][-1] >= _TAIL:
        if len(out) > _MAX_COUNT:
            raise NotImplementedError(
                f"More than {_MAX_COUNT} failures in the window are "
                "likely: the numerical route stops there. Use "
                "method='simulate'."
            )
        out.append(grid.convolve(out[-1], kernel))
    return out


def _weight(m, k, p: float, extra: float = 1.0) -> np.ndarray:
    """``C(m, k) p^(m - k) extra``, for arrays ``m``, ``k`` (from logs)."""
    m = np.asarray(m, dtype=float)
    k = np.asarray(k, dtype=float)
    j = m - k
    if p == 0.0:
        return np.where(j == 0, extra, 0.0)
    log = gammaln(m + 1) - gammaln(k + 1) - gammaln(j + 1) + j * math.log(p)
    return extra * np.exp(log)


# -- run to failure and age replacement ---------------------------------------


class _Cycles:
    """The part's cycles under age replacement at ``T`` (``inf``: run to
    failure) from its age, and the distribution functions of their totals
    on a grid of step ``h`` over ``[0, horizon]``: ``A[i - 1]``, the first
    failure cycle and ``i - 1`` new ones; ``B[i - 1]``, ``i`` new ones."""

    def __init__(
        self, life: _Life, T: float, horizon: float, h: float, later: bool
    ):
        self.life, self.T, self.h = life, T, h
        self.tau = max(T - life.age, 0.0)
        if self.tau == 0.0:
            self.p1 = 1.0  # at or past the age: replaced at once
        elif math.isinf(self.tau):
            self.p1 = 0.0
        else:
            self.p1 = float(life.S(self.tau))
        self.p = 0.0 if math.isinf(T) else float(life.R(T))
        self.grid = _Grid(h, horizon)
        self.kernel = self.grid.kernel(self.G(self.grid.x))
        # The totals from the part in now: only for a window that starts
        # later (the renewals before it).
        first = self.G1(self.grid.x)
        self.A = _powers(self.grid, first, self.kernel) if later else []
        self.B = _powers(self.grid, self.G(self.grid.x), self.kernel)

    def G(self, x) -> np.ndarray:
        """A new part's failure cycle: ``F(min(x, T))``."""
        return 1.0 - self.life.R(np.minimum(x, self.T))

    def G1(self, x) -> np.ndarray:
        """The part in now's: ``1 - S(min(x, tau))``."""
        if self.tau == 0.0:
            return np.zeros(np.shape(x))
        return 1.0 - self.life.S(np.minimum(x, self.tau))

    def at(self, which: str, i: int, y) -> np.ndarray:
        """``A_i`` or ``B_i`` at the times ``y``: the first exactly, the
        others from the grid (0 past the sequence's end)."""
        y = np.asarray(y, dtype=float)
        if i == 1:
            exact = self.G1 if which == "A" else self.G
            return np.where(y > 0.0, exact(np.maximum(y, 0.0)), 0.0)
        sequence = self.A if which == "A" else self.B
        if i > len(sequence):
            return np.zeros(y.shape)
        return self.grid.at(sequence[i - 1], y)

    def renewals(self, s) -> np.ndarray:
        """``M_c(s)``: the expected number of renewals in ``[0, s]`` after
        at least one failure (the others are the planned replacements'
        chain from now, at ``tau + k T``)."""
        s = np.asarray(s, dtype=float)
        out = np.zeros(s.shape)
        T, p = self.T, self.p
        planned = 0 if math.isinf(T) else int(s.max() // T) + 2
        for i in range(1, max(len(self.A), len(self.B)) + 1):
            for j in range(planned + 1):
                shift = j * T if j else 0.0
                weight = float(_weight(i - 1 + j, i - 1, p))
                if weight > 0.0:
                    out += weight * self.at("A", i, s - shift)
                if j and self.p1 > 0.0:
                    weight = float(_weight(i + j - 1, i, p, self.p1))
                    start = self.tau + (j - 1) * T
                    out += weight * self.at("B", i, s - start)
        return out


class _First:
    """The first event at or after ``start``: its time from ``start`` as a
    failure (``failed(x)``, a sub-distribution function) or a planned
    replacement (``atoms``, each time and chance, and ``planned(x)``, the
    rest, continuous)."""

    def __init__(self, cycles: _Cycles, start: float):
        c, e = cycles, start
        life, T = c.life, c.T
        self.cycles, self.start = c, e
        self.pieces = []  # (chance, age at start, cap): exact failures
        self.atoms = []  # (time from start, chance)
        if e <= c.tau:  # the part in now, never renewed
            self.pieces.append(
                (1.0 / life.base, life.age + e, life.age + c.tau)
            )
            if not math.isinf(c.tau):
                self.atoms.append((c.tau - e, c.p1))
        elif not math.isinf(T):  # the last of the planned chain from now
            k = math.ceil((e - c.tau) / T) - 1
            theta = c.tau + k * T
            u = e - theta
            if 0.0 < u <= T:
                chance = c.p1 * c.p**k
                self.pieces.append((chance, u, T))
                self.atoms.append((T - u, chance * c.p))
        # Renewals after a failure, by the age ``u`` they leave at start:
        # cells of the grid, each at its middle.
        reach = min(T, e)
        self.cells = None
        if reach > 0.0:
            n = int(math.ceil(reach / c.h - 1e-9))
            u = np.minimum(c.h * np.arange(n + 1), reach)
            M = c.renewals(e - u)
            self.mass = np.maximum(M[:-1] - M[1:], 0.0)
            self.middle = c.h * (np.arange(n) + 0.5)
            self.cells = n
            self.low = float(c.renewals(np.array([max(e - T, 0.0)]))[0])
            self.high = float(M[0])

    def _exact_failed(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        R = self.cycles.life.R
        out = np.zeros(x.shape)
        for chance, u, cap in self.pieces:
            out += chance * (R(u) - R(np.minimum(u + x, cap)))
        return out

    def failed_on(self, grid: _Grid) -> np.ndarray:
        """``failed`` at the grid's points: the exact pieces, and the
        renewals' by FFT (each cell's mass at its middle age)."""
        out = self._exact_failed(grid.x)
        if self.cells:
            c = self.cycles
            R = c.life.R
            n = self.cells
            ages = c.h * (np.arange(n + grid.n + 1) + 0.5)
            curve = R(np.minimum(ages, c.T))
            size = 1 << int(math.ceil(math.log2(len(ages) + n)))
            spectrum = np.fft.rfft(self.mass[::-1], size) * np.fft.rfft(
                curve, size
            )
            later = np.fft.irfft(spectrum, size)[n - 1 : n - 1 + grid.n + 1]
            out += float(self.mass @ R(np.minimum(self.middle, c.T))) - later
        return np.maximum(out, 0.0)

    def failed(self, grid: _Grid, values: np.ndarray, x) -> np.ndarray:
        """``failed`` at the times ``x``: the exact pieces exactly, the
        renewals' between the grid's points."""
        x = np.asarray(x, dtype=float)
        rest = values - self._exact_failed(grid.x)
        return np.where(
            x > 0.0,
            self._exact_failed(np.maximum(x, 0.0)) + grid.at(rest, x),
            0.0,
        )

    def planned(self, x) -> np.ndarray:
        """The renewals' planned replacements by ``x``: those with an age
        at start of ``T - x`` or more, each surviving to ``T``."""
        c = self.cycles
        x = np.asarray(x, dtype=float)
        if self.cells is None or math.isinf(c.T):
            return np.zeros(x.shape)
        s = self.start - c.T + np.clip(x, 0.0, c.T)
        M = np.where(s > 0.0, c.renewals(np.maximum(s, 0.0)), 0.0)
        return np.where(x > 0.0, c.p * np.maximum(M - self.low, 0.0), 0.0)


def _window_counts(life: _Life, T: float, start: float, window: float, h):
    """``P[i, j]``: the chance of ``i`` failures and ``j`` planned
    replacements in ``[start, start + window)`` under age replacement at
    ``T`` (``inf``: run to failure) from the part's age now, on a grid of
    step ``h``."""
    cycles = _Cycles(life, T, start + window, h, start > 0.0)
    first = _First(cycles, start)
    w, p = window, cycles.p
    grid = _Grid(h, w)
    kernel = grid.kernel(cycles.G(grid.x))
    failed = first.failed_on(grid)
    A = _powers(grid, failed, kernel)
    D = _powers(grid, first.planned(grid.x), kernel)
    count = max(len(A), len(D), len(cycles.B))
    A += [grid.convolve(A[-1], kernel) for _ in range(count + 1 - len(A))]
    D += [grid.convolve(D[-1], kernel) for _ in range(count + 2 - len(D))]

    def A_at(i, y):
        if i == 1:
            return first.failed(grid, failed, y)
        return grid.at(A[i - 1], y)

    def D_at(i, y):
        return first.planned(y) if i == 0 else grid.at(D[i], y)

    def Q(at, i, y):
        # P(the cycles end before y and the next runs past it).
        y = np.asarray(y, dtype=float)
        out = at(i, y) - at(i + 1, y)
        if p > 0.0:
            out = out - p * at(i, y - T)
        return np.where(y > 0.0, np.maximum(out, 0.0), 0.0)

    planned = 0 if math.isinf(T) else int(math.ceil(w / T)) + 1
    j = np.arange(planned + 1)
    later = j[1:]
    P = np.zeros((count + 1, planned + 1))
    events = float(first.failed(grid, failed, np.array([w]))[0])
    events += float(first.planned(np.array([w]))[0])
    events += sum(chance for x0, chance in first.atoms if x0 < w)
    P[0, 0] = max(1.0 - events, 0.0)
    for x0, chance in first.atoms:
        if chance == 0.0:
            continue
        z = w - x0 - (later - 1) * T
        alive = np.where((z > 0.0) & (z <= T), life.R(np.clip(z, 0.0, T)), 0.0)
        P[0, 1:] += chance * np.power(p, later - 1) * alive
        for i in range(1, count + 1):
            P[i, 1:] += _weight(i + later - 1, i, p, chance) * Q(
                lambda k, y: cycles.at("B", k, y), i, z
            )
    if planned:
        y = w - (later - 1) * T
        for i in range(count + 1):
            P[i, 1:] += _weight(i + later - 1, i, p) * Q(D_at, i, y)
    for i in range(1, count + 1):
        y = w - j * T if planned else np.array([w])
        P[i] += _weight(i - 1 + j, i - 1, p) * Q(A_at, i, y)
    return P


# -- block replacement --------------------------------------------------------


def _block_window(life: _Life, plan: _Plan, start: float, window, h):
    """``P[i, j]`` in ``[start, start + window)`` under block replacement:
    the planned replacements fall at fixed times, so ``j`` is known, and
    the failures are the sum of each stretch's, run to failure (the first
    from the part in service at ``start``, the others from new)."""
    T = plan.every
    tau = T - float(plan.since or 0.0)
    end = start + window
    blocks = tau + T * np.arange(int(math.ceil(max(end - tau, 0.0) / T)) + 1)
    blocks = blocks[blocks < end]
    before = blocks[blocks < start]
    inside = blocks[blocks >= start]
    origin = float(before[-1]) if len(before) else 0.0
    stretch = (float(inside[0]) if len(inside) else end) - start
    new = life.new()
    pmf = _failures(life if origin == 0.0 else new, start - origin, stretch, h)
    lengths = np.minimum(T, end - inside)
    full = int(np.sum(lengths >= T))
    if full:
        pmf = np.convolve(pmf, _power(_failures(new, 0.0, T, h), full))
    for length in lengths[lengths < T]:
        pmf = np.convolve(pmf, _failures(new, 0.0, float(length), h))
    P = np.zeros((len(pmf), len(inside) + 1))
    P[:, len(inside)] = pmf
    return P


def _failures(life: _Life, start: float, length: float, h) -> np.ndarray:
    """The chance of each number of failures in ``[start, start +
    length)`` of a part run to failure from ``life``'s age."""
    if length <= 0.0:
        return np.ones(1)
    return _window_counts(life, math.inf, start, length, h)[:, 0]


def _power(pmf: np.ndarray, k: int) -> np.ndarray:
    """The distribution of the sum of ``k`` independent counts of
    distribution ``pmf``."""
    out = np.ones(1)
    while k:
        if k & 1:
            out = _trim(np.convolve(out, pmf))
        k >>= 1
        if k:
            pmf = _trim(np.convolve(pmf, pmf))
    return out


def _trim(pmf: np.ndarray) -> np.ndarray:
    keep = np.flatnonzero(pmf >= 1e-20)
    return pmf[: keep[-1] + 1] if len(keep) else pmf[:1]


# -- simulated ---------------------------------------------------------------


def _sampler(part):
    from repyability.rbd._sampling import stream_sampler

    quantile = stream_sampler(part.reliability)
    if quantile is None:
        raise NotImplementedError(
            "The part's life cannot be drawn by its quantile function: "
            "use the default, method='exact'."
        )
    return quantile


def _simulate(part, plan: _Plan, edges: np.ndarray, mc_samples: int, seed):
    """The failures and planned replacements of ``mc_samples`` seeded
    histories of the part in each window between ``edges`` (from 0):
    two arrays of shape ``(windows, mc_samples)``; for a calendar policy,
    the planned replacements by kind too, ``{kind: (price, array)}``."""
    rng = np.random.default_rng(seed)
    quantile = _sampler(part)
    life = _Life(part, plan.age)
    used = 1.0 - life.base  # F(age)
    horizon = float(edges[-1])
    windows = len(edges) - 1
    failures = np.zeros((windows, mc_samples), dtype=int)
    planned = np.zeros((windows, mc_samples), dtype=int)

    def draw(n: int, aged: bool) -> np.ndarray:
        u = rng.random(n)
        if aged:
            return np.asarray(quantile(used + u * life.base)) - plan.age
        return np.asarray(quantile(u), dtype=float)

    def tally(counts, who, when):
        np.add.at(counts, (np.searchsorted(edges, when, "right") - 1, who), 1)

    if plan.policy == CALENDAR:
        return _simulate_calendar(plan, edges, draw, tally, mc_samples)
    T = plan.every
    if plan.policy == BLOCK:
        tau = T - float(plan.since or 0.0)
        blocks = np.arange(tau, horizon, T)
        for b in blocks:
            planned[np.searchsorted(edges, b, "right") - 1] += 1
        starts = [0.0] + list(blocks)
        for k, start in enumerate(starts):
            end = starts[k + 1] if k + 1 < len(starts) else horizon
            t = start + draw(mc_samples, k == 0)
            active = np.flatnonzero(t < end)
            while len(active):
                tally(failures, active, t[active])
                t[active] += draw(len(active), False)
                active = active[t[active] < end]
        return failures, planned
    # Run to failure and age replacement: cycle by cycle.
    t = np.zeros(mc_samples)
    age = np.full(mc_samples, plan.age)
    life_left = draw(mc_samples, True)
    active = np.arange(mc_samples)
    while len(active):
        until = np.maximum(T - age[active], 0.0)
        failed = life_left[active] < until
        ends = t[active] + np.where(failed, life_left[active], until)
        inside = ends < horizon
        active, failed, ends = active[inside], failed[inside], ends[inside]
        tally(failures, active[failed], ends[failed])
        tally(planned, active[~failed], ends[~failed])
        t[active] = ends
        age[active] = 0.0
        life_left[active] = draw(len(active), False)
    return failures, planned


def _simulate_calendar(plan: _Plan, edges, draw, tally, mc_samples: int):
    """``_simulate`` under a calendar policy: each cycle ends in a failure
    or at the planned replacement the policy fixes for a unit put in when
    the cycle starts (the part in now: put in ``age`` before 0)."""
    policy = plan.calendar
    horizon = float(edges[-1])
    windows = len(edges) - 1
    failures = np.zeros((windows, mc_samples), dtype=int)
    by_kind: dict = {}
    t = np.zeros(mc_samples)
    put_in = np.full(mc_samples, -plan.age)
    life_left = draw(mc_samples, True)
    active = np.arange(mc_samples)
    while len(active):
        due = [
            policy._next_planned(put_in[r], max(put_in[r], 0.0))
            for r in active
        ]
        until = np.array([d[0] for d in due]) - t[active]
        failed = life_left[active] < until
        ends = t[active] + np.where(failed, life_left[active], until)
        inside = ends < horizon
        tally(failures, active[failed & inside], ends[failed & inside])
        for k in np.flatnonzero(~failed & inside):
            _, price, kind = due[k]
            if kind not in by_kind:
                by_kind[kind] = (
                    float(price),
                    np.zeros((windows, mc_samples), dtype=int),
                )
            tally(by_kind[kind][1], active[k : k + 1], ends[k : k + 1])
        active, ends = active[inside], ends[inside]
        t[active] = ends
        put_in[active] = ends
        life_left[active] = draw(len(active), False)
    planned = sum(
        (counts for _, counts in by_kind.values()),
        np.zeros((windows, mc_samples), dtype=int),
    )
    return failures, planned, by_kind


# -- the answer ---------------------------------------------------------------


def _percentile(costs, probabilities, q: float) -> float:
    cumulative = np.cumsum(probabilities)
    index = int(np.searchsorted(cumulative, q / 100.0 - 1e-12))
    return float(costs[min(index, len(costs) - 1)])


def _above(costs, probabilities, budget: float) -> float:
    return float(np.sum(probabilities[costs > budget]))


def _by_cost(P: np.ndarray, cu: float, cp: float):
    """The distinct costs of the outcomes, cheapest first, and the chance
    of each."""
    i, j = np.indices(P.shape)
    return _merged((cu * i + cp * j).ravel(), P.ravel())


def _merged(cost: np.ndarray, mass: np.ndarray):
    """The distinct values of ``cost``, cheapest first, and the total
    ``mass`` of each."""
    order = np.argsort(cost, kind="stable")
    cost, mass = cost[order], mass[order]
    scale = max(float(np.max(np.abs(cost))), 1.0)
    new = np.concatenate([[True], np.diff(cost) > 1e-12 * scale])
    groups = np.cumsum(new) - 1
    costs = cost[new]
    probabilities = np.bincount(groups, weights=mass)
    keep = probabilities > 0.0
    keep[0] = True
    return costs[keep], probabilities[keep]


class _ByKind(NamedTuple):
    """A calendar policy's counts by kind: the kinds and their prices, and
    for each vector of planned replacements by kind, the chance of each
    number of failures."""

    names: Tuple[str, ...]
    prices: Tuple[float, ...]
    by_vector: Dict[Tuple[int, ...], np.ndarray]

    def costs(self, cu: float):
        cost, mass = [], []
        for vector, column in self.by_vector.items():
            planned = float(np.dot(vector, self.prices))
            cost.append(cu * np.arange(len(column)) + planned)
            mass.append(column)
        return _merged(np.concatenate(cost), np.concatenate(mass))

    def outcomes(self, cu: float) -> list:
        out = []
        for vector, column in self.by_vector.items():
            planned = dict(zip(self.names, (int(v) for v in vector)))
            spent = float(np.dot(vector, self.prices))
            for i in np.flatnonzero(column >= 1e-12):
                out.append(
                    {
                        "failures": int(i),
                        "planned": planned,
                        "cost": cu * int(i) + spent,
                        "probability": float(column[i]),
                    }
                )
        return sorted(out, key=lambda o: -o["probability"])


def _counts(part, plan: _Plan, start: float, window: float, h: float):
    life = _Life(part, plan.age)
    if plan.policy == BLOCK:
        return _block_window(life, plan, start, window, h)
    return _window_counts(life, plan.every, start, window, h)


def _padded(P: np.ndarray, shape) -> np.ndarray:
    out = np.zeros(shape)
    out[: P.shape[0], : P.shape[1]] = P
    return out


def _change(fine: np.ndarray, coarse: np.ndarray) -> float:
    shape = np.maximum(fine.shape, coarse.shape)
    return float(np.abs(_padded(fine, shape) - _padded(coarse, shape)).max())


def _numerical(part, plan: _Plan, edges: np.ndarray, cache: dict) -> list:
    """Each window's counts' distribution, its precision (the largest
    change of a probability from a grid twice as coarse, plus the mass
    left out) and, for a calendar policy, its counts by kind (``cache``
    keeps what its windows share)."""
    if plan.policy == CALENDAR:
        return _numerical_calendar(part, plan, edges, cache)
    life = _Life(part, plan.age)
    span = float(edges[-1])
    if plan.policy == BLOCK:
        tau = plan.every - float(plan.since or 0.0)
        span = max(min(tau, span), min(plan.every, span))
    h = _step(life, part, span)
    out = []
    for start, end in zip(edges[:-1], edges[1:]):
        fine = _counts(part, plan, float(start), float(end - start), h)
        coarse = _counts(part, plan, float(start), float(end - start), 2 * h)
        missing = max(0.0, 1.0 - float(fine.sum()))
        out.append((fine, _change(fine, coarse) + missing, None))
    return out


def _numerical_calendar(part, plan: _Plan, edges, cache: dict) -> list:
    """``_numerical`` under a calendar policy: on grids of half the
    policy's step and of its step (``_window_calendar``)."""
    from repyability.maintenance._window_calendar import (
        grid_step,
        window_counts,
    )

    step = grid_step(plan.calendar)
    life = _Life(part, plan.age)
    out = []
    for start, end in zip(edges[:-1], edges[1:]):
        found = {}
        for h in (step / 2.0, step):
            found[h] = window_counts(
                life, plan.calendar, float(start), float(end), h, cache
            )
        fine, by_vector, dropped = found[step / 2.0]
        coarse = found[step][0]
        kinds = cache["kinds"]
        missing = max(0.0, 1.0 - float(fine.sum())) + dropped
        by_kind = _ByKind(
            tuple(kinds.names),
            tuple(kinds.prices),
            {kinds.full(v): c for v, c in by_vector.items()},
        )
        out.append((fine, _change(fine, coarse) + missing, by_kind))
    return out


def _simulated(part, plan: _Plan, edges, n: int, seed, whole: bool) -> list:
    """Each window's share of the simulations with each count, each
    simulation's cost in it and, for a calendar policy, the shares by
    kind; with ``whole``, the whole span's last, from the same
    histories."""
    found = _simulate(part, plan, edges, n, seed)
    failures, planned = found[0], found[1]
    by_kind = found[2] if plan.policy == CALENDAR else None
    if whole:
        failures = np.vstack([failures, failures.sum(axis=0)])
        planned = np.vstack([planned, planned.sum(axis=0)])
        if by_kind is not None:
            by_kind = {
                kind: (price, np.vstack([c, c.sum(axis=0)]))
                for kind, (price, c) in by_kind.items()
            }
    cu = float(part.cu)
    out: List[Tuple[np.ndarray, np.ndarray, Optional[_ByKind]]] = []
    for w, (f, j) in enumerate(zip(failures, planned)):
        P = np.zeros((f.max() + 1, j.max() + 1))
        np.add.at(P, (f, j), 1.0 / n)
        if by_kind is None:
            cp = 0.0 if plan.policy == RUN_TO_FAILURE else float(part.cp)
            out.append((P, cu * f + cp * j, None))
            continue
        names = tuple(by_kind)
        prices = tuple(by_kind[k][0] for k in names)
        counts = np.array([by_kind[k][1][w] for k in names]).reshape(
            len(names), n
        )
        spent = np.asarray(prices) @ counts if names else np.zeros(n)
        vectors: Dict[Tuple[int, ...], np.ndarray] = {}
        for r in range(n):
            vector = tuple(int(c) for c in counts[:, r])
            if vector not in vectors:
                vectors[vector] = np.zeros(f.max() + 1)
            vectors[vector][f[r]] += 1.0 / n
        out.append((P, cu * f + spent, _ByKind(names, prices, vectors)))
    return out


def _route(method, mc_samples, seed):
    """The method, and the number of simulations (None for the numerical
    route), checked."""
    checks.one_of("method", method, ("exact", "simulate"))
    seed = checks.seed(seed)
    if method == "exact":
        if mc_samples is not None or seed is not None:
            raise ValueError(
                "mc_samples and seed are for method='simulate'; the "
                "default, method='exact', draws nothing."
            )
        return None, None
    n = _SIMULATIONS if mc_samples is None else mc_samples
    if not (
        isinstance(n, (int, np.integer)) and not isinstance(n, bool) and n > 1
    ):
        raise ValueError(
            f"mc_samples must be a whole number, more than 1; got {n!r}."
        )
    return int(n), seed


def _distribution(
    part, plan: _Plan, start, end, found, levels, budget, simulated
) -> CostDistribution:
    """A window's ``CostDistribution`` from its counts' distribution:
    ``found`` is ``(P, precision or each simulation's cost, by kind)``."""
    P, extra, by_kind = found
    precision, costs_each = (None, extra) if simulated else (extra, None)
    cu = float(part.cu)
    priced = plan.policy in (AGE, BLOCK)
    cp = float(part.cp) if priced else 0.0
    if by_kind is None:
        costs, cost_probabilities = _by_cost(P, cu, cp)
    else:
        costs, cost_probabilities = by_kind.costs(cu)
    i = np.arange(P.shape[0])
    j = np.arange(P.shape[1])
    by_failures = P.sum(axis=1)
    by_planned = P.sum(axis=0)
    total = float(P.sum())
    mean = float(costs @ cost_probabilities) / total
    variance = float((costs - mean) ** 2 @ cost_probabilities) / total
    standard_error, n = None, None
    if costs_each is not None:
        n = len(costs_each)
        standard_error = float(np.std(costs_each, ddof=1) / math.sqrt(n))
    return CostDistribution(
        float(start),
        float(end - start),
        plan.policy,
        plan.every if priced else None,
        plan.age,
        plan.since,
        cu,
        cp if priced else None,
        "simulated" if simulated else "numerical",
        P,
        by_failures,
        by_planned,
        costs,
        cost_probabilities,
        mean,
        math.sqrt(max(variance, 0.0)),
        float(i @ by_failures),
        float(j @ by_planned),
        levels,
        np.array([_percentile(costs, cost_probabilities, q) for q in levels]),
        budget,
        None if budget is None else _above(costs, cost_probabilities, budget),
        precision,
        standard_error,
        n,
        None if by_kind is None else by_kind.names,
        None if by_kind is None else by_kind.prices,
        None if by_kind is None else by_kind.outcomes(cu),
    )


def distributions(
    part,
    edges,
    every,
    resets,
    age,
    since_renewal,
    budget,
    levels,
    method,
    mc_samples,
    seed,
    whole: bool = False,
    calendar=None,
) -> list:
    """A ``CostDistribution`` for each window between ``edges`` (from 0),
    the inputs checked (the work of ``cost_distribution`` and
    ``cost_by_window``); with ``whole``, the whole span's last."""
    _check_part(part)
    plan = _plan(part, every, resets, age, since_renewal, calendar)
    levels = tuple(_level(q) for q in np.atleast_1d(levels))
    if budget is not None:
        budget = _budget(budget)
    n, seed = _route(method, mc_samples, seed)
    edges = np.asarray(edges, dtype=float)
    pairs = list(zip(edges[:-1], edges[1:]))
    if whole:
        pairs.append((0.0, edges[-1]))
    if n is None:
        cache: dict = {}
        found = _numerical(part, plan, edges, cache)
        if whole:
            found += _numerical(part, plan, edges[[0, -1]], cache)
    else:
        found = _simulated(part, plan, edges, n, seed, whole)
    return [
        _distribution(part, plan, a, b, f, levels, budget, n is not None)
        for (a, b), f in zip(pairs, found)
    ]


def cost_distribution(
    part,
    window,
    every=None,
    *,
    resets=None,
    age=0.0,
    since_renewal=None,
    calendar=None,
    budget=None,
    levels=LEVELS,
    method: str = "exact",
    mc_samples: Optional[int] = None,
    seed=None,
) -> CostDistribution:
    """The work of ``NonRepairable.cost_distribution`` (see there)."""
    window = _number(window, "window", strict=True)
    return distributions(
        part,
        [0.0, window],
        every,
        resets,
        age,
        since_renewal,
        budget,
        levels,
        method,
        mc_samples,
        seed,
        calendar=calendar,
    )[0]
