"""The value of running one unit on an operating schedule (UC-20): is it
worth running it harder?

A unit earns while it works, and its life depends on how it is run (a
fitted surpyval regression model of its life against its covariates:
temperature, load, speed). Running it harder may earn more but wears it
out faster. The plant's operating schedule says, in model time from 0,
how it is run in each stretch of time and what it earns per unit of time
up in that stretch. The unit is replaced by a new one whenever it fails:
the replacement takes a down time, during which nothing is earned and the
unit does not age, and costs ``failure_cost``.

A unit put into service at time ``v`` runs along the rest of the plant's
schedule: at age ``x`` it is at the covariates in force at time ``v + x``.
So its life depends on when it was put in. Its survival is the model's
``sf_tvc`` along that shifted path (surpyval's ``StepSchedule``), and a
unit put in after the last change has the last stretch's fixed-covariate
life.

**How it is worked out (numerical, no simulation).** The expected number
of failures and the probability that the unit is up at each time solve a
renewal equation in the plant's time whose life depends on when each unit
was put in (a non-homogeneous renewal process). It is solved on a grid
whose cells are sized from the lives (never in absolute time) and whose
edges include every change of the schedule, so that nothing bends inside
a cell. Each cell's units are put in through it; within the stretch they
were put in, they fail at that stretch's fixed covariates, and the chance
of failing by each later grid time is that life's distribution averaged
exactly over the cell (from its integral, so a density that is infinite
at 0 is handled); past the stretch's end, it is the life along the path
of a unit put in at the cell's middle. A failure in a cell is taken at a
constant rate through it, and the replacement's end follows from the down
time's distribution integrated exactly over the cell (atoms, such as a
fixed down time, included). The value and the costs are the integrals of
the discounted probability of being up (linear between grid times) and
of the discounted failures. The grid is halved until every total (the
failures and the time up, plain and discounted) agrees with the halved
grid's to the tolerance, and the halved grid's answer is returned.
"""

import math
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from repyability.rbd._model_utils import MODEL_ERRORS
from repyability.rbd.results import _ResultMapping
from repyability.utils import checks
from repyability.utils.wrappers import outside_level

#: Grid cells per interquartile range of the shortest life (the first
#: grid; each refinement halves the cells).
_STEPS_PER_SPREAD = 8
#: The fewest cells in a stretch of the schedule, and over the horizon.
_MIN_CELLS, _MIN_TOTAL = 4, 32
#: The most cells a grid may have: past it the method stops refining and
#: warns.
_MAX_CELLS = 2**15
#: The relative agreement of a grid with its halving that is asked for by
#: default.
_TOLERANCE = 1e-6
#: The relative agreement asked by default of each NPV in a search over
#: switch times, which values the schedule some 50 times.
_SWITCH_TOLERANCE = 1e-4
#: Switch times scanned by ``best_switch_time`` across the horizon, ends
#: included, before the best is polished.
_SCAN = 41

_FAMILIES = (
    "accelerated-failure-time, proportional- and additive-hazards, and "
    "proportional-odds models"
)


def _number(value, name: str, strict: bool = False) -> float:
    number = checks.number_or_nan(value)
    low = number > 0.0 if strict else number >= 0.0
    if not (math.isfinite(number) and low):
        what = "more than 0" if strict else "at least 0"
        raise ValueError(
            f"{name} must be a finite number, {what}; got {value!r}."
        )
    return number


# -- inputs ------------------------------------------------------------------


def _schedule(times, covariates, value_rates, model):
    """The schedule's stretch starts, covariate rows ``(K, p)`` and value
    rates, checked."""
    starts = np.atleast_1d(checks.real_array(times, "times")).astype(float)
    ok = (
        starts.ndim == 1
        and starts.size >= 1
        and bool(np.all(np.isfinite(starts)))
        and starts[0] == 0.0
        and bool(np.all(np.diff(starts) > 0.0))
    )
    if not ok:
        raise ValueError(
            "times must be the times each stretch of the schedule starts: "
            f"increasing, finite, the first 0; got {times!r}."
        )
    count = starts.size
    Z = checks.real_array(covariates, "covariates")
    if Z.ndim == 1 and count == 1:
        Z = Z.reshape(1, -1)
    elif Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    if Z.ndim != 2 or Z.shape[0] != count or not np.all(np.isfinite(Z)):
        raise ValueError(
            f"covariates must give one row of finite covariates for each "
            f"of the {count} stretch(es) of times; got {covariates!r}."
        )
    from repyability.rbd.regression_node import _fitted_width

    fitted = _fitted_width(model)
    if fitted is not None and Z.shape[1] != fitted:
        raise ValueError(
            f"The model was fitted with {fitted} covariate(s); each row of "
            f"covariates has {Z.shape[1]}."
        )
    rates = np.atleast_1d(checks.real_array(value_rates, "value_rates"))
    if (
        rates.ndim != 1
        or rates.size != count
        or not np.all(np.isfinite(rates))
        or np.any(rates < 0.0)
    ):
        raise ValueError(
            f"value_rates must give a finite value per unit of time up, at "
            f"least 0, for each of the {count} stretch(es) of times; got "
            f"{value_rates!r}."
        )
    return starts, Z, rates.astype(float)


class _Life:
    """A unit's life along the plant's schedule (``starts``, ``Z``): the
    cumulative hazard of a unit put into service at any time, through the
    model's ``Hf_tvc``."""

    def __init__(self, model, starts: np.ndarray, Z: np.ndarray):
        from repyability.rbd.regression_node import _is_semiparametric

        checks.no_distribution(model, "model")
        if _is_semiparametric(model):
            raise ValueError(
                "The model is semiparametric (a Cox model): its baseline is "
                "an estimate on the observed range only, with no tail "
                "beyond it, so the life of a unit along the schedule has no "
                "proper curve to renew. Fit a parametric regression model "
                "in surpyval (e.g. WeibullPH) instead."
            )
        self.model = model
        self.starts = starts
        self.Z = Z
        for j in range(len(starts)):
            try:
                # Just after 0, in any unit of time.
                first = self.fixed(j, np.array([np.finfo(float).tiny]))[0]
            except MODEL_ERRORS as error:
                raise ValueError(
                    "The model must be a fitted surpyval regression model "
                    f"whose sf_tvc works along a schedule ({_FAMILIES}); "
                    f"evaluating it failed: {type(error).__name__}: "
                    f"{error}."
                ) from error
            if not first < -math.log(0.99):
                raise ValueError(
                    "The life must be a proper curve, starting at a "
                    "survival of 1 (sf(0+) ~ 1); at the covariates of "
                    f"stretch {j} it starts at {math.exp(-first):.4g}."
                )

    def _hazard(self, ages: np.ndarray, times, rows) -> np.ndarray:
        from surpyval import StepSchedule

        path = StepSchedule.from_changepoints(times, rows)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            H = np.asarray(self.model.Hf_tvc(ages, path), dtype=float)
        if not np.all(np.isfinite(H)) or np.any(H < 0.0):
            raise ValueError(
                "The life along the schedule is not defined at some ages "
                "(the model's cumulative hazard is not a finite, "
                "non-negative number there: e.g. an additive hazard that "
                "falls below 0)."
            )
        return H

    def fixed(self, j: int, ages: np.ndarray) -> np.ndarray:
        """The cumulative hazard at the covariates of stretch ``j``."""
        return self._hazard(ages, [0.0], self.Z[j : j + 1])

    def put_in(self, v: float, j: int, ages, aged: float = 0.0):
        """The cumulative hazard at ``ages`` of a unit put into service at
        time ``v`` in stretch ``j``, having spent ``aged`` before it at
        that stretch's covariates, counted from then."""
        later = self.starts[j + 1 :] - v + aged
        times = np.concatenate([[0.0], later])
        ages = np.asarray(ages, dtype=float)
        H = self._hazard(ages + aged, times, self.Z[j:])
        if aged > 0.0:
            H = H - self._hazard(np.array([aged]), times, self.Z[j:])[0]
        return np.maximum(H, 0.0)

    def spread(self, H) -> float:
        """The interquartile range of a life of cumulative hazard ``H``
        (or a typical time when it has none)."""
        from repyability.rbd.regression_node import _inverse_hazard

        targets = -np.log(np.array([0.75, 0.25]))
        q = _inverse_hazard(H, targets)
        if np.all(np.isfinite(q)) and q[1] > q[0]:
            return float(q[1] - q[0])
        finite = q[np.isfinite(q) & (q > 0.0)]
        return float(finite[0]) if finite.size else math.inf


class _DownTime:
    """How long a replacement takes: instant, a fixed time, or a surpyval
    model. ``done(x)`` is the integral of its CDF from 0 to ``x``: of a
    failure spread evenly over a cell of width ``h`` ending ``x`` ago, the
    share of whose replacements are over is ``(done(x + h) - done(x)) /
    h``."""

    def __init__(self, down_time):
        from repyability.rbd._model_utils import (
            distribution_name,
            never_fails,
        )

        self.fixed: Optional[float] = None
        if down_time is None:
            self.fixed = 0.0
            return
        if checks.is_number(down_time):
            self.fixed = _number(down_time, "down_time")
            return
        checks.no_distribution(down_time, "down_time")
        name = distribution_name(down_time)
        if name == "ExactEventTime":
            self.fixed = float(np.ravel(down_time.params)[0])
            return
        if name is None or not hasattr(down_time, "qf"):
            raise ValueError(
                "down_time must be a number (0 for instant) or a fitted "
                "surpyval distribution of the time a replacement takes; "
                f"got {down_time!r}."
            )
        if never_fails(down_time) > 0.0:
            raise ValueError(
                "Some of the down times never end (the down_time model has "
                "p < 1), so the unit would end up down for good."
            )
        self._table(down_time)

    def _table(self, model) -> None:
        """``done`` at the down time's quantiles (dense through its body,
        and into its tail to 1e-12), for cubic Hermite interpolation with
        its CDF as the slope: past the last, the down time is over."""
        from repyability.rbd._block_replacement import _cumulative

        body = np.linspace(0.0, 0.999, 1000)[1:]
        tail = 1.0 - 10.0 ** -np.linspace(3.0, 12.0, 73)
        x = np.ravel(model.qf(np.concatenate([body, tail])))
        x = np.unique(np.concatenate([[0.0], x[np.isfinite(x) & (x > 0)]]))
        cdf = np.clip(np.ravel(model.ff(x)), 0.0, 1.0)
        cdf[0] = 0.0

        def ff(u):
            return np.clip(np.ravel(model.ff(u)), 0.0, 1.0)

        self.x, self.G = x, cdf
        self.D = _cumulative(ff, x)

    @property
    def reach(self) -> float:
        """The time past which every down time is over (``done`` rises
        with slope 1 from there)."""
        return float(self.fixed if self.fixed is not None else self.x[-1])

    def done(self, x: np.ndarray) -> np.ndarray:
        x = np.maximum(np.asarray(x, dtype=float), 0.0)
        if self.fixed is not None:
            return np.maximum(x - self.fixed, 0.0)
        xs, D, G = self.x, self.D, self.G
        i = np.clip(np.searchsorted(xs, x, side="right") - 1, 0, len(xs) - 2)
        a, b = xs[i], xs[i + 1]
        width = b - a
        s = (np.minimum(x, b) - a) / width
        h00 = (1 + 2 * s) * (1 - s) ** 2
        h10 = s * (1 - s) ** 2
        h01 = s * s * (3 - 2 * s)
        h11 = s * s * (s - 1)
        inside = h00 * D[i] + h10 * width * G[i] + h01 * D[i + 1]
        inside = inside + h11 * width * G[i + 1]
        return np.where(x > xs[-1], D[-1] + (x - xs[-1]), inside)


# -- the grid and the renewal equation --------------------------------------


def _phi(x: np.ndarray):
    """``(1 - exp(-x)) / x`` and ``(1 - exp(-x) (1 + x)) / x**2``, the
    weights of the discounted integral over a cell of ``r h = x`` of a
    constant, and of a line rising from 0 to 1 across it (1 and 1/2 at
    ``x = 0``)."""
    x = np.asarray(x, dtype=float)
    small = x < 1e-4
    safe = np.where(small, 1.0, x)
    one = np.where(small, 1.0 - x / 2 + x * x / 6, -np.expm1(-safe) / safe)
    two = np.where(
        small,
        0.5 - x / 3 + x * x / 8,
        (-np.expm1(-safe) - safe * np.exp(-safe)) / (safe * safe),
    )
    return one, two


class _Grid:
    """The grid over ``[0, horizon]``: each stretch of the schedule that
    starts before the horizon, cut into ``counts`` equal cells."""

    def __init__(self, starts, horizon: float, counts: np.ndarray):
        bounds = np.concatenate([starts[starts < horizon], [horizon]])
        edges = [np.zeros(1)]
        piece = []
        for j, n in enumerate(counts):
            inner = np.linspace(bounds[j], bounds[j + 1], int(n) + 1)[1:]
            inner[-1] = bounds[j + 1]
            edges.append(inner)
            piece.append(np.full(int(n), j))
        self.t = np.concatenate(edges)
        self.piece = np.concatenate(piece)
        self.counts = np.asarray(counts, dtype=int)
        self.ends = np.cumsum(self.counts)  # the last edge of each piece
        self.h = np.diff(self.t)
        self.bounds = bounds


class _Totals:
    """A solved grid's values: each cell's expected failures and time up,
    plain and discounted."""

    def __init__(self, failures, up, failures_pv, up_pv, grid):
        self.failures = failures
        self.up = up
        self.failures_pv = failures_pv
        self.up_pv = up_pv
        self.grid = grid

    def sums(self) -> np.ndarray:
        return np.array(
            [
                self.failures.sum(),
                self.up.sum(),
                self.failures_pv.sum(),
                self.up_pv.sum(),
            ]
        )

    def by_piece(self, values: np.ndarray) -> np.ndarray:
        return np.bincount(
            self.grid.piece, weights=values, minlength=len(self.grid.counts)
        )


def _solve(life: _Life, down: _DownTime, grid: _Grid, rate: float, age):
    """The expected failures and time up in each cell of ``grid``."""
    from repyability.rbd._block_replacement import _cumulative

    t, h, N = grid.t, grid.h, len(grid.h)
    # The unit in service at 0, ``age`` old at the first covariates.
    H0 = life.put_in(0.0, 0, t, aged=age)
    acc_M = -np.expm1(-H0)  # failures by each grid time
    acc_S = np.exp(-H0)  # the chance a unit is up at each grid time
    acc_I = np.zeros(N + 1)  # replacements put in by each grid time

    # Within its stretch, a unit put in through a cell fails by a later
    # grid time with its fixed-covariate life averaged over the cell:
    # ``(C((m + 1) h) - C(m h)) / h``, ``m`` cells on, ``C`` the integral
    # of its CDF (from geometric pieces near 0, where the density may be
    # infinite).
    kernels = []
    for j, n in enumerate(grid.counts):
        step = h[grid.ends[j] - 1]
        near = step * 2.0 ** -np.arange(40, 0, -1)
        edges = np.concatenate([[0.0], near, step * np.arange(1, n + 1)])
        C = _cumulative(lambda u, j=j: -np.expm1(-life.fixed(j, u)), edges)
        lattice = np.concatenate([[0.0], C[len(near) + 1 :]])
        kernels.append(np.clip(np.diff(lattice) / step, 0.0, 1.0))

    # A failure in a cell at a constant rate through it has had its
    # replacement put in by a time ``x`` after the cell's end with
    # probability ``(done(x + h) - done(x)) / h``: 1 once ``x`` passes the
    # down time's reach, which ``over`` counts from (by edge) and
    # ``acc_I`` before it.
    over = np.zeros(N + 2)
    finished = 0.0  # the failures whose replacements are all in by now
    selves = {
        float(step): float(down.done(np.array([step]))[0]) / step
        for step in np.unique(h)
    }
    failures = np.zeros(N)
    put_in = np.zeros(N)
    M = installed = 0.0
    for c in range(N):
        j = grid.piece[c]
        end = grid.ends[j]  # the last edge of this stretch
        K = kernels[j]
        q = K[0]
        w = selves[float(h[c])]
        finished += over[c + 1]
        alpha = acc_M[c + 1] - M
        beta = acc_I[c + 1] + finished - installed
        dM = (alpha + q * beta) / (1.0 - q * w)
        dI = beta + w * dM
        failures[c], put_in[c] = dM, dI
        M += dM
        installed += dI
        # The units put in through this cell: failing within its stretch,
        # and past it along their path.
        within = K[: end - c]
        acc_M[c + 1 : end + 1] += dI * within
        acc_S[c + 1 : end + 1] += dI * (1.0 - within)
        if end < N and dI != 0.0:
            mid = 0.5 * (t[c] + t[c + 1])
            later = life.put_in(mid, j, t[end + 1 :] - mid)
            acc_M[end + 1 :] += dI * -np.expm1(-later)
            acc_S[end + 1 :] += dI * np.exp(-later)
        # The failures in this cell: their replacements put in by each
        # later grid time.
        if dM != 0.0:
            ago = t[c + 1 :] - t[c + 1]
            near = int(np.searchsorted(ago, down.reach, side="right"))
            share = (down.done(ago[:near] + h[c]) - down.done(ago[:near])) / h[
                c
            ]
            acc_I[c + 1 : c + 1 + near] += dM * share
            if near == 0:
                finished += dM
            else:
                over[c + 1 + near] += dM

    A = np.clip(acc_S, 0.0, 1.0)
    one, two = _phi(rate * h)
    start = np.exp(-rate * t[:-1])
    up = 0.5 * h * (A[:-1] + A[1:])
    up_pv = start * h * (A[:-1] * (one - two) + A[1:] * two)
    return _Totals(failures, up, failures * start * one, up_pv, grid)


def _first_counts(life: _Life, down: _DownTime, starts, horizon, age):
    """The cells of each stretch on the first grid: sized from the
    shortest interquartile range of the lives (each stretch's, and the
    unit's in service at 0), never in absolute time."""
    inside = int(np.sum(starts < horizon))
    spreads = [
        life.spread(lambda u, j=j: life.fixed(j, u)) for j in range(inside)
    ]
    if age > 0.0:
        spreads.append(life.spread(lambda u: life.put_in(0.0, 0, u, aged=age)))
    step = min(min(spreads), horizon) / _STEPS_PER_SPREAD
    step = min(step, horizon / _MIN_TOTAL)
    bounds = np.concatenate([starts[:inside], [horizon]])
    counts = np.ceil(np.diff(bounds) / step).astype(int)
    return np.maximum(counts, _MIN_CELLS)


def _refined(life, down, starts, horizon, rate, age, tolerance):
    """Solve on grids halved until each total agrees with the halved
    grid's to ``tolerance`` (relative): the finer solution and the
    largest relative difference."""
    counts = _first_counts(life, down, starts, horizon, age)
    coarse = _solve(life, down, _Grid(starts, horizon, counts), rate, age)
    difference = math.inf
    while True:
        counts = 2 * counts
        if counts.sum() > _MAX_CELLS and math.isfinite(difference):
            warnings.warn(
                "The value of the operating schedule did not reach the "
                f"tolerance {tolerance:g}: on the finest grid allowed "
                f"({int(counts.sum() // 2)} cells) it changed by "
                f"{difference:.2g} (relative) from the grid of half as "
                "many. The lives may be far shorter than the horizon, or "
                "the down time far longer than the lives.",
                RuntimeWarning,
                stacklevel=outside_level(),
            )
            return coarse, difference
        fine = _solve(life, down, _Grid(starts, horizon, counts), rate, age)
        a, b = coarse.sums(), fine.sums()
        scale = np.abs(b)
        with np.errstate(invalid="ignore", divide="ignore"):
            change = np.where(scale > 0.0, np.abs(a - b) / scale, 0.0)
        difference = float(np.max(change))
        if difference <= tolerance:
            return fine, difference
        coarse = fine


# -- results -----------------------------------------------------------------


@dataclass
class ScheduleValue(_ResultMapping):
    """What running one unit on an operating schedule is worth: returned
    by [`operating_schedule_value`][repyability.operating_schedule_value].

    In plain terms: the unit earns ``value_rates`` per unit of time while
    it works, and each failure costs ``failure_cost`` and stops it for
    the down time while a new one is put in. The net present value (NPV)
    is what it earns less what its failures cost, each amount counted at
    what it is worth now (an amount ``t`` from now is worth ``exp(-r t)``
    of itself). It is the average over every way the failures could fall,
    worked out numerically (no simulation); it says nothing of the spread.

    Attributes
    ----------
    horizon : float
        The length of time valued, in the model's unit.
    discount_rate : float
        The continuous rate per unit time the amounts were discounted at
        (0: not discounted).
    npv : float
        The expected net present value: ``value - cost``.
    value : float
        The present value of what the unit earns while it works.
    cost : float
        The present value of the failures' costs.
    eac : float
        The equivalent annual cost: ``cost`` spread as a steady cost per
        unit of the model's time over the horizon, ``cost`` over ``(1 -
        exp(-r H)) / r`` (``H`` with no discounting).
    failures : float
        The expected number of failures over the horizon.
    up_time : float
        The expected time the unit works over the horizon.
    undiscounted_npv : float
        ``npv`` with no discounting.
    undiscounted_value : float
        ``value`` with no discounting.
    undiscounted_cost : float
        ``cost`` with no discounting.
    stretches : list of dict
        One row per stretch of the schedule: ``start``, ``end`` (cut at
        the horizon), ``covariates``, ``value_rate``, and the ``value``,
        ``cost`` (present values), ``failures`` and ``up_time`` within it.
        A stretch that starts at or after the horizon has zeros.
    method : str
        ``"numerical"``: a renewal equation solved on a grid (see
        ``tolerance``).
    tolerance : float
        The relative agreement asked of the grid with its halving.
    error : float
        The largest relative difference, between the grid of the answer
        and the grid of half as many cells, of the expected failures and
        time up (plain and discounted): within ``tolerance`` unless a
        warning said otherwise.
    cells : int
        The grid cells of the answer.
    """

    horizon: float
    discount_rate: float
    npv: float
    value: float
    cost: float
    eac: float
    failures: float
    up_time: float
    undiscounted_npv: float
    undiscounted_value: float
    undiscounted_cost: float
    stretches: List[Dict[str, Any]]
    method: str
    tolerance: float
    error: float
    cells: int


@dataclass
class SwitchTime(_ResultMapping):
    """The best time to switch a unit from one way of running it to
    another: returned by
    [`best_switch_time`][repyability.best_switch_time].

    In plain terms: the unit runs at the first covariates until
    ``switch_time``, then at the second. Switching at 0 means running the
    second way from the start; switching at the horizon means never
    switching. ``gain`` is what the best switch is worth over the better
    of those two.

    Attributes
    ----------
    switch_time : float
        The switch time that maximises the NPV, in ``[0, horizon]``.
    choice : str
        ``"switch at once"`` (the best is 0: run the second way from the
        start), ``"never switch"`` (the best is the horizon), or
        ``"switch"`` (in between).
    npv : float
        The NPV when switching at ``switch_time``.
    never_switch_npv : float
        The NPV of running the first way throughout.
    switch_at_once_npv : float
        The NPV of running the second way throughout.
    gain : float
        ``npv`` less the better of the two above (0 when an end is best).
    best : ScheduleValue
        The full valuation at ``switch_time``.
    times : list of float
        The switch times scanned, ``0`` to the horizon.
    npvs : list of float
        The NPV at each scanned time, to audit the answer (the NPV need
        not have one peak).
    """

    switch_time: float
    choice: str
    npv: float
    never_switch_npv: float
    switch_at_once_npv: float
    gain: float
    best: ScheduleValue
    times: List[float]
    npvs: List[float]


# -- the calculators ---------------------------------------------------------


class _Valuer:
    """The checked inputs that do not change with the schedule, and the
    valuation of a schedule."""

    def __init__(
        self,
        model,
        horizon,
        failure_cost,
        down_time,
        discount_rate,
        age,
        tolerance,
    ):
        from repyability.rbd._common import _discount_rate

        self.model = model
        self.horizon = _number(horizon, "horizon", strict=True)
        self.cost = _number(failure_cost, "failure_cost")
        self.down = _DownTime(down_time)
        self.rate = _discount_rate(discount_rate)
        self.age = _number(age, "age")
        tol = checks.number_or_nan(tolerance)
        if not (0.0 < tol <= 0.01):
            raise ValueError(
                "tolerance must be a number more than 0 and at most 0.01; "
                f"got {tolerance!r}."
            )
        self.tolerance = tol

    def value(self, starts, Z, rates) -> ScheduleValue:
        life = _Life(self.model, starts, Z)
        H, r = self.horizon, self.rate
        totals, error = _refined(
            life, self.down, starts, H, r, self.age, self.tolerance
        )
        failures = totals.by_piece(totals.failures)
        up = totals.by_piece(totals.up)
        failures_pv = totals.by_piece(totals.failures_pv)
        up_pv = totals.by_piece(totals.up_pv)
        inside = len(failures)
        rows = []
        for j in range(len(starts)):
            there = j < inside
            rows.append(
                {
                    "start": float(starts[j]),
                    "end": float(
                        min(starts[j + 1] if j + 1 < len(starts) else H, H)
                    ),
                    "covariates": [float(z) for z in Z[j]],
                    "value_rate": float(rates[j]),
                    "value": float(rates[j] * up_pv[j]) if there else 0.0,
                    "cost": (
                        float(self.cost * failures_pv[j]) if there else 0.0
                    ),
                    "failures": float(failures[j]) if there else 0.0,
                    "up_time": float(up[j]) if there else 0.0,
                }
            )
        value = float(np.dot(rates[:inside], up_pv))
        cost = float(self.cost * failures_pv.sum())
        plain_value = float(np.dot(rates[:inside], up))
        plain_cost = float(self.cost * failures.sum())
        annuity = -math.expm1(-r * H) / r if r else H
        return ScheduleValue(
            horizon=H,
            discount_rate=r,
            npv=value - cost,
            value=value,
            cost=cost,
            eac=cost / annuity,
            failures=float(failures.sum()),
            up_time=float(up.sum()),
            undiscounted_npv=plain_value - plain_cost,
            undiscounted_value=plain_value,
            undiscounted_cost=plain_cost,
            stretches=rows,
            method="numerical",
            tolerance=self.tolerance,
            error=error,
            cells=int(totals.grid.counts.sum()),
        )


def operating_schedule_value(
    model,
    times: Sequence[float],
    covariates,
    value_rates,
    horizon: float,
    *,
    failure_cost: float = 0.0,
    down_time=0.0,
    discount_rate: float = 0.0,
    age: float = 0.0,
    tolerance: float = _TOLERANCE,
) -> ScheduleValue:
    """What one unit run on an operating schedule is worth over a horizon:
    the value it earns while it works, the cost of its failures, and the
    net present value. Numerical, with no simulation.

    In plain terms: the schedule says, in the model's time from 0, how the
    unit is run in each stretch (its covariates: a temperature, a load)
    and what it earns per unit of time up there. Each failure costs
    ``failure_cost`` and stops it for the ``down_time`` while a new unit
    is put in; nothing is earned while it is down. A unit put in at time
    ``v`` runs along the rest of the schedule, so its life is the model's
    survival along the schedule from ``v`` (``model.sf_tvc``). Running
    harder can earn more and cost more in failures; this says which way
    the balance falls. The guide's [load
    schedule](../guide/condition-based.md#is-it-worth-running-hot) section
    tells the story; [`best_switch_time`][repyability.best_switch_time]
    finds the best time to change from one way of running to another.

    The expected failures and the chance of being up at each time solve a
    renewal equation whose life depends on when each unit was put in. It
    is solved on a grid sized from the lives, with the schedule's changes
    at grid edges, halved until every total agrees with the halved grid's
    to ``tolerance`` (``error`` in the result says how closely it did).

    Parameters
    ----------
    model : surpyval regression model
        The unit's life against its covariates, fitted in surpyval
        (accelerated failure time, proportional hazards or odds, additive
        hazards: what ``sf_tvc`` evaluates along a schedule), e.g.
        ``surpyval.WeibullAFT.fit(x, Z=Z)``. A semiparametric (Cox) model
        has no proper curve and is refused.
    times : array_like
        When each stretch of the schedule starts, in the model's time:
        increasing, the first 0. The last stretch runs on past the
        horizon.
    covariates : array_like
        One row of covariates per stretch (a number per stretch for a
        model with one covariate).
    value_rates : array_like
        What the unit earns per unit of time up in each stretch, at least
        0.
    horizon : float
        The length of time to value, more than 0.
    failure_cost : float, optional
        The cost of each failure and its replacement, charged when it
        fails; by default 0.
    down_time : float or surpyval model, optional
        How long a replacement takes: a fixed time, or a fitted surpyval
        distribution of it; by default 0 (instant). The unit neither earns
        nor ages while down.
    discount_rate : float, optional
        The continuous discount rate per unit of the model's time, by
        default 0: for an annual rate ``i`` with the model in hours,
        ``math.log(1 + i) / 8760``.
    age : float, optional
        How long the unit in service at 0 has already run, at the first
        stretch's covariates; by default 0 (new).
    tolerance : float, optional
        The relative agreement asked of the grid with its halving, by
        default 1e-6.

    Returns
    -------
    ScheduleValue

    Raises
    ------
    ValueError
        If an input is not as described, or the model cannot be evaluated
        along the schedule.

    Warns
    -----
    RuntimeWarning
        If the grid reached its largest size before the tolerance.

    Examples
    --------
    A bearing whose life is exponential with a mean of 100 hours at load
    1 and 25 at load 2, earning 1 an hour at load 1 and 2 at load 2, each
    failure costing 20:

    >>> import numpy as np
    >>> import surpyval as surv
    >>> from repyability import operating_schedule_value
    >>> x = np.array([50.0, 100.0, 150.0, 12.5, 25.0, 37.5])
    >>> load = np.array([[1.0], [1.0], [1.0], [2.0], [2.0], [2.0]])
    >>> model = surv.ExponentialAFT.fit(x, Z=load)
    >>> cool = operating_schedule_value(
    ...     model, [0], [1.0], [1.0], 1000.0, failure_cost=20.0
    ... )
    >>> round(cool.value, 3), round(cool.failures, 3)  # 1000 h, 1000 / 100
    (1000.0, 10.0)
    >>> hot = operating_schedule_value(
    ...     model, [0], [2.0], [2.0], 1000.0, failure_cost=20.0
    ... )
    >>> round(hot.npv, 1), round(cool.npv, 1)  # 2000 - 40 x 20, 1000 - 200
    (1200.0, 800.0)
    """
    valuer = _Valuer(
        model, horizon, failure_cost, down_time, discount_rate, age, tolerance
    )
    starts, Z, rates = _schedule(times, covariates, value_rates, model)
    return valuer.value(starts, Z, rates)


def best_switch_time(
    model,
    covariates,
    value_rates,
    horizon: float,
    *,
    failure_cost: float = 0.0,
    down_time=0.0,
    discount_rate: float = 0.0,
    age: float = 0.0,
    tolerance: float = _SWITCH_TOLERANCE,
) -> SwitchTime:
    """The best time to switch a unit from one way of running it to
    another, over a horizon: the switch time that maximises the net
    present value of [`operating_schedule_value`]
    [repyability.operating_schedule_value].

    In plain terms: the unit runs at the first row of covariates (earning
    the first value rate) until the switch, and at the second after it.
    Switching at 0 runs it the second way from the start, and at the
    horizon never switches; the answer says which is best, and what the
    best switch is worth over both. The NPV as a function of the switch
    time need not have one peak, so it is scanned at 41 times across the
    horizon, the best polished between its neighbours (Brent's method),
    and the scan returned to audit the answer. Each NPV is numerical, to
    ``tolerance``, so the switch time is found to about the flatness of
    the NPV near its best. Each switch time is a valuation, so a search
    takes some 50 of them.

    Parameters
    ----------
    model : surpyval regression model
        As for ``operating_schedule_value``.
    covariates : array_like
        Two rows of covariates (two numbers for a model with one): before
        the switch, and after.
    value_rates : array_like
        What the unit earns per unit of time up before the switch, and
        after.
    horizon : float
        The length of time to value, more than 0.
    failure_cost : float, optional
        As for ``operating_schedule_value``.
    down_time : float or surpyval model, optional
        As for ``operating_schedule_value``.
    discount_rate : float, optional
        As for ``operating_schedule_value``.
    age : float, optional
        As for ``operating_schedule_value``.
    tolerance : float, optional
        The relative agreement asked of each NPV's grid with its halving,
        by default 1e-4 (looser than ``operating_schedule_value``'s, as
        the search values the schedule some 50 times; the switch time
        moves little within it). Give 1e-6 for the full precision.

    Returns
    -------
    SwitchTime

    Examples
    --------
    The bearing of ``operating_schedule_value``'s example, run cool (load
    1, earning 1 an hour) or hot (load 2, earning 2 an hour), over 200
    hours, each NPV to 1e-3. At 20 a failure, running hot pays from the
    start; at 40 it never does:

    >>> import numpy as np
    >>> import surpyval as surv
    >>> from repyability import best_switch_time
    >>> x = np.array([50.0, 100.0, 150.0, 12.5, 25.0, 37.5])
    >>> load = np.array([[1.0], [1.0], [1.0], [2.0], [2.0], [2.0]])
    >>> model = surv.ExponentialAFT.fit(x, Z=load)
    >>> def best(cost):
    ...     return best_switch_time(
    ...         model, [1.0, 2.0], [1.0, 2.0], 200.0, failure_cost=cost,
    ...         tolerance=1e-3,
    ...     ).choice
    >>> best(20.0), best(40.0)
    ('switch at once', 'never switch')
    """
    from scipy.optimize import minimize_scalar

    valuer = _Valuer(
        model, horizon, failure_cost, down_time, discount_rate, age, tolerance
    )
    _, Z, rates = _schedule([0.0, 1.0], covariates, value_rates, model)
    H = valuer.horizon

    def at(s: float) -> ScheduleValue:
        if s <= 0.0:
            return valuer.value(np.zeros(1), Z[1:], rates[1:])
        if s >= H:
            return valuer.value(np.zeros(1), Z[:1], rates[:1])
        return valuer.value(np.array([0.0, s]), Z, rates)

    times = np.linspace(0.0, H, _SCAN)
    values = [at(float(s)) for s in times]
    npvs = np.array([v.npv for v in values])
    i = int(np.argmax(npvs))
    best_s, best = float(times[i]), values[i]
    lo, hi = times[max(i - 1, 0)], times[min(i + 1, _SCAN - 1)]
    polished = minimize_scalar(
        lambda s: -at(float(s)).npv,
        bounds=(float(lo), float(hi)),
        method="bounded",
        options={"xatol": 1e-4 * H},
    )
    if -polished.fun > best.npv:
        best_s = float(polished.x)
        best = at(best_s)
    if best_s <= 0.0:
        choice = "switch at once"
    elif best_s >= H:
        choice = "never switch"
    else:
        choice = "switch"
    never, at_once = float(npvs[-1]), float(npvs[0])
    return SwitchTime(
        switch_time=best_s,
        choice=choice,
        npv=best.npv,
        never_switch_npv=never,
        switch_at_once_npv=at_once,
        gain=max(best.npv - max(never, at_once), 0.0),
        best=best,
        times=[float(s) for s in times],
        npvs=[float(v) for v in npvs],
    )
