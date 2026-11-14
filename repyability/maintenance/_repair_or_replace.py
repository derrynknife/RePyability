"""Repair again or replace? From a joint frailty model (UC-27).

Units that fail often also tend to be written off sooner. surpyval's
joint frailty model (``surpyval.JointFrailty``) fits a fleet's repeat
failures and its write-offs together, and gives each unit a frailty: how
much more often than a typical unit it fails (and, through ``alpha``, how
much sooner it is written off). Given a unit's frailty ``u``, its failures
are minimal repairs at the intensity ``u r(t)`` and its write-off has the
hazard ``u**alpha h(t)``, independent of each other; a new unit's frailty
is drawn from the frailty distribution.

Everything here is a ratio of expectations over one cycle of a position in
the fleet (a unit from new until it is replaced or written off), by the
renewal-reward theorem, with a continuous discount rate ``r`` (0 for
none) as in ``NonRepairable.age_replacement_value``. A cycle that ends at
``L`` with discounted cost ``C`` gives the equivalent cost rate
``E[C] / E[integral_0^L exp(-r t) dt]``, the long-run cost rate when
``r = 0``. Every integral is one over age (Gauss-Legendre on pieces
spaced by the logarithm of age) and over the frailty (Gauss-Hermite on
the normal score of its distribution), of the model's own ``iif``,
``cif``, ``hf`` and ``Hf`` at each frailty: numerical, with no
simulation. The bands over parameter draws are the only random part.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, Hashable, Mapping, Optional, Tuple

import numpy as np
from scipy import stats
from scipy.optimize import brentq, minimize_scalar
from scipy.special import roots_hermitenorm, roots_legendre

from repyability.rbd.results import _ResultMapping
from repyability.utils import checks
from repyability.utils.deprecation import refuse_removed_names

#: Gauss-Hermite nodes over the frailty's normal score.
_FRAILTY_NODES = 40
#: Gauss-Legendre nodes in each piece of age, and pieces per decade.
_ORDER = 8
_PIECES_PER_DECADE = 8
#: The cumulative hazard (and expected failures) at which a cycle is taken
#: to have ended: exp(-40) is 4e-18.
_END_HAZARD = 40.0
#: The youngest age on the grids, as a share of the shortest time scale.
_HEAD = 1e-6
#: The most decades of age a grid spans (a frailty near 0 can make a
#: unit's life astronomically long).
_MAX_DECADES = 40.0
#: Parameter draws for the bands, by default.
_DEFAULT_DRAWS = 500
#: The confidence of the bands: 90% two-sided.
_BAND = 0.90
#: The relative difference within which a simpler rule is kept.
_TIE = 1e-9

_BOUNDS = (
    "90% two-sided confidence bands, from draws of the fitted parameters "
    "from their covariance (a Wald approximation; surpyval 0.27 has no "
    "bootstrap for this model)"
)

_TAKES = (
    "RepairOrReplace takes a surpyval JointFrailtyModel (from "
    "surpyval.JointFrailty.fit, or JointFrailty.from_params)"
)


def _check_model(model) -> None:
    """Refuse anything but a joint frailty model on the calendar time
    scale."""
    from surpyval.recurrent import JointFrailtyModel

    checks.no_distribution(model, "model")
    if not isinstance(model, JointFrailtyModel):
        raise TypeError(
            f"{_TAKES}, got {type(model).__name__}. For a unit repaired "
            "without being written off, use Repairable."
        )
    if model.time_scale != "calendar":
        raise NotImplementedError(
            "the model is on the gap time scale (each failure renews the "
            "rate of failures): surpyval 0.27 gives such a unit's expected "
            "failures by an age no closed form, and cannot simulate a unit "
            "of a given frailty, so its economics cannot be worked out "
            "here. Fit the model with time_scale='calendar'."
        )


def _cost(value, name: str, positive: bool = False) -> float:
    number = checks.number_or_nan(value)
    if not math.isfinite(number) or number < 0 or (positive and number == 0):
        least = "more than 0" if positive else "at least 0"
        raise ValueError(
            f"{name} must be a finite number, {least}; got {value!r}."
        )
    return number


def _frailty_nodes(theta: float, family: str) -> Tuple[np.ndarray, np.ndarray]:
    """Frailties and weights whose sums are expectations over the frailty
    distribution: Gauss-Hermite on the normal score ``z``, the frailty
    being its quantile at ``Phi(z)`` (exact in ``z`` for the log-normal,
    smooth for the gamma). No variance, one unit of frailty 1."""
    if theta <= 0:
        return np.ones(1), np.ones(1)
    z, w = roots_hermitenorm(_FRAILTY_NODES)
    w = w / math.sqrt(2.0 * math.pi)
    if family == "lognormal":
        u = np.exp(math.sqrt(theta) * z)
    else:
        shape = 1.0 / theta
        u = np.where(
            z < 0,
            stats.gamma.ppf(stats.norm.cdf(z), shape, scale=theta),
            stats.gamma.isf(stats.norm.sf(z), shape, scale=theta),
        )
    u = np.maximum(u, np.finfo(float).tiny)
    return u, w / w.sum()


def _log_pieces(lo: float, hi: float):
    """Gauss-Legendre nodes and weights for integrals from ``lo`` to
    ``hi`` on pieces equal in the logarithm of age: the nodes, the weights
    (the age's Jacobian in), each node's piece, and the pieces' edges."""
    decades = math.log10(hi / lo)
    n = max(1, int(math.ceil(_PIECES_PER_DECADE * decades)))
    edges = np.geomspace(lo, hi, n + 1)
    y, w = roots_legendre(_ORDER)
    log_lo, log_hi = np.log(edges[:-1]), np.log(edges[1:])
    half = 0.5 * (log_hi - log_lo)
    s = (0.5 * (log_hi + log_lo))[:, None] + half[:, None] * y[None, :]
    t = np.exp(s)
    weights = half[:, None] * w[None, :] * t
    piece = np.repeat(np.arange(n), _ORDER)
    return t.ravel(), weights.ravel(), piece, edges


class _Cycle:
    """The integrals of what is left of a unit's cycle, from age ``a``:
    each node's age ``s`` from ``a``, the cost and the discounted time of
    keep repairing summed by piece, and the arrays a count of further
    failures needs."""

    def __init__(self, engine, u, wu, a: float):
        self.engine, self.u, self.wu, self.a = engine, u, wu, a
        e = engine
        start = e._at(np.full(u.shape, a), u) if a > 0 else None
        self.H0 = np.zeros(u.shape) if start is None else start["H"]
        self.L0 = np.zeros(u.shape) if start is None else start["Lam"]
        span = np.maximum(e._ends(u, self.H0) - a, 0.0)
        lo = _HEAD * float(min(np.min(e._starts(u)), np.max(span)))
        hi = max(float(np.max(span)), 2.0 * lo)
        hi = min(hi, lo * 10.0**_MAX_DECADES)
        s, ws, self.piece, self.edges = _log_pieces(lo, hi)
        self.head = lo
        S, U = np.meshgrid(s, u)
        self.q = e._at(a + S, U)
        self.alive = (
            np.exp(-(self.q["H"] - self.H0[:, None]) - e.r * s[None, :])
            * ws[None, :]
        )
        # The head, from a to a + lo: failures and write-offs to their
        # cumulative counts there, in service and undiscounted throughout.
        first = e._at(np.full(u.shape, a + lo), u)
        self.head_failures = wu @ (first["Lam"] - self.L0)
        self.head_write_offs = wu @ (first["H"] - self.H0)
        n = len(self.edges) - 1
        cost, time = e._keep_parts(self.q, self.alive, wu)
        head = (e.cr + e.cf) * self.head_failures + e.cw * self.head_write_offs
        self.cum_cost = (
            np.r_[0.0, np.bincount(self.piece, cost, minlength=n).cumsum()]
            + head
        )
        self.cum_time = (
            np.r_[0.0, np.bincount(self.piece, time, minlength=n).cumsum()]
            + lo
        )

    def surviving(self, s) -> np.ndarray:
        """The chance (weighted over the nodes) of not being written off
        by ``s`` from ``a``, for each of ``s``."""
        s = np.atleast_1d(np.asarray(s, dtype=float))
        U, Sm = np.meshgrid(self.u, s, indexing="ij")
        H = self.engine._at(self.a + Sm, U)["H"]
        return self.wu @ np.exp(-(H - self.H0[:, None]))

    def keep(self) -> Tuple[float, float]:
        """The cost and discounted time of keep repairing to write-off."""
        return float(self.cum_cost[-1]), float(self.cum_time[-1])

    def ages(self) -> Tuple[np.ndarray, np.ndarray]:
        """The cost and discounted time of replacing at each edge ``s``."""
        planned = (
            self.engine.cp
            * np.exp(-self.engine.r * self.edges)
            * self.surviving(self.edges)
        )
        return self.cum_cost + planned, self.cum_time

    def age(self, s: float, i: int) -> Tuple[float, float]:
        """The cost and discounted time of replacing at ``s`` from ``a``,
        from the ``i``-th edge (at or below ``s``) on."""
        e = self.engine
        cost0, time0, begin = self.cum_cost[i], self.cum_time[i], self.edges[i]
        if s > begin:
            y, w = roots_legendre(4 * _ORDER)
            lb, ub = math.log(begin), math.log(s)
            nodes = np.exp(0.5 * (lb + ub) + 0.5 * (ub - lb) * y)
            wt = 0.5 * (ub - lb) * w * nodes
            Sm, U = np.meshgrid(nodes, self.u)
            q = e._at(self.a + Sm, U)
            alive = (
                np.exp(-(q["H"] - self.H0[:, None]) - e.r * nodes[None, :])
                * wt[None, :]
            )
            cost, time = e._keep_parts(q, alive, self.wu)
            cost0, time0 = cost0 + cost.sum(), time0 + time.sum()
        planned = e.cp * math.exp(-e.r * s) * float(self.surviving(s)[0])
        return float(cost0 + planned), float(time0)

    def counts(self, kmax: int) -> Tuple[np.ndarray, np.ndarray]:
        """The cost and discounted time of replacing at the ``k``-th
        failure from ``a``, for ``k = 1 .. kmax``."""
        e, q, alive = self.engine, self.q, self.alive
        more = q["Lam"] - self.L0[:, None]
        cost = np.empty(kmax)
        time = np.empty(kmax)
        # The head: a failure in it is the first, a write-off ends it.
        cost[:] = (e.cr + e.cf) * self.head_failures
        cost[0] = (e.cf + e.cp) * self.head_failures
        cost += e.cw * self.head_write_offs
        pmf = np.exp(-more)
        below = np.zeros_like(pmf)
        for k in range(1, kmax + 1):
            if k > 1:
                pmf = pmf * more / (k - 1)
            cdf = below + pmf
            spend = (
                q["lam"] * ((e.cr + e.cf) * below + (e.cf + e.cp) * pmf)
                + e.cw * q["h"] * cdf
            )
            cost[k - 1] += self.wu @ (alive * spend).sum(axis=1)
            time[k - 1] = self.wu @ (alive * cdf).sum(axis=1) + self.head
            below = cdf
        return cost, time


class _Engine:
    """The numbers of one joint frailty model at given costs."""

    def __init__(self, model, Z, costs: Dict[str, float]):
        self.model = model
        self.Z = Z
        self.cr = costs["repair"]
        self.cp = costs["replacement"]
        self.cf = costs["failure"]
        self.cw = costs["write_off"]
        self.r = costs["discount"]
        p = np.nan_to_num(np.asarray(model.params, dtype=float), nan=0.0)
        self.rs, self.rk, self.ts, self.tk, self.theta, self.alpha = (
            float(v) for v in p[:6]
        )
        if model.n_covariates:
            row = np.asarray(Z, dtype=float).reshape(-1)
            beta = np.nan_to_num(np.asarray(model.beta, dtype=float))
            gamma = np.nan_to_num(np.asarray(model.gamma, dtype=float))
            self.er = float(np.exp(row @ beta))
            self.et = float(np.exp(row @ gamma))
        else:
            self.er = self.et = 1.0

    # -- the model at frailties -------------------------------------------
    def _at(self, t: np.ndarray, u: np.ndarray) -> Dict[str, np.ndarray]:
        """The failure intensity, expected failures, write-off hazard and
        cumulative hazard of units of frailty ``u`` at ages ``t`` (arrays
        of one shape), from the model's own functions."""
        shape = t.shape
        tt, uu = t.ravel(), u.ravel()
        out = {
            "lam": self.model.iif(tt, self.Z, frailty=uu),
            "Lam": self.model.cif(tt, self.Z, frailty=uu),
            "h": self.model.hf(tt, self.Z, frailty=uu),
            "H": self.model.Hf(tt, self.Z, frailty=uu),
        }
        return {
            k: np.asarray(v, dtype=float).reshape(shape)
            for k, v in out.items()
        }

    def _ends(self, u: np.ndarray, H0=0.0) -> np.ndarray:
        """For each frailty, the age by which a unit (with cumulative
        hazard ``H0`` already) has all but surely been written off."""
        with np.errstate(divide="ignore", over="ignore"):
            return self.ts * (
                (H0 + _END_HAZARD) / (u**self.alpha * self.et)
            ) ** (1.0 / self.tk)

    def _starts(self, u: np.ndarray) -> np.ndarray:
        """For each frailty, the age of its first expected failure or
        unit cumulative hazard of write-off, whichever is sooner."""
        with np.errstate(divide="ignore", over="ignore"):
            written_off = self.ts * (1.0 / (u**self.alpha * self.et)) ** (
                1.0 / self.tk
            )
            failed = self.rs * (1.0 / (u * self.er)) ** (1.0 / self.rk)
        return np.minimum(written_off, failed)

    def _keep_parts(self, q, alive, wu) -> Tuple[np.ndarray, np.ndarray]:
        """The cost and the discounted time per node of age, keep
        repairing: repairs and their downtime, and the write-off."""
        rate = (self.cr + self.cf) * q["lam"] + self.cw * q["h"]
        return wu @ (alive * rate), wu @ alive

    # -- when undiscounted expectations are infinite ----------------------
    def _infinite(self, power: float) -> bool:
        """Whether ``E[u ** -power]`` is infinite for the frailty."""
        return (
            self.r == 0
            and self.theta > 0
            and self.model.family == "gamma"
            and power >= 1.0 / self.theta
        )

    def keep_repairing_infinite(self) -> bool:
        """Whether, undiscounted, a new unit's expected life (or its
        expected repairs) before write-off is infinite: a gamma frailty
        with ``1 / theta`` at most ``alpha / k_t``."""
        if self.alpha <= 0:
            return False
        life = self.alpha / self.tk
        repairs = self.alpha * self.rk / self.tk - 1.0
        return self._infinite(life) or self._infinite(repairs)

    def count_infinite(self) -> bool:
        """Whether a failure-count cycle's expected length is infinite."""
        power = 1.0 / self.rk
        if self.alpha > 0:
            power = min(power, self.alpha / self.tk)
        return self._infinite(power)

    # -- a new unit, its frailty drawn ------------------------------------
    def fleet(self) -> _Cycle:
        u, wu = _frailty_nodes(self.theta, self.model.family)
        return _Cycle(self, u, wu, 0.0)

    def keep_rate(self, cycle=None) -> float:
        if self.keep_repairing_infinite():
            return math.nan
        cost, time = (cycle or self.fleet()).keep()
        return cost / time

    def age_rate(self, T: float, cycle=None) -> float:
        cycle = cycle or self.fleet()
        i = max(int(np.searchsorted(cycle.edges, T, side="right")) - 1, 0)
        cost, time = cycle.age(T, i)
        return cost / time

    def count_rates(self, kmax: int, cycle=None) -> np.ndarray:
        if self.count_infinite():
            return np.full(kmax, math.nan)
        cost, time = (cycle or self.fleet()).counts(kmax)
        return cost / time

    # -- one unit, its frailty known --------------------------------------
    def unit(self, a: float, u: float) -> _Cycle:
        return _Cycle(self, np.array([float(u)]), np.ones(1), a)

    def marginal(self, t: float, u: float) -> float:
        """The cost per unit time of keeping a unit of frailty ``u`` at
        age ``t`` rather than replacing it: its repairs and their
        downtime, the write-off's premium over a planned replacement, less
        the interest on the replacement put off."""
        q = self._at(np.array([float(t)]), np.array([float(u)]))
        return float(
            (self.cr + self.cf) * q["lam"][0]
            + (self.cw - self.cp) * q["h"][0]
            - self.r * self.cp
        )


def _refine(objective, edges: np.ndarray, i: int) -> Tuple[float, float]:
    """The minimum of ``objective(s, j)`` (``j`` the edge below ``s``)
    between the edges either side of the ``i``-th."""
    a = edges[max(i - 1, 0)]
    b = edges[min(i + 1, len(edges) - 1)]

    def f(s):
        j = max(int(np.searchsorted(edges, s, side="right")) - 1, 0)
        return objective(s, j)

    res = minimize_scalar(
        f, bounds=(a, b), method="bounded", options={"xatol": 1e-10 * b}
    )
    return float(res.x), float(res.fun)


@dataclass(frozen=True)
class RepairOrReplacePolicy(_ResultMapping):
    """The best rule for a new unit in the fleet: keep repairing it, or
    replace it at an age or at a count of failures.

    Returned by
    [`RepairOrReplace.fleet_policy`][repyability.RepairOrReplace.fleet_policy].
    Each cost rate is per unit of the model's time, for a position in the
    fleet: a new unit of unknown frailty, repaired at each failure until
    the rule replaces it or it is written off, then another. With a
    discount rate it is the equivalent cost rate (the discount rate times
    the present value of all future costs), which tends to the long-run
    cost rate as the rate goes to 0. It is an immutable dataclass, read as
    a mapping too, with ``to_dict()`` for JSON.

    Attributes
    ----------
    recommended : str
        The cheapest rule: ``"keep repairing"`` (until written off),
        ``"replace at age"`` (``best_age``) or ``"replace at failure"``
        (the ``best_failure_count``-th).
    cost_rate : float
        The cost rate of the recommended rule.
    keep_repairing_cost_rate : float
        The cost rate of repairing every failure until the unit is written
        off. ``nan`` where, undiscounted, a new unit's expected life or
        repairs before write-off are infinite (a gamma frailty whose
        variance is large against the write-off's shape): give a discount
        rate.
    best_age : float
        The age at which replacing a unit is cheapest; ``inf`` when
        replacing at an age never pays.
    age_cost_rate : float
        The cost rate of replacing at ``best_age``.
    best_failure_count : int
        The failure at which replacing (instead of repairing) is cheapest,
        from 1 (replace at every failure) to the ``max_failures``
        searched.
    failure_count_cost_rate : float
        The cost rate of replacing at the ``best_failure_count``-th
        failure.
    discount_rate : float
        The continuous discount rate per unit of time (0 for none).
    bands : dict or None
        For each rule, ``"keep repairing"``, ``"replace at age"`` and
        ``"replace at failure"``, the ``(lower, upper)`` band on its cost
        rate at its ``best_age`` or ``best_failure_count``; None where the
        model carries no covariance (see ``bounds``).
    bounds : str
        What the bands are, or why there are none.
    notes : tuple of str
        Anything to read the numbers with.

    Examples
    --------
    >>> from surpyval import JointFrailty
    >>> from repyability import RepairOrReplace
    >>> model = JointFrailty.from_params((100.0, 2.0), (400.0, 2.0), 0.5, 1.0)
    >>> fleet = RepairOrReplace(
    ...     model, repair_cost=1.0, replacement_cost=10.0, failure_cost=2.0
    ... )
    >>> policy = fleet.fleet_policy()
    >>> policy.recommended, policy.best_failure_count
    ('replace at failure', 4)
    """

    recommended: str
    cost_rate: float
    keep_repairing_cost_rate: float
    best_age: float
    age_cost_rate: float
    best_failure_count: int
    failure_count_cost_rate: float
    discount_rate: float
    bands: Optional[Dict[str, Tuple[float, float]]]
    bounds: str
    notes: Tuple[str, ...]


@dataclass(frozen=True)
class UnitDecision(_ResultMapping):
    """Repair again or replace, for one unit at its age, from its
    frailty.

    Returned by
    [`RepairOrReplace.unit_decision`][repyability.RepairOrReplace.unit_decision].
    Each rule for the unit (keep repairing it until it is written off,
    replace it at an age, replace it at a count of further failures) is
    priced against replacing it now, with what follows its replacement
    valued at the fleet's cost rate. It is an immutable dataclass, read as
    a mapping too, with ``to_dict()`` for JSON.

    Attributes
    ----------
    unit : object
        The unit's name in the fitted model, or None for a frailty given.
    age : float
        The unit's age now, in the model's time.
    frailty : float
        The unit's frailty, taken as known: the model's estimate for it
        (its posterior mean given its history) or the one given.
    just_failed : bool
        Whether the unit has just failed, so that keeping it means paying
        for a repair now.
    recommended : str
        ``"replace now"``, ``"keep repairing"`` (until written off),
        ``"replace at age"`` (``best_age``) or ``"replace at failure"``
        (the ``best_failure_count``-th from now).
    saving : float
        What the recommended rule is expected to save against replacing
        now, in money discounted to now: 0 for replacing now.
    keep_saving : float
        What repairing the unit until it is written off is expected to
        save against replacing it now; negative where it costs more.
    age_saving : float
        The same, replacing it at ``best_age`` (0 where that is now).
    failure_count_saving : float
        The same, replacing it at its ``best_failure_count``-th further
        failure.
    best_age : float
        The age at which replacing the unit is best; ``inf`` when
        replacing at an age never pays.
    best_failure_count : int
        The further failure at which replacing it is best (1 for the
        next).
    keep_cost_rate : float
        The unit's own cost per unit time if repaired until written off,
        from now: its expected costs over its expected time in service
        (each discounted).
    age_cost_rate : float
        The same, replaced at ``best_age``; ``nan`` where that is now.
    failure_count_cost_rate : float
        The same, replaced at its ``best_failure_count``-th further
        failure.
    cost_rate_now : float
        What keeping the unit costs per unit time now: its failure
        intensity times the repair and failure costs, plus its write-off
        hazard times the write-off's premium over a planned replacement,
        less the interest on the replacement put off. Above
        ``fleet_cost_rate``, it costs more to keep than a new unit would.
    fleet_cost_rate : float
        The cost rate of a new unit under the fleet's best rule
        (``RepairOrReplacePolicy.cost_rate``).
    break_even_frailty : float
        The frailty at which, at this age, replacing now and the best
        rule for keeping it are worth the same: a unit at least that frail
        is replaced now (where the answer turns once). ``inf`` where no
        frailty makes replacing now pay, 0 where every one does.
    bands : dict or None
        The ``(lower, upper)`` bands on ``keep_cost_rate``,
        ``age_cost_rate``, ``failure_count_cost_rate`` and
        ``fleet_cost_rate``, the unit's frailty held at its estimate; None
        where the model carries no covariance.
    bounds : str
        What the bands are, or why there are none.
    notes : tuple of str
        Anything to read the numbers with.

    Examples
    --------
    >>> from surpyval import JointFrailty
    >>> from repyability import RepairOrReplace
    >>> model = JointFrailty.from_params((100.0, 2.0), (400.0, 2.0), 0.5, 1.0)
    >>> fleet = RepairOrReplace(
    ...     model, repair_cost=1.0, replacement_cost=10.0, failure_cost=2.0
    ... )
    >>> fleet.unit_decision(150.0, frailty=3.0).recommended
    'replace now'
    """

    unit: Any
    age: float
    frailty: float
    just_failed: bool
    recommended: str
    saving: float
    keep_saving: float
    age_saving: float
    failure_count_saving: float
    best_age: float
    best_failure_count: int
    keep_cost_rate: float
    age_cost_rate: float
    failure_count_cost_rate: float
    cost_rate_now: float
    fleet_cost_rate: float
    break_even_frailty: float
    bands: Optional[Dict[str, Tuple[float, float]]]
    bounds: str
    notes: Tuple[str, ...]


_NOW = "replace now"
_KEEP = "keep repairing"
_AGE = "replace at age"
_COUNT = "replace at failure"


@refuse_removed_names
class RepairOrReplace:
    """Repair again or replace? A fleet's repeat failures and write-offs
    from a joint frailty model, priced.

    In plain words: units that fail often also tend to be written off
    sooner. surpyval's joint frailty model fits a fleet's repairs and its
    write-offs together and gives each unit a frailty, a score of how much
    of a lemon it is (1 is a typical unit; 2 fails twice as often). This
    prices the choices at each failure: repair it again, or replace it.

    For a **new unit**, whose frailty is not yet known,
    [`fleet_policy`][repyability.RepairOrReplace.fleet_policy] compares
    keeping on repairing it until it is written off, replacing it at a set
    age, and replacing it at its ``k``-th failure (a count of failures is
    the fleet's way of finding its lemons). For **one unit**, its frailty
    estimated from its history,
    [`unit_decision`][repyability.RepairOrReplace.unit_decision] says
    whether to replace it now or keep it (until written off, to an age,
    or to a further failure), what that saves, and the frailty at which
    the answer turns
    ([`break_even_frailty`][repyability.RepairOrReplace.break_even_frailty]).

    Given its frailty ``u`` a unit fails at the intensity ``u r(t)``
    (minimal repairs, the model's power law) and is written off at the
    hazard ``u**alpha h(t)``. Every value is worked out numerically from
    the model's own functions at each frailty, with no simulation; only
    the bands are drawn (seeded). See the guide's "Repair again or
    replace?".

    surpyval's joint frailty model is new in 0.27 and experimental: its
    design is proposed, not yet settled, so its names and defaults may
    change. It has Weibull baselines only and no bootstrap.

    Parameters
    ----------
    model : surpyval.recurrent.JointFrailtyModel
        A joint frailty model on the calendar time scale, fitted
        (``surpyval.JointFrailty.fit``) or built with ``from_params``.
        Ages and cost rates are in its time unit.
    repair_cost : float
        The cost of each repair, at least 0.
    replacement_cost : float
        The cost of replacing a unit by a new one when it is planned (at
        an age, or at a failure the rule says to replace at), more than 0.
    failure_cost : float, optional
        The cost of each failure besides its repair or replacement (the
        downtime, the lost production), at least 0; default 0.
    write_off_cost : float, optional
        The cost of a write-off (the replacement forced by it, and its
        downtime); default ``replacement_cost + failure_cost``.
    discount_rate : float, optional
        The continuous discount rate per unit of the model's time, at
        least 0; default 0 (none). With one, the cost rates are equivalent
        cost rates (see
        [`RepairOrReplacePolicy`][repyability.RepairOrReplacePolicy]).
    Z : array_like, optional
        The covariates of the units, one row, for a model fitted with
        covariates.

    Raises
    ------
    TypeError
        If ``model`` is not a surpyval ``JointFrailtyModel``.
    NotImplementedError
        If the model is on the gap time scale.
    ValueError
        If a cost or the discount rate is out of range, or the model has
        covariates and ``Z`` is not given.

    Examples
    --------
    A fleet whose units fail at a rising rate (shape 2: a typical unit
    expects its first failure at 100 hours, and four by 200) and are
    written off at a rising hazard, the frail ones sooner (``alpha = 1``).
    A repair costs 1, a replacement 10, and each failure's downtime 2:

    >>> from surpyval import JointFrailty
    >>> from repyability import RepairOrReplace
    >>> model = JointFrailty.from_params((100.0, 2.0), (400.0, 2.0), 0.5, 1.0)
    >>> fleet = RepairOrReplace(
    ...     model, repair_cost=1.0, replacement_cost=10.0, failure_cost=2.0
    ... )
    >>> policy = fleet.fleet_policy()
    >>> policy.recommended, policy.best_failure_count
    ('replace at failure', 4)
    >>> round(policy.cost_rate, 4), round(policy.keep_repairing_cost_rate, 4)
    (0.0899, 0.135)
    """

    def __init__(
        self,
        model,
        *,
        repair_cost,
        replacement_cost,
        failure_cost=0.0,
        write_off_cost=None,
        discount_rate=0.0,
        Z=None,
    ):
        _check_model(model)
        repair = _cost(repair_cost, "repair_cost")
        replacement = _cost(
            replacement_cost, "replacement_cost", positive=True
        )
        failure = _cost(failure_cost, "failure_cost")
        write_off = (
            replacement + failure
            if write_off_cost is None
            else _cost(write_off_cost, "write_off_cost")
        )
        discount = _cost(discount_rate, "discount_rate")
        if model.n_covariates:
            if Z is None:
                raise ValueError(
                    f"the model has {model.n_covariates} covariate(s): give "
                    "the units' covariates as Z, one row."
                )
            row = checks.real_array(Z, "Z").reshape(-1)
            if row.size != model.n_covariates:
                raise ValueError(
                    f"Z must have {model.n_covariates} value(s), one per "
                    f"covariate; got {row.size}."
                )
            Z = row
        else:
            Z = None
        self.model = model
        self.Z = Z
        self.costs = {
            "repair": repair,
            "replacement": replacement,
            "failure": failure,
            "write_off": write_off,
            "discount": discount,
        }
        self._engine = _Engine(model, Z, self.costs)
        self._fleet: Dict[int, Tuple[Any, ...]] = {}
        self._draws: Dict[Tuple[Any, ...], Tuple[Any, str]] = {}

    # -- the fleet ----------------------------------------------------------
    def _best(self, max_failures: int):
        """The best of each rule for a new unit: memoised by the largest
        count searched."""
        if max_failures in self._fleet:
            return self._fleet[max_failures]
        engine = self._engine
        cycle = engine.fleet()
        keep = engine.keep_rate(cycle)
        cost, time = cycle.ages()
        with np.errstate(divide="ignore", invalid="ignore"):
            ages = np.where(np.isfinite(cost / time), cost / time, np.inf)
        i = int(np.argmin(ages))
        best_age, age_rate = math.inf, keep
        if i < len(ages) - 1:

            def rate(s, j):
                c, t = cycle.age(s, j)
                return c / t

            T, value = _refine(rate, cycle.edges, i)
            if math.isnan(keep) or value < keep * (1 - _TIE):
                best_age, age_rate = T, value
        counts = engine.count_rates(max_failures, cycle)
        finite = np.isfinite(counts)
        k = int(np.nanargmin(counts)) + 1 if finite.any() else 1
        count_rate = float(counts[k - 1])
        options = [(_KEEP, keep), (_AGE, age_rate), (_COUNT, count_rate)]
        if math.isinf(best_age):
            options = [o for o in options if o[0] != _AGE]
        recommended, best = options[0]
        for name, value in options[1:]:
            if math.isfinite(value) and (
                not math.isfinite(best) or value < best * (1 - _TIE)
            ):
                recommended, best = name, value
        notes = []
        if math.isnan(keep):
            notes.append(
                "Undiscounted, a new unit's expected life (or repairs) "
                "before write-off is infinite for this frailty "
                "distribution, so keeping on repairing has no long-run "
                "cost rate: give a discount rate."
            )
        if k == max_failures:
            notes.append(
                f"The best failure count is the largest searched "
                f"({max_failures}): raise max_failures."
            )
        out = (
            recommended,
            best,
            keep,
            best_age,
            age_rate,
            k,
            count_rate,
            tuple(notes),
        )
        self._fleet[max_failures] = out
        return out

    def _drawn_engines(self, mc_samples: int, seed):
        """Engines at ``mc_samples`` draws of the fitted parameters from
        their covariance, with what the bands are; or None and why
        there are none."""
        key = (mc_samples, None if seed is None else repr(seed))
        if seed is not None and key in self._draws:
            return self._draws[key]
        from surpyval import JointFrailty as fitter_

        fitter: Any = fitter_

        model = self.model
        try:
            cov = np.asarray(model.covariance(), dtype=float)
        except (ValueError, np.linalg.LinAlgError):
            return (
                None,
                "none: the model carries no covariance (it was built from "
                "parameters, or restored from a file), so its cost rates "
                "have no bands",
            )
        params = np.asarray(model.params, dtype=float)
        diag = np.diag(cov) if cov.ndim == 2 else np.zeros(0)
        free = np.flatnonzero(
            np.isfinite(diag) & (diag > 0) & np.isfinite(params)
        )
        if not free.size or not np.all(np.isfinite(cov[np.ix_(free, free)])):
            return (
                None,
                "none: the fit's covariance could not be worked out (its "
                "information matrix is singular)",
            )
        logged = np.array([j < 5 and params[j] > 0 for j in free])
        centre = np.where(logged, np.log(params[free]), params[free])
        jac = np.where(logged, 1.0 / params[free], 1.0)
        sub = cov[np.ix_(free, free)] * np.outer(jac, jac)
        rng = np.random.default_rng(seed)
        drawn = rng.multivariate_normal(centre, sub, size=mc_samples)
        drawn = np.where(logged, np.exp(drawn), drawn)
        base = np.nan_to_num(params, nan=0.0)
        p = model.n_covariates
        engines = []
        for row in drawn:
            values = base.copy()
            values[free] = row
            values[4] = max(values[4], 0.0)
            draw = fitter.from_params(
                recurrent=values[0:2],
                terminal=values[2:4],
                theta=float(values[4]),
                alpha=float(values[5]),
                beta=values[6 : 6 + p] if p else None,
                gamma=values[6 + p :] if p else None,
                family=model.family,
                time_scale=model.time_scale,
            )
            engines.append(_Engine(draw, self.Z, self.costs))
        out = (engines, _BOUNDS)
        if seed is not None:
            self._draws[key] = out
        return out

    @staticmethod
    def _band(values) -> Tuple[float, float]:
        values = np.asarray(values, dtype=float)
        tail = 50.0 * (1.0 - _BAND)
        if not np.any(np.isfinite(values)):
            return (math.nan, math.nan)
        lo, hi = np.nanpercentile(values, [tail, 100.0 - tail])
        return (float(lo), float(hi))

    def _drawn_fleet(self, mc_samples: int, seed, kmax: int):
        """Each rule's cost rate for a new unit at each draw of the
        parameters (at the best age and count of the fitted model), with
        what the bands are; or None and why there are none."""
        engines, bounds = self._drawn_engines(mc_samples, seed)
        if engines is None:
            return None, bounds
        key = ("fleet", mc_samples, repr(seed), kmax)
        if seed is not None and key in self._draws:
            return self._draws[key]
        best_age, k = self._best(kmax)[3], self._best(kmax)[5]
        rates: Dict[str, list] = {_KEEP: [], _AGE: [], _COUNT: []}
        for e in engines:
            cycle = e.fleet()
            rates[_KEEP].append(e.keep_rate(cycle))
            rates[_AGE].append(
                e.age_rate(best_age, cycle)
                if math.isfinite(best_age)
                else rates[_KEEP][-1]
            )
            rates[_COUNT].append(float(e.count_rates(k, cycle)[k - 1]))
        out = ({n: np.asarray(v) for n, v in rates.items()}, bounds)
        if seed is not None:
            self._draws[key] = out
        return out

    def fleet_policy(
        self,
        *,
        max_failures: int = 30,
        uncertainty: bool = True,
        mc_samples: Optional[int] = None,
        seed=None,
    ) -> RepairOrReplacePolicy:
        """The best rule for a new unit: keep repairing it, replace it at
        an age, or replace it at a count of failures.

        A new unit's frailty is unknown: it is drawn from the fleet's
        frailty distribution. Each rule's cost rate is the expected cost
        of a cycle (from new until the rule replaces the unit or it is
        written off) over its expected length, each discounted:

        - keep repairing: every failure repaired (``repair_cost +
          failure_cost``) until the write-off (``write_off_cost``);
        - replace at age ``T``: as keep repairing, but a unit reaching
          ``T`` is replaced (``replacement_cost``); the best ``T`` is
          searched over every age;
        - replace at failure ``k``: repair the first ``k - 1`` failures
          and replace at the ``k``-th (``failure_cost +
          replacement_cost``), unless written off first; ``k`` from 1 to
          ``max_failures``.

        Counting failures pays for two reasons: where units differ (a
        frailty variance), a unit with many early failures is likely a
        lemon; and replacing at a failure saves that failure's repair.

        Parameters
        ----------
        max_failures : int, optional
            The largest failure count searched, default 30.
        uncertainty : bool, optional
            Whether to give the 90% two-sided bands on the cost rates,
            from draws of the fitted parameters (default True). A model
            built from parameters has none.
        mc_samples : int, optional
            The number of parameter draws for the bands, default 500.
        seed : int, optional
            Seeds the draws, for reproducible bands.

        Returns
        -------
        RepairOrReplacePolicy
            The recommended rule and the cost rate of each.

        Raises
        ------
        ValueError
            If ``max_failures`` or ``mc_samples`` is not a whole number
            of at least 1.

        Examples
        --------
        >>> from surpyval import JointFrailty
        >>> from repyability import RepairOrReplace
        >>> model = JointFrailty.from_params(
        ...     (100.0, 2.0), (400.0, 2.0), 0.5, 1.0
        ... )
        >>> fleet = RepairOrReplace(
        ...     model, repair_cost=1.0, replacement_cost=10.0, failure_cost=2.0
        ... )
        >>> policy = fleet.fleet_policy()
        >>> round(policy.best_age, 1), round(policy.age_cost_rate, 4)
        (199.2, 0.1108)
        >>> policy.bands is None   # built from parameters: no covariance
        True
        """
        kmax = checks.whole_number(max_failures, "max_failures")
        draws = checks.whole_number(
            _DEFAULT_DRAWS if mc_samples is None else mc_samples, "mc_samples"
        )
        seed = checks.seed(seed)
        recommended, rate, keep, best_age, age_rate, k, count_rate, notes = (
            self._best(kmax)
        )
        bands = None
        bounds = "none: not asked for (uncertainty=False)"
        if uncertainty:
            drawn, bounds = self._drawn_fleet(draws, seed, kmax)
            if drawn is not None:
                bands = {
                    _KEEP: self._band(drawn[_KEEP]),
                    _COUNT: self._band(drawn[_COUNT]),
                }
                if math.isfinite(best_age):
                    bands[_AGE] = self._band(drawn[_AGE])
        return RepairOrReplacePolicy(
            recommended=recommended,
            cost_rate=float(rate),
            keep_repairing_cost_rate=float(keep),
            best_age=float(best_age),
            age_cost_rate=float(age_rate),
            best_failure_count=int(k),
            failure_count_cost_rate=float(count_rate),
            discount_rate=self.costs["discount"],
            bands=bands,
            bounds=bounds,
            notes=notes,
        )

    # -- one unit -----------------------------------------------------------
    def _frailty_of(self, unit, frailty) -> float:
        if (unit is None) == (frailty is None):
            raise ValueError(
                "give the unit's name in the fitted model (unit=) or its "
                "frailty (frailty=), one of them."
            )
        if frailty is not None:
            value = checks.number_or_nan(frailty)
            if not (math.isfinite(value) and value > 0):
                raise ValueError(
                    "frailty must be a finite number more than 0, got "
                    f"{frailty!r}."
                )
            return value
        known = self.model.frailties
        if not known:
            raise ValueError(
                "the model has no fitted units' frailties (it was built "
                "from parameters, or restored from a file): give the "
                "unit's frailty as frailty=."
            )
        if unit not in known:
            raise ValueError(
                f"the model has no unit {unit!r}: its units are named as "
                "in the data it was fitted to (i)."
            )
        return float(known[unit])

    @staticmethod
    def _unit_rules(engine: _Engine, a: float, u: float, g: float, kmax: int):
        """For a unit of frailty ``u`` at age ``a``, at the fleet's cost
        rate ``g``: each rule's cost and discounted time from now, the
        best age and count, and each rule's value against replacing now
        (its cost less ``g`` times its time, less the replacement's
        cost)."""
        cycle = engine.unit(a, u)

        def value(cost, time):
            return cost - g * time - engine.cp

        keep = cycle.keep()
        cost, time = cycle.ages()
        values = value(cost, time)
        i = int(np.argmin(values))
        age = (math.inf, keep)
        if 0 < i < len(values) - 1:

            def at(s, j):
                return value(*cycle.age(s, j))

            s, _ = _refine(at, cycle.edges, i)
            age = (
                a + s,
                cycle.age(
                    s,
                    max(
                        int(np.searchsorted(cycle.edges, s, side="right")) - 1,
                        0,
                    ),
                ),
            )
        elif i == 0:
            age = (a, (math.nan, math.nan))
        cost, time = cycle.counts(kmax)
        counts = value(cost, time)
        k = int(np.argmin(counts)) + 1
        return (
            {
                _KEEP: keep,
                _AGE: age[1],
                _COUNT: (float(cost[k - 1]), float(time[k - 1])),
            },
            age[0],
            k,
            value,
        )

    def _unit_savings(self, engine, a, u, g, kmax, just_failed):
        rules, best_age, k, value = self._unit_rules(engine, a, u, g, kmax)
        repair = engine.cr if just_failed else 0.0
        savings = {name: -value(*ct) - repair for name, ct in rules.items()}
        if math.isinf(best_age):
            savings[_AGE] = savings[_KEEP]
        elif best_age <= a:
            savings[_AGE] = 0.0
        return rules, savings, best_age, k

    def _break_even(
        self, a: float, g: float, kmax: int, just_failed: bool
    ) -> float:
        """The frailty at which the best rule for keeping the unit saves
        nothing against replacing it now."""

        def f(log_u):
            _, savings, best_age, _ = self._unit_savings(
                self._engine, a, math.exp(log_u), g, kmax, just_failed
            )
            if best_age <= a:
                savings.pop(_AGE)
            return max(savings.values())

        grid = np.linspace(-12.0, 12.0, 25)
        values = np.array([f(v) for v in grid])
        if values[0] <= 0:
            return 0.0
        below = np.flatnonzero(values <= 0)
        if not below.size:
            return math.inf
        j = int(below[0])
        return math.exp(brentq(f, grid[j - 1], grid[j], xtol=1e-10))

    def break_even_frailty(
        self, age, *, just_failed: bool = False, max_failures: int = 30
    ) -> float:
        """The frailty at which a unit of age ``age`` should be replaced
        now.

        Below it, the best rule for keeping the unit (repairing it until
        it is written off, replacing it at a later age or at a further
        failure; see
        [`unit_decision`][repyability.RepairOrReplace.unit_decision])
        saves money against replacing it now; above it, replacing it now
        is best. It falls with age where units wear out.

        Parameters
        ----------
        age : float
            The unit's age, at least 0, in the model's time.
        just_failed : bool, optional
            Whether the unit has just failed, so that keeping it means
            paying for its repair now (default False).
        max_failures : int, optional
            The largest failure count searched, default 30.

        Returns
        -------
        float
            The frailty; ``inf`` when no frailty makes replacing now pay
            at this age, 0 when every one does.

        Raises
        ------
        ValueError
            If ``age`` is not a finite number of at least 0.

        Examples
        --------
        >>> from surpyval import JointFrailty
        >>> from repyability import RepairOrReplace
        >>> model = JointFrailty.from_params(
        ...     (100.0, 2.0), (400.0, 2.0), 0.5, 1.0
        ... )
        >>> fleet = RepairOrReplace(
        ...     model, repair_cost=1.0, replacement_cost=10.0, failure_cost=2.0
        ... )
        >>> round(fleet.break_even_frailty(150.0), 3)
        1.242
        >>> round(fleet.break_even_frailty(150.0, just_failed=True), 3)
        0.786
        """
        a = self._age(age)
        kmax = checks.whole_number(max_failures, "max_failures")
        return self._break_even(
            a, self._best(kmax)[1], kmax, bool(just_failed)
        )

    @staticmethod
    def _age(age) -> float:
        value = checks.number_or_nan(age)
        if not (math.isfinite(value) and value >= 0):
            raise ValueError(
                f"age must be a finite number, at least 0; got {age!r}."
            )
        return value

    def unit_decision(
        self,
        age,
        *,
        unit: Optional[Hashable] = None,
        frailty=None,
        just_failed: bool = False,
        max_failures: int = 30,
        uncertainty: bool = True,
        mc_samples: Optional[int] = None,
        seed=None,
    ) -> UnitDecision:
        """Repair again or replace, for one unit at its age.

        The unit's frailty is the model's estimate for it (``unit=``, its
        name in the data the model was fitted to) or one given
        (``frailty=``), and is taken as known. Each way of keeping the
        unit is priced against replacing it now: repair it until it is
        written off, replace it at a later age (the best one), or replace
        it at a further failure (the best count). What follows its
        replacement, now or later, is valued at the fleet's cost rate (a
        new unit under the fleet's best rule), so a rule saves money when
        the unit's costs until it leaves are less than the fleet's cost
        rate over that time, plus the replacement's cost put off. With
        ``just_failed``, keeping the unit also means paying for its repair
        now, the question at a failure: repair it again, or replace it?

        As its history grows its frailty estimate changes: decide again
        with a refitted model.

        Parameters
        ----------
        age : float
            The unit's age now, at least 0, in the model's time.
        unit : hashable, optional
            The unit's name in the fitted model (a key of
            ``model.frailties``).
        frailty : float, optional
            The unit's frailty, more than 0, in place of ``unit``.
        just_failed : bool, optional
            Whether the unit has just failed (default False).
        max_failures : int, optional
            The largest failure count searched, for the fleet and the
            unit, default 30.
        uncertainty : bool, optional
            Whether to give the 90% two-sided bands, default True.
        mc_samples : int, optional
            The number of parameter draws for the bands, default 500.
        seed : int, optional
            Seeds the draws, for reproducible bands.

        Returns
        -------
        UnitDecision
            The recommendation, each rule's saving and cost rate, and the
            break-even frailty.

        Raises
        ------
        ValueError
            If neither or both of ``unit`` and ``frailty`` are given, the
            model has no such unit or no fitted frailties, or an argument
            is out of range.

        Examples
        --------
        >>> from surpyval import JointFrailty
        >>> from repyability import RepairOrReplace
        >>> model = JointFrailty.from_params(
        ...     (100.0, 2.0), (400.0, 2.0), 0.5, 1.0
        ... )
        >>> fleet = RepairOrReplace(
        ...     model, repair_cost=1.0, replacement_cost=10.0, failure_cost=2.0
        ... )
        >>> lemon = fleet.unit_decision(150.0, frailty=3.0)
        >>> lemon.recommended, lemon.saving
        ('replace now', 0.0)
        >>> sound = fleet.unit_decision(150.0, frailty=0.5)
        >>> sound.recommended, sound.best_failure_count, round(sound.saving, 2)
        ('replace at failure', 3, 3.22)
        """
        a = self._age(age)
        u = self._frailty_of(unit, frailty)
        kmax = checks.whole_number(max_failures, "max_failures")
        draws = checks.whole_number(
            _DEFAULT_DRAWS if mc_samples is None else mc_samples, "mc_samples"
        )
        seed = checks.seed(seed)
        if uncertainty and seed is None:
            # One seed for the unit's draws and the fleet's, so that they
            # are the same draws.
            seed = int(np.random.default_rng().integers(2**32))
        failed = bool(just_failed)
        rule, g, _, fleet_age, _, fleet_k, _, notes = self._best(kmax)
        engine = self._engine
        rules, savings, best_age, k = self._unit_savings(
            engine, a, u, g, kmax, failed
        )
        recommended, saving = _NOW, 0.0
        for name in (_KEEP, _AGE, _COUNT):
            if name == _AGE and math.isinf(best_age):
                continue
            if savings[name] > max(saving, 0.0) + _TIE * engine.cp:
                recommended, saving = name, savings[name]
        with np.errstate(invalid="ignore"):
            rates = {name: c / t for name, (c, t) in rules.items()}
        bands = None
        bounds = "none: not asked for (uncertainty=False)"
        if uncertainty:
            engines, bounds = self._drawn_engines(draws, seed)
            if engines is not None:
                bounds = bounds + " (the unit's frailty held at its estimate)"
                fleet = self._drawn_fleet(draws, seed, kmax)[0][rule]
                drawn: Dict[str, list] = {
                    name: [] for name in (_KEEP, _AGE, _COUNT)
                }
                for e in engines:
                    cycle = e.unit(a, u)
                    drawn[_KEEP].append(cycle.keep()[0] / cycle.keep()[1])
                    if a < best_age < math.inf:
                        s = best_age - a
                        j = max(
                            int(np.searchsorted(cycle.edges, s, side="right"))
                            - 1,
                            0,
                        )
                        c, t = cycle.age(s, j)
                        drawn[_AGE].append(c / t)
                    c, t = cycle.counts(k)
                    drawn[_COUNT].append(c[k - 1] / t[k - 1])
                bands = {
                    "keep_cost_rate": self._band(drawn[_KEEP]),
                    "failure_count_cost_rate": self._band(drawn[_COUNT]),
                    "fleet_cost_rate": self._band(fleet),
                }
                if drawn[_AGE]:
                    bands["age_cost_rate"] = self._band(drawn[_AGE])
        notes = notes + (
            "The unit's frailty is taken as known at its estimate; decide "
            "again as its history grows.",
        )
        return UnitDecision(
            unit=unit,
            age=a,
            frailty=u,
            just_failed=failed,
            recommended=recommended,
            saving=float(saving),
            keep_saving=float(savings[_KEEP]),
            age_saving=float(savings[_AGE]),
            failure_count_saving=float(savings[_COUNT]),
            best_age=float(best_age),
            best_failure_count=int(k),
            keep_cost_rate=float(rates[_KEEP]),
            age_cost_rate=float(rates[_AGE]),
            failure_count_cost_rate=float(rates[_COUNT]),
            cost_rate_now=engine.marginal(a, u),
            fleet_cost_rate=float(g),
            break_even_frailty=self._break_even(a, g, kmax, failed),
            bands=bands,
            bounds=bounds,
            notes=notes,
        )

    def unit_decisions(
        self,
        ages: Mapping[Hashable, float],
        *,
        max_failures: int = 30,
        uncertainty: bool = True,
        mc_samples: Optional[int] = None,
        seed=None,
    ) -> Dict[Hashable, UnitDecision]:
        """[`unit_decision`][repyability.RepairOrReplace.unit_decision]
        for each unit of the fitted model named in ``ages``, at its age.

        Parameters
        ----------
        ages : mapping
            Each unit's age now, by its name in the fitted model.
        max_failures : int, optional
            The largest failure count searched for the fleet's best rule,
            default 30.
        uncertainty : bool, optional
            Whether to give the 90% two-sided bands, default True.
        mc_samples : int, optional
            The number of parameter draws for the bands, default 500.
        seed : int, optional
            Seeds the draws; every unit's bands use the same draws.

        Returns
        -------
        dict
            A [`UnitDecision`][repyability.UnitDecision] by unit, in the
            order of ``ages``.

        Raises
        ------
        ValueError
            As ``unit_decision``, for any unit.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import JointFrailty
        >>> from repyability import RepairOrReplace
        >>> truth = JointFrailty.from_params(
        ...     (100.0, 2.0), (400.0, 2.0), 0.5, 1.0
        ... )
        >>> rows = truth.random(60, end=250.0, random_state=2)
        >>> model = JointFrailty.fit(**rows)
        >>> fleet = RepairOrReplace(
        ...     model, repair_cost=1.0, replacement_cost=10.0, failure_cost=2.0
        ... )
        >>> decisions = fleet.unit_decisions(
        ...     {0: 120.0, 1: 120.0}, mc_samples=50, seed=1
        ... )
        >>> sorted(decisions)
        [0, 1]
        """
        if not isinstance(ages, Mapping):
            raise TypeError(
                "ages must be a mapping of each unit's age by its name, got "
                f"{type(ages).__name__}."
            )
        if seed is None and uncertainty:
            seed = int(np.random.default_rng().integers(2**32))
        return {
            name: self.unit_decision(
                age,
                unit=name,
                max_failures=max_failures,
                uncertainty=uncertainty,
                mc_samples=mc_samples,
                seed=seed,
            )
            for name, age in ages.items()
        }
