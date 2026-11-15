"""A part's replacement policy at a site whose shuts are on real dates
(UC-43 of this package's ``README.md``), worked out by the route the
site's shuts allow (``_shutdowns._decide``):

- **exact** (and **exact_long_cycle**): the shuts repeat in the part's
  clock, so the long-run average is the existing calendar policy's
  (``_replacement``: relative value iteration over the cycle on a grid,
  or ``_integration``'s quadrature) over the shuts laid on that clock.
- **sampling**: the same long-run average estimated from sampled slices
  of a long cycle (``_replacement._sampled_policy``).
- **averaged**: the renewal-reward formula with each kind of shut's
  timing taken as evenly spread within its usual gaps (``_averaged``):
  a part reaching the age from which a kind replaces it waits for that
  kind's next shut a time drawn from the equilibrium residual of its
  gaps, independently of the other kinds. Opportunity-based age
  replacement in the sense of Dekker & Dijkstra (1992), Naval Research
  Logistics 39(2):175-190, and Dekker & Smeitink (1991), European
  Journal of Operational Research 53(1):46-63.
- **horizon**: the lowest net present cost on the real dates to a
  horizon (``_horizon``).

Every result says which, in words.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, ClassVar, Dict, List, Optional, Tuple, cast

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from repyability.maintenance._calendar import ANYTIME, Calendar, Opportunity
from repyability.maintenance._shutdowns import (
    SlowSolverWarning,
    _day_text,
    _decide,
    _Decision,
    _Layout,
    _money,
    _number,
    _seconds_text,
    quantity,
)
from repyability.rbd.results import _ResultMapping
from repyability.utils import checks
from repyability.utils.wrappers import outside_level

#: How a planned replacement on any other day is named in results.
ANY_DAY = "any other day"
#: Separates a kind's name from the number of one of its shuts in a
#: cycle, where a kind has several shuts a cycle (each laid as an
#: opportunity of its own).
_PART = "\x1f"
#: Gauss-Legendre nodes a piece of the averaged formula's integrals.
_NODES = 24
#: The chance of a part outliving the end of the averaged formula's
#: integrals.
_TAIL = 1e-12
#: Seconds of quadrature over which a ``SlowSolverWarning`` is given.
SLOW_SECONDS = 5.0


# ---------------------------------------------------------------- results


@dataclass(frozen=True)
class ReplacementRule(_ResultMapping):
    """The best rule for a part at a site, in words and as data: part of
    [`ShutdownPolicy`][repyability.ShutdownPolicy].

    Attributes
    ----------
    text : str
        The rule in words.
    thresholds : dict
        Each kind of shut (and ``"any other day"`` where the site allows
        one), the youngest and oldest age at which it replaces the part,
        ``(inf, inf)`` for never. They differ where the age depends on
        when in the cycle the shut falls (a quarterly shut just before
        the annual waits for an older part).
    by_period : list or None
        For a horizon, the thresholds by calendar year:
        ``(first day, last day, {kind: (youngest, oldest)})``; None
        otherwise.
    unit : str or None
        The unit of the ages.
    """

    text: str
    thresholds: Dict[str, Tuple[float, float]]
    by_period: Optional[List[Any]] = None
    unit: Optional[str] = None


@dataclass(frozen=True)
class DatedReplacement(_ResultMapping):
    """A planned replacement on the real dates: when, at which shut, and
    how old the part will be.

    Attributes
    ----------
    date : str or None
        The day, as ISO text (None with no ``now``).
    kind : str or None
        The shut (``"any other day"`` for a day with none); None if no
        planned replacement comes.
    age : float
        The part's age then, in its unit.
    time : float
        Model time from now (the part's clock).
    price : float or None
        What the replacement costs.
    """

    date: Optional[str]
    kind: Optional[str]
    age: float
    time: float
    price: Optional[float]

    def __str__(self) -> str:
        if self.kind is None:
            return "No planned replacement comes."
        when = (
            _day_text(date.fromisoformat(self.date))
            if self.date
            else f"in {_number(self.time)}"
        )
        place = (
            "on " + when
            if self.kind == ANY_DAY
            else f"at the {self.kind}, {when}"
        )
        return f"{place[0].upper()}{place[1:]}, part {_number(self.age)} old"


@dataclass(frozen=True)
class RuleCost(_ResultMapping):
    """What a given rule costs at a site, and how that was worked out:
    returned by
    [`NonRepairable.calendar_policy_cost`][repyability.NonRepairable.calendar_policy_cost]
    with a site, and listed by
    [`ShutdownPolicy.compare`][repyability.ShutdownPolicy.compare].

    Attributes
    ----------
    name : str
        The rule in a few words.
    rule : dict
        The age from which each kind replaces (``inf``: never).
    cost : float
        The figure in ``measure``.
    measure : str
        ``"cost per year"``, ``"net present cost"`` or, where the part's
        clock has no pace in time, ``"cost per <unit>"``.
    cost_se : float or None
        Its standard error, where sampled.
    route : str
        The route it was priced by.
    route_text : str
        The figure in words.
    solver : str
        How the numbers were worked out.
    failure_timing : str or None
        How a failure's replacement was timed, where a solver times it.
    replacements_by_kind : dict
        Expected replacements a year by kind, and ``"failure"`` (over
        the horizon, for a horizon; per unit of the part's clock where it
        has no pace).
    cost_rate : float
        The long-run cost a unit of the part's clock (routes 1-3).
    """

    name: str
    rule: Dict[str, float]
    cost: float
    measure: str
    cost_se: Optional[float]
    route: str
    route_text: str
    solver: str
    failure_timing: Optional[str]
    replacements_by_kind: Dict[str, float]
    cost_rate: Optional[float]


@dataclass(frozen=True)
class ShutdownPolicy(_ResultMapping):
    """The best replacement rule for a part at a site whose shuts are on
    real dates, what it costs, and how that was worked out. Returned by
    [`NonRepairable.optimal_calendar_policy`][repyability.NonRepairable.optimal_calendar_policy]
    given a [`Shutdowns`][repyability.Shutdowns].

    Attributes
    ----------
    route : str
        ``"exact"``, ``"exact_long_cycle"``, ``"sampling"``,
        ``"averaged"`` or ``"horizon"``.
    route_text : str
        The answer and how it was worked out, in plain words.
    measure : str
        ``"cost per year"`` (routes 1-3), ``"net present cost"`` (the
        horizon), or ``"cost per <unit>"`` where the part's clock has no
        pace in time. Never compare figures of different measures.
    cost : float
        The figure in ``measure``.
    cost_se : float or None
        Its standard error, for sampling.
    solver : str
        ``"closed form"``, ``"quadrature"``, ``"grid"`` or ``"sampling"``.
    failure_timing : str or None
        ``"half_step"`` or ``"exact"`` where a solver times failures;
        None for the closed form.
    rule : ReplacementRule
        The rule, in words and as data.
    replacements_by_kind : dict
        Expected replacements a year by kind and ``"failure"`` (over the
        horizon, a year on average, for a horizon).
    cost_rate : float or None
        The long-run cost a unit of the part's clock (routes 1-3).
    unit : str or None
        The part's unit.
    now : str or None
        The day answers count from.
    age : float
        The age of the part in service at ``now``.
    notes : tuple of str
        What else was assumed.
    settings : dict
        How it was worked out (the solver's own settings).
    life : dict or None
        The saved form of the life it was worked out for.
    """

    # Bound after construction (``_bind``), not fields: how the dates are
    # answered, the layout of the shuts and the route decided.
    _engine: ClassVar[Any] = None
    _layout: ClassVar[Any] = None
    _decision: ClassVar[Any] = None

    route: str = ""
    route_text: str = ""
    measure: str = ""
    cost: float = math.nan
    cost_se: Optional[float] = None
    solver: str = ""
    failure_timing: Optional[str] = None
    rule: Optional[ReplacementRule] = None
    replacements_by_kind: Dict[str, float] = field(default_factory=dict)
    cost_rate: Optional[float] = None
    unit: Optional[str] = None
    now: Optional[str] = None
    age: float = 0.0
    notes: Tuple[str, ...] = ()
    settings: Dict[str, Any] = field(default_factory=dict)
    life: Optional[dict] = None

    def _bind(self, engine, layout, decision) -> "ShutdownPolicy":
        object.__setattr__(self, "_engine", engine)
        object.__setattr__(self, "_layout", layout)
        object.__setattr__(self, "_decision", decision)
        return self

    def __repr__(self) -> str:
        return (
            f"ShutdownPolicy(route={self.route!r}, measure={self.measure!r}, "
            f"cost={self.cost!r}, solver={self.solver!r})"
        )

    # -- the part in service ------------------------------------------

    @property
    def next_replacement(self) -> DatedReplacement:
        """When the part in service (``age`` old at ``now``) will be
        replaced as planned, if it does not fail first."""
        time, price, kind = self._engine.next_planned(-self.age, 0.0)
        return self._dated(time, price, kind, -self.age)

    def _dated(self, time, price, kind, installed) -> DatedReplacement:
        if kind is None:
            return DatedReplacement(None, None, math.inf, math.inf, None)
        layout = self._layout
        if kind in layout.site.names:
            shut = next(k for k in layout.kinds if k.name == kind)
            step = self._engine.step
            time, moment = shut.nearest(time, 0.5 * step if step else 0.0)
        else:
            moment = layout.moment_at([time])[0]
        day = (
            None
            if moment is None
            else self._layout.clock.local_naive(moment).date().isoformat()
        )
        return DatedReplacement(
            day, kind, float(time - installed), float(time), float(price)
        )

    def schedule(
        self, until
    ) -> List[Tuple[Optional[str], Optional[str], float]]:
        """The planned replacements of a part fitted at ``now`` and of its
        successors, to ``until`` (a date, or with no ``now`` a model
        time), if none fails: ``(date, kind, age)`` for each.

        Parameters
        ----------
        until : date, str or float
            The last day (included).

        Returns
        -------
        list of (str, str, float)
        """
        layout = self._layout
        if layout.now is None:
            end = float(until)
        else:
            from repyability.maintenance._shutdowns import _iso_date

            day = _iso_date(until, "until")
            end = float(
                layout.model_at([layout.clock.midnight(day + _ONE_DAY)])[0]
            )
        out: List[Tuple[Optional[str], Optional[str], float]] = []
        installed = 0.0
        after = 0.0
        while True:
            time, price, kind = self._engine.next_planned(installed, after)
            if kind is None or time > end:
                return out
            dated = self._dated(time, price, kind, installed)
            out.append((dated.date, dated.kind, dated.age))
            installed = after = float(time)

    def compare(self) -> List[RuleCost]:
        """The same site under simpler rules, priced the same way, for
        context: run to failure, age replacement on any day (where the
        site allows one), and every shut of each kind."""
        layout = self._layout
        part = layout.part
        rules: List[Tuple[str, Dict[str, float]]] = [("run to failure", {})]
        if layout.site.any_day is not None:
            best = _age_replacement(part, layout.site.any_day)
            rules.append(
                (
                    f"age replacement on any day at {_number(best)}",
                    {ANY_DAY: best},
                )
            )
        for name in layout.site.names:
            rules.append((f"every {name}", {name: 0.0}))
        return [
            _rule_cost(layout, self._decision, rule, name, self.settings)
            for name, rule in rules
        ]

    def _next_planned(self, installed, after):
        """``CalendarPolicy._next_planned`` on the part's clock: what a
        diagram's ``"calendar"`` policy asks at each renewal."""
        return self._engine.next_planned(installed, after)

    def _planned_price(self, t: float) -> float:
        return self._engine.price_at(t)

    @property
    def step(self) -> Optional[float]:
        """The grid's step the dates are answered on (None by
        quadrature or the closed form)."""
        return self._engine.step

    def _calendar_policy(self, end: float):
        """The policy as a grid ``CalendarPolicy`` to model time ``end``,
        for a cost's spread (``cost_distribution``)."""
        return self._engine.grid_policy(end)


# ------------------------------------------------------------ the engines


class _KindTimes:
    """The real model times of each kind's shuts, extended as asked."""

    def __init__(self, layout: _Layout):
        self.layout = layout
        self.end = 0.0
        self.times: Dict[str, np.ndarray] = {}

    def upto(self, end: float) -> Dict[str, np.ndarray]:
        if end > self.end or not self.times:
            end = max(end, 2.0 * self.end)
            self.times = {k.name: k.times(end) for k in self.layout.kinds}
            self.end = end
        return self.times


class _ThresholdEngine:
    """A rule of one threshold age a kind (and one for any other day),
    on the real shuts: the averaged route's dates."""

    step = None

    def __init__(self, layout: _Layout, thresholds: Dict[str, float], any_age):
        self.layout = layout
        self.thresholds = thresholds
        self.any_age = any_age
        self.prices = {k.name: k.price for k in layout.kinds}
        self.kind_times = _KindTimes(layout)
        self._grids: Dict[float, Any] = {}

    def next_planned(self, installed: float, after: float):
        best: Tuple[float, float, Optional[str]] = (math.inf, 0.0, None)
        strict = after <= installed
        tolerance = 1e-9 * max(1.0, abs(after))
        if self.any_age is not None and math.isfinite(self.any_age):
            t = max(installed + self.any_age, after)
            if not (strict and t <= after + tolerance):
                best = (t, float(self.layout.site.any_day or 0.0), ANY_DAY)
        reach = self._reach()
        for name, age in self.thresholds.items():
            if not math.isfinite(age):
                continue
            first = max(installed + age, after)
            end = first + reach
            times = self.kind_times.upto(end)[name]
            ok = times >= first - tolerance
            if strict:
                ok &= times > after + tolerance
            if ok.any():
                t = float(times[np.argmax(ok)])
                if t < best[0]:
                    best = (t, self.prices[name], name)
        return best

    def _reach(self) -> float:
        """How far past a threshold the next shut of a kind is looked for:
        the part's clock over the span repeating is judged over (28
        years), or with no dates, ten of the longest period."""
        layout = self.layout
        if layout.now is not None:
            return layout.span_model()
        periods = [cast(Any, k.pattern()).period for k in layout.kinds]
        return 10.0 * max(periods, default=1.0)

    def price_at(self, t: float) -> float:
        price = math.inf
        if self.layout.site.any_day is not None:
            price = float(self.layout.site.any_day)
        tolerance = 1e-9 * max(1.0, abs(t))
        for name, times in self.kind_times.upto(t + 1.0).items():
            if np.any(np.abs(times - t) <= tolerance):
                price = min(price, self.prices[name])
        return price

    def grid_policy(self, end: float):
        rule = dict(self.thresholds)
        if self.any_age is not None:
            rule[ANYTIME] = self.any_age
        return _dated_grid_policy(self.layout, rule, end)


class _InnerEngine:
    """The exact routes' engine: a ``CalendarPolicy`` over the shuts laid
    on the part's clock as a repeating calendar, its kinds' names mapped
    back."""

    def __init__(self, inner, layout: _Layout):
        self.inner = inner
        self.layout = layout
        self.step = inner.step

    def next_planned(self, installed, after):
        time, price, kind = self.inner._next_planned(installed, after)
        return time, price, _base(kind)

    def price_at(self, t):
        return self.inner._planned_price(t)

    def grid_policy(self, end: float):
        if self.inner.step is None:
            raise NotImplementedError(
                "A policy worked out by quadrature (failure_timing='exact') "
                "has no grid for a cost's spread: work it out with "
                "failure_timing='half_step', or use method='simulate'."
            )
        return self.inner


def _base(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    if name == ANYTIME:
        return ANY_DAY
    return name.split(_PART)[0]


# ------------------------------------------------------------- the routes


def _expanded(layout: _Layout, patterns) -> Calendar:
    """The patterns as a repeating ``Calendar`` on the part's clock: a
    kind with several shuts a cycle as an opportunity for each."""
    opportunities = []
    for kind in layout.kinds:
        pattern = patterns[kind.name]
        offsets = pattern.offsets
        for i, offset in enumerate(offsets):
            name = kind.name if len(offsets) == 1 else f"{kind.name}{_PART}{i}"
            opportunities.append(
                Opportunity(
                    name, kind.price, period=pattern.period, offset=offset
                )
            )
    return Calendar(opportunities, anytime=layout.site.any_day)


def _merged(values: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name, value in values.items():
        key = _base(name) or name
        out[key] = out.get(key, 0.0) + float(value)
    return out


def _exact(layout: _Layout, decision: _Decision, mc_samples, seed):
    """The exact, long-cycle and sampling routes: the inner policy."""
    from repyability.maintenance import _replacement as R
    from repyability.maintenance._integration import integrated_policy

    part = layout.part
    calendar = _expanded(layout, decision.patterns)
    failure, anytime = calendar._prices(part)
    life = R._saved_life(part)
    if decision.failure_timing == "exact" and decision.route != "sampling":
        _warn_slow(decision)
        try:
            return integrated_policy(part, calendar, failure, life=life)
        except NotImplementedError as error:
            raise ValueError(
                f"failure_timing='exact' is refused here ({error}): use "
                "failure_timing='half_step' (the default), which gives the "
                "same to about 1e-7."
            ) from None
    step = R._resolve_step(None, calendar, part)
    ages = R._Ages(part, step, None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        grid = R._Grid(calendar, anytime, step)
    if decision.route == "sampling":
        if decision.failure_timing == "exact":
            from repyability.maintenance._horizon import sampled_quadrature

            return sampled_quadrature(
                layout, decision, calendar, failure, mc_samples, seed, life
            )
        settle = R._settle(grid, ages)
        if (
            mc_samples is None
            or not checks.is_whole(mc_samples)
            or int(mc_samples) < 2
        ):
            raise ValueError(
                "mc_samples must be a whole number, at least 2; got "
                f"{mc_samples!r}."
            )
        return R._sampled_policy(
            grid,
            ages,
            failure,
            settle,
            int(mc_samples),
            seed,
            grid.names,
            life,
        )
    if decision.route == "exact_long_cycle":
        warnings.warn(
            f"The shuts repeat only after a long cycle: working it out "
            f"exactly takes {_seconds_text(decision.seconds or 0.0)}.",
            stacklevel=outside_level(),
        )
    return R._cycle_policy(grid, ages, failure, grid.names, life)


def _warn_slow(decision: _Decision) -> None:
    if decision.seconds is not None and decision.seconds > SLOW_SECONDS:
        warnings.warn(
            "failure_timing='exact' works this out by quadrature, which "
            "takes "
            f"{_seconds_text(decision.seconds)}; failure_timing='half_step' "
            "(the default) gives the same to about 1e-7, faster.",
            SlowSolverWarning,
            stacklevel=outside_level(),
        )


def _inner_thresholds(inner, names_order) -> Dict[str, Tuple[float, float]]:
    """Each kind's youngest and oldest threshold age over the inner
    policy's cycle (or the window it was worked out over)."""
    found: Dict[str, List[float]] = {n: [] for n in names_order}
    table = inner._table
    if inner.step is None:
        thresholds, kinds = table.thresholds, table.kinds
        for t, k in zip(thresholds, kinds):
            found.setdefault(table.names[k].split(_PART)[0], []).append(
                float(t)
            )
    else:
        n = int(getattr(table, "horizon", 1))
        if hasattr(table, "first") and getattr(table, "first").size >= 1:
            n = max(n, int(table.first.size))
        first, kinds = table.at(np.arange(n))
        for f, k in zip(first, kinds):
            if k < 0:
                continue
            name = str(_base(inner._names[k]))
            found.setdefault(name, []).append(
                float(f) * inner.step if f >= 0 else math.inf
            )
    out = {}
    for name, values in found.items():
        finite = [v for v in values if math.isfinite(v)]
        out[name] = (
            (min(finite), max(finite)) if finite else (math.inf, math.inf)
        )
    return out


def _a(name: str) -> str:
    return ("an " if name[:1].lower() in "aeiou" else "a ") + name


def _rule_text(
    thresholds, unit, any_day_allowed, varies="by where it falls in the cycle"
) -> str:
    """The rule in words: the kinds that replace, youngest age first, then
    those that never do."""
    unit_words = f" {unit}" if unit else ""
    used = []
    never = []
    for name, (low, high) in thresholds.items():
        if name == ANY_DAY:
            continue
        if not math.isfinite(low):
            never.append(name)
        elif abs(high - low) <= 1e-9 * max(1.0, high):
            used.append(
                (
                    low,
                    f"at {_a(name)} if older than {_number(low)}{unit_words}",
                )
            )
        else:
            used.append(
                (
                    low,
                    f"at {_a(name)} if older than {_number(low)} to "
                    f"{_number(high)}{unit_words} ({varies})",
                )
            )
    low = thresholds.get(ANY_DAY, (math.inf, math.inf))[0]
    if any_day_allowed and math.isfinite(low):
        used.append(
            (
                low,
                f"on any other day once older than {_number(low)}{unit_words}",
            )
        )
    parts = [text for _, text in sorted(used, key=lambda x: x[0])]
    nots = [f"at {_a(n)}" for n in never]
    if not (any_day_allowed and math.isfinite(low)):
        nots.append(
            "on any other day"
            if any_day_allowed
            else "on any other day (the site allows none)"
        )
    if not parts:
        return "Run to failure: never " + " or ".join(nots) + "."
    text = "Replace " + "; ".join(parts)
    if nots:
        text += "; never " + " or ".join(nots)
    return text + "."


# -------------------------------------------------------- averaged route


class _Averaged:
    """The renewal-reward formula with each kind's shuts spread evenly
    within its usual gaps (see the module's notes).

    With thresholds ``a_k`` a kind and ``a_0`` for any other day, the
    planned replacement comes at ``P = min(a_0, min_k a_k + U_k)``, each
    ``U_k`` the equilibrium residual of kind ``k``'s gaps (density
    ``Pr(G_k > x) / E[G_k]``), independent; the part at ``min(X, P)``.
    With ``Q(t) = Pr(P > t)`` and ``R`` the part's survival, a cycle
    lasts ``∫ R Q`` on average, kind ``k`` replaces with
    ``∫ R q_k`` (``q_k`` the density of ``P`` by ``k``), any other day
    with ``R(a_0) Q(a_0)``, and a failure the rest. Each integral is of
    ``R`` times a piecewise polynomial, by Gauss-Legendre on the pieces.
    """

    def __init__(self, part, kinds, any_day, failure):
        from repyability.maintenance._replacement import (
            _age_where,
            _mean,
            _survival,
        )

        self.sf = _survival(part)
        self.mean = _mean(part, self.sf)
        self.far = _age_where(self.sf, _TAIL, self.mean)
        self.kinds = kinds  # [(name, price, lengths, counts)]
        self.any_day = any_day
        self.failure = float(failure)
        self.nodes, self.weights = np.polynomial.legendre.leggauss(_NODES)

    def _survive(self, k, x):
        """``1 - H_k(x)``: the chance kind ``k``'s next shut is more than
        ``x`` after the part reaches its threshold."""
        _, _, lengths, counts = self.kinds[k]
        x = np.maximum(x, 0.0)
        total = float(np.dot(counts, lengths))
        used = (np.minimum(x[:, None], lengths[None, :]) * counts).sum(axis=1)
        return np.clip(1.0 - used / total, 0.0, 1.0)

    def _density(self, k, x):
        _, _, lengths, counts = self.kinds[k]
        total = float(np.dot(counts, lengths))
        inside = (x[:, None] >= 0.0) & (x[:, None] < lengths[None, :])
        return (inside * counts).sum(axis=1) / total

    def evaluate(self, a, a0):
        """The cycle's mean length, the chance each kind replaces, of any
        other day, and of a failure."""
        a = np.asarray(a, float)
        end = self.far
        if a0 is not None and math.isfinite(a0):
            end = min(end, a0)
        ends = [end]
        breaks = [0.0, end]
        for k, (_, _, lengths, _) in enumerate(self.kinds):
            if math.isfinite(a[k]):
                ends.append(a[k] + float(lengths.max()))
                breaks.append(a[k])
                breaks.extend(a[k] + lengths)
        end = min(ends)
        points = np.unique(np.clip(np.asarray(breaks), 0.0, end))
        points = points[points <= end]
        if points.size < 2:
            points = np.array([0.0, end])
        low, high = points[:-1], points[1:]
        half = 0.5 * (high - low)
        t = (0.5 * (high + low))[:, None] + half[:, None] * self.nodes[None, :]
        w = (half[:, None] * self.weights[None, :]).ravel()
        t = t.ravel()
        r = self.sf(t)
        survive = []
        for k in range(len(self.kinds)):
            if math.isfinite(a[k]):
                survive.append(self._survive(k, t - a[k]))
            else:
                survive.append(np.ones_like(t))
        q = np.prod(survive, axis=0) if survive else np.ones_like(t)
        length = float(np.dot(w, r * q))
        by_kind = []
        for k in range(len(self.kinds)):
            if not math.isfinite(a[k]):
                by_kind.append(0.0)
                continue
            others = np.ones_like(t)
            for j in range(len(self.kinds)):
                if j != k:
                    others = others * survive[j]
            density = self._density(k, t - a[k])
            by_kind.append(float(np.dot(w, r * density * others)))
        anyday = 0.0
        if a0 is not None and math.isfinite(a0) and a0 <= end * (1 + 1e-12):
            tail = np.array([a0])
            held = float(self.sf(tail)[0])
            rest = 1.0
            for k in range(len(self.kinds)):
                if math.isfinite(a[k]):
                    rest *= float(self._survive(k, tail - a[k])[0])
            anyday = held * rest
        planned = sum(by_kind) + anyday
        if not math.isfinite(end) or end >= self.far:
            length += 0.0
        return length, np.array(by_kind), anyday, max(0.0, 1.0 - planned)

    def rate(self, a, a0) -> float:
        length, by_kind, anyday, failed = self.evaluate(a, a0)
        cost = self.failure * failed
        cost += sum(p * self.kinds[k][1] for k, p in enumerate(by_kind))
        if anyday:
            cost += anyday * float(self.any_day)
        return cost / length

    def optimise(self) -> Tuple[np.ndarray, Optional[float], float]:
        """The thresholds that cost least: each in turn over a scan of
        ages and then Brent's method, against never, until none moves;
        then all together by Nelder-Mead."""
        K = len(self.kinds)
        upper = self.far
        scan = np.linspace(0.0, upper, 97)[1:]
        a = np.full(K, math.inf)
        a0 = math.inf if self.any_day is not None else None
        names = list(range(K)) + (["any"] if self.any_day is not None else [])

        def value(which, x):
            b, b0 = a.copy(), a0
            if which == "any":
                b0 = x
            else:
                b[which] = x
            return self.rate(b, b0)

        best = self.rate(a, a0)
        for _ in range(12):
            moved = False
            for which in names:
                current = a0 if which == "any" else a[which]
                values = [value(which, x) for x in scan]
                i = int(np.argmin(values))
                lo = scan[max(i - 1, 0)] if i > 0 else 0.0
                hi = scan[min(i + 1, scan.size - 1)]
                found = minimize_scalar(
                    lambda x: value(which, x),
                    bounds=(lo, hi),
                    method="bounded",
                    options={"xatol": 1e-10 * upper},
                )
                x, v = (float(found.x), float(found.fun))
                if values[i] < v:
                    x, v = float(scan[i]), float(values[i])
                never = value(which, math.inf)
                if never <= v:
                    x, v = math.inf, never
                if v < best * (1 - 1e-13):
                    if which == "any":
                        a0 = x
                    else:
                        a[which] = x
                    best = v
                    moved = moved or current != x
            if not moved:
                break
        finite = [i for i in range(K) if math.isfinite(a[i])]
        use_any = a0 is not None and math.isfinite(a0)
        start = [a[i] for i in finite] + ([a0] if use_any else [])
        if len(start) > 1:

            def joint(x):
                b, b0 = a.copy(), a0
                for j, i in enumerate(finite):
                    b[i] = max(x[j], 0.0)
                if use_any:
                    b0 = max(x[-1], 0.0)
                return self.rate(b, b0)

            found = minimize(
                joint,
                np.asarray(start),
                method="Nelder-Mead",
                options={
                    "xatol": 1e-9 * upper,
                    "fatol": 1e-15,
                    "maxiter": 4000,
                },
            )
            if found.fun < best:
                for j, i in enumerate(finite):
                    a[i] = max(float(found.x[j]), 0.0)
                if use_any:
                    a0 = max(float(found.x[-1]), 0.0)
                best = float(found.fun)
        return a, a0, best


def _averaged_model(layout: _Layout) -> _Averaged:
    part = layout.part
    kinds = []
    for kind in layout.kinds:
        lengths, counts, _ = kind.gaps()
        kinds.append(
            (kind.name, kind.price, np.asarray(lengths), np.asarray(counts))
        )
    part._require_costs() if False else None
    if not hasattr(part, "cu"):
        raise ValueError(
            "A failure costs the part's own unplanned_cost, which this part "
            "has none of: give it one (NonRepairable(..., unplanned_cost=))."
        )
    return _Averaged(part, kinds, layout.site.any_day, part.cu)


def _age_replacement(part, price: float) -> float:
    """The best age to replace the part on any day at ``price``."""
    model = _Averaged(part, [], price, part.cu)
    _, a0, _ = model.optimise()
    return math.inf if a0 is None else float(a0)


# ------------------------------------------------------------ grid helpers


def _dated_calendar(layout: _Layout, end: float) -> Calendar:
    times = _KindTimes(layout).upto(end)
    opportunities = [
        Opportunity(k.name, k.price, dates=tuple(times[k.name]))
        for k in layout.kinds
        if times[k.name].size
    ]
    return Calendar(opportunities, anytime=layout.site.any_day)


class _FixedTable:
    """Fixed thresholds by grid index (from 0, none past the end)."""

    def __init__(self, first, kinds):
        self.first, self.kinds = np.asarray(first), np.asarray(kinds)
        self.horizon = max(int(self.first.size), 1)

    def at(self, index):
        index = np.asarray(index, np.int64)
        inside = (index >= 0) & (index < self.first.size)
        first = np.full(index.shape, -1, np.int64)
        kinds = np.full(index.shape, -1, np.int64)
        first[inside] = self.first[index[inside]]
        kinds[inside] = self.kinds[index[inside]]
        return first, kinds

    def chances(self, n):
        return np.zeros(n)


def _dated_grid_policy(layout: _Layout, rule: Dict[str, float], end: float):
    """A grid ``CalendarPolicy`` of a fixed rule over the real shuts to
    model time ``end`` (for a cost's spread, and diagrams)."""
    from repyability.maintenance import _replacement as R

    part = layout.part
    sf = R._survival(part)
    mean = R._mean(part, sf)
    step = mean / R.STEPS_PER_LIFE
    calendar = _dated_calendar(layout, end)
    failure, anytime = calendar._prices(part)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        grid = R._Grid(calendar, anytime, step)
    n = int(math.ceil(end / step)) + 1
    costs, first = R._rule_prices(
        grid, {k: v for k, v in rule.items() if math.isfinite(v)}, 0, n
    )
    _, kinds = grid.prices(0, n)
    names = grid.names
    # The kind replacing at each step is the rule's (the youngest age).
    kind_index = np.full(n, -1, np.int64)
    for k, name in enumerate(names):
        if name not in rule or not math.isfinite(rule[name]):
            continue
    index = {name: k for k, name in enumerate(names)}
    low, high = -0.5 * step, (n - 0.5) * step
    best = np.full(n, math.inf)
    for o in calendar.opportunities:
        if o.name not in rule or not math.isfinite(rule[o.name]):
            continue
        at = np.rint(o.times(low - step, high + step) / step).astype(np.int64)
        at = at[(at >= 0) & (at < n)]
        better = rule[o.name] < best[at]
        best[at[better]] = rule[o.name]
        kind_index[at[better]] = index[o.name]
    if (
        ANYTIME in rule
        and math.isfinite(rule[ANYTIME])
        and anytime is not None
    ):
        free = rule[ANYTIME] < best
        kind_index[free] = index[ANYTIME]
    first = np.where(kind_index >= 0, first, -1)
    ages = R._Ages(part, step, None)
    policy = R.CalendarPolicy(
        cost_rate=math.nan,
        cost_rate_se=None,
        replacement_rates={},
        cost_rates={},
        method="dp",
        basis="fixed",
        step=step,
        period=None,
        determined_until=None,
        threshold_only=True,
        settings={"max_age": ages.size * step},
        life=R._saved_life(part),
    )
    return policy._bind(
        _FixedTable(first, kind_index), names, grid, ages.size * step
    )


# --------------------------------------------------------- building results


def _measure(layout: _Layout) -> Tuple[Optional[float], str]:
    pace = layout.pace()
    if pace is None:
        unit = layout.unit or "unit of the part's clock"
        return None, f"cost per {unit}"
    return pace, "cost per year"


def _figure(measure: str, cost: float, se: Optional[float]) -> str:
    if measure == "cost per year":
        money = _money(round(cost, 2)) if cost >= 0.01 else f"${cost:.3g}"
        text = f"{money} a year"
    else:
        text = f"{cost:.4g} a {measure.split('per ')[-1]}"
    if se is not None:
        text += f" ± {se:.2g}"
    return text


def find_policy(
    part,
    site,
    *,
    plan=None,
    now=None,
    readings=None,
    age=0.0,
    unit=None,
    method=None,
    horizon=None,
    discount=None,
    failure_timing="half_step",
    mc_samples=64,
    seed=None,
) -> ShutdownPolicy:
    """The work of ``NonRepairable.optimal_calendar_policy`` with a site
    (see there)."""
    from repyability.maintenance._replacement import _saved_life

    if not hasattr(part, "cu"):
        raise ValueError(
            "A failure costs the part's own unplanned_cost, which this part "
            "has none of: give it one (NonRepairable(..., unplanned_cost=))."
        )
    layout = _Layout(
        part, site, plan=plan, now=now, readings=readings, unit=unit
    )
    age = quantity(age, layout.unit, "age")
    if not 0.0 <= age < math.inf:
        raise ValueError(f"age must be at least 0; got {age!r}.")
    decision = _decide(
        layout,
        method=method,
        horizon=horizon,
        discount=discount,
        failure_timing=failure_timing,
    )
    pace, measure = _measure(layout)
    notes = list(layout.notes)
    life = _saved_life(part)
    if decision.route == "horizon":
        from repyability.maintenance._horizon import horizon_policy

        return horizon_policy(layout, decision, age, life)
    names = list(site.names)
    if (
        decision.route in ("exact", "exact_long_cycle", "sampling")
        and site.shutdowns
    ):
        inner = _exact(layout, decision, mc_samples, seed)
        engine: Any = _InnerEngine(inner, layout)
        rate = float(inner.cost_rate)
        se = inner.cost_rate_se
        rates = _merged(inner.replacement_rates)
        thresholds = _inner_thresholds(inner, names)
        if ANY_DAY in thresholds and site.any_day is None:
            thresholds.pop(ANY_DAY)
        settings = dict(inner.settings)
        settings.update(
            method=inner.method, basis=inner.basis, step=inner.step
        )
    else:
        model = _averaged_model(layout)
        a, a0, rate = model.optimise()
        length, by_kind, anyday, failed = model.evaluate(a, a0)
        thresholds = {
            name: (float(a[k]), float(a[k])) for k, name in enumerate(names)
        }
        if site.any_day is not None:
            thresholds[ANY_DAY] = (float(a0 or 0.0), float(a0 or 0.0))
        rates = {
            name: float(by_kind[k] / length) for k, name in enumerate(names)
        }
        if site.any_day is not None:
            rates[ANY_DAY] = anyday / length
        rates["failure"] = failed / length
        se = None
        engine = _ThresholdEngine(
            layout,
            {n: thresholds[n][0] for n in names},
            None if a0 is None else float(a0),
        )
        settings = {"nodes": _NODES, "mean_cycle": length}
    rates = {(ANY_DAY if k == ANYTIME else k): v for k, v in rates.items()}
    scale = pace if pace is not None else 1.0
    cost = rate * scale
    cost_se = None if se is None else se * scale
    by_kind = {k: v * scale for k, v in rates.items()}
    rule = ReplacementRule(
        _rule_text(thresholds, layout.unit, site.any_day is not None),
        thresholds,
        None,
        layout.unit,
    )
    statement = {
        "exact": "Exact long-run average",
        "exact_long_cycle": "Exact long-run average",
        "sampling": "Estimated long-run average",
        "averaged": (
            "Long-run average, assuming shuts fall evenly within their "
            "usual gaps"
        ),
    }[decision.route]
    figure = _figure(measure, cost, cost_se)
    if decision.route == "sampling":
        figure += f" ({int(mc_samples)} samples)"
    if decision.route == "averaged":
        text = (
            f"{statement}: {figure} ({decision.why}). Exact dates not used "
            "for the cost."
        )
    else:
        why = decision.why
        text = (
            f"{statement}: {figure}. {why[0].upper()}{why[1:]}; worked out "
            f"{decision.how}."
        )
    return ShutdownPolicy(
        route=decision.route,
        route_text=text,
        measure=measure,
        cost=float(cost),
        cost_se=cost_se,
        solver=decision.solver or "",
        failure_timing=decision.failure_timing,
        rule=rule,
        replacements_by_kind=by_kind,
        cost_rate=float(rate),
        unit=layout.unit,
        now=None if layout.now is None else layout.now.isoformat(),
        age=float(age),
        notes=tuple(notes),
        settings=settings,
        life=life,
    )._bind(engine, layout, decision)


# ------------------------------------------------------------ a given rule


def _named_rule(layout: _Layout, rule) -> Dict[str, float]:
    if not isinstance(rule, dict):
        raise TypeError(
            "rule must be a dict of the age from which each kind of shut (or "
            f"'{ANY_DAY}') replaces the part; got {rule!r}."
        )
    out = {}
    names = list(layout.site.names) + [ANY_DAY]
    for name, age in rule.items():
        key = ANY_DAY if name == ANYTIME else name
        if key not in names:
            raise ValueError(
                f"Unknown kind {name!r} in the rule: the site's are {names}."
            )
        value = quantity(age, layout.unit, f"The age for {name!r}")
        if not value >= 0.0:
            raise ValueError(f"The age for {name!r} must be at least 0.")
        out[key] = value
    if ANY_DAY in out and layout.site.any_day is None:
        raise ValueError(
            "The rule replaces on any other day, which the site does not "
            "allow "
            "(its any_day is None)."
        )
    return out


def _rule_cost(layout, decision, rule, name, settings) -> RuleCost:
    """A fixed rule priced by the decision's route."""
    pace, measure = _measure(layout)
    names = list(layout.site.names)
    if decision.route == "horizon":
        from repyability.maintenance._horizon import horizon_rule_cost

        return horizon_rule_cost(layout, decision, rule, name)
    if decision.route == "averaged" or not layout.site.shutdowns:
        model = _averaged_model(layout)
        a = np.array([rule.get(n, math.inf) for n in names])
        a0 = (
            rule.get(ANY_DAY, math.inf)
            if layout.site.any_day is not None
            else None
        )
        length, by_kind, anyday, failed = model.evaluate(a, a0)
        rate = model.rate(a, a0)
        rates = {n: float(by_kind[k] / length) for k, n in enumerate(names)}
        if layout.site.any_day is not None:
            rates[ANY_DAY] = anyday / length
        rates["failure"] = failed / length
        se = None
        solver, timing = "closed form", None
    else:
        rate, se, rates = _grid_rule_rate(layout, decision, rule, settings)
        solver, timing = decision.solver, decision.failure_timing
    scale = pace if pace is not None else 1.0
    scaled_se = None if se is None else se * scale
    return RuleCost(
        name=name,
        rule=dict(rule),
        cost=float(rate * scale),
        measure=measure,
        cost_se=scaled_se,
        route=decision.route,
        route_text=f"{name}: {_figure(measure, rate * scale, scaled_se)}",
        solver=solver or "",
        failure_timing=timing,
        replacements_by_kind={k: v * scale for k, v in rates.items()},
        cost_rate=float(rate),
    )


def _grid_rule_rate(layout, decision, rule, settings):
    """A fixed rule's long-run rate on the exact routes: relative value
    iteration with the decisions fixed over the cycle (quadrature with
    them fixed for ``failure_timing="exact"``), or sampled windows."""
    from repyability.maintenance import _replacement as R

    part = layout.part
    calendar = _expanded(layout, decision.patterns)
    failure, anytime = calendar._prices(part)
    expanded = {}
    for o in calendar.opportunities:
        base = _base(o.name)
        if base in rule:
            expanded[o.name] = rule[base]
    if ANY_DAY in rule:
        expanded[ANYTIME] = rule[ANY_DAY]
    expanded = {k: v for k, v in expanded.items() if math.isfinite(v)}
    if decision.failure_timing == "exact":
        from repyability.maintenance._integration import integrated_policy

        inner = integrated_policy(
            part, calendar, failure, life=None, fixed=expanded
        )
        return inner.cost_rate, None, _merged(inner.replacement_rates)
    step = R._resolve_step(None, calendar, part)
    ages = R._Ages(part, step, None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        grid = R._Grid(calendar, anytime, step)
    if decision.route == "sampling":
        return _sampled_rule_rate(grid, ages, failure, expanded, settings)
    cycle_steps = cast(int, grid.cycle)
    costs, first = R._rule_prices(grid, expanded, 0, cycle_steps)
    cycle = R._relative_values(costs, ages, failure, fixed=first)
    replaced, failures = R._stationary(first, ages)
    _, kinds = grid.prices(0, cycle_steps)
    # The kind replacing at each step is the rule's.
    kinds = _rule_kinds(grid, expanded, cycle_steps)
    rates, _ = R._summary(
        grid.names,
        anytime,
        kinds,
        costs,
        replaced,
        failures,
        failure,
        cycle_steps * step,
    )
    return cycle.rate, None, _merged(rates)


def _rule_kinds(grid, rule, n):
    """The kind (index in the grid's names) a fixed rule replaces with at
    each step: the youngest age, the cheapest at a tie."""
    step = grid.step
    kinds = np.full(n, -1, np.int64)
    best = np.full(n, math.inf)
    price = np.full(n, math.inf)
    index = {name: k for k, name in enumerate(grid.names)}
    if ANYTIME in rule and grid.anytime is not None:
        best[:] = rule[ANYTIME]
        price[:] = grid.anytime
        kinds[:] = index[ANYTIME]
    for o in grid.calendar.opportunities:
        if o.name not in rule:
            continue
        at = np.rint(
            grid._dates(o, -1.5 * step, (n + 0.5) * step) / step
        ).astype(np.int64)
        at = at[(at >= 0) & (at < n)]
        age = rule[o.name]
        better = (age < best[at]) | ((age == best[at]) & (o.cost < price[at]))
        chosen = at[better]
        best[chosen], price[chosen], kinds[chosen] = age, o.cost, index[o.name]
    return kinds


def _sampled_rule_rate(grid, ages, failure, rule, settings):
    """A fixed rule's rate over a long cycle, sampled as the policy's is:
    windows from uniformly sampled times, each solved with the decisions
    fixed, the rate differenced over each."""
    from repyability.maintenance import _replacement as R

    samples = int(settings.get("mc_samples") or 64)
    rng = np.random.default_rng(checks.seed(settings.get("seed")))
    settle = R._settle(grid, ages)
    span = int(
        math.ceil(
            max([o.period for o in grid.calendar.opportunities] + [ages.mean])
            / grid.step
        )
    )
    cycle = grid.cycle if grid.cycle is not None else 2**62
    values = np.empty(samples)
    for i in range(samples):
        start = int(rng.integers(0, cycle)) - settle
        n = settle + span + settle
        costs, first = R._rule_prices(grid, rule, start, n)
        record = np.empty(n)
        R._sweep(
            np.zeros(ages.size + 1),
            costs,
            ages.p,
            failure,
            fixed=first,
            record=record,
        )
        values[i] = R._differenced(record, settle, settle + span, grid.step)
    return (
        float(values.mean()),
        float(values.std(ddof=1) / math.sqrt(samples)),
        {},
    )


def price_rule(
    part,
    site,
    rule,
    *,
    plan=None,
    now=None,
    readings=None,
    unit=None,
    method=None,
    horizon=None,
    discount=None,
    failure_timing="half_step",
    mc_samples=64,
    seed=None,
) -> RuleCost:
    """The work of ``NonRepairable.calendar_policy_cost`` with a site."""
    if not hasattr(part, "cu"):
        raise ValueError(
            "A failure costs the part's own unplanned_cost, which this part "
            "has none of: give it one (NonRepairable(..., unplanned_cost=))."
        )
    layout = _Layout(
        part, site, plan=plan, now=now, readings=readings, unit=unit
    )
    decision = _decide(
        layout,
        method=method,
        horizon=horizon,
        discount=discount,
        failure_timing=failure_timing,
    )
    named = _named_rule(layout, rule)
    words = (
        ", ".join(f"{k} from {_number(v)}" for k, v in named.items())
        or "run to failure"
    )
    return _rule_cost(
        layout,
        decision,
        named,
        words,
        {"mc_samples": mc_samples, "seed": seed},
    )


# ------------------------------------------------ the part in service now


@dataclass(frozen=True)
class DatedChoice(_ResultMapping):
    """One choice of when to replace the part in service, on the real
    dates: an entry of
    [`NextReplacementOnDates`][repyability.NextReplacementOnDates].

    Attributes
    ----------
    kind : str
        ``"now"``, ``"shut"``, ``"any other day"`` (at its best time) or
        ``"failure"`` (run to failure).
    name : str or None
        The shut's name, for a shut.
    date : str or None
        The day (None at failure, or for a planned replacement that never
        beats running to failure).
    time : float
        When, in the part's clock from now.
    price : float
        What the replacement costs then (at failure, the failure price).
    cost : float
        The expected cost until the part is replaced, less the time it
        serves valued at ``cost_rate``: the lowest is the best.
    break_even_age : float or None
        The youngest age at which replacing now costs no more.
    """

    kind: str
    name: Optional[str]
    date: Optional[str]
    time: float
    price: float
    cost: float
    break_even_age: Optional[float]

    def __str__(self) -> str:
        day = _day_text(date.fromisoformat(self.date)) if self.date else None
        if self.kind == "now":
            return "Now"
        if self.kind == "failure":
            return "Run to failure"
        if self.kind == "shut":
            return f"At the {self.name}, {day}"
        if day is None:
            return "On any other day: never beats running to failure"
        return f"On any other day, {day}"


@dataclass(frozen=True)
class NextReplacementOnDates(_ResultMapping):
    """Replace the part in service now, at one of the next shuts, on any
    other day, or run it to failure: returned by
    [`NonRepairable.optimal_next_replacement`][repyability.NonRepairable.optimal_next_replacement]
    given a site.

    Attributes
    ----------
    best : str
        The cheapest choice in words: ``"At the annual shut, 2 Nov 2027"``.
    best_option : DatedChoice
        The cheapest choice.
    options : list of DatedChoice
        Every choice, by time.
    age : float
        The part's age now.
    cost_rate : float
        The value of a unit of the part's service (the policy's figure, a
        unit of the part's clock).
    valued_by : str
        Where that figure comes from, in words.
    """

    best: str
    best_option: DatedChoice
    options: List[DatedChoice]
    age: float
    cost_rate: float
    valued_by: str


def _baseline_rate(policy: ShutdownPolicy) -> Tuple[float, str]:
    """The policy's figure a unit of the part's clock, and its words."""
    layout = policy._layout
    pace = layout.pace() or 1.0
    if policy.route != "horizon":
        return float(cast(float, policy.cost_rate)), (
            f"the policy's long-run cost ({policy.route} route)"
        )
    discount = float(policy._decision.discount)
    end = policy._decision.horizon
    years = (end - layout.origin).total_seconds() / 3600.0 / (24 * 365.25)
    if discount > 0:
        yearly = policy.cost * discount / (1.0 - (1.0 + discount) ** -years)
    else:
        yearly = policy.cost / years
    return yearly / pace, (
        "the policy's net present cost spread evenly over the years to its "
        "horizon"
    )


def next_on_dates(
    part,
    age,
    site,
    *,
    plan=None,
    now=None,
    readings=None,
    unit=None,
    policy=None,
) -> NextReplacementOnDates:
    """The work of ``NonRepairable.optimal_next_replacement`` with a site:
    the next two shuts of each kind on the real dates, valued at the
    policy's figure beyond them (UC-12 on real dates)."""
    from repyability.maintenance._next import _beyond, optimal_next_replacement
    from repyability.rbd._model_utils import model_mean

    if policy is None:
        policy = find_policy(
            part, site, plan=plan, now=now, readings=readings, unit=unit
        )
    elif not isinstance(policy, ShutdownPolicy):
        raise TypeError(
            "With a site, policy must be its ShutdownPolicy "
            f"(optimal_calendar_policy(site, ...)); got {policy!r}."
        )
    layout = policy._layout
    for name, given, theirs in (
        ("site", site, layout.site),
        ("plan", plan, layout.plan),
    ):
        if given is not None and given is not theirs and given != theirs:
            raise ValueError(
                f"The {name} given is not the one the policy was worked out "
                "for: work the policy out for it."
            )
    if (
        now is not None
        and layout.now is not None
        and str(layout.now) != str(now)
    ):
        raise ValueError(
            f"now={now!r}, but the policy counts from {layout.now}: give the "
            "policy's day, or work the policy out from this one."
        )
    age = quantity(age, layout.unit, "age")
    rate, valued_by = _baseline_rate(policy)
    mean = model_mean(part.reliability)
    held = float(np.ravel(part.reliability_function(age))[0])
    if not held > 0.0:
        raise ValueError(
            f"The life gives no chance of a unit working at age {age:g}."
        )

    def lasting(u):
        return float(np.ravel(part.reliability_function(age + u))[0]) / held

    reach = _beyond(lasting, mean)
    outages = []
    names = []
    now_price = site.any_day
    for kind in layout.kinds:
        times = kind.times(reach)
        if times.size and times[0] <= 1e-9:
            now_price = (
                kind.price if now_price is None else min(now_price, kind.price)
            )
        later = times[times > 1e-9][:2]
        for t in later:
            outages.append((float(t), kind.price))
            names.append((float(t), kind.name))
    if now_price is None:
        now_price = float(part.cu)
    order = sorted(range(len(outages)), key=lambda i: outages[i][0])
    outages = [outages[i] for i in order]
    names = [names[i] for i in order]
    from scipy.integrate import IntegrationWarning

    with warnings.catch_warnings():
        # The run to failure's served time is an integral to infinity,
        # which quad flags on a long life (km) though its value holds.
        warnings.simplefilter("ignore", IntegrationWarning)
        found = optimal_next_replacement(
            part,
            age,
            now_cost=now_price,
            scheduled_outages=outages,
            planned=site.any_day,
            rate=rate,
        )
    options = []
    shut_index = 0
    for choice in found.choices:
        if choice.kind == "outage":
            t, name = names[shut_index]
            shut_index += 1
            kind = next(k for k in layout.kinds if k.name == name)
            moment = kind.moment_of(t)
            options.append(
                DatedChoice(
                    "shut",
                    name,
                    _iso(layout, moment),
                    t,
                    choice.price,
                    choice.cost,
                    choice.break_even_age,
                )
            )
        elif choice.kind == "now":
            options.append(
                DatedChoice(
                    "now",
                    None,
                    layout.now.isoformat() if layout.now else None,
                    0.0,
                    choice.price,
                    choice.cost,
                    None,
                )
            )
        elif choice.kind == "planned":
            moment = (
                layout.moment_at([choice.time])[0]
                if math.isfinite(choice.time)
                else None
            )
            options.append(
                DatedChoice(
                    ANY_DAY,
                    None,
                    _iso(layout, moment),
                    choice.time,
                    choice.price,
                    choice.cost,
                    choice.break_even_age,
                )
            )
        else:
            options.append(
                DatedChoice(
                    "failure",
                    None,
                    None,
                    math.inf,
                    choice.price,
                    choice.cost,
                    choice.break_even_age,
                )
            )
    best = min(options, key=lambda c: (c.cost, c.time))
    return NextReplacementOnDates(
        str(best), best, options, float(age), rate, valued_by
    )


def _iso(layout: _Layout, moment) -> Optional[str]:
    if moment is None:
        return None
    return layout.clock.local_naive(moment).date().isoformat()


# ------------------------------------------------------ costs on dates


def on_dates(
    calendar,
    policy,
    window,
    until,
    age,
    *,
    site,
    plan,
    now,
    readings,
    many=False,
):
    """A cost's spread on real dates: the grid policy, the window (or the
    windows' ends) in the part's clock, and the part's age, from a
    ``ShutdownPolicy`` given as ``calendar`` or ``policy``. Anything else
    passes through."""
    from repyability.maintenance._shutdowns import _iso_date

    if policy is not None:
        if calendar is not None:
            raise ValueError(
                "Give the policy as calendar= or policy=, not both."
            )
        calendar = policy
    if not isinstance(calendar, ShutdownPolicy):
        given = [
            k
            for k, v in (
                ("site", site),
                ("plan", plan),
                ("now", now),
                ("readings", readings),
                ("until", until),
            )
            if v is not None
        ]
        if given:
            raise ValueError(
                f"{', '.join(given)} go with a site's policy on real dates: "
                "give policy=, the ShutdownPolicy of "
                "optimal_calendar_policy(site, ...)."
            )
        return calendar, window, age
    layout = calendar._layout
    if site is not None and site != layout.site:
        raise ValueError(
            "The site given is not the one the policy was worked out for: "
            "work the policy out for it."
        )
    if plan is not None and plan is not layout.plan:
        raise ValueError(
            "The plan given is not the one the policy was worked out for."
        )
    if (
        now is not None
        and layout.now is not None
        and _iso_date(now, "now") != layout.now
    ):
        raise ValueError(
            f"now={now!r}, but the policy counts from {layout.now}."
        )
    if readings is not None and dict(readings) != layout.readings:
        raise ValueError(
            "The readings given are not the ones the policy was worked "
            "out with."
        )

    def model(value):
        if layout.now is None or checks.is_number(value):
            return float(value)
        day = _iso_date(value, "until")
        return float(
            layout.model_at([layout.clock.midnight(day + _ONE_DAY)])[0]
        )

    if many:
        if window is None:
            raise ValueError("Give the windows' ends (dates, or times).")
        ends = [model(e) for e in window]
        end = max(ends)
        window = ends
    else:
        if until is not None:
            if window is not None:
                raise ValueError("Give window or until, not both.")
            window = model(until)
        elif window is None:
            raise ValueError("Give until= (the last day) or window=.")
        else:
            window = model(window)
        end = window
    if age == 0.0:
        age = calendar.age
    grid = calendar._calendar_policy(end + (calendar.step or 0.0) * 2)
    return grid, window, age


_ONE_DAY = timedelta(days=1)
