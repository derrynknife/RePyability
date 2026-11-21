"""The horizon route of a policy at a site (UC-43 of this package's
``README.md``): the lowest net present cost on the real dates from now to
a horizon, at a yearly discount rate (0 for the plain total), with no
assumption about how the shuts repeat.

- **On a grid** (``failure_timing="half_step"``): the dynamic programme
  of ``_replacement._sweep`` backward from the horizon (nothing counts
  after it), over the real shuts laid on the grid of the part's clock,
  each step discounted by the calendar time it spans.
- **By quadrature** (``failure_timing="exact"``): ``_integration``'s
  gaps, backward over the real shuts, each gap discounted at its own
  rate (its calendar time over its length in the part's clock).

The decisions near the horizon depend on where it ends (nothing is worth
keeping a part for after it). The route says from when: the policy is
worked out again to a later horizon and the first decision that differs
is named.

The same backward quadrature, with no discount, solves the windows that
sample a long cycle for ``method="sampling"`` with
``failure_timing="exact"``.
"""

from __future__ import annotations

import math
import warnings
from datetime import timedelta
from typing import Any, Dict, List, Tuple, cast

import numpy as np

from repyability.maintenance._calendar import ANYTIME
from repyability.maintenance._shutdowns import (
    _day_text,
    _Decision,
    _Layout,
    _money,
)
from repyability.value._rates import continuous as _rate
from repyability.value._rates import equivalent as _equivalent

#: The horizon is moved on by at least this many years to see which
#: decisions depend on where it ends.
LATER_YEARS = 5.0


def _grid_setup(layout: _Layout, end: float):
    from repyability.maintenance import _replacement as R
    from repyability.maintenance._site_policy import _dated_calendar

    part = layout.part
    sf = R._survival(part)
    mean = R._mean(part, sf)
    step = mean / R.STEPS_PER_LIFE
    calendar = _dated_calendar(layout, end)
    failure, anytime = calendar._prices(part)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        grid = R._Grid(calendar, anytime, step)
    ages = R._Ages(part, step, None)
    return grid, ages, failure, step


def _betas(
    layout: _Layout, n: int, step: float, discount: float
) -> np.ndarray:
    """Each step's discount factor: the calendar time it spans, at the
    yearly rate."""
    years = layout.years_at(np.arange(n + 1) * step)
    return np.exp(-_rate(discount) * np.diff(years))


def backward(grid, ages, failure, n: int, beta, rule=None):
    """The backward sweep over the first ``n`` steps of ``grid`` with
    nothing counted after them (``_replacement._sweep`` from zero
    values), each step discounted by ``beta`` (a number, or one a step):
    the values at the first step by age, each step's first replacing age
    index and kind, the prices, and whether the replacing ages were
    thresholds. ``rule`` fixes the decisions (the age from which each
    kind replaces, by name). The horizon route's sweep, on a site's real
    dates and in model time alike."""
    from repyability.maintenance import _replacement as R
    from repyability.maintenance._site_policy import _rule_kinds

    if rule is None:
        costs, kinds = grid.prices(0, n)
        v, first, upper = R._sweep(
            np.zeros(ages.size + 1), costs, ages.p, failure, beta
        )
    else:
        costs, first = R._rule_prices(grid, rule, 0, n)
        kinds = _rule_kinds(grid, rule, n)
        v, first, upper = R._sweep(
            np.zeros(ages.size + 1), costs, ages.p, failure, beta, fixed=first
        )
    return v, first, kinds, costs, upper


def _grid_solve(layout: _Layout, end: float, discount: float, rule=None):
    """The backward sweep to model time ``end``: the values at the first
    step by age, each step's first replacing age index and kind, the
    prices, and the grid."""
    grid, ages, failure, step = _grid_setup(layout, end)
    n = max(int(math.ceil(end / step - 1e-9)), 1)
    beta = _betas(layout, n, step, discount)
    v, first, kinds, costs, _ = backward(grid, ages, failure, n, beta, rule)
    return v, first, kinds, costs, grid, ages, failure, step


def _value_at(values: np.ndarray, age: float, step: float) -> float:
    return float(np.interp(age / step, np.arange(values.size), values))


def _counts(first, kinds, ages, age, step, names, failure_name="failure"):
    """Expected replacements by kind and failures over the steps, from a
    part ``age`` old."""
    from repyability.maintenance import _replacement as R

    share = np.zeros(ages.size + 1)
    share[min(int(round(age / step)), ages.size)] = 1.0
    _, replaced, failures = R._propagate(share, first, ages.p)
    out = {}
    for k, name in enumerate(names):
        out[name] = float(replaced[kinds == k].sum())
    out[failure_name] = float(failures)
    return out


def _depends_from(
    layout: _Layout, end: float, discount: float, first, kinds, step
):
    """The model time from which the grid's decisions change when the
    horizon is moved later (None if none does)."""
    later = _later_end(layout, end)
    _, first2, kinds2, *_ = _grid_solve(layout, later, discount)
    n = min(first.size, first2.size)
    differ = np.flatnonzero(
        (first[:n] != first2[:n]) | (kinds[:n] != kinds2[:n])
    )
    return None if differ.size == 0 else float(differ[0] * step)


def _later_end(layout: _Layout, end: float) -> float:
    from repyability.maintenance._shutdowns import _life_scale

    pace = layout.pace() or 1.0
    _, _, span = _life_scale(layout.part)
    return end + max(LATER_YEARS * pace, 3.0 * span)


def _by_year(layout, first, kinds, names, step, until):
    """Thresholds by calendar year from now: ``(first day, last day,
    {kind: (youngest, oldest)})``."""
    out = []
    n = first.size
    times = np.arange(n) * step
    years = layout.years_at(times)
    last = int(math.floor(float(years[-1]))) + 1 if n else 0
    start = layout.now
    for y in range(last):
        here = (years >= y) & (years < y + 1) & (times < until)
        if not here.any():
            continue
        found: Dict[str, Tuple[float, float]] = {}
        for k, name in enumerate(names):
            use = here & (kinds == k) & (first >= 0)
            if use.any():
                ages = first[use] * step
                found[_label(name)] = (float(ages.min()), float(ages.max()))
        a = start.replace(year=start.year + y)
        b = start.replace(year=start.year + y + 1) - timedelta(days=1)
        out.append((a.isoformat(), b.isoformat(), found))
    return out


def _label(name: str) -> str:
    from repyability.maintenance._site_policy import ANY_DAY

    return ANY_DAY if name == ANYTIME else name


def _thresholds_until(first, kinds, names, step, until, site_names, any_day):
    from repyability.maintenance._site_policy import ANY_DAY

    out = {}
    times = np.arange(first.size) * step
    for k, name in enumerate(names):
        label = _label(name)
        if label == ANY_DAY and any_day is None:
            continue
        use = (kinds == k) & (first >= 0) & (times < until)
        if use.any():
            ages = first[use] * step
            out[label] = (float(ages.min()), float(ages.max()))
        else:
            out[label] = (math.inf, math.inf)
    for name in site_names:
        out.setdefault(name, (math.inf, math.inf))
    return out


# ---------------------------------------------------------- quadrature


class _Shuts:
    """Backward quadrature over a finite run of shuts (times in the
    part's clock, prices, kinds) from ``start`` to ``end``: the values,
    each shut's threshold, and the counts forward."""

    def __init__(
        self,
        part,
        times,
        prices,
        kinds,
        start,
        end,
        failure,
        rhos=None,
        rule_ages=None,
    ):
        from repyability.maintenance._integration import _Gap, run_ages
        from repyability.maintenance._replacement import (
            FAR,
            _age_where,
            _mean,
            _survival,
        )

        self.sf = _survival(part)
        mean = _mean(part, self.sf)
        self.oldest = _age_where(self.sf, FAR, mean)
        self.grid = np.linspace(0.0, self.oldest, run_ages(mean, self.oldest))
        self.times = np.asarray(times, float)
        self.prices = np.asarray(prices, float)
        self.kinds = list(kinds)
        self.edges = np.concatenate([[start], self.times, [end]])
        self.gaps = np.diff(self.edges)
        self.rhos = (
            np.zeros(self.gaps.size) if rhos is None else np.asarray(rhos)
        )
        self.rule_ages = rule_ages
        self.failure = failure
        self._shapes: Dict[Tuple[float, float], object] = {}
        self._Gap = _Gap

    def _gap(self, width: float, rho: float):
        key = (round(width, 9), round(rho, 12))
        if key not in self._shapes:
            self._shapes[key] = self._Gap(
                width, self.sf, self.grid, self.failure, rho
            )
        return self._shapes[key]

    def solve(self):
        """Backward from ``end`` (nothing after it): the values at
        ``start`` by age; each shut's threshold, and the cost to go of a
        new part fitted at it (``new_values``)."""
        from repyability.maintenance._integration import threshold

        grid = self.grid
        after = np.zeros(grid.size)
        m = self.times.size
        self.thresholds = np.full(m, math.inf)
        self.new_values = np.zeros(m)
        for k in range(self.gaps.size - 1, -1, -1):
            width = self.gaps[k]
            if width > 0.0:
                shape = self._gap(width, float(self.rhos[k]))
                g_nodes = shape.new_unit(after, grid)
                w = shape.kept(g_nodes, after, grid)
            else:
                shape, g_nodes, w = None, None, after
            if k == 0:
                return w
            i = k - 1  # the shut at the start of this gap
            price = self.prices[i]
            if self.rule_ages is not None:
                limit = float(self.rule_ages[i])
                self.thresholds[i] = limit
                act = grid >= limit * (1 - 1e-12)
                after = np.where(act, price + w[0], w)
            else:
                if shape is not None:
                    self.thresholds[i], _ = threshold(
                        shape, g_nodes, after, w, price, grid
                    )
                else:
                    act = price + w[0] < w
                    self.thresholds[i] = (
                        float(grid[np.argmax(act)]) if act.any() else math.inf
                    )
                after = np.minimum(w, price + w[0])
            self.new_values[i] = float(w[0])
        return after

    def counts(self, age: float):
        """Expected replacements at each shut and failures over the run,
        from a part ``age`` old at ``start``."""
        grid = self.grid
        share = np.zeros(grid.size)
        share[int(np.argmin(np.abs(grid - age)))] = 1.0
        replaced = np.zeros(self.times.size)
        failures = 0.0
        for k in range(self.gaps.size):
            if k > 0:
                i = k - 1
                act = grid >= self.thresholds[i] * (1 - 1e-12)
                moved = float(share[act].sum())
                share = np.where(act, 0.0, share)
                share[0] += moved
                replaced[i] = moved
            if self.gaps[k] > 0.0:
                shape = self._gap(self.gaps[k], 0.0)
                share, failed = shape.forward(share)
                failures += failed
        return replaced, failures


# -------------------------------------------------------------- engines


class _ShutEngine:
    """A rule given shut by shut (the quadrature's), with no planned
    replacement on any other day: what ``next_replacement`` and a
    diagram ask."""

    step = None

    def __init__(self, times, thresholds, kinds, prices):
        self.times = np.asarray(times, float)
        self.thresholds = np.asarray(thresholds, float)
        self.kinds = list(kinds)
        self.prices = np.asarray(prices, float)

    def next_planned(self, installed, after):
        tolerance = 1e-9 * max(1.0, abs(after))
        later = (
            self.times > after + tolerance
            if after <= installed
            else self.times >= after - tolerance
        )
        due = later & (self.times - installed >= self.thresholds - tolerance)
        if not due.any():
            return math.inf, 0.0, None
        i = int(np.argmax(due))
        return float(self.times[i]), float(self.prices[i]), self.kinds[i]

    def price_at(self, t):
        hit = np.abs(self.times - t) <= 1e-9 * max(1.0, abs(t))
        return float(self.prices[hit].min()) if hit.any() else math.inf

    def grid_policy(self, end):
        raise NotImplementedError(
            "A policy worked out by quadrature (failure_timing='exact') has "
            "no grid for a cost's spread: work it out with "
            "failure_timing='half_step', or use method='simulate'."
        )


def _real_shuts(layout: _Layout, end: float):
    """The real shuts from now to ``end``, in order: times in the part's
    clock, prices and kinds (one shut a time, the cheapest)."""
    rows = []
    for kind in layout.kinds:
        for t in kind.times(end):
            if 0.0 <= t <= end:
                rows.append((float(t), kind.price, kind.name))
    rows.sort(key=lambda r: (r[0], r[1]))
    out: List[Tuple[float, float, str]] = []
    for row in rows:
        if out and abs(row[0] - out[-1][0]) <= 1e-9 * max(1.0, row[0]):
            continue
        out.append(row)
    return out


def _segment_rates(layout: _Layout, edges: np.ndarray, discount: float):
    """Each gap's continuous discount rate a unit of the part's clock:
    the calendar years it spans over its length, at the yearly rate."""
    if not discount:
        return np.zeros(edges.size - 1)
    years = layout.years_at(edges)
    widths = np.diff(edges)
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = np.where(
            widths > 0, _rate(discount) * np.diff(years) / widths, 0.0
        )
    return rho


def _quadrature_solve(layout: _Layout, end: float, discount: float, rule=None):
    from repyability.maintenance._site_policy import ANY_DAY

    rows = _real_shuts(layout, end)
    times = [r[0] for r in rows]
    prices = [r[1] for r in rows]
    kinds = [r[2] for r in rows]
    edges = np.concatenate([[0.0], times, [end]])
    rhos = _segment_rates(layout, edges, discount)
    limits = None
    if rule is not None:
        limits = np.array([rule.get(k, math.inf) for k in kinds])
        if ANY_DAY in rule:
            raise ValueError(
                "failure_timing='exact' prices no replacement on any other "
                "day."
            )
    shuts = _Shuts(
        layout.part,
        times,
        prices,
        kinds,
        0.0,
        end,
        layout.part.cu,
        rhos,
        limits,
    )
    values = shuts.solve()
    return shuts, values


# ---------------------------------------------------------------- routes


def horizon_policy(layout: _Layout, decision: _Decision, age: float, life):
    """The horizon route's policy (see the module's notes)."""
    from repyability.maintenance import _replacement as R
    from repyability.maintenance._site_policy import (
        ANY_DAY,
        ReplacementRule,
        ShutdownPolicy,
        _InnerEngine,
        _rule_text,
        _warn_slow,
    )

    end_moment: Any = decision.horizon
    end = float(layout.model_at([end_moment])[0])
    discount = float(cast(float, decision.discount))
    engine: Any
    settings: Dict[str, Any]
    years = (
        (end_moment - layout.origin).total_seconds() / 3600.0 / (24 * 365.25)
    )
    site = layout.site
    names = list(site.names)
    if decision.failure_timing == "exact":
        _warn_slow(decision)
        shuts, values = _quadrature_solve(layout, end, discount)
        npc = float(np.interp(age, shuts.grid, values))
        later_end = _later_end(layout, end)
        shuts2, _ = _quadrature_solve(layout, later_end, discount)
        m = shuts.times.size
        same = np.isclose(
            shuts.thresholds, shuts2.thresholds[:m], rtol=1e-6
        ) | (np.isinf(shuts.thresholds) & np.isinf(shuts2.thresholds[:m]))
        differ = np.flatnonzero(~same)
        depends = None if differ.size == 0 else float(shuts.times[differ[0]])
        replaced, failures = shuts.counts(age)
        counts = {n: 0.0 for n in names}
        for k, n in zip(shuts.kinds, replaced):
            counts[k] += float(n)
        counts["failure"] = float(failures)
        until = depends if depends is not None else end
        thresholds: Dict[str, Tuple[float, float]] = {}
        for name in names:
            use = [
                t
                for t, k, x in zip(shuts.thresholds, shuts.kinds, shuts.times)
                if k == name and x < until and math.isfinite(t)
            ]
            thresholds[name] = (
                (min(use), max(use)) if use else (math.inf, math.inf)
            )
        engine = _ShutEngine(
            shuts.times, shuts.thresholds, shuts.kinds, shuts.prices
        )
        by_period = None
        settings = {"nodes": 32, "ages": int(shuts.grid.size), "shuts": int(m)}
    else:
        v, first, kinds, costs, grid, ages, failure, step = _grid_solve(
            layout, end, discount
        )
        npc = _value_at(v, age, step)
        depends = _depends_from(layout, end, discount, first, kinds, step)
        until = depends if depends is not None else end
        counted = _counts(first, kinds, ages, age, step, grid.names)
        counts = {_label(k): x for k, x in counted.items()}
        if site.any_day is None:
            counts.pop(ANY_DAY, None)
        thresholds = _thresholds_until(
            first, kinds, grid.names, step, until, names, site.any_day
        )
        by_period = _by_year(layout, first, kinds, grid.names, step, end)
        policy = R.CalendarPolicy(
            cost_rate=math.nan,
            cost_rate_se=None,
            replacement_rates={},
            cost_rates={},
            method="dp",
            basis="horizon",
            step=step,
            period=None,
            determined_until=until,
            threshold_only=True,
            settings={"max_age": ages.size * step},
            life=life,
        )
        from repyability.maintenance._site_policy import _FixedTable

        policy = policy._bind(
            _FixedTable(first, kinds), grid.names, grid, ages.size * step
        )
        engine = _InnerEngine(policy, layout)
        settings = {"step": step, "steps": int(first.size)}
    per_year = {k: x / years for k, x in counts.items()}
    when = _day_text(layout.clock.local_naive(end_moment).date())
    eac = _equivalent(npc, _rate(discount), years)
    text = (
        f"Lowest net present cost to {when} at {discount:.0%} a year: "
        f"{_cents(npc)} (an equivalent annual cost of {_cents(eac)} a "
        "year)."
    )
    if depends is not None:
        moment = layout.moment_at([depends])[0]
        if moment is not None:
            day = layout.clock.local_naive(moment).date()
            text += (
                f" Decisions from about {_day_text(day)} on depend on where "
                "the horizon ends (nothing counts after it); those before do "
                "not."
            )
    else:
        text += " No decision depends on where the horizon ends."
    if decision.solver == "quadrature":
        text += (
            " Worked out by quadrature across each gap between the real "
            "shuts, failures timed at the instant they happen."
        )
    else:
        text += (
            " Worked out on the real dates on a grid of steps sized from the "
            "life, each failure's replacement counted half a step old."
        )
    rule = ReplacementRule(
        _rule_text(
            thresholds,
            layout.unit,
            site.any_day is not None,
            "by when it falls",
        ),
        thresholds,
        by_period,
        layout.unit,
    )
    settings.update(
        horizon=end_moment.isoformat(),
        discount=discount,
        decisions_depend_from=depends,
    )
    return ShutdownPolicy(
        route="horizon",
        route_text=text,
        measure="net present cost",
        cost=npc,
        cost_se=None,
        solver=decision.solver or "",
        failure_timing=decision.failure_timing,
        rule=rule,
        replacements_by_kind=per_year,
        cost_rate=None,
        unit=layout.unit,
        now=layout.now.isoformat() if layout.now else None,
        age=float(age),
        notes=tuple(layout.notes),
        settings=settings,
        life=life,
        eac=eac,
    )._bind(engine, layout, decision)


def _cents(value: float) -> str:
    return _money(round(value, 2)) if value >= 0.01 else f"${value:.3g}"


def horizon_rule_cost(layout: _Layout, decision: _Decision, rule, name):
    """A fixed rule's net present cost to the horizon, from a new part."""
    from repyability.maintenance._site_policy import ANY_DAY, RuleCost

    horizon: Any = decision.horizon
    end = float(layout.model_at([horizon])[0])
    discount = float(cast(float, decision.discount))
    years = (horizon - layout.origin).total_seconds() / 3600.0 / (24 * 365.25)
    names = list(layout.site.names)
    if decision.failure_timing == "exact":
        shuts, values = _quadrature_solve(layout, end, discount, rule)
        npc = float(values[0])
        replaced, failures = shuts.counts(0.0)
        counts = {n: 0.0 for n in names}
        for k, x in zip(shuts.kinds, replaced):
            counts[k] += float(x)
        counts["failure"] = float(failures)
    else:
        grid_rule = {
            (ANYTIME if k == ANY_DAY else k): v for k, v in rule.items()
        }
        grid_rule = {k: v for k, v in grid_rule.items() if math.isfinite(v)}
        v, first, kinds, costs, grid, ages, failure, step = _grid_solve(
            layout, end, discount, grid_rule
        )
        npc = float(v[0])
        counted = _counts(first, kinds, ages, 0.0, step, grid.names)
        counts = {_label(k): x for k, x in counted.items()}
        if layout.site.any_day is None:
            counts.pop(ANY_DAY, None)
    eac = _equivalent(npc, _rate(discount), years)
    return RuleCost(
        name=name,
        rule=dict(rule),
        cost=npc,
        measure="net present cost",
        cost_se=None,
        route="horizon",
        route_text=(
            f"{name}: net present cost {_money(round(npc, 2))} (an "
            f"equivalent annual cost of {_cents(eac)} a year)"
        ),
        solver=decision.solver or "",
        failure_timing=decision.failure_timing,
        replacements_by_kind={k: x / years for k, x in counts.items()},
        cost_rate=None,
        npv=npc,
        eac=eac,
    )


def sampled_quadrature(
    layout, decision, calendar, failure, mc_samples, seed, life
):
    """The long-run rate of a long cycle sampled by quadrature: windows
    from uniformly sampled times of the cycle, each solved backward over
    its shuts with nothing after it, the rate differenced between the
    cost to go of a new part at the first shut and at the first shut a
    life on."""
    from repyability.maintenance import _replacement as R
    from repyability.maintenance._integration import _OpportunityTable
    from repyability.maintenance._shutdowns import _life_scale
    from repyability.utils import checks

    if (
        mc_samples is None
        or not checks.is_whole(mc_samples)
        or int(mc_samples) < 2
    ):
        raise ValueError(
            "mc_samples must be a whole number, at least 2; got "
            f"{mc_samples!r}."
        )
    samples = int(mc_samples)
    part = layout.part
    mean, oldest, span = _life_scale(part)
    periods = [o.period for o in calendar.opportunities]
    settle = 2.0 * max(periods) + 3.0 * span
    width = max(max(periods), mean)
    cycle = float(decision.cycle)
    rng = np.random.default_rng(checks.seed(seed))
    names = [o.name for o in calendar.opportunities]

    def shuts_between(a, b):
        rows = []
        for o in calendar.opportunities:
            for t in o.times(a, b):
                rows.append((float(t), o.cost, o.name))
        rows.sort(key=lambda r: (r[0], r[1]))
        out: List[Tuple[float, float, str]] = []
        for row in rows:
            if out and abs(row[0] - out[-1][0]) <= 1e-9 * max(
                1.0, abs(row[0])
            ):
                continue
            out.append(row)
        return out

    rates = np.empty(samples)
    by_kind = {n: 0.0 for n in names}
    failures = 0.0
    for s in range(samples):
        start = float(rng.uniform(0.0, cycle))
        rows = shuts_between(start, start + width + settle)
        times = np.array([r[0] for r in rows])
        shuts = _Shuts(
            part,
            times,
            [r[1] for r in rows],
            [r[2] for r in rows],
            start,
            start + width + settle,
            failure,
        )
        shuts.solve()
        i1 = 0
        i2 = int(np.searchsorted(times, times[0] + width))
        rates[s] = (shuts.new_values[i1] - shuts.new_values[i2]) / (
            times[i2] - times[i1]
        )
        replaced, failed = shuts.counts(0.0)
        inside = times < times[0] + width
        for k, x, keep in zip(shuts.kinds, replaced, inside):
            if keep:
                by_kind[k] += float(x)
        failures += float(failed) * width / (width + settle)
    total = samples * width
    rates_by_kind = {n: v / total for n, v in by_kind.items()}
    rates_by_kind["failure"] = failures / total
    # The thresholds from now: a window from 0.
    rows = shuts_between(0.0, width + settle)
    shuts = _Shuts(
        part,
        [r[0] for r in rows],
        [r[1] for r in rows],
        [r[2] for r in rows],
        0.0,
        width + settle,
        failure,
    )
    shuts.solve()
    keep = shuts.times < width
    table = _OpportunityTable(
        10.0 * (width + settle),
        shuts.times[keep],
        shuts.thresholds[keep],
        np.array(
            [names.index(k) for k, x in zip(shuts.kinds, keep) if x], np.int64
        ),
        names + [ANYTIME],
        0.0,
        shuts.prices[keep],
    )
    policy = R.CalendarPolicy(
        cost_rate=float(rates.mean()),
        cost_rate_se=float(rates.std(ddof=1) / math.sqrt(samples)),
        replacement_rates=rates_by_kind,
        cost_rates={},
        method="sampling",
        basis="sampled",
        step=None,
        period=None,
        determined_until=None,
        threshold_only=True,
        settings={"mc_samples": samples, "window": width, "settle": settle},
        life=life,
    )
    return policy._bind(table, names + [ANYTIME], None, oldest)


# ------------------------------------------------------ a ready horizon

#: A ready horizon (what a refusal offers in its place): costs past it,
#: discounted, count for under this share of the total.
READY_SHARE = 1e-9
#: Undiscounted, a ready horizon is the full cycle and this many of the
#: part's life spans (the age it outlives with a chance of 1e-3) past it.
READY_SPANS = 3
#: ... but no more than this many steps of the grid (about a minute's
#: sweep), where the full cycle is longer (co-prime periods).
READY_STEPS = 2_000_000


def ready_years(discount: float) -> int:
    """The whole years past which costs at the yearly ``discount`` (more
    than 0) are discounted to under ``READY_SHARE``."""
    return int(
        math.ceil(math.log(1.0 / READY_SHARE) / math.log1p(float(discount)))
    )


def _rounded_up(value: float) -> float:
    """``value`` rounded up to three significant figures."""
    if not value > 0.0:
        return value
    scale = 10.0 ** (math.floor(math.log10(value)) - 2)
    return float(math.ceil(value / scale - 1e-9) * scale)


def _literal(value: float) -> str:
    """A model time as it would be typed: ``112000``, ``2.5``."""
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.6g}"


def ready_model_horizon(
    rate: float, span: float, cycle=None, end=None, step=None
):
    """A horizon in model time that works for a calendar a long run
    refuses: with a continuous discount ``rate`` a unit of time, where
    costs are discounted to under ``READY_SHARE``; with none, the full
    ``cycle`` and ``READY_SPANS`` life spans past it, but at most
    ``READY_STEPS`` of the grid's ``step``. At least a plan's last date
    (``end``) and the spans past it. Rounded up to three significant
    figures; with the reason in words."""
    if rate > 0.0:
        value = math.log(1.0 / READY_SHARE) / rate
        why = (
            "at this rate, costs past then are discounted to under one part "
            "in a billion"
        )
    else:
        value = (cycle or 0.0) + READY_SPANS * span
        why = (
            "the full cycle and three of the part's life spans past it"
            if cycle
            else "three of the part's life spans"
        )
        if step is not None and value > READY_STEPS * step:
            value = READY_STEPS * step
            why = (
                f"{READY_STEPS:,} steps of the grid, about a minute's work: "
                "the full cycle is longer than is worth sweeping"
            )
    if end is not None:
        value = max(value, float(end) + READY_SPANS * span)
    return _rounded_up(value), why


def ready_model_text(
    rate: float, span: float, cycle=None, end=None, step=None
) -> str:
    """``horizon=<value>`` and why, for a refusal in model time."""
    value, why = ready_model_horizon(rate, span, cycle, end, step)
    return f"horizon={_literal(value)} (in the calendar's time; {why})"


def _years_after(day, years: int):
    try:
        return day.replace(year=day.year + years)
    except ValueError:  # 29 February
        return day.replace(year=day.year + years, day=28)


def ready_site_text(
    layout: _Layout, discount, cycle_years=None, step=None
) -> str:
    """``horizon='<date>'`` with ``discount=``, and why, for a refusal at
    a site: with a yearly ``discount`` the day past which costs are
    discounted to under ``READY_SHARE``; with none, the full cycle of the
    shuts' dates (``cycle_years``, by default the 28 years after which
    weekdays fall on the same dates again) and ``READY_SPANS`` life spans
    past it."""
    from repyability.maintenance._shutdowns import (
        SPAN_YEARS,
        _life_scale,
        _percent,
    )

    rate = float(discount or 0.0)
    if rate > 0.0:
        years = ready_years(rate)
        why = (
            f"at {_percent(rate)} a year, costs past then are discounted to "
            "under one part in a billion"
        )
    else:
        cycle = float(cycle_years if cycle_years is not None else SPAN_YEARS)
        pace = layout.pace()
        spans = 0.0
        if pace:
            spans = READY_SPANS * _life_scale(layout.part)[2] / pace
        why = (
            "the full cycle of the shuts' dates and three of the part's "
            "life spans past it"
        )
        if (
            pace
            and step is not None
            and cycle + spans > (READY_STEPS * step / pace)
        ):
            cycle, spans = READY_STEPS * step / pace, 0.0
            why = (
                f"{READY_STEPS:,} steps of the grid, about a minute's work: "
                "the full cycle is longer than is worth sweeping"
            )
        years = int(math.ceil(cycle + spans - 1e-9))
    given = f"discount={rate:g}"
    if layout.now is None:
        return (
            f"now= and a horizon= {years} years after it, with {given} "
            f"({why})"
        )
    day = _years_after(layout.now, years)
    return f"horizon='{day.isoformat()}' with {given} ({why})"


# ----------------------------------------------------- in model time


def _model_solve(
    part, calendar, anytime, failure, step, max_age, end, rate, rule=None
):
    """The backward sweep in model time to ``end``, from a new part at 0,
    each step discounted by ``exp(-rate * step)`` (1 at 0): the given
    dates as they are (no tail), periods as they fall."""
    from repyability.maintenance import _replacement as R

    ages = R._Ages(part, step, max_age)
    grid = R._Grid(calendar, anytime, step, tail=None)
    n = max(int(math.ceil(end / step - 1e-9)), 1)
    beta = math.exp(-rate * step)
    v, first, kinds, costs, upper = backward(
        grid, ages, failure, n, beta, rule
    )
    return v, first, kinds, costs, upper, grid, ages, n


def model_horizon_policy(
    part, calendar, anytime, failure, step, max_age, horizon, rate, life
):
    """A policy in model time to a horizon: the lowest net present cost
    of a new part at 0 and its successors to ``horizon``, at the
    continuous ``rate`` a unit of time (0: the plain total), on every
    opportunity as it falls (given dates as they are, nothing assumed
    past them). Its ``npv`` is that cost and ``eac`` the steady cost a
    unit of time worth it; ``cost_rate`` and the rates by kind are the
    rule's undiscounted average to the horizon, from a new part.
    ``determined_until`` is the first time a decision changes when the
    horizon is moved later (a settling length on)."""
    from repyability.maintenance import _replacement as R
    from repyability.maintenance._site_policy import _FixedTable

    v, first, kinds, costs, upper, grid, ages, n = _model_solve(
        part, calendar, anytime, failure, step, max_age, horizon, rate
    )
    length = n * step
    npc = float(v[0])
    later = length + R._settle(grid, ages) * step
    _, first2, kinds2, *_ = _model_solve(
        part, calendar, anytime, failure, step, max_age, later, rate
    )
    differ = np.flatnonzero((first != first2[:n]) | (kinds != kinds2[:n]))
    until = float(differ[0] * step) if differ.size else length
    new = np.zeros(ages.size + 1)
    new[0] = 1.0
    _, replaced, failures = R._propagate(new, first, ages.p)
    rates, spent = R._summary(
        grid.names, anytime, kinds, costs, replaced, failures, failure, length
    )
    R._warn_not_threshold(upper)
    policy = R.CalendarPolicy(
        cost_rate=float(sum(spent.values())),
        cost_rate_se=None,
        replacement_rates=rates,
        cost_rates=spent,
        method="dp",
        basis="horizon",
        step=step,
        period=None,
        determined_until=until,
        threshold_only=upper,
        settings={
            "horizon": length,
            "steps": int(n),
            "table_until": length,
            "max_age": ages.size * step,
        },
        life=life,
        discount_rate=rate,
        npv=npc,
        eac=_equivalent(npc, rate, length),
    )
    table = _FixedTable(first, kinds, replaced)
    return policy._bind(table, grid.names, grid, ages.size * step)


def model_horizon_rule_cost(
    part, calendar, anytime, failure, step, max_age, rule, horizon, rate
) -> float:
    """A fixed rule's net present cost (the plain total at ``rate`` 0) in
    model time to ``horizon``, from a new part at 0."""
    v, *_ = _model_solve(
        part, calendar, anytime, failure, step, max_age, horizon, rate, rule
    )
    return float(v[0])
