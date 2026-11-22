"""The dated window of a policy at a site (UC-43 of this package's
``README.md``): a site with fixed dates (turnarounds on 3 March 2027 and
9 September 2028) is worked out on its real dates to the last of them,
then by the long run of its recurring shuts.

- **The window** runs from now to the last fixed date (or to
  ``dated_until``, a later day). Inside it every shut is on its real
  dates, the recurring kinds' as well as the fixed dates, in order,
  through the plan's clock: fixed dates and rules combine freely. No
  shut in it is averaged.
- **After it** only the recurring shuts remain (a kind given
  ``then_every`` among them, recurring from its last date at the gap
  given). Their long-run average cost a year, by the route those shuts
  take on their own (exact where they repeat, averaged or sampled where
  asked, refused with what to choose otherwise), is the figure that
  values everything after the window.

Undiscounted, the long run's cost a year adds the same to every choice
made in the window; what the window's decisions depend on is the long
run's relative values (what a part of each age is worth carrying past
the window's end). They are worked out by the same dynamic programme as
the window's: the recurring shuts on their real dates past the window's
end, backward from far enough on that the window's decisions no longer
change (``_settled``: run on by ``_horizon.LATER_YEARS`` years or three
lives, doubled until two runs agree). The window's decisions are then
those of the long-run average, whatever lies past the run-on.

After the window, a part is replaced by the long run's own rule (its
thresholds), and ``next_replacement``, ``schedule`` and a diagram follow
the window's decisions and then that rule.
"""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from repyability.maintenance._calendar import ANYTIME
from repyability.maintenance._shutdowns import (
    Shutdowns,
    _day_text,
    _Decision,
    _iso_date,
    _Layout,
    _listed,
    _money,
    _number,
    _seconds_text,
)

#: Times the run-on past the window's end is doubled, at most, until the
#: window's decisions agree between two runs.
DOUBLINGS = 4


# ------------------------------------------------------------- the split


def _fixed(layout: _Layout) -> List[Tuple[str, date]]:
    """Every fixed date from now on, ``(kind, day)`` in order."""
    assert layout.now is not None
    out = [
        (s.name, d)
        for s in layout.site.shutdowns
        if s.dates is not None
        for d in s.dates
        if d >= layout.now
    ]
    return sorted(out, key=lambda x: (x[1], x[0]))


def _recurring_site(site: Shutdowns) -> Shutdowns:
    """The site's recurring shuts alone: those by a rule, a period or a
    use, and each kind given ``then_every`` as it recurs from its last
    date (under its own name)."""
    shuts = []
    for shut in site.shutdowns:
        if shut.dates is None:
            shuts.append(shut)
        else:
            after = shut._after_dates()
            if after is not None:
                shuts.append(after)
    return Shutdowns(shuts, any_day=site.any_day, time_zone=site.time_zone)


def _plural(name: str, n: int) -> str:
    return name if n == 1 or name.endswith("s") else name + "s"


def _fixed_words(fixed) -> str:
    """``"the turnarounds on 3 Mar 2027 and 9 Sep 2028"``."""
    by_kind: Dict[str, List[date]] = {}
    for name, day in fixed:
        by_kind.setdefault(name, []).append(day)
    return _listed(
        [
            f"the {_plural(name, len(days))} on "
            + _listed([_day_text(d) for d in days])
            for name, days in by_kind.items()
        ]
    )


def _then_every_notes(layout: _Layout, window_day: date) -> List[str]:
    from repyability.maintenance._shutdowns import _gap, _gap_words

    notes = []
    for shut in layout.site.shutdowns:
        if shut.then_every is not None and shut.dates:
            n, unit = _gap(shut.then_every, "then_every")
            notes.append(
                f"{_plural(shut.name, 2)} after {_day_text(shut.dates[-1])} "
                f"assumed every {_gap_words(n, unit)} (your setting)"
            )
    return notes


# ----------------------------------------------------------- the decision


def _run_to_failure(layout: _Layout) -> _Decision:
    """The long run with nothing planned: running to failure, by the
    closed form (discounted where the decision is given a discount)."""
    return _Decision(
        "run_to_failure",
        "After the window no shut recurs and the site allows no planned "
        "replacement on any other day, so the long run is running to "
        "failure (closed form: the failure price over the mean life).",
        "closed form",
        0.01,
        failure_timing=None,
        why="no shut recurs after the window",
        how="by the closed form",
    )


def dated_decision(
    layout: _Layout,
    *,
    method,
    failure_timing,
    refuse,
    dated_until,
    discount=None,
) -> Optional[_Decision]:
    """The dated window's decision, or None where the site has no window
    (no fixed date from now on and no ``dated_until``) and no fixed dates
    to leave out."""
    from repyability.maintenance._horizon import _later_end
    from repyability.maintenance._shutdowns import _decide, _life_scale

    site = layout.site
    has_dates = any(s.dates is not None for s in site.shutdowns)
    if not has_dates and dated_until is None:
        return None
    if layout.now is None:
        raise ValueError(
            "now= is needed, the day answers count from: the dated window "
            "is on dates."
        )
    fixed = _fixed(layout)
    last = fixed[-1][1] if fixed else None
    if dated_until is not None:
        until = _iso_date(dated_until, "dated_until")
        if last is not None and until < last:
            raise ValueError(
                f"dated_until ({until}) is before the last fixed date "
                f"({last}): the window runs at least to it, as fixed dates "
                "are never averaged; give a later day, or leave dated_until "
                "out."
            )
        if until < layout.now:
            raise ValueError(
                f"dated_until must be on or after now ({layout.now}); got "
                f"{dated_until!r}."
            )
        last = until
    recurring = _recurring_site(site)
    rec_layout = _Layout(layout.part, recurring, **layout.given)
    if not recurring.shutdowns and recurring.any_day is None:
        long_run = replace(_run_to_failure(rec_layout), discount=discount)
    else:
        long_run = _decide(
            rec_layout,
            method=method,
            horizon=None,
            discount=discount,
            failure_timing=failure_timing,
            refuse=False,
        )
    if last is None:
        # Fixed dates, all before now: the recurring shuts alone.
        if long_run.route in ("horizon_or_averaged", "long_cycle") and refuse:
            _decide(
                rec_layout,
                method=method,
                horizon=None,
                discount=discount,
                failure_timing=failure_timing,
            )
        return replace(long_run, long_run=long_run, recurring=rec_layout)
    end_moment = layout.clock.midnight(last + timedelta(days=1))
    window_end = float(layout.model_at([end_moment])[0])
    day = _day_text(last)
    lead = f"Dated window to {day}, then the long-run average."
    if discount is not None:
        from repyability.maintenance._shutdowns import _percent

        lead = (
            f"Dated window to {day}, then the discounted long run, at "
            f"{_percent(discount)} a year."
        )
    using = (
        f"Using {_fixed_words(fixed)} as dated"
        if fixed
        else "Using every shut on its real dates"
    )
    if long_run.route in ("horizon_or_averaged", "long_cycle"):
        text = (
            f"{lead} {using} to {day}; after it only the recurring shuts "
            f"remain, and their long run needs a choice: {long_run.text}"
        )
        if refuse:
            raise ValueError(text)
        return _Decision(
            long_run.route,
            text,
            None,
            None,
            discount=discount,
            window_day=last,
            long_run=long_run,
            recurring=rec_layout,
            fixed=tuple(fixed),
        )
    mean, oldest, _ = _life_scale(layout.part)
    end = _later_end(layout, window_end)
    if failure_timing == "exact":
        from repyability.maintenance._integration import run_ages

        shuts = sum(k.times(end).size for k in layout.kinds)
        ages = float(run_ages(mean, oldest))
        window_seconds = 2.0 * shuts * 2.0 * (5e-5 + 3e-7 * ages)
        how = (
            "by quadrature across each gap between the real shuts, failures "
            "timed at the instant they happen"
        )
        solver = "quadrature"
    else:
        steps = end / (mean / 1000.0)
        ages = oldest / (mean / 1000.0)
        window_seconds = 2.0 * 3.0 * steps * (1.2e-5 + 5.5e-9 * ages)
        how = (
            "on a grid of steps sized from the life (about "
            f"{_number(mean / 1000.0)} {layout.unit_words()}), each failure's "
            "replacement counted half a step old"
        )
        solver = "grid"
    seconds = window_seconds + (long_run.seconds or 0.0)
    notes = _then_every_notes(layout, last)
    text = (
        f"{lead} {using}, every shut in the window on its real dates, worked "
        f"out {how}; after {day} only the recurring shuts remain"
        + (f" ({'; '.join(notes)})" if notes else "")
        + (
            ", and their long-run average values everything beyond: "
            if discount is None
            else ", and their discounted long run values everything beyond "
            "(the window itself discounted to now too): "
        )
        + f"{long_run.text} {_seconds_text(seconds).capitalize()} in all."
    )
    return _Decision(
        "dated_then_long_run",
        text,
        solver,
        seconds,
        discount=discount,
        failure_timing=failure_timing,
        why=using,
        how=how,
        window_day=last,
        window_end=window_end,
        long_run=long_run,
        recurring=rec_layout,
        fixed=tuple(fixed),
    )


# ------------------------------------------------------------ the window


def _settled(layout: _Layout, decision: _Decision):
    """The window solved by the backward programme over every shut on its
    real dates, run on past its end until its decisions agree between
    two runs: the grid's (``_horizon._grid_solve``) or the quadrature's
    (``_horizon._quadrature_solve``). With a discount every step, the
    run-on's too, is discounted by the calendar time it spans, so what
    lies past the window is valued by the recurring shuts' discounted
    long run (their own on the real dates), not by undiscounted relative
    values."""
    from repyability.maintenance._horizon import (
        _grid_solve,
        _later_end,
        _quadrature_solve,
    )

    end = float(decision.window_end or 0.0)
    extra = _later_end(layout, end) - end
    discount = float(decision.discount or 0.0)
    previous = None
    found: Any = None
    for _ in range(DOUBLINGS + 1):
        if decision.failure_timing == "exact":
            shuts, values = _quadrature_solve(layout, end + extra, discount)
            inside = _within(shuts.times, end)
            seen = (
                tuple(np.asarray(shuts.kinds, object)[inside]),
                shuts.thresholds[inside],
            )
            found = ("exact", shuts, values, extra)
        else:
            solved = _grid_solve(layout, end + extra, discount)
            first, kinds, grid, step = (
                solved[1],
                solved[2],
                solved[4],
                solved[7],
            )
            n = _steps(end, step)
            names = np.array(list(grid.names) + [None], object)
            seen = (tuple(names[kinds[:n]]), first[:n])
            found = ("grid", solved, None, extra)
        if previous is not None and _agree(seen, previous, decision):
            break
        previous = seen
        extra *= 2.0
    return found


def _agree(seen, previous, decision) -> bool:
    """Whether two runs give the window the same decisions: the same
    kinds, and thresholds within a grid step and 0.1% (where the
    threshold climbs steeply before a cheap shut, a value's last digits
    move it several steps) or, by quadrature, within 1e-6."""
    kinds, limits = seen
    kinds_before, limits_before = previous
    if kinds != kinds_before or limits.shape != limits_before.shape:
        return False
    if decision.failure_timing == "exact":
        same = np.isclose(limits, limits_before, rtol=1e-6) | (
            np.isinf(limits) & np.isinf(limits_before)
        )
        return bool(same.all())
    never = (limits < 0) | (limits_before < 0)
    return bool(
        np.all((limits < 0) == (limits_before < 0))
        and np.all(
            np.abs(limits - limits_before)[~never]
            <= 1 + 1e-3 * np.abs(limits)[~never]
        )
    )


def _steps(end: float, step: float) -> int:
    """The grid steps in the window: to the one its end is laid on (a
    shut at the end, the last fixed date, is in it)."""
    return int(math.floor(end / step + 0.5)) + 1


def _within(times: np.ndarray, end: float) -> np.ndarray:
    """Which model times are in the window (its end included: on a clock
    that stands still, a weekend's shuts share the end's model time)."""
    return np.asarray(times) <= end + 1e-9 * max(1.0, abs(end))


class _TwoPhase:
    """The window's decisions to its end (included), then the long run's
    rule strictly after it."""

    def __init__(self, window, long_run, end: float, merge):
        self.window = window
        self.long_run = long_run
        self.end = end
        self.step = window.step
        self._merge = merge
        # A grid lays a shut up to half a step from its time.
        tiny = 1e-9 * max(1.0, abs(end))
        self.last = end + 0.5 * (window.step or 0.0) + tiny
        long_step = getattr(long_run, "step", None) or 0.0
        self.beyond = max(self.last, end + 0.5 * long_step) + tiny

    def next_planned(self, installed: float, after: float):
        if after <= self.last:
            time_, price, kind = self.window.next_planned(installed, after)
            if kind is not None and time_ <= self.last:
                return time_, price, kind
            after = self.beyond
        after = max(after, self.beyond)
        return self.long_run.next_planned(installed, after)

    def price_at(self, t: float) -> float:
        if t <= self.last:
            return self.window.price_at(t)
        return self.long_run.price_at(t)

    def grid_policy(self, end: float):
        return self._merge(end)


class _NoReplacement:
    """The long run of running to failure."""

    step = None

    def next_planned(self, installed, after):
        return math.inf, 0.0, None

    def price_at(self, t):
        return math.inf


def _constant_rule(policy) -> Optional[Dict[str, float]]:
    """The long run's rule as one age a kind (``ANYTIME`` for any other
    day), where its thresholds do not vary; else None."""
    from repyability.maintenance._site_policy import ANY_DAY

    rule = {}
    for name, (low, high) in policy.rule.thresholds.items():
        if abs(high - low) > 1e-9 * max(1.0, abs(high)) and math.isfinite(low):
            return None
        if math.isfinite(low):
            rule[ANYTIME if name == ANY_DAY else name] = float(low)
    return rule


def _merged_grid(layout, first, kinds, names, n, long_policy, end):
    """A grid ``CalendarPolicy`` to model time ``end`` (a cost's spread):
    the window's decisions for its ``n`` steps, then the long run's rule
    where it is one age a kind."""
    from repyability.maintenance._site_policy import (
        _dated_grid_policy,
        _FixedTable,
    )

    rule = {} if long_policy is None else _constant_rule(long_policy)
    if rule is None:
        raise NotImplementedError(
            "The long run's thresholds vary with where each shut falls in "
            "its cycle, which a cost's spread on a grid past the dated "
            "window does not follow: give until= on or before the window's "
            "end, or use method='simulate'."
        )
    policy = _dated_grid_policy(layout, rule, end)
    table = policy._table
    out_first = np.array(table.first, np.int64)
    out_kinds = np.array(table.kinds, np.int64)
    index = {name: k for k, name in enumerate(policy._names)}
    m = min(n, out_first.size)
    for i in range(m):
        k = int(kinds[i])
        name = names[k] if k >= 0 else None
        out_first[i] = first[i] if name in index else -1
        out_kinds[i] = index[name] if name in index else -1
    return policy._bind(
        _FixedTable(out_first, out_kinds),
        policy._names,
        policy._grid,
        policy._oldest,
    )


# ------------------------------------------------------------ the policy


def _long_run_policy(decision: _Decision, age, mc_samples, seed):
    """The long run's policy on the recurring shuts alone (None for
    running to failure)."""
    from repyability.maintenance._site_policy import _policy_for

    long_run = decision.long_run
    assert long_run is not None
    if long_run.route == "run_to_failure":
        return None
    return _policy_for(decision.recurring, long_run, age, mc_samples, seed)


def _run_to_failure_rate(layout: _Layout) -> float:
    from repyability.maintenance._replacement import _mean, _survival

    part = layout.part
    return float(part.cu) / _mean(part, _survival(part))


def _run_to_failure_eac(layout: _Layout, discount: float) -> float:
    """Running to failure's equivalent annual cost a unit of the part's
    clock, at the yearly ``discount``."""
    from repyability.maintenance._options import discounted_run_to_failure
    from repyability.maintenance._site_policy import model_rate

    rate = model_rate(layout, discount)
    return discounted_run_to_failure(layout.part, rate, float(layout.part.cu))


def _phrase_dated(layout, rows, unit) -> Tuple[List[str], List[str]]:
    """The window's shuts of kinds with fixed dates, one phrase each."""
    unit_words = f" {unit}" if unit else ""
    parts, nots = [], []
    for day, name, limit in rows:
        where = f"at the {name} on {_day_text(date.fromisoformat(day))}"
        if math.isfinite(limit):
            parts.append(f"{where} if older than {_number(limit)}{unit_words}")
        else:
            nots.append(where)
    return parts, nots


def _window_text(layout, day, rows, thresholds, long_text) -> str:
    from repyability.maintenance._site_policy import _rule_parts

    dated_parts, dated_nots = _phrase_dated(layout, rows, layout.unit)
    parts, nots = _rule_parts(
        thresholds,
        layout.unit,
        layout.site.any_day is not None,
        "by when it falls",
    )
    parts = dated_parts + parts
    nots = dated_nots + nots
    if parts:
        text = f"To {day}: replace " + "; ".join(parts)
        if nots:
            text += "; never " + " or ".join(nots)
    else:
        text = f"To {day}: run to failure"
    return f"{text}. After {day}: {long_text}"


def _dated_rows(layout, times_by_kind, threshold_at):
    """Each shut of a kind with fixed dates in the window: ``(day, kind,
    threshold)``."""
    rows = []
    for kind in layout.kinds:
        if kind.shut.dates is None:
            continue
        for t in times_by_kind.get(kind.name, ()):
            moment = kind.moment_of(float(t))
            if moment is None:
                continue
            day = layout.clock.local_naive(moment).date().isoformat()
            rows.append((day, kind.name, threshold_at(kind.name, float(t))))
    return sorted(rows)


def dated_policy(
    layout: _Layout, decision: _Decision, age, life, mc_samples, seed
):
    """The dated window's policy (see the module's notes)."""
    from repyability.maintenance import _replacement as R
    from repyability.maintenance._horizon import (
        _ShutEngine,
        _thresholds_until,
    )
    from repyability.maintenance._site_policy import (
        ANY_DAY,
        EAC,
        ReplacementRule,
        ShutdownPolicy,
        _figure,
        _FixedTable,
        _InnerEngine,
        _measure,
    )

    site = layout.site
    discount = decision.discount
    end = float(decision.window_end or 0.0)
    assert decision.window_day is not None
    day = _day_text(decision.window_day)
    long_policy = _long_run_policy(decision, age, mc_samples, seed)
    pace, measure = _measure(layout)
    scale = pace if pace is not None else 1.0
    if long_policy is None:
        rate = _run_to_failure_rate(layout)
        long_engine: Any = _NoReplacement()
        long_text = "Run to failure."
        long_thresholds: Dict[str, Tuple[float, float]] = {
            n: (math.inf, math.inf) for n in decision.recurring.site.names
        }
        long_route, long_solver = "run_to_failure", "closed form"
        cost, cost_se, cost_rate = rate * scale, None, rate
        if discount is not None:
            measure = EAC
            cost = _run_to_failure_eac(layout, discount) * scale
        long_statement = f"running to failure: {_figure(measure, cost, None)}"
        by_kind = {
            "failure": scale / R._mean(layout.part, R._survival(layout.part))
        }
    else:
        long_engine = long_policy._engine
        assert long_policy.rule is not None
        long_text = long_policy.rule.text
        long_thresholds = dict(long_policy.rule.thresholds)
        long_route, long_solver = long_policy.route, long_policy.solver
        long_statement = long_policy.route_text
        cost, cost_se = long_policy.cost, long_policy.cost_se
        cost_rate = long_policy.cost_rate
        measure = long_policy.measure
        by_kind = dict(long_policy.replacements_by_kind)
    kind, solved, values, extra = _settled(layout, decision)
    names_site = list(site.names)
    window_replacements: Dict[str, float] = {}
    if kind == "grid":
        v, first, kinds, costs, grid, ages, failure, step = solved
        n = _steps(end, step)
        names = list(grid.names)
        window = R.CalendarPolicy(
            cost_rate=math.nan,
            cost_rate_se=None,
            replacement_rates={},
            cost_rates={},
            method="dp",
            basis="dated window",
            step=step,
            period=None,
            determined_until=end,
            threshold_only=True,
            settings={"max_age": ages.size * step},
            life=life,
        )
        window = window._bind(
            _FixedTable(first[:n], kinds[:n]), names, grid, ages.size * step
        )
        window_engine: Any = _InnerEngine(window, layout)
        share = np.zeros(ages.size + 1)
        share[min(int(round(age / step)), ages.size)] = 1.0
        _, replaced, failures = R._propagate(share, first[:n], ages.p)
        if discount is None:
            planned = replaced > 0
            dated_cost = float(np.dot(replaced[planned], costs[:n][planned]))
            dated_cost += float(failures) * float(failure)
        else:
            # The window's own decisions, each step discounted by the
            # calendar time it spans: the present value to its end.
            from repyability.maintenance._horizon import _betas

            to_end, _, _ = R._sweep(
                np.zeros(ages.size + 1),
                costs[:n],
                ages.p,
                failure,
                _betas(layout, n, step, discount),
                fixed=first[:n],
            )
            dated_cost = float(to_end[min(int(round(age / step)), ages.size)])
        for k, name in enumerate(names):
            label = ANY_DAY if name == ANYTIME else name
            window_replacements[label] = float(replaced[kinds[:n] == k].sum())
        window_replacements["failure"] = float(failures)
        thresholds = _thresholds_until(
            first, kinds, names, step, n * step, names_site, site.any_day
        )

        def threshold_at(name, t):
            i = int(np.rint(t / step))
            if 0 <= i < n and kinds[i] >= 0 and names[kinds[i]] == name:
                return float(first[i] * step) if first[i] >= 0 else math.inf
            return math.inf

        def merge(until):
            if until <= end:
                return window
            return _merged_grid(
                layout, first, kinds, names, n, long_policy, until
            )

        settings: Dict[str, Any] = {"step": step, "window_steps": n}
    else:
        shuts = solved
        inside = _within(shuts.times, end)
        window_engine = _ShutEngine(
            shuts.times[inside],
            shuts.thresholds[inside],
            [k for k, x in zip(shuts.kinds, inside) if x],
            shuts.prices[inside],
        )
        rule_ages = {
            float(t): float(x)
            for t, x in zip(shuts.times[inside], shuts.thresholds[inside])
        }
        # The window's own decisions to its end, priced from the part now.
        from repyability.maintenance._horizon import _Shuts

        rhos = None
        if discount is not None:
            from repyability.maintenance._horizon import _segment_rates

            rhos = _segment_rates(
                layout,
                np.concatenate([[0.0], shuts.times[inside], [end]]),
                discount,
            )
        priced = _Shuts(
            layout.part,
            shuts.times[inside],
            shuts.prices[inside],
            [k for k, x in zip(shuts.kinds, inside) if x],
            0.0,
            end,
            layout.part.cu,
            rhos,
            shuts.thresholds[inside],
        )
        priced_values = priced.solve()
        dated_cost = float(np.interp(age, priced.grid, priced_values))
        replaced, failures = priced.counts(age)
        for k, x in zip(priced.kinds, replaced):
            window_replacements[k] = window_replacements.get(k, 0.0) + float(x)
        window_replacements["failure"] = float(failures)
        thresholds = {}
        for name in names_site:
            use = [
                x
                for x, k in zip(shuts.thresholds[inside], window_engine.kinds)
                if k == name and math.isfinite(x)
            ]
            thresholds[name] = (
                (min(use), max(use)) if use else (math.inf, math.inf)
            )

        def threshold_at(name, t):
            return rule_ages.get(float(t), math.inf)

        def merge(until):
            raise NotImplementedError(
                "A policy worked out by quadrature (failure_timing='exact') "
                "has no grid for a cost's spread: work it out with "
                "failure_timing='half_step', or use method='simulate'."
            )

        settings = {"nodes": 32, "window_shuts": int(inside.sum())}
    engine = _TwoPhase(window_engine, long_engine, end, merge)
    times_by_kind = {k.name: k.times(end) for k in layout.kinds}
    times_by_kind = {k: t[_within(t, end)] for k, t in times_by_kind.items()}
    rows = _dated_rows(layout, times_by_kind, threshold_at)
    dated_kinds = {s.name for s in site.shutdowns if s.dates is not None}
    window_thresholds = {
        k: v for k, v in thresholds.items() if k not in dated_kinds
    }
    text_rule = _window_text(layout, day, rows, window_thresholds, long_text)
    after_day = decision.window_day + timedelta(days=1)
    assert layout.now is not None
    rule = ReplacementRule(
        text_rule,
        long_thresholds,
        [
            (
                layout.now.isoformat(),
                decision.window_day.isoformat(),
                thresholds,
            ),
            (after_day.isoformat(), None, long_thresholds),
        ],
        layout.unit,
        rows,
    )
    fixed = decision.fixed
    using = (
        f"Using {_fixed_words(fixed)} as dated"
        if fixed
        else "Using every shut on its real dates"
    )
    notes = _then_every_notes(layout, decision.window_day)
    figure = _figure(measure, cost, cost_se)
    if discount is None:
        then = "the long-run average"
        cost_words = "Expected cost"
    else:
        from repyability.maintenance._shutdowns import _percent

        then = f"the discounted long run at {_percent(discount)} a year"
        figure += " (equivalent annual cost)"
        cost_words = "Present value of the cost"
    text = (
        f"Dated window to {day}, then {then}: {figure}. "
        f"{using}, then the recurring shuts' "
        + ("long-run average" if discount is None else "discounted long run")
        + f" after {day}"
        + (f"; {'; '.join(notes)}" if notes else "")
        + f". {cost_words} to {day}: "
        + (
            _money(round(dated_cost, 2))
            if dated_cost >= 0.01
            else f"${dated_cost:.3g}"
        )
        + f". The window is worked out on the real dates {decision.how}; "
        f"the long run: {long_statement}"
    )
    settings.update(
        dated_until=decision.window_day.isoformat(),
        run_on=extra,
        window_replacements=window_replacements,
        long_run_settings=(
            {} if long_policy is None else dict(long_policy.settings)
        ),
    )
    all_notes = list(layout.notes) + notes
    return ShutdownPolicy(
        route="dated_then_long_run",
        route_text=text,
        measure=measure,
        cost=float(cost),
        cost_se=cost_se,
        solver=decision.solver or "",
        failure_timing=decision.failure_timing,
        rule=rule,
        replacements_by_kind=by_kind,
        cost_rate=None if cost_rate is None else float(cost_rate),
        unit=layout.unit,
        now=layout.now.isoformat(),
        age=float(age),
        notes=tuple(all_notes),
        settings=settings,
        life=life,
        dated_until=decision.window_day.isoformat(),
        dated_cost=float(dated_cost),
        long_run_route=long_route,
        long_run_solver=long_solver,
        discount=None if discount is None else float(discount),
    )._bind(engine, layout, decision)


# ------------------------------------------------------------ a given rule


def recurring_rule(decision: _Decision, rule) -> Dict[str, float]:
    """A rule's ages for the recurring shuts and any other day only."""
    from repyability.maintenance._site_policy import ANY_DAY

    names = decision.recurring.site.names
    return {k: v for k, v in rule.items() if k == ANY_DAY or k in names}


def dated_rule_cost(
    layout: _Layout, decision: _Decision, rule, name, settings
):
    """A fixed rule at a site with a dated window: its long-run cost
    after the window (on the recurring shuts, by their route), and the
    expected cost of a new part to the window's end; with a discount,
    the recurring shuts' discounted long run (its equivalent annual
    cost) and the present value to the window's end."""
    from repyability.maintenance._horizon import _grid_solve, _quadrature_solve
    from repyability.maintenance._site_policy import (
        ANY_DAY,
        RuleCost,
        _figure,
        _measure,
        _rule_cost,
    )

    end = float(decision.window_end or 0.0)
    recurring = decision.recurring
    long_run = decision.long_run
    assert long_run is not None and decision.window_day is not None
    kept = recurring_rule(decision, rule)
    pace, measure = _measure(layout)
    scale = pace if pace is not None else 1.0
    discount = float(decision.discount or 0.0)
    eac = None
    if long_run.route == "run_to_failure":
        rate = _run_to_failure_rate(layout)
        long = None
        if discount:
            eac = _run_to_failure_eac(layout, discount) * scale
    else:
        long = _rule_cost(recurring, long_run, kept, name, settings)
        rate = float(long.cost_rate or 0.0)
        eac = long.eac
    if decision.failure_timing == "exact":
        _, values = _quadrature_solve(layout, end, discount, rule)
        dated_cost = float(values[0])
    else:
        grid_rule = {
            (ANYTIME if k == ANY_DAY else k): v for k, v in rule.items()
        }
        grid_rule = {k: v for k, v in grid_rule.items() if math.isfinite(v)}
        v, *_ = _grid_solve(layout, end, discount, grid_rule)
        dated_cost = float(v[0])
    day = _day_text(decision.window_day)
    figure = float(rate * scale) if eac is None else float(eac)
    if eac is not None:
        from repyability.maintenance._site_policy import EAC

        measure = EAC
    worth = "expected cost" if eac is None else "present value"
    return RuleCost(
        name=name,
        rule=dict(rule),
        cost=figure,
        measure=measure,
        cost_se=None if long is None else long.cost_se,
        route="dated_then_long_run",
        route_text=(
            f"{name}: {_figure(measure, figure, None)} after {day}; "
            f"{worth} to {day} {_money(round(dated_cost, 2))}"
        ),
        solver=decision.solver or "",
        failure_timing=decision.failure_timing,
        replacements_by_kind=(
            {"failure": 0.0}
            if long is None
            else dict(long.replacements_by_kind)
        ),
        cost_rate=rate,
        dated_cost=dated_cost,
        eac=eac,
    )
