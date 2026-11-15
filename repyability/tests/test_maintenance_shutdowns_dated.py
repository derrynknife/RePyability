"""Fixed dates at a site (UC-43 of the maintenance README): a dated
window to the last fixed date, every shut in it on its real dates, then
the long-run average of the recurring shuts; ``then_every``; and the part
in service seeing the fixed dates ahead."""

import json
import math
from datetime import date

import numpy as np
import pytest
import surpyval as surv

from repyability import (
    Calendar,
    NonRepairable,
    Shutdown,
    Shutdowns,
    SlowSolverWarning,
)
from repyability.schedule import DutyPlan, Recurrence, Shifts

LIFE = surv.Weibull.from_params([1000.0, 2.5])
NOW = "2026-11-16"
TURNAROUNDS = ["2027-03-03", "2028-09-09"]


def plant_part():
    return NonRepairable(LIFE, unplanned_cost=20.0)


def four_weekly():
    return Shutdown(
        "stop", Recurrence.every(4, "weeks", "2026-11-21"), replace_cost=3.0
    )


def turnaround(**options):
    return Shutdown(
        "turnaround", dates=TURNAROUNDS, replace_cost=0.5, **options
    )


def window_table(policy):
    """The window's decisions on the grid: each step's first replacing age
    index and the kind there, by name."""
    inner = policy._engine.window.inner
    n = policy.settings["window_steps"]
    first, kinds = inner._table.at(np.arange(n))
    names = [inner._names[k] if k >= 0 else None for k in kinds]
    return list(first), names


# ------------------------------------------------------ the plant example


PUMP_BEARING = NonRepairable(
    surv.Weibull.from_params([20_000, 2.5]), unplanned_cost=40.0
)
RECURRING = [
    Shutdown(
        "weekly stop",
        Recurrence.weekly("Sat"),
        replace_cost=3.0,
        duration="8h",
    ),
    Shutdown(
        "quarterly shut",
        Recurrence.monthly(
            weekday="Sunday", nth=1, every=3, anchor="2026-12-06"
        ),
        replace_cost=2.0,
        duration="1d",
    ),
    Shutdown(
        "annual shut",
        Recurrence.yearly(month="November", weekday="Tuesday", nth=1),
        replace_cost=1.0,
        duration="3d",
    ),
]
PLANT = Shutdowns(
    RECURRING + [turnaround(duration="14d")],
    any_day=4.0,
    time_zone="Australia/Brisbane",
)
SHIFTS = DutyPlan(
    Shifts("Mon-Fri", "06:00", "22:00"), time_zone="Australia/Brisbane"
)
WHEN = dict(plan=SHIFTS, now=NOW, unit="hours", age=3100)


@pytest.fixture(scope="module")
def plant_policy():
    return PUMP_BEARING.optimal_calendar_policy(
        PLANT, method="averaged", **WHEN
    )


def test_turnarounds_before_sep_2028_and_annual_shuts_after(plant_policy):
    best = plant_policy
    assert best.route == "dated_then_long_run"
    assert best.solver == "grid" and best.failure_timing == "half_step"
    assert best.long_run_route == "averaged"
    assert best.long_run_solver == "closed form"
    assert best.dated_until == "2028-09-09"
    assert best.measure == "cost per year"
    assert best.route_text.startswith(
        "Dated window to 9 Sep 2028, then the long-run average: $1.90 a year. "
        "Using the turnarounds on 3 Mar 2027 and 9 Sep 2028 as dated, then "
        "the recurring shuts' long-run average after 9 Sep 2028."
    )
    # The rule in two phases: the window's (each turnaround its own), then
    # the long run's.
    window, after = best.rule.text.split(" After 9 Sep 2028: ")
    assert window.startswith(
        "To 9 Sep 2028: replace at the turnaround on 3 Mar 2027 if older "
        "than 1,544 hours; at the turnaround on 9 Sep 2028 if older than "
        "798.5 hours;"
    )
    assert after.startswith("Replace at an annual shut if older than 1,949")
    assert [r[:2] for r in best.rule.dated] == [
        ("2027-03-03", "turnaround"),
        ("2028-09-09", "turnaround"),
    ]
    assert best.rule.by_period[1][1] is None
    assert str(best.next_replacement) == (
        "At the turnaround, 3 Mar 2027, part 4,332 old"
    )
    kinds = best.schedule("2031-12-31")
    before = [k for d, k, _ in kinds if d <= "2028-09-09"]
    later = [k for d, k, _ in kinds if d > "2028-09-09"]
    assert "turnaround" in before
    assert later and set(later) == {"annual shut"}
    assert best.dated_cost == pytest.approx(3.3073, abs=1e-3)
    data = json.loads(json.dumps(best.to_dict()))
    assert data["route"] == "dated_then_long_run"


def test_the_long_run_figure_is_the_recurring_shuts_alone(plant_policy):
    recurring = Shutdowns(
        RECURRING, any_day=4.0, time_zone="Australia/Brisbane"
    )
    alone = PUMP_BEARING.optimal_calendar_policy(
        recurring, method="averaged", **WHEN
    )
    assert plant_policy.cost == alone.cost
    assert plant_policy.cost_rate == alone.cost_rate
    assert plant_policy.rule.thresholds == alone.rule.thresholds
    assert plant_policy.replacements_by_kind == alone.replacements_by_kind


def test_the_route_check_names_the_dated_window():
    choice = PLANT.route(PUMP_BEARING, plan=SHIFTS, now=NOW, unit="hours")
    assert choice.route == "horizon_or_averaged" and choice.solver is None
    assert choice.text.startswith(
        "Dated window to 9 Sep 2028, then the long-run average."
    )
    assert "Choose method='averaged'" in choice.text
    route = PLANT.route(
        PUMP_BEARING, plan=SHIFTS, now=NOW, unit="hours", method="averaged"
    )
    assert route.route == "dated_then_long_run"
    assert route.solver == "grid"
    assert route.long_run_route == "averaged"
    assert route.long_run_solver == "closed form"
    assert route.dated_until == "2028-09-09"
    assert route.text.startswith(
        "Dated window to 9 Sep 2028, then the long-run average. Using the "
        "turnarounds on 3 Mar 2027 and 9 Sep 2028 as dated, every shut in "
        "the window on its real dates"
    )
    with pytest.raises(ValueError, match="Dated window to 9 Sep 2028.*Choose"):
        PUMP_BEARING.optimal_calendar_policy(PLANT, **WHEN)


def test_the_part_in_service_sees_the_turnarounds_ahead(plant_policy):
    choice = PUMP_BEARING.optimal_next_replacement(
        3100, site=PLANT, policy=plant_policy
    )
    shuts = [(o.name, o.date) for o in choice.options if o.kind == "shut"]
    assert ("turnaround", "2027-03-03") in shuts
    assert ("turnaround", "2028-09-09") in shuts
    assert choice.best == "At the turnaround, 3 Mar 2027"
    assert choice.cost_rate == pytest.approx(plant_policy.cost_rate)
    assert "after the dated window" in choice.valued_by


def test_compare_and_costs_on_dates(plant_policy):
    compared = plant_policy.compare()
    assert [c.name for c in compared][:2] == [
        "run to failure",
        "age replacement on any day at 7,091",
    ]
    for c in compared:
        assert c.route == "dated_then_long_run"
        assert c.dated_cost is not None and c.dated_cost > 0
    spend = PUMP_BEARING.cost_distribution(
        until="2030-12-31", policy=plant_policy
    )
    inside = PUMP_BEARING.cost_distribution(
        until="2027-12-31", policy=plant_policy
    )
    assert 0 < inside.mean < spend.mean


# ---------------------------------------------- the window and the horizon


def test_the_window_is_the_horizon_route_far_out_at_no_discount():
    part = plant_part()
    site = Shutdowns([four_weekly(), turnaround()], any_day=4.0)
    dated = part.optimal_calendar_policy(site, unit="days", now=NOW)
    assert dated.route == "dated_then_long_run"
    assert dated.long_run_route == "exact"
    far = part.optimal_calendar_policy(
        site, unit="days", now=NOW, horizon="2066-11-16", discount=0.0
    )
    n = dated.settings["window_steps"]
    inner = far._engine.inner
    first, kinds = inner._table.at(np.arange(n))
    names = [inner._names[k] if k >= 0 else None for k in kinds]
    window_first, window_names = window_table(dated)
    assert window_names == names
    # The same thresholds, to a grid step and 0.2% (where a threshold
    # climbs steeply before the turnaround, rounding moves it a few steps).
    gap = np.abs(np.array(window_first) - first)
    assert np.all(gap <= 1 + 2e-3 * np.abs(first))
    assert np.mean(gap == 0) > 0.95
    # And by quadrature (no replacement on any other day).
    exact_site = Shutdowns([four_weekly(), turnaround()])
    with pytest.warns(SlowSolverWarning):
        by_quadrature = part.optimal_calendar_policy(
            exact_site, unit="days", now=NOW, failure_timing="exact"
        )
    on_grid = part.optimal_calendar_policy(exact_site, unit="days", now=NOW)
    assert by_quadrature.solver == "quadrature"
    assert by_quadrature.long_run_solver == "quadrature"
    step = on_grid.settings["step"]
    for a, b in zip(by_quadrature.rule.dated, on_grid.rule.dated):
        assert a[:2] == b[:2]
        # The grid's threshold is a whole number of its steps, the first
        # age that replaces: within a step of the quadrature's.
        assert abs(a[2] - b[2]) <= step
    assert by_quadrature.cost == pytest.approx(on_grid.cost, rel=1e-4)
    # The grid's window ends on the step its end is laid on, up to half a
    # step later: about a failure's chance in half a day more.
    assert by_quadrature.dated_cost == pytest.approx(
        on_grid.dated_cost, rel=5e-3
    )


def _keep_less_replace(policy, at, ages, per_day=8, years=20.0):
    """An independent check of a threshold at a shut (``at`` days from
    now): what keeping a part of each age costs over replacing it there,
    every other decision held at the policy's (each later shut's
    threshold in the window, then the long run's), over ``years`` with
    nothing after, by the trapezoid rule on a grid of ``per_day`` points a
    day. The threshold is where it equals the shut's price."""
    eta, beta, failure = 1000.0, 2.5, 20.0
    window = policy._engine.window
    long_limit = policy.rule.thresholds["stop"][0]
    h = 1.0 / per_day
    end = at + years * 365.25
    times = list(window.times[window.times > at])
    limits = list(window.thresholds[window.times > at])
    prices = list(window.prices[window.times > at])
    t = 5.0 + 28.0 * math.ceil((window.times[-1] - 5.0) / 28.0 + 1e-9)
    while t <= end + 1000.0:
        times.append(t)
        limits.append(long_limit)
        prices.append(3.0)
        t += 28.0
    times, limits, prices = map(np.array, (times, limits, prices))

    def survival(x):
        return np.exp(-((x / eta) ** beta))

    def density(x):
        return beta / eta * (x / eta) ** (beta - 1) * survival(x)

    def first_due(t, age):
        due = (times > t + 1e-9) & (age + times - t >= limits - 1e-9)
        i = int(np.argmax(due))
        return (times[i], prices[i]) if due.any() else (math.inf, 0.0)

    def kept(age, i, values):
        due, cost = first_due(at + i * h, age)
        if due > end:
            n, cost = values.size - 1 - i, None
        else:
            n = int(round((due - at) / h)) - i
        w = density(age + np.arange(n + 1) * h) / survival(age) * h
        w[[0, -1]] *= 0.5
        value = np.dot(w, failure + values[i : i + n + 1])
        if cost is not None:
            value += (
                survival(age + n * h) / survival(age) * (cost + values[i + n])
            )
        return value

    n = int(round((end - at) / h))
    new = np.zeros(n + 1)
    for i in range(n, -1, -1):
        new[i] = kept(0.0, i, new)
    return np.array([kept(a, 0, new) - new[0] for a in ages])


def test_a_dated_threshold_is_where_keeping_costs_the_shuts_price():
    # Keeping a part at the turnaround on 3 Mar 2027 costs 0.5 more than
    # replacing it from 28.4 days old, the rest of the policy held: the
    # quadrature's threshold, and the grid's to its step. (The
    # quadrature's age grid was once too coarse for the kinks later
    # thresholds leave in the cost to go, and gave 30.66.)
    part = plant_part()
    site = Shutdowns([four_weekly(), turnaround()])
    with pytest.warns(SlowSolverWarning):
        exact = part.optimal_calendar_policy(
            site, unit="days", now=NOW, failure_timing="exact"
        )
    on_grid = part.optimal_calendar_policy(site, unit="days", now=NOW)
    first = exact.rule.dated[0]
    assert first[:2] == ("2027-03-03", "turnaround")
    at = 107.0  # 3 Mar 2027, days from now
    ages = np.array([28.0, 28.5, 29.0])
    extra = _keep_less_replace(exact, at, ages)
    crossing = float(np.interp(0.5, extra, ages))
    assert crossing == pytest.approx(28.40, abs=0.03)
    assert first[2] == pytest.approx(crossing, abs=0.03)
    step = on_grid.settings["step"]
    assert abs(on_grid.rule.dated[0][2] - crossing) <= step


def test_a_rule_and_dates_combine_freely_in_the_window():
    # A monthly rule and two fixed dates give the window's
    # decisions of the same shuts all as fixed dates (the monthly ones then
    # recurring from the last, as the rule does).
    part = plant_part()
    monthly = Recurrence.monthly(day=15)
    by_rule = Shutdowns(
        [Shutdown("monthly", monthly, replace_cost=3.0), turnaround()],
        any_day=4.0,
    )
    days = by_rule.dates(NOW, "2028-09-09")["monthly"]
    as_dates = Shutdowns(
        [
            Shutdown(
                "monthly", dates=days, then_every="1 month", replace_cost=3.0
            ),
            turnaround(),
        ],
        any_day=4.0,
    )
    options = dict(unit="days", now=NOW, method="averaged")
    a = part.optimal_calendar_policy(by_rule, **options)
    b = part.optimal_calendar_policy(as_dates, **options)
    assert a.dated_until == b.dated_until == "2028-09-09"
    assert window_table(a) == window_table(b)
    # The long run averages the monthly gaps over 28 years from the rule's
    # start or from the last date: the same to a few in 1e8.
    assert a.cost == pytest.approx(b.cost, rel=1e-6)
    assert a.dated_cost == pytest.approx(b.dated_cost, rel=1e-6)
    assert a.schedule("2031-12-31") == b.schedule("2031-12-31")


def test_then_every_is_a_recurring_kind_from_the_last_date():
    part = plant_part()
    options = dict(unit="days", now=NOW, age=200)
    given = Shutdowns([turnaround(then_every="4 years")], any_day=4.0)
    added = Shutdowns(
        [
            turnaround(),
            Shutdown(
                "turnaround after",
                Recurrence.every(4, "years", TURNAROUNDS[-1]),
                replace_cost=0.5,
            ),
        ],
        any_day=4.0,
    )
    a = part.optimal_calendar_policy(given, **options)
    b = part.optimal_calendar_policy(added, **options)
    assert a.long_run_route == b.long_run_route == "exact"
    assert a.cost == pytest.approx(b.cost, rel=1e-12)
    assert a.dated_cost == pytest.approx(b.dated_cost, rel=1e-12)
    assert np.allclose(
        sorted(a.rule.thresholds.values()), sorted(b.rule.thresholds.values())
    )
    named = [
        (d, k.replace(" after", ""), age)
        for d, k, age in b.schedule("2040-12-31")
    ]
    assert a.schedule("2040-12-31") == named
    assert (
        "turnarounds after 9 Sep 2028 assumed every 4 years (your setting)"
        in (a.route_text)
    )
    assert any("every 4 years" in n for n in a.notes)


# ------------------------------------------------------- the edges


def test_fixed_dates_with_nothing_recurring_run_to_failure_after():
    part = plant_part()
    site = Shutdowns([turnaround()])
    best = part.optimal_calendar_policy(site, unit="days", now=NOW)
    assert best.route == "dated_then_long_run"
    assert best.long_run_route == "run_to_failure"
    mean = float(LIFE.mean())
    assert best.cost == pytest.approx(20.0 / mean * 365.25, rel=1e-6)
    assert best.rule.text.endswith("After 9 Sep 2028: Run to failure.")


def test_dated_until_extends_the_window():
    part = plant_part()
    site = Shutdowns([four_weekly()], any_day=4.0)
    plain = part.optimal_calendar_policy(site, unit="days", now=NOW)
    assert plain.route == "exact"
    longer = part.optimal_calendar_policy(
        site, unit="days", now=NOW, dated_until="2029-12-31"
    )
    assert longer.route == "dated_then_long_run"
    assert longer.dated_until == "2029-12-31"
    assert longer.cost == pytest.approx(plain.cost, rel=1e-12)
    assert "Using every shut on its real dates" in longer.route_text
    route = site.route(part, unit="days", now=NOW, dated_until="2029-12-31")
    assert route.route == "dated_then_long_run"
    with_dates = Shutdowns([four_weekly(), turnaround()], any_day=4.0)
    later = part.optimal_calendar_policy(
        with_dates, unit="days", now=NOW, dated_until="2030-06-30"
    )
    assert later.dated_until == "2030-06-30"
    with pytest.raises(ValueError, match="for a site"):
        part.optimal_calendar_policy(Calendar(), dated_until="2030-01-01")


def test_fixed_dates_all_past_leave_the_recurring_shuts():
    part = plant_part()
    old = Shutdowns(
        [
            four_weekly(),
            Shutdown("old", dates=["2025-01-01"], replace_cost=0.1),
        ],
        any_day=4.0,
    )
    alone = Shutdowns([four_weekly()], any_day=4.0)
    a = part.optimal_calendar_policy(old, unit="days", now=NOW)
    b = part.optimal_calendar_policy(alone, unit="days", now=NOW)
    assert a.route == b.route == "exact"
    assert a.cost == b.cost
    priced = part.calendar_policy_cost(
        old, {"stop": 300.0}, unit="days", now=NOW
    )
    assert (
        priced.cost
        == part.calendar_policy_cost(
            alone, {"stop": 300.0}, unit="days", now=NOW
        ).cost
    )


def test_a_given_rule_on_the_dated_window():
    part = plant_part()
    site = Shutdowns([four_weekly(), turnaround()], any_day=4.0)
    priced = part.calendar_policy_cost(
        site, {"turnaround": 0.0, "stop": 400.0}, unit="days", now=NOW
    )
    assert priced.route == "dated_then_long_run"
    alone = part.calendar_policy_cost(
        Shutdowns([four_weekly()], any_day=4.0),
        {"stop": 400.0},
        unit="days",
        now=NOW,
    )
    assert priced.cost == pytest.approx(alone.cost, rel=1e-12)
    assert priced.dated_cost > 0 and "expected cost to 9 Sep 2028" in (
        priced.route_text
    )


def test_then_every_is_described_saved_and_checked():
    shut = turnaround(then_every="4 years")
    assert shut.describe() == (
        "turnaround (3 Mar 2027, 9 Sep 2028, then every 4 years, $0.50)"
    )
    again = Shutdown.from_dict(json.loads(json.dumps(shut.to_dict())))
    assert again == shut
    assert Shutdowns([shut]).dates(NOW, "2036-12-31")["turnaround"][-2:] == (
        date(2032, 9, 9),
        date(2036, 9, 9),
    )
    with pytest.raises(ValueError, match="give dates= too"):
        Shutdown(
            "t", Recurrence.weekly("Sat"), then_every="1 year", replace_cost=1
        )
    with pytest.raises(ValueError, match="whole number of days"):
        turnaround(then_every="4.5 years")
    assert not math.isnan(shut.replace_cost)


def test_the_guide_plant_example(plant_policy):
    # docs/guide/maintenance.md, Example 3, line by line.
    best = plant_policy
    assert round(best.cost, 2) == 1.9
    assert round(best.dated_cost, 2) == 3.31
    assert (best.solver, best.long_run_route) == ("grid", "averaged")
    assert best.route_text.split(" Expected")[0] == (
        "Dated window to 9 Sep 2028, then the long-run average: $1.90 a "
        "year. Using the turnarounds on 3 Mar 2027 and 9 Sep 2028 as dated, "
        "then the recurring shuts' long-run average after 9 Sep 2028."
    )
    assert [(d, k, round(a, 1)) for d, k, a in best.rule.dated] == [
        ("2027-03-03", "turnaround", 1543.8),
        ("2028-09-09", "turnaround", 798.5),
    ]
    assert [k for _, k, _ in best.schedule("2031-12-31")] == [
        "annual shut",
        "turnaround",
        "annual shut",
        "annual shut",
        "annual shut",
    ]
