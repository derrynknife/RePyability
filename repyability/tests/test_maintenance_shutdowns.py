"""A site's shuts on real dates (UC-43 of the maintenance README): the
record, the four routes (each named, none chosen silently), the results on
dates, and one record for policies and diagrams."""

import json
import math
import warnings

import numpy as np
import pytest
import surpyval as surv

from repyability import (
    Calendar,
    NonRepairable,
    Opportunity,
    RepairableRBD,
    Shutdown,
    ShutdownPolicy,
    Shutdowns,
    SlowSolverWarning,
)
from repyability.maintenance._site_policy import _Averaged
from repyability.schedule import DutyPlan, Rate, Recurrence, Shifts, Usage

# The 0.14 and 0.19 forms (Calendar, Shutdown, Shutdowns, prices on the
# opportunities, optimal_calendar_policy) still work, deprecated: these
# tests keep them unchanged, and test_opportunities.py the new forms.
pytestmark = pytest.mark.filterwarnings(
    "ignore:.*(is deprecated|puts a price on the opportunities):FutureWarning"
)

LIFE = surv.Weibull.from_params([1000.0, 2.5])
NOW = "2026-11-16"


def textbook():
    part = NonRepairable(LIFE)
    part.set_costs_planned_and_unplanned(1, 5)
    return part


def plant_part():
    return NonRepairable(LIFE, unplanned_cost=20.0)


ANNUAL = Recurrence.yearly(month="November", weekday="Tuesday", nth=1)
QUARTERLY_ANNUAL = Shutdowns(
    [
        Shutdown("quarterly", every="91 days", replace_cost=3.0),
        Shutdown("annual", every="364 days", replace_cost=1.0),
    ]
)


# ------------------------------------------------------------- the record


def test_the_record_describes_saves_and_lists_its_dates():
    site = Shutdowns(
        [
            Shutdown(
                "weekly stop",
                Recurrence.weekly("Sat"),
                replace_cost=3.0,
                duration="8h",
            ),
            Shutdown("annual shut", ANNUAL, replace_cost=1.0, duration="3d"),
            Shutdown(
                "turnaround",
                dates=["2027-03-03", "2028-09-09"],
                replace_cost=0.5,
                duration="14d",
            ),
            Shutdown(
                "service",
                every=Usage(15_000, "km"),
                also_every="6 months",
                replace_cost=9_000,
            ),
        ],
        any_day=4.0,
        time_zone="Australia/Brisbane",
    )
    text = site.describe()
    assert text.startswith("4 kinds of shut: weekly stop (every Saturday, $3)")
    assert "turnaround (3 Mar 2027, 9 Sep 2028, $0.50)" in text
    assert text.endswith("any other day $4.")
    saved = json.loads(json.dumps(site.to_dict()))
    assert Shutdowns.from_dict(saved) == site
    plan = DutyPlan(Rate(400, per="day", meter="km"))
    dates = site.dates(NOW, "2027-12-31", plan=plan, readings={"km": 212_000})
    assert len(dates["weekly stop"]) == 58
    assert [str(d) for d in dates["annual shut"]] == ["2027-11-02"]
    assert [str(d) for d in dates["turnaround"]] == ["2027-03-03"]
    # 13,000 km to the first service at 400 km a day, then every 37.5 days.
    assert str(dates["service"][0]) == "2026-12-18"


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({}, "one way"),
        ({"rule": ANNUAL, "every": "30 days"}, "one way"),
        ({"every": "30 parsecs"}, "unit of time"),
        ({"every": "30 days", "also_every": "6 months"}, "shut by use"),
        ({"every": Usage(500), "at": "06:00"}, "takes no clock time"),
        ({"rule": ANNUAL, "replace_cost": -1.0}, "at least 0"),
    ],
)
def test_a_shut_is_checked_where_it_is_given(kwargs, match):
    kwargs = {"replace_cost": 1.0, **kwargs}
    rule = kwargs.pop("rule", None)
    with pytest.raises((ValueError, TypeError), match=match):
        Shutdown("s", rule, **kwargs)


# ------------------------------------------------------- route 1, exact


def test_no_shuts_is_the_textbook_age_replacement():
    part = textbook()
    best = part.optimal_calendar_policy(Shutdowns([], any_day=1), unit="days")
    assert best.route == "exact" and best.solver == "closed form"
    assert best.failure_timing is None
    low, high = best.rule.thresholds["any other day"]
    assert low == pytest.approx(493.05, abs=0.01) and low == high
    textbook_policy = part.optimal_replacement_policy()
    assert best.cost_rate == pytest.approx(textbook_policy.cost_rate, rel=1e-7)
    assert best.cost == pytest.approx(best.cost_rate * 365.25)
    assert best.measure == "cost per year"
    assert "plain age replacement" in best.route_text


def test_a_shut_every_day_is_age_replacement():
    part = textbook()
    site = Shutdowns([Shutdown("day", every="1 days", replace_cost=1.0)])
    best = part.optimal_calendar_policy(site, unit="days")
    assert best.route == "exact" and best.solver == "grid"
    textbook_policy = part.optimal_replacement_policy()
    assert best.cost_rate == pytest.approx(textbook_policy.cost_rate, rel=1e-5)


def test_one_kind_every_t_days_is_the_calendar_policy_at_the_next_shut():
    part = textbook()
    site = Shutdowns(
        [Shutdown("monthly stop", every="30 days", replace_cost=1.0)]
    )
    best = part.optimal_calendar_policy(site, unit="days")
    old = part.optimal_calendar_policy(
        Calendar([Opportunity("m", 1.0, period=30.0)], anytime=None)
    )
    assert best.cost_rate == pytest.approx(old.cost_rate, rel=1e-12)
    assert best.rule.thresholds["monthly stop"][0] == pytest.approx(
        479.1, abs=0.05
    )
    assert best.route_text.startswith("Exact long-run average: $1.27 a year.")
    assert "repeats every 30 days" in best.route_text


def test_quadrature_agrees_with_the_grid():
    part = plant_part()
    grid = part.optimal_calendar_policy(QUARTERLY_ANNUAL, unit="days")
    exact = part.optimal_calendar_policy(
        QUARTERLY_ANNUAL, unit="days", failure_timing="exact"
    )
    assert grid.cost_rate == pytest.approx(0.0069430, abs=1e-7)
    assert exact.solver == "quadrature" and exact.failure_timing == "exact"
    assert exact.cost_rate == pytest.approx(grid.cost_rate, rel=1e-5)


def test_weekly_stops_on_weekday_shifts_repeat_in_operating_hours():
    part = NonRepairable(
        surv.Weibull.from_params([20_000.0, 2.5]), unplanned_cost=40.0
    )
    site = Shutdowns(
        [Shutdown("weekly stop", Recurrence.weekly("Sat"), replace_cost=3.0)],
        any_day=4.0,
    )
    plan = DutyPlan(Shifts("Mon-Fri", "06:00", "22:00"))
    route = site.route(part, plan=plan, now=NOW, unit="hours")
    assert route.route == "exact"
    assert "every 80 operating hours" in route.text
    old = part.optimal_calendar_policy(
        Calendar(
            [Opportunity("w", 3.0, period=80.0, offset=80.0)], anytime=4.0
        )
    )
    best = part.optimal_calendar_policy(site, plan=plan, now=NOW, unit="hours")
    assert best.cost_rate == pytest.approx(old.cost_rate, rel=1e-9)
    assert best.next_replacement.date.endswith(("-05-13",))
    assert best.cost == pytest.approx(
        best.cost_rate * 80 * 365.25 / 7, rel=1e-3
    )


def test_a_use_on_the_lifes_own_meter_is_exact_whatever_the_duty():
    gearbox = NonRepairable(
        surv.Weibull.from_params([180_000.0, 2.2]), unplanned_cost=40_000
    )
    depot = Shutdowns(
        [
            Shutdown(
                "service",
                every=Usage(15_000, "km"),
                also_every="6 months",
                replace_cost=9_000,
            )
        ],
        any_day=14_000,
    )
    steady = DutyPlan(Rate(400, per="day", meter="km"))
    rostered = DutyPlan(
        background=Rate(400, per="day", meter="km"),
        specific=[
            Rate(700, per="day", meter="km", from_=NOW, until="2026-11-30")
        ],
    )
    road = dict(now=NOW, readings={"km": 212_000}, unit="km")
    one = gearbox.optimal_calendar_policy(depot, plan=steady, **road)
    two = gearbox.optimal_calendar_policy(depot, plan=rostered, **road)
    assert one.route == two.route == "exact"
    assert one.cost_rate == pytest.approx(two.cost_rate, rel=1e-12)
    assert one.rule.thresholds["service"][0] == pytest.approx(89_000.0)
    # Only the dates differ.
    assert one.schedule("2027-12-31") != two.schedule("2027-12-31")


# ---------------------------------------------- route 2, a long cycle


def test_sampling_is_within_its_error_of_the_exact_long_cycle():
    part = NonRepairable(
        surv.Weibull.from_params([1000.0, 5.0]), unplanned_cost=40.0
    )
    site = Shutdowns(
        [
            Shutdown("weekly", every="7 days", replace_cost=3.0),
            Shutdown("quarterly", every="93 days", replace_cost=2.0),
            Shutdown("annual", every="400 days", replace_cost=1.0),
        ],
        any_day=4.0,
    )
    sampled = part.optimal_calendar_policy(
        site, unit="days", method="sampling", mc_samples=24, seed=2
    )
    assert sampled.route == "sampling" and sampled.cost_se > 0.0
    assert "(24 samples)" in sampled.route_text
    exact = part.optimal_calendar_policy(site, unit="days", method="exact")
    assert exact.cost == pytest.approx(1.2801, abs=1e-4)
    assert abs(sampled.cost - exact.cost) < 3.0 * sampled.cost_se


def test_a_long_cycle_is_refused_without_a_method():
    part = NonRepairable(
        surv.Weibull.from_params([300.0, 2.5]), unplanned_cost=40.0
    )
    site = Shutdowns(
        [
            Shutdown("weekly", every="7 days", replace_cost=3.0),
            Shutdown("quarterly", every="93 days", replace_cost=2.0),
            Shutdown("annual", every="400 days", replace_cost=1.0),
        ],
        any_day=4.0,
    )
    route = site.route(part, unit="days")
    assert route.route == "long_cycle" and route.solver is None
    with pytest.raises(ValueError, match="method='exact'.*method='sampling'"):
        part.optimal_calendar_policy(site, unit="days")
    assert site.route(part, unit="days", method="exact").route == (
        "exact_long_cycle"
    )


# --------------------------------------------------- route 3, averaged


def test_averaged_is_close_to_exact_for_one_kind_of_equal_gaps():
    part = textbook()
    site = Shutdowns(
        [Shutdown("monthly stop", every="30 days", replace_cost=1.0)]
    )
    exact = part.optimal_calendar_policy(site, unit="days")
    averaged = part.optimal_calendar_policy(
        site, unit="days", method="averaged"
    )
    assert averaged.route == "averaged" and averaged.solver == "closed form"
    # The averaged formula takes a random phase at each part; on a truly
    # repeating calendar a part fitted at a shut has a fixed one. For one
    # kind the two agree closely (README, UC-43, Learned).
    assert averaged.cost_rate == pytest.approx(exact.cost_rate, rel=5e-4)
    assert "assuming shuts fall evenly within their usual gaps" in (
        averaged.route_text
    )


def test_the_averaged_statement_lists_every_recurring_kind_with_its_gaps():
    site = Shutdowns(
        [
            Shutdown(
                "weekly stop", Recurrence.weekly("Sat"), replace_cost=3.0
            ),
            Shutdown("annual shut", ANNUAL, replace_cost=1.0),
            Shutdown(
                "turnaround",
                dates=["2027-03-03", "2028-09-09"],
                replace_cost=0.5,
            ),
        ],
        any_day=4.0,
    )
    part = plant_part()
    best = part.optimal_calendar_policy(
        site, unit="days", now=NOW, method="averaged"
    )
    # Averaging is for the recurring shuts' long run: the turnarounds are
    # fixed dates, used as dated, never averaged.
    assert best.route == "dated_then_long_run"
    assert best.long_run_route == "averaged"
    for words in (
        "Using the turnarounds on 3 Mar 2027 and 9 Sep 2028 as dated",
        "weekly stop: every 7 days",
        "annual shut: every 364 or 371 days, averaged",
        "Exact dates not used for the cost",
    ):
        assert words in best.route_text
    assert "one-offs" not in best.route_text


def test_the_averaged_formula_is_its_assumption_simulated():
    # Each part meets each kind's shuts at an equilibrium phase of its
    # gaps, independently: simulated plainly against the formula.
    kinds = [
        ("q", 2.0, np.array([84.0, 98.0]), np.array([3.0, 1.0])),
        ("a", 1.0, np.array([364.0, 371.0]), np.array([5.0, 2.0])),
    ]
    part = NonRepairable(LIFE, unplanned_cost=20.0)
    model = _Averaged(part, kinds, 4.0, 20.0)
    a, a0 = np.array([300.0, 150.0]), 700.0
    rate = model.rate(a, a0)
    rng = np.random.default_rng(1)
    n = 400_000
    life = LIFE.random(n) if hasattr(LIFE, "random") else None
    life = np.asarray(life, float).ravel()[:n]
    planned = np.full(n, a0)
    price = np.full(n, 4.0)
    for k, (_, cost, lengths, counts) in enumerate(kinds):
        weights = counts * lengths / np.dot(counts, lengths)
        gap = rng.choice(lengths, size=n, p=weights)
        when = a[k] + rng.uniform(0.0, 1.0, n) * gap
        sooner = when < planned
        planned[sooner], price[sooner] = when[sooner], cost
    failed = life < planned
    length = np.where(failed, life, planned)
    spend = np.where(failed, 20.0, price)
    estimate = spend.mean() / length.mean()
    error = estimate * 3e-3
    assert rate == pytest.approx(estimate, abs=3 * error)


# ---------------------------------------------------- route 4, horizon


def test_a_long_horizon_at_no_discount_tends_to_the_exact_rate():
    part = plant_part()
    exact = part.optimal_calendar_policy(
        QUARTERLY_ANNUAL, unit="days", now=NOW
    )
    twenty = part.optimal_calendar_policy(
        QUARTERLY_ANNUAL,
        unit="days",
        now=NOW,
        horizon="2046-11-16",
        discount=0,
    )
    forty = part.optimal_calendar_policy(
        QUARTERLY_ANNUAL,
        unit="days",
        now=NOW,
        horizon="2066-11-16",
        discount=0,
    )
    assert forty.measure == "net present cost" and forty.cost_rate is None
    per_year = (forty.cost - twenty.cost) / 20.0
    assert per_year == pytest.approx(exact.cost, rel=1e-2)
    assert forty.cost / 40.0 == pytest.approx(exact.cost, rel=2e-2)


def test_early_decisions_do_not_depend_on_a_horizon_past_five_years():
    part = plant_part()
    ten, twenty = (
        part.optimal_calendar_policy(
            QUARTERLY_ANNUAL,
            unit="days",
            now=NOW,
            horizon=end,
            discount=0.07,
        )
        for end in ("2036-11-16", "2046-11-16")
    )
    early = ten.schedule("2031-11-16")
    assert early and early == twenty.schedule("2031-11-16")
    first_years = [row for row in ten.rule.by_period[:5]]
    assert first_years == twenty.rule.by_period[:5]
    assert "depend on where the horizon ends" in ten.route_text


def test_the_horizon_by_quadrature_agrees_with_the_grid():
    part = plant_part()
    common = dict(unit="days", now=NOW, horizon="2036-11-16", discount=0.07)
    grid = part.optimal_calendar_policy(QUARTERLY_ANNUAL, **common)
    exact = part.optimal_calendar_policy(
        QUARTERLY_ANNUAL, failure_timing="exact", **common
    )
    assert exact.solver == "quadrature"
    assert exact.cost == pytest.approx(grid.cost, rel=1e-3)
    assert exact.next_replacement.date == grid.next_replacement.date


# ---------------------------------------------------------- one record


def test_one_record_moves_both_outages_and_opportunities():
    part = plant_part()

    def both(month):
        site = Shutdowns(
            [
                Shutdown(
                    "annual",
                    Recurrence.yearly(month=month, weekday="Tuesday", nth=1),
                    replace_cost=1.0,
                    duration="3d",
                ),
                Shutdown(
                    "quarterly",
                    every="91 days",
                    replace_cost=3.0,
                    duration="1d",
                ),
            ]
        )
        best = part.optimal_calendar_policy(
            site, unit="days", now=NOW, method="averaged"
        )
        outages = site.on_diagram(NOW, "2036-11-16", unit="days")
        return best, outages

    november, november_out = both("November")
    march, march_out = both("March")
    assert november_out["annual"]["times"][0] == 351.0
    assert march_out["annual"]["times"][0] == 106.0  # 2 March 2027
    assert november_out["annual"]["duration"] == 3.0
    assert march_out["quarterly"] == november_out["quarterly"]
    # The policy's opportunities moved with the rule.
    assert november.schedule("2030-12-31")[0][0].startswith("2027-11")
    assert march.schedule("2030-12-31")[0][0] == "2027-03-02"
    # A diagram takes both from the same record.
    rbd = RepairableRBD(
        [("s", "b"), ("b", "t")],
        {
            "b": {
                "reliability": LIFE,
                "repairability": "instant",
                "replace_cost": 20.0,
                "preventive": {"policy": "calendar", "calendar": march},
            }
        },
        shutdowns=march_out,
    )
    result = rbd.cost(t_simulation=3652.5, mc_samples=200, seed=1)
    assert result.mean / 10.0 == pytest.approx(march.cost, rel=0.05)


def test_a_policy_on_an_operating_plan_is_refused_by_a_diagram():
    part = NonRepairable(
        surv.Weibull.from_params([20_000.0, 2.5]), unplanned_cost=40.0
    )
    site = Shutdowns(
        [Shutdown("weekly stop", Recurrence.weekly("Sat"), replace_cost=3.0)],
        any_day=4.0,
    )
    plan = DutyPlan(Shifts("Mon-Fri", "06:00", "22:00"))
    best = part.optimal_calendar_policy(site, plan=plan, now=NOW, unit="hours")
    with pytest.raises(ValueError, match="operating plan"):
        RepairableRBD(
            [("s", "b"), ("b", "t")],
            {
                "b": {
                    "reliability": part.reliability,
                    "repairability": "instant",
                    "preventive": {"policy": "calendar", "calendar": best},
                }
            },
        )


# ------------------------------------------------- the part in service


def test_the_part_in_service_on_real_dates():
    gearbox = NonRepairable(
        surv.Weibull.from_params([180_000.0, 2.2]), unplanned_cost=40_000
    )
    depot = Shutdowns(
        [Shutdown("service", every=Usage(15_000, "km"), replace_cost=9_000)],
        any_day=14_000,
    )
    road = dict(
        plan=DutyPlan(Rate(400, per="day", meter="km")),
        now=NOW,
        readings={"km": 212_000},
        unit="km",
    )
    best = gearbox.optimal_calendar_policy(depot, age="28,000 km", **road)
    assert str(best.next_replacement) == (
        "At the service, 17 May 2027, part 101,000 old"
    )
    choice = gearbox.optimal_next_replacement(
        "120,000 km", site=depot, policy=best
    )
    assert choice.best == "At the service, 18 Dec 2026"
    kinds = [o.kind for o in choice.options]
    assert kinds == ["now", "shut", "shut", "any other day", "failure"]
    assert choice.cost_rate == pytest.approx(best.cost_rate)
    spend = gearbox.cost_distribution(
        until="2027-12-31", policy=best, budget=50_000
    )
    by_window = gearbox.cost_by_window(
        ["2027-06-30", "2027-12-31"], policy=best
    )
    assert spend.mean == pytest.approx(by_window.total.mean, rel=1e-6)
    with pytest.raises(ValueError, match="age.*km"):
        gearbox.optimal_next_replacement("120,000 h", site=depot, policy=best)


def test_compare_and_a_given_rule():
    part = plant_part()
    best = part.optimal_calendar_policy(QUARTERLY_ANNUAL, unit="days")
    names = [c.name for c in best.compare()]
    assert names == ["run to failure", "every quarterly", "every annual"]
    priced = part.calendar_policy_cost(
        QUARTERLY_ANNUAL, {"annual": 0.0}, unit="days"
    )
    old = part.calendar_policy_cost(
        Calendar(
            [
                Opportunity("q", 3.0, period=91.0, offset=91.0),
                Opportunity("annual", 1.0, period=364.0, offset=364.0),
            ],
            anytime=None,
        ),
        {"annual": 0.0},
    )
    assert isinstance(old, float)
    assert priced.cost_rate == pytest.approx(old, rel=1e-9)
    assert priced.measure == "cost per year" and priced.route == "exact"
    assert priced.replacements_by_kind["annual"] == pytest.approx(
        365.25 / 364.0, rel=1e-6
    )
    assert best.cost <= min(c.cost for c in best.compare())


def test_results_go_through_json():
    part = plant_part()
    for options in (
        {},
        {"method": "averaged"},
        {"horizon": "2030-11-16", "discount": 0.05},
    ):
        best = part.optimal_calendar_policy(
            QUARTERLY_ANNUAL, unit="days", now=NOW, **options
        )
        assert isinstance(best, ShutdownPolicy)
        data = json.loads(json.dumps(best.to_dict()))
        assert data["route"] == best.route
        assert "_engine" not in data


# ---------------------------------------------------- no silent choice


DATED = Shutdowns(
    [Shutdown("annual shut", ANNUAL, replace_cost=1.0)], any_day=4.0
)


@pytest.mark.parametrize(
    "options, match",
    [
        ({}, "don't repeat on a fixed cycle.*method='averaged'.*horizon="),
        ({"method": "exact"}, "no repeating cycle to be exact over"),
        ({"method": "sampling"}, "no repeating cycle to sample"),
        ({"horizon": "2036-11-16"}, "needs discount= too"),
        ({"discount": 0.07}, "discounted long run needs shuts that repeat"),
        (
            {"horizon": "2036-11-16", "discount": 0.07, "method": "averaged"},
            "its own route",
        ),
        (
            {"failure_timing": "exact", "method": "averaged"},
            "failure_timing='half_step'",
        ),
    ],
)
def test_every_choice_is_asked_for_not_made(options, match):
    with pytest.raises((ValueError, NotImplementedError), match=match):
        plant_part().optimal_calendar_policy(
            DATED, unit="days", now=NOW, **options
        )


def test_the_route_check_says_what_to_choose():
    route = DATED.route(plant_part(), unit="days", now=NOW)
    assert route.route == "horizon_or_averaged" and route.solver is None
    assert "the annual shut falls 364 or 371 days apart" in route.text
    averaged = DATED.route(
        plant_part(), unit="days", now=NOW, method="averaged"
    )
    assert averaged.solver == "closed form"
    assert averaged.estimate == "under a second"
    horizon = DATED.route(
        plant_part(), unit="days", now=NOW, horizon="2036-11-16", discount=0.07
    )
    assert horizon.route == "horizon" and horizon.solver == "grid"


@pytest.mark.parametrize(
    "site, options, match",
    [
        (DATED, {"unit": "days"}, "now= is needed"),
        (
            Shutdowns(
                [Shutdown("s", every=Usage(500, "starts"), replace_cost=1.0)]
            ),
            {"unit": "days", "now": NOW},
            "give plan=",
        ),
        (Shutdowns([], any_day=None), {"unit": "days"}, "no planned"),
        (
            Shutdowns([Shutdown("t", dates=["2027-03-03"], replace_cost=1.0)]),
            {"unit": "days", "now": NOW, "dated_until": "2027-01-01"},
            "before the last fixed date",
        ),
        (
            Shutdowns([Shutdown("t", dates=["2027-03-03"], replace_cost=1.0)]),
            {
                "unit": "days",
                "now": NOW,
                "dated_until": "2028-01-01",
                "horizon": "2030-01-01",
                "discount": 0.0,
            },
            "give one or the other",
        ),
        (
            Shutdowns([Shutdown("m", every="30 days", replace_cost=1.0)]),
            {},
            "give unit=",
        ),
        (
            Shutdowns(
                [Shutdown("m", every="30 days", replace_cost=1.0)],
                time_zone="Australia/Sydney",
            ),
            {
                "unit": "days",
                "plan": DutyPlan(
                    Shifts("Mon-Fri", "06:00", "22:00"),
                    time_zone="Australia/Brisbane",
                ),
                "now": NOW,
            },
            "same time zone",
        ),
    ],
)
def test_what_is_missing_is_asked_for(site, options, match):
    with pytest.raises(ValueError, match=match):
        plant_part().optimal_calendar_policy(site, **options)


def test_slow_quadrature_warns():
    from repyability.maintenance import _site_policy

    part = plant_part()
    old = _site_policy.SLOW_SECONDS
    _site_policy.SLOW_SECONDS = 0.0
    try:
        with pytest.warns(SlowSolverWarning, match="half_step"):
            part.optimal_calendar_policy(
                QUARTERLY_ANNUAL, unit="days", failure_timing="exact"
            )
    finally:
        _site_policy.SLOW_SECONDS = old


def test_the_model_time_calendar_is_unchanged():
    part = textbook()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # The 0.14 and 0.19 forms warn that they are deprecated.
        warnings.simplefilter("ignore", FutureWarning)
        policy = part.optimal_calendar_policy(Calendar())
    assert policy.cost_rate == pytest.approx(0.003462, abs=5e-7)
    with pytest.raises(ValueError, match="for a site"):
        part.optimal_calendar_policy(Calendar(), now=NOW)
    assert math.isfinite(
        part.calendar_policy_cost(Calendar(), {"anytime": 493.0})
    )


def test_sampling_and_a_rule_by_quadrature():
    part = plant_part()
    sampled = part.optimal_calendar_policy(
        QUARTERLY_ANNUAL,
        unit="days",
        method="sampling",
        failure_timing="exact",
        mc_samples=8,
        seed=1,
    )
    assert sampled.solver == "sampling" and sampled.failure_timing == "exact"
    assert sampled.cost_rate == pytest.approx(0.0069430, rel=1e-3)
    every_annual = part.calendar_policy_cost(
        QUARTERLY_ANNUAL, {"annual": 0.0}, unit="days", failure_timing="exact"
    )
    # UC-13: every annual shut costs 25.48 over ten years.
    assert every_annual.cost_rate * 3640 == pytest.approx(25.48, abs=0.01)
