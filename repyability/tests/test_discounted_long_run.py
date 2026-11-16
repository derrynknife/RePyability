"""The discounted long run (Fox, 1966): age replacement, a calendar of
opportunities in model time and a site's shuts on real dates, each
choosing the policy with the least equivalent annual cost (EAC) at a
discount rate, checked against independent workings: the present value
by quadrature of the density, a simulation of the discounted renewal
costs, a long finite-horizon discounted programme, and the
undiscounted answers the discounted ones come to as the rate goes to
0."""

import math
import warnings

import numpy as np
import pytest
import surpyval as surv
from scipy.integrate import quad

from repyability import (
    Calendar,
    NonRepairable,
    Opportunity,
    Shutdown,
    Shutdowns,
    StandbyModel,
)
from repyability.maintenance._replacement import _Ages, _Grid, _sweep
from repyability.schedule import DutyPlan, Recurrence, Shifts

# The 0.14 and 0.19 forms (Calendar, Shutdown, Shutdowns, prices on the
# opportunities, optimal_calendar_policy) still work, deprecated: these
# tests keep them unchanged, and test_opportunities.py the new forms.
pytestmark = pytest.mark.filterwarnings(
    "ignore:.*(is deprecated|puts a price on the opportunities):FutureWarning"
)

HOURS_A_YEAR = 8760.0


def bearing(cp=1.0, cu=5.0, life=None):
    unit = NonRepairable(life or surv.Weibull.from_params([1000, 2.5]))
    unit.set_costs_planned_and_unplanned(cp=cp, cu=cu)
    return unit


def a_year(rate):
    """A yearly rate as a continuous one an hour."""
    return math.log1p(rate) / HOURS_A_YEAR


def fox_eac(life, cp, cu, rate, age):
    """Fox's equivalent annual cost at ``age``, written out from the
    density by quadrature: ``r * [cu F_r(T) + cp D(T)] / [1 - F_r(T) -
    D(T)]``, ``F_r`` the discounted chance of failing first and ``D`` the
    discounted chance of reaching ``age``."""
    failed = quad(
        lambda t: math.exp(-rate * t) * float(np.ravel(life.df(t))[0]),
        0.0,
        age,
        limit=200,
    )[0]
    kept = math.exp(-rate * age) * float(np.ravel(life.sf(age))[0])
    return rate * (cu * failed + cp * kept) / (1.0 - failed - kept)


def fox_run_to_failure(life, cu, rate):
    """Running to failure's EAC: ``r cu L / (1 - L)``, ``L`` the Laplace
    transform of the life at ``rate``."""
    laplace = quad(
        lambda t: math.exp(-rate * t) * float(np.ravel(life.df(t))[0]),
        0.0,
        np.inf,
        limit=400,
    )[0]
    return rate * cu * laplace / (1.0 - laplace)


# ------------------------------------------------------- age replacement


def test_rate_zero_is_the_undiscounted_policy_unchanged():
    unit = bearing()
    plain = unit.optimal_replacement_policy()
    assert unit.optimal_replacement_policy(discount_rate=0.0) == plain
    assert unit.optimal_replacement_policy(discount_rate=0) == plain
    assert unit.find_optimal_replacement(discount_rate=0.0) == plain.interval
    assert plain.discount_rate == 0.0 and plain.npv == math.inf
    assert plain.eac == plain.cost_rate
    # The doctest's numbers, as before.
    assert round(plain.interval, 1) == 493.0
    assert round(plain.cost_rate, 6) == 0.003462


def test_the_discounted_age_minimises_fox_s_cost():
    unit = bearing()
    rate = 0.002  # a unit of the life's time: strong, to show the effect
    policy = unit.optimal_replacement_policy(discount_rate=rate)
    life = unit.reliability
    ages = np.linspace(300.0, 1200.0, 1801)
    costs = [fox_eac(life, 1.0, 5.0, rate, a) for a in ages]
    best = ages[int(np.argmin(costs))]
    assert policy.interval == pytest.approx(best, abs=0.6)
    assert policy.eac == pytest.approx(
        fox_eac(life, 1.0, 5.0, rate, policy.interval), rel=1e-7
    )
    assert policy.eac == pytest.approx(min(costs), rel=1e-6)
    assert policy.npv == pytest.approx(policy.eac / rate)
    # cost_rate keeps its meaning: the undiscounted rate at the age chosen.
    assert policy.cost_rate == pytest.approx(
        float(unit._cost_rate(policy.interval))
    )


def test_the_present_value_agrees_with_a_simulation():
    # Draw the renewal process and add up its discounted costs, to where
    # a cost is discounted to exp(-25).
    unit = bearing()
    rate = 0.002
    policy = unit.optimal_replacement_policy(discount_rate=rate)
    age = policy.interval
    rng = np.random.default_rng(7)
    n = 40_000
    clock = np.zeros(n)
    total = np.zeros(n)
    while clock.min() < 25.0 / rate:
        life = 1000.0 * rng.weibull(2.5, n)
        failed = life < age
        clock = clock + np.where(failed, life, age)
        total += np.exp(-rate * clock) * np.where(failed, 5.0, 1.0)
    error = total.std(ddof=1) / math.sqrt(n)
    assert abs(total.mean() - policy.npv) < 4.0 * error


@pytest.mark.parametrize("yearly", [0.05, 0.10])
def test_discounting_puts_the_replacement_off_for_a_part_that_wears_out(
    yearly,
):
    unit = bearing()
    plain = unit.find_optimal_replacement()
    found = unit.find_optimal_replacement(discount_rate=a_year(yearly))
    assert found > plain
    assert (
        unit.find_optimal_replacement(discount_rate=a_year(2 * yearly)) > found
    )
    days = bearing().find_optimal_replacement(
        discount_rate=math.log1p(yearly) / 365.0
    )
    assert days > found


def test_as_the_rate_goes_to_zero_it_comes_to_the_cost_rate_optimum():
    unit = bearing()
    plain = unit.optimal_replacement_policy()
    previous = math.inf
    for rate in (1e-4, 1e-5, 1e-6, 1e-8, 1e-10):
        policy = unit.optimal_replacement_policy(discount_rate=rate)
        gap = abs(policy.interval - plain.interval)
        assert gap <= previous + 1e-6
        previous = gap
        assert policy.eac < plain.cost_rate
    assert policy.interval == pytest.approx(plain.interval, rel=1e-6)
    assert policy.eac == pytest.approx(plain.cost_rate, rel=1e-6)


def test_discounting_can_make_running_to_failure_best():
    # cp close to cu: replacing pays undiscounted, but put off far enough
    # by a high rate, the failures are cheaper.
    unit = bearing(cp=4.0, cu=5.0)
    life = unit.reliability
    assert math.isfinite(unit.find_optimal_replacement())
    policy = unit.optimal_replacement_policy(discount_rate=3e-3)
    assert policy.interval == math.inf
    rtf = fox_run_to_failure(life, 5.0, 3e-3)
    assert policy.eac == pytest.approx(rtf, rel=1e-6)
    ages = np.linspace(200.0, 6000.0, 300)
    best = min(fox_eac(life, 4.0, 5.0, 3e-3, a) for a in ages)
    assert best >= rtf * (1 - 1e-9)
    # cost_rate is running to failure's undiscounted: cu / MTTF.
    assert policy.cost_rate == pytest.approx(5.0 / float(life.mean()))


@pytest.mark.parametrize(
    "life",
    [
        surv.Exponential.from_params([0.01]),
        surv.Weibull.from_params([1e3, 0.8]),
    ],
)
def test_a_life_that_does_not_wear_out_is_run_to_failure(life):
    unit = bearing(life=life)
    policy = unit.optimal_replacement_policy(discount_rate=1e-3)
    assert policy.interval == math.inf
    assert policy.eac == pytest.approx(
        fox_run_to_failure(life, 5.0, 1e-3), rel=1e-6
    )


def test_a_limited_failure_population_is_searched_when_discounted():
    # Undiscounted, running to failure is free in the long run; discounted
    # it is not, and the search says which wins.
    life = surv.Weibull.from_params([1000, 3.0], lfp_p=0.9)
    unit = bearing(life=life)
    assert unit.find_optimal_replacement() == math.inf
    policy = unit.optimal_replacement_policy(discount_rate=1e-3)
    sf = life.sf
    rtf = (
        1e-3
        * 5.0
        * (
            1.0
            - 1e-3
            * quad(
                lambda t: math.exp(-1e-3 * t) * float(np.ravel(sf(t))[0]),
                0.0,
                np.inf,
                limit=400,
            )[0]
        )
        / (
            1e-3
            * quad(
                lambda t: math.exp(-1e-3 * t) * float(np.ravel(sf(t))[0]),
                0.0,
                np.inf,
                limit=400,
            )[0]
        )
    )
    assert policy.eac <= rtf * (1 + 1e-6)
    if math.isfinite(policy.interval):
        assert policy.eac == pytest.approx(
            fox_eac(life, 1.0, 5.0, 1e-3, policy.interval), rel=1e-6
        )


def test_a_non_parametric_estimate_is_scanned_on_its_grid():
    # Failures at 10, 20, 30, 40: R(u) = 1 - u / 40 exactly.
    unit = bearing(life=surv.KaplanMeier.fit([10, 20, 30, 40]), cp=1, cu=4)
    rate = 0.02

    def eac(t):
        length = quad(lambda u: math.exp(-rate * u) * (1 - u / 40), 0, t)[0]
        failed = quad(lambda u: math.exp(-rate * u) / 40, 0, t)[0]
        kept = math.exp(-rate * t) * (1 - t / 40)
        return (4 * failed + kept) / length

    ages = np.linspace(0.01, 40.0, 4000)
    best = ages[int(np.argmin([eac(a) for a in ages]))]
    policy = unit.optimal_replacement_policy(discount_rate=rate)
    assert policy.interval == pytest.approx(best, abs=0.02)
    assert policy.eac == pytest.approx(eac(policy.interval), rel=1e-5)
    assert policy.interval > unit.find_optimal_replacement()


def test_a_standby_arrangement_is_discounted_too():
    # Two exponential units, cold standby: an Erlang-2 life, which wears
    # out.
    each = surv.Exponential.from_params([0.001])
    unit = bearing(life=StandbyModel([each, each]), cp=1.0, cu=10.0)
    rate = 5e-4

    def erlang_eac(t):
        lam = 0.001
        sf = lambda u: math.exp(-lam * u) * (1 + lam * u)  # noqa: E731
        length = quad(lambda u: math.exp(-rate * u) * sf(u), 0, t)[0]
        failed = quad(
            lambda u: math.exp(-rate * u) * lam**2 * u * math.exp(-lam * u),
            0,
            t,
        )[0]
        kept = math.exp(-rate * t) * sf(t)
        return (10.0 * failed + kept) / length

    ages = np.linspace(100.0, 3000.0, 2901)
    best = ages[int(np.argmin([erlang_eac(a) for a in ages]))]
    policy = unit.optimal_replacement_policy(discount_rate=rate)
    assert policy.interval == pytest.approx(best, rel=2e-3)
    assert policy.eac == pytest.approx(erlang_eac(policy.interval), rel=1e-4)


@pytest.mark.parametrize("rate", [-0.1, math.inf, "0.1", True])
def test_a_discount_rate_out_of_range_is_refused(rate):
    with pytest.raises(ValueError, match="discount_rate must be a number"):
        bearing().optimal_replacement_policy(discount_rate=rate)


# ------------------------------------------------- a calendar in model time

BEARING = NonRepairable(
    surv.Weibull.from_params([1000.0, 2.5]),
    planned_cost=1.0,
    unplanned_cost=5.0,
)
SHUTS = Calendar(
    [
        Opportunity("quarterly", 3.0, period=91.0),
        Opportunity("annual", 1.0, period=364.0),
    ],
    failure=20.0,
    anytime=None,
)
A_DAY_30 = math.log(1.3) / 364.0


@pytest.mark.parametrize("method", ["dp", "integration"])
def test_a_calendar_at_rate_zero_is_unchanged(method):
    plain = BEARING.optimal_calendar_policy(SHUTS, method=method)
    zero = BEARING.optimal_calendar_policy(
        SHUTS, method=method, discount_rate=0.0
    )
    assert zero == plain
    assert zero.npv == math.inf and zero.eac == plain.cost_rate


def test_a_calendar_with_only_any_day_is_discounted_age_replacement():
    policy = BEARING.optimal_calendar_policy(Calendar(), discount_rate=1e-3)
    age = BEARING.optimal_replacement_policy(discount_rate=1e-3)
    assert policy.eac == pytest.approx(age.eac, rel=1e-3)
    planned = policy.next_replacement(age=0.0).time
    assert planned == pytest.approx(age.interval, abs=2 * policy.step)
    assert policy.eac == pytest.approx(1e-3 * policy.npv)


def test_the_calendar_s_present_value_is_a_long_horizon_s():
    # A plain backward sweep over many cycles from nothing, discounted a
    # step at a time, comes to the long run's present value.
    policy = BEARING.optimal_calendar_policy(SHUTS, discount_rate=A_DAY_30)
    ages = _Ages(BEARING, policy.step, None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid = _Grid(SHUTS, None, policy.step)
    costs, _ = grid.prices(0, grid.cycle)
    cycles = int(math.ceil(40.0 / (A_DAY_30 * 364.0)))
    values, _, _ = _sweep(
        np.zeros(ages.size + 1),
        np.tile(costs, cycles),
        ages.p,
        20.0,
        math.exp(-A_DAY_30 * policy.step),
    )
    assert policy.npv == pytest.approx(values[0], rel=1e-9)
    low, high = policy.settings["npv_bounds"]
    assert low <= policy.npv <= high


def test_the_grid_and_the_quadrature_agree_discounted():
    dp = BEARING.optimal_calendar_policy(SHUTS, discount_rate=A_DAY_30)
    by_quadrature = BEARING.optimal_calendar_policy(
        SHUTS, method="integration", discount_rate=A_DAY_30
    )
    assert by_quadrature.npv == pytest.approx(dp.npv, rel=1e-3)
    assert by_quadrature._threshold(0.0) == pytest.approx(
        dp._threshold(0.0), abs=1.0
    )


def test_discounting_raises_the_calendar_s_thresholds():
    plain = BEARING.optimal_calendar_policy(SHUTS)
    policy = BEARING.optimal_calendar_policy(SHUTS, discount_rate=A_DAY_30)
    assert policy._threshold(0.0) > plain._threshold(0.0)
    assert policy.eac < policy.cost_rate
    # The long-run figures are the chosen rule's, undiscounted.
    rule = {"annual": policy._threshold(0.0)}
    assert policy.cost_rate == pytest.approx(
        BEARING.calendar_policy_cost(SHUTS, rule), rel=1e-6
    )


def test_a_calendar_s_rate_going_to_zero_comes_to_the_long_run_average():
    plain = BEARING.optimal_calendar_policy(SHUTS)
    policy = BEARING.optimal_calendar_policy(SHUTS, discount_rate=1e-9)
    assert policy.eac == pytest.approx(plain.cost_rate, rel=1e-6)
    assert policy.schedule() == plain.schedule()


@pytest.mark.parametrize(
    "calendar, options, match",
    [
        (
            SHUTS,
            {"method": "sampling"},
            "sampling.*no discount_rate.*horizon=",
        ),
        (
            Calendar([Opportunity("turnaround", 1.0, dates=[100.0, 400.0])]),
            {},
            "given dates.*horizon=",
        ),
        (
            Calendar(
                [
                    Opportunity("a", 1.0, period=7.0),
                    Opportunity("b", 1.0, period=93.0),
                    Opportunity("c", 1.0, period=401.0),
                ],
                anytime=None,
            ),
            {"method": "dp"},
            "too long to discount.*horizon=",
        ),
    ],
)
def test_a_calendar_not_discounted_exactly_is_refused(
    calendar, options, match
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(NotImplementedError, match=match):
            BEARING.optimal_calendar_policy(
                calendar, discount_rate=1e-3, **options
            )


def test_a_site_takes_discount_not_discount_rate():
    site = Shutdowns([], any_day=4.0)
    with pytest.raises(ValueError, match="a site takes discount="):
        BEARING.optimal_calendar_policy(site, discount_rate=1e-3)


# ---------------------------------------------- a site's shuts on dates

NOW = "2026-11-16"
PLANT_PART = NonRepairable(
    surv.Weibull.from_params([1000.0, 2.5]), unplanned_cost=20.0
)
STOP = Shutdown(
    "stop", Recurrence.every(4, "weeks", "2026-11-21"), replace_cost=3.0
)
TURNAROUND = Shutdown(
    "turnaround", dates=["2027-03-03", "2028-09-09"], replace_cost=0.5
)


def site_policy(site, **options):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return PLANT_PART.optimal_calendar_policy(
            site, unit="days", now=NOW, **options
        )


@pytest.mark.parametrize(
    "site", [Shutdowns([STOP]), Shutdowns([STOP, TURNAROUND])]
)
def test_a_site_at_zero_discount_is_unchanged(site):
    plain = site_policy(site)
    zero = site_policy(site, discount=0.0)
    assert zero.cost == plain.cost and zero.route_text == plain.route_text
    assert zero.discount is None and zero.measure == "cost per year"


def test_a_site_s_discounted_long_run_is_a_long_horizon_s():
    site = Shutdowns([STOP])
    policy = site_policy(site, discount=0.3)
    assert policy.route == "exact" and policy.discount == 0.3
    assert policy.measure == "equivalent annual cost"
    assert policy.cost == pytest.approx(math.log1p(0.3) * policy.npv)
    # The horizon route, its own programme on the real dates, to 50
    # years: what lies past it is discounted to about 2e-6.
    horizon = site_policy(site, horizon="2076-11-16", discount=0.3)
    assert horizon.cost == pytest.approx(policy.npv, rel=2e-4)
    plain = site_policy(site)
    assert policy.cost < plain.cost
    assert policy.rule.thresholds["stop"][0] > plain.rule.thresholds["stop"][0]
    assert policy.cost_rate == pytest.approx(plain.cost_rate, rel=1e-2)


def test_a_site_s_rate_going_to_zero_comes_to_the_long_run_average():
    site = Shutdowns([STOP])
    plain = site_policy(site)
    policy = site_policy(site, discount=1e-9)
    assert policy.cost == pytest.approx(plain.cost, rel=1e-6)
    assert policy.rule.thresholds == plain.rule.thresholds


def test_any_day_alone_is_discounted_age_replacement():
    site = Shutdowns([], any_day=4.0)
    part = NonRepairable(
        surv.Weibull.from_params([1000.0, 2.5]),
        planned_cost=4.0,
        unplanned_cost=20.0,
    )
    policy = part.optimal_calendar_policy(
        site, unit="days", now=NOW, discount=0.07
    )
    age = part.optimal_replacement_policy(
        discount_rate=math.log1p(0.07) / 365.25
    )
    assert policy.cost == pytest.approx(age.eac * 365.25, rel=1e-9)
    assert policy.rule.thresholds["any other day"][0] == pytest.approx(
        age.interval
    )


@pytest.mark.parametrize("failure_timing", ["half_step", "exact"])
def test_the_dated_window_is_discounted_with_the_long_run_after_it(
    failure_timing,
):
    site = Shutdowns([STOP, TURNAROUND])
    policy = site_policy(site, discount=0.3, failure_timing=failure_timing)
    assert policy.route == "dated_then_long_run"
    assert policy.discount == 0.3
    assert policy.measure == "equivalent annual cost"
    long_run = site_policy(
        Shutdowns([STOP]), discount=0.3, failure_timing=failure_timing
    )
    assert policy.cost == pytest.approx(long_run.cost)
    assert policy.rule.thresholds == long_run.rule.thresholds
    # The window's decisions are those of a long discounted horizon over
    # the same real dates.
    horizon = site_policy(
        site,
        horizon="2060-11-16",
        discount=0.3,
        failure_timing=failure_timing,
    )
    assert policy.schedule("2028-09-09") == horizon.schedule("2028-09-09")
    plain = site_policy(site, failure_timing=failure_timing)
    assert policy.dated_cost < plain.dated_cost
    assert "discounted long run at 30% a year" in policy.route_text


def test_a_dated_window_running_to_failure_after_it_is_discounted():
    policy = site_policy(Shutdowns([TURNAROUND]), discount=0.07)
    life = PLANT_PART.reliability
    rate = math.log1p(0.07) / 365.25
    assert policy.long_run_route == "run_to_failure"
    assert policy.cost == pytest.approx(
        fox_run_to_failure(life, 20.0, rate) * 365.25, rel=1e-6
    )


@pytest.mark.parametrize(
    "site, options, match",
    [
        (
            Shutdowns([STOP]),
            {"method": "averaged"},
            "long-run average, which is not discounted",
        ),
        (
            Shutdowns([STOP]),
            {"method": "sampling"},
            "long-run average, which is not discounted",
        ),
        (
            Shutdowns(
                [
                    Shutdown(
                        "annual",
                        Recurrence.yearly(
                            month="November", weekday="Tuesday", nth=1
                        ),
                        replace_cost=1.0,
                    )
                ]
            ),
            {},
            "discounted long run needs shuts that repeat",
        ),
        (Shutdowns([STOP]), {"discount": -0.1}, "yearly rate"),
    ],
)
def test_a_site_not_discounted_exactly_is_refused(site, options, match):
    options = {"discount": 0.07, **options}
    with pytest.raises(ValueError, match=match):
        site_policy(site, **options)


def test_a_duty_plan_with_a_discounted_long_run_is_refused():
    plan = DutyPlan(Shifts("Mon-Fri", "06:00", "22:00"))
    with pytest.raises(ValueError, match="steady pace"):
        site_policy(Shutdowns([STOP]), plan=plan, discount=0.07)


def test_compare_prices_each_rule_at_the_discounted_long_run():
    # Each simpler rule is priced as the policy is: its discounted long
    # run, exactly (its EAC), so none beats the policy's own.
    policy = site_policy(Shutdowns([STOP]), discount=0.07)
    rows = policy.compare()
    assert {r.measure for r in rows} == {"equivalent annual cost"}
    assert all(r.cost >= policy.cost * (1 - 1e-9) for r in rows)
    assert all(r.npv == pytest.approx(r.cost / math.log1p(0.07)) for r in rows)
