"""Exact by default; refuse what can't be done exactly; ``horizon=``
never refuses (UC-44, the maintainer's decision of 16 Nov 2026).

Checked against independent workings: the horizon's equivalent annual
cost against the exact discounted long run as the horizon lengthens;
each refusal's ready horizon run as given, and again doubled; a fixed
rule's discounted price against a plain backward sweep over many cycles
and a seeded simulation of its discounted costs; and the next
replacement's discounted choices against present values worked out by
quadrature from the density.
"""

import math
import re
import warnings
from datetime import date

import numpy as np
import pytest
import surpyval as surv
from scipy.integrate import quad

from repyability import NonRepairable, Opportunities, Opportunity
from repyability._version import __version__
from repyability.maintenance import _replacement as R
from repyability.schedule import DutyPlan, Recurrence, Shifts
from repyability.utils.deprecation import NEXT_REMOVAL

NOW = "2026-11-16"
A_DAY = math.log(1.07) / 365.25
LIFE = surv.Weibull.from_params([1000.0, 2.5])

# The 28-day stop: replace at a stop for 3, a failure 20, 7% a year.
PART = NonRepairable(LIFE, unplanned_cost=20.0, opportunity_cost=3.0)
STOPS = Opportunities([Opportunity("stop", period=28.0)], any_day=False)
STOPS_ON_DATES = Opportunities(
    [Opportunity("stop", Recurrence.every(4, "weeks", "2026-11-21"))],
    any_day=False,
)

# (a): the 7, 93 and 401-day shuts, any other day 4, failure 40.
WORN = NonRepairable(
    surv.Weibull.from_params([1000.0, 5.0]),
    planned_cost=4.0,
    unplanned_cost=40.0,
    opportunity_costs={"a": 3.0, "b": 2.0, "c": 1.0},
)
COPRIME = Opportunities(
    [
        Opportunity("a", period=7.0),
        Opportunity("b", period=93.0),
        Opportunity("c", period=401.0),
    ]
)
# (b), (c), (e): a part replaced at a shut for 1, any day 4, failure 20.
SHUT_PART = NonRepairable(
    LIFE, planned_cost=4.0, unplanned_cost=20.0, opportunity_cost=1.0
)
GIVEN = Opportunities([Opportunity("turnaround", dates=[100.0, 400.0])])
ANNUAL = Opportunities(
    [
        Opportunity(
            "annual shut",
            Recurrence.yearly(month="November", weekday="Tuesday", nth=1),
        )
    ],
    any_day=False,
)
# (d): weekday shifts, the bearing's life in hours.
SHIFTS = DutyPlan(Shifts("Mon-Fri", "06:00", "22:00"))
HOURS_PART = NonRepairable(
    surv.Weibull.from_params([20_000.0, 2.5]),
    planned_cost=4.0,
    unplanned_cost=40.0,
    opportunity_cost=1.0,
)


def model_horizon(message: str) -> float:
    found = re.search(r"horizon=(\d+(?:\.\d+)?)", message)
    assert found, message
    return float(found.group(1))


def site_horizon(message: str) -> str:
    found = re.search(r"horizon='(\d{4}-\d{2}-\d{2})'", message)
    assert found, message
    return found.group(1)


def doubled(day: str) -> str:
    now = date.fromisoformat(NOW)
    return (now + 2 * (date.fromisoformat(day) - now)).isoformat()


@pytest.fixture(scope="module")
def exact_28():
    return PART.optimal_opportunity_policy(STOPS, discount_rate=A_DAY)


# ------------------------------------------------ horizon against exact


def test_the_horizon_s_eac_comes_to_the_exact_one(exact_28):
    # The exact discounted long run: 4.216 a year, from 419.1 days.
    assert exact_28.eac * 365.25 == pytest.approx(4.21646, abs=1e-5)
    assert exact_28._threshold(0.0) == pytest.approx(419.125)
    gaps = []
    for years in (10, 50, 100, 307):
        policy = PART.optimal_opportunity_policy(
            STOPS, discount_rate=A_DAY, horizon=years * 365.25
        )
        assert policy.basis == "horizon"
        gaps.append(abs(policy.eac - exact_28.eac) / exact_28.eac)
        if years < 50:
            continue  # a part's life from the end: early decisions move
        # Its first decisions are the exact thresholds, within a step.
        for t in (0.0, 28.0, 420.0, 840.0):
            assert abs(policy._threshold(t) - exact_28._threshold(t)) <= (
                policy.step + 1e-9
            )
        assert policy.next_replacement(0.0) == exact_28.next_replacement(0.0)
    assert gaps == sorted(gaps, reverse=True)
    assert gaps[-1] < 1e-8
    # Its present value comes to the long run's too.
    assert policy.npv == pytest.approx(exact_28.npv, rel=1e-8)


def test_on_real_dates_the_horizon_comes_to_the_exact_one():
    exact = PART.optimal_opportunity_policy(
        STOPS_ON_DATES, unit="days", now=NOW, discount=0.07
    )
    assert exact.eac == exact.cost
    horizon = PART.optimal_opportunity_policy(
        STOPS_ON_DATES,
        unit="days",
        now=NOW,
        discount=0.07,
        horizon="2333-11-16",
    )
    # The horizon's grid is sized from the life alone (the stops laid at
    # its nearest steps), the exact route's put on the stops: the two
    # grids differ by about 1.5e-5.
    assert horizon.eac == pytest.approx(exact.cost, rel=5e-5)
    assert horizon.cost == pytest.approx(exact.npv, rel=5e-5)
    assert "equivalent annual cost" in horizon.route_text
    assert horizon.next_replacement == exact.next_replacement


def test_an_undiscounted_horizon_is_the_total_to_it():
    policy = PART.optimal_opportunity_policy(STOPS, horizon=36525.0)
    assert policy.discount_rate == 0.0
    # The total cost to the horizon, the expected costs of every step.
    # (The horizon rounded up to a whole step of the grid.)
    length = policy.settings["horizon"]
    assert 36525.0 <= length < 36525.0 + policy.step
    assert policy.npv == pytest.approx(policy.cost_rate * length, rel=1e-5)
    assert policy.eac == pytest.approx(policy.npv / length)
    plain = PART.optimal_opportunity_policy(STOPS)
    # Near the long-run rate, a little under it: the part in service at
    # the horizon is not paid for (about one cycle's cost over 100 years).
    assert plain.cost_rate * 0.99 < policy.eac < plain.cost_rate


def test_a_horizon_in_model_time_is_a_number():
    with pytest.raises(ValueError, match="in their own time"):
        PART.optimal_opportunity_policy(STOPS, horizon="2036-11-16")
    with pytest.raises(ValueError, match="its own route"):
        PART.optimal_opportunity_policy(
            STOPS, horizon=3650.0, method="sampling"
        )


# ------------------------------------------- the cases once refused


def test_a_periods_that_repeat_too_late_to_discount():
    with pytest.raises(NotImplementedError) as refused:
        WORN.optimal_opportunity_policy(COPRIME, discount_rate=A_DAY)
    message = str(refused.value)
    assert "too long to discount" in message
    assert "one part in a billion" in message
    horizon = model_horizon(message)
    assert horizon == 112_000.0
    policy = WORN.optimal_opportunity_policy(
        COPRIME, discount_rate=A_DAY, horizon=horizon
    )
    longer = WORN.optimal_opportunity_policy(
        COPRIME, discount_rate=A_DAY, horizon=2 * horizon
    )
    assert longer.npv == pytest.approx(policy.npv, rel=1e-8)
    assert longer.eac == pytest.approx(policy.eac, rel=1e-8)
    assert longer.next_replacement(0.0) == policy.next_replacement(0.0)


def test_b_given_dates_in_model_time_with_a_discount():
    with pytest.raises(NotImplementedError) as refused:
        SHUT_PART.optimal_opportunity_policy(GIVEN, discount_rate=A_DAY)
    message = str(refused.value)
    assert "given dates" in message and "nothing assumed" in message
    horizon = model_horizon(message)
    policy = SHUT_PART.optimal_opportunity_policy(
        GIVEN, discount_rate=A_DAY, horizon=horizon
    )
    longer = SHUT_PART.optimal_opportunity_policy(
        GIVEN, discount_rate=A_DAY, horizon=2 * horizon
    )
    assert longer.npv == pytest.approx(policy.npv, rel=1e-8)
    assert longer.schedule(until=500.0) == policy.schedule(until=500.0)


def test_c_a_shut_that_drifts_on_real_dates():
    with pytest.raises(ValueError) as refused:
        SHUT_PART.optimal_opportunity_policy(
            ANNUAL, unit="days", now=NOW, discount=0.07
        )
    message = str(refused.value)
    assert "364 or 371 days apart" in message
    horizon = site_horizon(message)
    assert horizon == "2333-11-16"
    assert "discount=0.07" in message and "one part in a billion" in message
    policy = SHUT_PART.optimal_opportunity_policy(
        ANNUAL, unit="days", now=NOW, discount=0.07, horizon=horizon
    )
    longer = SHUT_PART.optimal_opportunity_policy(
        ANNUAL, unit="days", now=NOW, discount=0.07, horizon=doubled(horizon)
    )
    assert policy.route == "horizon"
    assert longer.cost == pytest.approx(policy.cost, rel=1e-8)
    assert longer.eac == pytest.approx(policy.eac, rel=1e-8)
    assert longer.next_replacement == policy.next_replacement


def test_d_a_duty_plan_with_a_discount():
    with pytest.raises(ValueError) as refused:
        HOURS_PART.optimal_opportunity_policy(
            STOPS_ON_DATES, plan=SHIFTS, unit="hours", now=NOW, discount=0.07
        )
    message = str(refused.value)
    assert "steady pace" in message
    horizon = site_horizon(message)
    policy = HOURS_PART.optimal_opportunity_policy(
        STOPS_ON_DATES,
        plan=SHIFTS,
        unit="hours",
        now=NOW,
        discount=0.07,
        horizon=horizon,
    )
    longer = HOURS_PART.optimal_opportunity_policy(
        STOPS_ON_DATES,
        plan=SHIFTS,
        unit="hours",
        now=NOW,
        discount=0.07,
        horizon=doubled(horizon),
    )
    assert longer.cost == pytest.approx(policy.cost, rel=1e-8)
    assert longer.next_replacement == policy.next_replacement


def test_e_a_given_rule_at_a_discount_where_nothing_repeats():
    rule = {"annual shut": 400.0}
    with pytest.raises(ValueError) as refused:
        SHUT_PART.opportunity_policy_cost(
            ANNUAL, rule, unit="days", now=NOW, discount=0.07
        )
    horizon = site_horizon(str(refused.value))
    priced = SHUT_PART.opportunity_policy_cost(
        ANNUAL, rule, unit="days", now=NOW, discount=0.07, horizon=horizon
    )
    longer = SHUT_PART.opportunity_policy_cost(
        ANNUAL,
        rule,
        unit="days",
        now=NOW,
        discount=0.07,
        horizon=doubled(horizon),
    )
    assert priced.measure == "net present cost"
    assert longer.cost == pytest.approx(priced.cost, rel=1e-8)
    assert priced.eac == pytest.approx(priced.npv * 0.0676586, rel=1e-4)
    # And in model time: given dates refuse, the horizon prices them.
    with pytest.raises(NotImplementedError) as refused:
        SHUT_PART.opportunity_policy_cost(
            GIVEN, {"turnaround": 50.0}, discount_rate=A_DAY
        )
    horizon_t = model_horizon(str(refused.value))
    once = SHUT_PART.opportunity_policy_cost(
        GIVEN, {"turnaround": 50.0}, discount_rate=A_DAY, horizon=horizon_t
    )
    twice = SHUT_PART.opportunity_policy_cost(
        GIVEN,
        {"turnaround": 50.0},
        discount_rate=A_DAY,
        horizon=2 * horizon_t,
    )
    assert twice == pytest.approx(once, rel=1e-8)


# ------------------------------------------------ a fixed rule, priced


def test_the_optimiser_s_rule_prices_at_its_own_eac(exact_28):
    rule = {"stop": exact_28._threshold(0.0)}
    eac = PART.opportunity_policy_cost(STOPS, rule, discount_rate=A_DAY)
    assert eac == pytest.approx(exact_28.eac, rel=1e-9)
    on_dates = PART.optimal_opportunity_policy(
        STOPS_ON_DATES, unit="days", now=NOW, discount=0.07
    )
    priced = PART.opportunity_policy_cost(
        STOPS_ON_DATES,
        {"stop": on_dates.rule.thresholds["stop"][0]},
        unit="days",
        now=NOW,
        discount=0.07,
    )
    assert priced.measure == "equivalent annual cost"
    assert priced.cost == pytest.approx(on_dates.cost, rel=1e-9)
    assert priced.npv == pytest.approx(on_dates.npv, rel=1e-9)


def test_a_fixed_rule_s_eac_against_a_plain_backward_sweep():
    rule = {"stop": 300.0}
    eac = PART.opportunity_policy_cost(STOPS, rule, discount_rate=A_DAY)
    # A plain backward sweep with the decisions fixed, from nothing,
    # over enough cycles that what lies past them is discounted away.
    calendar, _ = __import__(
        "repyability.maintenance._opportunities", fromlist=["resolve"]
    ).resolve(PART, STOPS)
    step = R._resolve_step(None, calendar, PART)
    ages = R._Ages(PART, step, None)
    grid = R._Grid(calendar, None, step)
    n = int(math.ceil(150_000.0 / step))
    costs, first = R._rule_prices(grid, rule, 0, n)
    v, _, _ = R._sweep(
        np.zeros(ages.size + 1),
        costs,
        ages.p,
        20.0,
        math.exp(-A_DAY * step),
        fixed=first,
    )
    assert eac == pytest.approx(A_DAY * v[0], rel=1e-8)


def test_a_fixed_rule_s_present_value_against_a_simulation():
    # Replace at the first stop at which the part is 300 days old or
    # more, at 3; a failure renews it at once, at 20; every cost
    # discounted to now at 7% a year.
    rule = {"stop": 300.0}
    eac = PART.opportunity_policy_cost(STOPS, rule, discount_rate=A_DAY)
    rng = np.random.default_rng(20261116)
    paths = 20_000
    t = np.zeros(paths)
    total = np.zeros(paths)
    while True:
        alive = np.exp(-A_DAY * t) > 1e-10
        if not alive.any():
            break
        life = 1000.0 * rng.weibull(2.5, paths)
        due = np.ceil((t + 300.0) / 28.0 - 1e-12) * 28.0
        fails = t + life < due
        when = np.where(fails, t + life, due)
        price = np.where(fails, 20.0, 3.0)
        total += np.where(alive, price * np.exp(-A_DAY * when), 0.0)
        t = when
    npv = total.mean()
    se = total.std(ddof=1) / math.sqrt(paths)
    assert abs(eac / A_DAY - npv) < 4 * se


def test_a_fixed_rule_by_quadrature_discounted():
    policy = PART.optimal_opportunity_policy(
        STOPS_ON_DATES,
        unit="days",
        now=NOW,
        discount=0.07,
        failure_timing="exact",
    )
    priced = PART.opportunity_policy_cost(
        STOPS_ON_DATES,
        {"stop": policy.rule.thresholds["stop"][0]},
        unit="days",
        now=NOW,
        discount=0.07,
        failure_timing="exact",
    )
    assert priced.cost == pytest.approx(policy.cost, rel=1e-6)


def test_a_rule_whose_long_run_cannot_be_discounted_is_refused():
    with pytest.raises(NotImplementedError, match="horizon=") as refused:
        WORN.opportunity_policy_cost(COPRIME, {"c": 0.0}, discount_rate=A_DAY)
    assert "one part in a billion" in str(refused.value)


def test_rate_zero_prices_as_before():
    rule = {"stop": 300.0}
    assert PART.opportunity_policy_cost(
        STOPS, rule, discount_rate=0.0
    ) == PART.opportunity_policy_cost(STOPS, rule)


# ----------------------------------------- the next replacement


def brute_force(part, age, policy, u, price):
    """A choice's present value from the density, with every later
    replacement valued at the policy's present value ``V``, less ``V``:
    failures before ``u`` at the failure price, else ``price`` at ``u``,
    each discounted to when it falls."""
    r, value = policy.discount_rate, policy.npv
    life = part.reliability
    held = float(life.sf(age))

    def failing(t):
        return math.exp(-r * t) * float(life.df(age + t)) / held

    failed = quad(failing, 0.0, u, limit=400)[0]
    kept = (
        0.0
        if math.isinf(u)
        else math.exp(-r * u) * float(life.sf(age + u)) / held
    )
    return (part.cu + value) * failed + (price + value) * kept - value


def test_the_next_replacement_s_choices_are_discounted(exact_28):
    part = NonRepairable(
        LIFE, planned_cost=4.0, unplanned_cost=20.0, opportunity_cost=3.0
    )
    found = part.optimal_next_replacement(
        300.0,
        now_cost=20.0,
        scheduled_outages=[(20.0, 3.0), (48.0, 3.0), (300.0, 3.0)],
        policy=exact_28,
    )
    assert found.discount_rate == exact_28.discount_rate
    assert found.cost_rate == exact_28.eac
    for choice in found.choices[1:]:
        if choice.kind == "planned":
            continue
        assert choice.cost == pytest.approx(
            brute_force(part, 300.0, exact_28, choice.time, choice.price),
            rel=1e-6,
            abs=1e-9,
        )


def test_a_discounted_age_policy_s_planned_choice_against_brute_force():
    part = NonRepairable(LIFE, planned_cost=1.0, unplanned_cost=5.0)
    policy = part.optimal_replacement_policy(discount_rate=A_DAY)
    found = part.optimal_next_replacement(200.0, policy=policy)
    planned = next(c for c in found.choices if c.kind == "planned")
    assert planned.cost == pytest.approx(
        brute_force(part, 200.0, policy, planned.time, 1.0), rel=1e-6
    )
    # The best time is the discounted one: no time nearby is cheaper.
    for du in (-5.0, 5.0):
        assert (
            planned.cost
            <= brute_force(part, 200.0, policy, planned.time + du, 1.0) + 1e-12
        )
    # From the policy's own age replacing now is the best (Fox's
    # optimum, to the search's tolerance).
    assert found.replace_from_age == pytest.approx(policy.interval, rel=2e-3)


def test_the_next_replacement_at_rate_zero_is_unchanged():
    part = NonRepairable(
        surv.Weibull.from_params([365.0, 2.0]),
        planned_cost=150.0,
        unplanned_cost=400.0,
    )
    found = part.optimal_next_replacement(
        200.0, scheduled_outages=[(150.0, 100.0), (330.0, 100.0)]
    )
    assert found.discount_rate == 0.0
    assert round(found.replace_from_age, 1) == 297.6


def test_on_real_dates_the_choices_are_discounted_at_the_policy_s_rate():
    policy = PART.optimal_opportunity_policy(
        STOPS_ON_DATES, unit="days", now=NOW, discount=0.07
    )
    found = PART.optimal_next_replacement(
        300.0, site=STOPS_ON_DATES, unit="days", now=NOW, policy=policy
    )
    assert found.discount_rate == pytest.approx(A_DAY)
    assert "discounted at 7% a year" in found.valued_by


# ------------------------------------------ method="auto" sampling


def test_auto_sampling_warns_it_will_refuse_with_a_ready_horizon():
    with pytest.warns(FutureWarning, match=f"From {NEXT_REMOVAL}") as w:
        sampled = WORN.optimal_opportunity_policy(
            COPRIME, mc_samples=2, seed=1
        )
    assert sampled.method == "sampling"
    message = str(w[0].message)
    assert "method='sampling'" in message
    # The full cycle (261,051 days) and three life spans.
    assert model_horizon(message) == 266_000.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        WORN.optimal_opportunity_policy(
            COPRIME, method="sampling", mc_samples=2, seed=1
        )


def test_auto_sampling_refuses_in_0_21():
    # method="auto" sampling a long cycle warns (a FutureWarning) for
    # 0.20 and refuses from 0.21. This fails as a reminder once the
    # version reaches it: make "auto" refuse with its ready horizon.
    version = tuple(map(int, __version__.split(".")))[:2]
    assert version < tuple(map(int, NEXT_REMOVAL.split(".")))
