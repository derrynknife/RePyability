"""Replacement options compared from a state (``compare_replacement_options``),
a delay or a slipped task (``delay_effect``, ``cost_of_waiting_for_visit``,
``cost_of_missed_visit``, ``slack_effect``) and a part renewed at each
service (``renewal_forecast``).

The exact values are checked against closed forms for an exponential life
and against seeded simulations of the same plans (the standard error of
each simulated mean sets the tolerance), and the comparison against the
older ``skip_next_replacement``, which it must agree with."""

import math

import numpy as np
import pytest
import surpyval as surv

from repyability import NonRepairable, Opportunity

SCALE, SHAPE = 365.0, 2.0
CP, CU = 150.0, 400.0
DAY = math.log(1.08) / 365.25


def bearing():
    return NonRepairable(
        surv.Weibull.from_params([SCALE, SHAPE]),
        planned_cost=CP,
        unplanned_cost=CU,
    )


def draw_life(rng, size, age, scale=SCALE, shape=SHAPE):
    """A Weibull life given the part has reached ``age``: the time still to
    run, by inverting R(age + x) = u R(age)."""
    u = rng.random(size)
    survival = math.exp(-((age / scale) ** shape))
    return scale * (-np.log(u * survival)) ** (1.0 / shape) - age


def simulated_option(rng, n, age, time, price, horizon, rate):
    """The present value to ``horizon`` of a plan that replaces the part at
    ``time`` for ``price`` (never, if ``time`` is None): failures are
    replaced at once at CU, whether or not the planned work is done."""
    total = np.zeros(n)
    for stage_start, stage_end, start_age in (
        (0.0, horizon if time is None else time, age),
        *(() if time is None else ((time, horizon, 0.0),)),
    ):
        clock = np.full(n, stage_start)
        unit_age = np.full(n, start_age)
        live = np.ones(n, bool)
        while live.any():
            life = np.empty(n)
            life[live] = draw_life(rng, int(live.sum()), 0.0)
            if start_age > 0.0 and np.all(unit_age[live] == start_age):
                life[live] = draw_life(rng, int(live.sum()), start_age)
            failure = clock + life
            hit = live & (failure < stage_end)
            total[hit] += CU * np.exp(-rate * failure[hit])
            clock = np.where(hit, failure, clock)
            unit_age = np.where(hit, 0.0, unit_age)
            live = hit
            start_age = 0.0
    if time is not None:
        total += price * math.exp(-rate * time)
    return total


@pytest.mark.parametrize("rate", [0.0, 0.0005])
def test_options_match_simulated_plans(rate):
    age, due, every = 200.0, 30.0, 180.0
    result = bearing().compare_replacement_options(
        age,
        due,
        every,
        defer_by=[30.0],
        shuts={"shut": (90.0, 100.0)},
        discount_rate=rate,
    )
    horizon = due + every
    plans = {
        "replace": (due, CP),
        "now": (0.0, CP),
        "defer by 30": (due + 30.0, CP),
        "shut": (90.0, 100.0),
        "skip": (None, None),
    }
    assert [o.name for o in result.options] == list(plans)
    for option in result.options:
        time, price = plans[option.name]
        draws = simulated_option(
            np.random.default_rng(11), 200_000, age, time, price, horizon, rate
        )
        error = draws.std(ddof=1) / math.sqrt(len(draws))
        assert abs(option.cost - draws.mean()) < 4 * error, option.name


def test_the_comparison_agrees_with_skipping_the_next_replacement():
    part = bearing()
    for days in (1.0, 30.0, 90.0, 180.0):
        old = part.skip_next_replacement(
            180.0, resets=False, time_to_next=days, discount_rate=DAY
        )
        new = part.compare_replacement_options(
            0.0, days, 180.0, discount_rate=DAY
        )
        skip = new.options[-1]
        assert skip.kind == "skip"
        assert skip.saving == pytest.approx(old.npv, rel=1e-9)


def test_options_have_closed_forms_for_an_exponential_life():
    lam, rate = 0.004, 0.001
    part = NonRepairable(
        surv.Exponential.from_params([lam]),
        planned_cost=CP,
        unplanned_cost=CU,
    )

    def failures(t):
        # Failures of an exponential life with replacement at once are a
        # Poisson process of rate lam: PV = CU lam (1 - exp(-r t)) / r.
        return CU * lam * (1 - math.exp(-rate * t)) / rate

    # Memoryless: the age does not matter.
    due, every = 50.0, 120.0
    horizon = due + every
    result = part.compare_replacement_options(
        700.0, due, every, shuts=[(80.0, 90.0)], discount_rate=rate
    )
    by_name = {o.name: o.cost for o in result.options}
    assert by_name["skip"] == pytest.approx(failures(horizon), rel=1e-6)
    assert by_name["replace"] == pytest.approx(
        failures(due) + math.exp(-rate * due) * (CP + failures(horizon - due)),
        rel=1e-6,
    )
    assert by_name["shut 1"] == pytest.approx(
        failures(80.0)
        + math.exp(-rate * 80.0) * (90.0 + failures(horizon - 80.0)),
        rel=1e-6,
    )


def test_the_best_option_is_the_cheapest_and_savings_are_against_the_plan():
    result = bearing().compare_replacement_options(
        200.0, 30.0, 180.0, defer_by=[30.0], shuts={"shut": (90.0, 100.0)}
    )
    assert result.best.name == "shut"
    plan = result.options[0].cost
    for option in result.options:
        assert option.saving == pytest.approx(plan - option.cost)
    assert result.horizon == 210.0


def test_option_inputs_are_checked():
    part = bearing()
    with pytest.raises(ValueError, match="due_in"):
        part.compare_replacement_options(10.0, 200.0, 180.0)
    with pytest.raises(ValueError, match="defer_by"):
        part.compare_replacement_options(10.0, 30.0, 180.0, defer_by=[500.0])
    with pytest.raises(ValueError, match="past the following"):
        part.compare_replacement_options(
            10.0, 30.0, 180.0, shuts=[(500.0, 1.0)]
        )
    with pytest.raises(ValueError, match="planned_cost|costs"):
        NonRepairable(
            surv.Weibull.from_params([365.0, 2.0])
        ).compare_replacement_options(10.0, 30.0, 180.0)


# -- a delay -----------------------------------------------------------------


def test_a_delay_has_the_exponential_closed_form():
    lam = 0.004
    part = NonRepairable(
        surv.Exponential.from_params([lam]),
        planned_cost=CP,
        unplanned_cost=CU,
    )
    late = part.delay_effect(900.0, 10.0, 40.0)
    assert late.added_failure_probability == pytest.approx(
        math.exp(-lam * 10.0) - math.exp(-lam * 40.0), rel=1e-12
    )
    # cost(u) = CP + (CU - CP) (1 - exp(-lam u))
    assert late.added_cost == pytest.approx(
        (CU - CP) * (math.exp(-lam * 10.0) - math.exp(-lam * 40.0)),
        rel=1e-9,
    )
    assert late.delay == 30.0


@pytest.mark.parametrize("rate", [0.0, 0.001])
def test_a_delay_matches_a_simulated_cycle(rate):
    age, planned, done = 200.0, 10.0, 40.0
    late = bearing().delay_effect(age, planned, done, discount_rate=rate)
    life = draw_life(np.random.default_rng(5), 400_000, age)

    def cycle(until):
        failed = life < until
        return np.where(
            failed,
            CU * np.exp(-rate * life),
            CP * math.exp(-rate * until),
        )

    for until, exact in ((planned, late.cost_planned), (done, late.cost_done)):
        draws = cycle(until)
        error = draws.std(ddof=1) / math.sqrt(len(draws))
        assert abs(exact - draws.mean()) < 4 * error
    risk = (life < done).mean() - (life < planned).mean()
    assert late.added_failure_probability == pytest.approx(risk, abs=4e-3)


def test_a_missed_visit_is_weighted_by_the_chance_of_missing_it():
    part = bearing()
    sure = part.cost_of_missed_visit(200.0, 10.0, 40.0)
    some = part.cost_of_missed_visit(200.0, 10.0, 40.0, miss_probability=0.1)
    assert some.added_cost == sure.added_cost
    assert some.expected_added_cost == pytest.approx(0.1 * sure.added_cost)
    assert some.expected_added_failure_probability == pytest.approx(
        0.1 * sure.added_failure_probability
    )


def test_waiting_for_the_next_visit_rounds_the_due_time_up():
    part = bearing()
    weekly = Opportunity("weekly", 0.0, period=7.0)
    assert part.cost_of_waiting_for_visit(200.0, 10.0, weekly).done_in == 14.0
    # A visit exactly when it is due costs nothing.
    on_time = part.cost_of_waiting_for_visit(200.0, 14.0, weekly)
    assert on_time.delay == 0.0 and on_time.added_cost == 0.0
    listed = part.cost_of_waiting_for_visit(200.0, 10.0, [30.0, 12.0, 90.0])
    assert listed.done_in == 12.0
    dated = Opportunity("shuts", 0.0, dates=[5.0, 60.0])
    assert part.cost_of_waiting_for_visit(200.0, 10.0, dated).done_in == 60.0
    with pytest.raises(ValueError, match="no visit"):
        part.cost_of_waiting_for_visit(200.0, 100.0, dated)


def test_a_later_visit_adds_more_risk():
    part = bearing()
    added = [
        part.cost_of_waiting_for_visit(
            200.0, 10.0, Opportunity("v", 0.0, period=p)
        ).added_failure_probability
        for p in (3.0, 7.0, 30.0)
    ]
    assert added == sorted(added) and added[0] > 0.0


# -- a slipped task ----------------------------------------------------------


def simulated_rate(interval, n=600_000, seed=3):
    rng = np.random.default_rng(seed)
    life = draw_life(rng, n, 0.0)
    cost = np.where(life < interval, CU, CP)
    length = np.minimum(life, interval)
    rate = cost.mean() / length.mean()
    # The ratio estimator's standard error.
    resid = cost - rate * length
    error = resid.std(ddof=1) / math.sqrt(n) / length.mean()
    return rate, error


def test_a_slipped_task_costs_what_the_age_replacement_costs_later():
    part = bearing()
    effect = part.slack_effect(300.0, 60.0)
    on_time, error = simulated_rate(300.0)
    late, late_error = simulated_rate(360.0)
    assert abs(effect.cost_rate_on_time - on_time) < 4 * error
    assert abs(effect.cost_rate_at_limit - late) < 4 * late_error
    assert effect.added_risk == pytest.approx(
        math.exp(-((300.0 / SCALE) ** SHAPE))
        - math.exp(-((360.0 / SCALE) ** SHAPE))
    )
    assert effect.added_cost_rate > 0.0
    between = sorted([effect.cost_rate_on_time, effect.cost_rate_at_limit])
    assert between[0] <= effect.mean_cost_rate_late <= between[1] + 1e-3


def test_no_slack_adds_nothing():
    effect = bearing().slack_effect(300.0, 0.0)
    assert effect.added_risk == 0.0 and effect.added_cost_rate == 0.0
    assert effect.mean_cost_rate_late == effect.cost_rate_on_time


# -- a part renewed at each service ------------------------------------------


def test_the_seal_renewed_at_every_service():
    seal = NonRepairable(
        surv.Weibull.from_params([6000.0, 2.2]),
        planned_cost=200.0,
        unplanned_cost=900.0,
    )
    plan = seal.renewal_forecast(944.0, 500.0, resets=False, age=340.0)

    def sf(t):
        return seal.reliability_function(np.array([t]))

    assert plan.service_times == [160.0, 660.0]
    assert plan.risk_before_service[0] == pytest.approx(
        1 - float(sf(500.0)[0]) / float(sf(340.0)[0]), rel=1e-12
    )
    assert plan.risk_before_service[1] == pytest.approx(
        1 - float(sf(500.0)[0]), rel=1e-12
    )
    assert plan.expected_renewals == pytest.approx(2.0)
    assert plan.expected_cost == pytest.approx(
        200.0 * 2 + 900.0 * plan.expected_failures, rel=1e-9
    )
    assert plan.expected_failures == pytest.approx(0.00785, abs=5e-5)
    assert plan.cost_rate == pytest.approx(0.4076, abs=5e-4)
    assert plan.best_interval == pytest.approx(3179.0, abs=5.0)
    assert plan.best_cost_rate == pytest.approx(0.1198, abs=5e-4)
    assert not plan.worth_it


def simulated_failures(resets, n=400_000, seed=8):
    rng = np.random.default_rng(seed)
    scale, shape, every, window = 6000.0, 2.2, 500.0, 944.0
    counts = np.zeros(n)
    clock = np.zeros(n)
    age = np.full(n, 340.0)
    live = np.ones(n, bool)
    # Services at 160, 660 (fixed), or each interval from the last renewal.
    next_service = np.full(n, 160.0)
    while live.any():
        u = rng.random(n)
        start = np.exp(-((age / scale) ** shape))
        life = scale * (-np.log(u * start)) ** (1.0 / shape) - age
        failure = clock + life
        service = next_service
        fails = live & (failure < np.minimum(service, window))
        counts[fails] += 1
        serviced = live & ~fails & (service < window)
        clock = np.where(fails, failure, np.where(serviced, service, clock))
        age = np.where(fails | serviced, 0.0, age)
        if resets:
            next_service = np.where(
                fails | serviced, clock + every, next_service
            )
        else:
            next_service = np.where(
                serviced, next_service + every, next_service
            )
        live = fails | serviced
    return counts


@pytest.mark.parametrize("resets", [False, True])
def test_the_forecast_matches_a_simulated_plan(resets):
    seal = NonRepairable(
        surv.Weibull.from_params([6000.0, 2.2]),
        planned_cost=200.0,
        unplanned_cost=900.0,
    )
    plan = seal.renewal_forecast(944.0, 500.0, resets=resets, age=340.0)
    counts = simulated_failures(resets)
    error = counts.std(ddof=1) / math.sqrt(len(counts))
    assert abs(plan.expected_failures - counts.mean()) < 4 * error


def test_a_part_that_does_not_wear_out_is_not_worth_renewing():
    part = NonRepairable(
        surv.Exponential.from_params([0.001]),
        planned_cost=100.0,
        unplanned_cost=400.0,
    )
    plan = part.renewal_forecast(500.0, 200.0, resets=True, age=50.0)
    assert not plan.worth_it
    assert math.isinf(plan.best_interval)


def test_the_forecast_inputs_are_checked():
    part = bearing()
    with pytest.raises(TypeError, match="resets"):
        part.renewal_forecast(100.0, 50.0, resets=1)
    with pytest.raises(ValueError, match="since_renewal"):
        part.renewal_forecast(100.0, 50.0, resets=True, since_renewal=10.0)
    with pytest.raises(ValueError, match="into an interval"):
        part.renewal_forecast(
            100.0, 50.0, resets=False, age=80.0, since_renewal=60.0
        )
