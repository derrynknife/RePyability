"""The layered package of visits (``optimal_layered_visits``,
``layered_cost_rate``): checked against closed forms for exponential
lives, against a renewal function solved here by an independent
discretisation of the renewal equation, and against a seeded simulation
of the plan."""

import math

import numpy as np
import pytest
import surpyval as surv

from repyability import (
    NonRepairable,
    layered_cost_rate,
    optimal_layered_visits,
)

FILTER = dict(scale=90.0, shape=3.0, cp=20.0, cu=1000.0)
PUMP = dict(scale=700.0, shape=2.5, cp=400.0, cu=2500.0)


def part(spec):
    return NonRepairable(
        surv.Weibull.from_params([spec["scale"], spec["shape"]]),
        planned_cost=spec["cp"],
        unplanned_cost=spec["cu"],
    )


def renewal_function(spec, horizon, step=0.25):
    """M(t) on a grid, from the renewal equation M = F + M * dF, summed
    over the increments of F (first order in the step)."""
    t = np.arange(0.0, horizon + step, step)
    cdf = 1.0 - np.exp(-((t / spec["scale"]) ** spec["shape"]))
    increments = np.diff(cdf)
    m = np.zeros(len(t))
    for i in range(1, len(t)):
        m[i] = np.sum((1.0 + m[i - 1 :: -1][:i]) * increments[:i])
    return t, m


def block_rate(spec, t, m, period):
    i = int(round(period / (t[1] - t[0])))
    return (spec["cp"] + spec["cu"] * m[i]) / period


def test_exponential_lives_have_the_closed_form_cost_rate():
    light = NonRepairable(
        surv.Exponential.from_params([0.02]),
        planned_cost=3.0,
        unplanned_cost=40.0,
    )
    deep = NonRepairable(
        surv.Exponential.from_params([0.001]),
        planned_cost=90.0,
        unplanned_cost=500.0,
    )
    interval, ratio, visit = 30.0, 4, 7.0
    # Failures are a Poisson process: M(P) = lambda P.
    expected = (
        visit / interval
        + (3.0 + 40.0 * 0.02 * interval) / interval
        + (90.0 + 500.0 * 0.001 * ratio * interval) / (ratio * interval)
    )
    got = layered_cost_rate(light, deep, interval, ratio, visit_cost=visit)
    assert got == pytest.approx(expected, rel=1e-6)


def test_parts_that_do_not_wear_out_are_best_never_visited():
    light = NonRepairable(
        surv.Exponential.from_params([0.02]),
        planned_cost=3.0,
        unplanned_cost=40.0,
    )
    plan = optimal_layered_visits(light, None, visit_cost=1.0)
    assert math.isinf(plan.interval)
    assert plan.cost_rate == pytest.approx(0.02 * 40.0)


def test_one_layer_without_a_visit_cost_is_the_block_replacement_optimum():
    t, m = renewal_function(PUMP, 1500.0)
    periods = np.arange(40.0, 1400.0, 0.25)
    rates = np.array([block_rate(PUMP, t, m, p) for p in periods])
    best = optimal_layered_visits(None, part(PUMP))
    assert best.ratio == 1
    assert best.cost_rate == pytest.approx(rates.min(), rel=3e-3)
    assert best.interval == pytest.approx(periods[rates.argmin()], rel=0.03)


def test_the_optimum_is_the_cheapest_over_a_dense_independent_scan():
    visit = 50.0
    horizon = 1200.0
    tf, mf = renewal_function(FILTER, horizon)
    tp, mp = renewal_function(PUMP, horizon)
    plan = optimal_layered_visits(part(FILTER), part(PUMP), visit_cost=visit)

    def rate(interval, ratio):
        return (
            visit / interval
            + block_rate(FILTER, tf, mf, interval)
            + block_rate(PUMP, tp, mp, ratio * interval)
        )

    scan = {
        k: min(rate(T, k) for T in np.arange(10.0, 800.0 / k, 0.5))
        for k in range(1, 13)
    }
    best_k = min(scan, key=scan.get)
    assert plan.ratio == best_k
    assert plan.cost_rate == pytest.approx(scan[best_k], rel=4e-3)
    for choice in plan.by_ratio:
        assert choice.cost_rate == pytest.approx(scan[choice.ratio], rel=4e-3)
    assert plan.deep_interval == pytest.approx(plan.ratio * plan.interval)
    assert plan.cost_rate == pytest.approx(
        plan.visit_cost_rate + plan.light_cost_rate + plan.deep_cost_rate
    )


def test_the_cost_rate_matches_a_simulated_plan():
    interval, ratio, visit = 40.0, 5, 50.0
    exact = layered_cost_rate(
        part(FILTER), part(PUMP), interval, ratio, visit_cost=visit
    )
    rng = np.random.default_rng(21)
    cycles, cycle_length = 20_000, interval * ratio

    def block_cost(spec, period, blocks):
        """Total cost of ``blocks`` renewals of a part every ``period``:
        failures in between are replaced, the block replacement renews."""
        total = np.zeros(cycles)
        for _ in range(blocks):
            total += spec["cp"]
            clock = np.zeros(cycles)
            live = np.ones(cycles, bool)
            while live.any():
                life = spec["scale"] * rng.weibull(spec["shape"], cycles)
                clock = clock + life
                hit = live & (clock < period)
                total[hit] += spec["cu"]
                live = hit
        return total

    cost = (
        visit * ratio
        + block_cost(FILTER, interval, ratio)
        + block_cost(PUMP, cycle_length, 1)
    )
    rate = cost / cycle_length
    error = rate.std(ddof=1) / math.sqrt(cycles)
    assert abs(exact - rate.mean()) < 4 * error


def test_a_deeper_ratio_pays_when_the_deep_part_outlives_the_light_one():
    plan = optimal_layered_visits(part(FILTER), part(PUMP), visit_cost=50.0)
    assert plan.ratio > 1
    ratios = {c.ratio: c.cost_rate for c in plan.by_ratio}
    assert ratios[plan.ratio] < ratios[1]


def test_layer_and_input_checks():
    with pytest.raises(ValueError, match="at least one layer"):
        optimal_layered_visits(None, None)
    with pytest.raises(TypeError, match="NonRepairable"):
        optimal_layered_visits("filter", None)
    with pytest.raises(ValueError, match="visit_cost"):
        optimal_layered_visits(part(FILTER), None, visit_cost=-1.0)
    with pytest.raises(ValueError, match="interval"):
        layered_cost_rate(part(FILTER), None, 0.0, 1)
    unpriced = NonRepairable(surv.Weibull.from_params([90.0, 3.0]))
    with pytest.raises(ValueError):
        optimal_layered_visits(unpriced, None)
