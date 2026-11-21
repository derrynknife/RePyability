"""A value schedule and escalation in the simulated NPV
(``npv_distribution``, ``compare_npv``, ``best_switch_time``) and in
``plan_vs_actual``: each run priced against a reference written here from
its history, the runs' mean against ``expected_npv`` with the same
arguments, the real/nominal identity, and a plan against what happened
adding up (and averaging to the plan, the tower property)."""

import json
import math
import warnings

import numpy as np
import pytest
import surpyval as surv

from repyability import RepairableRBD, Timeline, ValueSchedule, value_plan
from repyability.tests.test_npv_distribution import age_replaced, pump

E = surv.Exponential.from_params
W = surv.Weibull.from_params
LINE = [("s", "p"), ("p", "t")]
H, R = 400.0, 2e-3
SCHEDULE = ValueSchedule([0.0, 150.0], [3.0, 5.0])
GROWTH = {"value": 1e-3, "repair": 3e-3, "system_downtime": -1e-3}


def span(a, b, r):
    """The integral of exp(-r t) from a to b."""
    if r == 0.0:
        return b - a
    return (math.exp(-r * a) - math.exp(-r * b)) / r


def by_hand(timeline, steps, rate, growth):
    """A history's value earned, written out: each up span, cut at each
    step, at the step's rate grown at ``growth`` and discounted."""
    times = [0.0] + list(timeline.changes) + [timeline.end]
    up = timeline.up
    total = 0.0
    for a, b in zip(times[:-1], times[1:]):
        if up:
            for lo, hi, worth in steps:
                x, y = max(a, lo), min(b, hi)
                if y > x:
                    total += worth * span(x, y, rate - growth)
        up = not up
    return total


def test_each_run_is_priced_as_written_out():
    rbd = pump()
    n = 40
    runs = rbd.npv_distribution(
        H,
        discount_rate=R,
        value_rate=SCHEDULE,
        escalation=GROWTH,
        mc_samples=n,
        seed=2,
    )
    sim = rbd.simulate_timelines(H, mc_samples=n, seed=2)
    steps = [(0.0, 150.0, 3.0), (150.0, H, 5.0)]
    for i in range(n):
        value = by_hand(sim.system[i], steps, R, GROWTH["value"])
        pump_log = sim.components["p"][i]
        failures = pump_log.changes[0::2]  # it starts up
        cost = 2000.0 + sum(
            500.0 * math.exp(-(R - GROWTH["repair"]) * t) for t in failures
        )
        assert runs.value_samples[i] == pytest.approx(value, rel=1e-12)
        assert runs.cost_samples[i] == pytest.approx(cost, rel=1e-12)
    assert math.isnan(runs.value_rate)
    assert runs.value_schedule == SCHEDULE
    assert runs.escalation == GROWTH and runs.rate_basis == "nominal"


@pytest.mark.parametrize("make", [pump, age_replaced])
def test_the_mean_is_the_expected_npv_with_the_same_arguments(make):
    rbd = make()
    kwargs = dict(discount_rate=R, value_rate=SCHEDULE, escalation=GROWTH)
    runs = rbd.npv_distribution(H, mc_samples=4000, seed=3, **kwargs)
    exact = rbd.expected_npv(H, **kwargs)
    interval = runs.mean_interval(0.9999)
    assert interval.lower < exact.npv < interval.upper
    for category, amount in exact.cost_by_category.items():
        assert runs.cost_by_category.get(category, 0.0) == pytest.approx(
            amount, rel=0.1, abs=1e-9
        ), category


def test_one_step_and_no_growth_are_the_rate_alone():
    rbd = age_replaced()
    plain = rbd.npv_distribution(
        H, discount_rate=R, value_rate=3.0, mc_samples=50, seed=1
    )
    stepped = rbd.npv_distribution(
        H,
        discount_rate=R,
        value_rate=ValueSchedule([0.0], [3.0]),
        escalation={},
        mc_samples=50,
        seed=1,
    )
    np.testing.assert_array_equal(plain.samples, stepped.samples)


def test_inflation_leaves_the_present_values_and_lifts_the_rest():
    rbd = age_replaced()
    kwargs = dict(
        discount_rate=R,
        value_rate=SCHEDULE,
        escalation=GROWTH,
        rates="real",
        mc_samples=50,
        seed=1,
    )
    real = rbd.npv_distribution(H, **kwargs)
    lifted = rbd.npv_distribution(H, inflation=1e-3, **kwargs)
    np.testing.assert_allclose(lifted.samples, real.samples, rtol=1e-12)
    assert lifted.undiscounted_cost > real.undiscounted_cost
    assert lifted.inflation == 1e-3 and lifted.rate_basis == "real"
    # No discounting and no growth: the undiscounted value is the
    # present one.
    flat = rbd.npv_distribution(H, value_rate=SCHEDULE, mc_samples=50, seed=1)
    assert flat.undiscounted_npv == pytest.approx(flat.mean, rel=1e-12)


def test_two_plans_take_one_escalation():
    # The escalation names a component only the other plan has.
    one = pump()
    two = RepairableRBD(
        [("s", "p"), ("p", "q"), ("q", "t")],
        {
            "p": dict(one._init_args["components"]["p"]),
            "q": {
                "reliability": E([1e-4]),
                "repairability": E([0.1]),
                "repair_cost": 50.0,
            },
        },
    )
    kwargs = dict(
        discount_rate=R,
        value_rate=SCHEDULE,
        escalation={"q": 1e-3, "value": 1e-3},
        mc_samples=200,
        seed=4,
    )
    gain = one.compare_npv(two, H, **kwargs)
    assert gain.this.escalation == {"value": 1e-3}
    assert gain.other.escalation == {"q": 1e-3, "value": 1e-3}
    # One that names a component of neither plan is refused.
    with pytest.raises(ValueError, match="'q', which is no component"):
        one.compare_npv(pump(), H, **kwargs)
    same = one.compare_npv(
        pump(), H, **{**kwargs, "escalation": {"value": 1e-3}}
    )
    np.testing.assert_array_equal(same.differences, np.zeros(200))


def pumped():
    """A pump that all but never fails: its plan's rest is its rest."""
    return RepairableRBD(
        LINE,
        {
            "p": {
                "reliability": E([1e-9]),
                "repairability": E([0.1]),
                "repair_cost": 500.0,
            }
        },
        downtime_cost_rate=10.0,
    )


def test_the_rest_from_now_is_grown_and_discounted_to_the_start():
    rbd, now = pumped(), 100.0
    kwargs = dict(discount_rate=R, value_rate=SCHEDULE, escalation=GROWTH)
    result = rbd.plan_vs_actual(
        {"p": Timeline([], end=now)}, now, H, periods=[now, H], **kwargs
    )
    assert (
        result.plan.npv == rbd.expected_npv(H, periods=[now, H], **kwargs).npv
    )
    # The rest is valued with the step at 150 at 50 on its own clock.
    assert result.rest.value_schedule == ValueSchedule([0.0, 50.0], [3.0, 5.0])
    # Up throughout: the value from now is the plan's after now. (Its
    # downtime is not: known up now, the rest starts with none.)
    assert result.rest_npv == pytest.approx(result.plan_rest_npv, rel=1e-6)
    by_row = {(r["period"], r["category"]): r for r in result.plan.cash_flows}
    for row in result.rest_cash_flows:
        if row["category"] != "value":
            continue
        planned = by_row[(row["period"], row["category"])]
        assert row["present_value"] == pytest.approx(
            planned["present_value"], rel=1e-6, abs=1e-9
        )
        assert row["undiscounted"] == pytest.approx(
            planned["undiscounted"], rel=1e-6, abs=1e-9
        )
    assert result.revalued_npv == pytest.approx(
        result.actual_so_far + result.as_planned_so_far + result.rest_npv,
        rel=1e-12,
    )
    value = by_hand(
        Timeline([], end=now), [(0.0, 150.0, 3.0)], R, GROWTH["value"]
    )
    assert result.actual_so_far == pytest.approx(value, rel=1e-12)


def test_so_far_and_the_rest_average_to_the_plan():
    # The tower property, with a schedule stepping inside the rest and
    # growing streams: what each history yielded plus the rest valued from
    # its state now averages to the plan's value.
    rbd, now, n = age_replaced(), 150.0, 60
    kwargs = dict(
        discount_rate=R,
        value_rate=ValueSchedule([0.0, 200.0], [3.0, 5.0]),
        escalation=GROWTH,
    )
    plan = rbd.expected_npv(300.0, periods=[now, 300.0], **kwargs)
    sim = rbd.simulate_timelines(now, mc_samples=n, seed=11)
    whole = np.zeros(n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(n):
            logs = {node: sim.components[node][i] for node in rbd.components}
            whole[i] = rbd.plan_vs_actual(
                logs, now, 300.0, periods=[now, 300.0], **kwargs
            ).revalued_npv
    se = whole.std(ddof=1) / math.sqrt(n)
    assert abs(whole.mean() - plan.npv) < 4.0 * se


def test_the_best_switch_time_takes_the_basis():
    rng = np.random.default_rng(3)
    load = np.repeat([1.0, 2.0], 40)
    x = 100.0 * rng.weibull(3.0, load.size) / load**2
    rbd = RepairableRBD(
        LINE,
        {
            "p": {
                "reliability": surv.WeibullPH.fit(x, Z=load[:, None]),
                "repairability": E([0.5]),
                "repair_cost": 5.0,
                "covariates": surv.StepSchedule.from_changepoints(
                    [0.0], [[1.0]]
                ),
            }
        },
    )
    kwargs = dict(
        discount_rate=R,
        mc_samples=50,
        seed=1,
        scan=3,
        escalation={"value": 1e-3},
        rates="real",
    )
    real = rbd.best_switch_time(
        {"p": [[1.5], [1.0]]}, [1.5, 1.0], 100.0, **kwargs
    )
    lifted = rbd.best_switch_time(
        {"p": [[1.5], [1.0]]}, [1.5, 1.0], 100.0, inflation=1e-3, **kwargs
    )
    np.testing.assert_allclose(lifted.npvs, real.npvs, rtol=1e-12)
    assert lifted.escalation == {"value": 1e-3}
    assert lifted.inflation == 1e-3 and lifted.rate_basis == "real"


def test_it_is_asked_and_answered_as_json():
    rbd = age_replaced()
    answer = value_plan(
        {
            "schema": 1,
            "diagram": rbd.to_dict(),
            "analysis": "npv_distribution",
            "inputs": {
                "horizon": H,
                "discount_rate": R,
                "value_rate": SCHEDULE.to_dict(),
                "escalation": GROWTH,
                "mc_samples": 20,
                "seed": 1,
            },
        }
    )
    direct = rbd.npv_distribution(
        H,
        discount_rate=R,
        value_rate=SCHEDULE,
        escalation=GROWTH,
        mc_samples=20,
        seed=1,
    )
    assert answer["result"]["mean"] == direct.mean
    assert answer["result"]["value_schedule"] == SCHEDULE.to_dict()
    json.dumps(answer, allow_nan=False)
