"""The valuations of simulated runs (``npv_distribution``,
``compare_npv``) take a unit's own retirement and salvage: the runs end
at the retirement, and each is credited the salvage at the end, against
the same runs valued by hand from a plan without them."""

import math

import numpy as np
import pytest
import surpyval as surv

from repyability import AgePolicy, Prices, RepairableRBD, UnitPlan
from repyability.rbd import routes

W = surv.Weibull.from_params([400.0, 2.0])
REPAIR = surv.LogNormal.from_params([1.0, 0.5])
HORIZON = 300.0
RATE = 1e-3
RUNS = dict(discount_rate=RATE, value_rate=1.0, mc_samples=200, seed=5)

PLAIN = UnitPlan(
    W,
    REPAIR,
    tasks=(AgePolicy(interval=250.0, cost=2.0, duration=REPAIR),),
    prices=Prices(repair=5.0, downtime=1.0, acquisition=20.0),
)


def with_money(**money) -> UnitPlan:
    from dataclasses import replace

    return replace(PLAIN, **money)


def test_a_salvage_is_credited_to_every_run_at_the_end():
    plan = with_money(salvage=70.0)
    runs = plan.npv_distribution(HORIZON, **RUNS)
    plain = PLAIN.npv_distribution(HORIZON, **RUNS)
    worth = 70.0 * math.exp(-RATE * HORIZON)
    assert runs.salvage == pytest.approx(worth, rel=1e-15)
    assert runs.undiscounted_salvage == 70.0
    assert runs.retire_at is None
    np.testing.assert_allclose(
        runs.samples, plain.samples + worth, rtol=0, atol=1e-9
    )
    assert runs.undiscounted_npv == pytest.approx(
        plain.undiscounted_npv + 70.0, rel=1e-13
    )
    # The same credit expected_npv gives.
    exact = plan.expected_npv(HORIZON, discount_rate=RATE, value_rate=1.0)
    assert runs.salvage == pytest.approx(exact.salvage, rel=1e-12)


def test_a_retirement_ends_the_runs_there():
    plan = with_money(retire_at=240.0, salvage=30.0, escalation=2e-4)
    runs = plan.npv_distribution(HORIZON, **RUNS)
    to_then = PLAIN.npv_distribution(
        240.0, escalation={UnitPlan.NODE: 2e-4}, **RUNS
    )
    # The salvage grows with the "salvage" stream alone, as expected_npv
    # credits it (the unit's own escalation is its costs').
    worth = 30.0 * math.exp(-RATE * 240.0)
    assert runs.retire_at == 240.0 and runs.horizon == 240.0
    assert runs.salvage == pytest.approx(worth, rel=1e-15)
    exact = plan.expected_npv(HORIZON, discount_rate=RATE, value_rate=1.0)
    assert runs.salvage == pytest.approx(exact.salvage, rel=1e-12)
    np.testing.assert_allclose(
        runs.samples, to_then.samples + worth, rtol=0, atol=1e-9
    )


def test_compare_npv_credits_each_plan_its_own():
    salvaged = with_money(salvage=70.0)
    rbd = salvaged.diagram()
    other = PLAIN.diagram()
    compared = rbd.compare_npv(other, HORIZON, **RUNS)
    worth = 70.0 * math.exp(-RATE * HORIZON)
    # The runs are matched, so each run differs by the salvage alone.
    np.testing.assert_allclose(compared.differences, worth, atol=1e-9)
    assert compared.this.salvage == pytest.approx(worth, rel=1e-15)
    assert compared.other.salvage == 0.0


def test_what_runs_still_refuse_says_what_to_give():
    by_age = with_money(salvage=lambda age: 100.0 - age / 10.0)
    with pytest.raises(NotImplementedError, match="salvage by age"):
        by_age.diagram().npv_distribution(HORIZON, **RUNS)
    route = by_age.diagram().analysis_routes()["npv_distribution"]
    assert route.route == routes.REFUSED
    assert "salvage by age" in route.reason
    # One unit of several retiring is expected_npv's alone.
    other = {"reliability": W, "repairability": REPAIR, "repair_cost": 3.0}
    several = RepairableRBD(
        [("s", "a"), ("a", "b"), ("b", "t")],
        {"a": with_money(retire_at=100.0), "b": other},
    )
    with pytest.raises(NotImplementedError, match="expected_npv"):
        several.npv_distribution(HORIZON, **RUNS)
    assert several.analysis_routes()["npv_distribution"].route == (
        routes.REFUSED
    )


# ------------------------------------------- switch times, logs, services


def _switching(**money):
    """A one-block diagram on a covariate schedule (as
    test_switch_time_diagram's), with a unit's own money."""
    rng = np.random.default_rng(3)
    load = np.repeat([1.0, 2.0], 40)
    x = 100.0 * rng.weibull(3.0, load.size) / load**2
    model = surv.WeibullPH.fit(x, Z=load[:, None])
    spec = {
        "reliability": model,
        "repairability": surv.Exponential.from_params([1.0]),
        "replace_cost": 80.0,
        "covariates": surv.StepSchedule.from_changepoints([0.0], [[1.0]]),
        **money,
    }
    return RepairableRBD([("s", "a"), ("a", "t")], {"a": spec})


def _search(rbd, horizon):
    return rbd.best_switch_time(
        {"a": [[1.5], [1.0]]},
        [1.5, 1.0],
        horizon,
        discount_rate=RATE,
        mc_samples=200,
        seed=11,
        scan=5,
    )


def test_best_switch_time_credits_every_candidate_its_own_end():
    own = _search(_switching(salvage=50.0, retire_at=150.0), 200.0)
    plain = _search(_switching(), 150.0)
    worth = 50.0 * math.exp(-RATE * 150.0)
    assert own.retire_at == 150.0 and own.horizon == 150.0
    assert own.salvage == pytest.approx(worth, rel=1e-15)
    assert own.undiscounted_salvage == 50.0
    # The same candidates, from the same seed: each run its own NPV plus
    # the salvage, and the same choice.
    assert own.times == plain.times
    np.testing.assert_allclose(
        own.npvs, np.array(plain.npvs) + worth, rtol=0, atol=1e-9
    )
    np.testing.assert_allclose(
        own.samples, plain.samples + worth, rtol=0, atol=1e-9
    )
    assert own.switch_time == plain.switch_time
    assert plain.salvage == 0.0 and plain.retire_at is None
    route = _switching(salvage=50.0).analysis_routes()["best_switch_time"]
    assert route.route == routes.SIMULATED


def test_plan_vs_actual_ends_at_its_own_retirement_with_its_salvage():
    from repyability import Timeline

    log = Timeline.from_outages([(50.0, 52.0)], 100.0)
    plan = with_money(retire_at=240.0, salvage=70.0)
    own = plan.plan_vs_actual(log, 100.0, HORIZON, discount_rate=RATE)
    plain = PLAIN.plan_vs_actual(log, 100.0, 240.0, discount_rate=RATE)
    worth = 70.0 * math.exp(-RATE * 240.0)
    assert own.horizon == 240.0
    assert own.plan_npv == pytest.approx(plain.plan_npv + worth, rel=1e-12)
    assert own.rest_npv == pytest.approx(plain.rest_npv + worth, rel=1e-12)
    assert own.revalued_npv == pytest.approx(
        plain.revalued_npv + worth, rel=1e-12
    )
    # What happened so far is the plan's without them.
    assert own.actual_so_far == pytest.approx(plain.actual_so_far)
    assert own.variance_so_far == pytest.approx(plain.variance_so_far)
    assert {"category": "salvage", "component": UnitPlan.NODE} in [
        {k: r[k] for k in ("category", "component")} for r in own.not_compared
    ]
    assert plan.analysis_routes()["plan_vs_actual"].route != routes.REFUSED
    with pytest.raises(ValueError, match="past the retirement on the plan"):
        with_money(retire_at=80.0).plan_vs_actual(log, 100.0, HORIZON)


def test_a_service_value_takes_its_plan_own_end():
    from repyability.maintenance import Restoration
    from repyability.schedule import (
        Recurrence,
        Services,
        Task,
        UsageForecast,
        task_value,
    )

    life = surv.Weibull.from_params([8760.0, 3.0])
    service = Task(
        "gearbox service",
        every=Recurrence.every(3, "months", "2027-04-01"),
        cost=800.0,
    )
    usage = UsageForecast.calendar("2027-01-01", "2028-01-01")

    def worth(plan):
        return task_value(
            service,
            Services(plan, Restoration(q=0.5)),
            usage,
            age=4000.0,
            discount=0.07,
            period="quarter",
            unit="hours",
            seed=5,
            mc_samples=500,
        )

    plain = UnitPlan(life, "instant", prices=Prices(repair=12000.0))
    # A salvage at the forecast's end: credited to both sides alike.
    from dataclasses import replace

    salvaged = worth(replace(plain, salvage=1000.0))
    base = worth(plain)
    credit = 1000.0 * 1.07 ** (-365 / 365.25)
    for side in ("with_task", "without_task"):
        assert salvaged.total[side] == pytest.approx(
            base.total[side] - credit, rel=1e-12
        )
    np.testing.assert_allclose(salvaged.value, base.value, atol=1e-9)
    # Retired half way: the October service is not done, and nothing
    # falls in the last quarter.
    retired = worth(replace(plain, retire_at=4380.0, salvage=1000.0))
    assert retired.total["count"] == 2
    assert retired.with_task[3] == 0.0 and retired.without_task[3] == 0.0
    assert any("are not done" in note for note in retired.notes)
