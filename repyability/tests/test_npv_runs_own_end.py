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
    # One unit of several retiring is not built (expected_npv's refusal).
    other = {"reliability": W, "repairability": REPAIR, "repair_cost": 3.0}
    several = RepairableRBD(
        [("s", "a"), ("a", "b"), ("b", "t")],
        {"a": with_money(retire_at=100.0), "b": other},
    )
    with pytest.raises(NotImplementedError, match="retiring one unit"):
        several.npv_distribution(HORIZON, **RUNS)
    assert several.analysis_routes()["npv_distribution"].route == (
        routes.REFUSED
    )
