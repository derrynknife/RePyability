"""An upgraded component simulated (UC-52): a life that starts at or
after the upgrade's time is the upgrade's, from the same uniform as the
old life would have taken it, and the capital cost is charged then.

The references are a plain replacement at the upgrade's time (an
upgrade to the same life at no cost is one, draw for draw), memoryless
units either side of the upgrade in closed form, and the numerical
values over a horizon (`mission_availability`, `expected_cost`,
`expected_npv`).
"""

import math

import numpy as np
import pytest
import surpyval as surv

from repyability import NodeState, RepairableRBD
from repyability.rbd import _compiled, _event_loop
from repyability.tests import catalogue

E = surv.Exponential.from_params
W = surv.Weibull.from_params
EDGES = [("s", "a"), ("a", "t")]
#: Standard errors a simulated estimate may be from the exact value.
Z = 4.0


def one(life, repair=None, **more):
    return RepairableRBD(
        EDGES,
        {
            "a": {
                "reliability": life,
                "repairability": E([0.05]) if repair is None else repair,
                "repair_cost": 6.0,
                **more,
            }
        },
    )


def plain(**more):
    return dict(control_variate=False, conditional=False, **more)


@pytest.mark.parametrize("repair", [None, "instant"])
def test_an_upgrade_to_the_same_life_at_no_cost_is_a_replacement(repair):
    life = W([300, 2.0])
    upgraded = one(life, repair, upgrade={"at": 200.0, "reliability": life})
    replaced = one(life, repair, preventive={"replace_at": [200.0]})
    for state in (None, {"a": NodeState(age=120.0)}):
        mine = upgraded.availability(
            900.0, mc_samples=2000, seed=5, state=state, **plain()
        )
        theirs = replaced.availability(
            900.0, mc_samples=2000, seed=5, state=state, **plain()
        )
        assert (
            mine.mean_availability_interval().estimate
            == theirs.mean_availability_interval().estimate
        )
        np.testing.assert_array_equal(mine.cost.samples, theirs.cost.samples)


def test_memoryless_units_either_side_of_the_upgrade():
    # Before the upgrade a two-state chain at rate lam; a unit up at it is
    # replaced in no time (memoryless, nothing changes), and from it the
    # chain is at the new rate from where it was.
    lam, new, mu, at, T = 1 / 100.0, 1 / 400.0, 1 / 10.0, 300.0, 800.0

    def mean_up(rate, up, length):
        total = rate + mu
        steady = mu / total
        return (
            steady * length
            + (up - steady) * -math.expm1(-total * length) / total
        )

    def up_at(rate, up, s):
        total = rate + mu
        return mu / total + (up - mu / total) * math.exp(-total * s)

    expected = (
        mean_up(lam, 1.0, at) + mean_up(new, up_at(lam, 1.0, at), T - at)
    ) / T
    rbd = one(
        E([lam]),
        E([mu]),
        upgrade={"at": at, "reliability": E([new]), "cost": 25.0},
    )
    assert rbd.mission_availability(T) == pytest.approx(expected, abs=2e-6)
    run = rbd.availability(T, mc_samples=20_000, seed=21, **plain())
    interval = run.mean_availability_interval()
    assert abs(interval.estimate - expected) <= Z * interval.standard_error
    # The capital cost in every simulation that reaches it, none before.
    assert run.cost.by_category["capital"] == pytest.approx(25.0)
    short = rbd.availability(at, mc_samples=200, seed=21, **plain())
    assert short.cost.by_category["capital"] == 0.0


def test_the_simulation_agrees_with_the_numerical_values():
    rbd = catalogue.repairable_kinds()["upgraded at a time"]
    T, rate = 1000.0, 1e-3
    run = rbd.availability(T, mc_samples=20_000, seed=3, **plain())
    interval = run.mean_availability_interval()
    exact = rbd.mission_availability(T)
    assert abs(interval.estimate - exact) <= Z * interval.standard_error
    cost = rbd.expected_cost(T)
    assert abs(run.cost.sample_mean - cost.mean) <= Z * run.cost.sample_se
    assert cost.by_category["capital"] == pytest.approx(50.0)
    # From a state: the unit in service ages towards the upgrade.
    state = {"a": NodeState(age=200.0)}
    run = rbd.availability(
        T, mc_samples=20_000, seed=4, state=state, **plain()
    )
    interval = run.mean_availability_interval()
    exact = rbd.mission_availability(T, state=state)
    assert abs(interval.estimate - exact) <= Z * interval.standard_error
    # The NPV's spread, and its mean against the expected NPV.
    spread = rbd.npv_distribution(
        T, discount_rate=rate, value_rate=2.0, mc_samples=20_000, seed=9
    )
    expected = rbd.expected_npv(T, discount_rate=rate, value_rate=2.0)
    se = float(np.std(spread.samples, ddof=1)) / math.sqrt(spread.samples.size)
    assert np.std(spread.samples) > 0.0
    assert abs(spread.mean - expected.npv) <= Z * se


def test_an_upgrade_is_compared_run_by_run_with_keeping_the_old_life():
    life = W([300, 2.0])
    kept = one(life)
    upgraded = one(
        life, upgrade={"at": 200.0, "reliability": W([900, 2.0]), "cost": 40.0}
    )
    T, rate = 1500.0, 1e-3
    comparison = upgraded.compare_npv(
        kept, T, discount_rate=rate, value_rate=3.0, mc_samples=10_000, seed=2
    )
    expected = (
        upgraded.expected_npv(T, discount_rate=rate, value_rate=3.0).npv
        - kept.expected_npv(T, discount_rate=rate, value_rate=3.0).npv
    )
    interval = comparison.interval
    assert abs(comparison.mean - expected) <= Z * interval.standard_error
    # Matched runs (the old and the new life from the same uniforms) pin
    # the difference down better than independent ones (0.51 against
    # 0.94 here).
    assert interval.standard_error < comparison.independent_standard_error


def test_the_compiled_engine_leaves_an_upgrade_to_python():
    rbd = catalogue.repairable_kinds()["upgraded at a time"]
    plan = _event_loop._stream_plan(rbd, 100.0, 0, False)[0]
    assert _compiled.unsupported(rbd, plan, None, numba=True) == "an upgrade"
    route = rbd.analysis_routes()["availability"]
    assert route.engine == "python"
