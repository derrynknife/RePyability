"""Planned against unplanned downtime, and the failures a plan avoids
(UC-25, ``RepairableRBD.maintenance_effect``).

The system is down unplanned when it would still be down with every
planned task, test and shut in progress finished; planned otherwise. The
checks: closed forms worked by hand (a block under age replacement, a
parallel pair, a whole-system shut, a timed test of a hidden failure),
the exact values against the simulation, both engines to the last bit,
the parts adding up to the downtime in every result, and the failures
avoided against the direct difference of two diagrams.
"""

import math
import warnings

import numpy as np
import pytest
import surpyval as surv
from scipy.integrate import quad

from repyability import MaintenanceEffect, NodeState, RepairableRBD
from repyability.rbd import _compiled, _unplanned

W = surv.Weibull.from_params
E = surv.Exponential.from_params
FIXED = surv.ExactEventTime.from_params
ONE = [("s", "p"), ("p", "t")]
PAIR = [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")]
YEAR = 8760.0

needs_numba = pytest.mark.skipif(
    not _compiled.available(), reason="numba is not installed"
)


def pump(**more):
    """A pump replaced at 600 hours of age (8 hours, fixed), repaired in
    about 20 hours when it fails."""
    return {
        "reliability": W([1000.0, 3.0]),
        "repairability": E([1 / 20.0]),
        "preventive": {"interval": 600.0, "duration": FIXED([8.0])},
        **more,
    }


def by_hand():
    """A pump's long-run probabilities of being in a repair and in its
    maintenance: renewal-reward over its cycle (up to a failure or its
    age, then a repair or the maintenance)."""
    up, _ = quad(lambda t: math.exp(-((t / 1000.0) ** 3)), 0.0, 600.0)
    fails = -math.expm1(-(0.6**3))
    cycle = up + fails * 20.0 + (1.0 - fails) * 8.0
    return fails * 20.0 / cycle, (1.0 - fails) * 8.0 / cycle


# -- by hand ------------------------------------------------------------------


def test_one_block_under_age_replacement_by_hand():
    failed, planned = by_hand()
    rbd = RepairableRBD(ONE, {"p": pump()}, downtime_cost_rate=100.0)
    effect = rbd.maintenance_effect()
    assert effect.window is None and effect.method == "exact"
    # Down for its maintenance, E[PM time] / E[cycle]: planned; in a
    # repair, E[repair time] / E[cycle]: unplanned.
    assert effect.system_planned_downtime == pytest.approx(planned, 1e-9)
    assert effect.system_unplanned_downtime == pytest.approx(failed, 1e-9)
    assert effect.node_unplanned_downtime["p"] == pytest.approx(failed, 1e-9)
    assert effect.lost_production_planned == pytest.approx(
        100.0 * planned, 1e-9
    )
    # Over a long window, the same shares of the time.
    window = rbd.expected_events(50 * YEAR)
    assert window.system_planned_downtime / (50 * YEAR) == pytest.approx(
        planned, 2e-3
    )
    assert window.system_unplanned_downtime / (50 * YEAR) == pytest.approx(
        failed, 2e-3
    )


def test_a_parallel_pair_is_down_planned_unless_both_have_failed():
    # Each pump in a repair with probability f and in its maintenance with
    # probability m, independently: the pair is down with (f + m) ** 2,
    # unplanned only with both failed, f ** 2. One in its maintenance and
    # the other failed is planned: with the maintenance over it would be
    # up (though the change that took it down was a failure).
    f, m = by_hand()
    rbd = RepairableRBD(PAIR, {"a": pump(), "b": pump()})
    effect = rbd.maintenance_effect()
    assert effect.system_downtime == pytest.approx((f + m) ** 2, 1e-9)
    assert effect.system_unplanned_downtime == pytest.approx(f**2, 1e-9)
    assert effect.system_planned_downtime == pytest.approx(
        (f + m) ** 2 - f**2, 1e-9
    )


def test_one_in_maintenance_and_the_other_failed_is_planned():
    # Held so: b failed throughout, a maintained every 50 for 10. The
    # system is down exactly while a is in its maintenance, all of it
    # planned, though a failure (b's) is what keeps it down then.
    a = {
        "reliability": W([1e12, 1.0]),
        "repairability": "instant",
        "preventive": {
            "interval": 50.0,
            "policy": "block",
            "duration": FIXED([10.0]),
        },
    }
    b = {"reliability": E([0.01]), "repairability": E([1.0])}
    rbd = RepairableRBD(PAIR, {"a": a, "b": b})
    events = rbd.expected_events(500.0, broken_nodes=["b"])
    assert events.system_downtime == pytest.approx(90.0, 1e-6)
    assert events.system_planned_downtime == pytest.approx(90.0, 1e-6)
    assert events.system_unplanned_downtime == pytest.approx(0.0, abs=1e-6)
    run = rbd.availability(
        500.0, broken_nodes=["b"], mc_samples=50, seed=1, engine="python"
    )
    assert run.system_planned_downtime == pytest.approx(50 * 90.0, 1e-9)
    assert run.system_unplanned_downtime == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("idle_ageing", [True, False])
def test_a_periodic_shut_of_the_system_by_hand(idle_ageing):
    # One exponential block, the line shut every 100 for 5: in its
    # long-run state the block has failed with probability q = lam / (lam
    # + mu) at every time, in a shut or not (ageing through it, or
    # standing still). Unplanned: q; planned: the shut's share of the
    # time when the block has not failed, D / P (1 - q).
    lam, mu = 0.01, 0.5
    q = lam / (lam + mu)
    rbd = RepairableRBD(
        ONE,
        {"p": {"reliability": E([lam]), "repairability": E([mu])}},
        shutdowns={
            "line": {
                "period": 100.0,
                "duration": 5.0,
                "idle_ageing": idle_ageing,
            }
        },
    )
    effect = rbd.maintenance_effect()
    assert effect.system_unplanned_downtime == pytest.approx(q, 1e-9)
    assert effect.system_planned_downtime == pytest.approx(
        0.05 * (1.0 - q), 1e-9
    )
    # The simulation (in Python, over the windows) agrees.
    n, t = 400, 5000.0
    run = rbd.availability(t, mc_samples=n, seed=2, control_variate=False)
    down = t - np.asarray(run.uptimes)
    error = math.sqrt(np.mean(down**2) / n) / t
    events = rbd.expected_events(t)
    assert abs(
        run.system_unplanned_downtime / (n * t)
        - events.system_unplanned_downtime / t
    ) <= (4 * error)
    assert abs(
        run.system_planned_downtime / (n * t)
        - events.system_planned_downtime / t
    ) <= (4 * error)


def test_a_timed_test_of_a_hidden_failure_by_hand():
    # Tested every T, each test taking a working unit off line for D (it
    # does not age then); a failure found is repaired in R, the new unit
    # in service D + R after its test. Working at a test with probability
    # w = w e^{-lam (T - D)} + (1 - w) e^{-lam (T - D - R)}, its tests
    # of a working unit are planned, a share w D / T of the time.
    lam, T, D, R = 1e-3, 100.0, 2.0, 5.0
    kept, back = math.exp(-lam * (T - D)), math.exp(-lam * (T - D - R))
    w = back / (1.0 - kept + back)
    rbd = RepairableRBD(
        ONE,
        {
            "p": {
                "reliability": E([lam]),
                "repairability": FIXED([R]),
                "inspection": {"interval": T, "duration": FIXED([D])},
            }
        },
    )
    effect = rbd.maintenance_effect()
    assert effect.system_planned_downtime == pytest.approx(w * D / T, 1e-6)
    assert (
        effect.system_planned_downtime + effect.system_unplanned_downtime
        == pytest.approx(rbd.mean_unavailability(), 1e-12)
    )


# -- exact against simulated, and the engines -------------------------------


def nested(crews=None):
    """A pair of pumps in a nested RBD, in series with a third, each under
    age replacement taking time: the nested RBD down planned when its
    pumps are down and either is in its maintenance."""
    inner = RepairableRBD(PAIR, {"a": pump(), "b": pump()}, repair_crews=crews)
    return RepairableRBD(
        [("s", "n"), ("n", "c"), ("c", "t")],
        {"n": inner, "c": pump(reliability=W([3000.0, 2.0]))},
        downtime_cost_rate=10.0,
    )


def test_a_nested_rbd_is_split_as_its_own_structure_says():
    rbd = nested()
    t, n = 3000.0, 1000
    events = rbd.expected_events(t)
    run = rbd.availability(t, mc_samples=n, seed=3, control_variate=False)
    down = t - np.asarray(run.uptimes)
    error = math.sqrt(np.mean(down**2) / n)
    for exact, simulated in (
        (events.system_planned_downtime, run.system_planned_downtime / n),
        (events.system_unplanned_downtime, run.system_unplanned_downtime / n),
    ):
        assert abs(exact - simulated) <= 4 * error
    # The nested node's own split, by its own structure.
    inner = rbd.components["n"].expected_events(t)
    assert events.node_unplanned_downtime["n"] == pytest.approx(
        inner.system_unplanned_downtime, 1e-6
    )


@needs_numba
@pytest.mark.parametrize("crews", [None, 1])
def test_the_engines_split_alike(crews):
    rbd = nested(crews)
    runs = [
        rbd.availability(2000.0, mc_samples=300, seed=4, engine=engine)
        for engine in ("python", "numba")
    ]
    assert runs[0].system_planned_downtime > 0.0
    assert runs[0].system_unplanned_downtime > 0.0
    for name in ("system_planned_downtime", "system_unplanned_downtime"):
        assert getattr(runs[0], name) == getattr(runs[1], name)
    assert (
        runs[0].cost.lost_production_planned
        == runs[1].cost.lost_production_planned
    )


def test_the_parts_add_up_to_the_downtime():
    rbd = nested()
    run = rbd.availability(1500.0, mc_samples=200, seed=5)
    assert run.system_planned_downtime + run.system_unplanned_downtime == (
        pytest.approx(run.system_downtime, rel=1e-12)
    )
    assert (
        run.cost.lost_production_planned + run.cost.lost_production_unplanned
        == pytest.approx(10.0 * run.system_downtime / 200, rel=1e-12)
    )
    events = rbd.expected_events([500.0, 1500.0])
    np.testing.assert_allclose(
        events.system_planned_downtime + events.system_unplanned_downtime,
        events.system_downtime,
        rtol=1e-12,
    )
    for node in rbd.components:
        np.testing.assert_allclose(
            events.node_planned_downtime[node]
            + events.node_unplanned_downtime[node],
            events.node_downtime[node],
            rtol=1e-12,
        )
    windows = rbd.expected_events_by_window(width=500.0, count=3)
    np.testing.assert_allclose(
        windows.system_planned_downtime + windows.system_unplanned_downtime,
        windows.system_downtime,
        rtol=1e-9,
        atol=1e-12,
    )
    for effect in (
        rbd.maintenance_effect(),
        rbd.maintenance_effect(1500.0),
        rbd.maintenance_effect(1500.0, method="simulate", mc_samples=100),
    ):
        assert (
            effect.system_planned_downtime + effect.system_unplanned_downtime
            == pytest.approx(effect.system_downtime, rel=1e-12)
        )


def test_without_planned_outages_all_downtime_is_unplanned():
    rbd = RepairableRBD(
        PAIR,
        {
            name: {"reliability": W([100.0, 2.0]), "repairability": E([0.5])}
            for name in "ab"
        },
    )
    assert not _unplanned.takes_planned_outages(rbd)
    run = rbd.availability(500.0, mc_samples=50, seed=1)
    assert run.system_planned_downtime == 0.0
    assert run.system_unplanned_downtime == run.system_downtime
    events = rbd.expected_events(500.0)
    assert events.system_planned_downtime == 0.0
    assert events.node_unplanned_downtime == events.node_downtime


def test_with_crews_and_planned_outages_the_split_is_simulated():
    # A crew's chain takes no scheduled maintenance; with maintenance in
    # a pair a crew could, but would not tell it from a repair.
    rbd = nested(crews=1)
    assert rbd.analysis_routes()["maintenance_effect"].route == "simulated"
    with pytest.raises(NotImplementedError):
        rbd.maintenance_effect(500.0, method="exact")
    effect = rbd.maintenance_effect(500.0, mc_samples=50, seed=1)
    assert effect.method == "simulated" and effect.mc_samples == 50
    assert effect.node_unplanned_downtime is None
    with pytest.raises(NotImplementedError, match="planned work from one"):
        _unplanned.require_split(
            RepairableRBD(
                PAIR,
                {
                    "a": {
                        "reliability": E([0.01]),
                        "repairability": E([0.5]),
                        "inspection": {
                            "interval": 50.0,
                            "duration": FIXED([1.0]),
                        },
                    },
                    "b": {"reliability": E([0.01]), "repairability": E([0.5])},
                },
                repair_crews=1,
            )
        )


# -- what the plan avoids ---------------------------------------------------


def run_to_failure():
    """``nested()`` drawn with no planned work."""
    unplanned = {
        key: value for key, value in pump().items() if key != "preventive"
    }
    inner = RepairableRBD(PAIR, {"a": unplanned, "b": dict(unplanned)})
    return RepairableRBD(
        [("s", "n"), ("n", "c"), ("c", "t")],
        {"n": inner, "c": {**unplanned, "reliability": W([3000.0, 2.0])}},
        downtime_cost_rate=10.0,
    )


def test_failures_avoided_are_the_difference_of_the_two_diagrams():
    rbd, without = nested(), run_to_failure()
    effect = rbd.maintenance_effect(2000.0)
    plan, none = rbd.expected_events(2000.0), without.expected_events(2000.0)
    assert effect.system_failures == pytest.approx(plan.system_failures)
    assert effect.system_failures_without == pytest.approx(
        none.system_failures
    )
    assert effect.system_failures_avoided == pytest.approx(
        none.system_failures - plan.system_failures
    )
    assert effect.node_failures_avoided["c"] == pytest.approx(
        none.node_failures["c"] - plan.node_failures["c"]
    )
    assert effect.cost_saved == pytest.approx(
        without.expected_cost(2000.0).mean - rbd.expected_cost(2000.0).mean
    )
    # In the long run, rates.
    rates = rbd.maintenance_effect()
    assert rates.system_failures_without == pytest.approx(
        without.system_failure_frequency()
    )


def test_simulated_failures_avoided_are_two_runs_differenced():
    rbd, without = nested(), run_to_failure()
    n = 300
    effect = rbd.maintenance_effect(
        2000.0, method="simulate", mc_samples=n, seed=6
    )
    runs = [
        diagram.availability(
            2000.0, mc_samples=n, seed=6, control_variate=False
        )
        for diagram in (rbd, without)
    ]
    assert effect.system_failures_avoided == pytest.approx(
        (runs[1].system_failures - runs[0].system_failures) / n, rel=1e-12
    )
    assert effect.downtime_saved == pytest.approx(
        (runs[1].system_downtime - runs[0].system_downtime) / n, rel=1e-12
    )
    assert effect.system_unplanned_downtime == pytest.approx(
        runs[0].system_unplanned_downtime / n, rel=1e-12
    )
    # Paired on the same streams: the difference's error is smaller than
    # the two runs' own, taken apart.
    apart = math.sqrt(
        sum(np.var(np.asarray(run.uptimes), ddof=1) / n for run in runs)
    )
    assert effect.standard_errors["downtime_saved"] < apart


def test_running_to_failure_keeps_the_tests():
    tested = {
        "reliability": E([1e-3]),
        "repairability": E([0.1]),
        "inspection": {"interval": 100.0, "duration": FIXED([2.0])},
    }
    rbd = RepairableRBD(PAIR, {"a": tested, "b": pump()})
    twin = _unplanned.without_planned_work(rbd)
    assert not twin._preventive and "a" in twin._inspection
    assert not _unplanned.has_plan(twin)
    # Its tests take time, planned, with the plan or without it.
    assert _unplanned.takes_planned_outages(twin)


def test_a_component_in_its_maintenance_now_is_new_without_the_plan():
    rbd = RepairableRBD(ONE, {"p": pump()})
    state = {"p": NodeState(alive=False, down_for=2.0, maintenance=True)}
    effect = rbd.maintenance_effect(500.0, state=state)
    plain = RepairableRBD(
        ONE,
        {"p": {k: v for k, v in pump().items() if k != "preventive"}},
    )
    assert effect.system_downtime_without == pytest.approx(
        plain.expected_events(500.0).system_downtime
    )
    # Its 6 hours of maintenance left are planned.
    assert effect.system_planned_downtime >= 6.0 - 1e-9


def test_the_arguments_are_checked():
    rbd = RepairableRBD(ONE, {"p": pump()})
    with pytest.raises(ValueError, match="method"):
        rbd.maintenance_effect(100.0, method="guess")
    with pytest.raises(ValueError, match="not simulated"):
        rbd.maintenance_effect(method="simulate")
    with pytest.raises(ValueError, match="states"):
        rbd.maintenance_effect(state={"p": NodeState(age=10.0)})
    with pytest.raises(TypeError, match="number"):
        rbd.maintenance_effect("100")
    with pytest.raises(ValueError, match="positive"):
        rbd.maintenance_effect(-1.0)


def test_the_result_is_plain_data():
    rbd = RepairableRBD(ONE, {"p": pump()}, downtime_cost_rate=100.0)
    effect = rbd.maintenance_effect(YEAR)
    assert isinstance(effect, MaintenanceEffect)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = effect.to_dict()
    assert out["system_failures_avoided"] == effect.system_failures_avoided
