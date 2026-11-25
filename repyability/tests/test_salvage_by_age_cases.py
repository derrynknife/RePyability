"""A salvage by age in the simulated runs for units whose age is more than
the time since they were put in as new: a unit that keeps a virtual age,
one with hidden failures, a group of units, a nested diagram's
components and a life on a duty plan that changes. Each is checked
against its age worked out by hand, run by run, from the run's own
histories, and against a closed form where there is one."""

import math

import numpy as np
import pytest
import surpyval as surv
from scipy import integrate

from repyability import RepairableRBD
from repyability.rbd import _npv_runs, _timeline_runs
from repyability.rbd._duty import PlannedLife
from repyability.schedule import DutySchedule

E = surv.Exponential.from_params
W = surv.Weibull.from_params
ONE = [("s", "a"), ("a", "t")]
T = 1000.0


def worth(age):
    return 1000.0 - 0.2 * np.asarray(age)


def one(**spec) -> RepairableRBD:
    return RepairableRBD(ONE, {"a": {"salvage": worth, **spec}})


def end_ages(rbd, runs=300, seed=4, horizon=T, engine="python"):
    """The runs (a TimelineSimulation) and each one's ages at the end."""
    found = _timeline_runs.simulate(
        rbd,
        horizon,
        runs,
        seed,
        None,
        None,
        False,
        engine,
        None,
        0,
        None,
        hidden=True,
        ages=True,
    )
    return found[0], found[2]


def changes(runs, node="a"):
    """Each run's changes of ``node``: whether it starts up, the times,
    and whether each was a planned change down."""
    data = runs.components[node]._data
    for s in range(data.size):
        a, z = int(data.offsets[s]), int(data.offsets[s + 1])
        yield bool(data.start[s]), data.times[a:z], data.planned[a:z]


def npv(rbd, runs=4000, seed=3, horizon=T):
    return rbd.npv_distribution(
        horizon, value_rate=1.0, mc_samples=runs, seed=seed
    )


def within(runs, hand, sd):
    """The runs' salvage is ``hand`` within four standard errors."""
    assert abs(runs.salvage - hand) < 4.0 * sd / math.sqrt(runs.samples.size)


# -- 1. a unit that keeps a virtual age ---------------------------------------


@pytest.mark.parametrize("model, q", [("kijima1", 1.0), ("kijima1", 0.5)])
@pytest.mark.parametrize("model2", [False, True])
def test_a_virtual_age_by_hand_from_each_run(model, q, model2):
    # Repaired imperfectly: each sojourn x since the last repair takes
    # the virtual age v to v + q x (Kijima I) or q (v + x) (Kijima II);
    # at the end a unit up is worth(v + its time since its last repair).
    # Minimal repair (Kijima I, q = 1) makes it its operating time since
    # new, the end less its repairs.
    model = "kijima2" if model2 else model
    rbd = one(
        reliability=W([300.0, 2.0]),
        repairability=E([0.05]),
        repair={"model": model, "q": q},
    )
    runs, ages = end_ages(rbd)
    for s, (up, times, _) in enumerate(changes(runs)):
        assert up
        if times.size % 2:
            assert math.isnan(ages[s, 0])
            continue
        v, since = 0.0, 0.0
        for down, back in zip(times[0::2], times[1::2]):
            x = down - since
            v = q * (v + x) if model2 else v + q * x
            since = back
        assert ages[s, 0] == pytest.approx(v + (T - since), rel=1e-12)
        if (model, q) == ("kijima1", 1.0):
            downtime = float(np.sum(times[1::2] - times[0::2]))
            assert ages[s, 0] == pytest.approx(T - downtime, rel=1e-12)


def test_a_fitted_process_s_own_age_by_hand():
    # G1 (one-renewal): after j failures each gap is scaled by (1 + q)^j,
    # and the age its life sees is the time since the last repair over
    # that scale.
    q = 0.3
    rbd = one(
        reliability=surv.GeneralizedOneRenewal.from_params([300.0, 2.0], q),
        repairability=E([0.05]),
        repair_cost=1.0,
    )
    runs, ages = end_ages(rbd)
    for s, (_, times, _) in enumerate(changes(runs)):
        if times.size % 2:
            assert math.isnan(ages[s, 0])
            continue
        since = float(times[-1]) if times.size else 0.0
        j = times.size // 2
        assert ages[s, 0] == pytest.approx(
            (T - since) / (1.0 + q) ** j, rel=1e-12
        )


def test_a_task_that_restores_to_an_age_by_hand():
    # Block tasks in no time restore the unit to the virtual age 100; a
    # repair renews it. At the end: 100 + the time since the last task,
    # or the time since the last repair, whichever came last.
    rbd = one(
        reliability=W([400.0, 2.0]),
        repairability=E([0.05]),
        preventive={"interval": 300.0, "policy": "block", "restore_to": 100.0},
    )
    runs, ages = end_ages(rbd)
    _, events = _timeline_runs.simulate(
        rbd,
        T,
        300,
        4,
        None,
        None,
        False,
        "python",
        None,
        0,
        None,
        counted=True,
    )
    from repyability.rbd._events import TALLY_PREVENTIVE

    tasks = events["a"]
    restored = 0
    for s, (_, times, _) in enumerate(changes(runs)):
        if times.size % 2:
            assert math.isnan(ages[s, 0])
            continue
        mine = (tasks.history == s) & ((tasks.kinds & TALLY_PREVENTIVE) != 0)
        task = float(tasks.times[mine].max()) if mine.any() else -1.0
        repaired = float(times[-1]) if times.size else 0.0
        hand = 100.0 + (T - task) if task > repaired else T - repaired
        restored += task > repaired
        assert ages[s, 0] == pytest.approx(hand, rel=1e-12)
    assert restored > 100


def test_npv_distribution_credits_the_virtual_age():
    rbd = one(
        reliability=W([300.0, 2.0]),
        repairability=E([0.05]),
        repair={"model": "kijima1", "q": 1.0},
        repair_cost=1.0,
    )
    result = npv(rbd, runs=300, seed=4)
    _, ages = end_ages(rbd, runs=300, seed=4)
    up = np.isfinite(ages[:, 0])
    assert result.salvage == pytest.approx(
        float(np.sum(worth(ages[up, 0]))) / 300, rel=1e-12
    )


# -- 2. hidden failures -------------------------------------------------------


def test_hidden_failures_against_the_closed_form():
    # Exponential(LAM) units tested at 600 (in no time), and a failure
    # found then repaired at once. At T = 1000 the unit is up as the
    # first, never failed (age T), or as the second, put in at 600 and
    # not failed since (age 400); a unit failed unseen is credited
    # nothing.
    lam, tau = 1e-3, 600.0
    rbd = one(
        reliability=E([lam]),
        repairability="instant",
        inspection={"interval": tau},
    )
    first = math.exp(-lam * T) * float(worth(T))
    second = (
        (1.0 - math.exp(-lam * tau))
        * math.exp(-lam * (T - tau))
        * float(worth(T - tau))
    )
    within(npv(rbd), first + second, 1000.0)
    _, ages = end_ages(rbd)
    found = ages[np.isfinite(ages[:, 0]), 0]
    assert set(np.round(found, 9)) <= {T, T - tau}
    # (the share failed unseen is credited nothing)
    down = 1.0 - math.exp(-lam * T) - second / float(worth(T - tau))
    assert np.isnan(ages[:, 0]).mean() == pytest.approx(down, abs=0.1)


def test_a_test_off_line_does_not_age_the_unit():
    # Tests of 5 h each: a unit never failed is as old as the end less
    # its tests; one repaired, as old as the time since its repair less
    # the tests since.
    rbd = one(
        reliability=W([2000.0, 2.0]),
        repairability=E([0.05]),
        inspection={"interval": 200.0, "duration": 5.0},
    )
    runs, ages = end_ages(rbd)
    whole = 0
    for s, (up, times, planned) in enumerate(changes(runs)):
        assert up
        if times.size % 2:
            assert math.isnan(ages[s, 0])
            continue
        new, tests = 0.0, 0.0
        for k in range(0, times.size, 2):
            if planned[k]:
                tests += times[k + 1] - times[k]
            else:
                new, tests = float(times[k + 1]), 0.0
        assert ages[s, 0] == pytest.approx(T - new - tests, rel=1e-12)
        if planned[0::2].all():
            whole += 1
            assert ages[s, 0] == pytest.approx(T - 5.0 * (times.size // 2))
    assert whole > 50


# -- 3. groups of units -------------------------------------------------------

#: A repair that never ends in the window: what failed stays down.
NEVER = E([1e-9])


def test_a_cold_standby_pair_against_the_closed_form():
    # Unit 1 operates; at its failure L1 the cold spare (age 0) takes
    # over. At T: both up if L1 > T (worth(T) + worth(0)); the spare
    # alone if L1 < T < L1 + L2 (worth(T - L1)), whose chance and mean
    # make lam e^{-lam T} (c T - k T^2 / 2) for worth(a) = c - k a.
    lam = 1e-3
    rbd = one(
        reliability=E([lam]),
        repairability=NEVER,
        standby={"units": 2, "k": 1},
    )
    c, k = 1000.0, 0.2
    hand = math.exp(-lam * T) * (float(worth(T)) + float(worth(0.0)))
    hand += lam * math.exp(-lam * T) * (c * T - k * T**2 / 2.0)
    within(npv(rbd), hand, 2000.0)


def test_a_warm_spare_ages_at_its_dormancy():
    # The spare waits at half the rate (dormancy 0.5): at T beside the
    # first unit it is worth(T / 2) if it has not failed waiting; once
    # switched in at L1 it is aged L1 / 2 and goes on at rate 1.
    lam, d = 1e-3, 0.5
    rbd = one(
        reliability=E([lam]),
        repairability=NEVER,
        standby={"units": 2, "k": 1, "dormancy_factor": d},
    )
    both = math.exp(-lam * T) * (
        float(worth(T)) + math.exp(-d * lam * T) * float(worth(d * T))
    )
    switched = integrate.quad(
        lambda t: lam
        * math.exp(-lam * t)
        * math.exp(-d * lam * t)
        * math.exp(-lam * (T - t))
        * float(worth(d * t + (T - t))),
        0.0,
        T,
    )[0]
    within(npv(rbd), both + switched, 2000.0)
    _, ages = end_ages(rbd)
    assert ages.shape == (300, 2)


def test_identical_units_are_each_credited():
    # Two identical units, each up at T with chance e^{-lam T}, at age T.
    lam = 1e-3
    rbd = one(
        reliability=E([lam]),
        repairability=NEVER,
        identical={"units": 2, "k": 1},
    )
    within(npv(rbd), 2.0 * math.exp(-lam * T) * float(worth(T)), 2000.0)


# -- 4. a nested diagram ------------------------------------------------------

LAM = 1e-3


def nested():
    deep = RepairableRBD(
        ONE,
        {
            "a": {
                "reliability": E([LAM]),
                "repairability": "instant",
                "salvage": worth,
            }
        },
    )
    middle = RepairableRBD(
        [("s", "x"), ("x", "deep"), ("deep", "t")],
        {
            "x": {
                "reliability": E([LAM]),
                "repairability": "instant",
                "salvage": worth,
                "repair_cost": 1.0,
            },
            "deep": deep,
        },
    )
    return RepairableRBD(
        [("s", "m"), ("m", "y"), ("y", "t")],
        {
            "m": middle,
            "y": {
                "reliability": E([LAM]),
                "repairability": "instant",
                "salvage": 50.0,
            },
        },
    )


def test_a_nested_diagram_s_components_each_by_their_own_rule():
    # x (in m) and a (in m's deep), each always up (repaired at once),
    # are each credited worth(min(Exp(LAM), T)), whose mean is c - k (1 -
    # e^{-LAM T}) / LAM; y's own salvage is the number 50.
    rbd = nested()
    one_unit = 1000.0 - 0.2 * (1.0 - math.exp(-LAM * T)) / LAM
    result = npv(rbd)
    within(result, 50.0 + 2.0 * one_unit, 0.2 * 2.0 * 300.0)


def test_a_nested_unit_is_credited_its_age_in_the_system_s_run():
    # Run by run, as the nested diagram's own run (its streams named by
    # its place) gives it: the ages it has inside the system's runs.
    rbd = nested()
    money = _npv_runs._npv.Money(rbd, 0.0, 1.0)
    ledger, _ = _npv_runs.priced_chunk(
        rbd, T, np.array([T]), money, 200, 9, "auto", 0, None
    )
    credited = {
        k[2]: v[0] for k, v in ledger.cells.items() if k[1] == "salvage"
    }
    # (y's number is credited to every run alike, by _npv.own_end.)
    assert set(credited) == {("m", "x"), ("m", "deep", "a")}
    middle = rbd.components["m"]
    _, ages = _timeline_runs.simulate(
        middle,
        T,
        200,
        9,
        None,
        None,
        False,
        "python",
        None,
        0,
        None,
        hidden=True,
        prefix=("m",),
        ages=True,
    )[0::2]
    np.testing.assert_array_equal(credited[("m", "x")], -worth(ages[:, 0]))


def test_a_nested_unit_s_other_terms_are_still_refused():
    rbd = nested()
    rbd.components["m"]._valuation["x"]["escalation"] = 1e-4
    with pytest.raises(NotImplementedError, match="escalation"):
        npv(rbd, runs=10)


# -- 5. a duty plan that changes ----------------------------------------------

PLAN = DutySchedule(
    "hours", (0.0, 1000.0, 2000.0, 3000.0), (1.0, 0.25, 0.0, 0.5)
)


def test_a_life_on_a_duty_plan_ages_in_operating_time():
    # Repaired at once, an Exponential(LAM) unit in service at 3600 h is
    # as old as min(Exp(LAM), D(3600)) operating hours, D(3600) = 1550:
    # mean (1 - e^{-LAM D}) / LAM.
    rbd = one(reliability=E([LAM]), repairability="instant", duty=PLAN)
    hours = 1550.0
    hand = 1000.0 - 0.2 * (1.0 - math.exp(-LAM * hours)) / LAM
    within(npv(rbd, horizon=3600.0), hand, 0.2 * hours)


def test_operating_age_by_hand_from_each_run():
    # With repairs that take time: the plan's operating hours from the
    # last repair's end to the end.
    rbd = one(reliability=W([400.0, 2.0]), repairability=E([0.05]), duty=PLAN)
    clock = PlannedLife(W([400.0, 2.0]), PLAN, "a")
    runs, ages = end_ages(rbd, horizon=3600.0)
    for s, (_, times, _) in enumerate(changes(runs)):
        if times.size % 2:
            assert math.isnan(ages[s, 0])
            continue
        since = float(times[-1]) if times.size else 0.0
        hand = float(clock.operating(3600.0)) - float(clock.operating(since))
        assert ages[s, 0] == pytest.approx(hand, rel=1e-9, abs=1e-9)


# -- the engines --------------------------------------------------------------


@pytest.mark.parametrize("kind", ["hidden failures", "nested", "duty plan"])
def test_numba_s_runs_credit_what_the_python_loop_s_do(kind):
    # To the bit: a tested unit's age from numba's histories, and a nested
    # diagram's components' from its own run (the Python loop's, on either
    # engine of the system).
    pytest.importorskip("numba")
    rbd = (
        one(
            reliability=W([2000.0, 2.0]),
            repairability=E([0.05]),
            inspection={"interval": 200.0, "duration": E([0.2])},
        )
        if kind == "hidden failures"
        else (
            one(
                reliability=W([400.0, 2.0]), repairability=E([0.05]), duty=PLAN
            )
            if kind == "duty plan"
            else nested()
        )
    )
    kwargs = dict(value_rate=1.0, mc_samples=3000, seed=8)
    py = rbd.npv_distribution(T, engine="python", **kwargs)
    nb = rbd.npv_distribution(T, engine="numba", **kwargs)
    assert nb.engine == "numba"
    np.testing.assert_array_equal(nb.samples, py.samples)
    assert nb.salvage == py.salvage


def test_a_group_s_route_says_it_runs_in_python():
    rbd = one(reliability=E([1e-3]), repairability=NEVER, standby={"units": 2})
    route = rbd.analysis_routes()["npv_distribution"]
    assert route.engine == "python" and "Python engine" in route.engine_reason
