"""Age replacement and tests on an upgraded component (UC-72): the unit
up at the upgrade is replaced then (as at its age, or in no time before
its tests), and every unit put into service from then has the new life;
or, with ``"replace_in_service": False``, the unit in service stays
until its next failure or due replacement.

The references are age replacement alone (an upgrade to the same life
of a memoryless unit replaced in no time changes nothing but one more
replacement; one kept in service is nothing at all, run for run), a
memoryless unit's two-state chain whose rate changes at the upgrade,
the closed form of a memoryless unit tested and renewed, and the
simulation.
"""

import math

import numpy as np
import pytest
import surpyval as surv

from repyability import NodeState, RepairableRBD
from repyability.maintenance import Upgrade
from repyability.rbd import routes
from repyability.rbd.serialisation import rbd_from_dict, rbd_to_dict

E = surv.Exponential.from_params
W = surv.Weibull.from_params
FIXED = surv.ExactEventTime.from_params
EDGES = [("s", "a"), ("a", "t")]
LIFE = W([300, 2.0])
NEW = W([900, 2.0])
AGE = {"policy": "age", "interval": 150.0, "cost": 10.0}
TESTS = {"interval": 50.0, "cost": 1.0}
Z = 4.0


def one(life=LIFE, repair=None, **more):
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


def upgrade(at=400.0, life=NEW, **more):
    return {"at": at, "reliability": life, **more}


def plain_run(rbd, T, seed, state=None, **more):
    return rbd.availability(
        T,
        mc_samples=20_000,
        seed=seed,
        state=state,
        control_variate=False,
        conditional=False,
        **more,
    )


# -- the same life -------------------------------------------------------


def test_a_memoryless_unit_upgraded_to_its_own_life_is_age_replacement():
    # Exponential life and repair, replaced at its age in no time: a
    # replacement changes nothing but the count of replacements, so the
    # upgrade to the same life adds the unit up at the upgrade's.
    life, repair = E([1 / 200.0]), E([0.1])
    upgraded = one(life, repair, preventive=AGE, upgrade=upgrade(life=life))
    alone = one(life, repair, preventive=AGE)
    # Away from the replacements at fixed ages (the grid's own error,
    # some 1e-5 within a step of them, is alike in both).
    t = np.arange(0.0, 1000.0, 7.3)
    np.testing.assert_allclose(
        upgraded.point_availability(t), alone.point_availability(t), atol=1e-6
    )
    mine, theirs = upgraded.expected_events(1000.0), alone.expected_events(
        1000.0
    )
    assert mine.node_failures["a"] == pytest.approx(
        theirs.node_failures["a"], abs=1e-6
    )
    # Its replacements: age replacement's to the upgrade, the one at it,
    # and from there those of a unit new (replaced) or in a repair (its
    # repair memoryless), each the component's from that state.
    up = float(alone.point_availability(np.array([400.0]))[0])
    down = {"a": NodeState(alive=False, down_for=1.0)}
    after = (
        up * alone.expected_events(600.0).node_preventive["a"]
        + (1.0 - up)
        * alone.expected_events(600.0, state=down).node_preventive["a"]
    )
    assert mine.node_preventive["a"] == pytest.approx(
        alone.expected_events(400.0).node_preventive["a"] + up + after,
        rel=1e-6,
    )


@pytest.mark.parametrize(
    "more",
    [
        {"preventive": {**AGE, "duration": FIXED([3.0])}},
        {"inspection": TESTS},
        {},
    ],
    ids=["age replacement", "tests", "no task"],
)
def test_a_unit_kept_in_service_to_its_own_life_is_nothing(more):
    # Kept in service, each unit's life is drawn from the life in service
    # when it starts, from the same uniform: the same life is the
    # component without its upgrade, run for run.
    kept = one(upgrade=upgrade(life=LIFE, replace_in_service=False), **more)
    alone = one(**more)
    for state in (None, {"a": NodeState(age=100.0)}):
        mine = kept.cost(1000.0, mc_samples=500, seed=3, state=state)
        theirs = alone.cost(1000.0, mc_samples=500, seed=3, state=state)
        np.testing.assert_array_equal(mine.samples, theirs.samples)
        up = kept.availability(1000.0, mc_samples=500, seed=3, state=state)
        down = alone.availability(1000.0, mc_samples=500, seed=3, state=state)
        np.testing.assert_array_equal(up.uptimes, down.uptimes)


# -- memoryless references -----------------------------------------------


def two_state(t, lam_at, mu, at):
    """A memoryless unit new at 0, repaired at rate ``mu``, failing at
    ``lam_at(s)``: its two-state chain, its rate changing at ``at``."""

    def chain(lam, s, up):
        total = lam + mu
        return mu / total + (up - mu / total) * np.exp(-total * s)

    t = np.asarray(t, dtype=float)
    before = chain(lam_at(0.0), np.minimum(t, at), 1.0)
    at_upgrade = float(chain(lam_at(0.0), np.array(at), 1.0))
    after = chain(lam_at(at), np.maximum(t - at, 0.0), at_upgrade)
    return np.where(t < at, before, after)


def test_a_memoryless_unit_against_its_chain():
    lam, new, mu, at = 1 / 200.0, 1 / 800.0, 1 / 10.0, 400.0
    rbd = one(E([lam]), E([mu]), preventive=AGE, upgrade=upgrade(at, E([new])))
    t = np.array([0.0, 50.0, 120.0, 200.0, 280.0, 350.0, 399.9, 400.1])
    t = np.concatenate([t, [420.0, 520.0, 600.0, 999.0]])

    def rate_at(s):
        return lam if s < at else new

    np.testing.assert_allclose(
        rbd.point_availability(t), two_state(t, rate_at, mu, at), atol=1e-6
    )


def found_by_tests(t, lam_at, interval, at):
    """A memoryless unit with hidden failures, tested every ``interval``
    and renewed in no time when found failed, and replaced at ``at``:
    up with probability ``exp(-lambda (t - last))``, ``last`` its last
    test or the upgrade."""
    t = np.asarray(t, dtype=float)
    last = np.floor(t / interval) * interval
    last = np.where((t >= at) & (last < at), at, last)
    return np.exp(-np.where(t < at, lam_at(0.0), lam_at(at)) * (t - last))


def test_a_memoryless_tested_unit_against_its_closed_form():
    lam, new, at = 1 / 200.0, 1 / 800.0, 420.0
    rbd = one(
        E([lam]),
        "instant",
        inspection=TESTS,
        upgrade=upgrade(at, E([new])),
    )
    t = np.array([0.0, 10.0, 49.9, 50.1, 419.9, 420.0, 430.0, 449.9, 450.1])
    t = np.concatenate([t, [700.0, 999.0]])

    def rate_at(s):
        return lam if s < at else new

    np.testing.assert_allclose(
        rbd.point_availability(t),
        found_by_tests(t, rate_at, 50.0, at),
        atol=1e-9,
    )
    # The unit in service at the upgrade is replaced: working, a
    # preventive action; failed unseen, it is never found.
    events = rbd.expected_events(1000.0)
    up = float(np.exp(-lam * (at - 400.0)))
    assert events.node_preventive["a"] == pytest.approx(up, rel=1e-9)


# -- the simulation -------------------------------------------------------


@pytest.mark.parametrize(
    "more, states",
    [
        (
            {"preventive": {**AGE, "duration": FIXED([3.0])}},
            (
                None,
                {"a": NodeState(age=100.0)},
                {"a": NodeState(alive=False, down_for=2.0)},
                {"a": NodeState(alive=False, down_for=1.0, maintenance=True)},
            ),
        ),
        (
            {
                "preventive": {**AGE, "duration": E([1 / 5.0])},
                "repair": None,
                "repairability": FIXED([7.0]),
            },
            (None, {"a": NodeState(age=100.0)}),
        ),
        (
            {"inspection": TESTS, "repairability": "instant"},
            (None, {"a": NodeState(age=100.0, since_scheduled=30.0)}),
        ),
    ],
    ids=["age, fixed replacement", "age, fixed repair", "tests"],
)
def test_the_simulation_agrees_from_new_and_from_a_state(more, states):
    more = {key: value for key, value in more.items() if value is not None}
    repair = more.pop("repairability", None)
    rbd = one(repair=repair, upgrade=upgrade(cost=50.0), **more)
    T = 1000.0
    for seed, state in enumerate(states, start=1):
        run = plain_run(rbd, T, seed, state)
        interval = run.mean_availability_interval()
        exact = rbd.mission_availability(T, state=state)
        assert abs(interval.estimate - exact) <= Z * interval.standard_error
        cost = rbd.expected_cost(T, state=state)
        assert abs(run.cost.sample_mean - cost.mean) <= (
            Z * run.cost.sample_se
        )


def test_its_npv_follows_its_events():
    rbd = one(
        preventive={**AGE, "duration": FIXED([3.0])},
        upgrade=upgrade(cost=50.0),
    )
    T = 1000.0
    priced = rbd.expected_npv(T, discount_rate=0.0)
    assert priced.cost == pytest.approx(rbd.expected_cost(T).mean, rel=1e-9)
    assert priced.cost_by_category["capital"] == pytest.approx(50.0)


# -- the unit kept in service -----------------------------------------------


def lives(old, new, **more):
    return one(
        FIXED([old]),
        "instant",
        upgrade=upgrade(life=FIXED([new]), **more),
    )


@pytest.mark.parametrize("keep, failures", [(False, 1.0), (True, 2.0)])
def test_kept_in_service_the_old_unit_runs_to_its_failure(keep, failures):
    # Lives of exactly 300 and, from 400, 1000: replaced at 400, the unit
    # put in at 300 never fails by 1000; kept, it fails at 600, and the
    # unit put in then has the new life.
    rbd = lives(300.0, 1000.0, replace_in_service=not keep)
    counted = rbd.cost(1000.0, mc_samples=50, seed=1)
    np.testing.assert_array_equal(counted.samples, 6.0 * failures)


@pytest.mark.parametrize(
    "more",
    [
        {"preventive": {**AGE, "duration": FIXED([3.0])}},
        {"preventive": {**AGE, "policy": "block"}},
        {"inspection": TESTS, "repairability": "instant"},
        {},
    ],
    ids=["age replacement", "block replacement", "tests", "no task"],
)
def test_a_unit_kept_in_service_is_simulated(more):
    repair = more.pop("repairability", None)
    rbd = one(
        repair=repair,
        upgrade=upgrade(replace_in_service=False),
        **more,
    )
    report = rbd.analysis_routes()
    for name in ("point_availability", "expected_cost", "expected_npv"):
        assert report[name].route == routes.REFUSED
        assert "replace_in_service False" in report[name].reason
    for name in ("availability", "cost"):
        assert report[name].route == routes.SIMULATED
    with pytest.raises(NotImplementedError) as error:
        rbd.point_availability(np.array([500.0]))
    assert "estimate them by simulation, with availability()" in str(
        error.value
    )
    rbd.availability(600.0, mc_samples=20, seed=1)


# -- refusals and the spec ----------------------------------------------------


def test_tests_that_take_time_are_simulated():
    rbd = one(
        inspection={**TESTS, "duration": FIXED([1.0])},
        upgrade=upgrade(),
    )
    route = rbd.analysis_routes()["point_availability"]
    assert route.route == routes.REFUSED
    with pytest.raises(NotImplementedError) as error:
        rbd.mission_availability(1000.0)
    assert str(error.value).endswith(
        "estimate them by simulation, with availability() or cost() (or "
        "npv_distribution())."
    )
    rbd.availability(1000.0, mc_samples=20, seed=1)


@pytest.mark.parametrize(
    "at, tests, found", [(402.0, 10, 2), (400.0, 9, 2), (460.0, 10, 1)]
)
def test_a_unit_off_line_for_its_test_is_replaced_once_it_is_back(
    at, tests, found
):
    # Tested every 50 for 5, repaired in no time; the old unit never
    # fails, the new one 30 after it is put in. Upgraded at 402, in the
    # test from 400, the unit is replaced at 405 and fails at 435 (found
    # at 450, back at 455) and 485 (found at 500). At 400 it is replaced
    # before that test, which is not done, and fails at 430 and 485. At
    # 460 it fails at 490 (found at 500). By 520: the tests done, at 1
    # each, and the repair of each failure found, at 6.
    rbd = one(
        FIXED([1e6]),
        "instant",
        inspection={**TESTS, "duration": FIXED([5.0])},
        upgrade=upgrade(at, FIXED([30.0])),
    )
    counted = rbd.cost(520.0, mc_samples=5, seed=1)
    np.testing.assert_array_equal(counted.samples, tests * 1.0 + 6.0 * found)


def test_tests_that_miss_a_failure_are_simulated():
    rbd = one(
        E([1 / 200.0]),
        "instant",
        inspection={**TESTS, "coverage": 0.5, "full_test": 200.0},
        upgrade=upgrade(life=E([1 / 800.0])),
    )
    assert rbd.analysis_routes()["expected_cost"].route == routes.REFUSED
    run = rbd.availability(1000.0, mc_samples=2000, seed=1)
    # Better than the old unit alone, worse than the new one from new.
    alone = one(
        E([1 / 200.0]),
        "instant",
        inspection={**TESTS, "coverage": 0.5, "full_test": 200.0},
    ).mission_availability(1000.0)
    assert run.mean_availability_interval().estimate > alone


def test_its_tests_calendar_is_its_own_but_it_has_no_long_run_state():
    rbd = one(inspection=TESTS, repair="instant", upgrade=upgrade())
    rbd.mission_availability(
        500.0, state={"a": NodeState(age=10.0, since_scheduled=10.0)}
    )
    with pytest.raises(ValueError, match="no long-run state"):
        rbd.mission_availability(
            500.0, state={"a": NodeState(stationary=True)}
        )


def test_the_option_is_saved_and_a_plan_writes_it():
    rbd = one(
        preventive=AGE,
        upgrade=upgrade(replace_in_service=False, cost=5.0),
    )
    again = rbd_from_dict(rbd_to_dict(rbd))
    assert again._upgrades["a"].kept
    task = Upgrade(at=400.0, life=NEW, replace_in_service=False)
    assert task.spec() == {
        "at": 400.0,
        "reliability": NEW,
        "replace_in_service": False,
    }
    assert "replace_in_service" not in Upgrade(at=1.0, life=NEW).spec()


def test_the_upgrade_takes_age_replacement_and_tests_together_with_none_else():
    with pytest.raises(ValueError, match="a preventive task other than"):
        one(
            preventive={
                "policy": "condition",
                "interval": 100.0,
                "threshold": 0.1,
            },
            upgrade=upgrade(),
        )
    assert math.isfinite(
        one(preventive=AGE, upgrade=upgrade()).mission_availability(500.0)
    )
