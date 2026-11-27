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
BLOCK = {"policy": "block", "interval": 150.0, "cost": 10.0}
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


KEPT_STATES = (
    None,
    {"a": NodeState(age=100.0)},
    {"a": NodeState(alive=False, down_for=2.0)},
)


@pytest.mark.parametrize(
    "more, states",
    [
        ({}, KEPT_STATES),
        ({"preventive": AGE}, KEPT_STATES),
        (
            {"preventive": {**AGE, "duration": FIXED([3.0])}},
            KEPT_STATES
            + ({"a": NodeState(alive=False, down_for=1.0, maintenance=True)},),
        ),
    ],
    ids=["no task", "age, in no time", "age, a fixed time"],
)
def test_kept_in_service_its_own_life_is_the_component_over_a_horizon(
    more, states
):
    # Each unit has the life in service when it is put in: to its own
    # life, the renewal equation split at the upgrade is the component's
    # own, but for rounding, as are its replacements one after another.
    kept = one(upgrade=upgrade(life=LIFE, replace_in_service=False), **more)
    alone = one(**more)
    t = np.concatenate([np.linspace(0.0, 1000.0, 2001), [151.0, 152.9]])
    for state in states:
        np.testing.assert_allclose(
            kept.point_availability(t, state=state),
            alone.point_availability(t, state=state),
            rtol=0.0,
            atol=1e-12,
        )
        mine = kept.expected_events(1000.0, state=state)
        theirs = alone.expected_events(1000.0, state=state)
        for name in ("node_failures", "node_preventive"):
            assert getattr(mine, name)["a"] == pytest.approx(
                getattr(theirs, name)["a"], rel=1e-12, abs=1e-12
            )
        assert kept.expected_npv(
            1000.0, discount_rate=1e-3, state=state
        ).npv == pytest.approx(
            alone.expected_npv(1000.0, discount_rate=1e-3, state=state).npv,
            rel=1e-12,
        )


def test_kept_in_service_its_own_life_with_a_varying_replacement_time():
    # A replacement of an exponential time: the replacements one after
    # another are followed exactly only while each unit surely starts on
    # one side of the upgrade, the grid taking the rest, so the integrals
    # agree to the grid's precision.
    more = {"preventive": {**AGE, "duration": E([1 / 5.0])}}
    kept = one(
        LIFE,
        FIXED([7.0]),
        upgrade=upgrade(life=LIFE, replace_in_service=False),
        **more,
    )
    alone = one(LIFE, FIXED([7.0]), **more)
    for state in (None, {"a": NodeState(age=100.0)}):
        assert kept.mission_availability(1000.0, state=state) == pytest.approx(
            alone.mission_availability(1000.0, state=state)
        )
        assert kept.expected_cost(1000.0, state=state).mean == pytest.approx(
            alone.expected_cost(1000.0, state=state).mean, rel=1e-6
        )


def kept_chain(t, lam, new, mu, at, up_at):
    """A memoryless unit kept in service at ``at``, up then with
    probability ``up_at``: the old unit up fails at ``lam``, a repair
    ends at ``mu``, and every unit put in from ``at`` fails at ``new``
    (states: the old unit up, down, a new unit up)."""
    from scipy.linalg import expm

    Q = np.array(
        [[-lam, lam, 0.0], [0.0, -mu, mu], [0.0, new, -new]], dtype=float
    )
    start = np.array([up_at, 1.0 - up_at, 0.0])
    return np.array(
        [(start @ expm(Q * (s - at)))[[0, 2]].sum() for s in np.ravel(t)]
    )


def test_a_memoryless_unit_kept_in_service_against_its_chain():
    lam, new, mu, at = 1 / 200.0, 1 / 800.0, 1 / 10.0, 400.0
    rbd = one(
        E([lam]),
        E([mu]),
        upgrade=upgrade(at, E([new]), replace_in_service=False),
    )
    before = np.array([0.0, 50.0, 200.0, 399.9])
    after = np.array([400.1, 420.0, 520.0, 600.0, 999.0])
    up_at = two_state(at, lambda s: lam, mu, at)
    np.testing.assert_allclose(
        rbd.point_availability(before),
        two_state(before, lambda s: lam, mu, at),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        rbd.point_availability(after),
        kept_chain(after, lam, new, mu, at, float(up_at)),
        atol=1e-6,
    )


@pytest.mark.parametrize(
    "more, states",
    [
        ({}, KEPT_STATES),
        (
            {"preventive": {**AGE, "duration": FIXED([3.0])}},
            KEPT_STATES
            + ({"a": NodeState(alive=False, down_for=1.0, maintenance=True)},),
        ),
        (
            {
                "preventive": {**AGE, "duration": E([1 / 5.0])},
                "repairability": FIXED([7.0]),
            },
            (None, {"a": NodeState(age=100.0)}),
        ),
    ],
    ids=["no task", "age, fixed replacement", "age, fixed repair"],
)
def test_kept_in_service_the_simulation_agrees(more, states):
    more = dict(more)
    repair = more.pop("repairability", None)
    rbd = one(
        repair=repair,
        upgrade=upgrade(cost=50.0, replace_in_service=False),
        **more,
    )
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


@pytest.mark.parametrize(
    "more",
    [
        {},
        {"preventive": {**AGE, "duration": FIXED([3.0])}},
        {"preventive": BLOCK},
        {"inspection": TESTS, "repairability": "instant"},
    ],
    ids=["no task", "age replacement", "block replacement", "tests"],
)
def test_a_unit_kept_in_service_is_worked_out_over_a_horizon(more):
    more = dict(more)
    repair = more.pop("repairability", None)
    rbd = one(repair=repair, upgrade=upgrade(replace_in_service=False), **more)
    report = rbd.analysis_routes()
    for name in ("point_availability", "expected_cost", "expected_npv"):
        assert report[name].route == routes.NUMERICAL


def test_a_unit_kept_in_service_with_tests_that_take_time_is_simulated():
    rbd = one(
        repair="instant",
        inspection={**TESTS, "duration": FIXED([1.0])},
        upgrade=upgrade(replace_in_service=False),
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


# -- kept in service under block replacement or tests ----------------------


@pytest.mark.parametrize(
    "more, at, states",
    [
        ({"preventive": BLOCK}, 400.0, KEPT_STATES),
        ({"preventive": BLOCK}, 100.0, KEPT_STATES),
        (
            {"preventive": {**BLOCK, "duration": FIXED([3.0])}},
            400.0,
            KEPT_STATES
            + ({"a": NodeState(alive=False, down_for=1.0, maintenance=True)},),
        ),
        (
            {"inspection": TESTS, "repairability": "instant"},
            400.0,
            (None, {"a": NodeState(age=100.0, since_scheduled=30.0)}),
        ),
        (
            {"inspection": TESTS, "repairability": "instant"},
            420.0,
            (None, {"a": NodeState(age=100.0, since_scheduled=30.0)}),
        ),
    ],
    ids=[
        "block",
        "block, upgraded before its first block time",
        "block, a fixed replacement",
        "tests, upgraded at a test",
        "tests, between tests",
    ],
)
def test_kept_in_service_its_own_life_is_the_component_with_its_task(
    more, at, states
):
    # The stretch (or tests) holding the upgrade is the component's own,
    # less what the units started from the upgrade bring with the old
    # life, plus what they bring with the new: to the same life, the
    # component to rounding (a plan of equal stretches is block
    # replacement on that period to rounding; the tested unit's renewals
    # are followed to 1e-13 of their long run).
    more = dict(more)
    repair = more.pop("repairability", None)
    kept = one(
        repair=repair,
        upgrade=upgrade(at, life=LIFE, replace_in_service=False),
        **more,
    )
    alone = one(repair=repair, **more)
    t = np.concatenate([np.linspace(0.0, 1000.0, 2001), [151.0, 421.0]])
    for state in states:
        np.testing.assert_allclose(
            kept.point_availability(t, state=state),
            alone.point_availability(t, state=state),
            rtol=0.0,
            atol=1e-12,
        )
        mine = kept.expected_events(1000.0, state=state)
        theirs = alone.expected_events(1000.0, state=state)
        for name in ("node_failures", "node_preventive", "node_corrective"):
            assert getattr(mine, name)["a"] == pytest.approx(
                getattr(theirs, name)["a"], rel=1e-12, abs=1e-12
            )
        assert kept.expected_npv(
            1000.0, discount_rate=1e-3, state=state
        ).npv == pytest.approx(
            alone.expected_npv(1000.0, discount_rate=1e-3, state=state).npv,
            rel=1e-12,
        )


def test_a_memoryless_unit_kept_with_its_tests_against_its_closed_form():
    # Tests every 50 in no time: up with probability exp(-lambda s), s
    # since the last test, lambda the life of the unit in service, which
    # is the new one once a test after the upgrade has renewed it. Kept
    # at 420, the unit up at its test at 400 is old until it fails and is
    # found (at 450 at the soonest).
    lam, new, at = 1 / 200.0, 1 / 800.0, 420.0
    rbd = one(
        E([lam]),
        "instant",
        inspection=TESTS,
        upgrade=upgrade(at, E([new]), replace_in_service=False),
    )

    def closed(x):
        # From the last test u: the unit then is old if no test in
        # (at, u] renewed it: with no test since the upgrade, or found
        # working at every one, the old unit carrying on.
        x = np.asarray(x, dtype=float)
        last = np.floor(x / 50.0) * 50.0
        tests = np.maximum((last - 400.0) / 50.0, 0.0)  # since the upgrade
        # P(the old unit still in service at u) = exp(-lam (u - 400)).
        old = np.where(last > 400.0, np.exp(-lam * (last - 400.0)), 1.0)
        s = x - last
        return np.where(
            x < at,
            np.exp(-lam * s),
            np.where(
                tests > 0,
                old * np.exp(-lam * s) + (1.0 - old) * np.exp(-new * s),
                np.exp(-lam * s),
            ),
        )

    t = np.array([10.0, 399.0, 419.0, 421.0, 449.9, 450.1, 499.0, 777.0])
    np.testing.assert_allclose(rbd.point_availability(t), closed(t), atol=1e-12)


@pytest.mark.parametrize(
    "more, at, states",
    [
        ({"preventive": BLOCK}, 400.0, KEPT_STATES),
        ({"preventive": BLOCK}, 100.0, KEPT_STATES),
        (
            {"preventive": {**BLOCK, "duration": FIXED([3.0])}},
            400.0,
            KEPT_STATES
            + ({"a": NodeState(alive=False, down_for=1.0, maintenance=True)},),
        ),
        (
            {"inspection": TESTS, "repairability": "instant"},
            400.0,
            (None, {"a": NodeState(age=100.0, since_scheduled=30.0)}),
        ),
    ],
    ids=[
        "block",
        "block, before its first block time",
        "block, a fixed replacement",
        "tests",
    ],
)
def test_kept_in_service_with_its_task_the_simulation_agrees(
    more, at, states
):
    more = dict(more)
    repair = more.pop("repairability", None)
    rbd = one(
        repair=repair,
        upgrade=upgrade(at, cost=50.0, replace_in_service=False),
        **more,
    )
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


# -- replaced at the upgrade, tests that take time or miss -----------------

TIMED = {**TESTS, "duration": FIXED([5.0])}
MISSING = {**TESTS, "coverage": 0.5, "full_test": 200.0}


@pytest.mark.parametrize(
    "life, new, repair, tests, at, states",
    [
        (
            LIFE,
            NEW,
            FIXED([7.0]),
            TIMED,
            402.0,
            (
                None,
                {"a": NodeState(age=100.0, since_scheduled=30.0)},
                {"a": NodeState(alive=False, down_for=2.0)},
            ),
        ),
        (LIFE, NEW, "instant", MISSING, 400.0, (None,)),
        (E([1 / 200.0]), E([1 / 800.0]), "instant", MISSING, 400.0, (None,)),
    ],
    ids=["tests and repairs of fixed lengths", "tests that miss", "memoryless"],
)
def test_tested_and_replaced_the_simulation_agrees(
    life, new, repair, tests, at, states
):
    # Replaced at the upgrade if up or failed unseen; off line for its
    # test (402 falls in the test at 400), once the test is over; found
    # failed, once its test and repair are.
    rbd = one(
        life,
        repair,
        inspection=tests,
        upgrade=upgrade(at, new, cost=50.0),
    )
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


def test_a_memoryless_unit_replaced_with_tests_that_miss_is_a_new_one():
    # Memoryless, every unit is replaced at 400, a full test's time (not
    # done then): from there it is a unit of the new life new at 0, whose
    # full tests fall alike.
    rbd = one(
        E([1 / 200.0]),
        "instant",
        inspection=MISSING,
        upgrade=upgrade(400.0, E([1 / 800.0])),
    )
    fresh = one(E([1 / 800.0]), "instant", inspection=MISSING)
    t = np.array([401.0, 449.0, 451.0, 600.0, 777.0, 999.0])
    np.testing.assert_allclose(
        rbd.point_availability(t), fresh.point_availability(t - 400.0),
        atol=1e-12,
    )
    assert rbd.mission_availability(1000.0) * 1000.0 == pytest.approx(
        one(E([1 / 200.0]), "instant", inspection=MISSING).mission_availability(
            400.0
        )
        * 400.0
        + fresh.mission_availability(600.0) * 600.0,
        rel=1e-8,  # the integrals' tolerance (_quadrature.TOLERANCE)
    )


def test_tests_of_varying_length_are_simulated():
    rbd = one(
        inspection={**TESTS, "duration": E([1.0])},
        upgrade=upgrade(),
    )
    route = rbd.analysis_routes()["point_availability"]
    assert route.route == routes.REFUSED
    assert "fixed lengths" in route.reason
    with pytest.raises(NotImplementedError) as error:
        rbd.mission_availability(1000.0)
    assert str(error.value).endswith(
        "estimate them by simulation, with availability() or cost() (or "
        "npv_distribution())."
    )
    rbd.availability(1000.0, mc_samples=20, seed=1)


# -- refusals and the spec ----------------------------------------------------


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
