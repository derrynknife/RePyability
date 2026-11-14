"""Covariate schedules on a ``RepairableRBD``'s blocks (a spec's
``"covariates"``): a block whose life is a surpyval regression model, run
under covariates that change over the diagram's time.

Each life is drawn by inverting the cumulative hazard along the schedule
from when the unit is put into service, so the draws are checked against
closed forms (an exponential and a Weibull baseline under proportional
hazards), against surpyval's own ``Hf_tvc`` for a first life from 0, and
the simulations against a Markov chain worked out by hand and against a
renewal process simulated plainly here.
"""

import json
import math

import numpy as np
import pytest
import surpyval as surv
from scipy.integrate import quad
from scipy.linalg import expm

from repyability import NodeState, NonRepairable, RepairableRBD
from repyability.rbd._covariate_schedule import ScheduledLife

E = surv.Exponential.from_params
W = surv.Weibull.from_params
EDGES = [("s", "a"), ("a", "t")]
STEPS = surv.StepSchedule.from_changepoints


def _fit(cls, scale=1.0):
    """A one-covariate regression of ``cls`` with no ties between the
    loads' lives: lives about ``100 * scale`` at load 1 and a quarter of
    that at load 2."""
    rng = np.random.default_rng(7)
    load = np.repeat([1.0, 2.0], 40)
    x = scale * 100.0 * rng.weibull(1.5, load.size) / load**2
    return cls.fit(x, Z=load[:, None])


@pytest.fixture(scope="module")
def exp_ph():
    return _fit(surv.ExponentialPH)


@pytest.fixture(scope="module")
def weibull_ph():
    return _fit(surv.WeibullPH)


@pytest.fixture(scope="module")
def weibull_aft():
    return _fit(surv.WeibullAFT)


def _H(model, x, z):
    return float(model.Hf(np.array([x]), np.array([[z]]))[0])


def _unit(model, schedule, repairability=None, **more):
    return {
        "reliability": model,
        "repairability": E([1.0]) if repairability is None else repairability,
        "covariates": schedule,
        **more,
    }


# -- The draws against closed forms -----------------------------------------


def test_an_exponential_baseline_crosses_its_steps_in_closed_form(exp_ph):
    """Under proportional hazards with an exponential baseline the rate is
    constant within a step: ``r1`` before the change at ``T`` and ``r2``
    after, so a life started at ``t0 < T`` with exposure ``e`` is ``e /
    r1`` if that ends before ``T``, else ``(T - t0) + (e - r1 (T - t0)) /
    r2``."""
    T = 40.0
    life = ScheduledLife(exp_ph, STEPS([0.0, T], [[1.0], [2.0]]), "a")
    r1, r2 = (
        float(exp_ph.hf(np.array([1.0]), np.array([[z]]))[0])
        for z in (1.0, 2.0)
    )
    for t0 in (0.0, 15.0, 39.0, 55.0):
        for u in (1e-9, 0.1, 0.5, 0.9, 0.999999):
            e = -math.log1p(-u)
            before = max(T - t0, 0.0)
            if e <= r1 * before:
                want = e / r1
            else:
                want = before + (e - r1 * before) / r2
            got = life.life_left(t0, 0.0, u)
            assert got == pytest.approx(want, rel=1e-12)


def test_a_weibull_baseline_takes_its_age_into_each_step(weibull_ph):
    """A Weibull baseline under proportional hazards: ``H(a, Z) = c_Z (a /
    alpha) ** beta``. A unit of age ``a0`` started at ``t0``, before the
    change at ``T``, takes on ``c_1 ((a0 + T - t0) ** beta - a0 ** beta)
    / alpha ** beta`` by ``T`` and its age goes on from there at ``c_2``:
    its life inverts that in closed form."""
    T = 30.0
    life = ScheduledLife(weibull_ph, STEPS([0.0, T], [[1.0], [2.0]]), "a")
    # Read beta and c_Z / alpha ** beta off the model itself.
    beta = math.log(_H(weibull_ph, 2.0, 1.0) / _H(weibull_ph, 1.0, 1.0)) / (
        math.log(2.0)
    )
    k1, k2 = _H(weibull_ph, 1.0, 1.0), _H(weibull_ph, 1.0, 2.0)
    for t0, a0 in ((0.0, 0.0), (10.0, 25.0), (29.0, 5.0), (40.0, 12.0)):
        for u in (1e-6, 0.3, 0.8, 0.99999):
            e = -math.log1p(-u)
            before = max(T - t0, 0.0)
            first = k1 * ((a0 + before) ** beta - a0**beta)
            if e <= first:
                want = (e / k1 + a0**beta) ** (1.0 / beta) - a0
            else:
                age = a0 + before
                want = (
                    before
                    + ((e - first) / k2 + age**beta) ** (1.0 / beta)
                    - age
                )
            got = life.life_left(t0, a0, u)
            assert got == pytest.approx(want, rel=1e-10)


@pytest.mark.parametrize(
    "cls",
    [
        surv.WeibullPH,
        surv.WeibullAFT,
        surv.LogNormalAFT,
        surv.WeibullAH,
        surv.ExponentialPH,
    ],
)
def test_a_first_life_from_0_is_surpyvals_life_along_the_path(cls):
    """A new unit put in at 0 has the life surpyval gives along the path:
    its cumulative hazard at the life drawn is ``Hf_tvc``'s, for each kind
    taken (proportional and additive hazards, accelerated failure time)."""
    model = _fit(cls)
    schedule = STEPS([0.0, 20.0, 45.0], [[1.0], [2.0], [1.5]])
    life = ScheduledLife(model, schedule, "a")
    u = np.array([1e-6, 0.05, 0.3, 0.6, 0.9, 0.999])
    drawn = np.array([life.life_left(0.0, 0.0, v) for v in u])
    H = np.asarray(model.Hf_tvc(drawn, schedule), dtype=float)
    np.testing.assert_allclose(H, -np.log1p(-u), rtol=1e-9)


def test_a_life_started_later_runs_on_the_rest_of_the_schedule(weibull_aft):
    """A unit put in new at ``t0`` lives as one put in at 0 under the
    schedule shifted back by ``t0``: the schedule is on the diagram's
    clock, not the unit's."""
    t0 = 33.0
    schedule = STEPS([0.0, 20.0, 45.0], [[1.0], [2.0], [1.5]])
    shifted = STEPS([0.0, 45.0 - t0], [[2.0], [1.5]])
    later = ScheduledLife(weibull_aft, schedule, "a")
    fresh = ScheduledLife(weibull_aft, shifted, "a")
    for u in (0.01, 0.4, 0.95):
        assert later.life_left(t0, 0.0, u) == pytest.approx(
            fresh.life_left(0.0, 0.0, u), rel=1e-12
        )


def test_a_cyclic_schedule_is_its_steps_written_out(weibull_ph):
    """A cyclic schedule (60 at load 1, 40 at load 2, over and over) draws
    the lives of the same steps written out one by one."""
    cyclic = surv.StepSchedule.cyclic([0.0, 60.0], [[1.0], [2.0]], 100.0)
    times = [t for n in range(30) for t in (100.0 * n, 100.0 * n + 60.0)]
    rows = [[1.0], [2.0]] * 30
    written = ScheduledLife(weibull_ph, STEPS(times, rows), "a")
    cycled = ScheduledLife(weibull_ph, cyclic, "a")
    for t0 in (0.0, 75.0, 1234.5):
        for u in (0.2, 0.7, 0.99):
            assert cycled.life_left(t0, 3.0, u) == pytest.approx(
                written.life_left(t0, 3.0, u), rel=1e-12
            )


def test_a_life_of_no_exposure_is_none_and_of_all_is_endless(exp_ph):
    life = ScheduledLife(exp_ph, STEPS([0.0, 10.0], [[1.0], [2.0]]), "a")
    assert life.life_left(5.0, 0.0, 0.0) == 0.0
    assert life.life_left(5.0, 0.0, 1.0) == math.inf


# -- The simulations ---------------------------------------------------------


def _two_state_availability(rates, change, mu, horizon):
    """The mean availability over ``[0, horizon]`` of a unit with
    exponential lives at ``rates[0]`` before ``change`` and ``rates[1]``
    after, and exponential repairs at ``mu``: its two-state Markov chain,
    stepped through the two stretches by hand."""
    p = np.array([1.0, 0.0])
    up = 0.0
    for start, end, rate in (
        (0.0, change, rates[0]),
        (change, horizon, rates[1]),
    ):
        Q = np.array([[-rate, rate], [mu, -mu]])
        up += quad(lambda s: (p @ expm(Q * s))[0], 0.0, end - start)[0]
        p = p @ expm(Q * (end - start))
    return up / horizon


def test_availability_with_a_two_step_schedule_is_the_markov_chain(exp_ph):
    """One unit, exponential lives under proportional hazards on a
    two-step load and exponential repairs: the simulated mean availability
    against the chain worked out by hand, within four standard errors."""
    change, horizon, mu = 40.0, 100.0, 0.5
    rbd = RepairableRBD(
        EDGES,
        {"a": _unit(exp_ph, STEPS([0.0, change], [[1.0], [2.0]]), E([mu]))},
    )
    rates = [
        float(exp_ph.hf(np.array([1.0]), np.array([[z]]))[0])
        for z in (1.0, 2.0)
    ]
    want = _two_state_availability(rates, change, mu, horizon)
    run = rbd.availability(horizon, mc_samples=3000, seed=11)
    # About four standard errors either side.
    interval = run.mean_availability_interval(0.9999)
    assert interval.method == "simulated"
    assert interval.lower < want < interval.upper


def _renewals(model, change, horizon, runs, seed):
    """Failures over ``[0, horizon]`` of a unit renewed at each failure in
    no time, its life a Weibull proportional hazards model at load 1
    before ``change`` and 2 after: simulated plainly, each life inverted
    from the closed form ``H(a, Z) = k_Z a ** beta``."""
    rng = np.random.default_rng(seed)
    beta = math.log(_H(model, 2.0, 1.0) / _H(model, 1.0, 1.0)) / math.log(2)
    k1, k2 = _H(model, 1.0, 1.0), _H(model, 1.0, 2.0)
    counts = []
    for _ in range(runs):
        t, n = 0.0, 0
        while True:
            e = rng.exponential()
            before = max(change - t, 0.0)
            if e <= k1 * before**beta:
                life = (e / k1) ** (1.0 / beta)
            else:
                left = e - k1 * before**beta
                life = (left / k2 + before**beta) ** (1.0 / beta)
            t += life
            if t >= horizon:
                break
            n += 1
        counts.append(n)
    return np.asarray(counts, dtype=float)


def test_a_renewal_resets_the_age_while_the_schedule_runs_on(weibull_ph):
    """Each failure renews the unit (new, age 0), and the load it runs at
    is the diagram's clock's: the simulated failures over a window agree
    with a renewal process simulated plainly here, within four standard
    errors of their difference."""
    change, horizon = 50.0, 200.0
    rbd = RepairableRBD(
        EDGES,
        {
            "a": _unit(
                weibull_ph,
                STEPS([0.0, change], [[1.0], [2.0]]),
                "instant",
            )
        },
    )
    run = rbd.availability(horizon, mc_samples=1500, seed=5)
    mine = run.system_failures / run.n_simulations
    plain = _renewals(weibull_ph, change, horizon, 4000, seed=9)
    # The run's failures per simulation: its own spread, as the plain
    # run's (each simulation's count has about the same spread).
    spread = plain.std(ddof=1)
    error = spread * math.sqrt(1.0 / 1500 + 1.0 / plain.size)
    assert abs(mine - plain.mean()) < 4.0 * error


def test_imperfect_repair_keeps_a_virtual_age_under_the_schedule(
    weibull_ph,
):
    """A wearing-out unit repaired imperfectly (Kijima I, q = 1: as old as
    before) fails more often than one renewed, under the same schedule."""
    schedule = STEPS([0.0, 50.0], [[1.0], [2.0]])
    beta = math.log(_H(weibull_ph, 2.0, 1.0) / _H(weibull_ph, 1.0, 1.0))
    assert beta / math.log(2.0) > 1.0  # wears out
    renewed = RepairableRBD(EDGES, {"a": _unit(weibull_ph, schedule)})
    worn = RepairableRBD(
        EDGES,
        {
            "a": _unit(
                weibull_ph, schedule, repair={"model": "kijima1", "q": 1.0}
            )
        },
    )
    a = renewed.availability(200.0, mc_samples=300, seed=2)
    b = worn.availability(200.0, mc_samples=300, seed=2)
    assert b.system_failures > 1.2 * a.system_failures


def test_age_replacement_and_tests_run_with_a_schedule(weibull_ph):
    """Preventive replacement and tests for hidden failures run with a
    covariate schedule, and replacing before failure cuts the failures."""
    schedule = STEPS([0.0, 50.0], [[1.0], [2.0]])
    plain = RepairableRBD(EDGES, {"a": _unit(weibull_ph, schedule)})
    replaced = RepairableRBD(
        EDGES,
        {"a": _unit(weibull_ph, schedule, preventive={"interval": 5.0})},
    )
    tested = RepairableRBD(
        EDGES,
        {"a": _unit(weibull_ph, schedule, inspection={"interval": 10.0})},
    )
    a = plain.availability(200.0, mc_samples=200, seed=4)
    b = replaced.availability(200.0, mc_samples=200, seed=4)
    c = tested.availability(200.0, mc_samples=200, seed=4)
    assert b.system_failures < a.system_failures
    assert c.mean_availability < a.mean_availability


def test_seeded_runs_repeat_and_split_alike(exp_ph):
    """A seeded run gives the same numbers again, run in one block or
    in two processes; each simulation's draws come from its streams."""
    rbd = RepairableRBD(
        EDGES, {"a": _unit(exp_ph, STEPS([0.0, 30.0], [[1.0], [2.0]]))}
    )
    first = rbd.availability(100.0, mc_samples=200, seed=21)
    again = rbd.availability(100.0, mc_samples=200, seed=21)
    split = rbd.availability(100.0, mc_samples=200, seed=21, n_jobs=2)
    other = rbd.availability(100.0, mc_samples=200, seed=22)
    assert first.mean_availability == again.mean_availability
    assert first.mean_availability == split.mean_availability
    assert first.system_failures == split.system_failures
    assert first.mean_availability != other.mean_availability


def test_a_start_from_an_age_draws_the_life_left_from_0(weibull_ph):
    """A proportional hazards unit started at an age draws what is left
    of its life from that age, at the covariates in force from 0: an old
    wearing-out unit fails sooner than a new one."""
    rbd = RepairableRBD(
        EDGES,
        {"a": _unit(weibull_ph, STEPS([0.0, 50.0], [[1.0], [2.0]]))},
    )
    new = rbd.availability(20.0, mc_samples=300, seed=8)
    old = rbd.availability(
        20.0, mc_samples=300, seed=8, state={"a": NodeState(age=80.0)}
    )
    assert old.mean_availability < new.mean_availability


def test_a_timeline_shows_the_unit_failing_faster_after_the_step(exp_ph):
    """At four times the hazard after the change, the failures after it
    come about four times as often (in each stretch's up time)."""
    change, horizon = 100.0, 200.0
    rbd = RepairableRBD(
        EDGES,
        {"a": _unit(exp_ph, STEPS([0.0, change], [[1.0], [2.0]]), "instant")},
    )
    timelines = rbd.simulate_timelines(horizon, mc_samples=400, seed=3)
    before = after = 0
    for i in range(timelines.n_simulations):
        # Each failure's down and up, at one time (instant repairs).
        failures = timelines.components["a"][i].changes[::2]
        before += int(np.sum(failures < change))
        after += int(np.sum(failures >= change))
    ratio = (after / (horizon - change)) / (before / change)
    rates = [
        float(exp_ph.hf(np.array([1.0]), np.array([[z]]))[0])
        for z in (1.0, 2.0)
    ]
    assert ratio == pytest.approx(rates[1] / rates[0], rel=0.15)


def test_a_diagram_with_a_schedule_saves_and_loads(weibull_ph):
    cyclic = surv.StepSchedule.cyclic([0.0, 60.0], [[1.0], [2.0]], 100.0)
    for schedule in (STEPS([0.0, 50.0], [[1.0], [2.0]]), cyclic):
        rbd = RepairableRBD(EDGES, {"a": _unit(weibull_ph, schedule)})
        back = RepairableRBD.from_dict(json.loads(json.dumps(rbd.to_dict())))
        assert back._init_args["components"]["a"]["covariates"] == schedule
        assert (
            back.availability(100.0, mc_samples=40, seed=1).mean_availability
            == rbd.availability(100.0, mc_samples=40, seed=1).mean_availability
        )


# -- What is refused ---------------------------------------------------------


def test_the_exact_methods_refuse_a_schedule_saying_which_simulate(exp_ph):
    rbd = RepairableRBD(
        EDGES, {"a": _unit(exp_ph, STEPS([0.0, 30.0], [[1.0], [2.0]]))}
    )
    for call in (
        lambda: rbd.mean_availability(),
        lambda: rbd.point_availability(np.array([10.0])),
        lambda: rbd.mean_time_to_first_failure(),
        lambda: rbd.system_failure_frequency(),
    ):
        with pytest.raises(NotImplementedError, match="covariate schedule"):
            call()
    routes = rbd.analysis_routes()
    assert routes["mean_availability"].route == "refused"
    assert "availability, cost" in routes["mean_availability"].reason
    assert routes["availability"].route == "simulated"


def test_the_compiled_engine_leaves_a_schedule_to_python(exp_ph):
    rbd = RepairableRBD(
        EDGES, {"a": _unit(exp_ph, STEPS([0.0, 30.0], [[1.0], [2.0]]))}
    )
    with pytest.raises(NotImplementedError, match="covariate schedule"):
        rbd.availability(10.0, mc_samples=10, seed=1, engine="numba")


@pytest.mark.parametrize(
    "spec, error, match",
    [
        ({"reliability": "po"}, ValueError, "Proportional Odds"),
        ({"reliability": "cox"}, ValueError, "semiparametric"),
        ({"reliability": W([10, 2])}, ValueError, "regression model"),
        ({"covariates": [1.0]}, TypeError, "StepSchedule"),
        (
            {"covariates": STEPS([5.0, 30.0], [[1.0], [2.0]])},
            ValueError,
            "from 0",
        ),
        (
            {
                "covariates": surv.StepSchedule.from_intervals(
                    [0.0, 30.0], [30.0, 60.0], [[1.0], [2.0]]
                )
            },
            ValueError,
            "every time",
        ),
        (
            {"covariates": STEPS([0.0, 30.0], [[1.0, 0.0], [2.0, 0.0]])},
            ValueError,
            "fitted with 1",
        ),
        ({"duty": 0.5}, ValueError, "duty"),
        (
            {"standby": {"units": 2, "k": 1}},
            ValueError,
            "standby group",
        ),
        (
            {"reliability": "aft", "repair": {"model": "kijima1", "q": 0.5}},
            ValueError,
            "virtual age",
        ),
        (
            {
                "preventive": {
                    "interval": 10.0,
                    "policy": "condition",
                    "threshold": 0.1,
                }
            },
            ValueError,
            "on condition",
        ),
    ],
)
def test_what_a_schedule_cannot_go_with_is_refused(
    spec, error, match, exp_ph, weibull_aft
):
    x = np.array([5.0, 10.0, 20.0, 3.0, 4.0, 8.0, 12.0, 2.0])
    load = np.array([[1.0]] * 4 + [[2.0]] * 4)
    models = {
        "po": lambda: surv.WeibullPO.fit(x, Z=load),
        "cox": lambda: surv.CoxPH.fit(x, Z=load),
        "aft": lambda: weibull_aft,
    }
    spec = dict(spec)
    if isinstance(spec.get("reliability"), str):
        spec["reliability"] = models[spec["reliability"]]()
    unit = {**_unit(exp_ph, STEPS([0.0, 30.0], [[1.0], [2.0]])), **spec}
    with pytest.raises(error, match=match):
        RepairableRBD(EDGES, {"a": unit})


def test_a_common_cause_group_and_a_pausing_shut_are_refused(exp_ph):
    from repyability import BetaFactor, CCFGroup

    schedule = STEPS([0.0, 30.0], [[1.0], [2.0]])
    edges = [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")]
    unit = _unit(exp_ph, schedule)
    with pytest.raises(ValueError, match="common-cause group"):
        RepairableRBD(
            edges,
            {"a": unit, "b": dict(unit)},
            ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.1))],
        )
    with pytest.raises(ValueError, match="covariate schedule"):
        RepairableRBD(
            EDGES,
            {"a": unit},
            shutdowns={
                "shut": {
                    "period": 50.0,
                    "duration": 5.0,
                    "idle_ageing": False,
                }
            },
        )


def test_a_start_state_a_schedule_cannot_take_is_refused(exp_ph, weibull_aft):
    schedule = STEPS([0.0, 30.0], [[1.0], [2.0]])
    aft = RepairableRBD(EDGES, {"a": _unit(weibull_aft, schedule)})
    with pytest.raises(NotImplementedError, match="age has no one"):
        aft.availability(
            10.0, mc_samples=5, seed=1, state={"a": NodeState(age=3.0)}
        )
    tested = RepairableRBD(
        EDGES, {"a": _unit(exp_ph, schedule, inspection={"interval": 5.0})}
    )
    with pytest.raises(NotImplementedError, match="hidden"):
        tested.availability(
            10.0, mc_samples=5, seed=1, state={"a": NodeState(age=3.0)}
        )


def test_a_nonrepairable_given_a_regression_says_where_schedules_go(exp_ph):
    with pytest.raises(ValueError, match="'covariates'"):
        NonRepairable(exp_ph)
