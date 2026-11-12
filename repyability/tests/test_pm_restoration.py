"""Preventive maintenance that restores a unit only partly (Kijima): a
``"preventive"`` spec's ``"restoration"``, ``"replace_every"`` and
``"replacement_cost"`` (UC-22). The simulation is checked against a direct
simulation written here from Kijima's definition in plain numpy, against
an exact long-run cost rate (``Repairable.overhaul_replace_cost_rate``),
and against the diagram without the restoration where the two must agree.
"""

import math
import warnings

import numpy as np
import pytest
import surpyval as surv

from repyability import NodeState, Repairable, RepairableRBD
from repyability.rbd import _compiled, _event_loop, routes
from repyability.tests import catalogue

EDGES = [("s", "a"), ("a", "t")]
ALPHA, BETA = 1000.0, 2.5
LIFE = surv.Weibull.from_params([ALPHA, BETA])
#: Standard errors two estimates may be apart.
Z = 4.0


def unit(preventive, repairability="instant", **more):
    return RepairableRBD(
        EDGES,
        {
            "a": {
                "reliability": LIFE,
                "repairability": repairability,
                "preventive": preventive,
                **more,
            }
        },
    )


def direct(
    policy,
    interval,
    q,
    kijima,
    *,
    horizon,
    n,
    seed,
    every=None,
    duration=0.0,
    repair=0.0,
    threshold=None,
    virtual=0.0,
    age=0.0,
    since_scheduled=0.0,
):
    """Each simulation's failures, tasks that restore, tasks that replace
    and up time, written out from the definition: a life from virtual age
    ``a`` is the ``X`` with ``H(a + X) = H(a) + E``, ``E`` exponential; a
    failure is repaired as new (in ``repair``); a task after an operating
    time ``x`` since the last action takes the virtual age ``v`` to ``v +
    q x`` (Kijima I) or ``q (v + x)`` (Kijima II), or at every ``every``-th
    task since the unit was put in as new replaces it, and takes
    ``duration``, the unit not ageing meanwhile. Tasks fall at multiples
    of ``interval`` (``"block"``, skipped while the unit is down), at
    ``interval`` after the last action (``"age"``), or at the first
    inspection on the block calendar at which the unit is more likely
    than ``threshold`` to fail before the next, given its age then
    (``"condition"``). The unit starts at virtual ``virtual`` at its last
    action, ``age`` before 0, its calendar ``since_scheduled`` past its
    last multiple."""

    def H(x):
        return (x / ALPHA) ** BETA

    def multiple_after(t):
        due = (math.floor((t + since_scheduled) / interval) + 1) * interval
        due -= since_scheduled
        return due if due > t else due + interval

    rng = np.random.default_rng(seed)
    out = np.zeros((n, 4))
    for r in range(n):
        t, start, v, count = 0.0, -age, virtual, 0
        failures = restored = replaced = 0
        up = 0.0
        while True:
            now = v + t - start
            fail = t + ALPHA * (H(now) + rng.exponential()) ** (1 / BETA) - now
            if policy == "block":
                due = multiple_after(t)
            elif policy == "age":
                due = max(start + interval, t)
            else:
                due, check = math.inf, multiple_after(t)
                while check < min(fail, horizon):
                    later = multiple_after(check)
                    a, b = v + check - start, v + later - start
                    if -math.expm1(H(a) - H(b)) > threshold:
                        due = check
                        break
                    check = later
            if fail <= due:
                if fail >= horizon:
                    up += horizon - t
                    break
                up += fail - t
                failures += 1
                t = fail + repair
                start, v, count = t, 0.0, 0
                continue
            if due >= horizon:
                up += horizon - t
                break
            up += due - t
            x = due - start
            if every is not None and count + 1 >= every:
                v, count = 0.0, 0
                replaced += 1
            else:
                v = v + q * x if kijima == "kijima1" else q * (v + x)
                count += 1
                restored += 1
            t = start = due + duration
        out[r] = failures, restored, replaced, up
    return out


def counted(preventive, horizon, n, seed, **more):
    """The simulation's failures, tasks that restore and tasks that
    replace in each run, and its up times: three runs of the same
    simulations, each pricing one kind of event at 1."""

    def run(**costs):
        given = {
            key: value
            for key, value in preventive.items()
            if key not in ("cost", "replacement_cost")
        }
        for key in ("cost", "replacement_cost"):
            if key in costs:
                given[key] = costs.pop(key)
        rbd = unit(given, **costs, **more)
        return rbd.availability(
            horizon, mc_samples=n, seed=seed, control_variate=False
        )

    failed = run(repair_cost=1.0)
    columns = [failed.cost.samples]
    if "replace_every" in preventive:
        columns.append(run(cost=1.0, replacement_cost=0.0).cost.samples)
        columns.append(run(replacement_cost=1.0).cost.samples)
    else:
        columns.append(run(cost=1.0).cost.samples)
        columns.append(np.zeros(n))
    columns.append(failed.uptimes)
    return np.column_stack(columns)


def agree(simulated: np.ndarray, reference: np.ndarray) -> None:
    """Each column's mean within ``Z`` standard errors of the
    difference."""
    for column in range(simulated.shape[1]):
        a, b = simulated[:, column], reference[:, column]
        error = math.sqrt(
            a.var(ddof=1) / a.size + b.var(ddof=1) / b.size + 1e-300
        )
        assert abs(a.mean() - b.mean()) <= Z * error + 1e-12, (
            column,
            a.mean(),
            b.mean(),
            error,
        )


CASES = {
    "block, Kijima I": (
        dict(policy="block", interval=400.0, q=0.3, kijima="kijima1"),
        {},
    ),
    "block, Kijima II": (
        dict(policy="block", interval=400.0, q=0.3, kijima="kijima2"),
        {},
    ),
    "age, Kijima I": (
        dict(policy="age", interval=400.0, q=0.3, kijima="kijima1"),
        {},
    ),
    "age, Kijima II, repairs that take time": (
        dict(policy="age", interval=400.0, q=0.6, kijima="kijima2"),
        dict(repair=50.0),
    ),
    "block, every third replaces": (
        dict(policy="block", interval=400.0, q=0.3, kijima="kijima1"),
        dict(every=3),
    ),
    "block, tasks and repairs that take time": (
        dict(policy="block", interval=400.0, q=0.5, kijima="kijima1"),
        dict(duration=40.0, repair=50.0),
    ),
    "on condition, Kijima I": (
        dict(policy="condition", interval=100.0, q=0.3, kijima="kijima1"),
        dict(threshold=0.1),
    ),
    "on condition, Kijima II, every second replaces": (
        dict(policy="condition", interval=100.0, q=0.5, kijima="kijima2"),
        dict(threshold=0.05, every=2),
    ),
}


def preventive_of(case: dict, options: dict) -> dict:
    out = {
        "interval": case["interval"],
        "policy": case["policy"],
        "restoration": {"model": case["kijima"], "q": case["q"]},
    }
    if options.get("every") is not None:
        out["replace_every"] = options["every"]
    if options.get("duration"):
        out["duration"] = surv.ExactEventTime.from_params(
            [options["duration"]]
        )
    if options.get("threshold") is not None:
        out["threshold"] = options["threshold"]
    return out


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_simulation_follows_kijimas_definition(name):
    case, options = CASES[name]
    horizon, n = 5000.0, 3000
    repair = options.get("repair", 0.0)
    simulated = counted(
        preventive_of(case, options),
        horizon,
        n,
        seed=11,
        repairability=(
            surv.ExactEventTime.from_params([repair]) if repair else "instant"
        ),
    )
    reference = direct(
        **case, horizon=horizon, n=n, seed=12, **options  # type: ignore
    )
    agree(simulated, reference)
    # The tasks restore the unit (and with every, also replace it).
    assert reference[:, 1].mean() > 1.0


def test_tasks_that_restore_partly_raise_the_failures():
    # Against the same tasks renewing the unit: more failures, as the
    # unit keeps some of its age.
    renewing = direct(
        "block", 400.0, 0.0, "kijima1", horizon=5000.0, n=2000, seed=1
    )
    partly = direct(
        "block", 400.0, 0.3, "kijima1", horizon=5000.0, n=2000, seed=1
    )
    assert partly[:, 0].mean() > 1.5 * renewing[:, 0].mean()


@pytest.mark.parametrize("model", ["kijima1", "kijima2"])
@pytest.mark.parametrize("policy", ["age", "block"])
def test_from_a_virtual_age(model, policy):
    case = dict(policy=policy, interval=400.0, q=0.4, kijima=model)
    state = dict(virtual=700.0, age=150.0)
    if policy == "block":
        state["since_scheduled"] = 150.0
    n = 3000
    rbd = unit(
        dict(preventive_of(case, {})),
        repair_cost=1.0,
    )
    run = rbd.availability(
        3000.0,
        mc_samples=n,
        seed=4,
        control_variate=False,
        state={
            "a": NodeState(
                age=150.0,
                virtual_age=700.0,
                since_scheduled=state.get("since_scheduled") or None,
            )
        },
    )
    simulated = np.column_stack([run.cost.samples, run.uptimes])
    reference = direct(**case, horizon=3000.0, n=n, seed=5, **state)
    agree(simulated, reference[:, [0, 3]])
    # Older than new: more failures than from new.
    new = direct(**case, horizon=3000.0, n=n, seed=5)
    assert reference[:, 0].mean() > new[:, 0].mean() + 0.1
    # Seeded: the same again.
    again = rbd.availability(
        3000.0,
        mc_samples=n,
        seed=4,
        control_variate=False,
        state={
            "a": NodeState(
                age=150.0,
                virtual_age=700.0,
                since_scheduled=state.get("since_scheduled") or None,
            )
        },
    )
    np.testing.assert_array_equal(again.cost.samples, run.cost.samples)


def test_the_long_run_cost_rate_of_minimal_repair_and_overhauls_is_exact():
    # Minimal repairs in no time, Kijima II overhauls that keep 0.6 of the
    # age, every second replacing the unit: the cost rate over whole
    # cycles is Repairable.overhaul_replace_cost_rate's, exactly.
    interval, cycles, kept, cr, co, replace = 177.7, 2, 0.6, 1.0, 5.0, 20.0
    pump = Repairable(surv.CrowAMSAA.from_params([100.0, 2.5]))
    pump.set_repair_and_overhaul_costs(cr=cr, co=co)
    rate = pump.overhaul_replace_cost_rate(interval, cycles, kept, replace)
    rbd = RepairableRBD(
        EDGES,
        {
            "a": {
                "reliability": surv.Weibull.from_params([100.0, 2.5]),
                "repairability": "instant",
                "repair": {"model": "kijima1", "q": 1.0},
                "repair_cost": cr,
                "preventive": {
                    "interval": interval,
                    "policy": "block",
                    "cost": co,
                    "restoration": {"model": "kijima2", "q": kept},
                    "replace_every": cycles,
                    "replacement_cost": replace,
                },
            }
        },
    )
    whole = 20
    horizon = whole * cycles * interval
    n = 2000
    cost = rbd.cost(horizon, mc_samples=n, seed=3, control_variate=False)
    # The replacement at the window's end falls outside it.
    expected = rate * horizon - replace
    error = cost.samples.std(ddof=1) / math.sqrt(n)
    assert abs(cost.mean - expected) <= Z * error
    assert error < 0.01 * expected


@pytest.mark.parametrize("model", ["kijima1", "kijima2"])
def test_as_bad_as_old_tasks_leave_the_failures_as_without_them(model):
    # q = 1 and tasks in no time: a service that changes nothing, so the
    # failures are those of no maintenance at all (the renewal function,
    # worked out exactly).
    horizon, n = 3000.0, 4000
    serviced = unit(
        {
            "interval": 250.0,
            "policy": "block",
            "restoration": {"model": model, "q": 1.0},
        },
        repair_cost=1.0,
    )
    plain = RepairableRBD(
        EDGES,
        {"a": {"reliability": LIFE, "repairability": "instant"}},
    )
    exact = plain.expected_failures(horizon)
    run = serviced.cost(horizon, mc_samples=n, seed=8, control_variate=False)
    error = run.samples.std(ddof=1) / math.sqrt(n)
    assert abs(run.mean - exact) <= Z * error


def test_a_restoration_as_new_is_the_task_as_it_was():
    # q = 0 under Kijima II (or Kijima I, the repairs renewing): today's
    # diagram, to the last bit, and its exact routes.
    def diagram(**restoration):
        preventive = {"interval": 300.0, "policy": "block", "cost": 2.0}
        preventive.update(restoration)
        return unit(
            preventive,
            repairability=surv.Exponential.from_params([0.1]),
            repair_cost=1.0,
        )

    today = diagram()
    for model in ("kijima1", "kijima2"):
        as_new = diagram(restoration={"model": model, "q": 0.0})
        assert not as_new._restoring
        assert as_new.analysis_routes() == today.analysis_routes()
        for rbd in (today, as_new):
            assert rbd.analysis_routes()["mission_availability"].route != (
                routes.REFUSED
            )
        runs = [
            rbd.availability(800.0, mc_samples=300, seed=6)
            for rbd in (today, as_new)
        ]
        assert runs[0].mean_availability == runs[1].mean_availability
        np.testing.assert_array_equal(runs[0].uptimes, runs[1].uptimes)
        np.testing.assert_array_equal(
            runs[0].cost.samples, runs[1].cost.samples
        )


def test_kijima_one_at_zero_after_imperfect_repairs_is_not_a_renewal():
    # Repaired imperfectly, a unit is older than new at its tasks, and
    # Kijima I with q = 0 keeps that age: the restoration stays. With
    # repairs that renew it, its virtual age is 0 at every task: as new.
    def diagram(**more):
        return unit(
            {"interval": 300.0, "restoration": {"model": "kijima1", "q": 0.0}},
            **more,
        )

    assert not diagram()._restoring
    assert "a" in diagram(repair={"model": "kijima1", "q": 0.5})._restoring


def test_the_exact_methods_refuse_and_say_to_simulate():
    rbd = unit(
        {
            "interval": 400.0,
            "policy": "block",
            "cost": 1.0,
            "restoration": {"model": "kijima1", "q": 0.3},
        },
        repair_cost=1.0,
    )
    report = rbd.analysis_routes()
    for method in (
        "mean_availability",
        "mission_availability",
        "expected_cost",
        "expected_failures",
        "point_availability",
    ):
        assert report[method].route == routes.REFUSED
        assert "restored only partly" in report[method].reason
        assert "Kijima I, q = 0.3" in report[method].reason
    with pytest.raises(NotImplementedError, match="restored only partly"):
        rbd.mean_availability()
    with pytest.raises(NotImplementedError, match="restored only partly"):
        rbd.expected_failures(1000.0)
    for method in ("availability", "cost", "simulate_timelines"):
        assert report[method].route == routes.SIMULATED
    assert "partial restoration" in report["availability"].engine_reason


def test_the_compiled_engines_hand_it_to_python():
    rbd = unit(
        {"interval": 400.0, "restoration": {"model": "kijima1", "q": 0.3}}
    )
    plan, _ = _event_loop._stream_plan(rbd, 100.0, 1, False)
    reason = "partial restoration by preventive maintenance"
    assert _compiled.unsupported(rbd, plan, None) == reason
    assert _compiled.unsupported(rbd, plan, None, numba=True) == reason


def test_a_task_that_restores_uses_no_spare():
    preventive = {
        "interval": 400.0,
        "policy": "block",
        "restoration": {"model": "kijima1", "q": 0.3},
        "replace_every": 3,
    }
    rbd = unit(preventive)
    n = 400
    spares = rbd.spares_demand(5000.0, method="simulate", mc_samples=n, seed=2)
    counts = counted(preventive, 5000.0, n, seed=2)
    # Its failures (each repair renews it) and every third task.
    used = counts[:, 0] + counts[:, 2]
    assert spares["a"].mean == pytest.approx(used.mean(), rel=1e-12)
    with pytest.raises(NotImplementedError, match="restored only partly"):
        rbd.spares_demand(5000.0)


def test_the_replacement_is_charged_its_own_cost():
    preventive = {
        "interval": 400.0,
        "policy": "block",
        "cost": 2.0,
        "restoration": {"model": "kijima1", "q": 0.3},
        "replace_every": 3,
    }
    n = 300
    counts = counted(preventive, 5000.0, n, seed=9)
    priced = unit({**preventive, "replacement_cost": 7.0}).cost(
        5000.0, mc_samples=n, seed=9, control_variate=False
    )
    np.testing.assert_allclose(
        priced.samples, 2.0 * counts[:, 1] + 7.0 * counts[:, 2]
    )
    # By default, the task's own cost.
    default = unit(preventive).cost(
        5000.0, mc_samples=n, seed=9, control_variate=False
    )
    np.testing.assert_allclose(
        default.samples, 2.0 * (counts[:, 1] + counts[:, 2])
    )


def test_a_fitted_generalised_renewal_gives_its_restoration():
    times = np.cumsum(np.random.default_rng(0).weibull(1.5, 60) * 10)
    fit = surv.GeneralizedRenewal.fit(times, dist=surv.Weibull, kijima="ii")
    rbd = unit({"interval": 400.0, "restoration": fit})
    given = rbd._init_args["components"]["a"]["preventive"]["restoration"]
    assert given == {"model": "kijima2", "q": float(fit.restoration)}
    written = unit({"interval": 400.0, "restoration": given})
    runs = [
        diagram.availability(3000.0, mc_samples=100, seed=1)
        for diagram in (rbd, written)
    ]
    np.testing.assert_array_equal(runs[0].uptimes, runs[1].uptimes)
    with pytest.raises(ValueError, match="not Kijima's.*SurPyval#833"):
        unit(
            {
                "interval": 400.0,
                "restoration": surv.ARA.fit(times, dist=surv.Weibull),
            }
        )


def test_it_is_saved_and_loaded():
    rbd = unit(
        {
            "interval": 400.0,
            "policy": "block",
            "cost": surv.LogNormal.from_params([1.0, 0.3]),
            "restoration": {"model": "kijima2", "q": 0.4},
            "replace_every": 2,
            "replacement_cost": 0.0,
        },
        repair_cost=1.0,
    )
    back = RepairableRBD.from_dict(rbd.to_dict())
    assert back._restoring == rbd._restoring
    assert back.costs["a"]["replacement_cost"] == 0.0
    runs = [
        diagram.cost(3000.0, mc_samples=100, seed=1) for diagram in (rbd, back)
    ]
    np.testing.assert_array_equal(runs[0].samples, runs[1].samples)


def test_its_restoration_factor_is_a_lever():
    rbd = unit(
        {"interval": 400.0, "restoration": {"model": "kijima1", "q": 0.3}}
    )
    (lever,) = [
        x for x in rbd.levers() if x.name == "preventive.restoration.q"
    ]
    assert (lever.value, lever.bounds) == (0.3, (0.0, 1.0))
    moved = rbd.with_levers({lever: 0.8})
    assert moved._restoring["a"].q == 0.8
    with pytest.raises(ValueError, match="between 0 and 1"):
        rbd.with_levers({lever: 1.5})


@pytest.mark.parametrize(
    "preventive, match",
    [
        (
            {"interval": 400.0, "restoration": {"model": "kijima3", "q": 0.3}},
            "restoration must be a dict of a 'model'.*GeneralizedRenewal",
        ),
        (
            {"interval": 400.0, "restoration": {"model": "kijima1", "q": 2}},
            "q must be a number in \\[0, 1\\].*as old as it was",
        ),
        (
            {"interval": 400.0, "replace_every": 3},
            "replace_every needs a restoration",
        ),
        (
            {
                "interval": 400.0,
                "restoration": {"model": "kijima1", "q": 0.3},
                "replacement_cost": 5.0,
            },
            "replacement_cost .* needs replace_every",
        ),
        (
            {
                "interval": 400.0,
                "restoration": {"model": "kijima1", "q": 0.3},
                "replace_every": 1,
            },
            "replace_every of 1 replaces the unit at every task",
        ),
        (
            {
                "interval": 400.0,
                "restoration": {"model": "kijima1", "q": 0.3},
                "replace_every": 2.5,
            },
            "replace_every must be a whole number of tasks of at least 2",
        ),
        (
            {
                "interval": 400.0,
                "restoration": {"model": "kijima2", "q": 0.0},
                "replace_every": 2,
            },
            "replace_every needs a restoration with q above 0",
        ),
        (
            {
                "interval": 400.0,
                "restoration": {"model": "kijima1", "q": 0.3},
                "opportunity": 200.0,
            },
            "not taken with an opportunity",
        ),
        (
            {
                "interval": 400.0,
                "restoration": {"model": "kijima1", "q": 0.3},
                "replace_every": 2,
                "replacement_cost": -1.0,
            },
            "replacement_cost must be finite and non-negative",
        ),
    ],
)
def test_what_it_refuses(preventive, match):
    with pytest.raises(ValueError, match=match):
        unit(preventive)


def test_it_refuses_a_calendar_policy_and_a_level():
    with pytest.raises(ValueError, match="'calendar' policy is worked out"):
        unit(
            {
                "policy": "calendar",
                "calendar": catalogue.calendar_policy(LIFE),
                "restoration": {"model": "kijima1", "q": 0.3},
            },
            replace_cost=1.0,
        )
    with pytest.raises(ValueError, match="degradation level is not one"):
        RepairableRBD(
            EDGES,
            {
                "a": {
                    "reliability": catalogue.wear(),
                    "repairability": "instant",
                    "preventive": {
                        "interval": 60.0,
                        "policy": "condition",
                        "level": 6.0,
                        "restoration": {"model": "kijima1", "q": 0.3},
                    },
                }
            },
        )


def test_the_states_it_takes():
    rbd = unit(
        {
            "interval": 400.0,
            "policy": "block",
            "restoration": {"model": "kijima1", "q": 0.3},
            "replace_every": 3,
        }
    )
    with pytest.raises(NotImplementedError, match="how many it has had"):
        rbd.availability(
            100.0, mc_samples=2, seed=1, state={"a": NodeState(age=50.0)}
        )
    # A new unit on its calendar is one that has had none.
    rbd.availability(
        100.0,
        mc_samples=2,
        seed=1,
        state={"a": NodeState(since_scheduled=50.0)},
    )
    with pytest.raises(NotImplementedError, match="taken by the simulations"):
        rbd.point_availability(
            np.array([10.0]), state={"a": NodeState(age=50.0)}
        )
    plain = RepairableRBD(
        EDGES, {"a": {"reliability": LIFE, "repairability": "instant"}}
    )
    with pytest.raises(ValueError, match="nor restored partly"):
        plain.availability(
            100.0,
            mc_samples=2,
            seed=1,
            state={"a": NodeState(age=5.0, virtual_age=5.0)},
        )


def test_a_run_conditioned_on_its_module_agrees_with_a_plain_one():
    # The restored unit is a module that needs simulating, beside a pair
    # worked out exactly.
    rbd = RepairableRBD(
        [("s", "a"), ("a", "b"), ("a", "c"), ("b", "t"), ("c", "t")],
        {
            "a": {
                "reliability": LIFE,
                "repairability": surv.Exponential.from_params([0.05]),
                "preventive": {
                    "interval": 400.0,
                    "policy": "block",
                    "restoration": {"model": "kijima1", "q": 0.3},
                },
            },
            **{
                node: {
                    "reliability": surv.Weibull.from_params([300.0, 1.5]),
                    "repairability": surv.Exponential.from_params([0.1]),
                }
                for node in "bc"
            },
        },
    )
    assert rbd._conditional_modules(set(), set()) == ["a"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        conditioned = rbd.availability(3000.0, mc_samples=1500, seed=3)
        plain = rbd.availability(
            3000.0,
            mc_samples=1500,
            seed=3,
            control_variate=False,
            conditional=False,
        )
    assert conditioned.mean_availability_interval().method == "conditional"
    error = plain.uptimes.std(ddof=1) / math.sqrt(1500) / 3000.0
    assert (
        abs(conditioned.mean_availability - plain.mean_availability)
        <= Z * error
    )
