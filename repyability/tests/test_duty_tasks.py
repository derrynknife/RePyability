"""Tasks counted in operating time (``"basis": "operating"``) on a block
that operates part of the time (``"duty"``).

A block of duty ``d`` ages only while it operates, so its life on the
diagram's clock is its operating life over ``d``; a task counted in its
operating hours falls due every ``N`` of them, which is every ``N / d``
on the clock. The constructor puts the task there, so each check here is
the same block given on the calendar by hand: its life with its scale
over ``d``, its task every ``N / d``. Those are the same objects, so the
exact values and the seeded simulations are the same to the last bit.
"""

import math

import numpy as np
import pytest
import surpyval as surv

from repyability import RepairableRBD
from repyability.rbd import _compiled
from repyability.rbd.serialisation import rbd_from_json, rbd_to_json

W = surv.Weibull.from_params
E = surv.Exponential.from_params
FIXED = surv.ExactEventTime.from_params
ONE = [("s", "p"), ("p", "t")]
PAIR = [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")]

#: The pump runs 60% of the time; its life is a Weibull of 5,000 running
#: hours (shape 2.5), and it is overhauled every 4,000 running hours.
DUTY = 0.6
ALPHA, BETA = 5000.0, 2.5
EVERY = 4000.0

needs_numba = pytest.mark.skipif(
    not _compiled.available(), reason="numba is not installed"
)


def pump(task="preventive", **options):
    """The pump given in running hours, its task counted in them."""
    spec = {
        "reliability": W([ALPHA, BETA]),
        "repairability": FIXED([48.0]),
        "duty": DUTY,
        "replace_cost": 30000.0,
        task: {**options, "basis": "operating"},
    }
    if task == "preventive":
        spec[task].setdefault("cost", 8000.0)
    return spec


def by_hand(task="preventive", times=("interval",), **options):
    """The same pump given on the calendar by hand: its life's scale and
    its task's times over the duty."""
    spec = pump(task, **options)
    del spec["duty"]
    spec["reliability"] = W([ALPHA / DUTY, BETA])
    given = {k: v for k, v in spec[task].items() if k != "basis"}
    for name in times:
        if name in given:
            given[name] = given[name] / DUTY
    spec[task] = given
    return spec


def same(first, second):
    """Two runs' results the same to the last bit."""
    assert first.system_failures == second.system_failures
    assert first.system_planned_outages == second.system_planned_outages
    assert first.sample_mean_availability == second.sample_mean_availability
    np.testing.assert_array_equal(first.uptimes, second.uptimes)


# -- age replacement every N running hours --------------------------------


def test_age_replacement_in_running_hours_is_every_n_over_d_hours():
    counted = RepairableRBD(ONE, {"p": pump(interval=EVERY)})
    calendar = RepairableRBD(ONE, {"p": by_hand(interval=EVERY)})
    assert counted._preventive["p"].interval == EVERY / DUTY
    for method in ("mean_availability", "expected_cost_rate"):
        assert getattr(counted, method)() == pytest.approx(
            getattr(calendar, method)(), rel=1e-15
        )
    t = np.array([100.0, 5000.0, 20000.0])
    np.testing.assert_allclose(
        counted.point_availability(t),
        calendar.point_availability(t),
        rtol=1e-15,
    )
    left, right = counted.expected_events(20000.0), calendar.expected_events(
        20000.0
    )
    assert left.system_failures == pytest.approx(right.system_failures, 1e-15)
    assert left.system_planned_outages == pytest.approx(
        right.system_planned_outages, 1e-15
    )
    options = dict(t_simulation=30000.0, mc_samples=200, seed=7)
    same(counted.availability(**options), calendar.availability(**options))
    np.testing.assert_array_equal(
        counted.cost(**options).samples, calendar.cost(**options).samples
    )


@needs_numba
def test_the_engines_agree_on_a_task_in_running_hours():
    counted = RepairableRBD(
        PAIR, {"a": pump(interval=EVERY), "b": pump(interval=3000.0)}
    )
    options = dict(
        t_simulation=30000.0, mc_samples=100, seed=3, control_variate=False
    )
    same(
        counted.availability(engine="python", **options),
        counted.availability(engine="numba", **options),
    )


def test_a_block_policy_and_a_condition_policy_in_running_hours():
    for options in (
        dict(interval=EVERY, policy="block", duration=FIXED([24.0])),
        dict(interval=1000.0, policy="condition", threshold=0.05),
    ):
        counted = RepairableRBD(ONE, {"p": pump(**options)})
        calendar = RepairableRBD(ONE, {"p": by_hand(**options)})
        assert counted.mean_availability() == pytest.approx(
            calendar.mean_availability(), rel=1e-15
        )
        run = dict(t_simulation=20000.0, mc_samples=100, seed=2)
        same(counted.availability(**run), calendar.availability(**run))


def test_an_opportunity_in_running_hours():
    # Renewed early at a stop of its group from an age of 3,000 running
    # hours, 5,000 on the calendar.
    def system(spec):
        return RepairableRBD(
            PAIR,
            {"a": spec, "b": {**spec}},
            maintenance_groups={"g": {"setup_cost": 500.0}},
        )

    options = dict(interval=EVERY, opportunity=3000.0)
    counted = system({**pump(**options), "group": "g"})
    calendar = system(
        {**by_hand(times=("interval", "opportunity"), **options), "group": "g"}
    )
    assert counted._preventive["a"].opportunity == 3000.0 / DUTY
    run = dict(t_simulation=30000.0, mc_samples=100, seed=4)
    same(counted.availability(**run), calendar.availability(**run))


def test_tests_in_running_hours():
    # Tested every 500 running hours from 100, a full test every 1,500.
    options = dict(
        interval=500.0, offset=100.0, coverage=0.7, full_test=1500.0
    )
    times = ("interval", "offset", "full_test")
    counted = RepairableRBD(ONE, {"p": pump("inspection", **options)})
    calendar = RepairableRBD(
        ONE, {"p": by_hand("inspection", times, **options)}
    )
    schedule = counted._inspection["p"]
    assert schedule.interval == 500.0 / DUTY
    assert schedule.offset == 100.0 / DUTY
    assert schedule.full_test == pytest.approx(1500.0 / DUTY, rel=1e-15)
    run = dict(t_simulation=20000.0, mc_samples=100, seed=5)
    same(counted.availability(**run), calendar.availability(**run))


# -- with maintenance that restores partly ---------------------------------


def test_a_restoring_task_in_running_hours_is_on_the_calendar():
    # The virtual age is calendar time, as the life on the calendar is;
    # q is a fraction, with no unit.
    options = dict(
        interval=EVERY,
        policy="block",
        restoration={"model": "kijima1", "q": 0.3},
        replace_every=3,
        replacement_cost=30000.0,
    )
    counted = RepairableRBD(ONE, {"p": pump(**options)})
    calendar = RepairableRBD(ONE, {"p": by_hand(**options)})
    run = dict(t_simulation=40000.0, mc_samples=100, seed=3)
    same(counted.availability(**run), calendar.availability(**run))
    np.testing.assert_array_equal(
        counted.cost(**run).samples, calendar.cost(**run).samples
    )


# -- with shutdowns ---------------------------------------------------------


def shut(spec, idle_ageing):
    return RepairableRBD(
        ONE,
        {"p": spec},
        shutdowns={
            "line": {
                "period": 8760.0,
                "offset": 2000.0,
                "duration": 240.0,
                "idle_ageing": idle_ageing,
            }
        },
    )


@pytest.mark.parametrize("idle_ageing", [True, False])
def test_a_block_on_a_duty_in_a_shut_ages_at_its_duty_or_stands_still(
    idle_ageing,
):
    # Idle in the shut it keeps ageing on its calendar life (at its duty,
    # on average), or it stands still: as the same life given on the
    # calendar does. Ageing, its overhaul in running hours goes on; a
    # block with a schedule of its own is not stood still yet.
    def given(spec):
        if not idle_ageing:
            del spec["preventive"]
        return spec

    counted = shut(given(pump(interval=EVERY)), idle_ageing)
    calendar = shut(given(by_hand(interval=EVERY)), idle_ageing)
    assert counted.mean_availability() == pytest.approx(
        calendar.mean_availability(), rel=1e-15
    )
    events = [d.expected_events(30000.0) for d in (counted, calendar)]
    assert events[0].system_failures == pytest.approx(
        events[1].system_failures, rel=1e-15
    )
    run = dict(t_simulation=30000.0, mc_samples=100, seed=8)
    same(counted.availability(**run), calendar.availability(**run))


@pytest.mark.parametrize("task", ["preventive", "inspection"])
def test_a_task_at_a_shut_is_not_counted_in_running_hours(task):
    with pytest.raises(ValueError, match="at a shutdown is on the calendar"):
        shut(pump(task, shutdown="line"), True)
    # On the calendar (the default), it is done at the shut.
    spec = pump(task, shutdown="line")
    spec[task]["basis"] = "calendar"
    assert task in ("preventive", "inspection")
    shut(spec, True).mean_availability()


# -- planned against unplanned ---------------------------------------------


def test_the_split_on_a_duty_with_a_task_in_running_hours():
    # The pump and a spare in parallel, each overhauled every 4,000
    # running hours for a day: exact against simulated.
    spec = pump(interval=EVERY, duration=FIXED([24.0]))
    rbd = RepairableRBD(PAIR, {"a": spec, "b": dict(spec)})
    calendar = RepairableRBD(
        PAIR,
        {
            "a": by_hand(interval=EVERY, duration=FIXED([24.0])),
            "b": by_hand(interval=EVERY, duration=FIXED([24.0])),
        },
    )
    effect = rbd.maintenance_effect()
    assert effect.method == "exact"
    other = calendar.maintenance_effect()
    for name in (
        "system_planned_downtime",
        "system_unplanned_downtime",
        "system_failures_avoided",
    ):
        assert getattr(effect, name) == pytest.approx(
            getattr(other, name), rel=1e-15
        )
    t, n = 40000.0, 2000
    events = rbd.expected_events(t)
    run = rbd.availability(t, mc_samples=n, seed=3, control_variate=False)
    down = t - np.asarray(run.uptimes)
    error = math.sqrt(np.mean(down**2) / n)
    for exact, simulated in (
        (events.system_planned_downtime, run.system_planned_downtime / n),
        (events.system_unplanned_downtime, run.system_unplanned_downtime / n),
    ):
        assert abs(exact - simulated) <= 4 * error
    window = rbd.maintenance_effect(t)
    simulated = rbd.maintenance_effect(t, method="simulate", mc_samples=n)
    assert (
        abs(window.system_failures_avoided - simulated.system_failures_avoided)
        <= 4 * simulated.standard_errors["downtime_saved"] / 24.0 + 1.0 / n
    )


# -- levers, interval choices and saving -----------------------------------


def test_the_interval_lever_is_in_running_hours():
    counted = RepairableRBD(ONE, {"p": pump(interval=EVERY)})
    levers = {lever.name: lever.value for lever in counted.levers()}
    assert levers["preventive.interval"] == EVERY
    assert levers["reliability.alpha"] == ALPHA
    moved = counted.with_levers({("p", "preventive.interval"): 3000.0})
    assert moved._preventive["p"].interval == 3000.0 / DUTY
    assert (
        moved.mean_availability()
        == RepairableRBD(ONE, {"p": pump(interval=3000.0)}).mean_availability()
    )
    # Per running hour: the calendar's sensitivity, which is per hour on
    # the calendar, times the calendar hours a running hour takes.
    calendar = RepairableRBD(ONE, {"p": by_hand(interval=EVERY)})
    per_running = counted.parameter_sensitivity()["p"]["preventive.interval"]
    per_hour = calendar.parameter_sensitivity()["p"]["preventive.interval"]
    assert per_running == pytest.approx(per_hour / DUTY, rel=1e-6)


def test_the_interval_choices_are_in_running_hours():
    counted = RepairableRBD(ONE, {"p": pump(interval=EVERY)})
    calendar = RepairableRBD(ONE, {"p": by_hand(interval=EVERY)})
    best, other = (
        counted.optimal_replacement_intervals(),
        calendar.optimal_replacement_intervals(),
    )
    assert best.intervals["p"] == pytest.approx(
        other.intervals["p"] * DUTY, rel=1e-6
    )
    assert best.cost_rate == pytest.approx(other.cost_rate, rel=1e-9)
    allowed = [2000.0, 3000.0, 4000.0, 5000.0]
    chosen = counted.optimal_replacement_intervals(allowed=allowed)
    from_calendar = calendar.optimal_replacement_intervals(
        allowed=[a / DUTY for a in allowed]
    )
    assert chosen.intervals["p"] * 1.0 / DUTY == pytest.approx(
        from_calendar.intervals["p"], rel=1e-15
    )
    assert chosen.cost_rate == from_calendar.cost_rate
    # A plan's intervals go back in as given.
    again = counted.with_intervals(chosen)
    assert again._init_args["components"]["p"]["preventive"]["interval"] == (
        chosen.intervals["p"]
    )
    assert again.expected_cost_rate() == pytest.approx(chosen.cost_rate, 1e-12)


def test_the_test_interval_choices_are_in_running_hours():
    def tested(spec):
        return RepairableRBD(
            PAIR, {"a": spec, "b": {**spec}, **{}}, on_infeasible_rbd="raise"
        )

    options = dict(
        interval=500.0, duration=FIXED([2.0]), cost=50.0, offset=100.0
    )
    counted = tested({**pump("inspection", **options), "repair_cost": 1.0})
    calendar = tested(
        {
            **by_hand("inspection", ("interval", "offset"), **options),
            "repair_cost": 1.0,
        }
    )
    allowed = [250.0, 500.0, 1000.0]
    chosen = counted.optimal_inspection_intervals(
        allowed=allowed, offset_shares="stagger"
    )
    from_calendar = calendar.optimal_inspection_intervals(
        allowed=[a / DUTY for a in allowed], offset_shares="stagger"
    )
    for node in "ab":
        assert chosen.intervals[node] / DUTY == pytest.approx(
            from_calendar.intervals[node], rel=1e-15
        )
    assert chosen.cost_rate == pytest.approx(from_calendar.cost_rate, 1e-12)
    again = counted.with_intervals(chosen)
    assert again.expected_cost_rate() == pytest.approx(chosen.cost_rate, 1e-12)


def test_saving_keeps_the_running_hours_and_the_basis():
    counted = RepairableRBD(ONE, {"p": pump(interval=EVERY)})
    again = rbd_from_json(rbd_to_json(counted))
    task = again._init_args["components"]["p"]["preventive"]
    assert task["interval"] == EVERY and task["basis"] == "operating"
    assert again._preventive["p"].interval == EVERY / DUTY
    assert again.mean_availability() == counted.mean_availability()


# -- what is refused ---------------------------------------------------------


def test_a_task_in_running_hours_needs_a_duty():
    spec = pump(interval=EVERY)
    del spec["duty"]
    with pytest.raises(ValueError, match="give the block's duty, the share"):
        RepairableRBD(ONE, {"p": spec})


def test_a_calendar_policy_is_not_counted_in_running_hours():
    from repyability.maintenance import CalendarPolicy

    spec = pump(policy="calendar")
    spec["preventive"]["calendar"] = object.__new__(CalendarPolicy)
    del spec["preventive"]["cost"]
    with pytest.raises(ValueError, match="'calendar' policy's thresholds"):
        RepairableRBD(ONE, {"p": spec})


def test_the_basis_is_checked():
    spec = pump(interval=EVERY)
    spec["preventive"]["basis"] = "running"
    with pytest.raises(ValueError, match="'calendar' or 'operating'"):
        RepairableRBD(ONE, {"p": spec})


def test_a_basis_on_the_calendar_or_a_duty_of_one_changes_nothing():
    plain = {
        "reliability": W([ALPHA, BETA]),
        "repairability": FIXED([48.0]),
        "preventive": {"interval": EVERY, "cost": 8000.0},
    }
    given = {**plain, "preventive": {**plain["preventive"], "basis": None}}
    calendar = {
        **plain,
        "preventive": {**plain["preventive"], "basis": "calendar"},
    }
    whole = {
        **plain,
        "duty": 1.0,
        "preventive": {**plain["preventive"], "basis": "operating"},
    }
    values = {
        RepairableRBD(ONE, {"p": spec}).expected_cost_rate()
        for spec in (plain, given, calendar, whole)
    }
    assert len(values) == 1
