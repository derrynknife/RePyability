"""A block inspected at a shutdown's given times and kept unless it is due
(``"older_than"``, or on condition with a threshold), valued stretch by
stretch over a horizon (``_condition_replacement.
condition_plan_availability``): against the same block at a shutdown on a
period when the times are equally spaced, against renewing it at every
outage (a threshold of 0) and against no task (an age past the horizon),
and against the simulation, from new and from a state."""

import numpy as np
import pytest
import surpyval as surv

from repyability import NodeState, RepairableRBD
from repyability.rbd import routes

E = surv.Exponential.from_params
W = surv.Weibull.from_params
LIFE = W([200.0, 2.0])
TIMES = [30.0, 90.0, 210.0, 240.0]
Z = 4.0


def unit(preventive=None, calendar=None, duration=10.0):
    """A wearing unit, repaired at rate 0.5, held by shutdown "line"."""
    spec = {
        "reliability": LIFE,
        "repairability": E([0.5]),
        "repair_cost": 4.0,
    }
    if preventive is not None:
        spec["preventive"] = {"shutdown": "line", "cost": 3.0, **preventive}
    shut = {"duration": duration, **(calendar or {"times": TIMES})}
    return RepairableRBD(
        [("s", "a"), ("a", "t")], {"a": spec}, shutdowns={"line": shut}
    )


def plain(**more):
    return dict(control_variate=False, conditional=False, **more)


def test_equally_spaced_times_are_the_shutdown_on_that_period():
    T = 300.0
    given = unit({"older_than": 100.0}, {"times": [20.0, 120.0, 220.0]})
    period = unit({"older_than": 100.0}, {"period": 100.0, "offset": 20.0})
    t = np.linspace(0.0, T, 61)
    np.testing.assert_allclose(
        given.point_availability(t), period.point_availability(t), atol=1e-7
    )
    assert given.expected_cost(T).mean == pytest.approx(
        period.expected_cost(T).mean, rel=1e-7
    )
    a, b = given.expected_events(T), period.expected_events(T)
    assert a.node_preventive["a"] == pytest.approx(
        b.node_preventive["a"], rel=1e-7
    )


def test_a_threshold_of_0_renews_at_every_outage():
    T = 300.0
    kept = unit({"policy": "condition", "threshold": 0.0})
    renewed = unit({})
    t = np.linspace(0.0, T, 61)
    np.testing.assert_allclose(
        kept.point_availability(t), renewed.point_availability(t), atol=1e-7
    )
    assert kept.expected_events(T).node_failures["a"] == pytest.approx(
        renewed.expected_events(T).node_failures["a"], rel=1e-7
    )


def test_an_age_past_the_horizon_renews_nothing(monkeypatch):
    """An age the unit never reaches: the same as no task, to the
    recursion's own discretisation error, which is first order in the
    grid step and is bounded from it.

    Each stretch is worked out on a grid of ``N`` steps (here every
    stretch is short beside the life's spread, so ``N`` is
    ``_block_replacement._MIN_STEPS``, 2,000, and the step ``h`` of each
    stretch is its length over ``N``). The kept side follows the units it
    keeps through the condition recursion, which carries their failures
    and the starts after them in cells of one step, a first-order rule;
    so the two sides differ by ``C / N + O(1 / N**2)``.

    The order and the constant were measured before this bound was set,
    as the largest difference over the times checked: 2.17e-7 at
    ``N`` = 2,000, 1.09e-7 at 4,000 and 5.5e-8 at 8,000. It halves each
    time ``N`` doubles (ratios 1.99 and 1.98), so it is first order, and
    ``N`` times it, the constant ``C``, is 4.35e-4, 4.36e-4 and 4.41e-4.
    (The same holds on a period: an age past the horizon against no task
    with a shut every 120 differs by 2.7e-7, 1.4e-7 and 7.0e-8 at the
    same ``N``.) The bound is ``C / N`` with ``C`` rounded up to its next
    power of ten, 1e-3: 5e-7 at ``N`` = 2,000.

    The test checks the order too, from the values at ``N`` and ``2 N``:
    the difference shrinks by a factor between sqrt(2) and 2 sqrt(2) (an
    order between 1/2 and 3/2), so it is the grid's error, which vanishes
    as the grid is refined, not a fixed one."""
    from repyability.rbd import _block_replacement

    T, C = 300.0, 1e-3
    t = np.linspace(0.0, T, 61)

    def difference():
        kept = unit({"older_than": 1e6}).point_availability(t)
        return float(np.max(np.abs(kept - unit().point_availability(t))))

    N = _block_replacement._MIN_STEPS
    coarse = difference()
    assert coarse <= C / N
    assert unit({"older_than": 1e6}).expected_events(T).node_preventive[
        "a"
    ] == pytest.approx(0.0, abs=1e-9)
    monkeypatch.setattr(_block_replacement, "_MIN_STEPS", 2 * N)
    fine = difference()
    assert np.sqrt(2.0) * fine <= coarse <= 2.0 * np.sqrt(2.0) * fine


@pytest.mark.parametrize(
    "preventive",
    [{"older_than": 70.0}, {"policy": "condition", "threshold": 0.2}],
)
@pytest.mark.parametrize(
    "state", [None, {"a": NodeState(age=120.0)}, {"a": NodeState(alive=False)}]
)
def test_the_simulation_agrees(preventive, state):
    T, rbd = 300.0, unit(preventive)
    route = rbd.analysis_routes()["mission_availability"].route
    assert route in (routes.EXACT, routes.NUMERICAL)
    run = rbd.availability(
        T, mc_samples=20_000, seed=3, state=state, **plain()
    )
    interval = run.mean_availability_interval()
    exact = rbd.mission_availability(T, state=state)
    assert abs(interval.estimate - exact) <= Z * interval.standard_error
    cost = rbd.expected_cost(T, state=state).mean
    assert abs(run.cost.sample_mean - cost) <= Z * run.cost.sample_se


def test_a_test_at_given_times_is_still_simulated():
    rbd = unit()
    tested = RepairableRBD(
        rbd._init_args["edges"],
        {
            "a": {
                **rbd._init_args["components"]["a"],
                "inspection": {"shutdown": "line"},
            }
        },
        shutdowns=rbd._init_args["shutdowns"],
    )
    assert tested.analysis_routes()["expected_npv"].route == routes.REFUSED
    assert unit({"older_than": 70.0}).analysis_routes()[
        "expected_npv"
    ].route in (routes.EXACT, routes.NUMERICAL)
