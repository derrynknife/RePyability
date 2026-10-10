"""Replacement on condition in the long run (#145): a component inspected
every interval and replaced, at an inspection, when it is more likely than
a threshold to fail before the next. Its long-run values from its renewal
cycle, checked against block replacement (a threshold of 0 replaces at
every inspection), run to failure (a threshold no age reaches), following a
cycle to its end, and the simulation."""

import inspect

import numpy as np
import pytest
import surpyval as surv

from repyability import RepairableRBD
from repyability.rbd import _condition_replacement as condition
from repyability.rbd import _curves, _long_run
from repyability.rbd import routes as r

W, E, LN = (
    surv.Weibull.from_params,
    surv.Exponential.from_params,
    surv.LogNormal.from_params,
)
LIFE, REPAIR = W([100.0, 2.5]), LN([1.0, 0.5])
PAIR = [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")]


def unit(policy="condition", threshold=0.3, duration=None, **costs):
    preventive = {"interval": 30.0, "policy": policy}
    if policy == "condition":
        preventive["threshold"] = threshold
    if duration is not None:
        preventive["duration"] = duration
    for key in ("cost", "inspection_cost"):
        if key in costs:
            preventive[key] = costs.pop(key)
    return {
        "reliability": LIFE,
        "repairability": REPAIR,
        "preventive": preventive,
        **costs,
    }


LONG_RUN = (
    "mean_availability",
    "mean_unavailability",
    "system_failure_frequency",
    "mean_time_between_failures",
    "expected_cost_rate",
)


@pytest.mark.parametrize("duration", [None, E([0.5])])
def test_a_threshold_of_0_replaces_at_every_inspection(duration):
    # Any unit up at an inspection is then more likely than 0 to fail before
    # the next: block replacement.
    costs = dict(cost=30.0, repair_cost=100.0)
    on_condition = RepairableRBD(
        PAIR,
        {
            "a": unit(threshold=0.0, duration=duration, **costs),
            "b": unit(threshold=0.0),
        },
        downtime_cost_rate=50.0,
    )
    block = RepairableRBD(
        PAIR,
        {
            "a": unit("block", duration=duration, **costs),
            "b": unit("block"),
        },
        downtime_cost_rate=50.0,
    )
    for name in LONG_RUN:
        assert getattr(on_condition, name)() == pytest.approx(
            getattr(block, name)(), rel=1e-12
        )
    for a, b in zip(
        on_condition.birnbaum_importance().values(),
        block.birnbaum_importance().values(),
    ):
        assert a == pytest.approx(b, rel=1e-11)


def test_a_threshold_no_inspection_reaches_runs_to_failure():
    never = RepairableRBD(PAIR, {"a": unit(threshold=1.0), "b": unit()})
    plain = RepairableRBD(
        PAIR,
        {"a": {"reliability": LIFE, "repairability": REPAIR}, "b": unit()},
    )
    for name in LONG_RUN:
        assert getattr(never, name)() == pytest.approx(
            getattr(plain, name)(), rel=1e-6
        )
    cycle = _long_run._block_cycle(never, "a")
    assert cycle.replaced == 0.0


def test_the_rest_of_a_long_cycle_is_summed_as_it_falls():
    # Rarely replaced: a cycle of some 800 hours and 8 failures, whose tail
    # is summed as a geometric series once it falls at a steady rate;
    # following it to its end gives the same.
    quick = condition.condition_cycle(LIFE, REPAIR, E([0.5]), 30.0, 0.7)
    source = inspect.getsource(condition.condition_cycle).replace(
        "len(ratios) > 20", "len(ratios) > 10**9"
    )
    namespace = dict(vars(condition))
    exec(source, namespace)
    slow = namespace["condition_cycle"](LIFE, REPAIR, E([0.5]), 30.0, 0.7)
    assert quick.length == pytest.approx(801.28, rel=1e-5)
    for field in ("up", "length", "failures", "before", "after"):
        assert getattr(quick, field) == pytest.approx(
            getattr(slow, field), rel=1e-9
        )
    np.testing.assert_allclose(
        quick.availability, slow.availability, rtol=1e-9
    )


def test_the_simulation_agrees():
    threshold = 0.35
    rbd = RepairableRBD(
        [("s", "a"), ("a", "t")],
        {
            "a": unit(
                threshold=threshold,
                duration=E([0.5]),
                cost=30.0,
                inspection_cost=2.0,
                repair_cost=100.0,
            )
        },
    )
    # A long window, so that the start from new does not weigh.
    result = rbd.availability(200_000.0, mc_samples=20, seed=7)
    window = result.n_simulations * result.time_simulated_to
    assert result.system_uptime / window == pytest.approx(
        rbd.mean_availability(), abs=6e-4
    )
    assert result.system_failures / window == pytest.approx(
        rbd.system_failure_frequency(), rel=0.03
    )
    cost = result.cost.samples / result.time_simulated_to
    assert cost.mean() == pytest.approx(rbd.expected_cost_rate(), rel=0.015)
    # Each inspection the unit is up at is charged.
    inspections = result.cost.by_category["inspection"]
    cycle = _long_run._block_cycle(rbd, "a")
    assert inspections / result.time_simulated_to == pytest.approx(
        2.0 * cycle.before / 30.0, rel=0.015
    )


def test_the_routes():
    rbd = RepairableRBD(PAIR, {"a": unit(), "b": unit("block")})
    report = rbd.analysis_routes()
    route = report["mean_availability"]
    assert route.route == r.NUMERICAL
    assert "replacement on condition" in route.reason
    assert "a" in route.nodes
    # Over time too (#161): followed from one inspection to the next.
    over = report["point_availability"]
    assert over.route == r.NUMERICAL
    route, reason = _curves._node_over_time(rbd, "a", "availability")
    assert route == r.NUMERICAL and "replacement on condition" in reason
    assert 0.0 < float(np.ravel(rbd.point_availability(10.0))[0]) < 1.0
