"""The long run of a diagram with an upgrade: once upgraded, each
upgraded component has its new life for ever, so the long-run methods
give the values of the diagram built with that life (the time before
the upgrade, and its capital cost, ignored)."""

import numpy as np
import pytest
import surpyval as surv

from repyability import RepairableRBD
from repyability.rbd import routes

E = surv.Exponential.from_params
W = surv.Weibull.from_params
EDGES = [("s", "a"), ("a", "b"), ("b", "t")]
OLD, NEW = W([300.0, 2.0]), W([500.0, 2.5])


def diagram(a_life, upgraded=False, block=True):
    a = {
        "reliability": a_life,
        "repairability": E([0.05]),
        "repair_cost": 40.0,
    }
    if block:
        a["preventive"] = {"policy": "block", "interval": 200.0, "cost": 5.0}
    if upgraded:
        a["upgrade"] = {"at": 1000.0, "reliability": NEW, "cost": 900.0}
    b = {"reliability": E([1e-3]), "repairability": E([0.1])}
    return RepairableRBD(EDGES, {"a": a, "b": b}, downtime_cost_rate=3.0)


LONG_RUN = [
    ("mean_availability", ()),
    ("mean_unavailability", ()),
    ("expected_cost_rate", ()),
    ("system_failure_frequency", ()),
    ("mean_time_between_failures", ()),
    ("mean_up_time", ()),
    ("mean_down_time", ()),
    ("node_availability", ()),
    ("birnbaum_importance", ()),
    ("risk_achievement_worth", ()),
    ("replacement_cycle", ("a",)),
]


@pytest.mark.parametrize("block", [True, False])
@pytest.mark.parametrize("name, args", LONG_RUN)
def test_the_long_run_is_the_diagram_built_with_the_new_life(
    name, args, block
):
    if name == "replacement_cycle" and not block:
        pytest.skip("a cycle is block replacement's")
    upgraded = diagram(OLD, upgraded=True, block=block)
    built = diagram(NEW, block=block)
    ours = getattr(upgraded, name)(*args)
    theirs = getattr(built, name)(*args)
    if isinstance(ours, dict):
        assert ours.keys() == theirs.keys()
        for key in ours:
            assert ours[key] == pytest.approx(theirs[key], rel=1e-12)
    elif isinstance(ours, float):
        assert ours == pytest.approx(theirs, rel=1e-12)
    else:
        a, b = ours.to_dict(), theirs.to_dict()
        assert a.keys() == b.keys()
        for key in a:
            np.testing.assert_allclose(a[key], b[key], rtol=1e-12)
    assert upgraded.analysis_routes()[name].route == (
        built.analysis_routes()[name].route
    )


def test_a_memoryless_unit_upgraded_in_closed_form():
    lam, new, mu = 0.01, 0.002, 0.5
    rbd = RepairableRBD(
        [("s", "a"), ("a", "t")],
        {
            "a": {
                "reliability": E([lam]),
                "repairability": E([mu]),
                "upgrade": {"at": 50.0, "reliability": E([new])},
            }
        },
    )
    assert rbd.mean_availability() == pytest.approx(mu / (new + mu), 1e-12)
    assert rbd.system_failure_frequency() == pytest.approx(
        new * mu / (new + mu), 1e-12
    )


def test_the_routes_say_the_long_run_is_once_upgraded():
    rbd = diagram(OLD, upgraded=True)
    report = rbd.analysis_routes()
    route = report["mean_availability"]
    assert route.route in (routes.EXACT, routes.NUMERICAL)
    assert route.reason.startswith(
        "The long run once upgraded (each upgraded component with its new "
        "life, the time before its upgrade ignored): "
    )
    # Over a horizon the upgrade is followed as before.
    assert report["expected_npv"].route == routes.NUMERICAL
    assert report["total_cost"].route == routes.REFUSED
    with pytest.raises(NotImplementedError, match="once upgraded"):
        rbd.total_cost(100.0)


def test_values_over_time_still_follow_the_upgrade():
    """Asked at times, an importance measure follows the diagram with its
    upgrade, not the diagram after it."""
    rbd, built = diagram(OLD, upgraded=True), diagram(NEW)
    early = rbd.birnbaum_importance(x=np.array([100.0]))["b"]
    after = built.birnbaum_importance(x=np.array([100.0]))["b"]
    before = diagram(OLD).birnbaum_importance(x=np.array([100.0]))["b"]
    # Before the upgrade, the diagram is the one with the old life.
    np.testing.assert_allclose(early, before, rtol=1e-10)
    assert not np.allclose(early, after, rtol=1e-6)
