"""Spares and the first failure across a repairable sub-system (a nested
``RepairableRBD``): each equal to the same system drawn flat."""

import math

import numpy as np
import pytest
import surpyval as surv

from repyability import PerfectReliability, RepairableRBD

E = surv.Exponential.from_params
W = surv.Weibull.from_params
PAIR = [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")]


def unit(rate, **more):
    return {"reliability": E([rate]), "repairability": E([0.05]), **more}


def worn(**more):
    return {
        "reliability": W([2000.0, 2.5]),
        "repairability": E([1 / 8.0]),
        "preventive": {"interval": 1000.0, "duration": E([0.25])},
        **more,
    }


def drawn_in(node, inner_edges, inner_components, prefix_edges, top):
    """The flat diagram: ``top``'s components, and the sub-system's at
    ``node``, each as ``(node, component)``, its ends junctions."""
    edges = [((node, u), (node, v)) for u, v in inner_edges]
    for u, v in prefix_edges:
        edges.append(
            ((node, "t") if u == node else u, (node, "s") if v == node else v)
        )
    components = dict(top)
    components.update({(node, n): c for n, c in inner_components.items()})
    components[(node, "s")] = components[(node, "t")] = PerfectReliability
    return RepairableRBD(edges, components)


def pumps_and_pair(inner_components=None):
    inner_components = inner_components or {"a": unit(1e-3), "b": unit(2e-3)}
    inner = RepairableRBD(PAIR, inner_components)
    outer = [("s", "p"), ("p", "q"), ("q", "t")]
    nested = RepairableRBD(outer, {"p": unit(5e-4), "q": inner})
    flat = drawn_in("q", PAIR, inner_components, outer, {"p": unit(5e-4)})
    return nested, flat


def test_spares_demand_counts_a_sub_system_component_as_flat():
    nested, flat = pumps_and_pair({"a": worn(), "b": unit(2e-3)})
    ours, theirs = nested.spares_demand(5000.0), flat.spares_demand(5000.0)
    assert list(ours) == ["p", ("q", "a"), ("q", "b")] == list(theirs)
    for key in ours:
        assert np.allclose(
            ours[key].probabilities, theirs[key].probabilities, atol=1e-9
        )
        assert ours[key].mean == pytest.approx(theirs[key].mean, abs=1e-9)
    fleet = nested.spares_demand(5000.0, fleet=3)[("q", "a")]
    assert fleet.mean == pytest.approx(
        flat.spares_demand(5000.0, fleet=3)[("q", "a")].mean, abs=1e-9
    )


def test_spares_stock_counts_a_sub_system_component_as_flat():
    nested, flat = pumps_and_pair({"a": worn(), "b": unit(2e-3)})
    ours = nested.spares_stock(500.0, fill_rate=0.95)
    theirs = flat.spares_stock(500.0, fill_rate=0.95)
    assert list(ours) == list(theirs)
    for key in ours:
        assert ours[key].stock == theirs[key].stock
        assert ours[key].fill_rate == pytest.approx(
            theirs[key].fill_rate, abs=1e-9
        )
        assert ours[key].stockout_probability == pytest.approx(
            theirs[key].stockout_probability, abs=1e-9
        )


def test_a_sub_system_within_a_sub_system_is_keyed_by_both():
    inner = RepairableRBD(PAIR, {"a": unit(1e-3), "b": unit(2e-3)})
    middle = RepairableRBD(
        [("s", "r"), ("r", "x"), ("x", "t")], {"r": inner, "x": unit(1e-3)}
    )
    top = RepairableRBD([("s", "q"), ("q", "t")], {"q": middle})
    counted = top.spares_demand(1000.0)
    assert list(counted) == [("q", ("r", "a")), ("q", ("r", "b")), ("q", "x")]
    assert counted[("q", ("r", "a"))].mean == pytest.approx(
        inner.spares_demand(1000.0)["a"].mean, abs=1e-12
    )


def test_one_sub_system_for_several_nodes_counts_each():
    inner = RepairableRBD(PAIR, {"a": unit(1e-3), "b": unit(2e-3)})
    twice = RepairableRBD(
        [("s", "q1"), ("q1", "q2"), ("q2", "t")], {"q1": inner, "q2": inner}
    )
    counted = twice.spares_demand(1000.0)
    assert set(counted) == {("q1", "a"), ("q1", "b"), ("q2", "a"), ("q2", "b")}


def test_the_simulated_counts_of_a_sub_system_are_its_own_runs():
    nested, flat = pumps_and_pair()
    # Each sub-system's component from the sub-system's own runs on the
    # same seed, keyed (node, component), against the exact count.
    n = 4000
    simulated = nested.spares_demand(
        5000.0, method="simulate", mc_samples=n, seed=2
    )
    assert list(simulated) == ["p", ("q", "a"), ("q", "b")]
    inner = nested.components["q"].spares_demand(
        5000.0, method="simulate", mc_samples=n, seed=2
    )
    assert np.array_equal(
        simulated[("q", "a")].probabilities, inner["a"].probabilities
    )
    exact = nested.spares_demand(5000.0)
    for key in simulated:
        se = exact[key].std / math.sqrt(n)
        assert abs(simulated[key].mean - exact[key].mean) < 4 * se
    # Its top-level components are counted as before.
    alone = nested.spares_demand(
        1000.0, nodes=["p"], method="simulate", mc_samples=10, seed=1
    )
    assert list(alone) == ["p"]
    # And a simulated count of the flat diagram agrees with the exact one.
    simulated = flat.spares_demand(
        5000.0, method="simulate", mc_samples=4000, seed=2
    )[("q", "a")]
    exact = nested.spares_demand(5000.0)[("q", "a")]
    se = math.sqrt(exact.std**2 / 4000)
    assert abs(simulated.mean - exact.mean) < 4 * se


def test_what_couples_a_sub_system_refuses_its_counts():
    inner = RepairableRBD(PAIR, {"a": unit(1e-3), "b": unit(2e-3)})
    # The diagram's crews serve its own components, not a sub-system's:
    # one crew for one pump leaves it as counted; for two, a job waits.
    one = RepairableRBD(
        [("s", "p"), ("p", "q"), ("q", "t")],
        {"p": unit(5e-4), "q": inner},
        repair_crews=1,
    )
    assert one.spares_demand(1000.0)[("q", "a")].mean == pytest.approx(
        inner.spares_demand(1000.0)["a"].mean, abs=1e-12
    )
    crews = RepairableRBD(
        [("s", "p"), ("p", "p2"), ("p2", "q"), ("q", "t")],
        {"p": unit(5e-4), "p2": unit(5e-4), "q": inner},
        repair_crews=1,
    )
    with pytest.raises(NotImplementedError, match="crew"):
        crews.spares_demand(1000.0)
    own_crews = RepairableRBD(
        PAIR, {"a": unit(1e-3), "b": unit(2e-3)}, repair_crews=1
    )
    nested = RepairableRBD(
        [("s", "p"), ("p", "q"), ("q", "t")], {"p": unit(5e-4), "q": own_crews}
    )
    with pytest.raises(NotImplementedError, match="crew"):
        nested.spares_stock(100.0, fill_rate=0.9)
    routes = nested.analysis_routes()
    assert routes["spares_stock"].route == "refused"
    nested, _ = pumps_and_pair()
    with pytest.raises(NotImplementedError, match="across levels"):
        nested.spares_stock(
            100.0, fill_rate=0.9, parts={"seal": ["p", ("q", "a")]}
        )


def instant(rate):
    return {"reliability": E([rate]), "repairability": "instant"}


def test_a_part_pools_spares_across_levels():
    # A top-level unit and a sub-system's, each replaced in no time at a
    # constant rate: their demands are independent Poisson processes, so
    # the pooled count over T is Poisson((l1 + l2) T), as for two units
    # at one level.
    inner = RepairableRBD(PAIR, {"a": instant(1e-3), "b": instant(2e-3)})
    nested = RepairableRBD(
        [("s", "p"), ("p", "q"), ("q", "t")], {"p": instant(5e-4), "q": inner}
    )
    T = 4000.0
    from scipy import stats

    pooled = nested.spares_demand(T, parts={"seal": ["p", ("q", "a")]})
    assert pooled["seal"].members == ("p", ("q", "a"))
    expected = stats.poisson.pmf(
        np.arange(len(pooled["seal"].probabilities)), 1.5e-3 * T
    )
    # The counts are worked out on a grid, to about 1e-6.
    assert np.allclose(pooled["seal"].probabilities, expected, atol=1e-6)
    simulated = nested.spares_demand(
        T,
        parts={"seal": ["p", ("q", "a")]},
        method="simulate",
        mc_samples=4000,
        seed=5,
    )["seal"]
    se = math.sqrt(1.5e-3 * T / 4000)
    assert abs(simulated.mean - 1.5e-3 * T) < 4 * se


def test_the_routes_say_a_sub_system_is_counted():
    nested, _ = pumps_and_pair()
    routes = nested.analysis_routes()
    for name in ("spares_demand", "spares_stock"):
        assert routes[name].route == "numerical"
        assert "(node, component)" in routes[name].reason


# The first failure.


@pytest.mark.parametrize(
    "inner_components, top",
    [
        ({"a": unit(1e-3), "b": unit(2e-3)}, {"p": unit(5e-4)}),
        # A series of wearing units: a series structure, any life.
        (
            {
                "a": {
                    "reliability": W([3000.0, 2.0]),
                    "repairability": E([0.1]),
                }
            },
            {
                "p": {
                    "reliability": W([5000.0, 1.5]),
                    "repairability": E([0.1]),
                }
            },
        ),
    ],
)
def test_the_first_failure_of_a_sub_system_is_the_flat_diagram(
    inner_components, top
):
    inner_edges = (
        PAIR if len(inner_components) == 2 else [("s", "a"), ("a", "t")]
    )
    inner = RepairableRBD(inner_edges, inner_components)
    outer = [("s", "p"), ("p", "q"), ("q", "t")]
    nested = RepairableRBD(outer, {**top, "q": inner})
    flat = drawn_in("q", inner_edges, inner_components, outer, top)
    x = np.array([0.0, 100.0, 1000.0, 5000.0])
    assert np.allclose(
        nested.first_failure_sf(x), flat.first_failure_sf(x), atol=1e-9
    )
    assert nested.mean_time_to_first_failure() == pytest.approx(
        flat.mean_time_to_first_failure(), abs=1e-9
    )
    assert nested.analysis_routes()["first_failure_sf"].route == "numerical"


def test_the_first_failure_simulated_agrees():
    nested, flat = pumps_and_pair()
    exact = flat.mean_time_to_first_failure()
    simulated = nested.mean_time_to_first_failure(
        method="simulate", t_simulation=1e6, mc_samples=4000, seed=1
    )
    assert abs(simulated.estimate - exact) < 4 * simulated.standard_error
    sf = nested.first_failure_sf(
        1000.0, method="simulate", mc_samples=4000, seed=1
    )
    p = flat.first_failure_sf(1000.0)
    assert abs(sf - p) < 4 * math.sqrt(p * (1 - p) / 4000)


def test_a_coupled_sub_system_keeps_its_refusal():
    own_crews = RepairableRBD(
        PAIR, {"a": unit(1e-3), "b": unit(2e-3)}, repair_crews=1
    )
    nested = RepairableRBD(
        [("s", "p"), ("p", "q"), ("q", "t")], {"p": unit(5e-4), "q": own_crews}
    )
    with pytest.raises(NotImplementedError, match="repair crews"):
        nested.mean_time_to_first_failure()
    assert nested.analysis_routes()["first_failure_sf"].route == "refused"
    inner = RepairableRBD(PAIR, {"a": unit(1e-3), "b": unit(2e-3)})
    around = RepairableRBD(
        [("s", "p"), ("p", "p2"), ("p2", "q"), ("q", "t")],
        {"p": unit(5e-4), "p2": unit(5e-4), "q": inner},
        repair_crews=1,
    )
    with pytest.raises(NotImplementedError, match="repair crews"):
        around.first_failure_sf(10.0)
    # Crews no job waits for leave it as drawn in with a crew each.
    one = RepairableRBD(
        [("s", "p"), ("p", "q"), ("q", "t")],
        {"p": unit(5e-4), "q": inner},
        repair_crews=1,
    )
    nested, flat = pumps_and_pair()
    assert one.mean_time_to_first_failure() == pytest.approx(
        flat.mean_time_to_first_failure(), abs=1e-9
    )


def test_the_simulated_shelf_takes_a_sub_system_by_its_components():
    nested, _ = pumps_and_pair()
    found = nested.spares_stock(
        100.0,
        method="simulate",
        horizon=1000.0,
        no_stockout=0.9,
        mc_samples=500,
        seed=1,
    )
    inner = [k for k in found if isinstance(k, tuple)]
    assert inner and all(k[0] == "q" for k in inner)
    with pytest.raises(NotImplementedError, match="nested RBD"):
        nested.least_spend_for_downtime(
            1000.0, 10.0, spares={"q": {"price": 1.0, "lead_time": 10.0}}
        )
    found = nested.spares_stock(
        100.0,
        method="simulate",
        horizon=1000.0,
        no_stockout=0.9,
        nodes=["p"],
        mc_samples=500,
        seed=1,
    )
    assert list(found) == ["p"]
