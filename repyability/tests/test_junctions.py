"""Junctions in a RepairableRBD (#175, #182): nodes given
``PerfectReliability``, which never fail, such as a k-out-of-n vote point.
They are folded out of the structure (``modular.fold``), so every analysis
and both simulation engines see the components alone."""

import dataclasses
import itertools
import json

import networkx as nx
import numpy as np
import pytest
import surpyval as surv

from repyability import (
    NonRepairable,
    NonRepairableRBD,
    PerfectReliability,
    RepairableRBD,
)
from repyability.rbd import modular
from repyability.rbd.rbd_graph import RBDGraph
from repyability.tests.test_analysis_routes import REPAIRABLE_CALLS
from repyability.tests.test_rbd_modular import random_diagram

E = surv.Exponential.from_params
W = surv.Weibull.from_params
TRAINS = ["a", "b", "c"]
CAPACITY = {"a": 50.0, "b": 50.0, "c": 50.0, "v": 120.0, "p": 200.0}


def unit(**more):
    return {
        "reliability": W([500, 1.5]),
        "repairability": E([0.5]),
        "repair_cost": 2.0,
        **more,
    }


def station(junction=PerfectReliability):
    """Three pump trains, two needed, voted at the junction 'h', then a
    valve and a controller in series."""
    edges = (
        [("s", x) for x in TRAINS]
        + [(x, "h") for x in TRAINS]
        + [("h", "v"), ("v", "p"), ("p", "t")]
    )
    components = {x: unit() for x in TRAINS + ["v", "p"]}
    return RepairableRBD(
        edges,
        {**components, "h": junction},
        k={"h": 2},
        capacity=CAPACITY,
        downtime_cost_rate=3.0,
    )


def reordered():
    """The station with the series parts first and the vote at the
    output, as it had to be drawn before junctions."""
    edges = (
        [("s", "p"), ("p", "v")]
        + [("v", x) for x in TRAINS]
        + [(x, "t") for x in TRAINS]
    )
    components = {x: unit() for x in TRAINS + ["v", "p"]}
    return RepairableRBD(
        edges,
        components,
        k={"t": 2},
        capacity=CAPACITY,
        downtime_cost_rate=3.0,
    )


def flattened(value, key="") -> dict:
    """A result's numbers and labels, by where they are in it."""
    out: dict = {}
    if isinstance(value, bytes):
        # A shard: JSON.
        out.update(flattened(json.loads(value), key))
    elif value is None or isinstance(value, (str, bool)):
        out[key] = value
    elif isinstance(value, (int, float, np.number)):
        out[key] = float(value)
    elif isinstance(value, np.ndarray):
        out[key] = value
    elif isinstance(value, dict):
        for k in sorted(value, key=str):
            out.update(flattened(value[k], f"{key}.{k}"))
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            out.update(flattened(item, f"{key}[{i}]"))
    elif dataclasses.is_dataclass(value):
        for f in dataclasses.fields(value):
            out.update(flattened(getattr(value, f.name), f"{key}.{f.name}"))
    elif hasattr(value, "__dict__"):
        for k, item in sorted(vars(value).items()):
            if not k.startswith("_"):
                out.update(flattened(item, f"{key}.{k}"))
    else:
        out[key] = repr(value)
    return out


# The saved system differs, as the diagrams do.
DIFFERENT_DIAGRAMS = ("fingerprint", "system")


@pytest.mark.parametrize("name", sorted(REPAIRABLE_CALLS))
def test_a_vote_at_a_junction_is_the_vote_at_the_output(name):
    # Every analysis of the station, drawn with its vote at a junction, is
    # that of the station drawn with the vote at the output: the same
    # numbers (the simulations' too, from the same streams), or the same
    # refusal.
    call = REPAIRABLE_CALLS[name]
    results = []
    for rbd in (station(), reordered()):
        try:
            result = flattened(call(rbd))
        except (ValueError, NotImplementedError) as error:
            result = {"refused": str(error)}
        results.append(
            {
                key: value
                for key, value in result.items()
                if not any(word in key for word in DIFFERENT_DIAGRAMS)
            }
        )
    got, expected = results
    assert set(got) == set(expected)
    for key, value in expected.items():
        if isinstance(value, (float, np.ndarray)) and not (
            isinstance(value, np.ndarray) and value.dtype.kind not in "fiub"
        ):
            np.testing.assert_allclose(
                np.asarray(got[key], float),
                np.asarray(value, float),
                rtol=1e-9,
                atol=1e-12,
                err_msg=key,
            )
        elif isinstance(value, np.ndarray):
            assert np.array_equal(got[key], value), key
        else:
            assert got[key] == value, key


def test_the_junction_is_no_component():
    rbd = station()
    assert rbd._junctions() == {"h"}
    assert "h" not in rbd.components and "h" not in rbd.nodes
    assert "junction(s) 'h'" in repr(rbd)
    assert all("h" not in path for path in rbd.get_min_path_sets())
    assert "h" not in rbd.birnbaum_importance()
    with pytest.raises(ValueError, match="'h' is a junction, which always"):
        rbd.mean_availability(working_nodes=["h"])
    with pytest.raises(ValueError, match="'h' is a junction, which always"):
        rbd.availability(10.0, mc_samples=4, seed=1, broken_nodes=["h"])


def test_two_votes_in_series_are_nested_votes():
    # Two 2-of-3 stages in series, which no reordering puts both at the
    # output: with junctions, as with each stage a nested RBD.
    xs, ys = ["x0", "x1", "x2"], ["y0", "y1", "y2"]
    flat = RepairableRBD(
        [("s", x) for x in xs]
        + [(x, "h1") for x in xs]
        + [("h1", y) for y in ys]
        + [(y, "h2") for y in ys]
        + [("h2", "t")],
        {x: unit() for x in xs + ys}
        | {"h1": PerfectReliability, "h2": PerfectReliability},
        k={"h1": 2, "h2": 2},
    )

    def stage(names):
        return RepairableRBD(
            [("s", n) for n in names] + [(n, "t") for n in names],
            {n: unit() for n in names},
            k={"t": 2},
        )

    nested = RepairableRBD(
        [("s", "X"), ("X", "Y"), ("Y", "t")], {"X": stage(xs), "Y": stage(ys)}
    )
    t = np.array([50.0, 400.0, 2000.0])
    assert flat.mean_availability() == pytest.approx(
        nested.mean_availability(), rel=1e-12
    )
    np.testing.assert_allclose(
        flat.point_availability(t), nested.point_availability(t), rtol=1e-9
    )
    np.testing.assert_allclose(
        flat.expected_failures(t), nested.expected_failures(t), rtol=1e-9
    )
    assert flat.system_failure_frequency() == pytest.approx(
        nested.system_failure_frequency(), rel=1e-12
    )


@pytest.mark.parametrize(
    "perfect",
    [
        PerfectReliability,
        {"reliability": PerfectReliability, "repairability": E([0.1])},
    ],
    ids=["itself", "as a life"],
)
def test_a_part_that_never_fails_changes_nothing(perfect):
    # #175: PerfectReliability as a repairable component's life.
    b = {"reliability": W([500, 1.5]), "repairability": E([0.5])}
    with_it = RepairableRBD(
        [("s", "a"), ("a", "b"), ("b", "t")], {"a": perfect, "b": b}
    )
    without = RepairableRBD([("s", "b"), ("b", "t")], {"b": b})
    assert with_it.mean_availability() == without.mean_availability()
    t = np.array([0.0, 10.0, 300.0])
    np.testing.assert_array_equal(
        with_it.point_availability(t), without.point_availability(t)
    )
    run = with_it.availability(300.0, mc_samples=50, seed=1)
    alone = without.availability(300.0, mc_samples=50, seed=1)
    np.testing.assert_array_equal(run.uptimes, alone.uptimes)


def test_a_junction_passes_what_reaches_it_up_to_its_capacity():
    def pumps(capacity):
        return RepairableRBD(
            [("s", "a"), ("s", "b"), ("a", "h"), ("b", "h"), ("h", "t")],
            {"a": unit(), "b": unit(), "h": PerfectReliability},
            capacity=capacity,
        )

    free = pumps({"a": 50.0, "b": 50.0}).capacity_distribution()
    capped = pumps({"a": 50.0, "b": 50.0, "h": 70.0}).capacity_distribution()
    assert free.levels.max() == 100.0
    assert capped.levels.max() == 70.0
    np.testing.assert_allclose(
        capped.probabilities.sum(), free.probabilities.sum()
    )


@pytest.mark.parametrize(
    "junction",
    [
        PerfectReliability,
        {"reliability": PerfectReliability, "repairability": E([0.1])},
    ],
)
def test_a_junction_is_saved(junction):
    rbd = station(junction)
    loaded = RepairableRBD.from_json(rbd.to_json())
    assert loaded._junctions() == {"h"}
    assert loaded.to_dict() == rbd.to_dict()
    assert loaded.mean_availability() == rbd.mean_availability()


def test_a_part_that_never_fails_takes_no_maintenance():
    spec = {
        "reliability": PerfectReliability,
        "repairability": E([0.1]),
        "preventive": {"interval": 10.0},
        "repair_cost": 5.0,
    }
    with pytest.raises(ValueError, match="takes no preventive, repair_cost"):
        RepairableRBD([("s", "a"), ("a", "t")], {"a": spec})


def test_the_messages_say_to_give_a_junction_perfect_reliability():
    edges = (
        [("s", x) for x in TRAINS] + [(x, "h") for x in TRAINS] + [("h", "t")]
    )
    life = W([100.0, 2.0])
    with pytest.raises(ValueError, match="'h'.*takes PerfectReliability"):
        NonRepairableRBD(edges, {x: life for x in TRAINS}, k={"h": 2})
    with pytest.raises(ValueError, match="'h'.*takes PerfectReliability"):
        RepairableRBD(edges, {x: unit() for x in TRAINS}, k={"h": 2})
    with pytest.raises(ValueError, match="give PerfectReliability itself"):
        NonRepairable(PerfectReliability)


def _reached(edges, k, perfect, state) -> bool:
    """Whether the output is reached: a node is when it works (a perfect
    one always does) and at least k of its inputs are reached."""
    graph = nx.DiGraph(edges)
    reached: dict = {}
    for v in nx.topological_sort(graph):
        if v == "s":
            reached[v] = True
            continue
        enough = sum(reached[u] for u in graph.predecessors(v)) >= k.get(v, 1)
        works = v == "t" or v in perfect or state[v]
        reached[v] = enough and works
    return reached["t"]


@pytest.mark.parametrize("core", ["paths", "bdd"])
@pytest.mark.parametrize("reduce", [True, False])
def test_folding_keeps_the_structure_function(core, reduce):
    rng = np.random.default_rng(7)
    for _ in range(150):
        n = int(rng.integers(2, 8))
        edges, k = random_diagram(rng, n)
        graph = RBDGraph(edges)
        for node in graph.nodes:
            graph.nodes[node]["k"] = k.get(node, 1)
        nodes = [v for v in range(n) if v in graph.nodes]
        perfect = {v for v in nodes if rng.random() < 0.35}
        try:
            whole = modular.decompose(
                graph, "s", "t", reduce=reduce, core=core
            )
        except ValueError:  # nothing reaches the output
            continue
        folded = modular.fold(whole, perfect)
        assert not folded.nodes & perfect
        rest = [v for v in nodes if v not in perfect]
        for bits in itertools.product([False, True], repeat=len(rest)):
            state = dict(zip(rest, bits))
            expected = _reached(edges, k, perfect, state)
            assert folded.works(state, "p") == expected
            assert folded.works(state, "c") == expected
        p = {v: float(rng.uniform(0.05, 0.95)) for v in rest}
        assert folded.probabilities(p)[0] == pytest.approx(
            whole.probabilities(p | {v: 1.0 for v in perfect})[0], abs=1e-12
        )
