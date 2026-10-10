"""The repairable simulation's streams and engines.

Every draw of a simulation comes from a stream of its own (see
``repyability.rbd._streams``), and these tests hold the simulation to that:

- each draw the event loop takes is the one the streams' definition gives
  (``keyed_draws``), for every kind of draw: failures, repairs, maintenance
  and test times, and costs, nested RBDs' included, with antithetic pairs;
- for plain components, a reference simulation written from scratch (a
  ``heapq`` of events, the definition's draws, and the timeline functions
  for the up and down times) gives the same results;
- each simulation is the same however the run is cut up: in one process or
  several, in a run to a tolerance or of a fixed size;
- the compiled engine (numba, when installed) gives the same results as the
  Python one, to the last bit, and ``engine="auto"`` chooses between them.
"""

import dataclasses
import heapq
import warnings
from collections.abc import Mapping

import numpy as np
import pytest
import surpyval as surv

from repyability import NodeState, PerfectReliability, RepairableRBD
from repyability.non_repairable import NonRepairable
from repyability.rbd import _compiled, _streams, repairable_rbd
from repyability.rbd.repairable_rbd import Event
from repyability.tests import timeline_reference
from repyability.tests.catalogue import systems_of_every_kind
from repyability.tests.keyed_draws import KeyedDraws, reference_draw
from repyability.tests.test_performance_equivalence import (
    binomial_first,
    overlaps_one_at_a_time,
    pumps_with_capacities,
    repairable_rbds,
)

E = surv.Exponential.from_params
W = surv.Weibull.from_params
L = surv.LogNormal.from_params
X = surv.ExactEventTime.from_params
G = surv.Gamma.from_params
BRIDGE = [
    ("s", "a"),
    ("s", "b"),
    ("a", "c"),
    ("b", "c"),
    ("a", "d"),
    ("c", "d"),
    ("b", "e"),
    ("d", "t"),
    ("e", "t"),
]

needs_numba = pytest.mark.skipif(
    not _compiled.available(), reason="numba is not installed"
)


def identical(a, b, path="result"):
    """Recursively compare results: everything exactly, NaN equal to NaN,
    and mappings in the same order."""
    if dataclasses.is_dataclass(a):
        assert type(a) is type(b), path
        for f in dataclasses.fields(a):
            identical(
                getattr(a, f.name), getattr(b, f.name), f"{path}.{f.name}"
            )
    elif isinstance(a, Mapping):
        assert list(a) == list(b), path
        for key in a:
            identical(a[key], b[key], f"{path}[{key!r}]")
    elif isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        assert np.array_equal(
            np.asarray(a), np.asarray(b), equal_nan=True
        ), path
    elif isinstance(a, float) and np.isnan(a):
        assert isinstance(b, float) and np.isnan(b), path
    else:
        assert a == b, path


def plain_rbds():
    """Systems of plain components, which the compiled engine runs."""
    return {
        "bridge": RepairableRBD(
            BRIDGE,
            {
                n: {
                    "reliability": W([70 + 10 * i, 1.5]),
                    "repairability": L([0.5, 0.6]),
                }
                for i, n in enumerate("abcde")
            },
        ),
        "koon": RepairableRBD(
            [("s", u) for u in "abc"]
            + [(u, "v") for u in "abc"]
            + [("v", "t")],
            {
                n: {
                    "reliability": W([60 + 10 * i, 1.8]),
                    "repairability": E([0.3]),
                }
                for i, n in enumerate("abcv")
            },
            k={"v": 2},
        ),
        "costed": RepairableRBD(
            [("s", "x"), ("x", "y"), ("s", "z"), ("y", "t"), ("z", "t")],
            {
                "x": {
                    "reliability": E([0.02]),
                    "repairability": "instant",
                    "replace_cost": surv.Gamma.from_params([4.0, 0.04]),
                    "downtime_cost": 2.0,
                },
                "y": {
                    "reliability": W([80, 2.5], gamma=5),
                    "repairability": E([0.5]),
                    "repair_cost": 40.0,
                    "replace_cost": L([2.0, 0.5]),
                },
                "z": {
                    "reliability": L([4.0, 0.7]),
                    "repairability": W([3, 1.2]),
                },
            },
            downtime_cost_rate=10.0,
        ),
        # Two 2-of-3 votes at junctions, which are folded out of the
        # structure (#182), and a junction where a bridge crosses.
        "junctions": RepairableRBD(
            [("s", f"x{i}") for i in range(3)]
            + [(f"x{i}", "h1") for i in range(3)]
            + [("h1", f"y{i}") for i in range(3)]
            + [(f"y{i}", "h2") for i in range(3)]
            + [("h2", "a"), ("h2", "b"), ("a", "j"), ("b", "j")]
            + [("a", "c"), ("j", "c"), ("j", "d"), ("b", "d")]
            + [("c", "t"), ("d", "t")],
            {
                n: {
                    "reliability": W([90 + 10 * i, 1.6]),
                    "repairability": E([0.4]),
                }
                for i, n in enumerate(
                    ["x0", "x1", "x2", "y0", "y1", "y2", "a", "b", "c", "d"]
                )
            }
            | {
                "h1": PerfectReliability,
                "h2": PerfectReliability,
                "j": PerfectReliability,
            },
            k={"h1": 2, "h2": 2},
        ),
        # Fixed lives and repairs: events at the same time, released in the
        # heap's order.
        "ties": RepairableRBD(
            [("s", "a"), ("s", "b"), ("a", "c"), ("b", "c"), ("c", "t")],
            {
                "a": {
                    "reliability": X(10.0),
                    "repairability": X(2.0),
                    "repair_cost": 3.0,
                },
                "b": {"reliability": X(10.0), "repairability": X(2.0)},
                "c": {"reliability": X(12.0), "repairability": "instant"},
            },
            downtime_cost_rate=1.0,
        ),
        # Units that never fail, units dead on arrival, and normal repair
        # times.
        "defective": RepairableRBD(
            [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
            {
                "a": {
                    "reliability": W([30, 2], lfp_p=0.7),
                    "repairability": surv.Normal.from_params([2, 0.3]),
                },
                "b": {
                    "reliability": W([40, 1.2], f0=0.05),
                    "repairability": E([0.5]),
                },
            },
        ),
        "objects": RepairableRBD(
            [("s", "a"), ("a", "b"), ("b", "t")],
            {
                "a": NonRepairable(W([50, 2]), E([1.0])),
                "b": NonRepairable(E([0.01]), L([0.5, 0.3])),
            },
        ),
    }


# -- the draws ----------------------------------------------------------------


@pytest.mark.parametrize("antithetic", [False, True])
@pytest.mark.parametrize("name", sorted(repairable_rbds()))
def test_every_draw_comes_from_its_stream(name, antithetic, monkeypatch):
    rbd = repairable_rbds()[name]
    drawn = []
    draw = _streams.Stream.draw

    def recorded(stream):
        value = draw(stream)
        drawn.append(
            (stream._spec, stream._run.replication, stream._k - 1, value)
        )
        return value

    monkeypatch.setattr(_streams.Stream, "draw", recorded)
    window = 150.0
    rbd.availability(
        window, mc_samples=12, seed=31, antithetic=antithetic, engine="python"
    )
    entropy = _streams.entropy_of(31)
    kinds = set()
    for spec, replication, k, value in drawn:
        kinds.add(spec.kind)
        expected = reference_draw(entropy, spec, antithetic, replication, k)
        np.testing.assert_allclose(value, expected, rtol=1e-12, atol=0)
    assert _streams.FAILURE in kinds and _streams.REPAIR in kinds
    if rbd._preventive or rbd._inspection:
        assert _streams.DURATION in kinds


def test_each_simulation_draws_from_the_start_of_its_columns():
    # The k-th draw of a stream in a simulation is its k-th, however many
    # draws the other simulations take, and a block extended by more rows
    # keeps its rows.
    rbd = plain_rbds()["bridge"]
    plan, complete = rbd._stream_plan(400.0, 5, False)
    assert complete
    spec = plan.specs[(("a",), _streams.FAILURE)]
    block = plan.block(spec, 1)
    first = block.values.copy()
    block.extend()
    block.extend()
    assert block.values.shape[1] == first.shape[1] + 3 * max(spec.rows, 8)
    np.testing.assert_array_equal(block.values[:, : first.shape[1]], first)
    for j in (0, spec.width - 1):
        for k in (0, first.shape[1] + 5):
            expected = reference_draw(5, spec, False, spec.width + j, k)
            np.testing.assert_allclose(
                block.values[j, k], expected, rtol=1e-12
            )


# -- a reference simulation ---------------------------------------------------


def reference(rbd, t_end, n, seed, working=(), broken=(), antithetic=False):
    """Each simulation of ``rbd``'s plain components stepped through with a
    ``heapq`` of events, drawing from the streams' definition, and measured
    with the timeline functions: per simulation, the system's up time, the
    components' up and overlap times, the counts, the system's changes and
    the cost."""
    draws = KeyedDraws(rbd, t_end, seed, antithetic)
    nodes = list(rbd.components)
    index = {node: c for c, node in enumerate(nodes)}
    out = []
    for r in range(n):
        status = {node: node not in broken for node in nodes}
        heap: list = []
        for node in nodes:
            if node not in working and node not in broken:
                t = draws.next((node,), _streams.FAILURE, r)
                if t < t_end:
                    heapq.heappush(heap, (t, Event(t, node, False)))
        up = rbd.is_system_working(status, "p")
        system = [(0.0, 1 if up else 0)]
        events, counts = [], np.zeros((4, len(nodes)), int)
        charges = []
        while heap:
            t, event = heapq.heappop(heap)
            node, c = event.component, index[event.component]
            status[node] = event.status
            events.append((t, c, 1 if event.status else -1))
            counts[2 if event.status else 0, c] += 1
            if not event.status:
                costs = rbd.costs.get(node, {})
                for key in rbd.PER_FAILURE_COST_KEYS:
                    if key in costs:
                        if isinstance(costs[key], float):
                            charges.append(costs[key])
                        else:
                            kind = _streams.COST_KINDS[key]
                            charges.append(draws.next((node,), kind, r))
            works = rbd.is_system_working(status, "p")
            if works != up:
                up = works
                system.append((t, 1 if up else -1))
                counts[3 if up else 1, c] += 1
            kind = _streams.FAILURE if event.status else _streams.REPAIR
            t_next = t + draws.next((node,), kind, r)
            if t_next < t_end:
                heapq.heappush(
                    heap, (t_next, Event(t_next, node, not event.status))
                )
        system.append((t_end, 0))
        starts = [1 if node not in broken else 0 for node in nodes]
        overlaps = overlaps_one_at_a_time(starts, events, system, t_end)
        uptime = timeline_reference.time_at_status(system, 1)
        for c, node in enumerate(nodes):
            rate = rbd.costs.get(node, {}).get("downtime_cost")
            if rate is not None:
                charges.append(rate * (t_end - overlaps[c][0]))
        if rbd.has_costs:
            charges.append(rbd.downtime_cost_rate * (t_end - uptime))
        out.append((uptime, overlaps, counts, system[1:-1], sum(charges)))
    return out


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"antithetic": True},
        {"working": ("a",)},
        {"broken": ("a",)},
    ],
    ids=["plain", "antithetic", "a working", "a broken"],
)
@pytest.mark.parametrize("name", sorted(plain_rbds()))
def test_plain_simulation_matches_the_reference(name, options):
    rbd = plain_rbds()[name]
    options = {
        k: (
            tuple(n for n in v if n in rbd.components)
            if k != "antithetic"
            else v
        )
        for k, v in options.items()
    }
    t_end, n = 120.0, 30
    result = rbd.availability(
        t_end,
        mc_samples=n,
        seed=41,
        engine="python",
        working_nodes=options.get("working"),
        broken_nodes=options.get("broken"),
        antithetic=options.get("antithetic", False),
    )
    expected = reference(rbd, t_end, n, 41, **options)
    np.testing.assert_allclose(
        result.uptimes, [e[0] for e in expected], rtol=1e-12, atol=1e-12
    )
    nodes = list(rbd.components)
    totals = np.sum([e[1] for e in expected], axis=0)  # nodes x 5
    for c, node in enumerate(nodes):
        np.testing.assert_allclose(
            [
                result.node_uptime[node],
                result.criticalities.operational_criticality_index.up[node]
                * result.system_uptime,
            ],
            [totals[c][0], totals[c][1]],
            rtol=1e-12,
            atol=1e-9,
        )
    counts = np.sum([e[2] for e in expected], axis=0)
    assert result.system_failures == counts[1].sum()
    assert result.system_restorations == counts[3].sum()
    fci = result.criticalities.failure_criticality_index.per_component_failure
    assert list(fci) == [n for c, n in enumerate(nodes) if counts[0][c]]
    # The availability over time: the same changes at the same times.
    changes: dict = {0.0: 0, t_end: 0}
    for e in expected:
        for t, change in e[3]:
            changes[t] = changes.get(t, 0) + change
    times = sorted(changes)
    np.testing.assert_array_equal(result.timeline, times)
    initial = n * rbd.is_system_working(
        {node: node not in options.get("broken", ()) for node in nodes}, "p"
    )
    working = initial + np.cumsum([changes[t] for t in times])
    np.testing.assert_allclose(result.availability, working / n, rtol=1e-15)
    if rbd.has_costs:
        np.testing.assert_allclose(
            result.cost.samples, [e[4] for e in expected], rtol=1e-12
        )


# -- how the run is cut up ----------------------------------------------------


@pytest.mark.parametrize("name", sorted(systems_of_every_kind()))
def test_a_simulation_is_the_same_however_the_run_is_cut_up(name):
    rbd = systems_of_every_kind()[name]
    window = 20000.0 if name == "instrument air" else 150.0
    whole = rbd.availability(window, mc_samples=40, seed=51, engine="python")
    identical(
        whole,
        rbd.availability(
            window, mc_samples=40, seed=51, engine="python", n_jobs=2
        ),
    )
    first = rbd.availability(window, mc_samples=15, seed=51, engine="python")
    np.testing.assert_array_equal(first.uptimes, whole.uptimes[:15])
    # A run to a tolerance that stops after n simulations is a run of n.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        stopped = rbd.availability(
            window,
            mc_samples=20,
            seed=51,
            engine="python",
            tolerance=1e-9,
            max_samples=40,
            control_variate=False,
        )
    n = stopped.n_simulations
    identical(
        stopped,
        rbd.availability(
            window,
            mc_samples=n,
            seed=51,
            engine="python",
            control_variate=False,
        ),
    )


@pytest.mark.parametrize("forced", ["working_nodes", "broken_nodes"])
@pytest.mark.parametrize(
    "name",
    [
        "nested_koon",
        "nested_one_level",
        "nested_two_levels",
        "nested_maintained",
    ],
)
def test_a_nested_rbd_can_be_held_working_or_broken(name, forced):
    rbd = repairable_rbds()[name]
    for node, component in rbd.components.items():
        if isinstance(component, RepairableRBD):
            options = dict(
                t_simulation=200.0, mc_samples=30, seed=26, **{forced: [node]}
            )
            result = rbd.availability(engine="python", **options)
            identical(
                result, rbd.availability(engine="python", n_jobs=2, **options)
            )
            held = 200.0 * 30 if forced == "working_nodes" else 0.0
            assert result.node_uptime[node] == held


def test_an_unstreamable_maintenance_time_still_takes_time():
    rbd = systems_of_every_kind()["unstreamable maintenance"]
    assert not rbd._stream_specs(200.0)[1]
    result = rbd.availability(200.0, mc_samples=30, seed=31)
    assert result.system_planned_outages > 0
    assert result.node_uptime["a"] < 200.0 * 30


def test_an_unstreamable_component_leaves_the_others_alone():
    # The streamed component draws the same numbers whatever the other
    # draws from, so its up time is the same.
    systems = systems_of_every_kind()
    mixed = systems["unstreamable"]
    assert not mixed._stream_specs(100.0)[1]
    plain = RepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        {
            "a": {"reliability": W([70, 1.5]), "repairability": E([0.8])},
            "b": {"reliability": W([40, 2]), "repairability": E([0.5])},
        },
    )
    a_mixed = mixed.availability(300.0, mc_samples=50, seed=3).node_uptime["a"]
    a_plain = plain.availability(300.0, mc_samples=50, seed=3).node_uptime["a"]
    assert a_mixed == a_plain
    # Antithetic pairs and common random numbers need every draw streamed.
    with pytest.raises(NotImplementedError):
        mixed.availability(100.0, mc_samples=10, seed=1, antithetic=True)
    with pytest.raises(NotImplementedError):
        mixed.compare(
            plain, 100.0, mc_samples=10, seed=1, control_variate=False
        )


@pytest.mark.parametrize("name", ["costed_pairs", "maintained", "nested_koon"])
def test_the_global_rng_is_left_as_it_was(name):
    rbd = repairable_rbds()[name]
    np.random.seed(7)
    before = np.random.get_state()[1].copy()
    rbd.availability(100.0, mc_samples=10, seed=3)
    assert np.array_equal(np.random.get_state()[1], before)
    # Without a seed, the run takes one number, and is the run that
    # seeding with the same number gives.
    np.random.seed(7)
    unseeded = rbd.availability(100.0, mc_samples=10)
    after = np.random.get_state()
    np.random.seed(7)
    np.random.randint(0, 2**62, dtype=np.int64)
    expected = np.random.get_state()
    assert np.array_equal(after[1], expected[1]) and after[2] == expected[2]
    identical(unseeded, rbd.availability(100.0, mc_samples=10, seed=7))


# -- the engines --------------------------------------------------------------


def test_an_unknown_engine_is_refused():
    with pytest.raises(ValueError, match="engine"):
        plain_rbds()["bridge"].availability(
            10.0, mc_samples=2, seed=1, engine="fast"
        )


@pytest.mark.parametrize(
    "name, reason",
    [
        ("maintained", "preventive maintenance"),
        ("inspected", "inspections"),
        ("nested_one_level", "nested RBDs"),
    ],
)
def test_the_compiled_engine_says_what_it_cannot_run(name, reason):
    rbd = repairable_rbds()[name]
    plan, _ = rbd._stream_plan(100.0, 1, False)
    assert reason in _compiled.unsupported(rbd, plan, None)
    # "auto" runs it in Python.
    identical(
        rbd.availability(100.0, mc_samples=5, seed=2),
        rbd.availability(100.0, mc_samples=5, seed=2, engine="python"),
    )


def test_what_the_compiled_engine_can_run():
    for rbd in plain_rbds().values():
        plan, _ = rbd._stream_plan(100.0, 1, False)
        assert _compiled.unsupported(rbd, plan, None) is None
    mixed = systems_of_every_kind()["unstreamable"]
    plan, _ = mixed._stream_plan(100.0, 1, False)
    assert "cannot be streamed" in _compiled.unsupported(mixed, plan, None)
    capacity = RepairableRBD(
        [("s", "a"), ("a", "t")],
        {"a": {"reliability": E([0.1]), "repairability": E([1.0])}},
        capacity={"a": 10.0},
    )
    plan, _ = capacity._stream_plan(100.0, 1, False)
    assert _compiled.unsupported(capacity, plan, object()) == "capacities"
    # numba's own loop follows them (#155).
    assert _compiled.unsupported(capacity, plan, object(), numba=True) is None


def test_a_subclassed_component_runs_in_python(monkeypatch):
    # A subclass may make its events its own way, which only the Python
    # loop asks it for: "auto" runs it in Python however long the run.
    class LoggedUnit(NonRepairable):
        pass

    rbd = RepairableRBD(
        [("s", "a"), ("a", "sub"), ("sub", "t")],
        {
            "a": {"reliability": W([70, 1.5]), "repairability": E([0.8])},
            "sub": LoggedUnit(W([40, 2]), E([0.5])),
        },
    )
    plan, _ = rbd._stream_plan(100.0, 1, False)
    assert (
        _compiled.unsupported(rbd, plan, None)
        == "the LoggedUnit of node 'sub'"
    )

    def compiled(*args, **kwargs):
        raise AssertionError("compiled")

    monkeypatch.setattr(_compiled, "worthwhile", lambda *args: True)
    monkeypatch.setattr(_compiled, "Runner", compiled)
    rbd.availability(100.0, mc_samples=5, seed=2)


@needs_numba
def test_the_compiled_engine_refuses_what_it_cannot_run():
    grouped = RepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        {
            node: {
                **unit_spec(70, 2.0),
                "preventive": {"interval": 30.0},
                "group": "g",
            }
            for node in "ab"
        },
        maintenance_groups={"g": {}},
    )
    with pytest.raises(NotImplementedError, match="maintenance groups"):
        grouped.availability(100.0, mc_samples=5, seed=2, engine="numba")
    timed = inspected_unit(
        {"interval": 30.0, "duration": binomial_first([2, 1])}
    )
    with pytest.raises(NotImplementedError, match="test time"):
        timed.availability(100.0, mc_samples=5, seed=2, engine="numba")
    with pytest.raises(NotImplementedError, match="on condition"):
        on_condition().availability(
            100.0, mc_samples=5, seed=2, engine="numba"
        )
    imperfect = RepairableRBD(
        [("s", "a"), ("a", "t")],
        {
            "a": {
                **unit_spec(70, 2.0),
                "repair": {"model": "kijima1", "q": 0.5},
            }
        },
    )
    with pytest.raises(NotImplementedError, match="imperfect repair"):
        imperfect.availability(100.0, mc_samples=5, seed=2, engine="numba")


def test_without_numba_the_compiled_engine_cannot_be_asked_for(monkeypatch):
    monkeypatch.setattr(_compiled, "available", lambda: False)
    rbd = plain_rbds()["bridge"]
    with pytest.raises(ImportError, match="repyability\\[fast\\]"):
        rbd.availability(10.0, mc_samples=2, seed=1, engine="numba")
    # "auto" runs in Python.
    rbd.availability(10.0, mc_samples=2, seed=1)
    # What it does not simulate is refused as such: installing numba would
    # not help.
    imperfect = RepairableRBD(
        [("s", "a"), ("a", "t")],
        {
            "a": {
                **unit_spec(70, 2.0),
                "repair": {"model": "kijima1", "q": 0.5},
            }
        },
    )
    with pytest.raises(NotImplementedError, match="imperfect repair"):
        imperfect.availability(10.0, mc_samples=2, seed=1, engine="numba")


@needs_numba
@pytest.mark.parametrize(
    "options",
    [
        dict(t_simulation=300.0, mc_samples=60, seed=21),
        dict(t_simulation=200.0, mc_samples=30, seed=22, method="c"),
        dict(t_simulation=200.0, mc_samples=40, seed=26, antithetic=True),
        dict(t_simulation=500.0, mc_samples=700, seed=27, n_jobs=3),
        dict(
            t_simulation=100.0,
            mc_samples=50,
            seed=28,
            tolerance=1e-9,
            max_samples=150,
        ),
    ],
    ids=["plain", "cut sets", "antithetic", "threads", "tolerance"],
)
@pytest.mark.parametrize("name", sorted(plain_rbds()))
def test_the_engines_give_the_same_results(name, options):
    rbd = plain_rbds()[name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        python = rbd.availability(engine="python", **options)
        compiled = rbd.availability(engine="numba", **options)
    identical(python, compiled)


@needs_numba
@pytest.mark.parametrize("forced", ["working_nodes", "broken_nodes"])
def test_the_engines_agree_with_forced_nodes(forced):
    for rbd in plain_rbds().values():
        first = rbd.nodes[0]
        for engine in ("python", "numba"):
            options = dict(
                t_simulation=200.0, mc_samples=30, seed=24, **{forced: [first]}
            )
            if engine == "python":
                python = rbd.availability(engine=engine, **options)
            else:
                identical(python, rbd.availability(engine=engine, **options))


def pairs_in_series(pairs):
    """``pairs`` pairs of redundant units, one pair after another."""
    edges, previous, components = [], ["s"], {}
    for i in range(pairs):
        pair = [f"a{i}", f"b{i}"]
        edges += [(p, unit) for p in previous for unit in pair]
        previous = pair
        for unit in pair:
            components[unit] = {
                "reliability": W([50 + i, 1.5]),
                "repairability": E([1.0]),
            }
    edges += [(p, "t") for p in previous]
    return RepairableRBD(edges, components)


@needs_numba
@pytest.mark.parametrize("pairs", [12, 35])
def test_the_engines_agree_on_large_systems(pairs):
    # More components than bits in a 64-bit mask: the compiled loop keeps
    # the structure up to date as components change.
    rbd = pairs_in_series(pairs)
    for options in ({}, {"antithetic": True}, {"n_jobs": 2}):
        identical(
            rbd.availability(
                200.0, mc_samples=40, seed=3, engine="python", **options
            ),
            rbd.availability(
                200.0, mc_samples=40, seed=3, engine="numba", **options
            ),
        )


def maintained_rbds():
    """Systems under age and block replacement, which numba's own loop
    simulates (#155): maintenance in zero time and taking time, preventive
    costs fixed and drawn, fixed lives that fall on the schedule (ties the
    heap orders), and a system of 24 components."""
    pm = {"interval": 30.0}
    timed = {"interval": 40.0, "duration": L([0.5, 0.4]), "cost": 7.0}
    block = {"interval": 25.0, "policy": "block", "cost": G([3.0, 0.5])}
    timed_block = {
        "interval": 35.0,
        "policy": "block",
        "duration": E([1.5]),
    }
    big = pairs_in_series(12)
    big_components = {
        node: {**spec, "preventive": dict(timed if i % 2 else pm)}
        for i, (node, spec) in enumerate(big._init_args["components"].items())
    }
    return {
        "age, instant and timed": RepairableRBD(
            BRIDGE,
            {
                "a": {**unit_spec(70, 2.5), "preventive": pm},
                "b": {**unit_spec(80, 2.0), "preventive": timed},
                "c": unit_spec(90, 1.5),
                "d": {**unit_spec(60, 3.0), "preventive": timed},
                "e": {**unit_spec(75, 1.2), "preventive": pm},
            },
            downtime_cost_rate=4.0,
        ),
        "block, priced": RepairableRBD(
            [("s", "a"), ("s", "b"), ("a", "c"), ("b", "c"), ("c", "t")],
            {
                "a": {**unit_spec(50, 2.0), "preventive": block},
                "b": {
                    **unit_spec(55, 2.0),
                    "preventive": timed_block,
                    "repair_cost": 4.0,
                },
                "c": {**unit_spec(200, 1.5), "preventive": pm},
            },
        ),
        "fixed lives on the schedule": RepairableRBD(
            [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
            {
                "a": {
                    "reliability": X(30.0),
                    "repairability": X(2.0),
                    "preventive": {"interval": 30.0},
                },
                "b": {
                    "reliability": X(15.0),
                    "repairability": "instant",
                    "preventive": {"interval": 15.0, "policy": "block"},
                    "replace_cost": 1.0,
                },
            },
        ),
        "large": RepairableRBD(
            [tuple(e) for e in big._init_args["edges"]], big_components
        ),
    }


def unit_spec(scale, shape):
    return {
        "reliability": W([scale, shape]),
        "repairability": L([0.3, 0.5]),
    }


#: The runs on which the engines are compared under maintenance and
#: inspections.
UPKEEP_RUNS = pytest.mark.parametrize(
    "options",
    [
        dict(t_simulation=300.0, mc_samples=60, seed=41),
        dict(t_simulation=200.0, mc_samples=30, seed=42, method="c"),
        dict(t_simulation=200.0, mc_samples=40, seed=43, antithetic=True),
        dict(t_simulation=500.0, mc_samples=300, seed=44, n_jobs=3),
        dict(t_simulation=200.0, mc_samples=40, seed=45, broken_nodes=["b"]),
        dict(
            t_simulation=100.0,
            mc_samples=50,
            seed=46,
            tolerance=1e-9,
            max_samples=150,
        ),
    ],
    ids=["plain", "cut sets", "antithetic", "threads", "held", "tolerance"],
)


def engines_agree(rbd, options):
    """Whether the Python loop and numba's give the same results, of
    ``availability`` and (with costs) ``cost``, on a system numba's own
    loop runs but an engine of the interface's version is not given."""
    if "broken_nodes" in options and "b" not in rbd.components:
        nodes = list(rbd.components)
        options = {**options, "broken_nodes": [nodes[min(1, len(nodes) - 1)]]}
    plan, _ = rbd._stream_plan(100.0, 1, False)
    assert _compiled.unsupported(rbd, plan, None) is not None
    assert _compiled.unsupported(rbd, plan, None, numba=True) is None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        python = rbd.availability(engine="python", **options)
        compiled = rbd.availability(engine="numba", **options)
        identical(python, compiled)
        if rbd.has_costs:
            identical(
                rbd.cost(engine="python", **options),
                rbd.cost(engine="numba", **options),
            )


@needs_numba
@UPKEEP_RUNS
@pytest.mark.parametrize("name", sorted(maintained_rbds()))
def test_the_engines_agree_on_maintenance(name, options):
    engines_agree(maintained_rbds()[name], options)


@needs_numba
def test_planned_outages_are_counted_as_the_python_loop_counts_them():
    rbd = maintained_rbds()["age, instant and timed"]
    run = dict(t_simulation=400.0, mc_samples=200, seed=7)
    python = rbd.availability(engine="python", **run)
    compiled = rbd.availability(engine="numba", **run)
    assert compiled.system_planned_outages == python.system_planned_outages
    assert compiled.system_planned_outages > 0
    assert compiled.cost.by_category["preventive"] > 0.0


@needs_numba
@pytest.mark.parametrize("upkeep", ["maintained", "inspected"])
def test_a_maintained_run_from_a_state_stays_in_python(upkeep, monkeypatch):
    # A unit new at its age 0 draws nothing at the start, but its phase
    # shifts its block or test calendar, which numba's loop does not follow.
    if upkeep == "maintained":
        rbd = maintained_rbds()["block, priced"]
    else:
        rbd = inspected_rbds()["staggered, priced"]
    state = {"a": NodeState(age=0.0, phase=10.0)}
    with pytest.raises(NotImplementedError, match="started from a state"):
        rbd.availability(
            100.0, mc_samples=5, seed=2, state=state, engine="numba"
        )
    monkeypatch.setattr(_compiled, "worthwhile", lambda *args: True)

    def compiled(*args, **kwargs):
        raise AssertionError("compiled")

    monkeypatch.setattr(_compiled, "Runner", compiled)
    rbd.availability(100.0, mc_samples=5, seed=2, state=state)


def on_condition():
    """A unit replaced on condition at its inspections."""
    return RepairableRBD(
        [("s", "a"), ("a", "t")],
        {
            "a": {
                **unit_spec(80, 2.5),
                "preventive": {
                    "interval": 20.0,
                    "policy": "condition",
                    "threshold": 0.1,
                },
            }
        },
    )


def test_what_numbas_loop_runs_besides_plain_components():
    routes = repairable_rbds()
    for rbd in [
        routes["maintained"],
        routes["inspected"],
        *maintained_rbds().values(),
        *inspected_rbds().values(),
        *crewed_rbds().values(),
        *standby_rbds().values(),
        *nested_rbds().values(),
    ]:
        plan, _ = rbd._stream_plan(100.0, 1, False)
        # Not given to an engine of the interface's version.
        if rbd._crews_limited():
            reason = "repair crews"
        elif rbd._standby:
            reason = "standby groups"
        elif rbd._preventive:
            reason = "preventive maintenance"
        elif rbd._inspection:
            reason = "inspections"
        else:
            reason = "nested RBDs"
        assert reason in _compiled.unsupported(rbd, plan, None)
        assert _compiled.unsupported(rbd, plan, None, numba=True) is None
    timed = {"interval": 30.0, "duration": binomial_first([2, 1.5])}
    unstreamed = RepairableRBD(
        [("s", "a"), ("a", "t")],
        {
            "a": {
                "reliability": binomial_first([80, 2.5]),
                "repairability": E([1.0]),
                "inspection": {**timed, "duration": E([1.0])},
            }
        },
    )
    for rbd, reason in [
        (on_condition(), "replacement on condition"),
        (systems_of_every_kind()["unstreamable maintenance"], "maintenance"),
        (inspected_unit(timed), "the test time of node 'a'"),
        # Its test time could be streamed, its life cannot.
        (unstreamed, "the models of node 'a'"),
    ]:
        plan, _ = rbd._stream_plan(100.0, 1, False)
        assert reason in _compiled.unsupported(rbd, plan, None, numba=True)


def inspected_unit(inspection):
    """A unit whose failures are hidden until ``inspection`` finds them."""
    return RepairableRBD(
        [("s", "a"), ("a", "t")],
        {"a": {**unit_spec(80, 2.5), "inspection": inspection}},
    )


def inspected_rbds():
    """Systems whose failures are hidden until a test finds them, which
    numba's own loop simulates (#155): tests in zero time and taking time,
    staggered, and partial (which can miss a failure, found by a full
    test); inspection and corrective costs fixed and drawn (charged when a
    test finds the failure); fixed lives that fail on a test (found by it);
    inspections beside age and block replacement; and more components than
    the loop tabulates."""
    instant = {"interval": 30.0, "cost": 2.0}
    staggered = {"interval": 30.0, "offset": 15.0, "cost": G([2.0, 1.5])}
    timed = {"interval": 40.0, "offset": 10.0, "duration": L([0.2, 0.4])}
    partial = {
        "interval": 20.0,
        "coverage": 0.6,
        "full_test": 80.0,
        "cost": 1.0,
    }
    partial_timed = {
        "interval": 25.0,
        "offset": 5.0,
        "coverage": 0.8,
        "full_test": 100.0,
        "duration": E([2.0]),
        "cost": G([1.0, 2.0]),
    }
    big = pairs_in_series(12)
    big_components = {
        node: (
            {**spec, "inspection": dict(partial_timed if i % 4 else instant)}
            if i % 2
            else spec
        )
        for i, (node, spec) in enumerate(big._init_args["components"].items())
    }
    return {
        "staggered, priced": RepairableRBD(
            BRIDGE,
            {
                "a": {
                    **unit_spec(70, 2.5),
                    "inspection": instant,
                    "repair_cost": 4.0,
                },
                "b": {
                    **unit_spec(80, 2.0),
                    "inspection": staggered,
                    "replace_cost": G([5.0, 2.0]),
                },
                "c": {**unit_spec(90, 1.5), "repair_cost": 1.0},
                "d": {
                    **unit_spec(60, 3.0),
                    "inspection": timed,
                    "repair_cost": L([1.0, 0.5]),
                    "replace_cost": 3.0,
                },
                "e": {**unit_spec(75, 1.2), "inspection": instant},
            },
            downtime_cost_rate=4.0,
        ),
        "partial tests": RepairableRBD(
            [("s", "a"), ("s", "b"), ("a", "c"), ("b", "c"), ("c", "t")],
            {
                "a": {
                    **unit_spec(50, 2.0),
                    "inspection": partial,
                    "repair_cost": G([2.0, 1.0]),
                },
                "b": {
                    **unit_spec(55, 2.0),
                    "inspection": partial_timed,
                    "replace_cost": 6.0,
                },
                "c": {
                    **unit_spec(200, 1.5),
                    "inspection": {**partial, "offset": 10.0},
                },
            },
        ),
        "fixed lives on the tests": RepairableRBD(
            [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
            {
                "a": {
                    "reliability": X(30.0),
                    "repairability": X(2.0),
                    "inspection": {"interval": 30.0, "cost": 1.0},
                    "repair_cost": 3.0,
                },
                "b": {
                    "reliability": X(20.0),
                    "repairability": "instant",
                    "inspection": {
                        "interval": 10.0,
                        "offset": 5.0,
                        "coverage": 0.5,
                        "full_test": 40.0,
                        "duration": X(1.0),
                    },
                    "replace_cost": 1.0,
                },
            },
        ),
        "beside maintenance": RepairableRBD(
            BRIDGE,
            {
                "a": {**unit_spec(70, 2.5), "preventive": {"interval": 30.0}},
                "b": {
                    **unit_spec(80, 2.0),
                    "preventive": {
                        "interval": 25.0,
                        "policy": "block",
                        "duration": E([1.5]),
                        "cost": 2.0,
                    },
                },
                "c": {**unit_spec(90, 1.5), "inspection": partial_timed},
                "d": {
                    **unit_spec(60, 3.0),
                    "inspection": instant,
                    "repair_cost": 5.0,
                },
                "e": unit_spec(75, 1.2),
            },
            downtime_cost_rate=2.0,
        ),
        "large": RepairableRBD(
            [tuple(e) for e in big._init_args["edges"]], big_components
        ),
    }


@needs_numba
@UPKEEP_RUNS
@pytest.mark.parametrize("name", sorted(inspected_rbds()))
def test_the_engines_agree_on_inspections(name, options):
    engines_agree(inspected_rbds()[name], options)


def crewed_rbds():
    """Systems with fewer repair crews than components, which numba's own
    loop simulates (#155): one crew and two, with priorities and without
    (the job due first, then the one queued first, goes first), maintenance
    and tests that wait for a crew (a unit off line for a test does not age
    while it waits), fixed lives whose jobs fall due together, and more
    components than the loop tabulates."""

    def slow(scale, shape, **extra):
        # Repairs long enough for jobs to wait.
        return {
            "reliability": W([scale, shape]),
            "repairability": L([2.0, 0.5]),
            **extra,
        }

    big = pairs_in_series(12)
    return {
        "one crew, priorities": RepairableRBD(
            BRIDGE,
            {
                "a": slow(70, 2.5, priority=2.0, repair_cost=3.0),
                "b": slow(80, 2.0, priority=1.0),
                "c": slow(90, 1.5),
                "d": slow(60, 3.0, priority=1.0, replace_cost=G([2.0, 1.0])),
                "e": slow(75, 1.2),
            },
            repair_crews=1,
            downtime_cost_rate=4.0,
        ),
        "two crews, maintenance and tests": RepairableRBD(
            BRIDGE,
            {
                "a": slow(
                    70,
                    2.5,
                    preventive={"interval": 30.0, "duration": L([0.5, 0.4])},
                    priority=1.0,
                ),
                "b": slow(
                    80,
                    2.0,
                    preventive={
                        "interval": 25.0,
                        "policy": "block",
                        "duration": E([0.3]),
                        "cost": 2.0,
                    },
                ),
                "c": slow(
                    90,
                    1.5,
                    inspection={
                        "interval": 25.0,
                        "offset": 5.0,
                        "coverage": 0.8,
                        "full_test": 100.0,
                        "duration": E([0.5]),
                    },
                ),
                "d": slow(
                    60,
                    3.0,
                    inspection={
                        "interval": 30.0,
                        "duration": X(1.0),
                        "cost": 1.0,
                    },
                    repair_cost=5.0,
                    priority=3.0,
                ),
                "e": slow(75, 1.2),
            },
            repair_crews=2,
            downtime_cost_rate=2.0,
        ),
        "fixed lives, falling due together": RepairableRBD(
            [
                ("s", "a"),
                ("s", "b"),
                ("s", "c"),
                ("a", "t"),
                ("b", "t"),
                ("c", "t"),
            ],
            {
                "a": {"reliability": X(10.0), "repairability": X(3.0)},
                "b": {"reliability": X(10.0), "repairability": X(2.0)},
                "c": {
                    "reliability": X(10.0),
                    "repairability": X(1.0),
                    "priority": 1.0,
                },
            },
            repair_crews=1,
        ),
        "large": RepairableRBD(
            [tuple(e) for e in big._init_args["edges"]],
            {
                node: {**spec, "repairability": L([2.5, 0.5])}
                for node, spec in big._init_args["components"].items()
            },
            repair_crews=3,
        ),
    }


@needs_numba
@UPKEEP_RUNS
@pytest.mark.parametrize("name", sorted(crewed_rbds()))
def test_the_engines_agree_with_repair_crews(name, options):
    engines_agree(crewed_rbds()[name], options)


@needs_numba
def test_jobs_wait_for_a_crew_in_the_compiled_loop():
    rbd = crewed_rbds()["one crew, priorities"]
    run = dict(t_simulation=1000.0, mc_samples=200, seed=9, engine="numba")
    crewed = rbd.availability(**run)
    unlimited = RepairableRBD(
        BRIDGE, rbd._init_args["components"], downtime_cost_rate=4.0
    ).availability(**run)
    # The same draws, but repairs that wait for the crew.
    for node in "ae":
        assert crewed.node_uptime[node] < unlimited.node_uptime[node]
    # Three units fail at 10, for one crew: the first to fail takes it, and
    # of the two left waiting, the one of higher priority goes first,
    # though it was queued second.
    fixed = crewed_rbds()["fixed lives, falling due together"]
    run = dict(t_simulation=20.0, mc_samples=1, seed=1)
    result = fixed.availability(engine="numba", **run)
    identical(result, fixed.availability(engine="python", **run))
    down = {node: 20.0 - up for node, up in result.node_uptime.items()}
    # a: repaired 10 to 13; c: waits 3, repaired by 14; b: waits 4 (its
    # repair of 2 then ends at 16).
    assert down == pytest.approx({"a": 3.0, "b": 6.0, "c": 4.0})
    assert result.system_uptime == pytest.approx(17.0)


def standby_rbds():
    """Systems with standby groups, which numba's own loop simulates
    (#155): cold, warm and hot spares, switches that fail, groups that
    share repair crews with each other and with plain, maintained and
    tested components, fixed lives whose events fall together, and more
    components than the loop tabulates."""

    def slow(scale, shape, **extra):
        return {
            "reliability": W([scale, shape]),
            "repairability": L([2.0, 0.5]),
            **extra,
        }

    big = pairs_in_series(12)
    return {
        "cold and warm, crews and the rest": RepairableRBD(
            BRIDGE,
            {
                "a": slow(
                    70,
                    2.5,
                    standby={"units": 3, "dormancy_factor": 0.2},
                    priority=1.0,
                    repair_cost=2.0,
                ),
                "b": slow(
                    80,
                    2.0,
                    standby={"units": 2, "switching_probability": 0.7},
                    replace_cost=G([2.0, 1.0]),
                ),
                "c": slow(90, 1.5),
                "d": slow(
                    60,
                    3.0,
                    preventive={"interval": 30.0, "duration": E([0.5])},
                ),
                "e": slow(
                    75,
                    1.2,
                    inspection={
                        "interval": 25.0,
                        "duration": X(1.0),
                        "coverage": 0.7,
                        "full_test": 75.0,
                    },
                ),
            },
            repair_crews=2,
            downtime_cost_rate=3.0,
        ),
        "hot and switching, crews enough": RepairableRBD(
            BRIDGE,
            {
                "a": slow(
                    70,
                    2.5,
                    standby={"units": 4, "k": 2, "dormancy_factor": 1.0},
                ),
                "b": slow(
                    80,
                    2.0,
                    standby={"units": 2, "switching_probability": 0.0},
                ),
                "c": slow(90, 1.5),
                "d": slow(60, 3.0),
                "e": slow(75, 1.2),
            },
        ),
        "fixed lives, falling together": RepairableRBD(
            [("s", "g"), ("s", "h"), ("g", "t"), ("h", "t")],
            {
                "g": {
                    "reliability": X(5.0),
                    "repairability": X(3.0),
                    "standby": {"units": 3},
                },
                "h": {
                    "reliability": X(5.0),
                    "repairability": X(2.0),
                    "standby": {"units": 2},
                    "priority": 1.0,
                },
            },
            repair_crews=1,
        ),
        "large": RepairableRBD(
            [tuple(e) for e in big._init_args["edges"]],
            {
                node: {
                    **spec,
                    "repairability": L([2.5, 0.5]),
                    **({"standby": {"units": 2}} if i % 3 == 0 else {}),
                }
                for i, (node, spec) in enumerate(
                    big._init_args["components"].items()
                )
            },
            repair_crews=4,
        ),
    }


@needs_numba
@UPKEEP_RUNS
@pytest.mark.parametrize("name", sorted(standby_rbds()))
def test_the_engines_agree_on_standby_groups(name, options):
    engines_agree(standby_rbds()[name], options)


def nested_rbds():
    """Systems with nested RBDs, which numba's own loop simulates (#155),
    each a level of its own stepped to its next change: nested RBDs
    maintained (their planned outages planned outages of the system),
    tested, with crews of their own beside the system's, with standby
    groups, nested three deep, and with fixed lives whose events fall
    together across levels."""

    def unit(scale, shape, repair=None, **extra):
        return {
            "reliability": W([scale, shape]),
            "repairability": L([1.0, 0.5]) if repair is None else repair,
            **extra,
        }

    pair = [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")]
    bridge = [
        ("s", "a"),
        ("s", "b"),
        ("a", "c"),
        ("b", "c"),
        ("a", "d"),
        ("b", "e"),
        ("c", "d"),
        ("c", "e"),
        ("d", "t"),
        ("e", "t"),
    ]
    slow = L([2.0, 0.5])
    crewed = RepairableRBD(
        [
            ("s", "a"),
            ("s", "b"),
            ("s", "c"),
            ("a", "t"),
            ("b", "t"),
            ("c", "t"),
        ],
        {
            "a": unit(40, 2.0, slow, priority=1.0),
            "b": unit(45, 2.0, slow),
            "c": unit(50, 2.0, slow),
        },
        repair_crews=1,
    )
    return {
        "maintained inside": RepairableRBD(
            [("s", "m"), ("m", "c"), ("c", "t")],
            {
                "m": RepairableRBD(
                    pair,
                    {
                        "a": unit(
                            50,
                            2.5,
                            preventive={
                                "interval": 20.0,
                                "duration": E([1.0]),
                            },
                        ),
                        "b": unit(
                            55,
                            2.0,
                            preventive={"interval": 15.0, "policy": "block"},
                        ),
                    },
                ),
                "c": unit(90, 1.5, repair_cost=2.0),
            },
            downtime_cost_rate=1.0,
        ),
        "tested inside": RepairableRBD(
            [("s", "m"), ("s", "c"), ("m", "t"), ("c", "t")],
            {
                "m": RepairableRBD(
                    pair,
                    {
                        "a": unit(
                            50,
                            2.5,
                            inspection={"interval": 10.0, "duration": X(0.5)},
                        ),
                        "b": unit(
                            55,
                            2.0,
                            inspection={
                                "interval": 12.0,
                                "offset": 6.0,
                                "coverage": 0.7,
                                "full_test": 48.0,
                            },
                        ),
                    },
                ),
                "c": unit(60, 1.5),
            },
        ),
        "crews inside and out": RepairableRBD(
            [
                ("s", "m"),
                ("s", "c"),
                ("s", "d"),
                ("m", "t"),
                ("c", "t"),
                ("d", "t"),
            ],
            {
                "m": crewed,
                "c": unit(40, 2.0, slow),
                "d": unit(45, 2.0, slow, priority=2.0),
            },
            repair_crews=1,
        ),
        "standby inside": RepairableRBD(
            [("s", "m"), ("m", "c"), ("c", "t")],
            {
                "m": RepairableRBD(
                    [("s", "g"), ("g", "x"), ("x", "t")],
                    {
                        "g": unit(
                            40,
                            2.0,
                            slow,
                            standby={
                                "units": 3,
                                "dormancy_factor": 0.3,
                                "switching_probability": 0.9,
                            },
                        ),
                        "x": unit(80, 1.5),
                    },
                    repair_crews=1,
                ),
                "c": unit(90, 1.5),
            },
        ),
        "three levels": RepairableRBD(
            [("s", "m"), ("m", "t")],
            {
                "m": RepairableRBD(
                    [("s", "x"), ("s", "y"), ("x", "t"), ("y", "t")],
                    {
                        "x": RepairableRBD(
                            pair, {"a": unit(30, 2.0), "b": unit(35, 2.0)}
                        ),
                        "y": crewed,
                    },
                )
            },
        ),
        "fixed lives across levels": RepairableRBD(
            [("s", "m"), ("m", "c"), ("c", "t")],
            {
                "m": RepairableRBD(
                    pair,
                    {
                        "a": {"reliability": X(10.0), "repairability": X(2.0)},
                        "b": {
                            "reliability": X(10.0),
                            "repairability": "instant",
                        },
                    },
                ),
                "c": {"reliability": X(10.0), "repairability": X(1.0)},
            },
        ),
        # More components inside than bits in a 64-bit mask; and a core of
        # path sets at both levels, whose working path sets each level
        # counts on its own (#255).
        "wide inside": RepairableRBD(
            [("s", "m"), ("m", "c"), ("c", "t")],
            {"m": pairs_in_series(35), "c": unit(60, 1.5)},
        ),
        "bridges inside and out": RepairableRBD(
            bridge,
            {
                "a": RepairableRBD(
                    bridge,
                    {x: unit(40 + 3 * i, 1.5) for i, x in enumerate("abcde")},
                ),
                **{x: unit(50 + 7 * i, 2.0) for i, x in enumerate("bcde")},
            },
        ),
    }


@needs_numba
@UPKEEP_RUNS
@pytest.mark.parametrize("name", sorted(nested_rbds()))
def test_the_engines_agree_on_nested_rbds(name, options):
    engines_agree(nested_rbds()[name], options)


@needs_numba
def test_a_nested_rbd_s_planned_outage_is_the_system_s():
    rbd = nested_rbds()["maintained inside"]
    run = dict(t_simulation=500.0, mc_samples=100, seed=3)
    compiled = rbd.availability(engine="numba", **run)
    identical(compiled, rbd.availability(engine="python", **run))
    assert compiled.system_planned_outages > 0
    # Held, a nested RBD is not simulated at all.
    for held in ("broken_nodes", "working_nodes"):
        identical(
            rbd.availability(engine="numba", **{held: ["m"]}, **run),
            rbd.availability(engine="python", **{held: ["m"]}, **run),
        )


def test_what_numba_does_not_run_inside_a_nested_rbd():
    unit = {"reliability": W([50, 2.0]), "repairability": L([1.0, 0.5])}

    def outer(inner):
        return RepairableRBD(
            [("s", "m"), ("m", "c"), ("c", "t")], {"m": inner, "c": dict(unit)}
        )

    imperfect = RepairableRBD(
        [("s", "a"), ("a", "t")],
        {"a": {**unit, "repair": {"model": "kijima1", "q": 0.5}}},
    )
    for inner, reason in [
        (imperfect, "imperfect repair"),
        (on_condition(), "replacement on condition"),
    ]:
        rbd = outer(inner)
        plan, _ = rbd._stream_plan(100.0, 1, False)
        assert reason in _compiled.unsupported(rbd, plan, None, numba=True)
        # An engine of the interface's version is given no nested RBD.
        assert _compiled.unsupported(rbd, plan, None) == "nested RBDs"
    rbd = nested_rbds()["maintained inside"]
    plan, _ = rbd._stream_plan(100.0, 1, False)
    state = {"c": NodeState(age=0.0, phase=1.0)}
    assert "started from a state" in _compiled.unsupported(
        rbd, plan, None, numba=True, states=state
    )


def capacity_rbds():
    """Systems with capacities, which numba's own loop follows (#155), and
    the demand each run is measured against: capacities of one level and
    of several (a node working at several levels), a node without one
    (unlimited), a demand given and the design capacity's, with
    maintenance, tests, crews and standby groups, a nested RBD's capacity,
    and a system of 24 components."""

    def unit(scale, shape, repair=None, **extra):
        return {
            "reliability": W([scale, shape]),
            "repairability": L([1.0, 0.5]) if repair is None else repair,
            **extra,
        }

    three = [("s", "a"), ("s", "b"), ("s", "c")]
    three += [("a", "t"), ("b", "t"), ("c", "t")]
    slow = L([2.0, 0.5])
    big = pairs_in_series(12)
    return {
        "pumps": (pumps_with_capacities(), {}),
        "levels, a demand": (
            RepairableRBD(
                three,
                {node: unit(50, 2.0) for node in "abc"},
                capacity={
                    "a": {100: 0.75, 40: 0.25},
                    "b": 50.0,
                    "c": {60: 0.5, 30: 0.5},
                },
            ),
            {"demand": 150.0},
        ),
        "unlimited": (
            RepairableRBD(
                [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
                {"a": unit(50, 2.0), "b": unit(60, 2.0)},
                capacity={"a": 10.0},
            ),
            {},
        ),
        "maintained, tested, crews and a group": (
            RepairableRBD(
                three,
                {
                    "a": unit(
                        50,
                        2.0,
                        preventive={"interval": 20.0, "duration": E([1.0])},
                    ),
                    "b": unit(55, 2.0, slow, standby={"units": 2}),
                    "c": unit(
                        60,
                        2.0,
                        slow,
                        inspection={"interval": 15.0, "duration": X(0.5)},
                    ),
                },
                capacity={"a": 30.0, "b": 40.0, "c": 50.0},
                repair_crews=1,
                downtime_cost_rate=1.0,
            ),
            {"demand": 90.0},
        ),
        "nested": (
            RepairableRBD(
                [("s", "m"), ("s", "c"), ("m", "t"), ("c", "t")],
                {
                    "m": RepairableRBD(
                        [("s", "x"), ("s", "y"), ("x", "t"), ("y", "t")],
                        {"x": unit(40, 2.0), "y": unit(45, 2.0)},
                    ),
                    "c": unit(60, 1.5),
                },
                capacity={"m": 20.0, "c": 30.0},
            ),
            {},
        ),
        "large": (
            RepairableRBD(
                [tuple(e) for e in big._init_args["edges"]],
                big._init_args["components"],
                capacity={
                    node: 10.0 + i
                    for i, node in enumerate(big._init_args["components"])
                },
            ),
            {},
        ),
    }


@needs_numba
@UPKEEP_RUNS
@pytest.mark.parametrize("name", sorted(capacity_rbds()))
def test_the_engines_agree_on_capacities(name, options):
    rbd, extra = capacity_rbds()[name]
    if "broken_nodes" in options and "b" not in rbd.components:
        options = {**options, "broken_nodes": [list(rbd.components)[0]]}
    plan, _ = rbd._stream_plan(100.0, 1, False)
    # An engine of the interface's version is given no capacities.
    assert _compiled.unsupported(rbd, plan, object()) is not None
    assert _compiled.unsupported(rbd, plan, object(), numba=True) is None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        python = rbd.availability(engine="python", **extra, **options)
        compiled = rbd.availability(engine="numba", **extra, **options)
    identical(python, compiled)
    assert compiled.capacity_time


def test_capacities_numba_does_not_follow():
    wide = pairs_in_series(32)
    plan, _ = wide._stream_plan(100.0, 1, False)
    assert "more than 63" in _compiled.unsupported(
        wide, plan, object(), numba=True
    )


@needs_numba
def test_found_failures_are_charged_as_the_python_loop_charges_them():
    rbd = inspected_rbds()["staggered, priced"]
    run = dict(t_simulation=400.0, mc_samples=200, seed=7)
    python = rbd.cost(engine="python", **run)
    compiled = rbd.cost(engine="numba", **run)
    identical(python, compiled)
    for category in ("repair", "replace", "inspection", "system_downtime"):
        assert compiled.by_category[category] > 0.0
    # Tests that take time are planned outages.
    timed = rbd.availability(engine="numba", **run)
    assert timed.system_planned_outages > 0
    partial = inspected_rbds()["partial tests"]
    run = dict(t_simulation=1000.0, mc_samples=50, seed=8)
    identical(
        partial.availability(engine="python", **run),
        partial.availability(engine="numba", **run),
    )


@needs_numba
def test_a_run_on_threads_leaves_numbas_thread_count_alone():
    import numba

    before = numba.get_num_threads()
    plain_rbds()["bridge"].availability(
        100.0, mc_samples=50, seed=1, engine="numba", n_jobs=2
    )
    assert numba.get_num_threads() == before


@needs_numba
def test_the_engines_agree_on_costs_and_comparisons():
    rbd = plain_rbds()["costed"]
    identical(
        rbd.cost(200.0, mc_samples=50, seed=5, engine="python"),
        rbd.cost(200.0, mc_samples=50, seed=5, engine="numba"),
    )
    faster = RepairableRBD(
        [("s", "x"), ("x", "y"), ("s", "z"), ("y", "t"), ("z", "t")],
        {
            node: {**spec, "repairability": E([2.0])}
            for node, spec in {
                "x": {"reliability": E([0.02]), "replace_cost": 5.0},
                "y": {
                    "reliability": W([80, 2.5], gamma=5),
                    "repair_cost": 40.0,
                },
                "z": {"reliability": L([4.0, 0.7])},
            }.items()
        },
        downtime_cost_rate=10.0,
    )
    for quantity in ("availability", "cost"):
        identical(
            rbd.compare(
                faster,
                100.0,
                mc_samples=300,
                seed=6,
                quantity=quantity,
                engine="python",
                control_variate=False,
            ),
            rbd.compare(
                faster,
                100.0,
                mc_samples=300,
                seed=6,
                quantity=quantity,
                engine="numba",
                control_variate=False,
            ),
        )


@needs_numba
def test_the_compiled_engine_runs_out_and_carries_on(monkeypatch):
    # Streams that start with a single row, and little room for the
    # system's changes: the compiled loop keeps running simulations again
    # with more, and ends with the same results.
    rbd = plain_rbds()["koon"]
    monkeypatch.setattr(_streams, "first_rows", lambda expected: 1)
    expected = rbd.availability(300.0, mc_samples=200, seed=8, engine="python")
    monkeypatch.setattr(_compiled, "BATCH_BYTES", 2**12)
    short = rbd.availability(300.0, mc_samples=200, seed=8, engine="numba")
    identical(expected, short)
    assert short.system_failures > 0


@needs_numba
def test_the_compiled_heap_releases_ties_as_heapq_does():
    from repyability.rbd import _kernel

    rng = np.random.default_rng(5)
    size = 64
    arrays = (
        np.empty(size),
        np.empty(size, np.int64),
        np.empty(size, np.int8),
        np.empty(size, np.int64),
    )
    heap: list = []
    count = 0
    for step in range(5000):
        if rng.random() < 0.55 and count < size or count == 0:
            t = float(rng.integers(0, 12))
            count = _kernel._push.py_func(arrays, count, t, step, 1, -step)
            heapq.heappush(heap, (t, Event(t, step, True)))
        else:
            t, node, _, tag, count = _kernel._pop.py_func(arrays, count)
            expected = heapq.heappop(heap)[1]
            assert (t, node, tag) == (
                expected.time,
                expected.component,
                -expected.component,
            )


@needs_numba
def test_auto_compiles_long_runs(monkeypatch):
    rbd = plain_rbds()["bridge"]
    chosen = []
    runner = _compiled.Runner

    def spy(*args, **kwargs):
        chosen.append(True)
        return runner(*args, **kwargs)

    monkeypatch.setattr(_compiled, "Runner", spy)
    monkeypatch.setattr(_compiled, "compiled", lambda: False)
    rbd.availability(100.0, mc_samples=10, seed=1)
    assert not chosen
    monkeypatch.setattr(_compiled, "AUTO_DRAWS", 1)
    rbd.availability(100.0, mc_samples=10, seed=1)
    assert chosen


def random_repairable(seed):
    """A random diagram (bridges, shared nodes, votes, the odd direct edge
    from the input to the output) of plain repairable components."""
    from repyability.tests.test_rbd_modular import random_diagram

    rng = np.random.default_rng(seed)
    edges, k = random_diagram(rng, int(rng.integers(3, 14)))
    nodes = sorted({v for e in edges for v in e} - {"s", "t"})
    return RepairableRBD(
        edges,
        {
            v: {
                "reliability": W([30 + 5 * i, 1.2 + 0.1 * (i % 4)]),
                "repairability": L([0.2, 0.5]),
            }
            for i, v in enumerate(nodes)
        },
        k=k,
    )


@needs_numba
@pytest.mark.parametrize("seed", range(12))
def test_the_engines_agree_keeping_the_structure_up_to_date(seed):
    # The compiled loop keeps whether the system works up to date as
    # components change, through modules and a core of path sets alike.
    rbd = random_repairable(seed)
    for extra in ({}, {"broken_nodes": [rbd.nodes[0]]}, {"n_jobs": 2}):
        options = dict(t_simulation=150.0, mc_samples=40, seed=seed, **extra)
        identical(
            rbd.availability(engine="python", **options),
            rbd.availability(engine="numba", **options),
        )


@pytest.mark.parametrize("seed", range(8))
def test_the_kept_structure_starts_as_the_structure_function(seed):
    rbd = random_repairable(seed)
    nodes = list(rbd.components)
    structure = _compiled._structure(
        rbd, {node: c for c, node in enumerate(nodes)}
    )
    rng = np.random.default_rng(seed)
    for _ in range(10):
        start = (rng.random(len(nodes)) < 0.6).astype(np.int8)
        kept = _compiled._kept([structure], start)
        root, always, value, working = kept[10], kept[11], kept[12], kept[15]
        if always[0]:
            works = True
        elif root[0] >= 0:
            works = bool(value[root[0]])
        else:
            works = working[0] > 0
        assert works == rbd.is_system_working(
            {node: bool(start[c]) for c, node in enumerate(nodes)}, "p"
        )


@pytest.mark.parametrize(
    "changes",
    ["none", "random", "at zero", "repeated times"],
)
def test_the_curve_is_the_same_from_changes_in_order(changes):
    # An engine may keep its changes in order of time; the curve worked
    # out from them as they come is the one worked out by sorting them.
    rng = np.random.default_rng(3)
    times = {
        "none": np.zeros(0),
        "random": rng.random(5000) * 100.0,
        "at zero": np.concatenate(([0.0, 0.0], rng.random(50) * 100.0)),
        "repeated times": np.repeat(rng.random(300) * 100.0, 3),
    }[changes]
    deltas = rng.choice([-1, 1], times.size).astype(np.int64)
    order = np.argsort(times, kind="stable")
    for start in (0, 40):
        shuffled = repairable_rbd._working_over_time(
            times, deltas, 100.0, start
        )
        ordered = repairable_rbd._working_over_time(
            times[order], deltas[order], 100.0, start
        )
        for a, b in zip(shuffled, ordered):
            assert a.dtype == b.dtype and np.array_equal(a, b)
        time, working = ordered
        assert time[0] == 0.0 and time[-1] == 100.0
        assert working[0] == start + deltas[times == 0.0].sum()
