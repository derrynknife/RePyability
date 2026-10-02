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

from repyability import RepairableRBD
from repyability.non_repairable import NonRepairable
from repyability.rbd import _compiled, _streams, repairable_rbd
from repyability.rbd.repairable_rbd import Event
from repyability.tests.keyed_draws import KeyedDraws, reference_draw
from repyability.tests.test_performance_equivalence import (
    binomial_first,
    instrument_air,
    overlaps_one_at_a_time,
    pumps_with_capacities,
    repairable_rbds,
)

E = surv.Exponential.from_params
W = surv.Weibull.from_params
L = surv.LogNormal.from_params
X = surv.ExactEventTime.from_params
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
                    "reliability": W([30, 2], p=0.7),
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
        uptime = repairable_rbd.time_at_status(system, 1)
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


def systems_of_every_kind():
    systems = dict(repairable_rbds())
    systems["instrument air"] = instrument_air()
    systems["capacities"] = pumps_with_capacities()
    # A maintenance time that cannot be streamed, likewise.
    systems["unstreamable maintenance"] = RepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        {
            name: {
                "reliability": W([70, 2]),
                "repairability": E([0.8]),
                "preventive": {
                    "interval": 30.0,
                    "duration": binomial_first([2, 1.5]),
                    "cost": 5.0,
                },
            }
            for name in "ab"
        },
    )
    # A component whose draws cannot be streamed draws from numpy's global
    # RNG, seeded for each simulation; the others still stream.
    systems["unstreamable"] = RepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        {
            "a": {"reliability": W([70, 1.5]), "repairability": E([0.8])},
            "b": {
                "reliability": binomial_first([40, 2]),
                "repairability": E([0.5]),
            },
        },
    )
    return systems


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
        )
    n = stopped.n_simulations
    identical(
        stopped,
        rbd.availability(window, mc_samples=n, seed=51, engine="python"),
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
        mixed.compare(plain, 100.0, mc_samples=10, seed=1)


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
    assert _compiled.unsupported(rbd, plan, None) == "node 'sub''s LoggedUnit"

    def compiled(*args, **kwargs):
        raise AssertionError("compiled")

    monkeypatch.setattr(_compiled, "worthwhile", lambda plan, N: True)
    monkeypatch.setattr(_compiled, "Runner", compiled)
    rbd.availability(100.0, mc_samples=5, seed=2)


@needs_numba
def test_the_compiled_engine_refuses_what_it_cannot_run():
    rbd = repairable_rbds()["maintained"]
    with pytest.raises(NotImplementedError, match="preventive maintenance"):
        rbd.availability(100.0, mc_samples=5, seed=2, engine="numba")
    bridge = plain_rbds()["bridge"]
    crewed = RepairableRBD(
        [tuple(e) for e in bridge._init_args["edges"]],
        bridge._init_args["components"],
        repair_crews=1,
    )
    with pytest.raises(NotImplementedError, match="repair crews"):
        crewed.availability(100.0, mc_samples=5, seed=2, engine="numba")


def test_without_numba_the_compiled_engine_cannot_be_asked_for(monkeypatch):
    monkeypatch.setattr(_compiled, "available", lambda: False)
    rbd = plain_rbds()["bridge"]
    with pytest.raises(ImportError, match="repyability\\[fast\\]"):
        rbd.availability(10.0, mc_samples=2, seed=1, engine="numba")
    # "auto" runs in Python.
    rbd.availability(10.0, mc_samples=2, seed=1)


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
    # More components than the compiled loop tabulates the system's states
    # for, and more than bits in a 64-bit mask: it keeps the structure up
    # to date as components change instead.
    rbd = pairs_in_series(pairs)
    assert len(rbd.components) > _compiled.MAX_TABLED
    for options in ({}, {"antithetic": True}, {"n_jobs": 2}):
        identical(
            rbd.availability(
                200.0, mc_samples=40, seed=3, engine="python", **options
            ),
            rbd.availability(
                200.0, mc_samples=40, seed=3, engine="numba", **options
            ),
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
            ),
            rbd.compare(
                faster,
                100.0,
                mc_samples=300,
                seed=6,
                quantity=quantity,
                engine="numba",
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
    times = np.empty(size)
    nodes = np.empty(size, np.int64)
    states = np.empty(size, np.int8)
    heap: list = []
    count = 0
    for step in range(5000):
        if rng.random() < 0.55 and count < size or count == 0:
            t = float(rng.integers(0, 12))
            count = _kernel._push.py_func(
                times, nodes, states, count, t, step, 1
            )
            heapq.heappush(heap, (t, Event(t, step, True)))
        else:
            t, node, _, count = _kernel._pop.py_func(
                times, nodes, states, count
            )
            expected = heapq.heappop(heap)[1]
            assert (t, node) == (expected.time, expected.component)


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
def test_the_engines_agree_keeping_the_structure_up_to_date(seed, monkeypatch):
    # Without a table of every state (forced here on small diagrams), the
    # compiled loop keeps whether the system works up to date as components
    # change, through modules and a core of path sets alike.
    monkeypatch.setattr(_compiled, "MAX_TABLED", 0)
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
    root, always = structure[6], structure[7]
    rng = np.random.default_rng(seed)
    for _ in range(10):
        start = (rng.random(len(nodes)) < 0.6).astype(np.int8)
        kept = _compiled._kept(structure, start)
        value, working_paths = kept[9], kept[12]
        if always:
            works = True
        elif root >= 0:
            works = bool(value[root])
        else:
            works = working_paths > 0
        assert works == rbd.is_system_working(
            {node: bool(start[c]) for c, node in enumerate(nodes)}, "p"
        )
