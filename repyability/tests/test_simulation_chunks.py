"""Chunks of a simulation run (#114): simulations a to b of a run, run
anywhere, saved, and merged into the run's result; and blocks of a
NonRepairableRBD's lifetimes."""

import numpy as np
import pytest
import surpyval as surv

from repyability import NonRepairableRBD, RepairableRBD, SimulationChunk

E, W = surv.Exponential.from_params, surv.Weibull.from_params

EDGES = [("s", "A"), ("s", "B"), ("A", "C"), ("B", "C"), ("C", "t")]


def plant(**options):
    def unit(scale, repair, **costs):
        return {
            "reliability": W([scale, 1.5]),
            "repairability": E([repair]),
            **costs,
        }

    return RepairableRBD(
        EDGES,
        {
            "A": unit(10.0, 1.0, repair_cost=5.0),
            "B": unit(10.0, 1.0),
            "C": unit(50.0, 0.5, replace_cost=E([0.01])),
        },
        downtime_cost_rate=2.0,
        **options,
    )


def same_result(a, b, rel=1e-12):
    """Whether two results hold the same simulations: per-simulation values
    and timeline exactly, totals to rounding."""
    assert np.array_equal(a.uptimes, b.uptimes)
    assert np.array_equal(a.timeline, b.timeline)
    assert np.array_equal(a.availability, b.availability)
    assert (a.system_failures, a.system_restorations) == (
        b.system_failures,
        b.system_restorations,
    )
    assert a.n_simulations == b.n_simulations
    for node in a.node_uptime:
        assert a.node_uptime[node] == pytest.approx(
            b.node_uptime[node], rel=rel
        )
    if a.cost is not None:
        assert np.array_equal(a.cost.samples, b.cost.samples)
        for key, value in a.cost.by_category.items():
            assert value == pytest.approx(b.cost.by_category[key], rel=rel)
    return True


@pytest.mark.parametrize("engine", ["python", "numba"])
def test_chunks_merge_into_the_run(engine):
    if engine == "numba":
        pytest.importorskip("numba")
    rbd = plant()
    whole = rbd.availability(100.0, mc_samples=1000, seed=4, engine=engine)
    chunks = [
        rbd.simulate_chunk(100.0, a, b, seed=4, engine=engine)
        for a, b in [(0, 250), (250, 600), (600, 1000)]
    ]
    # Saved, sent back out of order, merged.
    saved = [SimulationChunk.from_json(chunk.to_json()) for chunk in chunks]
    merged = rbd.availability_from_chunks([saved[2], saved[0], saved[1]])
    assert same_result(merged, whole)
    # Merged chunks are a chunk.
    one = SimulationChunk.merge(chunks)
    assert one.ranges == [(0, 1000)] and one.n_simulations == 1000
    assert same_result(rbd.availability_from_chunks(one), whole)
    # The inputs are untouched.
    assert chunks[0].n_simulations == 250
    assert rbd.availability_from_chunks(chunks[0]).n_simulations == 250


def test_a_chunk_is_the_same_however_it_is_run():
    rbd = plant()
    alone = rbd.simulate_chunk(100.0, 300, 500, seed=9, engine="python")
    parallel = rbd.simulate_chunk(
        100.0, 300, 500, seed=9, engine="python", n_jobs=2
    )
    whole = rbd.availability(100.0, mc_samples=500, seed=9)
    assert np.array_equal(
        rbd.availability_from_chunks(alone).uptimes, whole.uptimes[300:]
    )
    assert same_result(
        rbd.availability_from_chunks(parallel),
        rbd.availability_from_chunks(alone),
    )


def test_chunks_with_gaps_and_out_of_order():
    rbd = plant()
    a = rbd.simulate_chunk(100.0, 0, 100, seed=1)
    b = rbd.simulate_chunk(100.0, 200, 300, seed=1)
    c = rbd.simulate_chunk(100.0, 100, 200, seed=1)
    gappy = SimulationChunk.merge([b, a])
    assert gappy.ranges == [(0, 100), (200, 300)]
    assert gappy.n_simulations == 200
    result = rbd.availability_from_chunks(gappy)
    whole = rbd.availability(100.0, mc_samples=300, seed=1)
    assert np.array_equal(
        result.uptimes,
        np.concatenate([whole.uptimes[:100], whole.uptimes[200:]]),
    )
    # The missing simulations can't come between the gappy chunk's.
    with pytest.raises(ValueError, match="interleave"):
        SimulationChunk.merge([gappy, c])
    with pytest.raises(ValueError, match="overlap"):
        SimulationChunk.merge([a, rbd.simulate_chunk(100.0, 50, 150, seed=1)])
    with pytest.raises(ValueError, match="no chunks"):
        SimulationChunk.merge([])


@pytest.mark.parametrize(
    "other",
    [
        {"t_simulation": 50.0},
        {"seed": 2},
        {"working_nodes": ["A"]},
        {"broken_nodes": ["A"]},
        {"method": "c"},
        {"antithetic": True},
    ],
    ids=lambda o: next(iter(o)),
)
def test_chunks_of_different_runs_do_not_merge(other):
    rbd = plant()
    settings = dict(t_simulation=100.0, seed=1)
    a = rbd.simulate_chunk(start=0, stop=100, **settings)
    b = rbd.simulate_chunk(start=100, stop=200, **{**settings, **other})
    with pytest.raises(ValueError, match="different runs"):
        SimulationChunk.merge([a, b])


def test_a_chunk_belongs_to_its_system():
    rbd = plant()
    chunk = rbd.simulate_chunk(100.0, 0, 50, seed=1)
    # The same system, saved and loaded, is the same system.
    loaded = RepairableRBD.from_json(rbd.to_json())
    other = loaded.simulate_chunk(100.0, 50, 100, seed=1)
    assert rbd.availability_from_chunks([chunk, other]).n_simulations == 100
    changed = plant(repair_crews=1)
    with pytest.raises(ValueError, match="another system"):
        changed.availability_from_chunks(chunk)


def test_antithetic_chunks_hold_whole_pairs():
    rbd = plant()
    whole = rbd.availability(100.0, mc_samples=400, seed=3, antithetic=True)
    chunks = [
        rbd.simulate_chunk(100.0, a, b, seed=3, antithetic=True)
        for a, b in [(0, 160), (160, 400)]
    ]
    merged = rbd.availability_from_chunks(chunks)
    assert same_result(merged, whole)
    assert (
        merged.mean_availability_interval().standard_error
        == whole.mean_availability_interval().standard_error
    )
    with pytest.raises(ValueError, match="even"):
        rbd.simulate_chunk(100.0, 0, 151, seed=3, antithetic=True)


def test_chunks_follow_the_capacity():
    rbd = RepairableRBD(
        EDGES,
        {
            node: {"reliability": E([0.1]), "repairability": E([1.0])}
            for node in "ABC"
        },
        capacity={"A": 60.0, "B": 60.0, "C": 100.0},
    )
    whole = rbd.availability(50.0, mc_samples=300, seed=2, demand=100.0)
    chunks = [
        rbd.simulate_chunk(50.0, a, b, seed=2, demand=100.0)
        for a, b in [(0, 120), (120, 300)]
    ]
    merged = rbd.availability_from_chunks(chunks)
    assert np.array_equal(merged.delivered, whole.delivered)
    assert np.array_equal(merged.capacity_timeline, whole.capacity_timeline)
    np.testing.assert_allclose(merged.capacity, whole.capacity, rtol=1e-12)
    assert merged.capacity_time.keys() == whole.capacity_time.keys()


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(start=0, stop=10, seed=None), "needs the run's seed"),
        (dict(start=10, stop=10, seed=1), "start < stop"),
        (dict(start=-1, stop=10, seed=1), "start < stop"),
        (dict(start=0.5, stop=10, seed=1), "whole number"),
        (dict(start=0, stop=True, seed=1), "whole number"),
        (dict(start=0, stop=10, seed=1, demand=5.0), "capacities"),
        (dict(start=0, stop=10, seed=1, method="x"), "'p' or 'c'"),
    ],
)
def test_the_chunk_is_checked(kwargs, message):
    with pytest.raises(ValueError, match=message):
        plant().simulate_chunk(100.0, **kwargs)
    with pytest.raises(ValueError, match="not a saved SimulationChunk"):
        SimulationChunk.from_dict({"kind": "AvailabilityResult"})


def test_random_blocks_are_the_draws_blocks():
    unit = W([100.0, 2.0])
    rbd = NonRepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        {"a": unit, "b": unit},
    )
    whole = rbd.random(25_000, seed=5, n_jobs=2)
    for block in range(3):
        part = whole[block * 10_000 : (block + 1) * 10_000]
        assert np.array_equal(
            rbd.random_block(block, seed=5)[: part.size], part
        )
    paired = rbd.random(20_000, seed=6, n_jobs=1, antithetic=True)
    assert np.array_equal(
        rbd.random_block(1, seed=6, antithetic=True), paired[10_000:]
    )
    for block, seed in ((-1, 5), (1.0, 5), (True, 5), (0, None)):
        with pytest.raises(ValueError):
            rbd.random_block(block, seed=seed)
