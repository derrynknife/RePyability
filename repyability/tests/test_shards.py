"""Shards (#152): a run's simulations as plain data, simulated anywhere
(``RepairableRBD.shards``, ``run_shard``, ``python -m
repyability.rbd.shards``), whose partials, put together in any order, are
the run's result to the last bit (``availability_from_chunks``);
``availability(..., shard_map=...)`` and ``cost`` do it all through any
map; and ``n_jobs``' workers send back their blocks' totals rather than
every simulation."""

import io
import json
import pickle
import random
import subprocess
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pytest
import surpyval as surv

from repyability import NodeState, RepairableRBD, SimulationChunk, run_shard
from repyability.rbd import _streams
from repyability.rbd.repairable_rbd import (
    _WORKER,
    _simulate_block,
    _start_worker,
    _Tally,
)
from repyability.rbd.shards import main
from repyability.tests.test_performance_equivalence import binomial_first
from repyability.tests.test_simulation_chunks import identical, plant
from repyability.tests.test_simulation_engines import systems_of_every_kind

E, W = surv.Exponential.from_params, surv.Weibull.from_params


@pytest.fixture(scope="module")
def pool():
    with ProcessPoolExecutor(2) as executor:
        yield executor


def spans(shards):
    """Each shard's range of simulations."""
    return [(json.loads(s)["start"], json.loads(s)["stop"]) for s in shards]


def engine_here(engine):
    if engine == "numba":
        pytest.importorskip("numba")
    return engine


@pytest.mark.parametrize("engine", ["python", "numba"])
@pytest.mark.parametrize(
    "settings",
    [
        {},
        {"antithetic": True},
        {"broken_nodes": ["B"]},
        {"curve_points": 25},
    ],
    ids=["plain", "antithetic", "held broken", "curve on a grid"],
)
def test_shards_put_together_in_any_order_are_the_run(engine, settings):
    rbd = plant()
    run = dict(seed=8, engine=engine_here(engine), **settings)
    whole = rbd.availability(100.0, mc_samples=2500, **run)
    shards = rbd.shards(100.0, 2500, size=600, **run)
    partials = [run_shard(shard) for shard in shards]
    random.Random(1).shuffle(partials)
    identical(whole, rbd.availability_from_chunks(partials, mc_samples=2500))


def test_shards_from_a_state_and_with_capacities_are_the_run():
    rbd = plant(capacity={"A": 5.0, "B": 5.0, "C": 10.0})
    state = {
        "A": NodeState(alive=True, age=12.0),
        "C": NodeState(alive=False, down_for=0.5),
    }
    run = dict(seed=2, state=state, demand=6.0)
    whole = rbd.availability(80.0, mc_samples=2500, **run)
    partials = [run_shard(s) for s in rbd.shards(80.0, 2500, size=1, **run)]
    assert len(partials) > 1
    identical(
        whole,
        rbd.availability_from_chunks(partials[::-1], mc_samples=2500),
    )


def test_shards_start_at_whole_blocks_of_draws():
    rbd = plant()
    # Each shard holds a whole number of the run's widest block of draws,
    # at least the size asked for; the last holds what is left.
    for size, antithetic in [(1, False), (700, False), (1, True)]:
        shards = rbd.shards(
            100.0, 5000, seed=1, size=size, antithetic=antithetic
        )
        ranges = spans(shards)
        step = ranges[0][1]
        assert step >= size and step % (2 if antithetic else 1) == 0
        assert [a for a, _ in ranges] == list(range(0, 5000, step))
        assert [b for _, b in ranges][:-1] == list(range(step, 5000, step))
        assert ranges[-1][1] == 5000
    # The widest block here: a size of 1 gives one block to a shard.
    one = spans(rbd.shards(100.0, 5000, seed=1, size=1))[0][1]
    assert spans(rbd.shards(100.0, 5000, seed=1, size=one + 1))[0][1] == (
        2 * one
    )
    # The default: as many as make 1024 or more.
    assert spans(rbd.shards(100.0, 5000, seed=1))[0][1] >= 1024


@pytest.mark.parametrize("engine", ["python", "numba"])
def test_availability_and_cost_through_a_map(engine, pool):
    engine = engine_here(engine)
    rbd = plant()
    run = dict(mc_samples=1500, seed=4, engine=engine)
    whole = rbd.availability(100.0, **run)
    for shard_map in (map, pool.map):
        identical(
            whole,
            rbd.availability(
                100.0, shard_map=shard_map, shard_size=400, **run
            ),
        )
    # A map that gives the partials back in another order.
    identical(
        whole,
        rbd.availability(
            100.0, shard_map=lambda f, xs: [f(x) for x in xs][::-1], **run
        ),
    )
    identical(
        rbd.cost(100.0, **run),
        rbd.cost(100.0, shard_map=pool.map, shard_size=400, **run),
    )
    # A run to a tolerance maps a round of shards at a time.
    run = dict(run, mc_samples=500, tolerance=0.002)
    whole = rbd.availability(100.0, **run)
    assert whole.n_simulations > 500
    identical(
        whole, rbd.availability(100.0, shard_map=pool.map, shard_size=1, **run)
    )


@pytest.mark.parametrize("name", sorted(systems_of_every_kind()))
def test_every_kind_of_system_runs_as_shards(name):
    rbd = systems_of_every_kind()[name]
    window = 20000.0 if name == "instrument air" else 150.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        route = rbd.analysis_routes()["shards"]
        if route.route == "refused":
            with pytest.raises((NotImplementedError, ValueError)) as error:
                rbd.availability(window, mc_samples=40, seed=5, shard_map=map)
            assert str(error.value) == route.reason
            return
        identical(
            rbd.availability(window, mc_samples=40, seed=5),
            rbd.availability(
                window, mc_samples=40, seed=5, shard_map=map, shard_size=16
            ),
        )


def test_a_missing_or_doubled_partial_is_refused():
    rbd = plant()
    shards = rbd.shards(100.0, 3000, seed=3, size=1000)
    partials = [run_shard(s) for s in shards]
    assert len(partials) == 3
    with pytest.raises(ValueError, match="some are missing"):
        rbd.availability_from_chunks(partials[:-1], mc_samples=3000)
    with pytest.raises(ValueError, match="some are missing"):
        rbd.availability_from_chunks(
            [partials[0], partials[2]], mc_samples=3000
        )
    with pytest.raises(ValueError, match="overlap"):
        rbd.availability_from_chunks(partials + partials[:1])
    # Without mc_samples, the simulations they hold.
    held = 3000 - spans(shards)[0][1]
    assert rbd.availability_from_chunks(partials[1:]).n_simulations == held


def test_a_map_that_gives_back_other_partials_is_refused():
    rbd = plant()
    other = plant()
    elsewhere = other.shards(100.0, 1000, seed=99, size=1)
    wrong = {
        "one short": lambda f, xs: [f(x) for x in xs][:-1],
        "one twice": lambda f, xs: [f(xs[0])] + [f(x) for x in xs],
        "another run's": lambda f, xs: [f(x) for x in elsewhere],
    }
    for shard_map in wrong.values():
        with pytest.raises(ValueError, match="shard_map gave back"):
            rbd.availability(
                100.0, mc_samples=1000, seed=3, shard_map=shard_map
            )


def test_a_partial_is_an_npz_read_without_pickle(tmp_path):
    rbd = plant()
    (shard,) = rbd.shards(100.0, 300, seed=6)
    partial = run_shard(shard)
    with np.load(io.BytesIO(partial), allow_pickle=False) as npz:
        assert {"header", "uptimes", "cost_samples", "changes"} <= set(
            npz.files
        )
        assert npz["uptimes"].shape == (300,)
    chunk = SimulationChunk.from_npz(partial)
    assert chunk.ranges == [(0, 300)]
    assert SimulationChunk.from_npz(chunk.to_npz()) == chunk
    for bad in (b"not a chunk", b""):
        with pytest.raises(ValueError, match="not a saved SimulationChunk"):
            SimulationChunk.from_npz(bad)
    # Nothing is unpickled: an array of objects is refused.
    buffer = io.BytesIO()
    np.savez(buffer, header=np.array([{"kind": 1}], dtype=object))
    with pytest.raises(ValueError, match="not a saved SimulationChunk"):
        SimulationChunk.from_npz(buffer.getvalue())


def test_the_command_line_runs_a_shard(tmp_path):
    rbd = plant()
    first, second = rbd.shards(100.0, 1200, seed=7, size=600)
    command = [sys.executable, "-m", "repyability.rbd.shards"]
    # From standard input to standard output.
    piped = subprocess.run(
        command, input=first, capture_output=True, check=True
    ).stdout
    # From a file to a file.
    (tmp_path / "shard.json").write_bytes(second)
    subprocess.run(
        command + [str(tmp_path / "shard.json"), str(tmp_path / "out.npz")],
        check=True,
    )
    written = (tmp_path / "out.npz").read_bytes()
    identical(
        rbd.availability(100.0, mc_samples=1200, seed=7),
        rbd.availability_from_chunks([written, piped], mc_samples=1200),
    )
    assert main(["--help"]) == 0
    assert main(["a", "b", "c"]) == 2


def test_what_cannot_be_sharded_is_refused():
    unstreamable = RepairableRBD(
        [("s", "a"), ("a", "t")],
        {
            "a": {
                "reliability": binomial_first([40, 2]),
                "repairability": E([0.5]),
            }
        },
    )
    # A model that would load back as another class: a worker would
    # simulate another system.
    message = "saves as surpyval's Parametric"
    with pytest.raises(NotImplementedError, match=message):
        unstreamable.shards(100.0, 100, seed=1)
    with pytest.raises(NotImplementedError, match=message):
        unstreamable.availability(100.0, mc_samples=100, shard_map=map)
    # Only shards, and those of this version.
    with pytest.raises(ValueError, match="not a shard"):
        run_shard(b'{"kind": "SimulationChunk"}')
    with pytest.raises(ValueError, match="not a shard"):
        run_shard(b"\x00 not JSON")
    (shard,) = plant().shards(100.0, 10, seed=1)
    older = json.loads(shard)
    older["repyability_version"] = "0.1"
    with pytest.raises(ValueError, match="made by RePyability 0.1"):
        run_shard(json.dumps(older))


def test_shard_arguments_are_checked():
    rbd = plant()
    run = dict(mc_samples=100, seed=1)
    for options, match in [
        ({"shard_map": "map"}, "shard_map must be a map"),
        ({"shard_map": map, "n_jobs": 2}, "leave out n_jobs"),
        ({"shard_size": 100}, "give shard_map too"),
        ({"shard_map": map, "shard_size": 0}, "at least 1"),
        ({"shard_map": map, "shard_size": 2.5}, "at least 1"),
        ({"shard_map": map, "shard_size": True}, "at least 1"),
        ({"shard_map": map, "engine": "fast"}, "engine must be one of"),
        ({"shard_map": map, "demand": 1.0}, "no node has one"),
    ]:
        with pytest.raises(ValueError, match=match):
            rbd.availability(100.0, **run, **options)
    for options, match in [
        ({"size": 0}, "at least 1"),
        ({"antithetic": True, "mc_samples": 101}, "even"),
        ({"engine": "fast"}, "engine must be one of"),
        ({"method": "x"}, "method"),
    ]:
        with pytest.raises(ValueError, match=match):
            rbd.shards(100.0, **{**run, **options})
    # What the compiled engine does not simulate is refused here, whether
    # or not numba is installed here.
    state = {"A": NodeState(alive=True, age=3.0)}
    with pytest.raises(NotImplementedError, match="started from a state"):
        rbd.shards(100.0, 10, seed=1, engine="numba", state=state)


def test_workers_send_back_their_blocks_totals():
    rbd = plant()
    entropy = _streams.entropy_of(3)
    args = (100.0, set(), set(), "p", None, entropy, False, None, {})
    _start_worker(pickle.dumps((rbd, args, None, False)))
    try:
        block = _simulate_block((40, 90))
    finally:
        _WORKER.clear()
    # The block's totals, compact: no simulation left to fold in, the
    # changes in arrays.
    assert isinstance(block, _Tally) and block.n == 50
    assert block._rows == [] and block.changes == []
    assert len(block.uptimes) == 50
    # And the same totals as the block's simulations added one by one.
    serial = rbd._run(
        100.0,
        set(),
        set(),
        "p",
        50,
        False,
        None,
        first=40,
        entropy=entropy,
        engine="python",
    )
    serial.merge(_Tally(list(rbd.components), rbd.costs, 100.0))
    assert block.to_dict() == serial.to_dict()


def test_parallel_workers_keep_each_simulations_replacements():
    rbd = plant()
    run = dict(entropy=_streams.entropy_of(2), replacements=True)
    serial = rbd._run(100.0, set(), set(), "p", 600, False, None, **run)
    parallel = rbd._run(
        100.0, set(), set(), "p", 600, False, None, jobs=2, **run
    )
    assert parallel.replacements == serial.replacements
    assert len(parallel.replacements) == 600
