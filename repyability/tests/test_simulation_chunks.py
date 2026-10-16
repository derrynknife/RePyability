"""Chunks of a simulation run (#114): simulations a to b of a run, run
anywhere, saved, and merged into the run's result, the same to the last bit
as the run's however it is cut, as its totals are kept exactly (#151); and
blocks of a NonRepairableRBD's lifetimes."""

import dataclasses
import math
from collections.abc import Mapping

import numpy as np
import pytest
import surpyval as surv

from repyability import NonRepairableRBD, RepairableRBD, SimulationChunk
from repyability.rbd import _montecarlo as montecarlo
from repyability.rbd._exact import ExactSum, expansion
from repyability.rbd._tally import _Tally
from repyability.rbd._time_order import _group_totals

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


def identical(a, b, path="result"):
    """Whether two results are the same to the last bit: every field, the
    totals too, which are kept exactly however the run is split (#151)."""
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
        assert np.array_equal(a, b, equal_nan=True), path
    elif isinstance(a, float):
        assert a == b or (math.isnan(a) and math.isnan(b)), (path, a, b)
    else:
        assert a == b, path
    return True


def same_result(a, b):
    """Whether two results hold the same simulations, to the last bit."""
    return identical(a, b)


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
        rbd.availability_from_chunks(alone, allow_gaps=True).uptimes,
        whole.uptimes[300:],
    )
    assert same_result(
        rbd.availability_from_chunks(parallel, allow_gaps=True),
        rbd.availability_from_chunks(alone, allow_gaps=True),
    )


def test_chunks_with_gaps_and_out_of_order():
    rbd = plant()
    a = rbd.simulate_chunk(100.0, 0, 100, seed=1)
    b = rbd.simulate_chunk(100.0, 200, 300, seed=1)
    c = rbd.simulate_chunk(100.0, 100, 200, seed=1)
    gappy = SimulationChunk.merge([b, a])
    assert gappy.ranges == [(0, 100), (200, 300)]
    assert gappy.n_simulations == 200
    # A missing chunk is not taken for a smaller run (#176) ...
    with pytest.raises(ValueError, match="between or before them are missing"):
        rbd.availability_from_chunks(gappy)
    with pytest.raises(ValueError, match="0 to 99, 200 to 299"):
        rbd.availability_from_chunks([b, a])
    with pytest.raises(ValueError, match="allow_gaps"):
        rbd.availability_from_chunks(b)
    # ... unless asked for: the result of the simulations held.
    result = rbd.availability_from_chunks(gappy, allow_gaps=True)
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
    # Their pairs are the run's: the plain run's interval is theirs.
    plain = rbd.availability(
        100.0, mc_samples=400, seed=3, antithetic=True, control_variate=False
    )
    assert montecarlo.standard_error(
        merged.uptimes / 100.0, True
    ) == pytest.approx(
        plain.mean_availability_interval().standard_error, rel=1e-15
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
    assert identical(merged, whole)


def test_chunks_count_the_capacity_on_the_grid():
    # With curve_points the capacity's changes are counted in the grid's
    # bins, exactly (#190): chunks, through JSON too, merge into the run.
    rbd = RepairableRBD(
        EDGES,
        {
            node: {"reliability": E([0.1]), "repairability": E([1.0])}
            for node in "ABC"
        },
        capacity={"A": 60.0, "B": 60.0, "C": 100.0},
    )
    run = dict(seed=2, demand=100.0, curve_points=25)
    whole = rbd.availability(50.0, mc_samples=300, **run)
    assert len(whole.capacity_timeline) == 26
    chunks = [
        SimulationChunk.from_json(
            rbd.simulate_chunk(50.0, a, b, **run).to_json()
        )
        for a, b in [(0, 120), (120, 300)]
    ]
    merged = rbd.availability_from_chunks(chunks)
    assert np.array_equal(merged.capacity_timeline, whole.capacity_timeline)
    assert np.array_equal(merged.capacity, whole.capacity)
    assert identical(merged, whole)


@pytest.mark.parametrize("engine", ["python", "numba"])
def test_a_capacity_chunk_is_saved_alike_by_either_engine(engine):
    # Each time's change of the capacity curve is saved as its exact total
    # (#155): the same chunk from either engine, and through JSON.
    if engine == "numba":
        pytest.importorskip("numba")
    rbd = RepairableRBD(
        EDGES,
        {
            node: {"reliability": E([0.1]), "repairability": E([1.0])}
            for node in "ABC"
        },
        capacity={"A": {60.0: 0.5, 30.0: 0.5}},
    )
    run = dict(seed=2, demand=100.0)
    python = rbd.simulate_chunk(50.0, 0, 200, engine="python", **run)
    chunk = rbd.simulate_chunk(50.0, 0, 200, engine=engine, **run)
    saved = python.to_dict()["totals"]
    assert saved["unlimited_changes"]  # unlimited while B and C work
    assert chunk.to_dict()["totals"] == saved
    back = SimulationChunk.from_json(chunk.to_json())
    assert back.to_dict()["totals"] == saved


def test_capacity_changes_at_one_time_add_up_exactly():
    tally = _Tally(["a"], {}, 10.0)
    for time, step in [
        (1.0, 1e100),
        (2.0, 5.0),
        (1.0, 1.0),
        (3.0, 2.0),
        (1.0, -1e100),
        (3.0, -2.0),
    ]:
        tally.capacity_times.append(time)
        tally.capacity_steps.append(step)
    for time, step in [(3.0, 1), (4.0, -1), (4.0, 1)]:
        tally.unlimited_times.append(time)
        tally.unlimited_steps.append(step)
    at, steps, starts, free_at, counts = tally.capacity_changes()
    assert at.tolist() == [1.0, 2.0, 3.0]
    assert _group_totals(steps, starts).tolist() == [1.0, 5.0, 0.0]
    assert (free_at.tolist(), counts.tolist()) == ([3.0, 4.0], [1, 0])
    # A time whose changes cancel out is kept, as no change.
    saved = tally.to_dict()
    assert saved["capacity_changes"] == [(1.0, [1.0]), (2.0, [5.0]), (3.0, [])]
    assert saved["unlimited_changes"] == [(3.0, 1), (4.0, 0)]
    back = _Tally.from_dict(saved)
    assert back.to_dict() == saved
    back.merge(_Tally.from_dict(saved))
    merged = back.to_dict()
    assert merged["capacity_changes"] == [
        (1.0, [2.0]),
        (2.0, [10.0]),
        (3.0, []),
    ]
    assert merged["unlimited_changes"] == [(3.0, 2), (4.0, 0)]


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(start=0, stop=10, seed=None), "needs the run's seed"),
        (dict(start=10, stop=10, seed=1), "start < stop"),
        (dict(start=-1, stop=10, seed=1), "start < stop"),
        (dict(start=0.5, stop=10, seed=1), "whole number"),
        (dict(start=0, stop=True, seed=1), "whole number"),
        (dict(start=0, stop=10, seed=1, demand=5.0), "capacities"),
        (
            dict(start=0, stop=10, seed=1, method="x"),
            r"'p' \(or 'paths'\) or 'c'",
        ),
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


# -- exact totals (#151) ------------------------------------------------------


def test_exact_sums_are_the_same_however_they_are_split():
    rng = np.random.default_rng(0)
    values = rng.standard_normal(5000) * 10.0 ** rng.integers(-12, 13, 5000)
    whole = math.fsum(values)
    canonical = ExactSum(values).partials
    for pieces in (2, 7, 64):
        cuts = np.sort(
            rng.choice(np.arange(1, len(values)), pieces - 1, False)
        )
        parts = [ExactSum(part) for part in np.split(values, cuts)]
        order = rng.permutation(len(parts))
        total = ExactSum()
        for i in order:
            total += parts[i]
        assert float(total) == whole
        assert total.partials == canonical
    # One value at a time, in any order.
    one = ExactSum()
    for value in rng.permutation(values):
        one += float(value)
    assert float(one) == whole and one.partials == canonical
    # What rounding in order loses, it keeps.
    assert float(ExactSum([1e100, 1.0, -1e100])) == 1.0
    assert expansion([1e100, 1.0, -1e100]) == [1.0]
    assert expansion([1.0, 2.0**-60]) == [1.0, 2.0**-60]
    assert float(ExactSum()) == 0.0 and ExactSum().partials == []
    assert float(ExactSum([1.0, math.inf])) == math.inf
    assert math.isnan(float(ExactSum([math.inf, -math.inf])))


@pytest.mark.parametrize("engine", ["python", "numba"])
@pytest.mark.parametrize(
    "settings",
    [{}, {"antithetic": True}, {"broken_nodes": ["B"]}],
    ids=["plain", "antithetic", "held broken"],
)
def test_a_run_cut_anywhere_is_the_run(engine, settings):
    # Uneven pieces, merged in any order, give the run to the last bit:
    # its curve, its per-simulation values and its totals.
    if engine == "numba":
        pytest.importorskip("numba")
    rbd = plant()
    whole = rbd.availability(
        100.0, mc_samples=600, seed=5, engine=engine, **settings
    )
    rng = np.random.default_rng(1)
    for cuts in ([0, 600], [0, 2, 600], [0, 2, 132, 400, 598, 600]):
        chunks = [
            rbd.simulate_chunk(100.0, a, b, seed=5, engine=engine, **settings)
            for a, b in zip(cuts, cuts[1:])
        ]
        chunks = [chunks[i] for i in rng.permutation(len(chunks))]
        assert identical(rbd.availability_from_chunks(chunks), whole)


def test_the_totals_are_rounded_once():
    rbd = plant()
    result = rbd.availability(100.0, mc_samples=500, seed=3)
    assert result.system_uptime == math.fsum(result.uptimes)
    assert result.system_downtime == math.fsum(100.0 - result.uptimes)
    costs = rbd.cost(100.0, mc_samples=500, seed=3, control_variate=False)
    assert math.fsum(costs.samples) == pytest.approx(
        500 * sum(costs.by_category.values()), rel=1e-14
    )


# -- #236: chunks of a plain run --------------------------------------------


def modular():
    """A plant with a module the exact methods do not take (imperfect
    repair of a unit that wears): its runs' means are conditional by
    default."""
    worn = {
        "reliability": W([10.0, 2.0]),
        "repairability": E([1.0]),
        "repair": {"model": "kijima1", "q": 0.5},
        "repair_cost": 5.0,
    }
    plain = {"reliability": E([0.05]), "repairability": E([1.0])}
    return RepairableRBD(
        [("s", "a"), ("a", "b"), ("b", "t")],
        {"a": worn, "b": plain},
        downtime_cost_rate=2.0,
    )


@pytest.mark.parametrize(
    "rbd, options",
    [
        (plant(), {"control_variate": False}),
        (modular(), {"control_variate": False}),
        (modular(), {"conditional": False}),
    ],
)
def test_chunks_merge_into_a_plain_run(rbd, options):
    whole = rbd.availability(30.0, mc_samples=40, seed=4, **options)
    chunks = [
        rbd.simulate_chunk(30.0, a, b, seed=4) for a, b in ((0, 25), (25, 40))
    ]
    assert same_result(rbd.availability_from_chunks(chunks, **options), whole)
    # Or kept with the chunks, through their saved form too.
    kept = [
        SimulationChunk.from_dict(
            rbd.simulate_chunk(30.0, a, b, seed=4, **options).to_dict()
        )
        for a, b in ((0, 25), (25, 40))
    ]
    assert same_result(rbd.availability_from_chunks(kept), whole)
    # By default, the means availability takes by default.
    default = rbd.availability(30.0, mc_samples=40, seed=4)
    assert same_result(rbd.availability_from_chunks(chunks), default)
    assert whole.mean_availability_interval().method == "simulated"


def test_chunks_with_other_means_do_not_merge():
    rbd = plant()
    first = rbd.simulate_chunk(30.0, 0, 10, seed=4, control_variate=False)
    rest = rbd.simulate_chunk(30.0, 10, 20, seed=4)
    with pytest.raises(ValueError, match="different runs"):
        rbd.availability_from_chunks([first, rest])
    # A chunk made with the defaults is saved as before them.
    assert "control_variate" not in rest.settings


@pytest.mark.parametrize("name", ["control_variate", "conditional"])
@pytest.mark.parametrize("value", [True, 1, "no"])
def test_a_chunk_holds_no_twin_or_modules_alone(name, value):
    rbd = plant()
    with pytest.raises(ValueError, match=f"{name} must be None or False"):
        rbd.simulate_chunk(30.0, 0, 10, seed=4, **{name: value})
    chunk = rbd.simulate_chunk(30.0, 0, 10, seed=4)
    with pytest.raises(ValueError, match=f"{name} must be None or False"):
        rbd.availability_from_chunks([chunk], **{name: value})
