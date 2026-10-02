"""The availability curve on a grid of times (#153): ``availability(...,
curve_points=G)`` counts the simulations' changes of state in ``G`` steps,
and its values at the grid's times are the full curve's there, exactly;
everything else in the result is the same as without it."""

import dataclasses
import math
from collections.abc import Mapping

import numpy as np
import pytest
import surpyval as surv

from repyability import NodeState, RepairableRBD, SimulationChunk

E, W = surv.Exponential.from_params, surv.Weibull.from_params

EDGES = [("s", "A"), ("s", "B"), ("A", "C"), ("B", "C"), ("C", "t")]


def plant(**options):
    return RepairableRBD(
        EDGES,
        {
            "A": {
                "reliability": W([10.0, 1.5]),
                "repairability": E([1.0]),
                "repair_cost": 5.0,
            },
            "B": {"reliability": W([10.0, 1.5]), "repairability": E([1.0])},
            "C": {"reliability": W([50.0, 1.5]), "repairability": E([0.5])},
        },
        downtime_cost_rate=2.0,
        **options,
    )


def on_the_grid(full, grid):
    """The full curve's values at the grid's times."""
    index = np.searchsorted(full.timeline, grid.timeline, side="right") - 1
    return full.availability[index]


def identical(a, b, path="result"):
    """Whether two values are the same to the last bit, field by field."""
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


def the_rest_is_the_same(full, grid):
    for f in dataclasses.fields(full):
        if f.name not in ("timeline", "availability"):
            identical(getattr(full, f.name), getattr(grid, f.name), f.name)
    return True


@pytest.mark.parametrize("engine", ["python", "numba"])
@pytest.mark.parametrize(
    "settings",
    [{}, {"antithetic": True}, {"broken_nodes": ["B"]}],
    ids=["plain", "antithetic", "held broken"],
)
def test_the_grid_holds_the_full_curve_at_its_times(engine, settings):
    if engine == "numba":
        pytest.importorskip("numba")
    rbd = plant()
    run = dict(mc_samples=1000, seed=4, engine=engine, **settings)
    full = rbd.availability(100.0, **run)
    grid = rbd.availability(100.0, curve_points=40, **run)
    np.testing.assert_array_equal(grid.timeline, 100.0 * np.arange(41) / 40)
    np.testing.assert_array_equal(grid.availability, on_the_grid(full, grid))
    assert len(full.timeline) > 10 * len(grid.timeline)
    assert the_rest_is_the_same(full, grid)


def test_a_change_at_0_and_on_a_grid_time():
    # From a state with the system down at 0, and a component whose failure
    # falls exactly on a grid time: the grid's value there is the full
    # curve's from that time on.
    rbd = RepairableRBD(
        [("s", "a"), ("a", "t")],
        {
            "a": {
                "reliability": surv.ExactEventTime.from_params(25.0),
                "repairability": surv.ExactEventTime.from_params(5.0),
            }
        },
    )
    state = {"a": NodeState(alive=False, down_for=0.0)}
    full = rbd.availability(100.0, mc_samples=4, seed=1, state=state)
    grid = rbd.availability(
        100.0, mc_samples=4, seed=1, state=state, curve_points=20
    )
    np.testing.assert_array_equal(grid.availability, on_the_grid(full, grid))
    # Down at 0, up from 5 to 30, down from 30 (a failure on the grid time).
    assert grid.availability[0] == 0.0 and grid.availability[1] == 1.0
    assert grid.availability[6] == 0.0


def test_parallel_and_chunked_runs_count_the_same():
    rbd = plant()
    whole = rbd.availability(100.0, mc_samples=600, seed=7, curve_points=50)
    parallel = rbd.availability(
        100.0,
        mc_samples=600,
        seed=7,
        curve_points=50,
        engine="python",
        n_jobs=2,
    )
    np.testing.assert_array_equal(parallel.availability, whole.availability)
    chunks = [
        rbd.simulate_chunk(100.0, a, b, seed=7, curve_points=50)
        for a, b in [(0, 250), (250, 600)]
    ]
    data = chunks[0].to_dict()
    # A chunk carries the grid's counts, not every change.
    assert data["totals"]["changes"] == []
    assert len(data["totals"]["binned"]) == 51
    sent = [SimulationChunk.from_json(c.to_json()) for c in reversed(chunks)]
    merged = rbd.availability_from_chunks(sent)
    np.testing.assert_array_equal(merged.availability, whole.availability)
    np.testing.assert_array_equal(merged.timeline, whole.timeline)
    assert merged.system_uptime == whole.system_uptime
    # Chunks counted on different grids, or one with every change, are of
    # different runs.
    other = rbd.simulate_chunk(100.0, 250, 600, seed=7, curve_points=20)
    full = rbd.simulate_chunk(100.0, 250, 600, seed=7)
    for stranger in (other, full):
        with pytest.raises(ValueError, match="different runs"):
            SimulationChunk.merge([chunks[0], stranger])


def test_costs_keep_no_curve():
    rbd = plant()
    costs = rbd.cost(100.0, mc_samples=300, seed=2)
    run = rbd.availability(100.0, mc_samples=300, seed=2)
    np.testing.assert_array_equal(costs.samples, run.cost.samples)
    assert costs.by_category == run.cost.by_category


@pytest.mark.parametrize("bad", [0, -3, 2.5, True, "10"])
def test_curve_points_is_checked(bad):
    rbd = plant()
    with pytest.raises(ValueError, match="curve_points"):
        rbd.availability(10.0, mc_samples=10, seed=1, curve_points=bad)
    with pytest.raises(ValueError, match="curve_points"):
        rbd.simulate_chunk(10.0, 0, 10, seed=1, curve_points=bad)
