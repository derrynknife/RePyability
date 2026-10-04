"""A run's changes put in time order by the compiled radix sort and merge
(#201, ``repyability.rbd._time_order``): to the last bit what numpy's
stable ``argsort``, ``np.unique``, ``searchsorted`` and ``np.cumsum`` give,
and the same results for whole runs and their chunks, whichever path builds
them. Without numba every run takes numpy's path, which the rest of the
suite tests."""

import json

import numpy as np
import pytest

from repyability.rbd import repairable_rbd
from repyability.tests.test_simulation_engines import capacity_rbds

numba = pytest.importorskip("numba")

from repyability.rbd import _time_order  # noqa: E402


def bits(array: np.ndarray) -> np.ndarray:
    """An array's bits, so that ``-0.0`` and ``0.0`` differ."""
    array = np.asarray(array)
    return array.view(np.uint64) if array.dtype == np.float64 else array


def same(a, b) -> None:
    a, b = np.asarray(a), np.asarray(b)
    assert a.dtype == b.dtype and a.shape == b.shape
    assert np.array_equal(bits(a), bits(b))


@pytest.fixture
def numpy_path(monkeypatch):
    """Run what follows on numpy's path, whatever the size."""

    def run(function, *args, **kwargs):
        with monkeypatch.context() as patch:
            patch.setattr(repairable_rbd, "_COMPILED_ORDER", np.inf)
            return function(*args, **kwargs)

    return run


def runs(rng, count=300, longest=60):
    """Changes as a run records them: a sorted run per simulation, one
    after another, with times the runs share (ties), 0, the smallest
    float, +inf and a time beyond the end among them."""
    shared = np.array([0.0, 5e-324, 1e-300, 0.5, 1.0, 50.0, 100.0, np.inf])
    times = []
    for _ in range(count):
        size = int(rng.integers(0, longest))
        mine = np.where(
            rng.random(size) < 0.3,
            rng.choice(shared, size),
            rng.random(size) * 120.0,
        )
        times.append(np.sort(mine))
    times = np.concatenate(times)
    steps = np.where(
        rng.random(times.size) < 0.1,
        rng.choice([0.0, -0.0], times.size),
        rng.normal(size=times.size),
    )
    deltas = rng.choice([-1, 1], times.size).astype(np.int64)
    return times, steps, deltas


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("block", [1 << 16, 97])
def test_the_radix_sort_is_a_stable_argsort(seed, block, monkeypatch):
    # In one block, and in many (each sorted on a thread of its own).
    monkeypatch.setattr(_time_order, "_BLOCK", block)
    times, steps, deltas = runs(np.random.default_rng(seed))
    order = np.argsort(times, kind="stable")
    for values in (steps, deltas):
        kept = times.copy(), values.copy()
        sorted_times, sorted_values = _time_order.sort_by_time(times, values)
        same(sorted_times, times[order])
        same(sorted_values, values[order])
        same(times, kept[0])  # left as they were, without reuse
        same(values, kept[1])
        reused = _time_order.sort_by_time(
            times.copy(), values.copy(), reuse=True
        )
        same(reused[0], times[order])
        same(reused[1], values[order])


@pytest.mark.parametrize("seed", range(4))
def test_grouped_by_time_as_numpy_groups_them(seed, numpy_path):
    times, steps, deltas = runs(np.random.default_rng(seed))
    for values in (steps, deltas):
        compiled = _time_order.by_time(times.copy(), values.copy())
        expected = numpy_path(repairable_rbd._by_time, times, values)
        for got, want in zip(compiled, expected):
            same(got, want)
        at, ordered, starts = compiled
        same(
            repairable_rbd._group_totals(ordered, starts),
            numpy_path(repairable_rbd._group_totals, ordered, starts),
        )


@pytest.mark.parametrize(
    "times",
    [
        np.zeros(0),
        np.array([3.0]),
        np.array([1.0, 2.0, 2.0, 7.0]),  # in order already
        np.array([2.0, 2.0, 2.0]),  # one time: no digit to sort by
    ],
)
def test_short_and_ordered_changes(times, numpy_path):
    values = np.arange(times.size, dtype=float)
    compiled = _time_order.by_time(times, values)
    for got, want in zip(
        compiled, numpy_path(repairable_rbd._by_time, times, values)
    ):
        same(got, want)


@pytest.mark.parametrize("odd", [-1.0, -0.0, np.nan])
def test_times_whose_bits_do_not_sort_are_left_to_numpy(odd):
    times = np.array([3.0, 1.0, odd, 2.0])
    values = np.arange(4.0)
    assert _time_order.sort_by_time(times, values) is None
    assert _time_order.by_time(times, values) is None


def merge_cases():
    rng = np.random.default_rng(7)
    cases = {
        "empty": (np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0, int)),
        # A first total of -0.0 stays -0.0 in np.cumsum; adding the 0.0 of a
        # time with no change makes it 0.0.
        "negative zero first": (
            np.array([0.0, 4.0]),
            np.array([-0.0, -0.0]),
            np.array([2.0]),
            np.array([1]),
        ),
        "only unlimited": (
            np.zeros(0),
            np.zeros(0),
            np.array([0.0, 3.0, 10.0]),
            np.array([2, -1, -1]),
        ),
        "at the ends and beyond": (
            np.array([0.0, 2.5, 10.0, 12.0]),
            np.array([3.0, -1.5, 0.25, 7.0]),
            np.array([2.5, 11.0]),
            np.array([1, -1]),
        ),
    }
    at = np.unique(rng.random(4000) * 10.0)
    free_at = np.unique(
        np.r_[rng.choice(at, 300), rng.random(300) * 10.0, 0.0]
    )
    cases["random"] = (
        at,
        np.where(rng.random(at.size) < 0.2, -0.0, rng.normal(size=at.size)),
        free_at,
        rng.choice([-1, 1], free_at.size).astype(np.int64),
    )
    return cases


@pytest.mark.parametrize("case", sorted(merge_cases()))
def test_the_merge_is_numpy_s_unique_searchsorted_and_cumsum(case, numpy_path):
    at, change, free_at, counts = merge_cases()[case]
    compiled = _time_order.capacity_totals(at, change, free_at, counts, 10.0)
    expected = numpy_path(
        repairable_rbd._capacity_totals, at, change, free_at, counts, 10.0
    )
    for got, want in zip(compiled, expected):
        same(got, want)


def test_an_undefined_time_is_left_to_numpy():
    at = np.array([1.0, np.nan])
    assert (
        _time_order.capacity_totals(
            at, np.ones(2), np.zeros(0), np.zeros(0, int), 10.0
        )
        is None
    )


def bit_identical(a, b, path="result"):
    """Two results the same to the last bit (``-0.0`` apart from ``0.0``)."""
    import dataclasses

    if dataclasses.is_dataclass(a):
        assert type(a) is type(b), path
        for f in dataclasses.fields(a):
            bit_identical(getattr(a, f.name), getattr(b, f.name), path)
    elif isinstance(a, dict):
        assert list(a) == list(b), path
        for key in a:
            bit_identical(a[key], b[key], f"{path}[{key!r}]")
    elif isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        same(a, b)
    elif isinstance(a, float):
        assert np.float64(a).view(np.uint64) == np.float64(b).view(
            np.uint64
        ), path
    else:
        assert a == b, path


@pytest.mark.parametrize("name", sorted(capacity_rbds()))
def test_runs_and_chunks_are_the_same_on_either_path(
    name, monkeypatch, numpy_path
):
    rbd, kwargs = capacity_rbds()[name]
    expected = numpy_path(
        rbd.availability, 200.0, mc_samples=24, seed=5, **kwargs
    )
    chunk = numpy_path(rbd.simulate_chunk, 200.0, 0, 24, seed=5, **kwargs)
    monkeypatch.setattr(repairable_rbd, "_COMPILED_ORDER", 0)
    bit_identical(
        rbd.availability(200.0, mc_samples=24, seed=5, **kwargs), expected
    )
    compiled = rbd.simulate_chunk(200.0, 0, 24, seed=5, **kwargs)
    assert json.dumps(compiled.to_dict()) == json.dumps(chunk.to_dict())
