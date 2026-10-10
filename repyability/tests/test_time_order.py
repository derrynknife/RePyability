"""A run's changes put in time order (#201, #208): each simulation's changes
come as a sorted run, one run after another, and ``_time_order`` nets
the +1 and -1 changes of state by time, groups the capacity's changes by
time and merges its totals. Each is checked here against its definition,
written out plainly: a dict of the changes at each time, their exact sums
and ``np.cumsum`` of the totals in time order."""

import math
from collections import defaultdict

import numpy as np
import pytest

from repyability.rbd import _time_order
from repyability.rbd._exact import ExactSum


def bits(array: np.ndarray) -> np.ndarray:
    """An array's bits, so that ``-0.0`` and ``0.0`` differ."""
    array = np.asarray(array)
    return array.view(np.uint64) if array.dtype == np.float64 else array


def same(a, b) -> None:
    a, b = np.asarray(a), np.asarray(b)
    assert a.dtype == b.dtype and a.shape == b.shape
    assert np.array_equal(bits(a), bits(b))


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


def by_time(times, values) -> dict:
    """Each time's values, the times in order."""
    grouped = defaultdict(list)
    for t, v in zip(times.tolist(), values.tolist()):
        grouped[t + 0.0].append(v)  # -0.0 is 0.0's time
    return dict(sorted(grouped.items()))


CASES = {
    "none": np.zeros(0),
    "at zero, as -0.0 too": np.array([0.0, -0.0, 0.0, 3.0]),
    "at the end and beyond": np.array([1.0, 100.0, 100.0, 130.0]),
}


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("start", [0, 40])
def test_the_working_count_after_each_time(seed, start):
    rng = np.random.default_rng(seed)
    times, _, deltas = runs(rng)
    expected = by_time(np.r_[times, 0.0, 100.0], np.r_[deltas, 0, 0])
    count = start + np.cumsum([sum(d) for d in expected.values()])
    time, working = _time_order._working_over_time(times, deltas, 100.0, start)
    same(time, np.array(list(expected), dtype=float))
    same(working, count.astype(float))


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_working_count_at_the_edges(case):
    times = CASES[case]
    deltas = np.resize(np.array([-1, 1], dtype=np.int64), times.size)
    expected = by_time(np.r_[times, 0.0, 100.0], np.r_[deltas, 0, 0])
    time, working = _time_order._working_over_time(times, deltas, 100.0, 7)
    same(time, np.array(list(expected), dtype=float))
    same(working, 7.0 + np.cumsum([sum(d) for d in expected.values()]))


def exact_total(values: list) -> float:
    """A time's total change: a lone change itself (``-0.0`` stays), else
    the exact sum, rounded once."""
    return values[0] if len(values) == 1 else float(ExactSum(values))


@pytest.mark.parametrize("seed", range(4))
def test_the_capacity_changes_grouped_by_time(seed):
    rng = np.random.default_rng(seed)
    times, steps, _ = runs(rng)
    expected = by_time(times, steps)
    at, values, starts = _time_order._by_time(times, steps)
    same(at, np.array(list(expected), dtype=float))
    stops = np.append(starts[1:], values.size)
    for a, b, want in zip(starts, stops, expected.values()):
        assert sorted(values[a:b].tolist()) == sorted(want)
    totals = _time_order._group_totals(values, starts)
    same(totals, np.array([exact_total(v) for v in expected.values()]))
    assert all(
        math.isclose(t, math.fsum(v), abs_tol=0.0)
        for t, v in zip(totals.tolist(), expected.values())
    )


@pytest.mark.parametrize("seed", range(4))
def test_the_net_changes_by_time(seed):
    rng = np.random.default_rng(seed)
    times, _, deltas = runs(rng)
    expected = by_time(times, deltas)
    at, net = _time_order._net_by_time(times, deltas)
    same(at, np.array(list(expected), dtype=float))
    same(net, np.array([sum(d) for d in expected.values()], dtype=np.int64))


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
def test_the_capacity_totals_merged(case):
    at, change, free_at, counts = merge_cases()[case]
    every = sorted(set(at.tolist()) | set(free_at.tolist()) | {0.0, 10.0})
    step = dict.fromkeys(every, 0.0)
    step.update(zip(at.tolist(), change.tolist()))
    freed = dict.fromkeys(every, 0)
    freed.update(zip(free_at.tolist(), counts.tolist()))
    times, totals, unlimited = _time_order._capacity_totals(
        at, change, free_at, counts, 10.0
    )
    same(times, np.array(every, dtype=float))
    same(totals, np.cumsum(np.array(list(step.values()), dtype=float)))
    same(unlimited, np.cumsum(list(freed.values())) > 0)
