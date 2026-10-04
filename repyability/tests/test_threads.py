"""Simulations called from several threads at once (#216).

A diagram kept in memory and simulated from several threads (as a tool or
an API server does) gives each call what it gives alone: the same for the
same seed, and no error. The event loop keeps a run's state on the diagram,
and draws that cannot be streamed come from numpy's global RNG, so the runs
take turns (``repyability.utils.wrappers.SIMULATIONS``).
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import surpyval as surv

from repyability import RepairableRBD
from repyability.tests.catalogue import (
    nonrepairable_kinds,
    repairable_kinds,
)
from repyability.tests.test_simulation_engines import identical


def together(job, n: int = 6, threads: int = 4, rounds: int = 2) -> None:
    """Each of ``n`` seeded calls of ``job``, run alone and then all at
    once on ``threads`` threads, gives the same."""
    alone = [job(i) for i in range(n)]
    for _ in range(rounds):
        with ThreadPoolExecutor(threads) as pool:
            for i, result in enumerate(pool.map(job, range(n))):
                identical(result, alone[i])


def test_the_issue_s_diagram():
    unit = {
        "reliability": surv.Weibull.from_params([500, 2.0]),
        "repairability": surv.LogNormal.from_params([2.5, 0.5]),
    }
    rbd = RepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        {"a": unit, "b": unit},
    )
    together(
        lambda i: rbd.availability(
            t_simulation=1000.0,
            mc_samples=100,
            seed=i,
            control_variate=False,
            engine="python",
        ).uptimes
    )


@pytest.mark.parametrize(
    "name",
    [
        "unstreamable",
        "unstreamable maintenance",
        "nested_maintained",
        "one repair crew",
        "standby group",
        "common cause, tested",
    ],
)
def test_availability_and_cost_from_threads(name):
    rbd = repairable_kinds()[name]
    options = dict(t_simulation=300.0, mc_samples=40, control_variate=False)

    def job(i):
        out = {"availability": rbd.availability(seed=i, **options)}
        if rbd.has_costs:
            out["cost"] = rbd.cost(seed=i, **options)
        return out

    together(job)


def test_different_diagrams_at_once():
    # Their draws that cannot be streamed share numpy's global RNG.
    first = repairable_kinds()["unstreamable"]
    second = repairable_kinds()["unstreamable maintenance"]

    def job(i):
        rbd = first if i % 2 else second
        return rbd.availability(
            300.0, mc_samples=40, seed=i, control_variate=False
        ).uptimes

    together(job)


def test_timelines_from_threads():
    rbd = repairable_kinds()["maintained"]
    together(lambda i: rbd.simulate_timelines(200.0, mc_samples=20, seed=i))


@pytest.mark.parametrize("name", ["plain", "unreplayable", "nested"])
def test_lifetimes_from_threads(name):
    rbd = nonrepairable_kinds()[name]
    together(lambda i: rbd.random(500, seed=i))


def test_the_global_rng_is_left_as_it_was():
    rbd = repairable_kinds()["unstreamable"]
    np.random.seed(3)
    expected = np.random.random(4)
    np.random.seed(3)
    with ThreadPoolExecutor(4) as pool:
        list(
            pool.map(
                lambda i: rbd.availability(
                    300.0, mc_samples=20, seed=i, control_variate=False
                ),
                range(6),
            )
        )
    np.testing.assert_array_equal(np.random.random(4), expected)


def test_shards_on_threads():
    # A run's shards each take their turn where they run: on threads, they
    # must not wait for the run that sent them.
    rbd = repairable_kinds()["nested_maintained"]
    options = dict(mc_samples=40, seed=4, control_variate=False)
    alone = rbd.availability(300.0, **options)
    with ThreadPoolExecutor(3) as pool:
        sharded = rbd.availability(
            300.0, shard_map=pool.map, shard_size=10, **options
        )
    np.testing.assert_array_equal(sharded.uptimes, alone.uptimes)
