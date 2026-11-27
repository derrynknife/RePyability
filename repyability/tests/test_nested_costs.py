"""A nested ``RepairableRBD``'s costs counted in the diagram it is a node
of (see ``rbd/_nested_costs.py``): exactly, in the long run, simulated on
either engine, discounted, in a valuation of runs and against a log.

A nested diagram runs on its own, so the checks are against its own
methods: a system whose only block is a nested diagram costs what the
nested diagram costs, and one with other blocks costs the hand sum of
the parts, by category, each labelled with where it falls. Simulated,
a nested diagram's run alone under its place's streams draws what it
draws inside the system's run, so the sums hold simulation by
simulation, to the bit.
"""

import math

import numpy as np
import pytest
import surpyval as surv

from repyability import NodeState, RepairableRBD
from repyability.rbd import _compiled, _event_loop, _timeline_runs
from repyability.tests.test_performance_equivalence import binomial_first
from repyability.timelines import Timeline

E = surv.Exponential.from_params
W = surv.Weibull.from_params
T = 200.0
RATE = math.log(1.07) / 1000.0  # a discount rate per unit time

needs_numba = pytest.mark.skipif(
    not _compiled.available(), reason="numba is not installed"
)


def inner(priced: bool = True) -> RepairableRBD:
    """A pump pair, each pump repaired (one also maintained, off line),
    with its own downtime cost, crews' salaries and a purchase price;
    ``priced=False``, the same diagram with no costs."""

    def cost(**amounts):
        return amounts if priced else {}

    return RepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        {
            "a": {
                "reliability": W([50.0, 1.5]),
                "repairability": E([0.5]),
                **cost(
                    repair_cost=surv.Normal.from_params([100.0, 10.0]),
                    downtime_cost=2.0,
                    acquisition_cost=1000.0,
                ),
            },
            "b": {
                "reliability": E([0.02]),
                "repairability": E([1.0]),
                "preventive": {
                    "interval": 30.0,
                    "duration": E([2.0]),
                    **({"cost": 5.0} if priced else {}),
                },
                **cost(repair_cost=40.0),
            },
        },
        downtime_cost_rate=10.0 if priced else 0.0,
        repair_crews=2,
        crew_cost_rate=0.25 if priced else 0.0,
    )


def valve():
    return {
        "reliability": E([0.01]),
        "repairability": E([0.2]),
        "repair_cost": 50.0,
        "downtime_cost": 1.0,
    }


def alone(sub: RepairableRBD) -> RepairableRBD:
    """A system whose only block is ``sub``."""
    return RepairableRBD([("s", "sub"), ("sub", "t")], {"sub": sub})


def plant(sub: RepairableRBD) -> RepairableRBD:
    """``sub`` in series with a valve, with a system downtime cost."""
    return RepairableRBD(
        [("s", "sub"), ("sub", "v"), ("v", "t")],
        {"sub": sub, "v": valve()},
        downtime_cost_rate=20.0,
    )


def as_parent(category: str) -> str:
    return "component_downtime" if category == "system_downtime" else category


def nested_alone(sub, mc_samples, seed, engine="python", **options):
    """``sub``'s own cost run, under the streams of its place in the
    system (node ``"sub"``)."""
    return _event_loop._run(
        sub,
        T,
        set(),
        set(),
        "p",
        mc_samples,
        False,
        seed,
        False,
        engine=engine,
        curve_points=1,
        prefix=("sub",),
        **options,
    )


# -- exact ------------------------------------------------------------------


def test_a_system_of_one_nested_diagram_costs_what_it_costs():
    sub = inner()
    system = alone(sub)
    assert system.has_costs
    got, own = system.expected_cost(T), sub.expected_cost(T)
    assert got.mean == own.mean
    for category, value in own.by_category.items():
        if category == "system_downtime":
            continue
        want = value + (
            own.by_category["system_downtime"]
            if category == "component_downtime"
            else 0.0
        )
        assert got.by_category[category] == pytest.approx(want, rel=1e-15)
    assert got.by_category["system_downtime"] == 0.0
    # Each component labelled with its path, and what the nested diagram
    # pays itself (its downtime and crews) with its node's.
    assert set(got.by_component) == {("sub", "a"), ("sub", "b"), "sub"}
    assert got.by_component[("sub", "a")] == own.by_component["a"]
    assert got.by_component["sub"] == pytest.approx(
        own.by_category["system_downtime"] + own.by_category["crews"],
        rel=1e-15,
    )
    assert system.expected_cost_rate() == sub.expected_cost_rate()
    assert system.acquisition_cost == sub.acquisition_cost == 1000.0
    assert system.total_cost(T) == pytest.approx(sub.total_cost(T))


def test_a_system_with_other_blocks_costs_the_hand_sum_of_its_parts():
    sub = inner()
    system = plant(sub)
    # The same system with the nested diagram unpriced: its own costs.
    own = plant(inner(priced=False))
    got = system.expected_cost(T)
    mine, theirs = own.expected_cost(T), sub.expected_cost(T)
    assert got.mean == pytest.approx(mine.mean + theirs.mean, rel=1e-14)
    for category in got.by_category:
        want = mine.by_category[category]
        for key, value in theirs.by_category.items():
            if as_parent(key) == category:
                want = want + value
        assert got.by_category[category] == pytest.approx(
            want, rel=1e-14, abs=1e-12
        ), category
    assert got.by_component["v"] == mine.by_component["v"]
    assert system.expected_cost_rate() == pytest.approx(
        own.expected_cost_rate() + sub.expected_cost_rate(), rel=1e-14
    )


def test_the_npv_is_the_discounted_sum():
    sub = inner()
    system, own = plant(sub), plant(inner(priced=False))
    got = system.expected_npv(T, discount_rate=RATE, value_rate=3.0)
    mine = own.expected_npv(T, discount_rate=RATE, value_rate=3.0)
    theirs = sub.expected_npv(T, discount_rate=RATE, value_rate=0.0)
    assert got.cost == pytest.approx(mine.cost + theirs.cost, rel=1e-13)
    assert got.value == mine.value
    assert got.npv == pytest.approx(mine.npv - theirs.cost, rel=1e-13)
    # The discounted expected cost is the same sum.
    assert system.expected_cost(T, discount_rate=RATE).total == pytest.approx(
        own.expected_cost(T, discount_rate=RATE).total
        + sub.expected_cost(T, discount_rate=RATE).total,
        rel=1e-12,
    )
    labels = {row["component"] for row in got.cash_flows}
    assert {("sub", "a"), ("sub", "b"), "sub", "v"} <= labels
    assert got.cost_by_component[("sub", "b")] == pytest.approx(
        theirs.cost_by_component["b"], rel=1e-13
    )


def test_a_nested_node_held_working_costs_nothing_and_broken_its_downtime():
    sub = inner()
    system = plant(sub)
    own = plant(inner(priced=False))
    for held in ("working_nodes", "broken_nodes"):
        got = system.expected_cost(T, **{held: ["sub"]})
        mine = own.expected_cost(T, **{held: ["sub"]})
        extra = 10.0 * T if held == "broken_nodes" else 0.0
        assert got.mean == pytest.approx(mine.mean + extra, rel=1e-14)
        rate = system.expected_cost_rate(**{held: ["sub"]})
        assert rate == pytest.approx(
            own.expected_cost_rate(**{held: ["sub"]}) + extra / T, rel=1e-14
        )
        simulated = system.cost(
            T, mc_samples=50, seed=2, control_variate=False, **{held: ["sub"]}
        )
        plain = own.cost(
            T, mc_samples=50, seed=2, control_variate=False, **{held: ["sub"]}
        )
        assert np.array_equal(simulated.samples, plain.samples + extra)


# -- simulated ----------------------------------------------------------


def test_the_nested_diagram_draws_alone_what_it_draws_in_the_system():
    sub = inner()
    system = plant(sub)
    engines = ["python"] + (["numba"] if _compiled.available() else [])
    for engine in engines:
        runs = system.simulate_timelines(
            300.0, mc_samples=200, seed=5, engine=engine
        )
        own = _timeline_runs.simulate(
            sub,
            300.0,
            200,
            5,
            None,
            None,
            False,
            "python",
            None,
            0,
            prefix=("sub",),
        )
        inside, outside = runs.components["sub"]._data, own.system._data
        assert inside.times.size > 100
        assert np.array_equal(inside.times, outside.times)
        assert np.array_equal(inside.offsets, outside.offsets)
        assert np.array_equal(inside.planned, outside.planned)


def test_each_simulation_costs_the_sum_of_its_parts_to_the_bit():
    sub = inner()
    system, own = alone(sub), alone(inner(priced=False))
    got = system.cost(T, mc_samples=300, seed=7, control_variate=False)
    theirs = nested_alone(sub, 300, 7)
    assert np.array_equal(got.samples, np.asarray(theirs.cost_samples))
    system, own = plant(sub), plant(inner(priced=False))
    got = system.cost(T, mc_samples=300, seed=7, control_variate=False)
    mine = own.cost(T, mc_samples=300, seed=7, control_variate=False)
    assert np.array_equal(
        got.samples, mine.samples + np.asarray(theirs.cost_samples)
    )
    assert set(got.by_component) == {("sub", "a"), ("sub", "b"), "sub", "v"}
    assert got.by_component["v"] == mine.by_component["v"]


@needs_numba
def test_both_engines_agree_to_the_bit():
    system = plant(inner())
    options = dict(mc_samples=300, seed=9, control_variate=False)
    python = system.cost(T, engine="python", **options)
    numba = system.cost(T, engine="numba", **options)
    assert np.array_equal(python.samples, numba.samples)
    assert python.by_category == numba.by_category
    assert python.by_component == numba.by_component
    a = system.availability(T, engine="python", **options).cost
    b = system.availability(T, engine="numba", **options).cost
    assert np.array_equal(a.samples, b.samples)


def test_runs_in_parts_and_in_processes_are_the_run():
    system = plant(inner())
    whole = system.cost(T, mc_samples=600, seed=3, control_variate=False)
    parallel = system.cost(
        T, mc_samples=600, seed=3, control_variate=False, n_jobs=2
    )
    assert np.array_equal(whole.samples, parallel.samples)
    assert whole.by_component == parallel.by_component
    chunks = [
        system.simulate_chunk(T, first, first + 300, seed=3)
        for first in (0, 300)
    ]
    merged = system.availability_from_chunks(chunks, control_variate=False)
    assert np.array_equal(merged.cost.samples, whole.samples)


def test_the_simulation_agrees_with_the_exact_value():
    system = plant(inner())
    exact = system.expected_cost(T)
    got = system.cost(T, mc_samples=20_000, seed=11, control_variate=False)
    se = got.samples.std(ddof=1) / math.sqrt(got.samples.size)
    assert abs(got.samples.mean() - exact.mean) < 4.0 * se
    for label, value in got.by_component.items():
        assert value == pytest.approx(exact.by_component[label], rel=0.03)
    # By default the mean is the exact one.
    assert system.cost(T, mc_samples=200, seed=1).mean == pytest.approx(
        exact.mean, rel=1e-12
    )


def test_a_run_from_a_state_starts_the_nested_diagram_from_its_own():
    sub = inner()
    state = {"sub": {"a": NodeState(age=40.0), "b": NodeState(alive=False)}}
    system = alone(sub)
    exact = system.expected_cost(T, state=state)
    # The same parts, added up in another order (a last bit apart).
    own = sub.expected_cost(T, state=state["sub"])
    assert exact.by_category == own.by_category
    assert exact.mean == pytest.approx(own.mean, rel=1e-14)
    got = system.cost(
        T, mc_samples=200, seed=4, control_variate=False, state=state
    )
    theirs = nested_alone(sub, 200, 4, states=state["sub"])
    assert np.array_equal(got.samples, np.asarray(theirs.cost_samples))


def test_nested_inside_nested_is_labelled_with_its_path():
    deep = alone(inner())
    system = RepairableRBD(
        [("s", "top"), ("top", "v"), ("v", "t")],
        {"top": deep, "v": valve()},
    )
    got = system.expected_cost(T)
    assert ("top", "sub", "a") in got.by_component
    assert ("top", "sub") in got.by_component
    # (The valve's curve is solved on a grid to about 1e-7, a grid of its
    # own in a diagram of its own.)
    assert got.mean == pytest.approx(
        deep.expected_cost(T).mean
        + RepairableRBD([("s", "v"), ("v", "t")], {"v": valve()})
        .expected_cost(T)
        .mean,
        rel=1e-9,
    )
    simulated = system.cost(T, mc_samples=2000, seed=6, control_variate=False)
    assert set(simulated.by_component) == set(got.by_component)


# -- valuations of runs and logs ---------------------------------------------


def test_the_npv_distribution_counts_the_nested_costs():
    system = plant(inner())
    exact = system.expected_npv(T, discount_rate=RATE, value_rate=3.0)
    runs = system.npv_distribution(
        T, discount_rate=RATE, value_rate=3.0, mc_samples=4000, seed=8
    )
    se = runs.cost_samples.std(ddof=1) / math.sqrt(runs.n_simulations)
    assert abs(runs.cost - exact.cost) < 4.0 * se
    assert set(runs.cost_by_component) == set(exact.cost_by_component)
    assert runs.cost_by_component["sub"] == pytest.approx(
        exact.cost_by_component["sub"], rel=0.01
    )


def test_plan_vs_actual_carries_the_nested_labels():
    system = plant(inner())
    logs = {
        "sub": Timeline([50.0, 52.0], end=100.0),
        "v": Timeline([], end=100.0),
    }
    result = system.plan_vs_actual(
        logs,
        100.0,
        T,
        discount_rate=RATE,
        value_rate=3.0,
        state={"sub": {"a": NodeState(), "b": NodeState()}},
    )
    actual = {
        (r["category"], r["component"]) for r in result.actual_cash_flows
    }
    # What the nested node's log gives: its downtime and its crews.
    assert ("component_downtime", "sub") in actual
    assert ("crews", "sub") in actual
    left_out = {(n["category"], n["component"]) for n in result.not_compared}
    assert ("repair", ("sub", "a")) in left_out
    assert not {c for _, c in actual if isinstance(c, tuple)}
    planned = {r["component"] for r in result.rest_cash_flows}
    assert {("sub", "a"), ("sub", "b")} <= planned
    # Its downtime over the log's 2 time units down, at 10 a unit.
    down = sum(
        -r["undiscounted"]
        for r in result.actual_cash_flows
        if (r["category"], r["component"]) == ("component_downtime", "sub")
    )
    assert down == pytest.approx(20.0)


# -- what refuses -------------------------------------------------------------


def test_a_nested_downtime_cost_in_a_shutdown_scope_refuses():
    system = RepairableRBD(
        [("s", "sub"), ("sub", "v"), ("v", "t")],
        {"sub": inner(), "v": valve()},
        shutdowns={"overhaul": {"times": [50.0, 150.0], "duration": 5.0}},
    )
    for call in (
        lambda: system.expected_cost(T),
        lambda: system.expected_cost_rate(),
        lambda: system.cost(T, mc_samples=10, seed=1),
    ):
        with pytest.raises(NotImplementedError, match="shutdown 'overhaul'"):
            call()
    routes = system.analysis_routes()
    assert routes["expected_cost"].route == "refused"
    assert routes["cost"].route == "refused"


def test_a_nested_diagram_whose_draws_are_not_streamed_refuses_to_simulate():
    sub = RepairableRBD(
        [("s", "a"), ("a", "t")],
        {
            "a": {
                "reliability": binomial_first([40, 2]),
                "repairability": E([0.5]),
                "repair_cost": 5.0,
            }
        },
    )
    system = plant(sub)
    with pytest.raises(NotImplementedError, match="do not come from a"):
        system.cost(T, mc_samples=10, seed=1)
    assert system.analysis_routes()["cost"].route == "refused"
