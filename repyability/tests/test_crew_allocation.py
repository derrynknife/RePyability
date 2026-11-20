"""Redundancy allocation with limited repair crews
(``allocate_redundancy``, ``_crew_allocation``): a copy is one more job
for the crews, so each design is worked out whole, from the crews' Markov
chain of its components and their copies.

Here: a design's values against the same design drawn out as a diagram
(every copy a node of its own, behind a junction); the search against
every design in its box; the limit of a crew for every job against the
search for independent components; the bound the search prunes by; and
what refuses."""

import itertools
import math

import numpy as np
import pytest
import surpyval as surv

from repyability import (
    BetaFactor,
    CCFGroup,
    PerfectReliability,
    RepairableRBD,
)
from repyability.rbd import _crew_allocation, _crew_chain
from repyability.rbd import routes as r

E = surv.Exponential.from_params
W = surv.Weibull.from_params
HORIZON = 300.0
#: Three components in series: failure rate, repair rate, price, options.
SPECS = {
    "a": (0.01, 0.1, 400.0, {}),
    "b": (0.02, 0.2, 250.0, {"downtime_cost": 0.3}),
    "c": (0.005, 0.05, 900.0, {"priority": 1}),
}


def unit(rate, repair, price, **options):
    return {
        "reliability": E([rate]),
        "repairability": E([repair]),
        "acquisition_cost": price,
        "repair_cost": 5.0,
        **options,
    }


def spec(node):
    rate, repair, price, options = SPECS[node]
    return unit(rate, repair, price, **options)


def line(crews=2, **options):
    """``a``, ``b`` and ``c`` in series, sharing the crews."""
    return RepairableRBD(
        [("s", "a"), ("a", "b"), ("b", "c"), ("c", "t")],
        {node: spec(node) for node in SPECS},
        repair_crews=crews,
        downtime_cost_rate=20.0,
        **options,
    )


def drawn(counts, crews=2, **options):
    """The line with ``counts`` copies of each node drawn out: every copy
    a node of its own, a node's copies in parallel behind a junction."""
    components, edges, before = {}, [], "s"
    for node, n in zip(SPECS, counts):
        junction = ("after", node)
        components[junction] = PerfectReliability
        for copy in range(n):
            name = (node, copy)
            components[name] = spec(node)
            edges += [(before, name), (name, junction)]
        before = junction
    edges.append((before, "t"))
    return RepairableRBD(
        edges,
        components,
        repair_crews=crews,
        downtime_cost_rate=20.0,
        **options,
    )


BOX = {"a": 2, "b": 3, "c": 2}


def every_design(**options):
    """Each design of the box and its total cost, drawn out."""
    return {
        counts: drawn(counts, **options).total_cost(HORIZON)
        for counts in itertools.product(
            *(range(1, BOX[node] + 1) for node in SPECS)
        )
    }


# -- a design's values --------------------------------------------------------


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"crew_cost_rate": 0.4},
        {"call_out": {"when_waiting": 1, "fee": 30.0, "rate": 2.0}},
    ],
    ids=["plain", "salaries", "a call-out crew"],
)
def test_a_design_is_the_diagram_with_its_copies_drawn_out(options):
    rbd = line(**options)
    evaluate = _crew_allocation.Evaluator(rbd, list(SPECS))
    for counts in [(1, 1, 1), (2, 1, 1), (1, 3, 2), (2, 2, 2)]:
        design, diagram = evaluate(counts), drawn(counts, **options)
        assert design.unavailability == pytest.approx(
            diagram.mean_unavailability(), rel=1e-12
        )
        assert design.cost_rate == pytest.approx(
            diagram.expected_cost_rate(), rel=1e-12
        )
        assert design.acquisition == diagram.acquisition_cost
    # One of each is the diagram itself.
    assert evaluate((1, 1, 1)).cost_rate == pytest.approx(
        rbd.expected_cost_rate(), rel=1e-13
    )


def test_alike_copies_are_counted_not_told_apart():
    # The chain of copies counted against the chain of every copy.
    rates, repairs, priorities = [0.02, 0.05], [0.1, 0.3], [0.0, 1.0]
    counted = _crew_chain.solve(
        "ab", rates, repairs, priorities, 2, 2, (), [3, 2]
    )
    each = _crew_chain.solve(
        range(5),
        [0.02] * 3 + [0.05] * 2,
        [0.1] * 3 + [0.3] * 2,
        [0.0] * 3 + [1.0] * 2,
        2,
        2,
    )
    assert len(counted.states) < len(each.states)
    assert _crew_chain.state_count(
        priorities, [False, False], 2, 2, [3, 2]
    ) == len(counted.states)
    for k, copies in enumerate(([0, 1, 2], [3, 4])):
        # Those up, on average, and whether any is.
        assert float(
            counted.probabilities @ counted.working[:, k]
        ) == pytest.approx(
            float(each.probabilities @ each.up[:, copies].sum(axis=1)),
            rel=1e-12,
        )
        assert float(
            counted.probabilities @ counted.up[:, k]
        ) == pytest.approx(
            float(each.probabilities @ each.up[:, copies].any(axis=1)),
            rel=1e-12,
        )
    # The call-out crew's call-outs and time engaged.
    for vector in ("calls", "engaged"):
        assert float(
            counted.probabilities @ getattr(counted, vector)
        ) == pytest.approx(
            float(each.probabilities @ getattr(each, vector)), rel=1e-12
        )


# -- the search ---------------------------------------------------------------


@pytest.mark.parametrize(
    "options",
    [{}, {"call_out": {"when_waiting": 2, "fee": 30.0, "rate": 2.0}}],
    ids=["plain", "a call-out crew"],
)
def test_the_search_finds_the_best_design_of_the_box(options):
    designs = every_design(**options)
    best = min(designs, key=designs.get)
    result = line(**options).allocate_redundancy(HORIZON, max_units=BOX)
    assert tuple(result.units[node] for node in SPECS) == best
    assert result.total_cost == pytest.approx(designs[best], rel=1e-12)
    assert result.availability == pytest.approx(
        drawn(best, **options).mean_availability(), rel=1e-12
    )
    # Not at a corner of the box: the search chose it.
    assert best not in ((1, 1, 1), tuple(BOX.values()))
    assert result.method == "exact"
    assert "repair crews' Markov chain" in result.basis
    # One step at a time does no better.
    greedy = line(**options).allocate_redundancy(
        HORIZON, max_units=BOX, method="greedy"
    )
    assert greedy.total_cost >= result.total_cost - 1e-9
    assert greedy.method == "greedy"


def test_the_search_keeps_to_an_availability():
    designs = every_design()
    target = 0.97
    within = {
        counts: cost
        for counts, cost in designs.items()
        if drawn(counts).mean_availability() >= target
    }
    assert 0 < len(within) < len(designs)
    best = min(within, key=within.get)
    result = line().allocate_redundancy(
        HORIZON, max_units=BOX, min_availability=target
    )
    assert tuple(result.units[node] for node in SPECS) == best
    assert result.availability >= target
    with pytest.raises(ValueError, match="even with a crew for every job"):
        line().allocate_redundancy(
            HORIZON, max_units=1, min_availability=0.999999
        )
    # Within the least any crews could give, but not with one crew.
    one = line(crews=1)
    floor = _crew_allocation.Unlimited(one, list(SPECS))((2, 2, 2))
    least = min(
        _crew_allocation.Evaluator(one, list(SPECS))(counts).unavailability
        for counts in itertools.product((1, 2), repeat=3)
    )
    assert floor < least
    with pytest.raises(ValueError, match="waits too long"):
        one.allocate_redundancy(
            HORIZON,
            max_units=2,
            min_availability=1.0 - 0.5 * (floor + least),
        )


def test_value_earned_is_downtime_that_costs():
    rbd = line()
    valued = rbd.allocate_redundancy(
        HORIZON, max_units=BOX, objective="npv", value_rate=15.0
    )
    dearer = RepairableRBD(
        [("s", "a"), ("a", "b"), ("b", "c"), ("c", "t")],
        {node: spec(node) for node in SPECS},
        repair_crews=2,
        downtime_cost_rate=35.0,
    ).allocate_redundancy(HORIZON, max_units=BOX)
    assert valued.units == dearer.units
    assert valued.npv == pytest.approx(
        15.0 * HORIZON * valued.availability - valued.total_cost, rel=1e-12
    )
    assert "repair crews' Markov chain" in valued.basis


def test_a_crew_for_every_job_is_the_search_for_independent_components():
    # Nine crews for at most nine jobs: none waits, and the chain's
    # designs are those of independent components.
    free = line(crews=None).allocate_redundancy(HORIZON, max_units=3)
    counts, design = _crew_allocation.search(
        line(crews=9), list(SPECS), [3, 3, 3], None, "exact", HORIZON, 0.0
    )
    assert dict(zip(SPECS, counts)) == free.units
    assert design.acquisition + HORIZON * design.cost_rate == pytest.approx(
        free.total_cost, rel=1e-10
    )
    assert 1.0 - design.unavailability == pytest.approx(
        free.availability, rel=1e-12
    )
    # With as many crews as components the diagram itself takes the
    # search for independent components: copies are then given crews too.
    assert (
        line(crews=3).allocate_redundancy(HORIZON, max_units=3).units
        == free.units
    )


def test_no_design_is_down_less_than_with_a_crew_for_every_job():
    rng = np.random.default_rng(89)
    for _ in range(12):
        rates = rng.uniform(0.005, 0.05, 3)
        repairs = rng.uniform(0.05, 0.5, 3)
        k = int(rng.integers(1, 3))
        rbd = RepairableRBD(
            [("s", n) for n in "abc"] + [(n, "t") for n in "abc"],
            {
                n: unit(rates[i], repairs[i], 10.0, priority=int(i == 1))
                for i, n in enumerate("abc")
            },
            k={"t": k},
            repair_crews=int(rng.integers(1, 3)),
            downtime_cost_rate=1.0,
        )
        evaluate = _crew_allocation.Evaluator(rbd, list("abc"))
        unlimited = _crew_allocation.Unlimited(rbd, list("abc"))
        for counts in itertools.product((1, 2, 3), repeat=3):
            floor = unlimited(counts)
            assert floor <= evaluate(counts).unavailability * (1 + 1e-12)
            # More copies of anything lower the floor.
            for i in range(3):
                more = counts[:i] + (counts[i] + 1,) + counts[i + 1 :]
                assert unlimited(more) <= floor * (1 + 1e-12)
        assert unlimited((math.inf,) * 3) == 0.0


def test_a_copy_can_make_the_system_worse():
    # Why the search for independent components' bounds do not hold: a
    # copy of the spare path's unit takes the one crew from the unit
    # the system needs.
    rbd = RepairableRBD(
        [("s", "a"), ("a", "b"), ("a", "z"), ("b", "t"), ("z", "t")],
        {
            "a": unit(0.02, 0.05, 10.0),
            "b": unit(0.2, 0.05, 10.0, priority=1),
            "z": unit(0.001, 1.0, 10.0),
        },
        repair_crews=1,
        downtime_cost_rate=1.0,
    )
    evaluate = _crew_allocation.Evaluator(rbd, ["b"])
    assert evaluate((2,)).unavailability > evaluate((1,)).unavailability


# -- what refuses -------------------------------------------------------------


def test_a_design_too_large_for_the_chain_is_refused_by_name():
    nodes = "abcdefg"
    rbd = RepairableRBD(
        [("s", nodes[0])] + list(zip(nodes, nodes[1:])) + [(nodes[-1], "t")],
        {
            n: unit(0.01 + 0.001 * i, 0.1, 1.0, priority=i)
            for i, n in enumerate(nodes)
        },
        repair_crews=1,
        downtime_cost_rate=1000.0,
    )
    with pytest.raises(NotImplementedError, match="needs the design") as e:
        rbd.allocate_redundancy(HORIZON, max_units=4)
    assert "15,000 states" in str(e.value) and "compare()" in str(e.value)
    # Bounded, it is worked out.
    assert rbd.allocate_redundancy(HORIZON, nodes=["a"], max_units=2).units


def test_a_copy_that_costs_nothing_to_buy_needs_a_cap():
    free = RepairableRBD(
        [("s", "a"), ("a", "b"), ("b", "t")],
        {
            "a": {"reliability": E([0.01]), "repairability": E([0.1])},
            "b": unit(0.01, 0.1, 10.0),
        },
        repair_crews=1,
        downtime_cost_rate=5.0,
    )
    with pytest.raises(ValueError, match="costs nothing to buy"):
        free.allocate_redundancy(HORIZON, nodes=["a", "b"])
    assert free.allocate_redundancy(
        HORIZON, nodes=["a", "b"], max_units={"a": 2}
    ).units["a"] in (1, 2)


def test_what_the_chain_of_copies_does_not_take_refuses():
    with pytest.raises(NotImplementedError, match="whole trains"):
        line().allocate_redundancy(HORIZON, trains={"ab": ["a", "b"]})
    grouped = RepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "c"), ("b", "c"), ("c", "t")],
        {n: unit(0.01, 0.1, 10.0) for n in "abc"},
        repair_crews=1,
        ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.1, basis="rate"))],
        downtime_cost_rate=5.0,
    )
    with pytest.raises(NotImplementedError, match="common-cause") as e:
        grouped.allocate_redundancy(HORIZON, max_units=2)
    assert "compare()" in str(e.value)
    assert grouped.analysis_routes()["allocate_redundancy"].route == r.REFUSED
    weibull = RepairableRBD(
        [("s", "a"), ("a", "b"), ("b", "t")],
        {
            "a": {
                "reliability": W([100.0, 2.0]),
                "repairability": E([0.1]),
                "acquisition_cost": 10.0,
            },
            "b": unit(0.01, 0.1, 10.0),
        },
        repair_crews=1,
        downtime_cost_rate=5.0,
    )
    with pytest.raises(NotImplementedError, match="exponential lives"):
        weibull.allocate_redundancy(HORIZON, max_units=2)
    assert weibull.analysis_routes()["allocate_redundancy"].route == r.REFUSED


def test_the_report_and_the_other_allocations():
    rbd = line()
    report = rbd.analysis_routes()
    assert report["allocate_redundancy"].route == r.EXACT
    assert "crews' Markov chain" in report["allocate_redundancy"].reason
    # Targets of availability move one component at a time: they still
    # assume a crew each, and say what does not.
    for method, call in (
        ("availability_allocation", lambda: rbd.availability_allocation(0.9)),
        ("mttf_mttr_allocation", lambda: rbd.mttf_mttr_allocation(0.9)),
    ):
        assert report[method].route == r.REFUSED
        with pytest.raises(NotImplementedError, match="allocate_redundancy"):
            call()
