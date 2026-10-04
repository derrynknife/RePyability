"""Safety rough edges (#237): an inspection offset within a billionth of
the interval is no offset, and a diagram's fault tree keeps a common-cause
member its logic makes irrelevant. (Unavailability over time to its own
precision is in ``test_unavailability.py``.)"""

import warnings

import numpy as np
import pytest
import surpyval as surv

from repyability import BetaFactor, CCFGroup, FaultTree, RepairableRBD

E = surv.Exponential.from_params
YEAR = 8760.0


# -- an offset near 0 ---------------------------------------------------------


def inspected(offset):
    return RepairableRBD(
        [("s", "a"), ("a", "t")],
        {
            "a": {
                "reliability": E([1e-5]),
                "repairability": E([1 / 24]),
                "inspection": {
                    "interval": YEAR,
                    "duration": surv.ExactEventTime.from_params([8.0]),
                    "cost": 100.0,
                    "offset": offset,
                },
            }
        },
    )


@pytest.mark.parametrize(
    "offset, tests",
    [(0.0, 2), (1e-6, 2), (YEAR * 1e-9, 2), (YEAR * 2e-9, 3), (1.0, 3)],
    ids=["0", "1e-6", "a billionth", "two billionths", "1"],
)
def test_an_offset_within_a_billionth_of_the_interval_is_none(offset, tests):
    # Over three years, tests at 1 and 2 years; a positive offset adds one
    # at the start, unless it is within a billionth of the interval.
    rbd = inspected(offset)
    assert rbd.expected_events(3 * YEAR).node_inspections["a"] == tests
    cost = rbd.expected_cost(3 * YEAR)
    assert cost.by_category["inspection"] == pytest.approx(100.0 * tests)
    if tests == 2:
        none = inspected(0.0)
        times = np.array([4.0, 12.0, YEAR + 4.0])
        assert np.array_equal(
            rbd.point_unavailability(times), none.point_unavailability(times)
        )


# -- a member the tree's logic makes irrelevant -------------------------------


def test_a_member_the_logic_makes_irrelevant_stays_in_the_tree():
    # TOP = a OR (a AND b): b cannot change it, but the group's shared
    # cause strikes both.
    group = CCFGroup(["a", "b"], BetaFactor(0.1))
    tree = FaultTree(
        {"TOP": ("or", ["a", "G1"]), "G1": ("and", ["a", "b"])},
        {"a": 0.01, "b": 0.01},
        ccf_groups=[group],
    )
    rbd = tree.to_rbd()
    back = FaultTree.from_rbd(rbd)
    assert set(back.events) == {"a", "b"}
    assert [list(g.members) for g in back.ccf_groups] == [["a", "b"]]
    expected = tree.top_event_probability()
    assert back.top_event_probability() == pytest.approx(expected, rel=1e-12)
    assert 1 - rbd.sf() == pytest.approx(expected, rel=1e-12)
    # b alone, or its own failure, never fails the top event.
    assert back.minimal_cut_sets() == tree.minimal_cut_sets()


def random_tree(rng):
    """Gates of AND and OR over three to five events, some repeated, and
    the top gate."""
    events = [f"e{i}" for i in range(int(rng.integers(3, 6)))]
    nodes, gates = list(events), {}
    for g in range(int(rng.integers(2, 5))):
        size = min(int(rng.integers(2, 4)), len(nodes))
        inputs = [str(x) for x in rng.choice(nodes, size=size, replace=False)]
        gates[f"g{g}"] = (str(rng.choice(["and", "or"])), inputs)
        nodes.append(f"g{g}")
    top = f"g{len(gates) - 1}"
    reached, pending = set(), [top]
    while pending:
        x = pending.pop()
        if x not in reached:
            reached.add(x)
            pending += gates[x][1] if x in gates else []
    gates = {k: v for k, v in gates.items() if k in reached}
    return gates, [e for e in events if e in reached], top


def test_random_trees_with_a_common_cause_pair_go_round():
    rng = np.random.default_rng(237)
    trees = irrelevant = 0
    while trees < 60:
        gates, events, top = random_tree(rng)
        if len(events) < 2:
            continue
        probabilities = {e: float(rng.uniform(0.01, 0.2)) for e in events}
        pair = [str(x) for x in rng.choice(events, size=2, replace=False)]
        probabilities[pair[1]] = probabilities[pair[0]]
        group = CCFGroup(pair, BetaFactor(0.1))
        with warnings.catch_warnings():
            # Probabilities beyond the split's rare-event range.
            warnings.simplefilter("ignore", UserWarning)
            tree = FaultTree(gates, probabilities, top=top, ccf_groups=[group])
            rbd = tree.to_rbd()
            back = FaultTree.from_rbd(rbd)
            expected = tree.top_event_probability()
            assert back.top_event_probability() == pytest.approx(
                expected, rel=1e-12
            )
        trees += 1
        relevant = set(rbd._decomposition().nodes) & set(pair)
        if relevant:
            # A group with a member that matters keeps every member.
            assert set(pair) <= set(back.events)
            irrelevant += relevant != set(pair)
    # The trees the round trip refused before #237.
    assert irrelevant >= 5
