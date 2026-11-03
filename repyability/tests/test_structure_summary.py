"""The structure's size without listing its sets (``structure_summary`` and
the counts): checked against the listed sets, which are an independent
route (they are enumerated from the diagram, the counts from the
reduction), and against hand counts."""

import pytest

from repyability import RBD, StructureSummary
from repyability.tests.catalogue import BRIDGE, nonrepairable_kinds, too_meshed
from repyability.tests.test_meshed_structures import (
    ladder,
    meshed_dag,
)

SERIES = [("s", "a"), ("a", "b"), ("b", "c"), ("c", "t")]
PARALLEL = [
    ("s", "a"),
    ("s", "b"),
    ("s", "c"),
    ("a", "t"),
    ("b", "t"),
    ("c", "t"),
]
CROSS = [
    ("s", "a"),
    ("s", "b"),
    ("a", "c"),
    ("b", "d"),
    ("a", "e"),
    ("e", "d"),
    ("c", "t"),
    ("d", "t"),
]


def listed(rbd):
    return len(rbd.minimal_path_sets()), len(rbd.minimal_cut_sets())


def test_series_has_one_path_set_and_a_cut_set_for_each_component():
    info = RBD(SERIES).structure_summary()
    assert (info.n_path_sets, info.n_cut_sets) == (1, 3)
    assert (info.n_components, info.module_depth, info.core_size) == (3, 1, 0)
    assert info.series_parallel and info.counts_exact


def test_parallel_has_a_path_set_for_each_component_and_one_cut_set():
    info = RBD(PARALLEL).structure_summary()
    assert (info.n_path_sets, info.n_cut_sets) == (3, 1)


def test_two_of_three_voting_node_is_a_component_in_series():
    edges = [
        ("s", "a"),
        ("s", "b"),
        ("s", "c"),
        ("a", "v"),
        ("b", "v"),
        ("c", "v"),
        ("v", "t"),
    ]
    rbd = RBD(edges, k={"v": 2})
    assert rbd.count_minimal_path_sets() == 3
    # By hand: the vote node alone, or any two of a, b, c.
    assert rbd.count_minimal_cut_sets() == 4
    assert listed(rbd) == (3, 4)


def test_the_cross_linked_diagram_has_a_core_of_three_paths():
    info = RBD(CROSS).structure_summary()
    # Paths a-c, b-d and a-e-d.
    assert info.n_path_sets == 3
    assert not info.series_parallel
    assert info.core_path_sets == 3
    assert info.n_components == 5
    assert (info.n_path_sets, info.n_cut_sets) == listed(RBD(CROSS))


def test_an_edge_straight_to_the_output_leaves_nothing_to_fail():
    rbd = RBD([("s", "a"), ("s", "t"), ("a", "t")])
    info = rbd.structure_summary()
    assert info.always_works and info.n_irrelevant == 1
    assert (info.n_path_sets, info.n_cut_sets) == (1, 0)


def test_nesting_depth_counts_groups_held_in_groups():
    # (a || b) in series with c, all in parallel with d: depth 3.
    edges = [
        ("s", "a"),
        ("s", "b"),
        ("a", "c"),
        ("b", "c"),
        ("c", "t"),
        ("s", "d"),
        ("d", "t"),
    ]
    info = RBD(edges).structure_summary()
    assert info.module_depth == 3
    assert info.n_modules == 3
    assert (info.n_path_sets, info.n_cut_sets) == listed(RBD(edges))


@pytest.mark.parametrize(
    "edges",
    [
        meshed_dag(n, seed=seed, p=p)
        for n, seed, p in [
            (8, 1, 0.3),
            (10, 2, 0.25),
            (12, 3, 0.2),
            (9, 4, 0.4),
            (14, 5, 0.15),
        ]
    ]
    + [BRIDGE, ladder(2)],
)
def test_counts_equal_the_listed_sets_on_random_diagrams(edges):
    rbd = RBD(edges)
    info = rbd.structure_summary()
    assert info.counts_exact
    assert (info.n_path_sets, info.n_cut_sets) == listed(rbd)
    assert rbd.count_minimal_path_sets() == info.n_path_sets
    assert rbd.count_minimal_cut_sets() == info.n_cut_sets


@pytest.mark.parametrize("name", list(nonrepairable_kinds()))
def test_counts_equal_the_listed_sets_on_the_catalogue(name):
    rbd = nonrepairable_kinds()[name]
    if rbd.structure_check["is_too_meshed"]:
        pytest.skip("too meshed to list")
    info = rbd.structure_summary()
    if info.counts_exact:
        assert (info.n_path_sets, info.n_cut_sets) == listed(rbd)
    else:
        # A repeated node: the counts bound the listed sets above.
        assert info.n_path_sets >= listed(rbd)[0]
        assert info.n_cut_sets >= listed(rbd)[1]


def test_a_diagram_too_meshed_gives_no_counts_and_says_why():
    rbd = too_meshed(lambda: RBD(BRIDGE))
    info = rbd.structure_summary()
    assert info.n_path_sets is None and info.n_cut_sets is None
    assert not info.counts_exact and info.too_meshed
    with pytest.raises(NotImplementedError):
        rbd.count_minimal_path_sets()


def test_the_summary_is_plain_data():
    info = RBD(CROSS).structure_summary()
    assert isinstance(info, StructureSummary)
    out = info.to_dict()
    assert out["n_path_sets"] == 3 and out["too_meshed"] is None
