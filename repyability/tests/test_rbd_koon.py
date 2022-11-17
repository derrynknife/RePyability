"""
Tests many of the RBD class methods.

Uses pytest fixtures located in conftest.py in the tests/ directory.
"""

from repyability.rbd.rbd import RBD


def test_rbd_koon_default_k(rbd_series: RBD):
    assert rbd_series.G.nodes[1]["k"] == 1
    assert rbd_series.G.nodes[2]["k"] == 1
    assert rbd_series.G.nodes[3]["k"] == 1
    assert rbd_series.G.nodes[4]["k"] == 1
    assert rbd_series.G.nodes[5]["k"] == 1


def test_rbd_koon_k_given(rbd1_koon: RBD):
    assert rbd1_koon.G.nodes["source"]["k"] == 1
    assert rbd1_koon.G.nodes["pump1"]["k"] == 1
    assert rbd1_koon.G.nodes["pump2"]["k"] == 1
    assert rbd1_koon.G.nodes["valve"]["k"] == 2
    assert rbd1_koon.G.nodes["sink"]["k"] == 1


def test_rbd_get_min_path_sets_rbd_series_koon(rbd_series_koon: RBD):
    assert set() == rbd_series_koon.get_min_path_sets()


def test_rbd_get_min_path_sets_rbd1_koon(rbd1_koon: RBD):
    assert {
        frozenset(["source", "pump1", "pump2", "valve", "sink"])
    } == rbd1_koon.get_min_path_sets()


def test_rbd_get_min_path_sets_rbd1_exclude_in_out_node_koon(rbd1_koon: RBD):
    assert {
        frozenset(["pump1", "pump2", "valve"])
    } == rbd1_koon.get_min_path_sets(include_in_out_nodes=False)


def test_rbd_get_min_path_sets_rbd2_koon(rbd2_koon: RBD):
    assert {
        frozenset(frozenset([1, 2, 3, 4, 5, 6, 7, 8])),
    } == rbd2_koon.get_min_path_sets()


def test_rbd_get_min_path_sets_rbd3_koon1(rbd3_koon1: RBD):
    assert {
        frozenset([0, 1, 2, 6]),
        frozenset([0, 3, 4, 6]),
    } == rbd3_koon1.get_min_path_sets()


def test_rbd_get_min_path_sets_rbd3_koon2(rbd3_koon2: RBD):
    assert {
        frozenset([0, 1, 2, 5, 6]),
        frozenset([0, 1, 5, 4, 6]),
        frozenset([0, 3, 4, 6]),
    } == rbd3_koon2.get_min_path_sets()


def test_rbd_get_min_path_sets_rbd3_koon3(rbd3_koon3: RBD):
    assert {
        frozenset([0, 1, 2, 3, 5, 6]),
        frozenset({0, 3, 4, 6}),
    } == rbd3_koon3.get_min_path_sets()


def test_rbd_get_min_path_sets_rbd_koon_parallel_k1(rbd_koon_parallel_args):
    rbd = RBD(**rbd_koon_parallel_args, k={"v": 1})
    assert {
        frozenset(["s", 1, "v", "t"]),
        frozenset(["s", 2, "v", "t"]),
        frozenset(["s", 3, "v", "t"]),
    } == rbd.get_min_path_sets()


def test_rbd_get_min_path_sets_rbd_koon_parallel_k2(rbd_koon_parallel_args):
    rbd = RBD(**rbd_koon_parallel_args, k={"v": 2})
    assert {
        frozenset(["s", 1, 2, "v", "t"]),
        frozenset(["s", 1, 3, "v", "t"]),
        frozenset(["s", 2, 3, "v", "t"]),
    } == rbd.get_min_path_sets()


def test_rbd_get_min_path_sets_rbd_koon_parallel_k3(rbd_koon_parallel_args):
    rbd = RBD(**rbd_koon_parallel_args, k={"v": 3})
    assert {
        frozenset(["s", 1, 2, 3, "v", "t"]),
    } == rbd.get_min_path_sets()


# There are 4 combinations of (k("v1"), k("v2")) for the rbd_koon_composite
# RBD, let's test all of them...
def test_rbd_get_min_path_sets_rbd_koon_composite_k11(rbd_koon_composite_args):
    rbd = RBD(**rbd_koon_composite_args, k={"v1": 1, "v2": 1})
    assert {
        frozenset(["s", "a1", "v1", "b1", "v2", "t"]),
        frozenset(["s", "a1", "v1", "b2", "v2", "t"]),
        frozenset(["s", "a2", "v1", "b1", "v2", "t"]),
        frozenset(["s", "a2", "v1", "b2", "v2", "t"]),
    } == rbd.get_min_path_sets()


def test_rbd_get_min_path_sets_rbd_koon_composite_k12(rbd_koon_composite_args):
    rbd = RBD(**rbd_koon_composite_args, k={"v1": 1, "v2": 2})
    assert {
        frozenset(["s", "a1", "v1", "b1", "b2", "v2", "t"]),
        frozenset(["s", "a2", "v1", "b1", "b2", "v2", "t"]),
    } == rbd.get_min_path_sets()


def test_rbd_get_min_path_sets_rbd_koon_composite_k21(rbd_koon_composite_args):
    rbd = RBD(**rbd_koon_composite_args, k={"v1": 2, "v2": 1})
    assert {
        frozenset(["s", "a1", "a2", "v1", "b1", "v2", "t"]),
        frozenset(["s", "a1", "a2", "v1", "b2", "v2", "t"]),
    } == rbd.get_min_path_sets()


def test_rbd_get_min_path_sets_rbd_koon_composite_k22(rbd_koon_composite_args):
    rbd = RBD(**rbd_koon_composite_args, k={"v1": 2, "v2": 2})
    assert {
        frozenset(["s", "a1", "a2", "v1", "b1", "b2", "v2", "t"]),
    } == rbd.get_min_path_sets()


# And test impossible cases!


def test_rbd_get_min_path_sets_rbd_koon_composite_nonfunctioning(
    rbd_koon_composite_args,
):
    assert (
        set()
        == RBD(
            **rbd_koon_composite_args, k={"v1": 2, "v2": 3}
        ).get_min_path_sets()
    )
    assert (
        set()
        == RBD(
            **rbd_koon_composite_args, k={"v1": 3, "v2": 2}
        ).get_min_path_sets()
    )
    assert (
        set()
        == RBD(
            **rbd_koon_composite_args, k={"v1": 3, "v2": 3}
        ).get_min_path_sets()
    )


def test_rbd_get_min_path_sets_rbd_koon_simple(rbd_koon_simple: RBD):
    assert {frozenset(["s", 1, 2, "t"])} == rbd_koon_simple.get_min_path_sets()


def test_rbd_get_min_path_sets_rbd_koon_nonminimal(
    rbd_koon_nonminimal_args: dict,
):
    rbd_k1 = RBD(**rbd_koon_nonminimal_args, k={2: 1})
    assert {frozenset(["s", 1, 2, "t"])} == rbd_k1.get_min_path_sets()

    rbd_k2 = RBD(**rbd_koon_nonminimal_args, k={2: 2})
    assert {frozenset(["s", 1, 2, 3, 4, 5, "t"])} == rbd_k2.get_min_path_sets()


def test_rbd_get_min_path_sets_rbd_double_parallel_koon(
    rbd_double_parallel_args: dict,
):
    rbd = RBD(**rbd_double_parallel_args, k={3: 2})
    assert {
        frozenset(["s", 1, 2, 3, 4, "t"]),
        frozenset(["s", 1, 2, 3, 5, "t"]),
    } == rbd.get_min_path_sets()
