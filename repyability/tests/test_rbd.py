"""
Tests many of the RBD class methods.

Uses pytest fixtures located in conftest.py in the tests/ directory.
"""
import pytest

from repyability.rbd.rbd import RBD


# Check components are correct lengths
def test_rbd_components(rbd1: RBD, rbd2: RBD):
    assert len(rbd1.components) == 5
    assert len(rbd2.components) == 8


# Test get_all_path_sets()
def test_rbd_get_all_path_sets(rbd1: RBD, rbd2: RBD):
    assert list(rbd1.get_all_path_sets()) == [
        ["source", "pump1", "valve", "sink"],
        ["source", "pump2", "valve", "sink"],
    ]
    assert list(rbd2.get_all_path_sets()) == [
        [1, 2, 3, 5, 6, 7, 8],
        [1, 2, 4, 7, 8],
    ]


# sf()

# Test sf() w/ simple series RBD
def test_rbd_sf_series(rbd_series: RBD):
    t = 5
    assert (
        pytest.approx(
            rbd_series.components[2].sf(t)
            * rbd_series.components[3].sf(t)
            * rbd_series.components[4].sf(t)
        )
        == rbd_series.sf(t)[0]
    )


# Test get_min_path_sets()


def test_rbd_get_min_path_sets_rbd_series(rbd_series: RBD):
    assert {(1, 2, 3, 4, 5)} == rbd_series.get_min_path_sets()


def test_rbd_get_min_path_sets_rbd_parallel(rbd_parallel: RBD):
    assert {
        (1, 2, 5),
        (1, 3, 5),
        (1, 4, 5),
    } == rbd_parallel.get_min_path_sets()


def test_rbd_get_min_path_sets_rbd1(rbd1: RBD):
    assert {
        ("source", "pump1", "valve", "sink"),
        ("source", "pump2", "valve", "sink"),
    } == rbd1.get_min_path_sets()


def test_rbd_get_min_path_sets_rbd1_exclude_in_out_node(rbd1: RBD):
    assert {
        ("pump1", "valve"),
        ("pump2", "valve"),
    } == rbd1.get_min_path_sets(include_in_out_nodes=False)


def test_rbd_get_min_path_sets_rbd2(rbd2: RBD):
    assert {(1, 2, 3, 5, 6, 7, 8), (1, 2, 4, 7, 8)} == rbd2.get_min_path_sets()


def test_rbd_get_min_path_sets_rbd3(rbd3: RBD):
    assert {
        (0, 1, 2, 6),
        (0, 1, 5, 4, 6),
        (0, 3, 4, 6),
        (0, 3, 5, 2, 6),
    } == rbd3.get_min_path_sets()


# Test get_min_cut_sets()


def test_rbd_get_min_cut_sets_rbd_series(rbd_series: RBD):
    assert {
        frozenset([2]),
        frozenset([3]),
        frozenset([4]),
    } == rbd_series.get_min_cut_sets()


def test_rbd_get_min_cut_sets_rbd_parallel(rbd_parallel: RBD):
    assert {frozenset([2, 3, 4])} == rbd_parallel.get_min_cut_sets()


def test_rbd_get_min_cut_sets_rbd1(rbd1: RBD):
    assert {
        frozenset(["pump1", "pump2"]),
        frozenset(["valve"]),
    } == rbd1.get_min_cut_sets()


def test_rbd_get_min_cut_sets_rbd2(rbd2: RBD):
    assert {
        frozenset([2]),
        frozenset([3, 4]),
        frozenset([4, 5]),
        frozenset([4, 6]),
        frozenset([7]),
    } == rbd2.get_min_cut_sets()


def test_rbd_get_min_cut_sets_rbd3(rbd3: RBD):
    assert {
        frozenset([1, 3]),
        frozenset([2, 4]),
        frozenset([1, 4, 5]),
        frozenset([2, 3, 5]),
    } == rbd3.get_min_cut_sets()


# Test sf() w/ simple series RBD
def test_rbd_sf_parallel(rbd_parallel: RBD):
    t = 2
    assert (
        pytest.approx(
            (1 - rbd_parallel.components[2].sf(t))
            * (1 - rbd_parallel.components[3].sf(t))
            * (1 - rbd_parallel.components[4].sf(t))
        )
        == 1 - rbd_parallel.sf(t)[0]
    )


# Test sf() w/ a simple RBD with both parallel and series components
def test_rbd_sf_composite(rbd1: RBD):
    """Tests with an RBD with both parallel and series components."""
    t = 2
    assert (
        pytest.approx(
            (
                1
                - (1 - rbd1.components["pump1"].sf(t))
                * (1 - rbd1.components["pump2"].sf(t))
            )
            * rbd1.components["valve"].sf(t)
        )
        == rbd1.sf(t)[0]
    )


# Test sf() w/ an RBD that cannot be reduced to parallel or series
def test_rbd_sf_complex(rbd3: RBD):
    assert pytest.approx(0.994780625) == rbd3.sf(1000)[0]


# Test sf() w/ repeated component
# i.e. Two nodes correspond to one node
def test_rbd_sf_repeated_component(rbd_repeated_component_parallel: RBD):
    t = 2
    assert (
        pytest.approx(
            (1 - rbd_repeated_component_parallel.components[2].sf(t))
            * (1 - rbd_repeated_component_parallel.components[3].sf(t))
            * (1 - rbd_repeated_component_parallel.components[4].sf(t))
            # Not 'component 5' as nodes 2 and 5 are both component 2
        )
        == 1 - rbd_repeated_component_parallel.sf(t)[0]
    )


# Test sf() w/ broken node
def test_rbd_sf_broken_node(rbd_parallel: RBD):
    t = 2
    assert (
        pytest.approx(
            (1 - rbd_parallel.components[2].sf(t))
            * (1 - rbd_parallel.components[4].sf(t))
        )
        == 1 - rbd_parallel.sf(t, broken_nodes=[3])[0]
    )


# Test sf() w/ broken component
def test_rbd_sf_broken_component(rbd_repeated_component_parallel: RBD):
    t = 2
    assert (
        pytest.approx(
            (1 - rbd_repeated_component_parallel.components[2].sf(t))
            * (1 - rbd_repeated_component_parallel.components[4].sf(t))
        )
        == 1 - rbd_repeated_component_parallel.sf(t, broken_components=[3])[0]
    )
    assert (
        pytest.approx(
            (1 - rbd_repeated_component_parallel.components[3].sf(t))
            * (1 - rbd_repeated_component_parallel.components[4].sf(t))
        )
        == 1 - rbd_repeated_component_parallel.sf(t, broken_components=[2])[0]
    )


# Test sf() w/ working node
def test_rbd_sf_working_node(rbd_series: RBD):
    t = 2
    assert (
        pytest.approx(
            rbd_series.components[2].sf(t) * rbd_series.components[4].sf(t)
        )
        == rbd_series.sf(t, working_nodes=[3])[0]
    )


# Test sf() w/ working component
def test_rbd_sf_working_component(rbd_repeated_component_parallel: RBD):
    t = 2
    assert (
        pytest.approx(1)
        == rbd_repeated_component_parallel.sf(t, working_components=[2])[0]
    )
    assert (
        pytest.approx(1)
        == rbd_repeated_component_parallel.sf(t, working_components=[3])[0]
    )


# Test sf() w/ working node with repeated component
def test_rbd_sf_working_node_repeated_component(
    rbd_repeated_component_composite: RBD,
):
    rbd = rbd_repeated_component_composite
    t = 2
    assert (
        pytest.approx(1 - rbd.components[2].ff(t) * rbd.components[3].ff(t))
        == rbd.sf(t, working_nodes=[3])[0]
    )
    print(rbd.components[2].sf(t))


# ff()

# Test ff(), can just test parallel as ff() just calls 1 - sf()
def test_rbd_ff(rbd_parallel: RBD):
    t = 5
    assert (
        pytest.approx(
            (1 - rbd_parallel.components[2].sf(t))
            * (1 - rbd_parallel.components[3].sf(t))
            * (1 - rbd_parallel.components[4].sf(t))
        )
        == rbd_parallel.ff(t)[0]
    )


# Importance calcs

# Test birnbaum_importance() w/ composite RBD
def test_rbd_birnbaum_importance(rbd1: RBD):
    t = 2
    birnbaum_importance_dict = rbd1.birnbaum_importance(t)
    assert len(birnbaum_importance_dict) == 3
    assert (
        pytest.approx(
            rbd1.components["valve"].sf(t)
            - rbd1.components["pump2"].sf(t) * rbd1.components["valve"].sf(t)
        )
        == birnbaum_importance_dict["pump1"]
    )
    assert (
        pytest.approx(
            rbd1.components["valve"].sf(t)
            - rbd1.components["pump1"].sf(t) * rbd1.components["valve"].sf(t)
        )
        == birnbaum_importance_dict["pump2"]
    )
    assert (
        pytest.approx(
            1 - rbd1.components["pump1"].ff(t) * rbd1.components["pump2"].ff(t)
        )
        == birnbaum_importance_dict["valve"]
    )


# Test birnbaum_importance() raises warning when dependence is found
def test_rbd_birnbaum_importance_dependence_warning(
    rbd_repeated_component_series: RBD,
):
    with pytest.warns(UserWarning):
        rbd_repeated_component_series.birnbaum_importance(2)


# Test improvement_potential()
def test_rbd_improvement_potential(rbd1: RBD):
    t = 2
    improvement_potential = rbd1.improvement_potential(t)
    assert len(improvement_potential) == 3
    assert (
        pytest.approx(rbd1.components["valve"].sf(t) - rbd1.sf(t))
        == improvement_potential["pump1"]
    )
    assert (
        pytest.approx(rbd1.components["valve"].sf(t) - rbd1.sf(t))
        == improvement_potential["pump2"]
    )
    assert (
        pytest.approx(
            (
                1
                - rbd1.components["pump1"].ff(t)
                * rbd1.components["pump2"].ff(t)
            )
            - rbd1.sf(t)
        )
        == improvement_potential["valve"]
    )


# Test risk_achievement_worth()
def test_rbd_risk_achievement_worth(rbd1: RBD):
    t = 2
    raw = rbd1.risk_achievement_worth(t)
    assert len(raw) == 3
    assert (
        pytest.approx(
            (
                1
                - rbd1.components["pump2"].sf(t)
                * rbd1.components["valve"].sf(t)
            )
            / rbd1.ff(t)
        )
        == raw["pump1"]
    )
    assert (
        pytest.approx(
            (
                1
                - rbd1.components["pump1"].sf(t)
                * rbd1.components["valve"].sf(t)
            )
            / rbd1.ff(t)
        )
        == raw["pump2"]
    )
    assert pytest.approx(1 / rbd1.ff(t)) == raw["valve"]


# Test risk_reduction_worth()
def test_rbd_risk_reduction_worth(rbd1: RBD):
    t = 2
    rrw = rbd1.risk_reduction_worth(t)
    assert len(rrw) == 3
    assert (
        pytest.approx(rbd1.ff(t) / rbd1.components["valve"].ff(t))
        == rrw["pump1"]
    )
    assert (
        pytest.approx(
            pytest.approx(rbd1.ff(t) / rbd1.components["valve"].ff(t))
        )
        == rrw["pump2"]
    )
    assert (
        pytest.approx(
            rbd1.ff(t)
            / (rbd1.components["pump1"].ff(t) * rbd1.components["pump2"].ff(t))
        )
        == rrw["valve"]
    )


# Test criticality_importance()
def test_rbd_criticality_importance(rbd1: RBD):
    t = 2
    criticality_importance = rbd1.criticality_importance(t)
    assert len(criticality_importance) == 3
    assert (
        pytest.approx(
            # Birnbaum importance:
            (
                rbd1.components["valve"].sf(t)
                - rbd1.components["pump2"].sf(t)
                * rbd1.components["valve"].sf(t)
            )
            # Correction factor:
            * (rbd1.components["pump1"].sf(t) / rbd1.sf(t))
        )
        == criticality_importance["pump1"]
    )
    assert (
        pytest.approx(
            # Birnbaum importance:
            (
                rbd1.components["valve"].sf(t)
                - rbd1.components["pump1"].sf(t)
                * rbd1.components["valve"].sf(t)
            )
            # Correction factor:
            * (rbd1.components["pump2"].sf(t) / rbd1.sf(t))
        )
        == criticality_importance["pump2"]
    )
    assert (
        pytest.approx(
            # Birnbaum importance:
            (
                1
                - rbd1.components["pump1"].ff(t)
                * rbd1.components["pump2"].ff(t)
            )
            # Correction factor:
            * (rbd1.components["valve"].sf(t) / rbd1.sf(t))
        )
        == criticality_importance["valve"]
    )


# Test fussel_vesely() w/ cut-set method


def test_fussel_vesely_c_rbd1(rbd1: RBD):
    t = 2
    fv_importance = rbd1.fussel_vesely(t, fv_type="c")
    assert (
        pytest.approx(
            rbd1.components["pump1"].ff(t)
            * rbd1.components["pump2"].ff(t)
            / rbd1.ff(t)
        )
        == fv_importance["pump1"]
    )
    assert (
        pytest.approx(
            rbd1.components["pump1"].ff(t)
            * rbd1.components["pump2"].ff(t)
            / rbd1.ff(t)
        )
        == fv_importance["pump2"]
    )
    assert (
        pytest.approx(rbd1.components["valve"].ff(t) / rbd1.ff(t))
        == fv_importance["valve"]
    )


def test_fussel_vesely_c_series(rbd_series: RBD):
    t = 2
    fv_importance = rbd_series.fussel_vesely(t, fv_type="c")
    assert (
        pytest.approx(rbd_series.components[2].ff(t) / rbd_series.ff(t))
        == fv_importance[2]
    )
    assert (
        pytest.approx(rbd_series.components[3].ff(t) / rbd_series.ff(t))
        == fv_importance[3]
    )
    assert (
        pytest.approx(rbd_series.components[4].ff(t) / rbd_series.ff(t))
        == fv_importance[4]
    )


def test_fussel_vesely_c_parallel(rbd_parallel: RBD):
    t = 2
    fv_importance = rbd_parallel.fussel_vesely(t, fv_type="c")
    fv_expected = (
        rbd_parallel.components[2].ff(t)
        * rbd_parallel.components[3].ff(t)
        * rbd_parallel.components[4].ff(t)
        / rbd_parallel.ff(t)
    )
    assert pytest.approx(fv_expected) == fv_importance[2]
    assert pytest.approx(fv_expected) == fv_importance[3]
    assert pytest.approx(fv_expected) == fv_importance[4]


def test_fussel_vesely_c_rbd2(rbd2: RBD):
    t = 2
    fv_importance = rbd2.fussel_vesely(t, fv_type="c")
    assert (
        pytest.approx(rbd2.components[2].ff(t) / rbd2.ff(t))
        == fv_importance[2]
    )
    assert (
        pytest.approx(
            rbd2.components[3].ff(t) * rbd2.components[4].ff(t) / rbd2.ff(t)
        )
        == fv_importance[3]
    )
    assert (
        pytest.approx(
            (
                rbd2.components[4].ff(t) * rbd2.components[5].ff(t)
                + rbd2.components[3].ff(t) * rbd2.components[4].ff(t)
                + rbd2.components[4].ff(t) * rbd2.components[6].ff(t)
            )
            / rbd2.ff(t)
        )
        == fv_importance[4]
    )
    assert (
        pytest.approx(
            rbd2.components[4].ff(t) * rbd2.components[5].ff(t) / rbd2.ff(t)
        )
        == fv_importance[5]
    )
    assert (
        pytest.approx(
            rbd2.components[4].ff(t) * rbd2.components[6].ff(t) / rbd2.ff(t)
        )
        == fv_importance[6]
    )
    assert (
        pytest.approx(rbd2.components[7].ff(t) / rbd2.ff(t))
        == fv_importance[7]
    )


def test_fussel_vesely_c_rbd3(rbd3: RBD):
    t = 2
    fv_importance = rbd3.fussel_vesely(t, fv_type="c")
    assert (
        pytest.approx(
            (
                rbd3.components[1].ff(t) * rbd3.components[3].ff(t)
                + rbd3.components[1].ff(t)
                * rbd3.components[4].ff(t)
                * rbd3.components[5].ff(t)
            )
            / rbd3.ff(t)
        )
        == fv_importance[1]
    )
    assert (
        pytest.approx(
            (
                rbd3.components[2].ff(t) * rbd3.components[4].ff(t)
                + rbd3.components[2].ff(t)
                * rbd3.components[3].ff(t)
                * rbd3.components[5].ff(t)
            )
            / rbd3.ff(t)
        )
        == fv_importance[2]
    )
    assert (
        pytest.approx(
            (
                rbd3.components[1].ff(t) * rbd3.components[3].ff(t)
                + rbd3.components[2].ff(t)
                * rbd3.components[3].ff(t)
                * rbd3.components[5].ff(t)
            )
            / rbd3.ff(t)
        )
        == fv_importance[3]
    )
    assert (
        pytest.approx(
            (
                rbd3.components[2].ff(t) * rbd3.components[4].ff(t)
                + rbd3.components[1].ff(t)
                * rbd3.components[4].ff(t)
                * rbd3.components[5].ff(t)
            )
            / rbd3.ff(t)
        )
        == fv_importance[4]
    )
    assert (
        pytest.approx(
            (
                rbd3.components[1].ff(t)
                * rbd3.components[4].ff(t)
                * rbd3.components[5].ff(t)
                + rbd3.components[2].ff(t)
                * rbd3.components[3].ff(t)
                * rbd3.components[5].ff(t)
            )
            / rbd3.ff(t)
        )
        == fv_importance[5]
    )


def test_fussel_vesely_c_repeated_component_parallel(
    rbd_repeated_component_parallel: RBD,
):
    rbd = rbd_repeated_component_parallel
    t = 2
    fv_importance = rbd.fussel_vesely(t, fv_type="c")
    fv_expected = (
        rbd.components[2].ff(t)
        * rbd.components[3].ff(t)
        * rbd.components[4].ff(t)
        / rbd.ff(t)
    )
    assert pytest.approx(fv_expected) == fv_importance[2]
    assert pytest.approx(fv_expected) == fv_importance[3]
    assert pytest.approx(fv_expected) == fv_importance[4]
    assert pytest.approx(fv_expected) == fv_importance[5]


def test_fussel_vesely_c_rbd_repeated_component_series(
    rbd_repeated_component_series: RBD,
):
    rbd = rbd_repeated_component_series
    t = 2
    fv_importance = rbd.fussel_vesely(t, fv_type="c")
    assert (
        pytest.approx(rbd.components[2].ff(t) / rbd.ff(t)) == fv_importance[2]
    )
    assert (
        pytest.approx(rbd.components[3].ff(t) / rbd.ff(t)) == fv_importance[3]
    )
    assert (
        pytest.approx(rbd.components[2].ff(t) / rbd.ff(t)) == fv_importance[4]
    )


def test_fussel_vesely_c_rbd_repeated_component_composite(
    rbd_repeated_component_composite: RBD,
):
    rbd = rbd_repeated_component_composite
    t = 2
    fv_importance = rbd.fussel_vesely(t, fv_type="c")
    assert (
        pytest.approx(
            (
                rbd.components[2].ff(t)
                + rbd.components[2].ff(t) * rbd.components[3].ff(t)
            )
            / rbd.ff(t)
        )
        == fv_importance[2]
    )
    assert (
        pytest.approx(rbd.components[2].ff(t) / rbd.ff(t)) == fv_importance[3]
    )
    assert (
        pytest.approx(
            rbd.components[2].ff(t) * rbd.components[3].ff(t) / rbd.ff(t)
        )
        == fv_importance[4]
    )


# Test fussel_vesely() w/ path-set method


def test_fussel_vesely_p_rbd1(rbd1: RBD):
    t = 2
    fv_importance = rbd1.fussel_vesely(t, fv_type="p")
    assert (
        pytest.approx(
            rbd1.components["pump1"].ff(t)
            * rbd1.components["valve"].ff(t)
            / rbd1.ff(t)
        )
        == fv_importance["pump1"]
    )
    assert (
        pytest.approx(
            rbd1.components["pump2"].ff(t)
            * rbd1.components["valve"].ff(t)
            / rbd1.ff(t)
        )
        == fv_importance["pump2"]
    )
    assert (
        pytest.approx(
            (
                rbd1.components["pump1"].ff(t) * rbd1.components["valve"].ff(t)
                + rbd1.components["pump2"].ff(t)
                * rbd1.components["valve"].ff(t)
            )
            / rbd1.ff(t)
        )
        == fv_importance["valve"]
    )


def test_fussel_vesely_p_series(rbd_series: RBD):
    t = 2
    fv_importance = rbd_series.fussel_vesely(t, fv_type="p")
    expected_fv_importance = (
        rbd_series.components[2].ff(t)
        * rbd_series.components[3].ff(t)
        * rbd_series.components[4].ff(t)
        / rbd_series.ff(t)
    )
    assert pytest.approx(expected_fv_importance) == fv_importance[2]
    assert pytest.approx(expected_fv_importance) == fv_importance[3]
    assert pytest.approx(expected_fv_importance) == fv_importance[4]


def test_fussel_vesely_p_parallel(rbd_parallel: RBD):
    t = 2
    fv_importance = rbd_parallel.fussel_vesely(t, fv_type="p")
    assert (
        pytest.approx(rbd_parallel.components[2].ff(t) / rbd_parallel.ff(t))
        == fv_importance[2]
    )
    assert (
        pytest.approx(rbd_parallel.components[3].ff(t) / rbd_parallel.ff(t))
        == fv_importance[3]
    )
    assert (
        pytest.approx(rbd_parallel.components[4].ff(t) / rbd_parallel.ff(t))
        == fv_importance[4]
    )


def test_fussel_vesely_p_rbd2(rbd2: RBD):
    t = 2
    fv_importance = rbd2.fussel_vesely(t, fv_type="p")
    assert (
        pytest.approx(
            (
                (
                    rbd2.components[2].ff(t)
                    * rbd2.components[3].ff(t)
                    * rbd2.components[5].ff(t)
                    * rbd2.components[6].ff(t)
                    * rbd2.components[7].ff(t)
                )
                + (
                    rbd2.components[2].ff(t)
                    * rbd2.components[4].ff(t)
                    * rbd2.components[7].ff(t)
                )
            )
            / rbd2.ff(t)
        )
        == fv_importance[2]
    )
    assert (
        pytest.approx(
            (
                rbd2.components[2].ff(t)
                * rbd2.components[3].ff(t)
                * rbd2.components[5].ff(t)
                * rbd2.components[6].ff(t)
                * rbd2.components[7].ff(t)
            )
            / rbd2.ff(t)
        )
        == fv_importance[3]
    )
    assert (
        pytest.approx(
            (
                rbd2.components[2].ff(t)
                * rbd2.components[4].ff(t)
                * rbd2.components[7].ff(t)
            )
            / rbd2.ff(t)
        )
        == fv_importance[4]
    )
    assert (
        pytest.approx(
            (
                rbd2.components[2].ff(t)
                * rbd2.components[3].ff(t)
                * rbd2.components[5].ff(t)
                * rbd2.components[6].ff(t)
                * rbd2.components[7].ff(t)
            )
            / rbd2.ff(t)
        )
        == fv_importance[5]
    )
    assert (
        pytest.approx(
            (
                rbd2.components[2].ff(t)
                * rbd2.components[3].ff(t)
                * rbd2.components[5].ff(t)
                * rbd2.components[6].ff(t)
                * rbd2.components[7].ff(t)
            )
            / rbd2.ff(t)
        )
        == fv_importance[6]
    )
    assert (
        pytest.approx(
            (
                (
                    rbd2.components[2].ff(t)
                    * rbd2.components[3].ff(t)
                    * rbd2.components[5].ff(t)
                    * rbd2.components[6].ff(t)
                    * rbd2.components[7].ff(t)
                )
                + (
                    rbd2.components[2].ff(t)
                    * rbd2.components[4].ff(t)
                    * rbd2.components[7].ff(t)
                )
            )
            / rbd2.ff(t)
        )
    ) == fv_importance[7]


def test_fussel_vesely_p_rbd3(rbd3: RBD):
    t = 2
    fv_importance = rbd3.fussel_vesely(t, fv_type="p")
    assert (
        pytest.approx(
            (
                (rbd3.components[1].ff(t) * rbd3.components[2].ff(t))
                + (
                    rbd3.components[1].ff(t)
                    * rbd3.components[5].ff(t)
                    * rbd3.components[4].ff(t)
                )
            )
            / rbd3.ff(t)
        )
        == fv_importance[1]
    )
    assert (
        pytest.approx(
            (
                (rbd3.components[1].ff(t) * rbd3.components[2].ff(t))
                + (
                    rbd3.components[3].ff(t)
                    * rbd3.components[5].ff(t)
                    * rbd3.components[2].ff(t)
                )
            )
            / rbd3.ff(t)
        )
        == fv_importance[2]
    )
    assert (
        pytest.approx(
            (
                (rbd3.components[3].ff(t) * rbd3.components[4].ff(t))
                + (
                    rbd3.components[3].ff(t)
                    * rbd3.components[5].ff(t)
                    * rbd3.components[2].ff(t)
                )
            )
            / rbd3.ff(t)
        )
        == fv_importance[3]
    )
    assert (
        pytest.approx(
            (
                rbd3.components[3].ff(t) * rbd3.components[4].ff(t)
                + rbd3.components[1].ff(t)
                * rbd3.components[4].ff(t)
                * rbd3.components[5].ff(t)
            )
            / rbd3.ff(t)
        )
        == fv_importance[4]
    )
    assert (
        pytest.approx(
            (
                rbd3.components[1].ff(t)
                * rbd3.components[5].ff(t)
                * rbd3.components[4].ff(t)
                + rbd3.components[3].ff(t)
                * rbd3.components[5].ff(t)
                * rbd3.components[2].ff(t)
            )
            / rbd3.ff(t)
        )
        == fv_importance[5]
    )


def test_fussel_vesely_p_repeated_component_parallel(
    rbd_repeated_component_parallel: RBD,
):
    rbd = rbd_repeated_component_parallel
    t = 2
    fv_importance = rbd.fussel_vesely(t, fv_type="p")
    assert (
        pytest.approx(rbd.components[2].ff(t) / rbd.ff(t)) == fv_importance[2]
    )
    assert (
        pytest.approx(rbd.components[3].ff(t) / rbd.ff(t)) == fv_importance[3]
    )
    assert (
        pytest.approx(rbd.components[4].ff(t) / rbd.ff(t)) == fv_importance[4]
    )
    assert (
        pytest.approx(rbd.components[2].ff(t) / rbd.ff(t)) == fv_importance[5]
    )


def test_fussel_vesely_p_rbd_repeated_component_series(
    rbd_repeated_component_series: RBD,
):
    rbd = rbd_repeated_component_series
    t = 2
    fv_importance = rbd.fussel_vesely(t, fv_type="p")
    expected_fv_importance = (
        rbd.components[2].ff(t) * rbd.components[3].ff(t) / rbd.ff(t)
    )
    assert pytest.approx(expected_fv_importance) == fv_importance[2]
    assert pytest.approx(expected_fv_importance) == fv_importance[3]
    assert pytest.approx(expected_fv_importance) == fv_importance[4]
