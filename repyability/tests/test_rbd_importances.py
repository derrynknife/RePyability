"""
Tests RBD's importance methods.

Uses pytest fixtures located in conftest.py in the tests/ directory.
"""
import pytest

from repyability.rbd.rbd import RBD


# Test birnbaum_importance() w/ composite RBD
def test_rbd_birnbaum_importance(rbd1: RBD):
    t = 2
    birnbaum_importance_dict = rbd1.birnbaum_importance(t)
    assert len(birnbaum_importance_dict) == 3
    assert (
        pytest.approx(
            rbd1.reliability["valve"].sf(t)
            - rbd1.reliability["pump2"].sf(t) * rbd1.reliability["valve"].sf(t)
        )
        == birnbaum_importance_dict["pump1"]
    )
    assert (
        pytest.approx(
            rbd1.reliability["valve"].sf(t)
            - rbd1.reliability["pump1"].sf(t) * rbd1.reliability["valve"].sf(t)
        )
        == birnbaum_importance_dict["pump2"]
    )
    assert (
        pytest.approx(
            1
            - rbd1.reliability["pump1"].ff(t) * rbd1.reliability["pump2"].ff(t)
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
        pytest.approx(rbd1.reliability["valve"].sf(t) - rbd1.sf(t))
        == improvement_potential["pump1"]
    )
    assert (
        pytest.approx(rbd1.reliability["valve"].sf(t) - rbd1.sf(t))
        == improvement_potential["pump2"]
    )
    assert (
        pytest.approx(
            (
                1
                - rbd1.reliability["pump1"].ff(t)
                * rbd1.reliability["pump2"].ff(t)
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
                - rbd1.reliability["pump2"].sf(t)
                * rbd1.reliability["valve"].sf(t)
            )
            / rbd1.ff(t)
        )
        == raw["pump1"]
    )
    assert (
        pytest.approx(
            (
                1
                - rbd1.reliability["pump1"].sf(t)
                * rbd1.reliability["valve"].sf(t)
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
        pytest.approx(rbd1.ff(t) / rbd1.reliability["valve"].ff(t))
        == rrw["pump1"]
    )
    assert (
        pytest.approx(
            pytest.approx(rbd1.ff(t) / rbd1.reliability["valve"].ff(t))
        )
        == rrw["pump2"]
    )
    assert (
        pytest.approx(
            rbd1.ff(t)
            / (
                rbd1.reliability["pump1"].ff(t)
                * rbd1.reliability["pump2"].ff(t)
            )
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
                rbd1.reliability["valve"].sf(t)
                - rbd1.reliability["pump2"].sf(t)
                * rbd1.reliability["valve"].sf(t)
            )
            # Correction factor:
            * (rbd1.reliability["pump1"].sf(t) / rbd1.sf(t))
        )
        == criticality_importance["pump1"]
    )
    assert (
        pytest.approx(
            # Birnbaum importance:
            (
                rbd1.reliability["valve"].sf(t)
                - rbd1.reliability["pump1"].sf(t)
                * rbd1.reliability["valve"].sf(t)
            )
            # Correction factor:
            * (rbd1.reliability["pump2"].sf(t) / rbd1.sf(t))
        )
        == criticality_importance["pump2"]
    )
    assert (
        pytest.approx(
            # Birnbaum importance:
            (
                1
                - rbd1.reliability["pump1"].ff(t)
                * rbd1.reliability["pump2"].ff(t)
            )
            # Correction factor:
            * (rbd1.reliability["valve"].sf(t) / rbd1.sf(t))
        )
        == criticality_importance["valve"]
    )


# Test fussel_vesely() w/ cut-set method


def test_fussel_vesely_incorrect_fv_type(rbd1: RBD):
    t = 2
    with pytest.raises(ValueError):
        rbd1.fussel_vesely(t, fv_type="a")


def test_fussel_vesely_c_rbd1(rbd1: RBD):
    t = 2
    fv_importance = rbd1.fussel_vesely(t, fv_type="c")
    assert (
        pytest.approx(
            rbd1.reliability["pump1"].ff(t)
            * rbd1.reliability["pump2"].ff(t)
            / rbd1.ff(t)
        )
        == fv_importance["pump1"]
    )
    assert (
        pytest.approx(
            rbd1.reliability["pump1"].ff(t)
            * rbd1.reliability["pump2"].ff(t)
            / rbd1.ff(t)
        )
        == fv_importance["pump2"]
    )
    assert (
        pytest.approx(rbd1.reliability["valve"].ff(t) / rbd1.ff(t))
        == fv_importance["valve"]
    )


def test_fussel_vesely_c_series(rbd_series: RBD):
    t = 2
    fv_importance = rbd_series.fussel_vesely(t, fv_type="c")
    assert (
        pytest.approx(rbd_series.reliability[2].ff(t) / rbd_series.ff(t))
        == fv_importance[2]
    )
    assert (
        pytest.approx(rbd_series.reliability[3].ff(t) / rbd_series.ff(t))
        == fv_importance[3]
    )
    assert (
        pytest.approx(rbd_series.reliability[4].ff(t) / rbd_series.ff(t))
        == fv_importance[4]
    )


def test_fussel_vesely_c_parallel(rbd_parallel: RBD):
    t = 2
    fv_importance = rbd_parallel.fussel_vesely(t, fv_type="c")
    fv_expected = (
        rbd_parallel.reliability[2].ff(t)
        * rbd_parallel.reliability[3].ff(t)
        * rbd_parallel.reliability[4].ff(t)
        / rbd_parallel.ff(t)
    )
    assert pytest.approx(fv_expected) == fv_importance[2]
    assert pytest.approx(fv_expected) == fv_importance[3]
    assert pytest.approx(fv_expected) == fv_importance[4]


def test_fussel_vesely_c_rbd2(rbd2: RBD):
    t = 2
    fv_importance = rbd2.fussel_vesely(t, fv_type="c")
    assert (
        pytest.approx(rbd2.reliability[2].ff(t) / rbd2.ff(t))
        == fv_importance[2]
    )
    assert (
        pytest.approx(
            rbd2.reliability[3].ff(t) * rbd2.reliability[4].ff(t) / rbd2.ff(t)
        )
        == fv_importance[3]
    )
    assert (
        pytest.approx(
            (
                rbd2.reliability[4].ff(t) * rbd2.reliability[5].ff(t)
                + rbd2.reliability[3].ff(t) * rbd2.reliability[4].ff(t)
                + rbd2.reliability[4].ff(t) * rbd2.reliability[6].ff(t)
            )
            / rbd2.ff(t)
        )
        == fv_importance[4]
    )
    assert (
        pytest.approx(
            rbd2.reliability[4].ff(t) * rbd2.reliability[5].ff(t) / rbd2.ff(t)
        )
        == fv_importance[5]
    )
    assert (
        pytest.approx(
            rbd2.reliability[4].ff(t) * rbd2.reliability[6].ff(t) / rbd2.ff(t)
        )
        == fv_importance[6]
    )
    assert (
        pytest.approx(rbd2.reliability[7].ff(t) / rbd2.ff(t))
        == fv_importance[7]
    )


def test_fussel_vesely_c_rbd3(rbd3: RBD):
    t = 2
    fv_importance = rbd3.fussel_vesely(t, fv_type="c")
    assert (
        pytest.approx(
            (
                rbd3.reliability[1].ff(t) * rbd3.reliability[3].ff(t)
                + rbd3.reliability[1].ff(t)
                * rbd3.reliability[4].ff(t)
                * rbd3.reliability[5].ff(t)
            )
            / rbd3.ff(t)
        )
        == fv_importance[1]
    )
    assert (
        pytest.approx(
            (
                rbd3.reliability[2].ff(t) * rbd3.reliability[4].ff(t)
                + rbd3.reliability[2].ff(t)
                * rbd3.reliability[3].ff(t)
                * rbd3.reliability[5].ff(t)
            )
            / rbd3.ff(t)
        )
        == fv_importance[2]
    )
    assert (
        pytest.approx(
            (
                rbd3.reliability[1].ff(t) * rbd3.reliability[3].ff(t)
                + rbd3.reliability[2].ff(t)
                * rbd3.reliability[3].ff(t)
                * rbd3.reliability[5].ff(t)
            )
            / rbd3.ff(t)
        )
        == fv_importance[3]
    )
    assert (
        pytest.approx(
            (
                rbd3.reliability[2].ff(t) * rbd3.reliability[4].ff(t)
                + rbd3.reliability[1].ff(t)
                * rbd3.reliability[4].ff(t)
                * rbd3.reliability[5].ff(t)
            )
            / rbd3.ff(t)
        )
        == fv_importance[4]
    )
    assert (
        pytest.approx(
            (
                rbd3.reliability[1].ff(t)
                * rbd3.reliability[4].ff(t)
                * rbd3.reliability[5].ff(t)
                + rbd3.reliability[2].ff(t)
                * rbd3.reliability[3].ff(t)
                * rbd3.reliability[5].ff(t)
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
        rbd.reliability[2].ff(t)
        * rbd.reliability[3].ff(t)
        * rbd.reliability[4].ff(t)
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
        pytest.approx(rbd.reliability[2].ff(t) / rbd.ff(t)) == fv_importance[2]
    )
    assert (
        pytest.approx(rbd.reliability[3].ff(t) / rbd.ff(t)) == fv_importance[3]
    )
    assert (
        pytest.approx(rbd.reliability[2].ff(t) / rbd.ff(t)) == fv_importance[4]
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
                rbd.reliability[2].ff(t)
                + rbd.reliability[2].ff(t) * rbd.reliability[3].ff(t)
            )
            / rbd.ff(t)
        )
        == fv_importance[2]
    )
    assert (
        pytest.approx(rbd.reliability[2].ff(t) / rbd.ff(t)) == fv_importance[3]
    )
    assert (
        pytest.approx(
            rbd.reliability[2].ff(t) * rbd.reliability[3].ff(t) / rbd.ff(t)
        )
        == fv_importance[4]
    )


# Test fussel_vesely() w/ path-set method


def test_fussel_vesely_p_rbd1(rbd1: RBD):
    t = 2
    fv_importance = rbd1.fussel_vesely(t, fv_type="p")
    assert (
        pytest.approx(
            rbd1.reliability["pump1"].ff(t)
            * rbd1.reliability["valve"].ff(t)
            / rbd1.ff(t)
        )
        == fv_importance["pump1"]
    )
    assert (
        pytest.approx(
            rbd1.reliability["pump2"].ff(t)
            * rbd1.reliability["valve"].ff(t)
            / rbd1.ff(t)
        )
        == fv_importance["pump2"]
    )
    assert (
        pytest.approx(
            (
                rbd1.reliability["pump1"].ff(t)
                * rbd1.reliability["valve"].ff(t)
                + rbd1.reliability["pump2"].ff(t)
                * rbd1.reliability["valve"].ff(t)
            )
            / rbd1.ff(t)
        )
        == fv_importance["valve"]
    )


def test_fussel_vesely_p_series(rbd_series: RBD):
    t = 2
    fv_importance = rbd_series.fussel_vesely(t, fv_type="p")
    expected_fv_importance = (
        rbd_series.reliability[2].ff(t)
        * rbd_series.reliability[3].ff(t)
        * rbd_series.reliability[4].ff(t)
        / rbd_series.ff(t)
    )
    assert pytest.approx(expected_fv_importance) == fv_importance[2]
    assert pytest.approx(expected_fv_importance) == fv_importance[3]
    assert pytest.approx(expected_fv_importance) == fv_importance[4]


def test_fussel_vesely_p_parallel(rbd_parallel: RBD):
    t = 2
    fv_importance = rbd_parallel.fussel_vesely(t, fv_type="p")
    assert (
        pytest.approx(rbd_parallel.reliability[2].ff(t) / rbd_parallel.ff(t))
        == fv_importance[2]
    )
    assert (
        pytest.approx(rbd_parallel.reliability[3].ff(t) / rbd_parallel.ff(t))
        == fv_importance[3]
    )
    assert (
        pytest.approx(rbd_parallel.reliability[4].ff(t) / rbd_parallel.ff(t))
        == fv_importance[4]
    )


def test_fussel_vesely_p_rbd2(rbd2: RBD):
    t = 2
    fv_importance = rbd2.fussel_vesely(t, fv_type="p")
    assert (
        pytest.approx(
            (
                (
                    rbd2.reliability[2].ff(t)
                    * rbd2.reliability[3].ff(t)
                    * rbd2.reliability[5].ff(t)
                    * rbd2.reliability[6].ff(t)
                    * rbd2.reliability[7].ff(t)
                )
                + (
                    rbd2.reliability[2].ff(t)
                    * rbd2.reliability[4].ff(t)
                    * rbd2.reliability[7].ff(t)
                )
            )
            / rbd2.ff(t)
        )
        == fv_importance[2]
    )
    assert (
        pytest.approx(
            (
                rbd2.reliability[2].ff(t)
                * rbd2.reliability[3].ff(t)
                * rbd2.reliability[5].ff(t)
                * rbd2.reliability[6].ff(t)
                * rbd2.reliability[7].ff(t)
            )
            / rbd2.ff(t)
        )
        == fv_importance[3]
    )
    assert (
        pytest.approx(
            (
                rbd2.reliability[2].ff(t)
                * rbd2.reliability[4].ff(t)
                * rbd2.reliability[7].ff(t)
            )
            / rbd2.ff(t)
        )
        == fv_importance[4]
    )
    assert (
        pytest.approx(
            (
                rbd2.reliability[2].ff(t)
                * rbd2.reliability[3].ff(t)
                * rbd2.reliability[5].ff(t)
                * rbd2.reliability[6].ff(t)
                * rbd2.reliability[7].ff(t)
            )
            / rbd2.ff(t)
        )
        == fv_importance[5]
    )
    assert (
        pytest.approx(
            (
                rbd2.reliability[2].ff(t)
                * rbd2.reliability[3].ff(t)
                * rbd2.reliability[5].ff(t)
                * rbd2.reliability[6].ff(t)
                * rbd2.reliability[7].ff(t)
            )
            / rbd2.ff(t)
        )
        == fv_importance[6]
    )
    assert (
        pytest.approx(
            (
                (
                    rbd2.reliability[2].ff(t)
                    * rbd2.reliability[3].ff(t)
                    * rbd2.reliability[5].ff(t)
                    * rbd2.reliability[6].ff(t)
                    * rbd2.reliability[7].ff(t)
                )
                + (
                    rbd2.reliability[2].ff(t)
                    * rbd2.reliability[4].ff(t)
                    * rbd2.reliability[7].ff(t)
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
                (rbd3.reliability[1].ff(t) * rbd3.reliability[2].ff(t))
                + (
                    rbd3.reliability[1].ff(t)
                    * rbd3.reliability[5].ff(t)
                    * rbd3.reliability[4].ff(t)
                )
            )
            / rbd3.ff(t)
        )
        == fv_importance[1]
    )
    assert (
        pytest.approx(
            (
                (rbd3.reliability[1].ff(t) * rbd3.reliability[2].ff(t))
                + (
                    rbd3.reliability[3].ff(t)
                    * rbd3.reliability[5].ff(t)
                    * rbd3.reliability[2].ff(t)
                )
            )
            / rbd3.ff(t)
        )
        == fv_importance[2]
    )
    assert (
        pytest.approx(
            (
                (rbd3.reliability[3].ff(t) * rbd3.reliability[4].ff(t))
                + (
                    rbd3.reliability[3].ff(t)
                    * rbd3.reliability[5].ff(t)
                    * rbd3.reliability[2].ff(t)
                )
            )
            / rbd3.ff(t)
        )
        == fv_importance[3]
    )
    assert (
        pytest.approx(
            (
                rbd3.reliability[3].ff(t) * rbd3.reliability[4].ff(t)
                + rbd3.reliability[1].ff(t)
                * rbd3.reliability[4].ff(t)
                * rbd3.reliability[5].ff(t)
            )
            / rbd3.ff(t)
        )
        == fv_importance[4]
    )
    assert (
        pytest.approx(
            (
                rbd3.reliability[1].ff(t)
                * rbd3.reliability[5].ff(t)
                * rbd3.reliability[4].ff(t)
                + rbd3.reliability[3].ff(t)
                * rbd3.reliability[5].ff(t)
                * rbd3.reliability[2].ff(t)
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
        pytest.approx(rbd.reliability[2].ff(t) / rbd.ff(t)) == fv_importance[2]
    )
    assert (
        pytest.approx(rbd.reliability[3].ff(t) / rbd.ff(t)) == fv_importance[3]
    )
    assert (
        pytest.approx(rbd.reliability[4].ff(t) / rbd.ff(t)) == fv_importance[4]
    )
    assert (
        pytest.approx(rbd.reliability[2].ff(t) / rbd.ff(t)) == fv_importance[5]
    )


def test_fussel_vesely_p_rbd_repeated_component_series(
    rbd_repeated_component_series: RBD,
):
    rbd = rbd_repeated_component_series
    t = 2
    fv_importance = rbd.fussel_vesely(t, fv_type="p")
    expected_fv_importance = (
        rbd.reliability[2].ff(t) * rbd.reliability[3].ff(t) / rbd.ff(t)
    )
    assert pytest.approx(expected_fv_importance) == fv_importance[2]
    assert pytest.approx(expected_fv_importance) == fv_importance[3]
    assert pytest.approx(expected_fv_importance) == fv_importance[4]
