"""
Tests NonRepairableRBD's importance methods.

Uses pytest fixtures located in conftest.py in the tests/ directory.
"""
import pytest

from repyability.rbd.non_repairable_rbd import NonRepairableRBD


# Test birnbaum_importance() w/ composite NonRepairableRBD
def test_rbd_birnbaum_importance(rbd1: NonRepairableRBD):
    t = 2
    birnbaum_importance_dict = rbd1.birnbaum_importance(t)
    assert len(birnbaum_importance_dict) == 3
    assert (
        pytest.approx(
            rbd1.reliabilities["valve"].sf(t)
            - rbd1.reliabilities["pump2"].sf(t)
            * rbd1.reliabilities["valve"].sf(t)
        )
        == birnbaum_importance_dict["pump1"]
    )
    assert (
        pytest.approx(
            rbd1.reliabilities["valve"].sf(t)
            - rbd1.reliabilities["pump1"].sf(t)
            * rbd1.reliabilities["valve"].sf(t)
        )
        == birnbaum_importance_dict["pump2"]
    )
    assert (
        pytest.approx(
            1
            - rbd1.reliabilities["pump1"].ff(t)
            * rbd1.reliabilities["pump2"].ff(t)
        )
        == birnbaum_importance_dict["valve"]
    )


# Test improvement_potential()
def test_rbd_improvement_potential(rbd1: NonRepairableRBD):
    t = 2
    improvement_potential = rbd1.improvement_potential(t)
    assert len(improvement_potential) == 3
    assert (
        pytest.approx(rbd1.reliabilities["valve"].sf(t) - rbd1.sf(t))
        == improvement_potential["pump1"]
    )
    assert (
        pytest.approx(rbd1.reliabilities["valve"].sf(t) - rbd1.sf(t))
        == improvement_potential["pump2"]
    )
    assert (
        pytest.approx(
            (
                1
                - rbd1.reliabilities["pump1"].ff(t)
                * rbd1.reliabilities["pump2"].ff(t)
            )
            - rbd1.sf(t)
        )
        == improvement_potential["valve"]
    )


# Test risk_achievement_worth()
def test_rbd_risk_achievement_worth(rbd1: NonRepairableRBD):
    t = 2
    raw = rbd1.risk_achievement_worth(t)
    assert len(raw) == 3
    assert (
        pytest.approx(
            (
                1
                - rbd1.reliabilities["pump2"].sf(t)
                * rbd1.reliabilities["valve"].sf(t)
            )
            / rbd1.ff(t)
        )
        == raw["pump1"]
    )
    assert (
        pytest.approx(
            (
                1
                - rbd1.reliabilities["pump1"].sf(t)
                * rbd1.reliabilities["valve"].sf(t)
            )
            / rbd1.ff(t)
        )
        == raw["pump2"]
    )
    assert pytest.approx(1 / rbd1.ff(t)) == raw["valve"]


# Test risk_reduction_worth()
def test_rbd_risk_reduction_worth(rbd1: NonRepairableRBD):
    t = 2
    rrw = rbd1.risk_reduction_worth(t)
    assert len(rrw) == 3
    assert (
        pytest.approx(rbd1.ff(t) / rbd1.reliabilities["valve"].ff(t))
        == rrw["pump1"]
    )
    assert (
        pytest.approx(
            pytest.approx(rbd1.ff(t) / rbd1.reliabilities["valve"].ff(t))
        )
        == rrw["pump2"]
    )
    assert (
        pytest.approx(
            rbd1.ff(t)
            / (
                rbd1.reliabilities["pump1"].ff(t)
                * rbd1.reliabilities["pump2"].ff(t)
            )
        )
        == rrw["valve"]
    )


# Test criticality_importance()
def test_rbd_criticality_importance(rbd1: NonRepairableRBD):
    t = 2
    criticality_importance = rbd1.criticality_importance(t)
    assert len(criticality_importance) == 3
    assert (
        pytest.approx(
            # Birnbaum importance:
            (
                rbd1.reliabilities["valve"].sf(t)
                - rbd1.reliabilities["pump2"].sf(t)
                * rbd1.reliabilities["valve"].sf(t)
            )
            # Correction factor:
            * (rbd1.reliabilities["pump1"].sf(t) / rbd1.sf(t))
        )
        == criticality_importance["pump1"]
    )
    assert (
        pytest.approx(
            # Birnbaum importance:
            (
                rbd1.reliabilities["valve"].sf(t)
                - rbd1.reliabilities["pump1"].sf(t)
                * rbd1.reliabilities["valve"].sf(t)
            )
            # Correction factor:
            * (rbd1.reliabilities["pump2"].sf(t) / rbd1.sf(t))
        )
        == criticality_importance["pump2"]
    )
    assert (
        pytest.approx(
            # Birnbaum importance:
            (
                1
                - rbd1.reliabilities["pump1"].ff(t)
                * rbd1.reliabilities["pump2"].ff(t)
            )
            # Correction factor:
            * (rbd1.reliabilities["valve"].sf(t) / rbd1.sf(t))
        )
        == criticality_importance["valve"]
    )


# Test fussel_vesely() w/ cut-set method


def test_fussel_vesely_incorrect_fv_type(rbd1: NonRepairableRBD):
    t = 2
    with pytest.raises(ValueError):
        rbd1.fussel_vesely(t, fv_type="a")


def test_fussel_vesely_c_rbd1(rbd1: NonRepairableRBD):
    t = 2
    fv_importance = rbd1.fussel_vesely(t, fv_type="c")
    assert (
        pytest.approx(
            rbd1.reliabilities["pump1"].ff(t)
            * rbd1.reliabilities["pump2"].ff(t)
            / rbd1.ff(t)
        )
        == fv_importance["pump1"]
    )
    assert (
        pytest.approx(
            rbd1.reliabilities["pump1"].ff(t)
            * rbd1.reliabilities["pump2"].ff(t)
            / rbd1.ff(t)
        )
        == fv_importance["pump2"]
    )
    assert (
        pytest.approx(rbd1.reliabilities["valve"].ff(t) / rbd1.ff(t))
        == fv_importance["valve"]
    )


def test_fussel_vesely_c_series(rbd_series: NonRepairableRBD):
    t = 2
    fv_importance = rbd_series.fussel_vesely(t, fv_type="c")
    assert (
        pytest.approx(rbd_series.reliabilities[2].ff(t) / rbd_series.ff(t))
        == fv_importance[2]
    )
    assert (
        pytest.approx(rbd_series.reliabilities[3].ff(t) / rbd_series.ff(t))
        == fv_importance[3]
    )
    assert (
        pytest.approx(rbd_series.reliabilities[4].ff(t) / rbd_series.ff(t))
        == fv_importance[4]
    )


def test_fussel_vesely_c_parallel(rbd_parallel: NonRepairableRBD):
    fv_importance = rbd_parallel.fussel_vesely(fv_type="c")
    # TODO: Remove need for value in FixedEventProbability
    fv_expected = (
        rbd_parallel.reliabilities[2].ff(1.0)
        * rbd_parallel.reliabilities[3].ff(1.0)
        * rbd_parallel.reliabilities[4].ff(1.0)
        / rbd_parallel.ff()
    )
    assert pytest.approx(fv_expected) == fv_importance[2]
    assert pytest.approx(fv_expected) == fv_importance[3]
    assert pytest.approx(fv_expected) == fv_importance[4]


def test_fussel_vesely_c_rbd2(rbd2: NonRepairableRBD):
    t = 2
    fv_importance = rbd2.fussel_vesely(t, fv_type="c")
    assert (
        pytest.approx(rbd2.reliabilities[2].ff(t) / rbd2.ff(t))
        == fv_importance[2]
    )
    assert (
        pytest.approx(
            rbd2.reliabilities[3].ff(t)
            * rbd2.reliabilities[4].ff(t)
            / rbd2.ff(t)
        )
        == fv_importance[3]
    )
    assert (
        pytest.approx(
            (
                rbd2.reliabilities[4].ff(t) * rbd2.reliabilities[5].ff(t)
                + rbd2.reliabilities[3].ff(t) * rbd2.reliabilities[4].ff(t)
                + rbd2.reliabilities[4].ff(t) * rbd2.reliabilities[6].ff(t)
            )
            / rbd2.ff(t)
        )
        == fv_importance[4]
    )
    assert (
        pytest.approx(
            rbd2.reliabilities[4].ff(t)
            * rbd2.reliabilities[5].ff(t)
            / rbd2.ff(t)
        )
        == fv_importance[5]
    )
    assert (
        pytest.approx(
            rbd2.reliabilities[4].ff(t)
            * rbd2.reliabilities[6].ff(t)
            / rbd2.ff(t)
        )
        == fv_importance[6]
    )
    assert (
        pytest.approx(rbd2.reliabilities[7].ff(t) / rbd2.ff(t))
        == fv_importance[7]
    )


def test_fussel_vesely_c_rbd3(rbd3: NonRepairableRBD):
    t = 2
    fv_importance = rbd3.fussel_vesely(t, fv_type="c")
    assert (
        pytest.approx(
            (
                rbd3.reliabilities[1].ff(t) * rbd3.reliabilities[3].ff(t)
                + rbd3.reliabilities[1].ff(t)
                * rbd3.reliabilities[4].ff(t)
                * rbd3.reliabilities[5].ff(t)
            )
            / rbd3.ff(t)
        )
        == fv_importance[1]
    )
    assert (
        pytest.approx(
            (
                rbd3.reliabilities[2].ff(t) * rbd3.reliabilities[4].ff(t)
                + rbd3.reliabilities[2].ff(t)
                * rbd3.reliabilities[3].ff(t)
                * rbd3.reliabilities[5].ff(t)
            )
            / rbd3.ff(t)
        )
        == fv_importance[2]
    )
    assert (
        pytest.approx(
            (
                rbd3.reliabilities[1].ff(t) * rbd3.reliabilities[3].ff(t)
                + rbd3.reliabilities[2].ff(t)
                * rbd3.reliabilities[3].ff(t)
                * rbd3.reliabilities[5].ff(t)
            )
            / rbd3.ff(t)
        )
        == fv_importance[3]
    )
    assert (
        pytest.approx(
            (
                rbd3.reliabilities[2].ff(t) * rbd3.reliabilities[4].ff(t)
                + rbd3.reliabilities[1].ff(t)
                * rbd3.reliabilities[4].ff(t)
                * rbd3.reliabilities[5].ff(t)
            )
            / rbd3.ff(t)
        )
        == fv_importance[4]
    )
    assert (
        pytest.approx(
            (
                rbd3.reliabilities[1].ff(t)
                * rbd3.reliabilities[4].ff(t)
                * rbd3.reliabilities[5].ff(t)
                + rbd3.reliabilities[2].ff(t)
                * rbd3.reliabilities[3].ff(t)
                * rbd3.reliabilities[5].ff(t)
            )
            / rbd3.ff(t)
        )
        == fv_importance[5]
    )


def test_fussel_vesely_c_repeated_component_parallel(
    rbd_repeated_component_parallel: NonRepairableRBD,
):
    rbd = rbd_repeated_component_parallel
    t = 2
    fv_importance = rbd.fussel_vesely(t, fv_type="c")
    fv_expected = (
        rbd.reliabilities[2].ff(t)
        * rbd.reliabilities[3].ff(t)
        * rbd.reliabilities[4].ff(t)
        / rbd.ff(t)
    )
    assert pytest.approx(fv_expected) == fv_importance[2]
    assert pytest.approx(fv_expected) == fv_importance[3]
    assert pytest.approx(fv_expected) == fv_importance[4]


# Test fussel_vesely() w/ path-set method


def test_fussel_vesely_p_rbd1(rbd1: NonRepairableRBD):
    t = 2
    fv_importance = rbd1.fussel_vesely(t, fv_type="p")
    assert (
        pytest.approx(
            rbd1.reliabilities["pump1"].ff(t)
            * rbd1.reliabilities["valve"].ff(t)
            / rbd1.ff(t)
        )
        == fv_importance["pump1"]
    )
    assert (
        pytest.approx(
            rbd1.reliabilities["pump2"].ff(t)
            * rbd1.reliabilities["valve"].ff(t)
            / rbd1.ff(t)
        )
        == fv_importance["pump2"]
    )
    assert (
        pytest.approx(
            (
                rbd1.reliabilities["pump1"].ff(t)
                * rbd1.reliabilities["valve"].ff(t)
                + rbd1.reliabilities["pump2"].ff(t)
                * rbd1.reliabilities["valve"].ff(t)
            )
            / rbd1.ff(t)
        )
        == fv_importance["valve"]
    )


def test_fussel_vesely_p_series(rbd_series: NonRepairableRBD):
    t = 2
    fv_importance = rbd_series.fussel_vesely(t, fv_type="p")
    expected_fv_importance = (
        rbd_series.reliabilities[2].ff(t)
        * rbd_series.reliabilities[3].ff(t)
        * rbd_series.reliabilities[4].ff(t)
        / rbd_series.ff(t)
    )
    assert pytest.approx(expected_fv_importance) == fv_importance[2]
    assert pytest.approx(expected_fv_importance) == fv_importance[3]
    assert pytest.approx(expected_fv_importance) == fv_importance[4]


def test_fussel_vesely_p_parallel(rbd_parallel: NonRepairableRBD):
    t = 2
    fv_importance = rbd_parallel.fussel_vesely(t, fv_type="p")
    assert (
        pytest.approx(rbd_parallel.reliabilities[2].ff(t) / rbd_parallel.ff(t))
        == fv_importance[2]
    )
    assert (
        pytest.approx(rbd_parallel.reliabilities[3].ff(t) / rbd_parallel.ff(t))
        == fv_importance[3]
    )
    assert (
        pytest.approx(rbd_parallel.reliabilities[4].ff(t) / rbd_parallel.ff(t))
        == fv_importance[4]
    )


def test_fussel_vesely_p_rbd2(rbd2: NonRepairableRBD):
    t = 2
    fv_importance = rbd2.fussel_vesely(t, fv_type="p")
    assert (
        pytest.approx(
            (
                (
                    rbd2.reliabilities[2].ff(t)
                    * rbd2.reliabilities[3].ff(t)
                    * rbd2.reliabilities[5].ff(t)
                    * rbd2.reliabilities[6].ff(t)
                    * rbd2.reliabilities[7].ff(t)
                )
                + (
                    rbd2.reliabilities[2].ff(t)
                    * rbd2.reliabilities[4].ff(t)
                    * rbd2.reliabilities[7].ff(t)
                )
            )
            / rbd2.ff(t)
        )
        == fv_importance[2]
    )
    assert (
        pytest.approx(
            (
                rbd2.reliabilities[2].ff(t)
                * rbd2.reliabilities[3].ff(t)
                * rbd2.reliabilities[5].ff(t)
                * rbd2.reliabilities[6].ff(t)
                * rbd2.reliabilities[7].ff(t)
            )
            / rbd2.ff(t)
        )
        == fv_importance[3]
    )
    assert (
        pytest.approx(
            (
                rbd2.reliabilities[2].ff(t)
                * rbd2.reliabilities[4].ff(t)
                * rbd2.reliabilities[7].ff(t)
            )
            / rbd2.ff(t)
        )
        == fv_importance[4]
    )
    assert (
        pytest.approx(
            (
                rbd2.reliabilities[2].ff(t)
                * rbd2.reliabilities[3].ff(t)
                * rbd2.reliabilities[5].ff(t)
                * rbd2.reliabilities[6].ff(t)
                * rbd2.reliabilities[7].ff(t)
            )
            / rbd2.ff(t)
        )
        == fv_importance[5]
    )
    assert (
        pytest.approx(
            (
                rbd2.reliabilities[2].ff(t)
                * rbd2.reliabilities[3].ff(t)
                * rbd2.reliabilities[5].ff(t)
                * rbd2.reliabilities[6].ff(t)
                * rbd2.reliabilities[7].ff(t)
            )
            / rbd2.ff(t)
        )
        == fv_importance[6]
    )
    assert (
        pytest.approx(
            (
                (
                    rbd2.reliabilities[2].ff(t)
                    * rbd2.reliabilities[3].ff(t)
                    * rbd2.reliabilities[5].ff(t)
                    * rbd2.reliabilities[6].ff(t)
                    * rbd2.reliabilities[7].ff(t)
                )
                + (
                    rbd2.reliabilities[2].ff(t)
                    * rbd2.reliabilities[4].ff(t)
                    * rbd2.reliabilities[7].ff(t)
                )
            )
            / rbd2.ff(t)
        )
    ) == fv_importance[7]


def test_fussel_vesely_p_rbd3(rbd3: NonRepairableRBD):
    t = 2
    fv_importance = rbd3.fussel_vesely(t, fv_type="p")
    assert (
        pytest.approx(
            (
                (rbd3.reliabilities[1].ff(t) * rbd3.reliabilities[2].ff(t))
                + (
                    rbd3.reliabilities[1].ff(t)
                    * rbd3.reliabilities[5].ff(t)
                    * rbd3.reliabilities[4].ff(t)
                )
            )
            / rbd3.ff(t)
        )
        == fv_importance[1]
    )
    assert (
        pytest.approx(
            (
                (rbd3.reliabilities[1].ff(t) * rbd3.reliabilities[2].ff(t))
                + (
                    rbd3.reliabilities[3].ff(t)
                    * rbd3.reliabilities[5].ff(t)
                    * rbd3.reliabilities[2].ff(t)
                )
            )
            / rbd3.ff(t)
        )
        == fv_importance[2]
    )
    assert (
        pytest.approx(
            (
                (rbd3.reliabilities[3].ff(t) * rbd3.reliabilities[4].ff(t))
                + (
                    rbd3.reliabilities[3].ff(t)
                    * rbd3.reliabilities[5].ff(t)
                    * rbd3.reliabilities[2].ff(t)
                )
            )
            / rbd3.ff(t)
        )
        == fv_importance[3]
    )
    assert (
        pytest.approx(
            (
                rbd3.reliabilities[3].ff(t) * rbd3.reliabilities[4].ff(t)
                + rbd3.reliabilities[1].ff(t)
                * rbd3.reliabilities[4].ff(t)
                * rbd3.reliabilities[5].ff(t)
            )
            / rbd3.ff(t)
        )
        == fv_importance[4]
    )
    assert (
        pytest.approx(
            (
                rbd3.reliabilities[1].ff(t)
                * rbd3.reliabilities[5].ff(t)
                * rbd3.reliabilities[4].ff(t)
                + rbd3.reliabilities[3].ff(t)
                * rbd3.reliabilities[5].ff(t)
                * rbd3.reliabilities[2].ff(t)
            )
            / rbd3.ff(t)
        )
        == fv_importance[5]
    )


def test_fussel_vesely_p_repeated_component_parallel(
    rbd_repeated_component_parallel: NonRepairableRBD,
):
    rbd = rbd_repeated_component_parallel
    t = 2
    fv_importance = rbd.fussel_vesely(t, fv_type="p")
    assert (
        pytest.approx(rbd.reliabilities[2].ff(t) / rbd.ff(t))
        == fv_importance[2]
    )
    assert (
        pytest.approx(rbd.reliabilities[3].ff(t) / rbd.ff(t))
        == fv_importance[3]
    )
    assert (
        pytest.approx(rbd.reliabilities[4].ff(t) / rbd.ff(t))
        == fv_importance[4]
    )
