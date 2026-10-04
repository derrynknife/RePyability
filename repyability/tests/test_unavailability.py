"""Unavailability over time to its own precision (#237):
``point_unavailability`` and ``mission_unavailability`` are not
``1 - point_availability`` and ``1 - mission_availability``, which round to
0 below about 1e-16, but agree with them where both are representable."""

import math

import numpy as np
import pytest
import surpyval as surv

from repyability import BetaFactor, CCFGroup, RepairableRBD

W, E, LN = (
    surv.Weibull.from_params,
    surv.Exponential.from_params,
    surv.LogNormal.from_params,
)
PAIR = [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")]
SERIES = [("s", "a"), ("a", "b"), ("b", "t")]
FAIL, MTTR = 1e-9, 8.0


def valve(rate=FAIL, mttr=MTTR):
    return {"reliability": E([rate]), "repairability": E([1 / mttr])}


def one_down(t, rate=FAIL, mttr=MTTR):
    """An exponential unit's probability of being down at ``t``, from new."""
    total = rate + 1 / mttr
    return -(rate / total) * np.expm1(-total * np.asarray(t, float))


def pair_mission(t, rate=FAIL, mttr=MTTR):
    """The mean over ``[0, t]`` of ``one_down`` squared."""
    total = rate + 1 / mttr
    steady = rate / total
    s = total * t
    return steady**2 * (
        1 - 2 * (-np.expm1(-s)) / s + (-np.expm1(-2 * s)) / (2 * s)
    )


def test_a_pair_of_valves_to_its_own_precision():
    pair = RepairableRBD(PAIR, {"a": valve(), "b": valve()})
    times = np.array([0.0, 1.0, 10.0, 100.0, 1000.0, 87600.0])
    got = pair.point_unavailability(times)
    assert got[0] == 0.0
    assert got[1:] == pytest.approx(one_down(times[1:]) ** 2, rel=1e-9)
    for t in (100.0, 8760.0):
        assert pair.mission_unavailability(t) == pytest.approx(
            pair_mission(t), rel=1e-6
        )
    assert pair.mission_unavailability(0.0) == 0.0


def test_a_unit_that_is_down_counts_in_full():
    pair = RepairableRBD(PAIR, {"a": valve(), "b": valve()})
    times = np.array([1.0, 50.0, 500.0])
    assert pair.point_unavailability(times, broken_nodes=["a"]) == (
        pytest.approx(one_down(times), rel=1e-9)
    )
    assert np.all(pair.point_unavailability(times, working_nodes=["a"]) == 0)


def test_a_series_pair_is_the_union():
    series = RepairableRBD(SERIES, {"a": valve(1e-3), "b": valve(2e-3)})
    times = np.array([5.0, 50.0, 5000.0])
    a, b = one_down(times, 1e-3), one_down(times, 2e-3)
    assert series.point_unavailability(times) == pytest.approx(
        a + b - a * b, rel=1e-9
    )


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"repair_crews": 1},
        {"ccf_groups": [CCFGroup(["a", "b"], BetaFactor(0.1))]},
    ],
    ids=["independent", "one repair crew", "a common-cause group"],
)
def test_it_is_one_less_the_availability_where_both_hold(options):
    if options:
        # Shared crews and common causes are worked out by Markov chains,
        # of exponential units; a group's members are alike.
        unit = {"reliability": E([1 / 200]), "repairability": E([0.3])}
        units = {"a": unit, "b": unit}
    else:
        units = {
            "a": {"reliability": W([200, 1.5]), "repairability": LN([1, 0.5])},
            "b": {"reliability": E([1 / 150]), "repairability": E([0.2])},
        }
    rbd = RepairableRBD(PAIR, units, **options)
    times = np.array([0.0, 3.0, 40.0, 400.0])
    down = rbd.point_unavailability(times)
    assert np.all(down >= 0.0)
    # The availability's curves are good to about 1e-6, and its early
    # values' small complements to about 1e-4 of themselves.
    assert down == pytest.approx(
        1 - rbd.point_availability(times), rel=1e-4, abs=1e-9
    )
    for t in (50.0, 400.0):
        assert rbd.mission_unavailability(t) == pytest.approx(
            1 - rbd.mission_availability(t), rel=1e-4, abs=1e-9
        )


def test_an_instant_repair_is_never_down():
    rbd = RepairableRBD(
        PAIR,
        {
            "a": {"reliability": E([1e-3]), "repairability": "instant"},
            "b": valve(1e-3, 10.0),
        },
    )
    assert np.all(rbd.point_unavailability([0.0, 10.0, 1000.0]) == 0.0)
    assert rbd.mission_unavailability(1000.0) == 0.0


def test_the_routes_say_how_each_is_worked_out():
    routes = RepairableRBD(
        PAIR, {"a": valve(), "b": valve()}
    ).analysis_routes()
    for name in ("point_unavailability", "mission_unavailability"):
        assert routes[name].route == routes[name.replace("un", "", 1)].route
    assert not math.isnan(
        RepairableRBD(PAIR, {"a": valve(), "b": valve()}).mean_unavailability()
    )
