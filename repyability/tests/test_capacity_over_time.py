"""The capacity distribution over time from new (#124):
``RepairableRBD.point_capacity`` and ``mission_capacity``.

Checked against closed forms (parallel exponential pumps are binomial at
their point availabilities; a degrading pump with exponential stages and
repair is a Markov chain), against the long run it settles at, the
availability over time, a nested diagram drawn flat, and the simulation's
delivered fraction.
"""

import warnings

import numpy as np
import pytest
import surpyval as surv
from scipy.integrate import quad
from scipy.linalg import expm
from scipy.stats import binom

from repyability import CapacityDistribution, DegradingNode, RepairableRBD
from repyability.rbd import routes

E = surv.Exponential.from_params
W = surv.Weibull.from_params
L = surv.LogNormal.from_params

PUMPS = [
    ("in", "a"),
    ("in", "b"),
    ("in", "c"),
    ("a", "out"),
    ("b", "out"),
    ("c", "out"),
]


def exponential_pumps(failure_rate=0.1, repair_rate=1.0, **options):
    pump = {
        "reliability": E([failure_rate]),
        "repairability": E([repair_rate]),
    }
    return RepairableRBD(
        PUMPS,
        {n: pump for n in "abc"},
        capacity={n: 50.0 for n in "abc"},
        **options,
    )


def available(t, failure_rate=0.1, repair_rate=1.0):
    total = failure_rate + repair_rate
    return repair_rate / total + failure_rate / total * np.exp(-total * t)


def test_exponential_pumps_are_binomial_at_their_availabilities():
    plant = exponential_pumps()
    x = np.array([0.0, 0.5, 1.0, 5.0, 100.0])
    capacity = plant.point_capacity(x)
    assert isinstance(capacity, CapacityDistribution)
    assert capacity.levels.tolist() == [0.0, 50.0, 100.0, 150.0]
    exact = np.array([binom.pmf(k, 3, available(x)) for k in range(4)])
    np.testing.assert_allclose(capacity.probabilities, exact, atol=1.2e-6)
    # The capacity is positive exactly when the system is up.
    np.testing.assert_allclose(
        1.0 - capacity.probabilities[0],
        plant.point_availability(x),
        atol=1e-15,
    )
    # Over a window, the mean of the distribution over time.
    t = np.array([1.0, 10.0, 1000.0])
    window = plant.mission_capacity(t)
    exact = np.array(
        [
            [
                quad(lambda s: binom.pmf(k, 3, available(s)), 0.0, u)[0] / u
                for u in t
            ]
            for k in range(4)
        ]
    )
    np.testing.assert_allclose(window.probabilities, exact, atol=1.2e-6)
    np.testing.assert_allclose(
        window.meets(1e-9), plant.mission_availability(t), atol=1e-12
    )


def test_a_scalar_time_gives_one_distribution():
    plant = exponential_pumps()
    at_one = plant.point_capacity(1.0)
    assert at_one.probabilities.shape == (4,)
    assert isinstance(at_one.meets(100.0), float)
    window = plant.mission_capacity(10.0)
    assert window.probabilities.shape == (4,)
    assert window.probabilities.sum() == pytest.approx(1.0)
    # A window of 0 is the distribution at 0: every pump new and up.
    start = plant.mission_capacity(0.0)
    assert start.levels.tolist() == [150.0]


def test_the_capacity_settles_at_the_long_run_distribution():
    plant = exponential_pumps()
    long_run = plant.capacity_distribution()
    late = plant.point_capacity(1000.0)
    np.testing.assert_allclose(
        late.probabilities, long_run.probabilities, atol=1e-9
    )
    window = plant.mission_capacity(1e7)
    np.testing.assert_allclose(
        window.probabilities, long_run.probabilities, atol=1e-7
    )


def degrading_pump(a=0.01, b=0.02, m=0.1):
    worn = DegradingNode([(50.0, E([a])), (25.0, E([b]))])
    return RepairableRBD(
        [("s", "p"), ("p", "t")],
        {"p": {"reliability": worn, "repairability": E([m])}},
    )


def test_a_degrading_pump_follows_its_markov_chain():
    # Full output for an exponential time, then half output for another,
    # then down for an exponential repair: a three-state Markov chain.
    a, b, m = 0.01, 0.02, 0.1
    rbd = degrading_pump(a, b, m)
    chain = np.array([[-a, a, 0.0], [0.0, -b, b], [m, 0.0, -m]])

    def states(t):
        full, half, down = expm(chain * t)[0]
        return np.array([down, half, full])

    x = np.array([0.0, 10.0, 50.0, 100.0, 1000.0])
    capacity = rbd.point_capacity(x)
    assert capacity.levels.tolist() == [0.0, 25.0, 50.0]
    exact = np.array([states(t) for t in x]).T
    np.testing.assert_allclose(capacity.probabilities, exact, atol=2e-7)
    np.testing.assert_allclose(
        capacity.probabilities[:, -1],
        rbd.capacity_distribution().probabilities,
        atol=1e-7,
    )
    window = rbd.mission_capacity(200.0)
    integral = np.array(
        [quad(lambda s: states(s)[k], 0.0, 200.0)[0] / 200.0 for k in range(3)]
    )
    np.testing.assert_allclose(window.probabilities, integral, atol=2e-7)


def test_levels_while_up_share_the_availability():
    pump = {"reliability": E([0.1]), "repairability": E([1.0])}
    rbd = RepairableRBD(
        [("s", "p"), ("p", "t")],
        {"p": pump},
        capacity={"p": {50.0: 0.8, 25.0: 0.2}},
    )
    x = np.array([0.5, 3.0])
    capacity = rbd.point_capacity(x)
    assert capacity.levels.tolist() == [0.0, 25.0, 50.0]
    up = available(x)
    np.testing.assert_allclose(
        capacity.probabilities,
        np.vstack([1.0 - up, 0.2 * up, 0.8 * up]),
        atol=3e-7,
    )


def skid_and_pump():
    unit = {"reliability": W([500.0, 1.8]), "repairability": L([2.5, 0.6])}
    skid = RepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        {"a": unit, "b": unit},
        capacity={"a": 50.0, "b": 50.0},
    )
    nested = RepairableRBD(
        [("in", "skid"), ("in", "c"), ("skid", "out"), ("c", "out")],
        {"skid": skid, "c": unit},
        capacity={"c": 50.0},
    )
    flat = RepairableRBD(
        PUMPS, {n: unit for n in "abc"}, capacity={n: 50.0 for n in "abc"}
    )
    return nested, flat


def test_a_nested_rbd_brings_its_own_distribution():
    nested, flat = skid_and_pump()
    x = np.array([1.0, 50.0, 500.0, 5000.0])
    for method in ("point_capacity", "mission_capacity"):
        left = getattr(nested, method)(x)
        right = getattr(flat, method)(x)
        assert left.levels.tolist() == right.levels.tolist()
        np.testing.assert_allclose(
            left.probabilities, right.probabilities, atol=1e-12
        )


@pytest.mark.parametrize("t", [200.0, 2000.0])
def test_the_delivered_fraction_is_what_the_simulation_estimates(t):
    _, plant = skid_and_pump()
    exact = plant.mission_capacity(t)
    simulated = plant.availability(t, mc_samples=3000, seed=3, demand=100.0)
    interval = simulated.delivered_fraction_interval(0.999)
    assert interval.lower <= exact.delivered_fraction(100.0) <= interval.upper
    assert exact.mean() == pytest.approx(simulated.mean_capacity, rel=2e-3)


def test_inspected_pumps_settle_into_their_calendar():
    tested = {
        "reliability": E([0.002]),
        "repairability": "instant",
        "inspection": {"interval": 100.0},
    }
    rbd = RepairableRBD(
        PUMPS,
        {n: tested for n in "abc"},
        capacity={n: 50.0 for n in "abc"},
    )
    long_run = rbd.capacity_distribution()
    window = rbd.mission_capacity(1e6)
    np.testing.assert_allclose(
        window.probabilities, long_run.probabilities, atol=1e-9
    )
    # Over whole intervals, the average is the long run's exactly.
    np.testing.assert_allclose(
        rbd.mission_capacity(300.0).probabilities,
        long_run.probabilities,
        atol=1e-12,
    )


def test_forced_nodes():
    plant = exponential_pumps()
    held = plant.point_capacity(1.0, working_nodes=["a"])
    up = available(1.0)
    assert held.levels.tolist() == [50.0, 100.0, 150.0]
    np.testing.assert_allclose(
        held.probabilities, binom.pmf([0, 1, 2], 2, up), atol=3e-7
    )
    broken = plant.point_capacity(1.0, broken_nodes=["a"])
    assert broken.levels.tolist() == [0.0, 50.0, 100.0]
    # A degrading pump held up is at its stages in proportion.
    rbd = degrading_pump()
    capacity = rbd.point_capacity([0.0, 50.0], working_nodes=["p"])
    shares = rbd.point_capacity([0.0, 50.0]).probabilities[1:]
    np.testing.assert_allclose(
        capacity.probabilities, shares / shares.sum(axis=0), atol=1e-12
    )


def test_what_the_capacity_over_time_refuses():
    plain = RepairableRBD(
        [("s", "p"), ("p", "t")],
        {"p": {"reliability": E([0.1]), "repairability": E([1.0])}},
    )
    report = plain.analysis_routes()
    for name in ("point_capacity", "mission_capacity"):
        assert report[name].route == routes.REFUSED
        with pytest.raises(ValueError) as error:
            getattr(plain, name)(10.0)
        assert str(error.value) == report[name].reason
    worn = DegradingNode([(50.0, E([0.01])), (25.0, E([0.02]))])
    scheduled = RepairableRBD(
        [("s", "p"), ("p", "t")],
        {
            "p": {
                "reliability": worn,
                "repairability": E([0.1]),
                "preventive": {"interval": 100.0},
            }
        },
    )
    route = scheduled.analysis_routes()["point_capacity"]
    assert route.route == routes.REFUSED
    with pytest.raises(NotImplementedError) as error:
        scheduled.point_capacity(10.0)
    assert str(error.value) == route.reason
    # With a shared crew it comes from the crews' chain (#146; see
    # test_chains_over_time.py), settling at the long-run distribution.
    crews = exponential_pumps(repair_crews=1)
    assert crews.analysis_routes()["mission_capacity"].route == (
        routes.NUMERICAL
    )
    settled = crews.mission_capacity(1e8)
    np.testing.assert_allclose(
        settled.probabilities,
        crews.capacity_distribution().probabilities,
        rtol=1e-7,
    )
    report = exponential_pumps().analysis_routes()
    assert report["point_capacity"].route == routes.NUMERICAL
    assert report["mission_capacity"].route == routes.NUMERICAL
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        exponential_pumps().point_capacity(1.0)
