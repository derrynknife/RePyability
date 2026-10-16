"""Minimal repair whose repairs take time (#318): a component repaired
minimally (Kijima, ``q = 1``) fails at its life's hazard at its operating
age, the time it has been up, and each failure's repair takes time. Its
availability over time and its expected failures are worked out on a grid
of its operating age (``MinimalRepairCurve``, by
``minimal_repair_lattice``), and the values over a window, the importance
over time and the expected cost follow for any system with it. Its
long-run values stay refused: its rate of failures need not settle."""

import math

import numpy as np
import pytest
import surpyval as surv

from repyability import RepairableRBD
from repyability.rbd import _curves
from repyability.rbd import routes as r
from repyability.rbd._point_availability import (
    MinimalRepairCurve,
    _cells,
    _lattice,
    cumulative_hazard,
)
from repyability.rbd._point_availability import knots as point_knots
from repyability.rbd._point_availability import minimal_repair_lattice
from repyability.tests import catalogue

W = surv.Weibull.from_params
E = surv.Exponential.from_params
L = surv.LogNormal.from_params
EDGES = [("s", "c"), ("c", "t")]
MINIMAL = {"model": "kijima1", "q": 1.0}


def single(spec):
    return RepairableRBD(EDGES, {"c": spec})


def minimal(life, repairs, **more):
    """A unit minimally repaired, its repairs taking ``repairs``."""
    return {
        "reliability": life,
        "repairability": repairs,
        "repair": MINIMAL,
        **more,
    }


def closed_forms(rate: float, mu: float, t: np.ndarray):
    """An exponential unit (failure rate ``rate``, repair rate ``mu``):
    its point availability and expected failures by ``t``, the two-state
    Markov chain's."""
    total = rate + mu
    up = mu / total + rate / total * np.exp(-total * t)
    failures = rate * (
        mu / total * t + rate / total**2 * (1.0 - np.exp(-total * t))
    )
    return up, failures


# -- the lattice -------------------------------------------------------------


def test_the_lattice_against_the_two_state_chain():
    # An exponential life keeps no age, so the sum over the powers of an
    # exponential repair time is the two-state Markov chain, in closed form.
    rate, mu = 0.02, 0.2
    t = 0.05 * np.arange(4001)
    life, repair = E([rate]), E([mu])
    lattice = _lattice(
        *_cells(
            lambda s: 1.0 - np.asarray(repair.sf(s), float),
            t,
            point_knots(repair),
        )
    )
    up, failures = minimal_repair_lattice(cumulative_hazard(life, t), lattice)
    exact_up, exact_failures = closed_forms(rate, mu, t)
    np.testing.assert_allclose(up, exact_up, atol=1e-6)
    np.testing.assert_allclose(failures, exact_failures, atol=1e-6)


def test_the_lattice_error_falls_as_the_square_of_the_step():
    rate, mu = 0.02, 0.2
    life, repair = E([rate]), E([mu])
    errors = []
    for n in (500, 1000, 2000):
        t = (100.0 / n) * np.arange(n + 1)
        lattice = _lattice(
            *_cells(
                lambda s: 1.0 - np.asarray(repair.sf(s), float),
                t,
                point_knots(repair),
            )
        )
        up, _ = minimal_repair_lattice(cumulative_hazard(life, t), lattice)
        errors.append(np.abs(up - closed_forms(rate, mu, t)[0]).max())
    assert errors[0] / errors[1] > 3.5
    assert errors[1] / errors[2] > 3.5


def test_repairs_in_no_time_on_the_lattice_give_the_closed_form():
    # The atom at 0 for every power: the sums are 1 and H(t), which the
    # curve takes in closed form.
    life = W([100.0, 2.0])
    t = 0.1 * np.arange(2001)
    lattice = np.zeros(len(t))
    lattice[0] = 1.0
    up, failures = minimal_repair_lattice(cumulative_hazard(life, t), lattice)
    np.testing.assert_allclose(up, 1.0, atol=1e-13)
    np.testing.assert_allclose(failures, (t / 100.0) ** 2.0, atol=1e-11)


def test_the_curve_keeps_the_interface():
    life, repair = W([100.0, 2.0]), L([2.0, 0.5])
    curve = MinimalRepairCurve(
        life,
        100.0,
        1000,
        True,
        repair_sf=lambda s: np.asarray(repair.sf(s), float),
        repair_splits=point_knots(repair),
        step=0.1,
        n=1000,
    )
    assert curve.settle == np.inf and curve.period is None
    assert curve.grids() == [(0.1, 100.0)]
    np.testing.assert_array_equal(curve.breaks(0.0, 100.0), [0.0, 100.0])
    assert len(curve.knots(0.0, 100.0)) == 1003
    x = np.array([0.0, 0.05, 50.0, 100.0])
    values = curve.at(x)
    assert values[0] == 1.0 and np.all(np.diff(values) < 0.0)
    events = curve.events(x)
    assert events["failures"][0] == 0.0
    np.testing.assert_array_equal(events["corrective"], events["failures"])
    assert np.all(events["preventive"] == 0.0)
    assert curve.atoms(100.0).times.size == 0


# -- against the exact cases --------------------------------------------------


def test_repairs_of_no_length_are_minimal_repair_in_no_time():
    # A repair of 1e-8 on average: up throughout, failing H(t) times, as
    # the closed form for repairs in no time has it.
    life = W([100.0, 2.5])
    timed = single(minimal(life, E([1e8]), repair_cost=1.0))
    instant = single(minimal(life, "instant", repair_cost=1.0))
    t = np.array([10.0, 100.0, 250.0])
    np.testing.assert_allclose(timed.point_availability(t), 1.0, atol=1e-6)
    # Linear between its grid's points, where the closed form is convex:
    # to 1e-7 or so, as any curve on a grid.
    np.testing.assert_allclose(
        timed.expected_failures(t),
        instant.expected_failures(t),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        timed.expected_failures(t),
        cumulative_hazard(life, t),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        timed.expected_cost(t).mean,
        instant.expected_cost(t).mean,
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.parametrize(
    "repair",
    [E([0.2]), L([2.0, 0.5]), W([3.0, 3.0])],
    ids=["exponential", "lognormal", "weibull"],
)
def test_with_an_exponential_life_minimal_repair_is_renewal(repair):
    # A memoryless unit is as good as new after any repair, so its minimal
    # repair is the plain renewal unit, worked out by its renewal equation.
    life = E([0.02])
    minimally = single(minimal(life, repair, repair_cost=3.0))
    renewed = single(
        {"reliability": life, "repairability": repair, "repair_cost": 3.0}
    )
    t = np.array([5.0, 25.0, 100.0, 300.0])
    np.testing.assert_allclose(
        minimally.point_availability(t),
        renewed.point_availability(t),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        minimally.expected_failures(t), renewed.expected_failures(t), atol=1e-6
    )
    np.testing.assert_allclose(
        minimally.mission_availability(t),
        renewed.mission_availability(t),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        minimally.expected_cost(t).mean,
        renewed.expected_cost(t).mean,
        rtol=1e-6,
    )
    one, other = minimally.expected_events(t), renewed.expected_events(t)
    np.testing.assert_allclose(
        one.node_downtime["c"], other.node_downtime["c"], atol=1e-5
    )


def test_an_exponential_unit_against_its_two_state_chain():
    rate, mu = 0.02, 0.2
    rbd = single(minimal(E([rate]), E([mu])))
    t = np.array([5.0, 25.0, 100.0, 300.0])
    up, failures = closed_forms(rate, mu, t)
    np.testing.assert_allclose(rbd.point_availability(t), up, atol=1e-6)
    np.testing.assert_allclose(rbd.expected_failures(t), failures, atol=1e-6)


# -- against the simulation --------------------------------------------------


def simulated(rbd, horizon: float, mc_samples: int, seed: int):
    """A plain run (its means its own, not the exact methods'): the
    system's availability on a grid, its mean, and its failures."""
    return rbd.availability(
        horizon,
        mc_samples=mc_samples,
        seed=seed,
        curve_points=10,
        control_variate=False,
    )


@pytest.mark.parametrize(
    "life, repair",
    [
        (W([100.0, 2.0]), L([2.0, 0.5])),
        (W([100.0, 3.0]), L([2.5, 1.0])),
        (W([100.0, 0.8]), E([0.1])),
    ],
    ids=["shape 2, lognormal", "shape 3, lognormal", "wearing in"],
)
def test_a_timed_unit_against_the_simulation(life, repair):
    rbd = single(minimal(life, repair, repair_cost=1.0))
    horizon, n = 300.0, 4000
    sim = simulated(rbd, horizon, n, 2)
    # The curve: the fraction up at each grid time, binomial.
    grid = sim.timeline[1:-1]
    up = rbd.point_availability(grid)
    error = np.sqrt(up * (1.0 - up) / n)
    assert np.all(np.abs(sim.availability[1:-1] - up) < 4.0 * error + 1e-3)
    # The failures: only the repairs are priced, at 1 each.
    failures = rbd.expected_failures(horizon)
    error = float(np.std(sim.cost.samples)) / math.sqrt(n)
    assert abs(sim.cost.mean - failures) < 4.0 * error
    assert sim.system_failures / n == pytest.approx(failures, rel=0.05)
    # The mean availability over the window.
    error = float(np.std(sim.uptimes)) / math.sqrt(n) / horizon
    assert (
        abs(sim.sample_mean_availability - rbd.mission_availability(horizon))
        < 4.0 * error
    )


def test_a_fitted_crow_amsaa_process_against_the_simulation():
    rbd = catalogue.repairable_kinds()["fitted Crow-AMSAA process"]
    horizon, n = 1000.0, 4000
    sim = simulated(rbd, horizon, n, 5)
    counted = rbd.expected_events(horizon)
    error = float(np.std(sim.cost.samples)) / math.sqrt(n)
    assert abs(sim.cost.mean - counted.node_failures["a"]) < 4.0 * error
    assert sim.system_failures / n == pytest.approx(
        counted.system_failures, rel=0.05
    )
    error = float(np.std(sim.uptimes)) / math.sqrt(n) / horizon
    assert (
        abs(sim.sample_mean_availability - rbd.mission_availability(horizon))
        < 4.0 * error
    )


def pair(pump, **options):
    """A pump minimally repaired and a plain spare, in parallel."""
    return RepairableRBD(
        [("s", "pump"), ("s", "spare"), ("pump", "t"), ("spare", "t")],
        {
            "pump": pump,
            "spare": {
                "reliability": W([150.0, 1.5]),
                "repairability": E([0.1]),
                "repair_cost": 2.0,
            },
        },
        **options,
    )


def test_a_system_with_a_timed_unit_against_the_simulation():
    pump = minimal(W([100.0, 2.5]), L([2.0, 0.5]), repair_cost=1.0)
    rbd = pair(pump, downtime_cost_rate=5.0)
    horizon, n = 300.0, 4000
    sim = simulated(rbd, horizon, n, 3)
    grid = sim.timeline[1:-1]
    up = rbd.point_availability(grid)
    error = np.sqrt(up * (1.0 - up) / n)
    assert np.all(np.abs(sim.availability[1:-1] - up) < 4.0 * error + 1e-3)
    error = float(np.std(sim.uptimes)) / math.sqrt(n) / horizon
    assert (
        abs(sim.sample_mean_availability - rbd.mission_availability(horizon))
        < 4.0 * error
    )
    expected = rbd.expected_cost(horizon)
    error = float(np.std(sim.cost.samples)) / math.sqrt(n)
    assert abs(sim.cost.mean - expected.mean) < 4.0 * error
    assert sim.system_failures / n == pytest.approx(
        rbd.expected_events(horizon).system_failures, rel=0.1
    )


def test_importance_over_time_with_a_timed_unit():
    # In parallel, the spare's Birnbaum importance at t is the pump's
    # unavailability then: against the pump simulated alone.
    pump = minimal(W([100.0, 2.5]), L([2.0, 0.5]))
    rbd = pair(pump)
    alone = single(pump)
    n = 4000
    sim = simulated(alone, 300.0, n, 4)
    grid = sim.timeline[1:-1]
    importance = rbd.birnbaum_importance(x=grid)
    down = 1.0 - sim.availability[1:-1]
    error = np.sqrt(down * (1.0 - down) / n)
    assert np.all(np.abs(importance["spare"] - down) < 4.0 * error + 1e-3)
    # And the pump's is the spare's unavailability, from its renewal
    # equation: the exact structure at the curves.
    spare = single({"reliability": W([150.0, 1.5]), "repairability": E([0.1])})
    np.testing.assert_allclose(
        importance["pump"], 1.0 - spare.point_availability(grid), atol=1e-6
    )


# -- the routes and the refusals ---------------------------------------------


def test_the_routes_say_numerical_over_a_window_and_refuse_the_long_run():
    rbd = single(minimal(W([100.0, 2.0]), E([0.5]), repair_cost=1.0))
    report = rbd.analysis_routes()
    for name in (
        "point_availability",
        "mission_availability",
        "expected_failures",
        "expected_events",
        "expected_cost",
        "availability_rate",
    ):
        assert report[name].route == r.NUMERICAL, name
    route, how = _curves._node_over_time(rbd, "c")
    assert route == r.NUMERICAL and "operating age followed on a grid" in how
    for name in (
        "mean_availability",
        "expected_cost_rate",
        "birnbaum_importance",
    ):
        assert report[name].route == r.REFUSED, name
    assert "numerical, its operating age" in report["mean_availability"].reason
    with pytest.raises(NotImplementedError, match="operating age followed"):
        rbd.mean_availability()
    assert report["availability"].route == r.SIMULATED
    # And the run's means are the numerical values by default (#187).
    assert "need none" in report["availability"].reason


def test_the_other_blockers_still_refuse():
    for spec, blocker in [
        (
            minimal(
                W([100.0, 2.0]),
                E([0.5]),
                repair={"model": "kijima1", "q": 0.9},
            ),
            "q below 1",
        ),
        (
            minimal(W([100.0, 2.0]), E([0.5]), replace_after=3),
            "replaced at its failure 3",
        ),
        (
            minimal(W([100.0, 2.0]), E([0.5]), preventive={"interval": 50.0}),
            "preventive maintenance",
        ),
        (minimal(W([100.0, 2.0], f0=0.1), E([0.5])), "dead on arrival"),
    ]:
        rbd = single(spec)
        with pytest.raises(NotImplementedError, match=blocker):
            rbd.point_availability([50.0])
        assert rbd.analysis_routes()["expected_failures"].route == r.REFUSED


def test_too_long_a_window_is_refused():
    # A repair far shorter than the step costs nothing, but a window of
    # thousands of lives would need more grid points than are followed.
    rbd = single(minimal(W([1.0, 2.0]), E([100.0])))
    with pytest.raises(NotImplementedError, match="grid points"):
        rbd.point_availability([1e6])


def test_a_tested_minimally_repaired_unit_is_refused_as_the_route_says():
    # A component with hidden failures repaired minimally: the values over
    # time are refused (its failures are found only by its tests), as
    # analysis_routes says, by every method, with the route's message;
    # they were computed as if every repair renewed it. A run's default
    # means then fall back to the simulations' own, not a refused value.
    spec = {
        "reliability": W([10.0, 1.5]),
        "repairability": surv.LogNormal.from_params([1.5, 0.5]),
        "repair": {"model": "kijima1", "q": 1.0},
        "inspection": {"interval": 10.0},
    }
    rbd = RepairableRBD([("s", "A"), ("A", "t")], {"A": spec})
    report = rbd.analysis_routes()
    calls = {
        "point_availability": lambda: rbd.point_availability([5.0, 40.0]),
        "mission_availability": lambda: rbd.mission_availability(40.0),
        "expected_events": lambda: rbd.expected_events(40.0),
        "expected_failures": lambda: rbd.expected_failures(40.0),
    }
    for name, call in calls.items():
        assert report[name].route == r.REFUSED
        with pytest.raises(NotImplementedError) as error:
            call()
        assert str(error.value) == report[name].reason
        assert "found only by its tests" in report[name].reason
    result = rbd.availability(40.0, mc_samples=200, seed=3)
    assert result.mean_availability_interval().method != "exact"
    assert result.mean_availability == pytest.approx(
        result.sample_mean_availability, rel=1e-12
    )
    assert result.control_variate is None
    # The renewed unit (q = 0) is what was computed: its values differ.
    renewed = RepairableRBD(
        [("s", "A"), ("A", "t")],
        {"A": {**spec, "repair": {"model": "kijima1", "q": 0.0}}},
    )
    assert renewed.analysis_routes()["mission_availability"].route != (
        r.REFUSED
    )
    assert renewed.mission_availability(40.0) != pytest.approx(
        result.mean_availability, abs=0.02
    )
