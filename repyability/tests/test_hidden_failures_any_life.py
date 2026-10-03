"""Hidden failures with any life (#144): a component tested every
``interval``, in no time, and renewed, in no time, at the test that finds
it failed, with a life that is not exponential. The exact (numerical)
long-run and from-new values against references written from the
definition: the renewal-reward ratio of a cycle, and the units in service
followed test by test; and the simulation against them."""

import math

import numpy as np
import pytest
from scipy import integrate
from surpyval import Exponential, Gamma, LogNormal, Weibull

from repyability import NodeState, RepairableRBD
from repyability.rbd import _hidden_life, routes
from repyability.rbd._point_availability import InspectionCurve

E = Exponential.from_params
W = Weibull.from_params
SINGLE = [("s", "a"), ("a", "t")]
PAIR = [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")]

LIVES = {
    "wearing out": (W([200.0, 2.5]), 50.0),
    "infant mortality": (W([30.0, 0.7]), 50.0),
    "lognormal": (LogNormal.from_params([4.0, 0.5]), 20.0),
    "gamma": (Gamma.from_params([3.0, 0.05]), 10.0),
    # Its survival bends at 75, inside the second interval: that term is
    # summed directly rather than interpolated.
    "with a threshold": (W([200.0, 2.5], gamma=75.0), 50.0),
}


def hidden(model, interval, **spec):
    costs = {
        key: spec.pop(key)
        for key in ("repair_cost", "downtime_cost")
        if key in spec
    }
    return {
        "reliability": model,
        "repairability": "instant",
        "inspection": {"interval": interval, **spec},
        **costs,
    }


def sf(model, x):
    return np.asarray(model.sf(np.asarray(x, dtype=float)), dtype=float)


def cycle(model, interval):
    """The mean number of intervals in a cycle: a unit that fails in ``((k
    - 1) T, k T]`` (or at once) is found and renewed at the k-th test, so
    the cycle lasts ``k`` or more tests with chance ``R((k - 1) T)`` (and
    surely one): ``1 + sum_{k >= 1} R(k T)``, summed until ``R`` is
    nothing."""
    total, start = 1.0, 1
    while start <= 200_000:
        survive = sf(model, interval * np.arange(start, start + 4096))
        total += float(survive.sum())
        if survive[-1] < 1e-300:
            break
        start += 4096
    return total


def renewal_reward(model, interval):
    """Up for the life ``X`` of each cycle, of mean length ``T S``: the
    long-run availability and the failures per unit time."""
    length = interval * cycle(model, interval)
    return model.mean() / length, 1.0 / length


def profile(model, interval, u):
    """The long-run chance of being up ``u`` after a test, by its
    definition: renewed ``m`` intervals before with chance ``R(m T) / S``
    (``1 / S`` at the test), and up since."""
    m = np.arange(0, 20_000)
    weights = np.concatenate([[1.0], sf(model, interval * m[1:])])
    u = np.atleast_1d(np.asarray(u, dtype=float))
    up = sf(model, interval * m[:, None] + u[None, :])
    return up.sum(axis=0) / weights.sum()


class Units:
    """The units in service, followed test by test from the definition:
    each test finds the unit in service failed with the chance that it
    failed since it was last known up, and the renewal puts a new one in
    service. The unit in service at 0 is ``age`` old and was last known up
    at age ``known``; the tests are at ``first`` and every ``interval``
    after."""

    def __init__(self, model, interval, first, horizon, age=0.0, known=0.0):
        self.model, self.age, self.known = model, age, known
        prob, birth, last = [1.0], [-age], [known]
        self.tests, self.renewals = [], []
        self.snapshots = []
        t = first
        while t <= horizon:
            p, b, k = np.array(prob), np.array(birth), np.array(last)
            survive = sf(model, t - b) / sf(model, k)
            renewed = float(p @ (1.0 - survive))
            prob = list(p * survive) + [renewed]
            birth = list(b) + [t]
            last = list(t - b) + [0.0]
            self.tests.append(t)
            self.renewals.append(renewed)
            self.snapshots.append(
                (np.array(prob), np.array(birth), np.array(last))
            )
            t += interval

    def up(self, x):
        """The chance of being up at each ``x`` (a renewal at ``x``
        counted)."""
        out = []
        for value in np.atleast_1d(x):
            n = sum(t <= value for t in self.tests)
            if n == 0:
                up = sf(self.model, self.age + value) / sf(
                    self.model, self.known
                )
                out.append(float(up))
                continue
            p, b, k = self.snapshots[n - 1]
            out.append(
                float(p @ (sf(self.model, value - b) / sf(self.model, k)))
            )
        return np.array(out)

    def events(self, x):
        """Failures in ``[0, x)``, tests before ``x`` and the failures they
        found, for each ``x``."""
        failures, found, tests = [], [], []
        start = sf(self.model, self.age) / sf(self.model, self.known)
        for value in np.atleast_1d(x):
            before = [
                (t, r) for t, r in zip(self.tests, self.renewals) if t < value
            ]
            first = start - (
                sf(self.model, self.age + value) / sf(self.model, self.known)
            )
            later = sum(
                r * float(self.model.ff(np.array([value - t]))[0])
                for t, r in before
            )
            failures.append(float(first) + later)
            found.append(sum(r for _, r in before))
            tests.append(len(before))
        return np.array(failures), np.array(found), np.array(tests)


# -- the long run -------------------------------------------------------------


@pytest.mark.parametrize("name", list(LIVES))
def test_one_unit_by_renewal_reward(name):
    model, interval = LIVES[name]
    rbd = RepairableRBD(SINGLE, {"a": hidden(model, interval)})
    availability, failures = renewal_reward(model, interval)
    assert rbd.node_availability()["a"] == pytest.approx(
        availability, rel=1e-12
    )
    assert rbd.mean_availability() == pytest.approx(availability, rel=1e-12)
    assert rbd.mean_unavailability() == pytest.approx(
        1.0 - availability, rel=1e-11
    )
    assert rbd.system_failure_frequency() == pytest.approx(failures, rel=1e-12)


@pytest.mark.parametrize("name", list(LIVES))
def test_the_long_run_profile_by_its_definition(name):
    model, interval = LIVES[name]
    rbd = RepairableRBD(SINGLE, {"a": hidden(model, interval)})
    # At a test (the last time), the unit it finds failed is renewed.
    u = np.array([0.0, 1e-9, 0.3, 0.5, 0.77, 0.999, 1.0]) * interval
    stationary = {"a": NodeState(stationary=True)}
    np.testing.assert_allclose(
        rbd.point_availability(u, state=stationary),
        profile(model, interval, np.mod(u, interval)),
        rtol=1e-13,
        atol=1e-15,
    )
    # Its last test 0.3 intervals before 0.
    later = {"a": NodeState(stationary=True, phase=0.3 * interval)}
    np.testing.assert_allclose(
        rbd.point_availability(u, state=later),
        profile(model, interval, np.mod(u + 0.3 * interval, interval)),
        rtol=1e-13,
        atol=1e-15,
    )


def test_a_threshold_is_summed_directly():
    model, interval = LIVES["with a threshold"]
    life = _hidden_life.TestedLife(model, interval)
    table, rough = life.smooth()
    assert list(rough) == [1]
    assert not table[0].any()
    assert not list(_hidden_life.TestedLife(W([200.0, 2.5]), 50.0).smooth()[1])


def test_the_exponential_case_is_the_closed_form():
    # The module with a constant rate: exp(-lambda u) after a test; a mean
    # unavailability of 1 - (1 - exp(-lambda T)) / (lambda T), about
    # lambda T / 2 for a small lambda T (the PFDavg).
    rate, interval = 2e-3, 50.0
    life = _hidden_life.TestedLife(E([rate]), interval)
    u = np.linspace(0.0, interval, 11)
    up, down = life.profile(u)
    np.testing.assert_allclose(up, np.exp(-rate * u), rtol=1e-14)
    np.testing.assert_allclose(down, -np.expm1(-rate * u), rtol=1e-12)
    x = rate * interval
    assert life.unavailability == pytest.approx(
        1.0 + math.expm1(-x) / x, rel=1e-12
    )
    assert life.unavailability == pytest.approx(x / 2, rel=0.04)
    assert life.failures == pytest.approx(-math.expm1(-x) / interval)
    np.testing.assert_allclose(
        life.intensity(u), rate * np.exp(-rate * u), rtol=1e-12
    )
    # From new, and in the long run, as the closed form's curve.
    times = np.array([0.0, 10.0, 50.0, 75.0, 333.0, 2e4 + 7.0])
    new = _hidden_life.TestedLifeCurve(life, interval)
    closed = InspectionCurve(rate, interval)
    np.testing.assert_allclose(new.at(times), closed.at(times), rtol=1e-13)
    for key in ("failures", "corrective", "inspections"):
        np.testing.assert_allclose(
            new.events(times)[key],
            closed.events(times)[key],
            rtol=1e-12,
            atol=1e-15,
        )
    steady = _hidden_life.TestedLifeSteady(life, 20.0)
    np.testing.assert_allclose(
        steady.at(times), InspectionCurve(rate, interval, 20.0).at(times)
    )


def test_a_pair_by_integration():
    # A 1oo2 pair of different wearing lives, tested together or staggered:
    # the system is down when both are, and fails when one fails while the
    # other is down.
    a, b, interval = W([200.0, 2.5]), W([150.0, 1.7]), 50.0

    def down(model, u):
        return 1.0 - profile(model, interval, u)[0]

    def rate(model, u):
        m = np.arange(0, 2000)
        weights = np.concatenate([[1.0], sf(model, interval * m[1:])])
        return float(model.df(interval * m + u).sum() / weights.sum())

    for offset in (0.0, 25.0):
        rbd = RepairableRBD(
            PAIR,
            {
                "a": hidden(a, interval),
                "b": hidden(b, interval, offset=offset),
            },
        )

        def both(u):
            return down(a, u) * down(b, (u - offset) % interval)

        def fails(u):
            other = (u - offset) % interval
            return rate(a, u) * down(b, other) + rate(b, other) * down(a, u)

        breaks = sorted({0.0, offset, interval})
        pieces = list(zip(breaks[:-1], breaks[1:]))
        unavailable = sum(
            integrate.quad(both, lo, hi, epsabs=0, epsrel=1e-12)[0]
            for lo, hi in pieces
        )
        frequency = sum(
            integrate.quad(fails, lo, hi, epsabs=0, epsrel=1e-10)[0]
            for lo, hi in pieces
        )
        assert rbd.mean_unavailability() == pytest.approx(
            unavailable / interval, rel=1e-11
        )
        assert rbd.system_failure_frequency() == pytest.approx(
            frequency / interval, rel=1e-9
        )
        # In a 1oo2 pair a unit is critical while the other is down.
        assert rbd.birnbaum_importance()["a"] == pytest.approx(
            1.0 - rbd.node_availability()["b"], rel=1e-11
        )


def test_the_cost_rate():
    model, interval = W([200.0, 2.5]), 50.0
    rbd = RepairableRBD(
        SINGLE,
        {
            "a": hidden(
                model,
                interval,
                cost=30.0,
                repair_cost=500.0,
                downtime_cost=4.0,
            )
        },
        downtime_cost_rate=10.0,
    )
    availability, failures = renewal_reward(model, interval)
    want = 30.0 / interval + 500.0 * failures + 14.0 * (1.0 - availability)
    assert rbd.expected_cost_rate() == pytest.approx(want, rel=1e-11)


def test_the_best_test_interval_for_a_wearing_life():
    # The cost rate c / T + d (1 - E[X] / (T S(T))), by brute force.
    model = W([400.0, 2.0])
    c, d = 20.0, 5.0
    rbd = RepairableRBD(
        SINGLE,
        {"a": hidden(model, 100.0, cost=c, downtime_cost=d)},
    )
    plan = rbd.optimal_inspection_intervals()

    def cost_rate(interval):
        availability, _ = renewal_reward(model, interval)
        return c / interval + d * (1.0 - availability)

    grid = np.linspace(20.0, 200.0, 1801)
    best = grid[np.argmin([cost_rate(t) for t in grid])]
    assert plan.intervals["a"] == pytest.approx(best, rel=2e-3)
    assert plan.cost_rate == pytest.approx(cost_rate(best), rel=1e-6)
    assert plan.cost_rate == pytest.approx(
        rbd._with_intervals(
            inspection={"a": plan.intervals["a"]}
        ).expected_cost_rate(),
        rel=1e-12,
    )


# -- from new, and from a state ----------------------------------------------


@pytest.mark.parametrize("name", ["wearing out", "infant mortality"])
@pytest.mark.parametrize("offset", [0.0, 20.0])
def test_from_new_by_the_units_in_service(name, offset):
    model, interval = LIVES[name]
    rbd = RepairableRBD(SINGLE, {"a": hidden(model, interval, offset=offset)})
    first = offset or interval
    reference = Units(model, interval, first, horizon=40 * interval)
    x = np.array(
        [3.0, first, first + 0.4 * interval, 3 * interval + offset]
        + list(np.linspace(0.0, 35 * interval, 23)[1:])
    )
    np.testing.assert_allclose(
        rbd.point_availability(x), reference.up(x), rtol=1e-12, atol=1e-15
    )
    failures, found, tests = reference.events(x)
    events = rbd.expected_events(x)
    np.testing.assert_allclose(
        events.node_failures["a"], failures, rtol=1e-11, atol=1e-14
    )
    np.testing.assert_allclose(
        events.node_corrective["a"], found, rtol=1e-11, atol=1e-14
    )
    np.testing.assert_array_equal(events.node_inspections["a"], tests)
    # One unit alone: the system fails when it does.
    np.testing.assert_allclose(
        events.system_failures, failures, rtol=1e-9, atol=1e-12
    )
    # Long after, the long-run profile.
    late = 1e5 * interval + offset + np.array([0.0, 0.25, 0.6]) * interval
    np.testing.assert_allclose(
        rbd.point_availability(late),
        profile(model, interval, np.array([0.0, 0.25, 0.6]) * interval),
        rtol=1e-12,
    )


def test_the_mission_availability_from_new():
    model, interval = W([200.0, 2.5]), 50.0
    rbd = RepairableRBD(SINGLE, {"a": hidden(model, interval)})
    reference = Units(model, interval, interval, horizon=12 * interval)
    window = 10.5 * interval
    edges = np.linspace(0.0, window, 22)
    uptime = sum(
        integrate.quad(
            lambda t: reference.up(t)[0], lo, hi, epsabs=0, epsrel=1e-12
        )[0]
        for lo, hi in zip(edges[:-1], edges[1:])
    )
    assert rbd.mission_availability(window) == pytest.approx(
        uptime / window, rel=1e-9
    )


@pytest.mark.parametrize(
    "age, phase",
    [(130.0, 20.0), (8.0, 20.0), (0.0, 35.0), (400.0, 0.0)],
)
def test_from_a_state(age, phase):
    # Found working at its last test, ``phase`` ago, at age ``age - phase``;
    # or, younger than that, put into service since, and not yet tested.
    model, interval = W([200.0, 2.5]), 50.0
    rbd = RepairableRBD(SINGLE, {"a": hidden(model, interval)})
    state = {"a": NodeState(age=age, phase=phase)}
    known = max(age - phase, 0.0)
    first = interval - phase if phase else interval
    reference = Units(
        model, interval, first, horizon=30 * interval, age=age, known=known
    )
    x = np.array([0.0, 5.0, first, first + 1.0, 4 * interval + 3.0, 1111.0])
    np.testing.assert_allclose(
        rbd.point_availability(x, state=state),
        reference.up(x),
        rtol=1e-12,
        atol=1e-15,
    )
    failures, found, tests = reference.events(x)
    events = rbd.expected_events(x, state=state)
    np.testing.assert_allclose(
        events.node_failures["a"], failures, rtol=1e-11, atol=1e-14
    )
    np.testing.assert_allclose(
        events.node_corrective["a"], found, rtol=1e-11, atol=1e-14
    )
    np.testing.assert_array_equal(events.node_inspections["a"], tests)


def test_a_long_life_against_its_interval():
    # A weekly test of a life of some 20 years: thousands of intervals to
    # follow, quickly, and the values still exact.
    model, interval = W([1000.0, 1.5]), 1.0
    rbd = RepairableRBD(SINGLE, {"a": hidden(model, interval)})
    availability, failures = renewal_reward(model, interval)
    assert rbd.mean_availability() == pytest.approx(availability, rel=1e-12)
    reference = Units(model, interval, interval, horizon=60.0)
    x = np.array([0.5, 7.25, 31.0, 59.9])
    np.testing.assert_allclose(
        rbd.point_availability(x), reference.up(x), rtol=1e-12
    )
    late = np.array([1e7 + 0.5])
    np.testing.assert_allclose(
        rbd.point_availability(late),
        profile(model, interval, np.array([0.5])),
        rtol=1e-12,
    )
    assert rbd.expected_failures(20_000.0) == pytest.approx(
        rbd.expected_events(20_000.0).node_failures["a"], rel=1e-9
    )


# -- what stays simulated -----------------------------------------------------


def test_what_stays_simulated():
    # Tests that miss failures, and tests and repairs that take time, are
    # numerical (#159, see test_hidden_failures_timed.py); a test that can
    # last a whole interval, or a life with units dead on arrival, are not.
    model = W([200.0, 2.5])
    for rbd in (
        RepairableRBD(
            SINGLE,
            {"a": hidden(model, 50.0, coverage=0.8, full_test=200.0)},
        ),
        RepairableRBD(SINGLE, {"a": hidden(model, 50.0, duration=E([1.0]))}),
        RepairableRBD(
            SINGLE,
            {"a": {**hidden(model, 50.0), "repairability": E([1.0])}},
        ),
    ):
        assert 0.0 < rbd.mean_availability() < 1.0
        assert 0.0 < float(np.ravel(rbd.point_availability(10.0))[0]) <= 1.0
    slow = RepairableRBD(
        SINGLE, {"a": hidden(model, 50.0, duration=E([0.01]))}
    )
    with pytest.raises(NotImplementedError, match="within its test interval"):
        slow.mean_availability()
    with pytest.raises(NotImplementedError, match="within its test interval"):
        slow.point_availability(10.0)


def test_the_routes():
    rbd = RepairableRBD(SINGLE, {"a": hidden(W([200.0, 2.5]), 50.0)})
    report = rbd.analysis_routes()
    for name in (
        "mean_availability",
        "node_availability",
        "system_failure_frequency",
        "birnbaum_importance",
    ):
        assert report[name].route == routes.NUMERICAL
        assert report[name].nodes == ("a",)
        assert "renewed at the tests" in report[name].reason
    for name in ("point_availability", "expected_events"):
        assert report[name].route == routes.NUMERICAL
    # Tests that miss failures: the cycle followed test by test (#159).
    missing = RepairableRBD(
        SINGLE,
        {"a": hidden(W([200.0, 2.5]), 50.0, coverage=0.8, full_test=200.0)},
    )
    route = missing.analysis_routes()["mean_availability"]
    assert route.route == routes.NUMERICAL
    assert "followed test by test" in route.reason
    # Tests that can last a whole interval stay simulated.
    slow = RepairableRBD(
        SINGLE, {"a": hidden(W([200.0, 2.5]), 50.0, duration=E([0.01]))}
    )
    route = slow.analysis_routes()["mean_availability"]
    assert route.route == routes.REFUSED
    with pytest.raises(NotImplementedError) as error:
        slow.mean_availability()
    assert str(error.value) == route.reason


# -- the simulation -----------------------------------------------------------


def test_the_simulation_agrees():
    a, b, interval = W([80.0, 1.8]), W([120.0, 1.3]), 50.0
    rbd = RepairableRBD(
        PAIR,
        {
            "a": hidden(a, interval, repair_cost=10.0),
            "b": hidden(b, interval, offset=20.0, repair_cost=10.0),
        },
        downtime_cost_rate=100.0,
    )
    window = 500.0
    result = rbd.availability(window, mc_samples=4000, seed=11)
    n = result.n_simulations
    events = rbd.expected_events(window)
    assert result.system_failures / n == pytest.approx(
        events.system_failures, rel=0.04
    )
    assert result.system_downtime / n == pytest.approx(
        events.system_downtime, rel=0.04
    )
    for node in "ab":
        assert result.node_downtime[node] / n == pytest.approx(
            events.node_downtime[node], rel=0.03
        )
    assert result.cost.mean == pytest.approx(
        rbd.expected_cost(window).mean, rel=0.03
    )
