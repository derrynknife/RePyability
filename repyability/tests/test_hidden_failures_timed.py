"""Hidden failures whose tests or repairs take time, or whose tests miss
them, for any life (#159): the cycle from one test that finds a failure to
the next, followed test by test on a grid (``_hidden_tests``), or the chain
of an exponential life's states from test to test. Checked against the sums
of #144 and the closed forms where tests and repairs take no time, the walk
against the chain for an exponential life, the long run against the values
from new once they settle, and everything against the simulation."""

import math

import numpy as np
import pytest
import surpyval as surv

from repyability import (
    BetaFactor,
    CCFGroup,
    NodeState,
    RepairableRBD,
)
from repyability.rbd import _hidden_life
from repyability.rbd import _hidden_tests as ht
from repyability.rbd import _long_run, routes
from repyability.rbd._point_availability import (
    InspectionCurve,
    PartialTestCurve,
)

W, E, LN, X = (
    surv.Weibull.from_params,
    surv.Exponential.from_params,
    surv.LogNormal.from_params,
    surv.ExactEventTime.from_params,
)
INSTANT = X([0.0])
SINGLE = [("s", "c"), ("c", "t")]


def hidden(life, repair, duration=None, interval=1.0, **inspection):
    spec = {"interval": interval, "cost": 1.0, **inspection}
    if duration is not None:
        spec["duration"] = duration
    return {
        "reliability": life,
        "repairability": repair,
        "inspection": spec,
        "replace_cost": 1.0,
    }


def single(spec) -> RepairableRBD:
    return RepairableRBD(SINGLE, {"c": spec})


# -- where tests and repairs take no time ------------------------------------


def test_the_cycle_walk_is_the_sums_over_the_test_intervals():
    life = W([6.0, 2.5])
    unit = ht.TestedUnit(life, INSTANT, None, 1.0)
    sums = _hidden_life.TestedLife(life, 1.0)
    long_run = unit.long_run
    assert long_run.availability == pytest.approx(sums.availability, abs=1e-12)
    assert long_run.unavailability == pytest.approx(
        sums.unavailability, rel=1e-10
    )
    assert long_run.failures == pytest.approx(sums.failures, rel=1e-12)
    phase = np.linspace(0.0, 1.0, 11)
    np.testing.assert_allclose(
        long_run.profile(phase)[0], sums.profile(phase)[0], atol=1e-10
    )
    # Away from the tests themselves (where the sums place a time a
    # rounding before its test, as the simulation does not).
    x = np.linspace(0.0, 40.0, 4001) + 1e-7
    for curve, reference in (
        (unit.curve(40.0, 0.4, 0), _hidden_life.TestedLifeCurve(sums, 0.4)),
        (unit.curve(np.inf, 0.4, 0), _hidden_life.TestedLifeCurve(sums, 0.4)),
        (
            unit.curve(40.0, -0.3, 1, ("stationary",)),
            _hidden_life.TestedLifeSteady(sums, 0.3),
        ),
    ):
        np.testing.assert_allclose(curve.at(x), reference.at(x), atol=1e-10)
        ours, theirs = curve.events(x), reference.events(x)
        for key in ("failures", "corrective", "inspections"):
            np.testing.assert_allclose(ours[key], theirs[key], atol=1e-10)
    # Up at age 3, known up at its last test 0.3 before 0.
    known = 3.0 - 0.3

    def initial(t):
        return life.sf(3.0 + np.asarray(t)) / life.sf(known)

    def initial_down(t):
        failed = life.ff(3.0 + np.asarray(t)) - life.ff(known)
        return np.clip(failed / life.sf(known), 0.0, 1.0)

    curve = unit.curve(40.0, -0.3, 1, ("alive", 3.0, 0.3))
    reference = _hidden_life.TestedLifeCurve(sums, 0.7, initial, initial_down)
    np.testing.assert_allclose(curve.at(x), reference.at(x), atol=1e-10)
    ours, theirs = curve.events(x), reference.events(x)
    for key in ("failures", "corrective", "inspections"):
        np.testing.assert_allclose(ours[key], theirs[key], atol=1e-10)


@pytest.mark.parametrize("walk", [False, True], ids=["chain", "walk"])
def test_an_exponential_life_takes_the_closed_forms(walk):
    rate, T = 0.3, 1.0
    life = E([rate])
    kw = {} if walk else {"rate": rate}
    full = ht.TestedUnit(life, INSTANT, None, T, **kw)
    exact = -math.expm1(-rate * T) / (rate * T)
    assert full.long_run.availability == pytest.approx(exact, rel=1e-8)
    x = np.linspace(0.0, 20.0, 2001) + 1e-7
    curve = full.curve(20.0, 0.4, 0)
    reference = InspectionCurve(rate, T, T - 0.4, 0.0)
    np.testing.assert_allclose(curve.at(x), reference.at(x), atol=1e-10)
    # Tests that miss: rho ** k exp(-rate u), k tests since a full one.
    c, per = 0.6, 3
    partial = ht.TestedUnit(life, INSTANT, None, T, c, per, **kw)
    kept = math.log1p(-(1 - c) * -math.expm1(-rate * T))
    exact = -math.expm1(per * kept) / ((1 - c) * rate * per * T)
    assert partial.long_run.availability == pytest.approx(exact, rel=1e-8)
    for base, k0, offset in ((0.4, 0, 0.4), (0.0, 1, 0.0)):
        curve = partial.curve(20.0, base, k0)
        reference = PartialTestCurve(rate, T, offset, c, per)
        np.testing.assert_allclose(curve.at(x), reference.at(x), atol=1e-10)
        ours, theirs = curve.events(x), reference.events(x)
        for key in ("failures", "corrective", "inspections"):
            np.testing.assert_allclose(ours[key], theirs[key], atol=1e-10)


def test_a_fixed_test_and_an_instant_repair_in_closed_form():
    # Every unit is new when its test is over: up exp(-rate (s - D)).
    rate, T, D = 0.05, 1.0, 0.1
    unit = ht.TestedUnit(E([rate]), INSTANT, X([D]), T, rate=rate)
    kept = math.exp(-rate * (T - D))
    long_run = unit.long_run
    assert long_run.availability == pytest.approx(
        (1 - kept) / (rate * T), rel=1e-8
    )
    assert long_run.failures == pytest.approx((1 - kept) / T, rel=1e-12)
    assert long_run.planned == pytest.approx(kept / T, rel=1e-12)
    s = np.array([0.05, 0.1, 0.5, 0.99])
    up = np.where(s >= D, np.exp(-rate * (s - D)), 0.0)
    np.testing.assert_allclose(long_run.profile(s)[0], up, atol=1e-12)


# -- the walk, the chain and the long run -------------------------------------


@pytest.mark.parametrize(
    "repair, duration",
    [
        (E([20.0]), X([0.02])),
        (X([0.3]), None),
        (LN([np.log(0.1), 0.5]), E([50.0])),
        (X([1.65]), X([0.05])),
    ],
    ids=["fixed test", "fixed repair", "both spread", "repair over a test"],
)
@pytest.mark.parametrize("coverage, per", [(1.0, 1), (0.7, 3)])
def test_the_walk_is_the_chain(repair, duration, coverage, per):
    rate = 0.05
    args = (E([rate]), repair, duration, 1.0, coverage, per)
    chain = ht.TestedUnit(*args, rate=rate).long_run
    walk = ht.TestedUnit(*args).long_run
    for name in (
        "availability",
        "unavailability",
        "failures",
        "inspections",
        "planned",
        "corrective",
    ):
        assert getattr(walk, name) == pytest.approx(
            getattr(chain, name), rel=1e-11, abs=1e-14
        )
    phase = np.linspace(0.0, per * 1.0, 61)[:-1] + 0.0123
    for ours, theirs in zip(walk.profile(phase), chain.profile(phase)):
        np.testing.assert_allclose(ours, theirs, atol=1e-12)
    # Up and down add up, each worked out in its own right.
    up, down = chain.profile(phase)
    np.testing.assert_allclose(up + down, 1.0, atol=1e-12)


@pytest.mark.parametrize(
    "spec",
    [
        dict(
            life=W([3.0, 1.5]),
            repair=X([0.4]),
            test=E([30.0]),
            coverage=0.5,
            per=3,
        ),
        dict(
            life=W([4.0, 2.0]), repair=LN([np.log(0.2), 0.6]), test=X([0.05])
        ),
        dict(life=W([20.0, 4.0]), repair=X([0.1]), test=X([0.02])),
        dict(
            life=E([0.3]),
            repair=LN([np.log(0.3), 0.5]),
            test=LN([np.log(0.05), 0.3]),
            coverage=0.6,
            per=2,
            rate=0.3,
        ),
    ],
    ids=["missing", "timed", "wearing out", "exponential, missing"],
)
def test_from_new_it_settles_into_its_long_run(spec):
    unit = ht.TestedUnit(interval=1.0, **spec)
    long_run = unit.long_run
    for base, k0 in ((0.3, 0), (0.0, 1)):
        curve = unit.curve(np.inf, base, k0)
        assert curve.settled and curve.period == long_run.period
        ends = np.array([curve.settle, curve.settle + curve.period])
        events = curve.events(ends)
        for key, rate in (
            ("failures", long_run.failures),
            ("inspections", long_run.inspections),
            ("planned", long_run.planned),
            ("corrective", long_run.corrective),
        ):
            assert np.diff(events[key])[0] / curve.period == pytest.approx(
                rate, rel=1e-10, abs=1e-14
            )


@pytest.mark.parametrize("nudge", [0.0, 2e-15, 1e-14])
def test_it_settles_with_its_long_run_a_few_roundings_off(monkeypatch, nudge):
    # The state iterated from new comes within a few roundings of the
    # stationary one, how few depending on the platform's arithmetic: a bound
    # at the rounding was never met on Python 3.12's, and the curve refused.
    def unit():
        return ht.TestedUnit(
            interval=1.0,
            life=E([0.3]),
            repair=LN([np.log(0.3), 0.5]),
            test=LN([np.log(0.05), 0.3]),
            coverage=0.6,
            per=2,
            rate=0.3,
        )

    exact = unit().curve(np.inf, 0.0, 1)
    stationary = ht._Chain.stationary
    monkeypatch.setattr(
        ht._Chain, "stationary", lambda chain: stationary(chain) + nudge
    )
    curve = unit().curve(np.inf, 0.0, 1)
    assert curve.settled and curve.settle == exact.settle
    x = np.linspace(0.0, 2 * curve.settle, 201)
    assert np.allclose(curve.at(x), exact.at(x), rtol=0.0, atol=1e-11)


CASES = {
    "timed, offset": (
        hidden(W([4.0, 2.0]), LN([np.log(0.2), 0.6]), X([0.05]), offset=0.3),
        None,
    ),
    "missing": (
        hidden(
            W([3.0, 1.5]), X([0.4]), E([30.0]), coverage=0.5, full_test=3.0
        ),
        None,
    ),
    "in a repair": (
        hidden(W([4.0, 2.0]), E([5.0]), X([0.05])),
        NodeState(alive=False, down_for=0.1, since_scheduled=0.6),
    ),
    "up at an age": (
        hidden(W([4.0, 2.0]), E([5.0]), X([0.05])),
        NodeState(age=2.0, since_scheduled=0.4),
    ),
    "exponential, missing": (
        hidden(
            E([0.3]),
            LN([np.log(0.3), 0.5]),
            LN([np.log(0.05), 0.3]),
            offset=0.5,
            coverage=0.6,
            full_test=2.0,
        ),
        None,
    ),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_against_the_simulation(name):
    spec, state = CASES[name]
    rbd = single(spec)
    kw = {} if state is None else {"state": {"c": state}}
    horizon, n = 10.0, 8_000
    mission = float(np.ravel(rbd.mission_availability(horizon, **kw))[0])
    events = rbd.expected_events(horizon, **kw)
    simulated = rbd.availability(
        horizon, mc_samples=n, seed=7, control_variate=False, **kw
    )
    up = np.asarray(simulated.uptimes, dtype=float).ravel() / horizon
    assert mission == pytest.approx(up.mean(), abs=4 * up.std() / np.sqrt(n))
    failures = float(np.ravel(events.system_failures)[0])
    counted = simulated.system_failures / n
    assert failures == pytest.approx(counted, abs=4 * np.sqrt(failures / n))
    planned = float(np.ravel(events.system_planned_outages)[0])
    counted = simulated.system_planned_outages / n
    assert planned == pytest.approx(counted, abs=4 * np.sqrt(planned / n))
    cost = simulated.cost.by_category
    tests = float(np.ravel(events.node_inspections["c"])[0])
    assert tests == pytest.approx(np.mean(cost["inspection"]), abs=0.02)
    found = float(np.ravel(events.node_corrective["c"])[0])
    assert found == pytest.approx(
        np.mean(cost["replace"]), abs=4 * np.sqrt(found / n)
    )


def test_spares_with_tests_that_take_time():
    rbd = single(hidden(W([4.0, 2.0]), LN([np.log(0.2), 0.6]), X([0.05])))
    exact = rbd.spares_demand(10.0)["c"]
    simulated = rbd.spares_demand(
        10.0, method="simulate", mc_samples=8_000, seed=3
    )["c"]
    assert exact.mean == pytest.approx(simulated.mean, abs=0.05)
    np.testing.assert_allclose(
        exact.probabilities[:5], simulated.probabilities[:5], atol=0.02
    )
    rbd.spares_stock(2.0, stockout_probability=0.05)
    # Tests that miss: where a cycle ends depends on where it began, between
    # the full tests (a Markov renewal process over those places).
    missing = single(
        hidden(W([3.0, 1.5]), X([0.4]), coverage=0.5, full_test=3.0)
    )
    exact = missing.spares_demand(10.0)["c"]
    simulated = missing.spares_demand(
        10.0, method="simulate", mc_samples=8_000, seed=3
    )["c"]
    assert exact.mean == pytest.approx(simulated.mean, abs=0.05)
    np.testing.assert_allclose(
        exact.probabilities[:5], simulated.probabilities[:5], atol=0.02
    )
    missing.spares_stock(2.0, stockout_probability=0.05)


# -- systems ----------------------------------------------------------------


def valve(offset=0.0):
    return hidden(
        E([2e-6]),
        LN([np.log(24), 0.5]),
        W([4, 3]),
        interval=8760.0,
        offset=offset,
    )


def test_tests_at_once_take_the_function_off_line():
    pair = [("s", "v1"), ("s", "v2"), ("v1", "t"), ("v2", "t")]
    both = RepairableRBD(pair, {"v1": valve(), "v2": valve()})
    apart = RepairableRBD(pair, {"v1": valve(), "v2": valve(4380.0)})
    assert both.mean_unavailability() == pytest.approx(4.28047e-4, rel=1e-5)
    assert apart.mean_unavailability() == pytest.approx(7.11935e-5, rel=1e-5)
    # Each test of both takes the function down, unless one is failed.
    planned = _long_run._outage_frequencies(both)[1] * 8760.0
    assert planned == pytest.approx(0.9997, abs=1e-4)
    assert _long_run._outage_frequencies(apart)[1] * 8760.0 < 0.02


def test_a_system_s_long_run_is_its_settled_values():
    rbd = RepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "c"), ("b", "c"), ("c", "t")],
        {
            "a": hidden(W([4.0, 2.0]), LN([np.log(0.2), 0.6]), X([0.05])),
            "b": hidden(
                W([4.0, 2.0]), LN([np.log(0.2), 0.6]), X([0.05]), offset=0.5
            ),
            "c": hidden(
                E([0.05]), X([0.1]), E([40.0]), coverage=0.6, full_test=2.0
            ),
        },
    )
    availability = rbd.mean_availability()
    assert availability + rbd.mean_unavailability() == pytest.approx(
        1.0, abs=1e-12
    )
    failures, planned = _long_run._outage_frequencies(rbd)
    ends = np.array([60.0, 80.0])
    mission = np.ravel(rbd.mission_availability(ends))
    late = (mission[1] * ends[1] - mission[0] * ends[0]) / 20.0
    assert late == pytest.approx(availability, abs=1e-10)
    events = rbd.expected_events(ends)
    assert np.diff(np.ravel(events.system_failures))[0] / 20.0 == (
        pytest.approx(failures, rel=1e-9)
    )
    assert np.diff(np.ravel(events.system_planned_outages))[0] / 20.0 == (
        pytest.approx(planned, rel=1e-9)
    )
    cost = np.ravel(rbd.expected_cost(ends).total)
    assert (cost[1] - cost[0]) / 20.0 == pytest.approx(
        rbd.expected_cost_rate(), rel=1e-8
    )


def test_the_interval_is_chosen():
    spec = hidden(W([50.0, 1.8]), X([0.02]), X([0.01]), cost=20.0)
    spec.update(replace_cost=300.0, downtime_cost=2000.0)
    rbd = single(spec)
    plan = rbd.optimal_inspection_intervals()
    best = plan.intervals["c"]
    for interval in (0.8 * best, 1.25 * best):
        worse = rbd.with_intervals({"c": interval}).expected_cost_rate()
        assert worse > plan.cost_rate
    chosen = rbd.optimal_inspection_intervals(allowed=[0.5, 1.0, 2.0])
    assert chosen.intervals == {"c": 1.0}


# -- what stays simulated ----------------------------------------------------


def test_what_stays_simulated():
    # A test that can last as long as its interval.
    slow = single(hidden(W([4.0, 2.0]), INSTANT, E([0.5])))
    route = slow.analysis_routes()["mean_availability"]
    assert route.route == routes.REFUSED
    with pytest.raises(NotImplementedError) as error:
        slow.mean_availability()
    assert str(error.value) == route.reason
    assert "within its test interval" in route.reason
    # A common-cause group's chain takes its members' tests of a fixed
    # length too (#220): with no shared cause, as the members' own models.
    pair = [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")]
    member = hidden(E([0.01]), INSTANT, X([0.1]), interval=10.0)
    grouped = RepairableRBD(
        pair,
        {"a": member, "b": member},
        ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.0))],
    )
    plain = RepairableRBD(pair, {"a": member, "b": member})
    assert grouped.mean_availability() == pytest.approx(
        plain.mean_availability(), rel=1e-6
    )


def test_the_routes():
    rbd = single(hidden(W([4.0, 2.0]), E([5.0]), X([0.05])))
    report = rbd.analysis_routes()
    for name in ("mean_availability", "node_availability"):
        assert report[name].route == routes.NUMERICAL
        assert "followed test by test" in report[name].reason
    for name in ("point_availability", "expected_events", "expected_cost"):
        assert report[name].route == routes.NUMERICAL
    assert report["spares_demand"].route == routes.NUMERICAL


# -- an exponential life's failures within a cell of the grid ---------------


@pytest.mark.parametrize("test_time", [None, 6.0])
def test_an_exponential_unit_fails_at_its_rate_while_it_is_up(test_time):
    # A repair of a few cells of the grid: the unit's availability bends
    # within them, and so do its failures. In the long run, with W working
    # at each test and K back from a repair by s after it (were the repair
    # to start at once), it is up with W e^(-rate u) + K mu (e^(-rate u) -
    # e^(-mu u)) / (mu - rate), u the time since the test was over.
    rate, mu, interval = 2e-5, 1.0 / 24.0, 8760.0
    test = None if test_time is None else X([test_time])
    unit = ht.TestedUnit(E([rate]), E([mu]), test, interval, rate=rate)
    long_run = unit.long_run
    rows, h = long_run.rows, unit.setup.h
    assert 2.0 * h < 1.0 / mu < 10.0 * h
    lags = np.arange(rows.coef.shape[1])
    back = float(rows.coef[0] @ np.exp(-mu * lags * interval))
    s = np.concatenate(
        [
            np.linspace(0.0, 40.0 * h, 1601),
            np.linspace(40.0 * h, interval, 3001),
        ]
    )
    u = np.maximum(s - (test_time or 0.0), 0.0)
    over = s >= (test_time or 0.0)
    exact = over * (
        rows.tested[0] * np.exp(-rate * u)
        + back * mu * (np.exp(-rate * u) - np.exp(-mu * u)) / (mu - rate)
    )
    # (Failures linear across each cell were 2.9e-7 off, and 1.7e-5 in
    # the cell the test ends in.)
    np.testing.assert_allclose(long_run.profile(s)[0], exact, atol=5e-9)
    np.testing.assert_allclose(
        long_run.intensity(s), rate * long_run.profile(s)[0], rtol=1e-13
    )


def test_units_down_in_the_same_repairs_fail_the_system_as_over_time():
    # Two units tested together, each in its repair just after a test
    # with the same chance: the system fails where one fails in the
    # other's repair, which bends within the grid's cells.
    spec = hidden(E([2e-5]), E([1.0 / 24.0]), interval=8760.0)
    rbd = RepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        {"a": spec, "b": dict(spec)},
    )
    frequency = rbd.system_failure_frequency()
    ends = 8760.0 * np.array([40.0, 44.0])
    counted = np.diff(np.ravel(rbd.expected_failures(ends)))[0]
    # (5.7e-8 apart with the entries in a cell taken midway; 6.4e-10.)
    assert counted / (4.0 * 8760.0) == pytest.approx(frequency, rel=1e-8)


# -- any other life's failures within a cell of the grid ---------------------


def _weibull_series(eta, beta, interval, mean_repair, s):
    """A Weibull unit tested at once every ``interval`` and repaired in an
    exponential time, in the long run, written out: a unit found failed
    is back ``V`` after its test, is working at the ``k``-th test after
    with ``E R(k T - V)`` and has run for its mean life when it fails, so
    a cycle is ``N = 1 + sum_k E R(k T - V)`` tests long. ``s`` after a
    test it is up with ``sum_k E R(k T + s - V) / N`` (the units back by
    then) and failing at the rate ``sum_k E f(k T + s - V) / N``. The
    expectations are by Gauss-Laguerre quadrature, but the units back
    since the last test, whose ages start at 0: by adaptive quadrature,
    with the density's power of the age as a weight."""
    from scipy import integrate

    def survival(age):
        return np.exp(-((np.maximum(age, 0.0) / eta) ** beta))

    def density(age):
        return beta / eta * (age / eta) ** (beta - 1.0) * survival(age)

    nodes, weights = np.polynomial.laguerre.laggauss(80)
    back = mean_repair * nodes
    k = np.arange(1, int(40.0 ** (1.0 / beta) * eta / interval) + 4)

    def later(function, since):
        ages = k[:, None] * interval + since - back[None, :]
        return float((function(ages) @ weights).sum())

    tests = 1.0 + later(survival, 0.0)
    up, failing = [], []
    for since in s:

        def entered(age):
            return math.exp(-(since - age) / mean_repair) / mean_repair

        closing = [since * 2.0**-j for j in range(20, 0, -1)]
        first = integrate.quad(
            lambda age: float(survival(age)) * entered(age),
            0.0,
            since,
            points=closing,
            limit=400,
            epsabs=0.0,
            epsrel=1e-13,
        )[0]
        new = integrate.quad(
            lambda age: beta / eta**beta * float(survival(age)) * entered(age),
            0.0,
            since,
            weight="alg",
            wvar=(beta - 1.0, 0.0),
            limit=400,
            epsabs=0.0,
            epsrel=1e-13,
        )[0]
        up.append((first + later(survival, since)) / tests)
        failing.append((new + later(density, since)) / tests)
    mean_life = eta * math.gamma(1.0 + 1.0 / beta)
    return (
        mean_life / (tests * interval),
        1.0 / (tests * interval),
        np.array(up),
        np.array(failing),
    )


#: A Weibull life's scale and shape, and how near its rate of failing
#: comes within the first cells after a test (the rest is asked the same
#: of every shape).
WEIBULL = {
    "shape 3": (50000.0, 3.0, 1e-9),
    "shape 1.5": (50000.0, 1.5, 1e-7),
    "shape 0.8": (50000.0, 0.8, 2e-5),
    "shape 0.6": (20000.0, 0.6, 5e-4),
}


@pytest.mark.parametrize("name", sorted(WEIBULL))
def test_a_weibull_unit_in_its_repair_against_its_series(name):
    # A repair of a few cells of the grid, after which a new unit's
    # failures rise as its age to the life's shape: not a line across a
    # cell, for a shape below 1 least of all. (As lines across the cells,
    # with the new units at the grid's points: the profile 1.0e-6 off for
    # a shape of 0.8 and 3.2e-5 for 0.6, the unavailability 5.8e-8 and
    # 9.2e-7, the rate of failing 8% and 56%.)
    eta, beta, near = WEIBULL[name]
    interval, mean_repair = 8760.0, 24.0
    unit = ht.TestedUnit(
        W([eta, beta]), E([1.0 / mean_repair]), None, interval
    )
    long_run, h = unit.long_run, unit.setup.h
    assert 2.0 * h < mean_repair < 10.0 * h
    s = np.array(
        [0.3, 1.0, 3.3, h, 1.5 * h, 7.77, 2.0 * h, 25.0, 60.0]
        + [16.0 * h, 16.4 * h, 32.0 * h, 33.3 * h, 200.0, 4000.0, 8000.0]
    )
    availability, failures, up, failing = _weibull_series(
        eta, beta, interval, mean_repair, s
    )
    assert long_run.availability == pytest.approx(availability, rel=1e-10)
    assert long_run.unavailability == pytest.approx(
        1.0 - availability, rel=1e-9
    )
    assert long_run.failures == pytest.approx(failures, rel=1e-11)
    np.testing.assert_allclose(long_run.profile(s)[0], up, atol=5e-9)
    np.testing.assert_allclose(long_run.intensity(s), failing, rtol=near)


@pytest.mark.parametrize("name", ["shape 1.5", "shape 0.8", "shape 0.6"])
def test_a_pair_of_weibull_units_fails_as_their_series(name):
    # Two units tested together: the system fails where one fails with
    # the other down, hidden or in its repair, so its failures weigh one's
    # rate of failing by the other's downtime, both of which bend within
    # the first cells after a test. (As lines across the cells: 6.9e-8
    # off for a shape of 1.5, 1.1e-6 for 0.8 and 1.1e-4 for 0.6.)
    eta, beta, _ = WEIBULL[name]
    interval, mean_repair = 8760.0, 24.0
    points, shares = np.polynomial.legendre.leggauss(12)
    edges = np.concatenate(
        [
            [0.0],
            mean_repair * 2.0 ** np.arange(-30.0, 1.0),
            mean_repair * np.arange(2, 41),
            np.linspace(40.0 * mean_repair, interval, 41)[1:],
        ]
    )
    middle, half = 0.5 * (edges[1:] + edges[:-1]), 0.5 * np.diff(edges)
    s = (middle[:, None] + half[:, None] * points[None, :]).ravel()
    weights = (half[:, None] * shares[None, :]).ravel()
    _, _, up, failing = _weibull_series(eta, beta, interval, mean_repair, s)
    frequency = 2.0 * float(weights @ (failing * (1.0 - up))) / interval
    unavailability = float(weights @ (1.0 - up) ** 2) / interval
    spec = hidden(W([eta, beta]), E([1.0 / mean_repair]), interval=interval)
    rbd = RepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        {"a": spec, "b": dict(spec)},
    )
    bound = 2e-8 if beta < 0.7 else 1e-9
    assert rbd.system_failure_frequency() == pytest.approx(
        frequency, rel=bound
    )
    assert rbd.mean_unavailability() == pytest.approx(
        unavailability, rel=1e-10
    )


@pytest.mark.parametrize("beta", [0.6, 0.8, 1.5, 3.0])
def test_a_weibull_unit_back_at_a_fixed_time_in_closed_form(beta):
    # Back in service a fixed time after the test that finds it failed,
    # a unit is of an exact age at every test after: working at the k-th
    # with R(k T - r), and so a cycle of N = 1 + sum_k R(k T - r) tests,
    # in which it runs for its mean life. The time is not on the grid.
    eta, interval, repair = 50000.0, 8760.0, 24.0
    unit = ht.TestedUnit(W([eta, beta]), X([repair]), None, interval)
    assert abs(repair / unit.setup.h - round(repair / unit.setup.h)) > 0.1
    k = np.arange(1, int(40.0 ** (1.0 / beta) * eta / interval) + 4)
    tests = 1.0 + float(
        np.exp(-(((k * interval - repair) / eta) ** beta)).sum()
    )
    mean_life = eta * math.gamma(1.0 + 1.0 / beta)
    long_run = unit.long_run
    assert long_run.availability == pytest.approx(
        mean_life / (tests * interval), rel=1e-12
    )
    assert long_run.failures == pytest.approx(
        1.0 / (tests * interval), rel=1e-12
    )
    # s after the test, the unit back at r has failed with F(s - r).
    s = repair + np.array([0.01, 0.5, 1.0, 3.3, 10.0, 100.0, 5000.0])
    ages = (k[:, None] - 1) * interval + s[None, :] - repair
    up = np.exp(-((ages / eta) ** beta)).sum(axis=0) / tests
    np.testing.assert_allclose(long_run.profile(s)[0], up, atol=1e-13)
    density = (
        beta
        / eta
        * (ages / eta) ** (beta - 1.0)
        * np.exp(-((ages / eta) ** beta))
    )
    np.testing.assert_allclose(
        long_run.intensity(s), density.sum(axis=0) / tests, rtol=1e-10
    )


@pytest.mark.parametrize(
    "spec",
    [
        dict(
            life=W([4.0, 2.0]), repair=LN([np.log(0.2), 0.6]), test=X([0.05])
        ),
        dict(life=W([3.0, 0.8]), repair=X([0.3]), test=E([50.0])),
        dict(
            life=W([3.0, 1.5]),
            repair=X([0.4]),
            test=E([30.0]),
            coverage=0.5,
            per=3,
        ),
    ],
    ids=["fixed test", "spread test", "fixed repair, missing"],
)
def test_the_long_run_is_its_profile_s_mean(spec):
    # The mean availability, the failures and the rate of failing are
    # one curve's: its mean over the period, its rise over it and its
    # rate of change, between the grid's points as on them.
    unit = ht.TestedUnit(interval=1.0, **spec)
    long_run = unit.long_run
    edges = np.append(long_run.edges(), long_run.period)
    points, shares = np.polynomial.legendre.leggauss(8)
    middle, half = 0.5 * (edges[1:] + edges[:-1]), 0.5 * np.diff(edges)
    phase = (middle[:, None] + half[:, None] * points[None, :]).ravel()
    weights = (half[:, None] * shares[None, :]).ravel() / long_run.period
    up, down = long_run.profile(phase)
    assert float(weights @ up) == pytest.approx(
        long_run.availability, rel=1e-10
    )
    assert float(weights @ down) == pytest.approx(
        long_run.unavailability, rel=1e-9
    )
    assert float(weights @ long_run.intensity(phase)) == pytest.approx(
        long_run.failures, rel=1e-9
    )


def test_units_working_at_a_test_are_those_not_failed_before_it():
    # New units' failures are worked out from when each goes into
    # service, and their survival at the tests after from masses on the
    # age grid: the two must agree on who is left.
    unit = ht.TestedUnit(W([50000.0, 0.8]), E([1.0 / 24.0]), None, 8760.0)
    cycle = ht._walk(unit.setup, unit.setup.restart)
    assert cycle.p[1] == pytest.approx(1.0 - cycle.fail[0], rel=1e-12)
    assert cycle.fail.sum() == pytest.approx(1.0, rel=1e-12)
