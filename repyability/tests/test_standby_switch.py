"""A cold standby pair of general lives and repairs whose switch can
fail (``switching_probability`` below 1): its long-run values, with one
repair at a time (``_standby_cycle``) and with both units repaired at
once (``_standby_both``), and with one repair at a time its values over
time (``_standby_cycle.over_time``)."""

import math

import numpy as np
import pytest
import surpyval as surv

from repyability import RepairableRBD
from repyability.rbd import (
    _curves,
    _standby_both,
    _standby_chain,
    _standby_cycle,
)

E = surv.Exponential.from_params
W = surv.Weibull.from_params
LN = surv.LogNormal.from_params

MODULES = {1: _standby_cycle, None: _standby_both}


def pair(life, repair, crews, switching):
    return RepairableRBD(
        [("s", "p"), ("p", "t")],
        {
            "p": {
                "reliability": life,
                "repairability": repair,
                "standby": {
                    "units": 2,
                    "k": 1,
                    "switching_probability": switching,
                },
            }
        },
        repair_crews=crews,
    )


@pytest.mark.parametrize("crews", [1, None])
@pytest.mark.parametrize("switching", [0.9, 0.5, 0.0])
@pytest.mark.parametrize(
    "lam, mu", [(1e-3, 0.05), (1e-3, 1 / 300), (0.1, 0.1)]
)
def test_exponential_units_are_the_chain(lam, mu, switching, crews):
    chain = _standby_chain.long_run(
        2, 1, lam, mu, switching=switching, crews=crews
    )
    found = MODULES[crews].long_run(E([lam]), E([mu]), switching)
    for field in chain._fields:
        assert getattr(found, field) == pytest.approx(
            getattr(chain, field), rel=1e-9
        ), field


@pytest.mark.parametrize("crews", [1, None])
def test_exponential_repairs_by_hand(crews):
    # Exponential repairs at rate mu are memoryless, so the pair is in
    # one of two states whenever a unit starts to operate: A, the other
    # under repair, or S, the other ready. After the unit's life X the
    # other is still under repair with probability L = E[exp(-mu X)] in
    # A: the pair is down until a repair ends (the other's, 1 / mu, with
    # one crew; the first of two, 1 / (2 mu), with two), and is in A
    # again. Otherwise (and always in S) the switch works with
    # probability p, to A, or the pair is down for a repair, 1 / mu, and
    # then in S. The states are entered in the ratio p to (1 - L)(1 - p).
    from scipy import integrate

    life, mean = W([1000.0, 3.0]), 1000.0 * math.gamma(1 + 1 / 3)
    mu, p = 1 / 400.0, 0.8
    edges = np.ravel(life.qf(np.array([1e-9, 0.01, 0.5, 0.99, 1 - 1e-9])))
    laplace = sum(
        integrate.quad(
            lambda x: mu
            * math.exp(-mu * x)
            * float(life.ff(np.array([x]))[0]),
            lo,
            hi,
            epsabs=0.0,
            epsrel=1e-12,
            limit=200,
        )[0]
        for lo, hi in zip([0.0, *edges], [*edges, math.inf])
    )
    q = 1 - p
    both_down = 1 / mu if crews == 1 else 1 / (2 * mu)
    a_down = laplace * both_down + (1 - laplace) * q / mu
    a_fails = laplace + (1 - laplace) * q
    s_down, s_fails = q / mu, q
    a, s = p, (1 - laplace) * q
    time = a * (mean + a_down) + s * (mean + s_down)
    found = MODULES[crews].long_run(life, E([mu]), p)
    assert found.unavailability == pytest.approx(
        (a * a_down + s * s_down) / time, rel=1e-7
    )
    assert found.failure_frequency == pytest.approx(
        (a * a_fails + s * s_fails) / time, rel=1e-7
    )
    assert found.unit_failure_frequency == pytest.approx(
        (a + s) / time, rel=1e-7
    )


@pytest.mark.parametrize("crews", [1, None])
def test_a_sure_switch_and_one_that_never_works(crews):
    life, repair = W([1000.0, 2.5]), W([300.0, 1.5])
    module = MODULES[crews]
    sure = module.long_run(life, repair, 1.0)
    assert sure == module.long_run(life, repair)
    worse = module.long_run(life, repair, 0.9)
    assert worse.unavailability > sure.unavailability
    # Never switched: one unit, a life and a repair in turn.
    up = 1000.0 * math.gamma(1 + 1 / 2.5)
    down = 300.0 * math.gamma(1 + 1 / 1.5)
    never = module.long_run(life, repair, 0.0)
    assert never.unavailability == pytest.approx(down / (up + down), rel=1e-9)
    assert never.failure_frequency == pytest.approx(1 / (up + down), rel=1e-9)
    assert module.long_run(life, repair, 1e-6).unavailability == pytest.approx(
        never.unavailability, rel=1e-4
    )


@pytest.mark.parametrize("crews, seed", [(1, 7), (None, 11)])
def test_a_wear_out_pair_against_the_simulation(crews, seed):
    # Weibull(1000, 2.5) lives, Weibull(300, 1.5) repairs and a switch
    # that works nine times in ten: 400 runs of 1,000,000 h within four
    # standard errors, and the system's failures within 2%.
    life, repair = W([1000.0, 2.5]), W([300.0, 1.5])
    rbd = pair(life, repair, crews, 0.9)
    values = MODULES[crews].long_run(life, repair, 0.9)
    exact = rbd.mean_availability()
    assert exact == values.availability
    runs, horizon = 400, 1e6
    run = rbd.availability(
        horizon,
        mc_samples=runs,
        seed=seed,
        control_variate=False,
        conditional=False,
    )
    up = np.asarray(run.uptimes) / horizon
    assert abs(up.mean() - exact) < 4 * up.std(ddof=1) / math.sqrt(runs)
    rate = run.system_failures / (runs * horizon)
    assert rate == pytest.approx(values.failure_frequency, rel=0.02)
    assert rbd.analysis_routes()["mean_availability"].route == "numerical"


@pytest.mark.parametrize("switching", [0.9, 0.5, 0.0])
@pytest.mark.parametrize("lam, mu", [(1e-3, 0.05), (1e-3, 1 / 300), (0.01, 1)])
def test_over_time_exponential_units_are_the_chain(lam, mu, switching):
    # The chain of exponential units, followed by uniformization, against
    # the two states' renewal equations with the same laws: at the
    # grid's points to 1e-9. Between them the curve is linear, so it is
    # off by up to an eighth of the chain's second difference over a
    # step.
    chain = _curves._standby_curve(pair(E([lam]), E([mu]), 1, switching), "p")
    step, n = 1 / (1000 * lam), 8000
    curve = _standby_cycle.over_time(E([lam]), E([mu]), step, n, switching)

    def both(x):
        expected, found = chain.events(x), curve.events(x)
        return [
            (curve.down_at(x), 1 - chain.at(x)),
            (found["failures"], expected["failures"]),
            (found["corrective"], expected["corrective"]),
        ]

    for found, expected in both(curve.times[97::97]):
        assert found == pytest.approx(expected, rel=1e-9)
    between = np.random.default_rng(1).uniform(0.0, n * step, 200)
    bends = [np.abs(np.diff(e, 2)).max() for _, e in both(curve.times)]
    for (found, expected), bend in zip(both(between), bends):
        assert np.abs(found - expected).max() <= bend / 8 + 1e-9 * (
            np.abs(expected).max()
        )


def test_over_time_a_sure_switch_is_unchanged():
    life, repair = W([1000.0, 2.5]), W([300.0, 1.5])
    sure = _standby_cycle.over_time(life, repair, 5.0, 2000, 1.0)
    plain = _standby_cycle.over_time(life, repair, 5.0, 2000)
    for name in ("down", "failures", "units"):
        assert np.array_equal(getattr(sure, name), getattr(plain, name))
    # A switch that almost never fails is close to it.
    nearly = _standby_cycle.over_time(life, repair, 5.0, 2000, 1 - 1e-9)
    assert nearly.down == pytest.approx(sure.down, abs=1e-8)


def _laplace(function, s, cuts):
    """``int exp(-s t) function(t) dt`` from 0 to the last of ``cuts``,
    by quadrature on the pieces between them."""
    from scipy import integrate

    edges = [0.0, *sorted(cuts)]
    return sum(
        integrate.quad(
            lambda t: math.exp(-s * t) * function(t),
            lo,
            hi,
            epsabs=0.0,
            epsrel=1e-10,
            limit=400,
        )[0]
        for lo, hi in zip(edges, edges[1:])
    )


@pytest.mark.parametrize(
    "life, repair, switching",
    [
        (W([1000.0, 2.5]), W([300.0, 1.5]), 0.9),
        (W([1000.0, 1.3]), LN([4.0, 0.7]), 0.6),
    ],
)
def test_over_time_the_transforms_by_hand(life, repair, switching):
    # The pair starts afresh in two states: A, a unit starting to
    # operate as the other starts its repair, and S, with the other
    # ready (where it is at 0). With f*, g*, c* and r* the Laplace
    # transforms of dF, dG, d(F G) and G dF (a life's end with the other
    # ready), p the switch and q = 1 - p: S leads to A with p f* and to
    # S with q f* g*; A to A with c* - q r* and to S with q r* g*. Each
    # visit to S is followed by a repair down with q f*, and each to A
    # by K* (K = F (1 - G), down within a cycle) and by a repair with
    # q r*; a repair under way has the transform (1 - g*) / s. The
    # system fails q f* times a visit to S, and f* - p r* a visit to A.
    def F(t):
        return float(life.ff(np.array([t]))[0])

    def G(t):
        return float(repair.ff(np.array([t]))[0])

    def left(t):
        return float(repair.sf(np.array([t]))[0])

    def f(t):
        return float(life.df(np.array([t]))[0])

    def g(t):
        return float(repair.df(np.array([t]))[0])

    cuts = np.unique(
        np.concatenate(
            [
                np.ravel(
                    m.qf(np.array([1e-3, 0.1, 0.5, 0.9, 1 - 1e-3, 1 - 1e-13]))
                )
                for m in (life, repair)
            ]
        )
    )
    p, q = switching, 1 - switching
    rbd = pair(life, repair, 1, switching)
    curve = _curves._pair_curve(rbd, "p", 1e9)
    assert np.isfinite(curve.settle)
    t = curve.times
    values = rbd.mean_unavailability(), rbd.system_failure_frequency()
    for s in (1 / 3000.0, 1 / 800.0, 1 / 200.0):
        first = _laplace(f, s, cuts)
        repaired = _laplace(g, s, cuts)
        cycle = _laplace(lambda x: f(x) * G(x) + F(x) * g(x), s, cuts)
        ready = _laplace(lambda x: f(x) * G(x), s, cuts)
        within = _laplace(lambda x: F(x) * left(x), s, cuts)
        repairing = (1 - repaired) / s
        a_to_a, a_to_s = cycle - q * ready, q * ready * repaired
        s_to_a, s_to_s = p * first, q * first * repaired
        both = (1 - s_to_s) * (1 - a_to_a) - s_to_a * a_to_s
        in_s, in_a = (1 - a_to_a) / both, s_to_a / both
        down = in_s * q * first * repairing + in_a * (
            within + q * ready * repairing
        )
        fails = in_s * q * first + in_a * (first - p * ready)
        decay = np.exp(-s * t)
        tail = math.exp(-s * t[-1]) / s
        found = np.trapezoid(decay * curve.down, t) + values[0] * tail
        assert found == pytest.approx(down, rel=1e-6)
        # int exp(-s t) dN(t) = s int exp(-s t) N(t) dt.
        counted = s * np.trapezoid(decay * curve.failures, t) + tail * (
            s * curve.failures[-1] + values[1]
        )
        assert counted == pytest.approx(fails, rel=1e-6)


def test_over_time_it_settles_at_the_long_run():
    life, repair = W([1000.0, 2.5]), W([300.0, 1.5])
    rbd = pair(life, repair, 1, 0.9)
    values = _standby_cycle.long_run(life, repair, 0.9)
    late = np.array([1e5, 1e6])
    assert rbd.point_unavailability(late) == pytest.approx(
        values.unavailability, rel=1e-7
    )
    assert np.diff(rbd.expected_failures(late))[0] / 9e5 == pytest.approx(
        values.failure_frequency, rel=1e-7
    )
    assert rbd.point_availability([0.0])[0] == 1.0
    assert rbd.expected_failures([0.0])[0] == 0.0
    route = rbd.analysis_routes()["point_availability"]
    assert route.route == "numerical"


def test_over_time_a_switch_that_never_works_is_one_unit():
    # One unit, a life and a repair in turn: with exponential ones it is
    # down at t with probability lam / (lam + mu) (1 - exp(-(lam + mu)
    # t)), and has failed lam (t - the time down so far) times.
    lam, mu = 1e-3, 1 / 300
    step, n = 1.0, 8000
    curve = _standby_cycle.over_time(E([lam]), E([mu]), step, n, 0.0)
    x = curve.times[97::97]
    total = lam + mu
    down = lam / total * -np.expm1(-total * x)
    assert curve.down_at(x) == pytest.approx(down, rel=1e-9)
    been_down = lam / total * (x + np.expm1(-total * x) / total)
    failures = curve.events(x)["failures"]
    assert failures == pytest.approx(lam * (x - been_down), rel=1e-9)
    assert curve.events(x)["corrective"] == pytest.approx(failures, rel=1e-12)


def test_over_time_a_switch_that_never_works_is_one_weibull_unit():
    # The same with Weibull(1000, 2.5) lives and Weibull(300, 1.5)
    # repairs, against the one unit's own curve on its renewal grid:
    # 6.2e-8 apart at most (1.0e-7 before that grid's early grids, and
    # 2.3e-7 at 900 h on the grid this was first tried on). The unit's
    # expected failures are 1.9e-7 off far from the start (its grid's
    # sums of times are spread a little wider than the times are).
    life, repair = W([1000.0, 2.5]), W([300.0, 1.5])
    curve = _standby_cycle.over_time(life, repair, 1.0, 8000, 0.0)
    unit = RepairableRBD(
        [("s", "a"), ("a", "t")],
        {"a": {"reliability": life, "repairability": repair}},
    )
    x = curve.times[97::97]
    assert curve.down_at(x) == pytest.approx(
        unit.point_unavailability(x), abs=1e-7
    )
    assert curve.events(x)["failures"] == pytest.approx(
        np.ravel(unit.expected_failures(x)), abs=3e-7
    )


@pytest.mark.parametrize(
    "life, repair, switching, window, seed",
    [
        (W([1000.0, 2.5]), W([300.0, 1.5]), 0.9, 2500.0, 23),
        (W([1000.0, 1.5]), LN([5.0, 0.6]), 0.7, 1200.0, 29),
    ],
)
def test_over_time_against_the_simulation(
    life, repair, switching, window, seed
):
    # 20,000 runs of a window a few lives long, from new: the mission
    # availability within four standard errors, and the system's
    # failures within four of their count's (its square root: a count of
    # failures in a window is no more spread than a Poisson one).
    rbd = pair(life, repair, 1, switching)
    runs = 20_000
    run = rbd.availability(
        window,
        mc_samples=runs,
        seed=seed,
        control_variate=False,
        conditional=False,
    )
    up = np.asarray(run.uptimes) / window
    exact = rbd.mission_availability(window)
    assert abs(up.mean() - exact) < 4 * up.std(ddof=1) / math.sqrt(runs)
    expected = float(rbd.expected_failures([window])[0])
    count = run.system_failures
    assert abs(count / runs - expected) < 4 * math.sqrt(count) / runs


def test_over_time_both_repaired_at_once_is_refused():
    rbd = pair(W([1000.0, 2.5]), W([300.0, 1.5]), None, 0.9)
    with pytest.raises(NotImplementedError, match="repair_crews=1") as error:
        rbd.point_availability([10.0])
    route = rbd.analysis_routes()["point_availability"]
    assert route.route == "refused" and route.reason == str(error.value)
