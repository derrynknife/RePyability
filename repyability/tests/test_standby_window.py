"""A cold standby pair with general lives and repairs and one repair at
a time, over time from new: its cycle as a renewal equation on a grid
(``repyability.rbd._standby_cycle.over_time``)."""

import math

import numpy as np
import pytest
import surpyval as surv

from repyability import NodeState, RepairableRBD
from repyability.rbd import _curves, _standby_cycle

E = surv.Exponential.from_params
W = surv.Weibull.from_params
LN = surv.LogNormal.from_params


def pair(life, repair, crews=1, **standby):
    return RepairableRBD(
        [("s", "p"), ("p", "t")],
        {
            "p": {
                "reliability": life,
                "repairability": repair,
                "standby": {"units": 2, "k": 1, **standby},
            }
        },
        repair_crews=crews,
    )


@pytest.mark.parametrize("lam, mu", [(1e-3, 0.05), (1e-3, 1 / 300), (0.01, 1)])
def test_exponential_units_are_the_chain(lam, mu):
    # The chain of exponential units, followed by uniformization, against
    # the cycle's renewal equation with the same laws: at the grid's
    # points to 1e-9. Between them the curve is linear, so it is off by
    # up to an eighth of the chain's second difference over a step.
    chain = _curves._standby_curve(pair(E([lam]), E([mu])), "p")
    step, n = 1 / (1000 * lam), 8000
    curve = _standby_cycle.over_time(E([lam]), E([mu]), step, n)

    def both(x):
        expected, found = chain.events(x), curve.events(x)
        assert not found["planned"].any() and not found["preventive"].any()
        return [
            (curve.down_at(x), 1 - chain.at(x)),
            (found["failures"], expected["failures"]),
            (found["corrective"], expected["corrective"]),
        ]

    for found, expected in both(curve.times[97::97]):
        assert found == pytest.approx(expected, rel=1e-9)
    x = curve.times[97::97]
    assert curve.at(x) == pytest.approx(chain.at(x), rel=1e-12)
    between = np.random.default_rng(1).uniform(0.0, n * step, 200)
    bends = [np.abs(np.diff(e, 2)).max() for _, e in both(curve.times)]
    for (found, expected), bend in zip(both(between), bends):
        assert np.abs(found - expected).max() <= bend / 8 + 1e-9 * (
            np.abs(expected).max()
        )


def _laplace(function, s, cuts):
    """``int exp(-s t) function(t) dt`` from 0 to the last of ``cuts``,
    by quadrature on the pieces between them."""
    from scipy import integrate

    edges = [0.0, *sorted(cuts)]
    total = 0.0
    for lo, hi in zip(edges, edges[1:]):
        total += integrate.quad(
            lambda t: math.exp(-s * t) * function(t),
            lo,
            hi,
            epsabs=0.0,
            epsrel=1e-10,
            limit=400,
        )[0]
    return total


@pytest.mark.parametrize(
    "life, repair",
    [
        (W([1000.0, 2.5]), W([300.0, 1.5])),
        (W([1000.0, 1.3]), LN([4.0, 0.7])),
    ],
)
def test_the_transforms_by_hand(life, repair):
    # From new the pair regenerates first at X, and then after each
    # cycle, max(X, R). With f*, c* and K* the Laplace transforms of dF,
    # d(F G) and K = F (1 - G) (down within a cycle), the unavailability's
    # is f* K* / (1 - c*), and the transform of the expected failures'
    # derivative is f* q* / (1 - c*), q* that of (1 - G) dF: each by
    # quadrature of the models, against the curve's own transform (the
    # trapezoid rule on its grid, and its long-run values past it).
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
    rbd = pair(life, repair)
    curve = _curves._pair_curve(rbd, "p", 1e9)
    assert np.isfinite(curve.settle)
    t = curve.times
    values = rbd.mean_unavailability(), rbd.system_failure_frequency()
    for s in (1 / 3000.0, 1 / 800.0, 1 / 200.0):
        first = _laplace(f, s, cuts)
        cycle = _laplace(lambda x: f(x) * G(x) + F(x) * g(x), s, cuts)
        down = _laplace(lambda x: F(x) * left(x), s, cuts)
        fails = _laplace(lambda x: f(x) * left(x), s, cuts)
        decay = np.exp(-s * t)
        tail = math.exp(-s * t[-1]) / s
        found = np.trapezoid(decay * curve.down, t) + values[0] * tail
        assert found == pytest.approx(first * down / (1 - cycle), rel=1e-6)
        # int exp(-s t) dN(t) = s int exp(-s t) N(t) dt.
        counted = s * np.trapezoid(decay * curve.failures, t) + tail * (
            s * curve.failures[-1] + values[1]
        )
        assert counted == pytest.approx(first * fails / (1 - cycle), rel=1e-6)


def test_it_settles_at_the_long_run():
    life, repair = W([1000.0, 2.5]), W([300.0, 1.5])
    rbd = pair(life, repair)
    values = _standby_cycle.long_run(life, repair)
    late = np.array([1e5, 1e6])
    assert rbd.point_availability(late) == pytest.approx(
        values.availability, abs=1e-10
    )
    assert rbd.point_unavailability(late) == pytest.approx(
        values.unavailability, rel=1e-7
    )
    assert np.diff(rbd.expected_failures(late))[0] / 9e5 == pytest.approx(
        values.failure_frequency, rel=1e-7
    )
    # New, both ready: up until two lives have ended at the soonest.
    assert rbd.point_availability([0.0])[0] == 1.0
    assert rbd.expected_failures([0.0])[0] == 0.0
    # A mission average tends to the long run from above.
    assert rbd.mission_availability(1e7) == pytest.approx(
        values.availability, abs=1e-5
    )
    assert rbd.mission_availability(1e7) > values.availability
    route = rbd.analysis_routes()["point_availability"]
    assert route.route == "numerical"


@pytest.mark.parametrize(
    "life, repair, window, seed",
    [
        (W([1000.0, 2.5]), W([300.0, 1.5]), 2500.0, 7),
        (W([1000.0, 1.5]), LN([5.0, 0.6]), 1200.0, 11),
    ],
)
def test_wear_out_pairs_against_the_simulation(life, repair, window, seed):
    # 20,000 runs of a window a few lives long, from new: the mission
    # availability within four standard errors, and the system's
    # failures within four of their count's (its square root: a count of
    # failures in a window is no more spread than a Poisson one).
    rbd = pair(life, repair)
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


def test_long_in_service_it_is_in_its_long_run_state():
    life, repair = W([1000.0, 2.5]), W([300.0, 1.5])
    rbd = pair(life, repair)
    values = _standby_cycle.long_run(life, repair)
    state = {"p": NodeState(stationary=True)}
    x = np.array([0.0, 10.0, 1e4])
    assert rbd.point_availability(x, state=state) == pytest.approx(
        values.availability, rel=1e-14
    )
    assert rbd.expected_failures(x, state=state) == pytest.approx(
        values.failure_frequency * x, rel=1e-12
    )


def test_the_unit_of_time_does_not_matter():
    scale = 1e-6
    hours = pair(W([1000.0, 2.5]), LN([5.0, 0.6]))
    other = pair(W([1000.0 * scale, 2.5]), LN([5.0 + math.log(scale), 0.6]))
    x = np.array([300.0, 1500.0, 4000.0, 1e5])
    assert other.point_unavailability(x * scale) == pytest.approx(
        hours.point_unavailability(x), rel=1e-7
    )
    assert other.expected_failures(x * scale) == pytest.approx(
        hours.expected_failures(x), rel=1e-7
    )


@pytest.mark.parametrize(
    "words, rbd",
    [
        (
            "repair_crews=1",
            pair(W([1000.0, 2.5]), W([300.0, 1.5]), crews=None),
        ),
        (
            "warm spare",
            pair(W([1000.0, 2.5]), E([0.01]), dormancy_factor=0.5),
        ),
        (
            "switching",
            pair(W([1000.0, 2.5]), E([0.01]), switching_probability=0.9),
        ),
    ],
)
def test_what_is_not_followed_over_time_is_refused(words, rbd):
    with pytest.raises(NotImplementedError, match=words) as error:
        rbd.point_availability([10.0])
    route = rbd.analysis_routes()["point_availability"]
    assert route.route == "refused" and route.reason == str(error.value)
    with pytest.raises(NotImplementedError, match=words):
        rbd.expected_failures([10.0])
