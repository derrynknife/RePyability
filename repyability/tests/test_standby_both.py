"""The long-run values of a cold standby pair with general lives and
repairs whose units are both repaired at once, from the Markov renewal
process on the other unit's elapsed repair
(``repyability.rbd._standby_both``)."""

import math
import warnings

import numpy as np
import pytest
import surpyval as surv

from repyability import RepairableRBD
from repyability.rbd import _standby_both, _standby_chain, _standby_cycle

E = surv.Exponential.from_params
W = surv.Weibull.from_params
LN = surv.LogNormal.from_params


def pair(life, repair, crews=None, **standby):
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


@pytest.mark.parametrize(
    "lam, mu", [(1e-3, 0.05), (1e-3, 1 / 300), (0.1, 0.1), (1.0, 0.2)]
)
def test_exponential_units_are_the_chain(lam, mu):
    chain = _standby_chain.long_run(2, 1, lam, mu)
    found = _standby_both.long_run(E([lam]), E([mu]))
    for field in chain._fields:
        assert getattr(found, field) == pytest.approx(
            getattr(chain, field), rel=1e-9
        ), field
    # A Weibull of shape 1 is the exponential.
    one = _standby_both.long_run(W([1 / lam, 1.0]), W([1 / mu, 1.0]))
    assert one.unavailability == pytest.approx(chain.unavailability, rel=1e-9)


@pytest.mark.parametrize(
    "life, mean",
    [
        (W([1000.0, 3.0]), 1000.0 * math.gamma(1 + 1 / 3)),
        (W([500.0, 0.8]), 500.0 * math.gamma(1 + 1 / 0.8)),
        (LN([6.0, 0.5]), math.exp(6.0 + 0.125)),
    ],
)
def test_exponential_repairs_by_hand(life, mean):
    # Exponential repairs at rate mu are memoryless, so every step is
    # alike: the operating unit's life X, then, if the other's repair is
    # not over (probability L = E[exp(-mu X)]), the pair down until the
    # first of two repairs ends, 1 / (2 mu) on average. So per step the
    # time is E[X] + L / (2 mu), with L system failures and one unit's.
    from scipy import integrate

    mu = 1 / 400.0
    # L = int mu exp(-mu x) F(x) dx (by parts: no density needed).
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
    down = laplace / (2 * mu)
    step = mean + down
    found = _standby_both.long_run(life, E([mu]))
    assert found.unavailability == pytest.approx(down / step, rel=1e-7)
    assert found.failure_frequency == pytest.approx(laplace / step, rel=1e-7)
    assert found.unit_failure_frequency == pytest.approx(1 / step, rel=1e-7)


def test_a_second_crew_is_never_worse():
    life, repair = W([1000.0, 2.5]), W([300.0, 1.5])
    one = _standby_cycle.long_run(life, repair)
    both = _standby_both.long_run(life, repair)
    assert both.unavailability < one.unavailability
    # The units fail more often for being up more of the time.
    assert both.unit_failure_frequency > one.unit_failure_frequency


def test_the_unit_of_time_does_not_matter():
    hours = _standby_both.long_run(W([1000.0, 2.5]), LN([5.0, 0.6]))
    scale = 1e-6
    other = _standby_both.long_run(
        W([1000.0 * scale, 2.5]), LN([5.0 + math.log(scale), 0.6])
    )
    assert other.unavailability == pytest.approx(
        hours.unavailability, rel=1e-9
    )
    assert other.failure_frequency * scale == pytest.approx(
        hours.failure_frequency, rel=1e-9
    )
    assert other.unit_failure_frequency * scale == pytest.approx(
        hours.unit_failure_frequency, rel=1e-9
    )


def test_a_small_unavailability_keeps_its_digits():
    # Lives far longer than the repairs: the pair is down about 4e-13 of
    # the time, worked out as itself. Both down takes a life shorter
    # than a repair, R > X, and then the shorter of what two repairs
    # have left: E[min(R - X, R'); R > X], here by quadrature over X
    # (the pair regenerates at nearly every failure, so one step from
    # a = 0 is the cycle to 1e-10).
    from scipy import integrate

    life, repair = W([1e5, 2.0]), LN([2.0, 0.5])
    end = float(np.ravel(repair.qf(1 - 1e-13))[0])

    def survival(t):
        return float(np.ravel(repair.sf(np.array([t])))[0])

    def shorter(x):
        return integrate.quad(
            lambda s: survival(x + s) * survival(s),
            0.0,
            end,
            epsabs=0.0,
            epsrel=1e-10,
            limit=200,
        )[0]

    down = integrate.quad(
        lambda x: float(life.df(np.array([x]))[0]) * shorter(x),
        0.0,
        end,
        epsabs=0.0,
        epsrel=1e-9,
        limit=200,
    )[0]
    mean = 1e5 * math.gamma(1.5)
    found = _standby_both.long_run(life, repair)
    assert found.unavailability == pytest.approx(down / mean, rel=1e-6)
    assert found.unavailability < 1e-12


@pytest.mark.parametrize(
    "life, repair, seed",
    [
        (W([1000.0, 2.5]), W([300.0, 1.5]), 7),
        (W([1000.0, 1.5]), LN([5.0, 0.6]), 11),
    ],
)
def test_wear_out_pairs_against_the_simulation(life, repair, seed):
    # 400 runs of 1,000,000 h (from new, a small part of it) within four
    # standard errors: the availability, and the system's and the units'
    # failures by their counts' spread between runs.
    rbd = pair(life, repair)
    exact = rbd.mean_availability()
    values = _standby_both.long_run(life, repair)
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
    route = rbd.analysis_routes()["mean_availability"]
    assert route.route == "numerical" and "elapsed repair" in route.reason
    assert rbd.system_failure_frequency() == values.failure_frequency
    assert rbd.mean_unavailability() == values.unavailability


def test_inside_a_larger_diagram_with_crews_to_spare():
    # Two crews and a third component: the pair's units never wait.
    life, repair = W([1000.0, 2.5]), W([300.0, 1.5])
    rbd = RepairableRBD(
        [("s", "p"), ("p", "c"), ("c", "t")],
        {
            "p": {
                "reliability": life,
                "repairability": repair,
                "standby": {"units": 2, "k": 1},
            },
            "c": {"reliability": E([1e-4]), "repairability": E([0.1])},
        },
    )
    values = _standby_both.long_run(life, repair)
    other = 0.1 / (0.1 + 1e-4)
    assert rbd.mean_availability() == pytest.approx(
        values.availability * other, rel=1e-12
    )


def test_a_grid_that_does_not_settle_warns(monkeypatch):
    monkeypatch.setattr(_standby_both, "MAX_STEPS", 2**11)
    with pytest.warns(UserWarning, match="did not settle to 1e-08"):
        found = _standby_both.long_run(W([500.0, 1.2]), LN([4.0, 0.8]))
    monkeypatch.undo()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        settled = _standby_both.long_run(W([500.0, 1.2]), LN([4.0, 0.8]))
    assert found.unavailability == pytest.approx(
        settled.unavailability, rel=0.05
    )


def test_lives_too_short_for_the_grid_are_refused(monkeypatch):
    # Lives a millionth of the repairs: every life rounds to 0 on any
    # grid allowed, and the pair never regenerates on it.
    with pytest.raises(NotImplementedError, match="simulate"):
        _standby_both.long_run(E([1e3]), E([1e-4]))


@pytest.mark.parametrize(
    "words, rbd",
    [
        (
            "warm spare",
            pair(W([1000.0, 2.5]), E([0.01]), dormancy_factor=0.5),
        ),
        (
            "switching",
            pair(W([1000.0, 2.5]), E([0.01]), switching_probability=0.9),
        ),
        (
            "3 units",
            pair(W([1000.0, 2.5]), E([0.01]), units=3),
        ),
    ],
)
def test_what_is_not_taken_is_refused(words, rbd):
    with pytest.raises(NotImplementedError, match=words):
        rbd.mean_availability()
    assert rbd.analysis_routes()["mean_availability"].route == "refused"
