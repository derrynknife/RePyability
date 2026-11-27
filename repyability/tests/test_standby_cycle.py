"""The long-run values of a cold standby pair with general lives and
repairs and one repair at a time, from its regeneration cycle
(``repyability.rbd._standby_cycle``)."""

import math

import numpy as np
import pytest
import surpyval as surv

from repyability import RepairableRBD
from repyability.rbd import _standby_chain, _standby_cycle

E = surv.Exponential.from_params
W = surv.Weibull.from_params


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


@pytest.mark.parametrize(
    "lam, mu", [(1e-3, 0.05), (1e-3, 1 / 300), (0.1, 0.1)]
)
def test_exponential_units_are_the_chain(lam, mu):
    chain = _standby_chain.long_run(2, 1, lam, mu, crews=1)
    cycle = _standby_cycle.long_run(E([lam]), E([mu]))
    for field in chain._fields:
        assert getattr(cycle, field) == pytest.approx(
            getattr(chain, field), rel=1e-9
        ), field
    # A Weibull of shape 1 is the exponential.
    one = _standby_cycle.long_run(W([1 / lam, 1.0]), E([mu]))
    assert one.unavailability == pytest.approx(chain.unavailability, rel=1e-9)


def test_the_cycle_by_hand():
    # Exponential repairs at rate mu: R is memoryless, so E[(R - X)+] =
    # E[exp(-mu X)] / mu and P(R >= X) = E[exp(-mu X)], and a cycle is
    # E[X] + E[(R - X)+]. With X a Weibull(1000, 3), E[X] = 1000
    # Gamma(4/3) and E[exp(-mu X)] its Laplace transform.
    from scipy import integrate, special

    mu, life = 1 / 800.0, W([1000.0, 3.0])
    mean = 1000.0 * special.gamma(1 + 1 / 3)
    laplace = integrate.quad(
        lambda x: math.exp(-mu * x) * float(life.df(np.array([x]))[0]),
        0.0,
        math.inf,
        epsabs=0.0,
        epsrel=1e-12,
    )[0]
    down = laplace / mu
    cycle = mean + down
    found = _standby_cycle.long_run(life, E([mu]))
    assert found.unavailability == pytest.approx(down / cycle, rel=1e-8)
    assert found.failure_frequency == pytest.approx(laplace / cycle, rel=1e-8)
    assert found.unit_failure_frequency == pytest.approx(1 / cycle, rel=1e-8)


def test_a_wear_out_pair_against_the_simulation():
    # Weibull(1000, 2.5) lives, Weibull(300, 1.5) repairs: 400 runs of
    # 1,000,000 h (from new, a small part of it) within four standard
    # errors, availability and the system's failures.
    rbd = pair(W([1000.0, 2.5]), W([300.0, 1.5]))
    exact = rbd.mean_availability()
    runs, horizon = 400, 1e6
    run = rbd.availability(
        horizon,
        mc_samples=runs,
        seed=7,
        control_variate=False,
        conditional=False,
    )
    up = np.asarray(run.uptimes) / horizon
    assert abs(up.mean() - exact) < 4 * up.std(ddof=1) / math.sqrt(runs)
    cycle = _standby_cycle.long_run(W([1000.0, 2.5]), W([300.0, 1.5]))
    rate = run.system_failures / (runs * horizon)
    assert rate == pytest.approx(cycle.failure_frequency, rel=0.05)
    route = rbd.analysis_routes()["mean_availability"]
    assert route.route == "numerical" and "cycle" in route.reason


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
    ],
)
def test_what_the_cycle_does_not_take_is_refused(words, rbd):
    with pytest.raises(NotImplementedError, match=words):
        rbd.mean_availability()
    assert rbd.analysis_routes()["mean_availability"].route == "refused"
