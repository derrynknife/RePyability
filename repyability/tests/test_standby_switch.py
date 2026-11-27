"""A cold standby pair of general lives and repairs whose switch can
fail (``switching_probability`` below 1): its long-run values, with one
repair at a time (``_standby_cycle``) and with both units repaired at
once (``_standby_both``)."""

import math

import numpy as np
import pytest
import surpyval as surv

from repyability import RepairableRBD
from repyability.rbd import _standby_both, _standby_chain, _standby_cycle

E = surv.Exponential.from_params
W = surv.Weibull.from_params

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


def test_over_time_a_switch_that_can_fail_is_refused():
    rbd = pair(W([1000.0, 2.5]), W([300.0, 1.5]), 1, 0.9)
    with pytest.raises(NotImplementedError, match="a sure switch") as error:
        rbd.point_availability([10.0])
    route = rbd.analysis_routes()["point_availability"]
    assert route.route == "refused" and route.reason == str(error.value)
