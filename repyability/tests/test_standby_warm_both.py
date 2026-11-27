"""The long-run values of a warm standby pair with general lives and
repairs whose units are both repaired at once, from the Markov renewal
process on its operating unit's used life and a repair's time left
(``repyability.rbd._standby_warm_both``)."""

import math
import warnings

import numpy as np
import pytest
import surpyval as surv

from repyability import RepairableRBD
from repyability.rbd import (
    _standby_both,
    _standby_chain,
    _standby_warm,
    _standby_warm_both,
)

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


@pytest.mark.parametrize("dormancy", [0.1, 0.5, 1.0])
@pytest.mark.parametrize(
    "lam, mu", [(1e-3, 0.05), (1e-3, 1 / 300), (0.1, 0.1)]
)
def test_exponential_units_are_the_chain(lam, mu, dormancy):
    # To RTOL: the halving ends when two extrapolated limits agree to
    # it, and exponential units are smooth, so the last limit is at
    # least that near.
    chain = _standby_chain.long_run(2, 1, lam, mu, dormancy=dormancy)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        found = _standby_warm_both.long_run(E([lam]), E([mu]), dormancy)
    for field in chain._fields:
        assert getattr(found, field) == pytest.approx(
            getattr(chain, field), rel=_standby_warm_both.RTOL
        ), field


def test_a_spare_that_hardly_ages_is_the_cold_pair():
    # A spare ageing at 1e-9 of the rate fails in standby about that
    # often in a life, and the cold pair's own values are to 1e-8: far
    # inside 1e-6, as with one repair at a time.
    life, repair = W([1000.0, 2.5]), W([300.0, 1.5])
    cold = _standby_both.long_run(life, repair)
    found = _standby_warm_both.long_run(life, repair, 1e-9)
    for field in cold._fields:
        assert getattr(found, field) == pytest.approx(
            getattr(cold, field), rel=1e-6
        ), field


def test_a_second_crew_is_never_worse_and_ageing_never_better():
    life, repair = W([1000.0, 2.5]), W([300.0, 1.5])
    cold = _standby_both.long_run(life, repair)
    both = [_standby_warm_both.long_run(life, repair, d) for d in (0.1, 1.0)]
    one = [_standby_warm.long_run(life, repair, d) for d in (0.1, 1.0)]
    down = [cold.unavailability] + [v.unavailability for v in both]
    assert down == sorted(down) and len(set(down)) == 3
    for two_crews, one_crew in zip(both, one):
        assert two_crews.unavailability < one_crew.unavailability


def test_a_hot_pair_is_two_units():
    # A hot spare ages as the operating unit does, and with a crew each
    # no repair waits: each unit is a life and a repair in turn, and
    # the pair is down while both are in repair.
    life, repair = W([1000.0, 2.5]), W([300.0, 1.5])
    up = 1000.0 * math.gamma(1 + 1 / 2.5)
    down = 300.0 * math.gamma(1 + 1 / 1.5)
    found = _standby_warm_both.long_run(life, repair, 1.0)
    # To ten times RTOL: these models are not smooth at 0.
    assert found.unit_failure_frequency == pytest.approx(
        2.0 / (up + down), rel=1e-8
    )
    assert found.unavailability == pytest.approx(
        (down / (up + down)) ** 2, rel=1e-8
    )


def test_the_unit_of_time_does_not_matter():
    hours = _standby_warm_both.long_run(W([1000.0, 2.5]), W([300.0, 1.5]), 0.5)
    scale = 1e-6
    other = _standby_warm_both.long_run(
        W([1000.0 * scale, 2.5]), W([300.0 * scale, 1.5]), 0.5
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


def simulated(rbd, seed, batches=8, runs=50, horizon=1e6):
    """Seeded batches of runs: each run's fraction of the time up, and
    each batch's system failures per unit time."""
    up, rates = [], []
    for batch in range(batches):
        run = rbd.availability(
            horizon,
            mc_samples=runs,
            seed=seed + batch,
            control_variate=False,
            conditional=False,
        )
        up.extend(np.asarray(run.uptimes) / horizon)
        rates.append(run.system_failures / (runs * horizon))
    return np.array(up), np.array(rates)


def within(estimates, exact, errors=4.0):
    error = estimates.std(ddof=1) / math.sqrt(len(estimates))
    return abs(estimates.mean() - exact) < errors * error


@pytest.mark.parametrize(
    "life, repair, dormancy, seed",
    [
        (W([1000.0, 2.5]), W([300.0, 1.5]), 0.5, 307),
        (W([1000.0, 1.5]), LN([5.0, 0.6]), 0.2, 401),
    ],
)
def test_wear_out_pairs_against_the_simulation(life, repair, dormancy, seed):
    # 8 batches of 50 runs of 1,000,000 h (seeds from ``seed``): the
    # availability within four standard errors of the 400 runs', and
    # the system's failures within four of the 8 batches'.
    rbd = pair(life, repair, dormancy_factor=dormancy)
    values = _standby_warm_both.long_run(life, repair, dormancy)
    assert rbd.mean_availability() == values.availability
    up, rates = simulated(rbd, seed)
    assert within(up, values.availability)
    assert within(rates, values.failure_frequency)
    route = rbd.analysis_routes()["mean_availability"]
    assert route.route == "numerical" and "both units" in route.reason


def test_the_pair_costs_its_repairs_and_its_down_time():
    life, repair = W([1000.0, 2.5]), W([300.0, 1.5])
    rbd = RepairableRBD(
        [("s", "p"), ("p", "t")],
        {
            "p": {
                "reliability": life,
                "repairability": repair,
                "standby": {"units": 2, "k": 1, "dormancy_factor": 0.5},
                "repair_cost": 3.0,
            }
        },
        downtime_cost_rate=5.0,
    )
    values = _standby_warm_both.long_run(life, repair, 0.5)
    assert rbd.expected_cost_rate() == pytest.approx(
        3.0 * values.unit_failure_frequency + 5.0 * values.unavailability,
        rel=1e-12,
    )


def test_a_failing_switch_with_both_repaired_at_once_is_refused():
    rbd = pair(
        W([1000.0, 2.5]),
        W([300.0, 1.5]),
        dormancy_factor=0.5,
        switching_probability=0.9,
    )
    with pytest.raises(NotImplementedError, match="repair_crews=1") as error:
        rbd.mean_availability()
    route = rbd.analysis_routes()["mean_availability"]
    assert route.route == "refused" and route.reason == str(error.value)
    assert "simulate it with availability()" in route.reason
