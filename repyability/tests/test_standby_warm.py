"""The long-run values of a warm standby pair with general lives and
repairs and one repair at a time, from the Markov renewal process on its
operating unit's used life (``repyability.rbd._standby_warm``)."""

import math
import warnings

import numpy as np
import pytest
import surpyval as surv

from repyability import RepairableRBD
from repyability.rbd import _standby_chain, _standby_cycle, _standby_warm

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


@pytest.mark.parametrize("dormancy", [0.1, 0.5, 1.0])
@pytest.mark.parametrize(
    "lam, mu", [(1e-3, 0.05), (1e-3, 1 / 300), (0.1, 0.1)]
)
def test_exponential_units_are_the_chain(lam, mu, dormancy):
    chain = _standby_chain.long_run(2, 1, lam, mu, dormancy=dormancy, crews=1)
    found = _standby_warm.long_run(E([lam]), E([mu]), dormancy)
    for field in chain._fields:
        assert getattr(found, field) == pytest.approx(
            getattr(chain, field), rel=1e-9
        ), field


def by_hand(lam, repair, dormancy):
    """Exponential lives at rate ``lam`` are memoryless, so the pair
    starts afresh each time a repair starts, whatever the operating unit
    has used. The repair ``R`` ends with the operating unit still up
    with probability ``L = E[exp(-lam R)]``; if not, the pair fails and
    is down until ``R``, ``int (1 - G(s)) (1 - exp(-lam s)) ds`` on
    average. If so, the next repair starts when either unit fails, after
    ``1 / (lam (1 + d))`` more. One unit fails each time."""
    from scipy import integrate

    def survives(s):
        return float(repair.sf(np.array([s]))[0])

    edges = np.ravel(
        repair.qf(np.array([1e-6, 0.01, 0.5, 0.99, 1 - 1e-6, 1 - 1e-12]))
    )

    def integral(f):
        return sum(
            integrate.quad(f, lo, hi, epsabs=0.0, epsrel=1e-12, limit=200)[0]
            for lo, hi in zip([0.0, *edges], [*edges, math.inf])
        )

    mean = integral(survives)
    down = integral(lambda s: survives(s) * -math.expm1(-lam * s))
    # 1 - L = int lam exp(-lam s) (1 - G(s)) ds.
    fails = integral(lambda s: lam * math.exp(-lam * s) * survives(s))
    time = mean + (1.0 - fails) / (lam * (1.0 + dormancy))
    return down / time, fails / time, 1.0 / time


@pytest.mark.parametrize("dormancy", [0.2, 1.0])
@pytest.mark.parametrize("repair", [W([300.0, 1.5]), LN([5.0, 0.6])])
def test_exponential_lives_by_hand(repair, dormancy):
    lam = 1e-3
    unavailability, fails, units = by_hand(lam, repair, dormancy)
    found = _standby_warm.long_run(E([lam]), repair, dormancy)
    assert found.unavailability == pytest.approx(unavailability, rel=1e-8)
    assert found.failure_frequency == pytest.approx(fails, rel=1e-8)
    assert found.unit_failure_frequency == pytest.approx(units, rel=1e-8)
    assert found.availability == pytest.approx(1 - unavailability, rel=1e-9)


def test_a_small_unavailability_keeps_its_digits():
    # Repairs of half an hour beside lives of 10,000 h: the pair is down
    # about a billionth of the time, and the grid's step is longer than
    # any repair.
    lam, repair = 1e-4, W([0.5, 1.5])
    unavailability, fails, _ = by_hand(lam, repair, 0.5)
    assert unavailability < 1e-8
    found = _standby_warm.long_run(E([lam]), repair, 0.5)
    assert found.unavailability == pytest.approx(unavailability, rel=1e-6)
    assert found.failure_frequency == pytest.approx(fails, rel=1e-6)


def test_a_spare_that_hardly_ages_is_the_cold_pair():
    life, repair = W([1000.0, 2.5]), W([300.0, 1.5])
    cold = _standby_cycle.long_run(life, repair)
    found = _standby_warm.long_run(life, repair, 1e-9)
    for field in cold._fields:
        assert getattr(found, field) == pytest.approx(
            getattr(cold, field), rel=1e-6
        ), field


def test_a_spare_that_ages_more_fails_more():
    life, repair = W([1000.0, 2.5]), W([300.0, 1.5])
    cold = _standby_cycle.long_run(life, repair)
    warm = [_standby_warm.long_run(life, repair, d) for d in (0.1, 0.5, 1.0)]
    down = [cold.unavailability] + [v.unavailability for v in warm]
    units = [cold.unit_failure_frequency] + [
        v.unit_failure_frequency for v in warm
    ]
    assert down == sorted(down) and units == sorted(units)


def test_a_hot_pair_with_short_repairs_is_two_units():
    # A hot spare ages as the operating unit does, so with repairs a
    # hundred-thousandth of the lives (hardly ever one waiting for the
    # other's) each unit is a life and a repair in turn.
    life, repair = W([1000.0, 2.5]), W([0.01, 1.5])
    up = 1000.0 * math.gamma(1 + 1 / 2.5)
    down = 0.01 * math.gamma(1 + 1 / 1.5)
    found = _standby_warm.long_run(life, repair, 1.0)
    assert found.unit_failure_frequency == pytest.approx(
        2.0 / (up + down), rel=1e-7
    )


def test_the_unit_of_time_does_not_matter():
    hours = _standby_warm.long_run(W([1000.0, 2.5]), W([300.0, 1.5]), 0.5)
    scale = 1e-6
    other = _standby_warm.long_run(
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


@pytest.mark.parametrize(
    "life, repair, dormancy, seed",
    [
        (W([1000.0, 2.5]), W([300.0, 1.5]), 0.5, 13),
        (W([1000.0, 1.5]), LN([5.0, 0.6]), 0.2, 17),
        (W([1000.0, 2.5]), W([300.0, 1.5]), 1.0, 19),
    ],
)
def test_wear_out_pairs_against_the_simulation(life, repair, dormancy, seed):
    # 400 runs of 1,000,000 h: the availability within four standard
    # errors, and the system's and the units' failures within 2%.
    rbd = pair(life, repair, dormancy_factor=dormancy)
    values = _standby_warm.long_run(life, repair, dormancy)
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
    route = rbd.analysis_routes()["mean_availability"]
    assert route.route == "numerical" and "warm standby pair" in route.reason


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
        repair_crews=1,
        downtime_cost_rate=5.0,
    )
    values = _standby_warm.long_run(life, repair, 0.5)
    assert rbd.system_failure_frequency() == pytest.approx(
        values.failure_frequency, rel=1e-12
    )
    assert rbd.mean_unavailability() == pytest.approx(
        values.unavailability, rel=1e-12
    )
    assert rbd.expected_cost_rate() == pytest.approx(
        3.0 * values.unit_failure_frequency + 5.0 * values.unavailability,
        rel=1e-12,
    )


def test_a_grid_that_does_not_settle_warns(monkeypatch):
    monkeypatch.setattr(_standby_warm, "MAX_STEPS", 2**9)
    with pytest.warns(UserWarning, match="did not settle"):
        rough = _standby_warm._worked_out(W([1000.0, 2.5]), E([0.01]), 0.5)
    monkeypatch.undo()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fine = _standby_warm._worked_out(W([1000.0, 2.5]), E([0.01]), 0.5)
    assert rough.unavailability == pytest.approx(fine.unavailability, rel=1e-4)


@pytest.mark.parametrize(
    "words, rbd",
    [
        (
            "both units can be repaired at once",
            pair(W([1000.0, 2.5]), E([0.01]), None, dormancy_factor=0.5),
        ),
        (
            "switching onto it may fail",
            pair(
                W([1000.0, 2.5]),
                E([0.01]),
                dormancy_factor=0.5,
                switching_probability=0.9,
            ),
        ),
        (
            "3 units",
            pair(W([1000.0, 2.5]), E([0.01]), units=3, dormancy_factor=0.5),
        ),
    ],
)
def test_what_is_not_taken_is_refused(words, rbd):
    with pytest.raises(NotImplementedError, match=words) as error:
        rbd.mean_availability()
    route = rbd.analysis_routes()["mean_availability"]
    assert route.route == "refused" and route.reason == str(error.value)
    assert "simulate it with availability()" in route.reason


def test_over_time_a_warm_pair_is_refused():
    rbd = pair(W([1000.0, 2.5]), W([300.0, 1.5]), dormancy_factor=0.5)
    with pytest.raises(NotImplementedError, match="warm spare") as error:
        rbd.point_availability([10.0])
    route = rbd.analysis_routes()["point_availability"]
    assert route.route == "refused" and route.reason == str(error.value)
