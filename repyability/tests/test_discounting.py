"""Discounted total costs (#184): the components bought at the start, and
the running costs, spent at a steady rate, discounted continuously, so that
a horizon ``H`` counts as ``(1 - exp(-r H)) / r``."""

import math
import warnings

import numpy as np
import pytest
import surpyval as surv

from repyability import RepairableRBD
from repyability.tests.test_allocation_trains import HORIZON, TRAIN, station

SEVEN = math.log(1.07) / 8760  # 7% a year, per hour


def pump():
    return {
        "reliability": surv.Exponential.from_params([1e-3]),
        "repairability": surv.Exponential.from_params([0.1]),
        "repair_cost": 500.0,
        "acquisition_cost": 20000.0,
    }


def pump_line():
    return RepairableRBD(
        [("s", "pump"), ("pump", "t")],
        {"pump": pump()},
        downtime_cost_rate=100.0,
    )


def pump_pair():
    return RepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        {"a": pump(), "b": pump()},
        downtime_cost_rate=100.0,
    )


def test_the_present_value_of_a_steady_cost():
    line = pump_line()
    rate = line.expected_cost_rate()
    # Ten years at 7% a year: each year's running cost is worth 1/1.07 of
    # the year before's.
    present = (1 - 1.07**-10) / SEVEN
    assert line.total_cost(HORIZON, discount_rate=SEVEN) == pytest.approx(
        20000.0 + rate * present, rel=1e-13
    )
    assert round(line.total_cost(HORIZON, discount_rate=SEVEN)) == 114538
    assert line.total_cost(HORIZON, discount_rate=0.0) == line.total_cost(
        HORIZON
    )
    assert line.total_cost(HORIZON, discount_rate=1e-15) == pytest.approx(
        line.total_cost(HORIZON), rel=1e-9
    )
    # Discounted heavily, the running costs count for an hour or so (with
    # a warning that they are discounted away, #231).
    with pytest.warns(UserWarning, match="discounts the costs"):
        heavily = line.total_cost(HORIZON, discount_rate=1.0)
    assert heavily == pytest.approx(20000.0 + rate, rel=1e-12)
    assert line.total_cost(0.0, discount_rate=SEVEN) == 20000.0


@pytest.mark.parametrize("bad", [-1e-6, math.nan, math.inf, "0.07", True])
def test_a_discount_rate_is_checked(bad):
    with pytest.raises(ValueError, match="discount_rate"):
        pump_line().total_cost(HORIZON, discount_rate=bad)
    with pytest.raises(ValueError, match="discount_rate"):
        pump_line().allocate_redundancy(HORIZON, discount_rate=bad)


@pytest.mark.parametrize("percent, trains", [(0, 4), (7, 4), (15, 3)])
def test_discounting_can_change_the_design(percent, trains):
    # A fourth pump train pays undiscounted and at 7% a year, but not at
    # 15%: the running costs it saves are later than its purchase.
    r = math.log(1 + percent / 100) / 8760
    best = station(3).allocate_redundancy(
        HORIZON, trains=TRAIN, discount_rate=r
    )
    totals = {
        n: station(2 + n).total_cost(HORIZON, discount_rate=r)
        for n in range(1, 5)
    }
    assert best.units == {"train 1": min(totals, key=totals.get)}
    assert best.units == {"train 1": trains - 2}
    assert best.total_cost == pytest.approx(totals[trains - 2], rel=1e-12)
    assert best.discount_rate == r


def test_a_copy_that_never_pays_when_discounted():
    # The second pump saves 0.980 an hour and costs 20,000 and 0.495 an
    # hour: it pays once the horizon counts for more than 41,200 hours.
    # Discounted at 1/40,000 an hour, no horizon does.
    line = pump_line()
    for years in (10, 50, 200):
        assert line.allocate_redundancy(
            years * 8760.0, discount_rate=1 / 40_000
        ).units == {"pump": 1}
        assert line.allocate_redundancy(
            years * 8760.0, discount_rate=1 / 40_000, method="greedy"
        ).units == {"pump": 1}
    assert line.allocate_redundancy(50 * 8760.0).units == {"pump": 2}
    # Over a horizon worth more, it does.
    best = line.allocate_redundancy(50 * 8760.0, discount_rate=1 / 60_000)
    assert best.units == {"pump": 2}


# -- #231: a rate per year with models in hours, endless and many horizons --


def test_a_rate_that_discounts_the_costs_away_is_warned_of():
    line = pump_line()
    with pytest.warns(UserWarning, match=r"math.log\(1 \+ i\) / 8760") as w:
        line.total_cost(HORIZON, discount_rate=0.07)
    assert w[0].filename == __file__  # it points at the call
    with pytest.warns(UserWarning, match="discounts the costs"):
        line.allocate_redundancy(HORIZON, discount_rate=0.07)
    with pytest.warns(UserWarning, match="mean life"):
        line.total_cost(math.inf, discount_rate=0.07)
    with pytest.warns(UserWarning, match="mean life"):
        line.expected_cost(HORIZON, discount_rate=0.07)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        line.total_cost(HORIZON, discount_rate=SEVEN)
        line.total_cost(math.inf, discount_rate=SEVEN)
        line.total_cost(HORIZON)
        # A long horizon at a high rate is no mistake: 200 years at 22%.
        line.total_cost(200 * 8760.0, discount_rate=1 / 40_000)


def test_an_endless_horizon_when_discounted():
    line = pump_line()
    rate = line.expected_cost_rate()
    endless = line.total_cost(math.inf, discount_rate=SEVEN)
    assert endless == pytest.approx(20000.0 + rate / SEVEN, rel=1e-13)
    with pytest.warns(UserWarning, match="at the horizon"):
        far = line.total_cost(1e9, discount_rate=SEVEN)
    assert far == pytest.approx(endless, rel=1e-12)
    best = line.allocate_redundancy(math.inf, discount_rate=SEVEN)
    assert best.horizon == math.inf
    one, two = (
        rbd.total_cost(math.inf, discount_rate=SEVEN)
        for rbd in (line, pump_pair())
    )
    assert best.units == {"pump": 1 if one <= two else 2}
    assert best.total_cost == pytest.approx(min(one, two), rel=1e-9)
    with pytest.raises(ValueError, match="finite without a discount_rate"):
        line.total_cost(math.inf)


def test_many_horizons_at_once():
    line = pump_line()
    horizons = [0.0, 8760.0, HORIZON, math.inf]
    totals = line.total_cost(horizons, discount_rate=SEVEN)
    assert isinstance(totals, np.ndarray) and totals.shape == (4,)
    for horizon, total in zip(horizons, totals):
        assert total == line.total_cost(horizon, discount_rate=SEVEN)
    assert line.total_cost(np.array([[8760.0]])).shape == (1, 1)
    with pytest.raises(ValueError, match="horizon"):
        line.total_cost([8760.0, -1.0])
    with pytest.raises(ValueError, match="a number for allocate_redundancy"):
        line.allocate_redundancy([8760.0, HORIZON])


# -- #231: the discounted expected cost from new ----------------------------


def test_the_discounted_cost_from_new_in_closed_form():
    # One unit failing at 1e-3 and repaired at 0.1 an hour, from new, is up
    # A(t) = mu / (l + mu) + l / (l + mu) exp(-(l + mu) t): its costs fall
    # at 500 l A(t) (repairs) and 100 (1 - A(t)) (downtime) an hour.
    line = pump_line()
    lam, mu = 1e-3, 0.1
    s = lam + mu

    def present(T):
        # integral of exp(-r t) (a + b exp(-s t)), with the rates of each.
        a = 500 * lam * mu / s + 100 * lam / s
        b = 500 * lam * lam / s - 100 * lam / s
        r = SEVEN
        return a * -math.expm1(-r * T) / r + b * -math.expm1(-(r + s) * T) / (
            r + s
        )

    got = line.expected_cost([8760.0, HORIZON], discount_rate=SEVEN)
    assert got.discount_rate == SEVEN
    for T, value in zip([8760.0, HORIZON], got.mean):
        assert value == pytest.approx(present(T), rel=1e-7)
    assert got.total[1] == pytest.approx(20000.0 + present(HORIZON), rel=1e-7)
    # The long-run approximation of the same.
    assert line.total_cost(HORIZON, discount_rate=SEVEN) == pytest.approx(
        got.total[1], rel=1e-3
    )
    undiscounted = line.expected_cost(HORIZON)
    assert line.expected_cost(HORIZON, discount_rate=0.0) == undiscounted
    assert undiscounted.discount_rate == 0.0


def scheduled():
    """A block-replaced unit and a tested one: their expected costs jump at
    each replacement and test."""
    E, W = surv.Exponential.from_params, surv.Weibull.from_params
    return RepairableRBD(
        [("s", "a"), ("a", "b"), ("b", "t")],
        {
            "a": {
                "reliability": W([2000.0, 2.5]),
                "repairability": E([0.1]),
                "repair_cost": 500.0,
                "preventive": {
                    "interval": 1000.0,
                    "policy": "block",
                    "cost": 200.0,
                },
            },
            "b": {
                "reliability": E([1e-4]),
                "repairability": E([0.05]),
                "repair_cost": 800.0,
                "inspection": {"interval": 730.0, "cost": 50.0},
            },
        },
        downtime_cost_rate=20.0,
    )


def test_the_discounted_cost_from_new_with_scheduled_jumps():
    rbd = scheduled()
    T = 2 * 8760.0
    got = rbd.expected_cost(T, discount_rate=SEVEN)
    # The sum of each step's cost, discounted from its middle, on a grid
    # fine enough that it is within 1e-6.
    grid = np.linspace(0.0, T, 100_001)
    cost = rbd.expected_cost(grid)
    middle = 0.5 * (grid[1:] + grid[:-1])
    weights = np.exp(-SEVEN * middle)
    assert got.mean == pytest.approx(
        np.sum(weights * np.diff(cost.mean)), rel=2e-6
    )
    for category, values in cost.by_category.items():
        assert got.by_category[category] == pytest.approx(
            np.sum(weights * np.diff(values)), rel=2e-6, abs=1e-6
        )
    assert sum(got.by_category.values()) == pytest.approx(got.mean, rel=1e-12)
