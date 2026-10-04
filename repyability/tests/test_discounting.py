"""Discounted total costs (#184): the components bought at the start, and
the running costs, spent at a steady rate, discounted continuously, so that
a horizon ``H`` counts as ``(1 - exp(-r H)) / r``."""

import math

import pytest
import surpyval as surv

from repyability import RepairableRBD
from repyability.tests.test_allocation_trains import HORIZON, TRAIN, station

SEVEN = math.log(1.07) / 8760  # 7% a year, per hour


def pump_line():
    return RepairableRBD(
        [("s", "pump"), ("pump", "t")],
        {
            "pump": {
                "reliability": surv.Exponential.from_params([1e-3]),
                "repairability": surv.Exponential.from_params([0.1]),
                "repair_cost": 500.0,
                "acquisition_cost": 20000.0,
            }
        },
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
    # Discounted heavily, the running costs count for an hour or so.
    assert line.total_cost(HORIZON, discount_rate=1.0) == pytest.approx(
        20000.0 + rate, rel=1e-12
    )
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
