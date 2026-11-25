"""The long-run spares stock with the units waiting for a spare making no
demand (``spares_stock``'s finite source): exact for a constant failure
rate replaced in no time, approximate otherwise, each against the shelf
in the simulation (``repyability.rbd._shelves``), stocks 0 to 5, a short
lead time and a long one."""

import dataclasses
import math

import numpy as np
import pytest
import surpyval as surv

from repyability import RepairableRBD
from repyability.rbd import _event_loop, _shelves

E = surv.Exponential.from_params
W = surv.Weibull.from_params
FAIL = 1e-3
#: Long runs, so the start from new with a full shelf is a small part.
HORIZON, RUNS = 4_000_000.0, 20
SHORT, LONG = 300.0, 3000.0


def diagram(spec, copies=1):
    """``copies`` units of ``spec`` side by side (their lives apart)."""
    names = ["p"] + [f"p (copy {i})" for i in range(1, copies)]
    return (
        RepairableRBD(
            [("s", n) for n in names] + [(n, "t") for n in names],
            {n: dict(spec) for n in names},
        ),
        names,
    )


def simulated(rbd, names, stock, lead, seed):
    """The fill rate and the share of time out of stock of one shelf the
    units ``names`` draw on, with their standard errors."""
    shelf = _shelves.Shelves(
        {"p": _shelves.shelf(stock, lead, False)}, {n: "p" for n in names}
    )
    tally = _event_loop._run(
        _shelves.shelved(rbd, shelf),
        HORIZON,
        set(),
        set(),
        "p",
        RUNS,
        False,
        seed,
    )
    rows = [run["p"] for run in tally.shelves]
    demands = np.array([r.demands for r in rows], float)
    waited = np.array([r.stockouts for r in rows], float)
    rate = waited.sum() / demands.sum()
    rate_se = math.sqrt(((waited - rate * demands) ** 2).sum()) / (
        demands.sum()
    )
    out = np.array([r.out_of_stock for r in rows]) / HORIZON
    return 1 - rate, rate_se, out.mean(), out.std(ddof=1) / math.sqrt(RUNS)


def long_run(spec, lead, copies=1):
    rbd, _ = diagram(spec)
    return rbd.spares_stock(lead, fill_rate=0.5, fleet=copies)["p"]


def close(got, want, se):
    # Within four standard errors (a value the runs never vary is the
    # long run's to a hair).
    return abs(got - want) <= 4 * se + 1e-4


CONSTANT = {"reliability": E([FAIL]), "repairability": "instant"}


@pytest.mark.parametrize("lead", [SHORT, LONG])
@pytest.mark.parametrize("copies", [1, 3])
def test_a_constant_rate_matches_the_shelf_exactly(lead, copies):
    stock = long_run(CONSTANT, lead, copies)
    assert stock.exact and stock.sources == copies
    rbd, names = diagram(CONSTANT, copies)
    for s in range(6):
        fill, fill_se, out, out_se = simulated(rbd, names, s, lead, 11 + s)
        assert close(stock.fill_rate_for(s), fill, fill_se), s
        assert close(stock.stockout_probability_for(s), out, out_se), s


def test_the_one_spare_case_before_and_after():
    # The value the demand going on gave, exp(-0.3) = 0.7408, against the
    # finite source's 1 / 1.3 = 0.7692, the shelf's within its error.
    stock = long_run(CONSTANT, SHORT)
    going_on = dataclasses.replace(stock, sources=None)
    assert going_on.fill_rate_for(1) == pytest.approx(0.740818, abs=1e-6)
    assert stock.fill_rate_for(1) == pytest.approx(0.769231, abs=1e-6)
    rbd, names = diagram(CONSTANT)
    fill, se, _, _ = simulated(rbd, names, 1, SHORT, 12)
    assert close(stock.fill_rate_for(1), fill, se)
    assert fill - going_on.fill_rate_for(1) > 10 * se


REPAIRED = {"reliability": E([FAIL]), "repairability": E([0.1])}
WEAR_OUT = {"reliability": W([1000.0, 3.0]), "repairability": "instant"}


@pytest.mark.parametrize("spec", [REPAIRED, WEAR_OUT], ids=["repair", "wear"])
def test_the_approximation_with_a_short_lead_time(spec):
    # Repairs that take time, or wear-out lives: approximate, and with a
    # lead time short of a life within the shelf's error.
    stock = long_run(spec, SHORT)
    assert not stock.exact
    rbd, names = diagram(spec)
    for s in range(6):
        fill, fill_se, out, out_se = simulated(rbd, names, s, SHORT, 21 + s)
        assert close(stock.fill_rate_for(s), fill, fill_se), s
        assert close(stock.stockout_probability_for(s), out, out_se), s


@pytest.mark.parametrize("spec", [REPAIRED, WEAR_OUT], ids=["repair", "wear"])
def test_the_approximation_with_a_long_lead_time_is_on_the_safe_side(spec):
    # A lead time of three lives: the approximation is short of the shelf's
    # fill rate (holding more spares, never fewer), and above what the
    # demand going on gave. With a constant rate it is within 0.01; with
    # wear-out lives, whose next demands the waits put in step with the
    # orders' arrivals, it is far short when stock-outs are common.
    stock = long_run(spec, LONG)
    going_on = dataclasses.replace(stock, sources=None)
    rbd, names = diagram(spec)
    for s in range(1, 6):
        fill, se, _, _ = simulated(rbd, names, s, LONG, 31 + s)
        held = stock.fill_rate_for(s)
        assert going_on.fill_rate_for(s) - 1e-12 <= held <= fill + 4 * se
        if spec is REPAIRED:
            assert fill - held < 0.01


def test_a_part_of_identical_constant_rates_is_exact():
    # A part of two positions with one rate is two units on one shelf, as
    # a fleet of two; with two rates it is approximate.
    rbd, names = diagram(CONSTANT, 2)
    part = rbd.spares_stock(LONG, fill_rate=0.5, parts={"seal": names})
    fleet = long_run(CONSTANT, LONG, 2)
    assert part["seal"].exact and part["seal"].sources == 2
    for s in range(8):
        assert part["seal"].fill_rate_for(s) == pytest.approx(
            fleet.fill_rate_for(s), abs=1e-12
        )
    mixed = RepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        {"a": dict(CONSTANT), "b": {**CONSTANT, "reliability": E([2e-3])}},
    )
    found = mixed.spares_stock(
        SHORT, fill_rate=0.5, fleet=2, parts={"seal": ["a", "b"]}
    )["seal"]
    assert not found.exact and found.sources == 4


def test_wear_out_with_a_long_lead_is_simulated_by_default():
    # Weibull(1000, 3), a lead time of 3,000 h (3.4 lives), one unit, an
    # 85% fill rate: the reweighting would hold 5 (0.824 at 4); the shelf
    # in the runs meets it with 4. The long run finds it so by default,
    # and says why; each stock tried against long runs of the shelf.
    rbd, names = diagram(WEAR_OUT)
    found = rbd.spares_stock(LONG, fill_rate=0.85)["p"]
    assert found.method == "simulated" and not found.exact
    assert "in step with the orders' arrivals" in found.reason
    assert found.stock == 4
    assert found.fill_rate == found.fill_rate_for(4) >= 0.85
    for s in range(1, 5):
        fill, fill_se, out, out_se = simulated(rbd, names, s, LONG, 41 + s)
        # The default's runs are shorter (about 500 demands each): their
        # own error is about as large again.
        assert close(found.fill_rate_for(s), fill, 3 * fill_se), s
        assert close(found.stockout_probability_for(s), out, 3 * out_se), s
    again = rbd.spares_stock(LONG, fill_rate=0.85)["p"]
    assert again.simulated == found.simulated
    # A short lead time, and constant rates, stay worked out.
    assert long_run(WEAR_OUT, SHORT).method == "approximate"
    assert long_run(CONSTANT, LONG).method == "exact"
    assert long_run(REPAIRED, LONG).method == "approximate"
