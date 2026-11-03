"""The expected net present value of a ``RepairableRBD`` and the present
value of an age replacement: checked against closed forms (exponential
lives, a constant discount rate), the undiscounted analyses and Fox's
(1966) formula integrated independently."""

import json
import math
import warnings

import numpy as np
import pytest
import surpyval as surv
from scipy import integrate

from repyability import (
    AgeReplacementValue,
    ExpectedNPV,
    NodeState,
    NonRepairable,
    RepairableRBD,
)

LAM, MU = 1e-3, 0.1
S = LAM + MU
DELTA = 2e-4
LINE = [("s", "p"), ("p", "t")]


def pump(**extra):
    spec = {
        "reliability": surv.Exponential.from_params([LAM]),
        "repairability": surv.Exponential.from_params([MU]),
    }
    spec.update(extra)
    return RepairableRBD(LINE, {"p": spec})


def up_time(horizon, delta=DELTA, start=0.0):
    """The integral of exp(-delta t) A(t) from 0 to ``horizon`` for a pump
    new at 0 (A = MU/S + LAM/S exp(-S t))."""
    steady = MU / S * -math.expm1(-delta * horizon) / delta
    return steady + LAM / S * -math.expm1(-(S + delta) * horizon) / (S + delta)


@pytest.mark.parametrize("horizon", [50.0, 1000.0, 2e5])
def test_value_is_the_discounted_expected_time_up(horizon):
    result = pump(repair_cost=7.0).expected_npv(
        horizon, discount_rate=DELTA, value_rate=3.0
    )
    assert result.value == pytest.approx(3.0 * up_time(horizon), rel=1e-7)
    # Failures come at LAM times the availability: each repair is worth
    # the repair cost times that share of the time up.
    assert result.cost == pytest.approx(7.0 * LAM * up_time(horizon), rel=1e-7)
    assert result.npv == pytest.approx(result.value - result.cost, rel=1e-12)


def test_with_no_discounting_it_is_the_expected_time_up_and_cost():
    rbd = pump(repair_cost=7.0, acquisition_cost=500.0)
    result = rbd.expected_npv(3000.0, value_rate=2.0)
    assert result.value == pytest.approx(
        2.0 * 3000.0 * rbd.mission_availability(3000.0), rel=1e-9
    )
    cost = rbd.expected_cost(3000.0)
    assert result.cost == pytest.approx(cost.total, rel=1e-9)
    assert result.undiscounted_npv == result.npv
    assert result.annuity_factor == 3000.0
    assert result.eac == pytest.approx(cost.total / 3000.0, rel=1e-9)


def test_from_a_state_and_the_long_run():
    rbd = pump(repair_cost=7.0)
    stationary = rbd.expected_npv(
        4000.0, discount_rate=DELTA, value_rate=1.0, state="stationary"
    )
    expected = MU / S * -math.expm1(-DELTA * 4000.0) / DELTA
    assert stationary.value == pytest.approx(expected, rel=1e-7)
    down = rbd.expected_npv(
        400.0,
        discount_rate=DELTA,
        value_rate=1.0,
        state={"p": NodeState(alive=False)},
    )
    assert (
        down.value
        < rbd.expected_npv(400.0, discount_rate=DELTA, value_rate=1.0).value
    )


def test_a_discount_rate_tending_to_zero_gives_the_undiscounted_values():
    rbd = pump(repair_cost=7.0, acquisition_cost=500.0)
    plain = rbd.expected_npv(3000.0, value_rate=2.0)
    near = rbd.expected_npv(3000.0, discount_rate=1e-12, value_rate=2.0)
    assert near.npv == pytest.approx(plain.npv, rel=1e-8)
    assert near.eac == pytest.approx(plain.eac, rel=1e-8)


def test_the_equivalent_annual_cost_approaches_the_cost_rate():
    rbd = pump(repair_cost=7.0, downtime_cost=0.5)
    result = rbd.expected_npv(1e7, discount_rate=1e-12)
    assert result.eac == pytest.approx(rbd.expected_cost_rate(), rel=1e-4)


def test_events_at_fixed_times_are_discounted_at_their_exact_times():
    # Exponential lives need no age: failures come at LAM whatever the
    # replacements, each replaced at once at 11; a block replacement
    # every 100 costs 2 at 100, 200, ..., 1000 (H = 1050).
    rbd = RepairableRBD(
        LINE,
        {
            "p": {
                "reliability": surv.Exponential.from_params([LAM]),
                "repairability": "instant",
                "replace_cost": 11.0,
                "preventive": {
                    "policy": "block",
                    "interval": 100.0,
                    "cost": 2,
                },
            }
        },
    )
    horizon, rate = 1050.0, 0.003
    result = rbd.expected_npv(horizon, discount_rate=rate)
    failures = 11.0 * LAM * -math.expm1(-rate * horizon) / rate
    blocks = 2.0 * sum(math.exp(-rate * 100.0 * k) for k in range(1, 11))
    assert result.cost == pytest.approx(failures + blocks, rel=1e-7)
    assert result.cost_by_category["preventive"] == pytest.approx(
        blocks, rel=1e-7
    )


def test_it_matches_fox_for_an_age_replacement_over_a_long_horizon():
    life = surv.Weibull.from_params([1000.0, 2.5])
    age, cp, cu = 500.0, 1.0, 5.0
    rate = 1e-3
    rbd = RepairableRBD(
        LINE,
        {
            "p": {
                "reliability": life,
                "repairability": "instant",
                "replace_cost": cu,
                "preventive": {"policy": "age", "interval": age, "cost": cp},
            }
        },
    )
    part = NonRepairable(life, planned_cost=cp, unplanned_cost=cu)
    fox = part.age_replacement_value(age, discount_rate=rate)
    result = rbd.expected_npv(40_000.0, discount_rate=rate)
    assert result.cost == pytest.approx(fox.npv, rel=1e-6)


def fox_by_hand(life, age, cp, cu, delta):
    """Fox (1966), integrated independently of the package."""

    def f(t):
        return float(np.ravel(life.df(t))[0]) * math.exp(-delta * t)

    failed = integrate.quad(f, 0.0, age, epsabs=1e-13, epsrel=1e-13)[0]
    kept = math.exp(-delta * age) * float(np.ravel(life.sf(age))[0])
    return (cu * failed + cp * kept) / (1.0 - failed - kept)


@pytest.mark.parametrize("percent", [2, 8, 15])
def test_age_replacement_value_is_fox(percent):
    life = surv.Weibull.from_params([1000.0, 2.5])
    delta = math.log(1 + percent / 100) / 8760
    part = NonRepairable(life, planned_cost=1.0, unplanned_cost=5.0)
    value = part.age_replacement_value(500.0, discount_rate=delta)
    assert isinstance(value, AgeReplacementValue)
    assert value.npv == pytest.approx(
        fox_by_hand(life, 500.0, 1.0, 5.0, delta), rel=1e-8
    )
    assert value.eac == pytest.approx(delta * value.npv, rel=1e-14)


def test_age_replacement_value_undiscounted_is_the_cost_rate():
    life = surv.Weibull.from_params([1000.0, 2.5])
    part = NonRepairable(life, planned_cost=1.0, unplanned_cost=5.0)
    value = part.age_replacement_value(493.0)
    assert value.eac == pytest.approx(part.cost_rate(493.0), rel=1e-9)
    assert value.npv == math.inf
    near = part.age_replacement_value(493.0, discount_rate=1e-10)
    assert near.eac == pytest.approx(value.eac, rel=1e-6)


def test_the_cash_flow_table_adds_up():
    rbd = RepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        {
            "a": {
                "reliability": surv.Exponential.from_params([LAM]),
                "repairability": surv.Exponential.from_params([MU]),
                "repair_cost": 7.0,
                "downtime_cost": 1.0,
                "acquisition_cost": 300.0,
            },
            "b": {
                "reliability": surv.Weibull.from_params([800.0, 2.0]),
                "repairability": surv.Exponential.from_params([MU]),
                "repair_cost": 9.0,
                "acquisition_cost": 100.0,
            },
        },
        downtime_cost_rate=4.0,
    )
    result = rbd.expected_npv(
        2000.0, discount_rate=DELTA, value_rate=5.0, periods=4
    )
    rows = result.cash_flows
    assert result.period_ends == [500.0, 1000.0, 1500.0, 2000.0]
    assert sum(r["present_value"] for r in rows) == pytest.approx(
        result.npv, rel=1e-9
    )
    assert sum(r["undiscounted"] for r in rows) == pytest.approx(
        result.undiscounted_npv, rel=1e-9
    )
    # By period it is the increments of the cost over longer windows.
    for k, end in enumerate(result.period_ends, start=1):
        so_far = rbd.expected_cost(end, discount_rate=DELTA)
        cost = -sum(
            r["present_value"]
            for r in rows
            if r["category"] != "value"
            and r["category"] != "acquisition"
            and r["period"] <= k
        )
        # (the discounted costs are integrated on pieces, to ~1e-6)
        assert cost == pytest.approx(so_far.mean, rel=1e-5)
    assert result.cost_by_category["acquisition"] == 400.0
    assert set(result.cost_by_component) == {"a", "b"}
    assert sum(result.cost_by_component.values()) + result.cost_by_category[
        "system_downtime"
    ] == pytest.approx(result.cost, rel=1e-9)
    assert {r["component"] for r in rows if r["category"] == "value"} == {None}
    assert result.value == pytest.approx(
        sum(r["present_value"] for r in rows if r["category"] == "value")
    )


def test_periods_may_be_given_as_their_ends():
    rbd = pump(repair_cost=7.0)
    a = rbd.expected_npv(900.0, periods=[100.0, 400.0, 900.0])
    assert [r["end"] for r in a.cash_flows] == [100.0, 400.0, 900.0]
    assert a.cash_flows[1]["start"] == 100.0
    b = rbd.expected_npv(900.0, periods=3)
    assert b.period_ends == [300.0, 600.0, 900.0]
    assert a.npv == pytest.approx(b.npv, rel=1e-12)


def test_it_is_plain_data():
    result = pump(repair_cost=7.0).expected_npv(
        500.0, discount_rate=DELTA, value_rate=1.0, periods=2
    )
    assert isinstance(result, ExpectedNPV)
    data = json.loads(json.dumps(result.to_dict()))
    assert data["npv"] == pytest.approx(result.npv)
    assert len(data["cash_flows"]) == len(result.cash_flows)


def test_value_alone_needs_no_prices_but_something_must_be_asked():
    rbd = pump()
    assert rbd.expected_npv(100.0, value_rate=2.0).cost == 0.0
    with pytest.raises(ValueError, match="Nothing is priced"):
        rbd.expected_npv(100.0)
    assert pump(acquisition_cost=9.0).expected_npv(100.0).cost == 9.0


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"horizon": 0.0}, "horizon"),
        ({"horizon": math.inf}, "horizon"),
        ({"horizon": "10"}, "horizon"),
        ({"discount_rate": -1.0}, "discount_rate"),
        ({"discount_rate": True}, "discount_rate"),
        ({"value_rate": -1.0}, "value_rate"),
        ({"value_rate": math.nan}, "value_rate"),
        ({"periods": 0}, "periods"),
        ({"periods": 2.5}, "periods"),
        ({"periods": [50.0, 80.0]}, "periods"),
        ({"periods": [50.0, 40.0, 100.0]}, "periods"),
    ],
)
def test_arguments_are_checked(kwargs, match):
    args = {"horizon": 100.0, "value_rate": 1.0}
    args.update(kwargs)
    with pytest.raises((ValueError, TypeError), match=match):
        pump(repair_cost=1.0).expected_npv(**args)


def test_a_rate_that_discounts_the_value_away_is_warned_of():
    with pytest.warns(UserWarning, match="discounts the costs"):
        pump().expected_npv(1e4, discount_rate=1.0, value_rate=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pump(repair_cost=1.0).expected_npv(
            1e4, discount_rate=DELTA, value_rate=1.0
        )


def test_age_replacement_value_is_checked():
    part = NonRepairable(
        surv.Weibull.from_params([1000.0, 2.5]),
        planned_cost=1.0,
        unplanned_cost=5.0,
    )
    for bad in (0.0, -1.0, math.inf, "5", True):
        with pytest.raises(ValueError, match="age"):
            part.age_replacement_value(bad)
    with pytest.raises(ValueError, match="discount_rate"):
        part.age_replacement_value(5.0, discount_rate=-1.0)
    with pytest.raises(ValueError, match="costs not set"):
        NonRepairable(
            surv.Weibull.from_params([1000.0, 2.5])
        ).age_replacement_value(5.0)
