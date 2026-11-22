"""Value and prices over time in a net present value (UC-57 of the
maintenance README): a value earned in steps (``ValueSchedule``), costs
and value that escalate, and rates that are real or nominal; and the
same escalation on dates, in ``repyability.schedule``."""

import json
import math
from datetime import date

import numpy as np
import pytest
import surpyval as surv

from repyability import RepairableRBD, ValueSchedule
from repyability.schedule import (
    Recurrence,
    Task,
    UsageForecast,
    events_by_period,
    work_by_period,
)

E = surv.Exponential.from_params
LAMBDA, MU, PRICE = 1e-3, 0.1, 500.0
T = 5 * 8760.0
R = math.log(1.08) / 8760
G = math.log(1.05) / 8760


def pump(**more):
    return RepairableRBD(
        [("s", "p"), ("p", "t")],
        {
            "p": {
                "reliability": E([LAMBDA]),
                "repairability": E([MU]),
                "repair_cost": PRICE,
                "acquisition_cost": 20000.0,
                **more,
            }
        },
    )


def up_time(rate, start=0.0, end=T):
    """The integral of exp(-rate t) A(t) from ``start`` to ``end``, in
    closed form (A = MU/S + LAMBDA/S exp(-S t), S = LAMBDA + MU)."""
    s = LAMBDA + MU

    def steady(d, t):
        return t if d == 0.0 else -math.expm1(-d * t) / d

    def to(t):
        return MU / s * steady(rate, t) + LAMBDA / s * steady(rate + s, t)

    return to(end) - to(start)


def test_a_growing_value_is_the_closed_form_s():
    # The value's part, apart from the costs, which do not change.
    rbd = pump()
    plain = rbd.expected_npv(T, discount_rate=R, value_rate=10.0)
    grown = rbd.expected_npv(
        T, discount_rate=R, value_rate=10.0, escalation={"value": G}
    )
    assert grown.value == pytest.approx(10.0 * up_time(R - G), rel=1e-7)
    assert grown.cost == plain.cost
    assert grown.undiscounted_value == pytest.approx(
        10.0 * up_time(-G), rel=1e-7
    )


@pytest.mark.parametrize("g", [G, 2.5 * R, -G])
def test_an_escalated_cost_is_the_cost_discounted_at_the_rate_less_it(g):
    # Also a growth above the rate: a rate below 0, over a horizon.
    rbd = pump()
    grown = rbd.expected_npv(
        T, discount_rate=R, value_rate=10.0, escalation={"repair": g}
    )
    exact = PRICE * LAMBDA * up_time(R - g)
    assert grown.cost_by_category["repair"] == pytest.approx(exact, rel=1e-7)
    assert grown.cost_by_category["acquisition"] == 20000.0


def test_a_component_s_growth_comes_before_its_category_s():
    rbd = pump()
    a = rbd.expected_npv(T, discount_rate=R, escalation={"p": G})
    b = rbd.expected_npv(
        T, discount_rate=R, escalation={"p": G, "repair": 3 * G}
    )
    c = rbd.expected_npv(T, discount_rate=R, escalation={"repair": G})
    assert a.npv == b.npv == c.npv


def test_steps_of_value_are_the_sum_of_the_steps_valued_apart():
    rbd = pump()
    schedule = ValueSchedule([0.0, 8760.0], [10.0, 12.0])
    stepped = rbd.expected_npv(
        T, discount_rate=R, value_rate=schedule, periods=5
    )
    exact = 10.0 * up_time(R, 0.0, 8760.0) + 12.0 * up_time(R, 8760.0, T)
    assert stepped.value == pytest.approx(exact, rel=1e-7)
    first = rbd.expected_npv(8760.0, discount_rate=R, value_rate=10.0)
    whole = rbd.expected_npv(T, discount_rate=R, value_rate=12.0)
    early = rbd.expected_npv(8760.0, discount_rate=R, value_rate=12.0)
    assert stepped.value == pytest.approx(
        first.value + whole.value - early.value, rel=1e-9
    )
    values = [r for r in stepped.cash_flows if r["category"] == "value"]
    assert len(values) == 5
    assert sum(r["present_value"] for r in values) == pytest.approx(
        stepped.value, rel=1e-12
    )
    assert math.isnan(stepped.value_rate)
    assert stepped.value_schedule == schedule
    out = json.loads(json.dumps(stepped.to_dict()))
    assert out["value_schedule"] == {
        "times": [0.0, 8760.0],
        "rates": [10.0, 12.0],
    }


def test_inflation_on_every_stream_leaves_the_npv_as_it_is():
    rbd = pump(replace_cost=100.0)
    i = math.log(1.03) / 8760
    kwargs = dict(value_rate=10.0, salvage=1000.0, periods=5)
    real = rbd.expected_npv(
        T,
        discount_rate=R,
        escalation={"repair": G},
        rates="real",
        inflation=i,
        **kwargs,
    )
    nominal = rbd.expected_npv(
        T,
        discount_rate=R + i,
        escalation={
            "repair": G + i,
            "replace": i,
            "value": i,
            "salvage": i,
        },
        **kwargs,
    )
    in_real_terms = rbd.expected_npv(
        T, discount_rate=R, escalation={"repair": G}, **kwargs
    )
    for result in (nominal, in_real_terms):
        assert real.npv == pytest.approx(result.npv, rel=1e-12)
    # Its undiscounted amounts are nominal, and its rate the real one.
    assert real.undiscounted_npv == pytest.approx(
        nominal.undiscounted_npv, rel=1e-12
    )
    assert real.discount_rate == R
    assert (real.rate_basis, real.inflation) == ("real", i)
    assert real.eac == pytest.approx(in_real_terms.eac, rel=1e-12)


def test_no_growth_and_one_rate_change_nothing():
    rbd = pump(replace_cost=100.0)
    kwargs = dict(discount_rate=R, value_rate=10.0, periods=5, salvage=50.0)
    before = rbd.expected_npv(T, **kwargs).to_dict()
    after = rbd.expected_npv(T, escalation={}, **kwargs).to_dict()
    assert after == before
    assert before["escalation"] == {} and before["rate_basis"] == "nominal"


def test_a_salvage_grows_at_its_own_rate():
    rbd = pump()
    plain = rbd.expected_npv(T, discount_rate=R, salvage=1000.0)
    grown = rbd.expected_npv(
        T, discount_rate=R, salvage=1000.0, escalation={"salvage": G}
    )
    assert grown.salvage == pytest.approx(
        plain.salvage * math.exp(G * T), rel=1e-12
    )
    assert grown.undiscounted_salvage == pytest.approx(
        1000.0 * math.exp(G * T), rel=1e-12
    )


@pytest.mark.parametrize(
    "kwargs, error, match",
    [
        ({"escalation": {"x": 0.1}}, ValueError, "no component"),
        ({"escalation": {"repair": "fast"}}, ValueError, "finite growth"),
        ({"escalation": 0.1}, ValueError, "must be a dict"),
        ({"rates": "both"}, ValueError, "rates"),
        ({"inflation": 0.01}, ValueError, 'rates="real"'),
        ({"value_rate": lambda t: 1.0}, ValueError, "ValueSchedule"),
    ],
)
def test_what_is_refused(kwargs, error, match):
    with pytest.raises(error, match=match):
        pump().expected_npv(T, **{"value_rate": 1.0, **kwargs})


def test_a_schedule_is_steps_of_plain_numbers():
    with pytest.raises(TypeError, match="plain numbers"):
        ValueSchedule(lambda t: t, [1.0])
    with pytest.raises(ValueError, match="the first 0"):
        ValueSchedule([1.0, 2.0], [1.0, 1.0])
    with pytest.raises(ValueError, match="one for each|for each of"):
        ValueSchedule([0.0, 2.0], [1.0])


# -- on dates -----------------------------------------------------------------


def test_a_task_s_present_value_takes_its_cost_grown():
    task = Task(
        "grease",
        every=Recurrence.every(30, "days", "2026-11-20"),
        cost=250.0,
    )
    due = task.due("2026-11-20", "2028-11-19", discount=0.07, escalation=0.03)
    start = date(2026, 11, 20)
    by_hand = sum(
        250.0 * (1.03 / 1.07) ** ((day - start).days / 365.25)
        for day in due.done
    )
    assert due.npv == pytest.approx(by_hand, rel=1e-12)
    assert due.discount == 0.07
    assert "escalate at 3% a year" in " ".join(due.notes)
    plain = task.due("2026-11-20", "2028-11-19", discount=0.07)
    assert due.npv > plain.npv
    assert due.cost == plain.cost  # at today's prices
    work = work_by_period(
        [task],
        "year",
        "2026-11-20",
        "2028-11-19",
        discount=0.07,
        escalation=0.03,
    )
    assert work.total_npv == pytest.approx(by_hand, rel=1e-12)


def test_for_ever_needs_the_escalation_below_the_discount():
    task = Task(
        "grease",
        every=Recurrence.every(30, "days", "2026-11-20"),
        cost=250.0,
    )
    grown = task.long_run_value("2026-11-20", 0.07, escalation=0.03)
    net = 1.07 / 1.03 - 1.0
    assert grown.npv == pytest.approx(
        task.long_run_value("2026-11-20", net).npv, rel=1e-12
    )
    with pytest.raises(ValueError, match="without end"):
        task.long_run_value("2026-11-20", 0.03, escalation=0.03)
    with pytest.raises(ValueError, match="give a discount"):
        task.due("2026-11-20", "2027-11-19", escalation=0.03)


@pytest.mark.parametrize("escalation", [0.03, 0.12])
def test_a_system_s_costs_on_dates_are_expected_npv_s(escalation):
    # The diagram's clock is calendar time in hours: the yearly rates are
    # continuous ones per hour, and a growth above the discount (12%
    # against 7%) is a rate below 0 over the window.
    rbd = RepairableRBD(
        [("s", "p"), ("p", "t")],
        {
            "p": {
                "reliability": E([1e-3]),
                "repairability": E([0.1]),
                "repair_cost": 500.0,
            }
        },
    )
    usage = UsageForecast.calendar("2027-01-01", "2029-01-01")
    found = events_by_period(
        rbd,
        usage,
        "year",
        unit="hours",
        cost=True,
        mc_samples=50,
        seed=1,
        discount=0.07,
        escalation=escalation,
    )
    hours = 24.0 * 365.25
    rate = math.log(1.07) / hours
    growth = math.log(1.0 + escalation) / hours
    end = float(found.model_ends[-1])
    npv = rbd.expected_npv(
        end, discount_rate=rate, escalation={"repair": growth}
    )
    assert found.npv == pytest.approx(npv.cost, rel=1e-6)
    assert np.all(np.diff(found.expected_npv) != 0.0)
