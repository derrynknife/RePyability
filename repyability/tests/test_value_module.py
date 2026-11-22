"""The money module (``repyability.value``): one convention for every
present value, converted from a yearly rate in one place, and the one
definition of each money helper the diagrams, the maintenance module and
the schedule module use."""

import math
from datetime import date
from fractions import Fraction

import numpy as np
import pytest

from repyability import ValueSchedule
from repyability.maintenance import _horizon
from repyability.rbd import _common, _npv, _pricing, _tally
from repyability.rbd import value_schedule as old_value_schedule
from repyability.schedule import _discount
from repyability.value import _escalation, _flows, _rates
from repyability.value._categories import CATEGORIES


def test_each_helper_has_one_definition():
    # The old homes hand out the money module's objects, not copies.
    assert _common._discount_rate is _rates.discount_rate
    assert _common._horizons is _rates.horizons
    assert _common._present_horizon is _rates.present_horizon
    assert _horizon._rate is _rates.continuous
    assert _horizon._equivalent is _rates.equivalent
    for name in (
        "yearly",
        "net",
        "continuous",
        "per_unit",
        "factor",
        "factor_on",
        "years_from",
        "equivalent_annual",
        "perpetuity",
        "DAYS_A_YEAR",
        "HOURS_A_YEAR",
    ):
        assert getattr(_discount, name) is getattr(_rates, name)
    assert _pricing.Ledger is _flows.Ledger
    assert _pricing.row is _flows.row
    assert _pricing.sort_rows is _flows.sort_rows
    assert _pricing.discounted_span is _rates.discounted_span
    assert _tally._CATEGORIES is CATEGORIES
    assert _npv._inflation is _escalation.inflation
    assert old_value_schedule.ValueSchedule is ValueSchedule


def test_the_categories_keep_their_order():
    # Their places index the engines' arrays and seeded costs.
    assert CATEGORIES == (
        "repair",
        "replace",
        "preventive",
        "inspection",
        "component_downtime",
        "system_downtime",
        "setup",
        "shutdown",
        "crews",
        "capital",
    )
    assert list(_flows.ORDER) == ["acquisition", *CATEGORIES, "value"]


def test_a_yearly_rate_is_converted_in_one_place():
    assert _rates.continuous(0.07) == math.log1p(0.07)
    # Per hour, a year being 365.25 days.
    assert _rates.HOURS_A_YEAR == 8766.0
    assert _rates.per_unit(0.07, 1.0) == math.log1p(0.07) / 8766.0
    assert _rates.per_unit(0.07, 24.0) == pytest.approx(
        math.log1p(0.07) / 365.25, rel=1e-15
    )
    # Discounting continuously at the converted rate is the yearly
    # convention: (1 + i) ** -y.
    for years in (0.5, 1.0, 3.0, 10.0):
        assert _rates.factor(0.07, years) == pytest.approx(
            1.07**-years, rel=1e-14
        )
    days = (date(2027, 11, 21) - date(2026, 11, 21)).days
    assert _rates.factor_on(
        0.07, date(2026, 11, 21), date(2027, 11, 21)
    ) == pytest.approx(1.07 ** -(days / 365.25), rel=1e-14)


def test_annuities_and_perpetuities_against_closed_forms():
    r = 0.01
    # A unit rate over 50 units: (1 - e^{-50 r}) / r.
    out = _rates.present_horizon(np.array([50.0, np.inf]), r)
    assert out[0] == pytest.approx(-math.expm1(-0.5) / r, rel=1e-15)
    assert out[1] == 1.0 / r
    # Undiscounted, the horizon itself.
    assert _rates.present_horizon(np.array([7.0]), 0.0)[0] == 7.0
    # The steady cost worth a present value inverts the annuity.
    npc = float(out[0])
    assert _rates.equivalent(npc, r, 50.0) == pytest.approx(1.0, rel=1e-14)
    assert _rates.equivalent(300.0, 0.0, 2.0) == 150.0
    # One a year from year 1, for ever, at 7%: 1 / 0.07.
    assert _rates.perpetuity(0.07, 1.0, 1.0) == pytest.approx(
        1 / 0.07, rel=1e-13
    )
    # A unit rate from a to b: (e^{-ra} - e^{-rb}) / r.
    span = _rates.discounted_span(10.0, 30.0, r)
    assert span == pytest.approx(
        (math.exp(-0.1) - math.exp(-0.3)) / r, rel=1e-14
    )
    assert _rates.discounted_span(10.0, 30.0, 0.0) == 20.0


def test_net_rate_of_escalating_costs():
    # (1 + i) / (1 + g) - 1, exactly as fractions.
    expected = Fraction(107, 100) / Fraction(103, 100) - 1
    assert _rates.net(0.07, 0.03) == pytest.approx(float(expected), 1e-15)
    assert _rates.net(0.07, None) == 0.07
    with pytest.raises(ValueError, match="give a discount= too"):
        _rates.net(None, 0.03)


def test_escalation_and_inflation_checks():
    growths = _escalation.escalation(("pump",), {"repair": 0.01, "pump": 2})
    assert growths == {"repair": 0.01, "pump": 2.0}
    assert _escalation.growth(growths, 0.5, "pump", "repair") == 2.5
    assert _escalation.growth(growths, 0.5, "valve", "repair") == 0.51
    assert _escalation.growth(growths, 0.5, "value") == 0.5
    with pytest.raises(ValueError, match="which is no component"):
        _escalation.escalation(("pump",), {"valve": 0.01})
    with pytest.raises(ValueError, match="must be a dict"):
        _escalation.escalation(("pump",), 0.01)
    assert _escalation.inflation(None, "nominal") == 0.0
    assert _escalation.inflation(0.02, "real") == 0.02
    with pytest.raises(ValueError, match='inflation is for rates="real"'):
        _escalation.inflation(0.02, "nominal")


def test_the_ledger_discounts_events_and_spans():
    r = 0.1
    ledger = _flows.Ledger(np.array([1.0, 2.0]), r, size=2)
    ledger.events([0, 1, 1], [0.5, 0.5, 1.5], 10.0, "repair", "pump")
    ledger.spans([0], [0.5], [1.5], 4.0, "value", None)
    totals = ledger.totals()
    present, nominal = totals[("repair", "pump")]
    assert present[0] == pytest.approx(10 * math.exp(-0.05), rel=1e-15)
    assert present[1] == pytest.approx(
        10 * (math.exp(-0.05) + math.exp(-0.15)), rel=1e-15
    )
    assert list(nominal) == [10.0, 20.0]
    value, _ = totals[("value", None)]
    assert value[0] == pytest.approx(
        4 * (math.exp(-0.05) - math.exp(-0.15)) / r, rel=1e-14
    )
    rows = ledger.rows(0)
    # A cell is kept for every history once any history has an amount in
    # it: history 0 has none in period 2's repairs.
    assert [(row["period"], row["category"]) for row in rows] == [
        (1, "repair"),
        (1, "value"),
        (2, "repair"),
        (2, "value"),
    ]
    assert rows[2]["present_value"] == 0.0
    # Costs are negative in the table, value positive.
    assert rows[0]["present_value"] < 0 < rows[1]["present_value"]
