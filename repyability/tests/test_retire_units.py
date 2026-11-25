"""One unit retired inside a diagram of several (``expected_npv(
retire_at={node: t}, retired=)``): from ``t`` it is held failed (its
paths lost) or as a junction, its own costs stop, the system's go on, and
its salvage is credited at ``t``."""

import math

import pytest
import surpyval as surv
from scipy import integrate

from repyability import RepairableRBD

E = surv.Exponential.from_params
LA, MA, LB, MB = 1e-3, 0.1, 2e-3, 0.05
H, R, V, T = 5000.0, 1e-4, 10.0, 2000.0
PAIR = [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")]


def unit(rate, repair, **more):
    return {
        "reliability": E([rate]),
        "repairability": E([repair]),
        "repair_cost": 500.0,
        **more,
    }


def pair(**a):
    return RepairableRBD(
        PAIR,
        {"a": unit(LA, MA, **a), "b": unit(LB, MB)},
        downtime_cost_rate=50.0,
    )


def up(lam, mu, s):
    """A unit's availability from new: mu/(lam+mu) + lam/(lam+mu)
    e^{-(lam+mu)s}."""
    return mu / (lam + mu) + lam / (lam + mu) * math.exp(-(lam + mu) * s)


def pv(f, a, b):
    return integrate.quad(lambda s: f(s) * math.exp(-R * s), a, b)[0]


def test_a_unit_retired_failed_against_the_closed_form():
    # Before T the pair; after it b alone. a's repairs and own downtime
    # stop at T, its purchase is paid; the system's downtime goes on.
    rbd = pair(downtime_cost=2.0, acquisition_cost=100.0)
    got = rbd.expected_npv(
        H, discount_rate=R, value_rate=V, retire_at={"a": T}
    )

    def both(s):
        return 1.0 - (1.0 - up(LA, MA, s)) * (1.0 - up(LB, MB, s))

    value = V * (pv(both, 0, T) + pv(lambda s: up(LB, MB, s), T, H))
    down = 50.0 * (
        pv(lambda s: 1 - both(s), 0, T) + pv(lambda s: 1 - up(LB, MB, s), T, H)
    )
    a_costs = (
        500.0 * LA * pv(lambda s: up(LA, MA, s), 0, T)
        + 2.0 * pv(lambda s: 1 - up(LA, MA, s), 0, T)
        + 100.0
    )
    b_costs = 500.0 * LB * pv(lambda s: up(LB, MB, s), 0, H)
    hand = value - down - a_costs - b_costs
    assert math.isclose(got.npv, hand, rel_tol=1e-6)
    assert math.isclose(got.cost_by_component["a"], a_costs, rel_tol=1e-6)
    assert math.isclose(got.value, value, rel_tol=1e-6)
    assert got.retired_units == {"a": {"at": T, "as": "failed"}}
    rows = sum(row["present_value"] for row in got.cash_flows)
    assert math.isclose(rows, got.npv, rel_tol=1e-12)


def test_a_unit_retired_as_a_junction():
    # From T the system always works: no value lost, no downtime.
    got = pair().expected_npv(
        H,
        discount_rate=R,
        value_rate=V,
        retire_at={"a": T},
        retired="junction",
    )

    def both(s):
        return 1.0 - (1.0 - up(LA, MA, s)) * (1.0 - up(LB, MB, s))

    value = V * (pv(both, 0, T) + pv(lambda s: 1.0, T, H))
    assert math.isclose(got.value, value, rel_tol=1e-6)
    assert "system_downtime" in got.cost_by_category
    assert got.retired_units["a"]["as"] == "junction"


def test_retired_at_the_horizon_changes_nothing():
    rbd = pair()
    plain = rbd.expected_npv(H, discount_rate=R, value_rate=V, periods=5)
    at_end = rbd.expected_npv(
        H, discount_rate=R, value_rate=V, periods=5, retire_at={"a": H}
    )
    assert at_end.npv == plain.npv
    assert at_end.cash_flows == plain.cash_flows


def test_retired_at_0_is_the_diagram_with_it_down():
    # With no costs of its own while down, retiring a at 0 is holding it
    # broken; with them, it is that less its own downtime's cost.
    rbd = pair()
    at_0 = rbd.expected_npv(
        H, discount_rate=R, value_rate=V, retire_at={"a": 0.0}
    )
    held = rbd.expected_npv(
        H, discount_rate=R, value_rate=V, broken_nodes=["a"]
    )
    assert math.isclose(at_0.npv, held.npv, rel_tol=1e-12)
    priced = pair(downtime_cost=2.0)
    at_0 = priced.expected_npv(
        H, discount_rate=R, value_rate=V, retire_at={"a": 0.0}
    )
    held = priced.expected_npv(
        H, discount_rate=R, value_rate=V, broken_nodes=["a"]
    )
    own = 2.0 * (1.0 - math.exp(-R * H)) / R
    assert math.isclose(held.npv + own, at_0.npv, rel_tol=1e-9)


def test_its_salvage_is_credited_at_its_retirement():
    rbd = pair()
    kwargs = dict(discount_rate=R, value_rate=V, retire_at={"a": T})
    with_salvage = rbd.expected_npv(H, salvage={"a": 300.0}, **kwargs)
    without = rbd.expected_npv(H, **kwargs)
    assert math.isclose(
        with_salvage.npv - without.npv, 300.0 * math.exp(-R * T), rel_tol=1e-12
    )
    row = [r for r in with_salvage.cash_flows if r["category"] == "salvage"]
    assert row[0]["component"] == "a" and row[0]["end"] >= T


def test_the_rate_sensitivity_and_periods():
    rbd = pair()
    got = rbd.expected_npv(
        H,
        discount_rate=R,
        value_rate=V,
        retire_at={"a": T},
        periods=4,
        rate_sensitivity=True,
    )
    assert [row["period"] for row in got.cash_flows] == sorted(
        row["period"] for row in got.cash_flows
    )
    step = 1e-7
    up_ = rbd.expected_npv(
        H, discount_rate=R + step, value_rate=V, retire_at={"a": T}
    ).npv
    down_ = rbd.expected_npv(
        H, discount_rate=R - step, value_rate=V, retire_at={"a": T}
    ).npv
    assert math.isclose(
        got.rate_sensitivity, (up_ - down_) / (2 * step), rel_tol=1e-4
    )


def test_what_is_refused():
    rbd = pair()
    with pytest.raises(ValueError, match="past the horizon"):
        rbd.expected_npv(H, retire_at={"a": 2 * H})
    with pytest.raises(ValueError, match="failed"):
        rbd.expected_npv(H, retire_at={"a": T}, retired="gone")
    with pytest.raises(ValueError, match="retire_at"):
        rbd.expected_npv(H, retired="junction")
    crews = RepairableRBD(
        PAIR, {"a": unit(LA, MA), "b": unit(LB, MB)}, repair_crews=1
    )
    with pytest.raises(NotImplementedError, match="repair crews"):
        crews.expected_npv(H, retire_at={"a": T})
