"""A plan's mid-life and end-of-life actions: an upgrade that gives a
component another life from a time, at a capital cost; retiring the
system before the horizon; and a salvage value at the end.

The references are a plain replacement at the upgrade's time (the same
life, at no cost); closed forms (memoryless units between and after the
upgrade, a unit that fails once at most, the age of a memoryless unit in
service); and the same valuation over a shorter horizon.
"""

import json
import math

import numpy as np
import pytest
import surpyval as surv

from repyability import NodeState, RepairableRBD

E = surv.Exponential.from_params
W = surv.Weibull.from_params
FIXED = surv.ExactEventTime.from_params
EDGES = [("s", "a"), ("a", "t")]
LAM, MU = 1 / 200.0, 1 / 20.0


def one(life=None, repair=None, **more):
    """A system of one component."""
    spec = {
        "reliability": E([LAM]) if life is None else life,
        "repairability": E([MU]) if repair is None else repair,
        **more,
    }
    return RepairableRBD(EDGES, {"a": spec})


def chain(t, up, lam, mu):
    """A two-state unit's probability of being up ``t`` after it was up
    with probability ``up``."""
    total = lam + mu
    return mu / total + (up - mu / total) * np.exp(-total * t)


# -- an upgrade ---------------------------------------------------------------


@pytest.mark.parametrize("repair", [E([MU]), "instant", FIXED([15.0])])
def test_an_upgrade_to_the_same_life_at_no_cost_is_a_replacement(repair):
    life = W([300, 2.0])
    upgraded = one(
        life,
        repair,
        upgrade={"at": 200.0, "reliability": life},
        repair_cost=6.0,
    )
    replaced = one(
        life, repair, preventive={"replace_at": [200.0]}, repair_cost=6.0
    )
    t = np.linspace(0.0, 900.0, 91)
    np.testing.assert_allclose(
        upgraded.point_availability(t),
        replaced.point_availability(t),
        rtol=0,
        atol=1e-10,
    )
    for T in (150.0, 200.0, 900.0):
        mine, theirs = upgraded.expected_events(T), replaced.expected_events(T)
        assert mine.node_failures["a"] == pytest.approx(
            theirs.node_failures["a"], abs=1e-10
        )
        assert upgraded.expected_cost(T).mean == pytest.approx(
            replaced.expected_cost(T).mean, abs=1e-9
        )
    assert "capital" in upgraded.expected_cost(900.0).by_category
    assert upgraded.expected_cost(900.0).by_category["capital"] == 0.0


def test_a_memoryless_unit_upgraded_to_a_better_one():
    # Up to the upgrade a two-state chain at the old rate; from it, at
    # the new rate from where the old one left it (a unit up is replaced
    # by one as good as any other of the new life; one in a repair comes
    # back with the new life).
    better, at = 1 / 800.0, 300.0
    rbd = one(
        upgrade={"at": at, "reliability": E([better]), "cost": 5000.0},
        repair_cost=10.0,
    )
    t = np.array([0.0, 50.0, 299.0, 300.0, 301.0, 320.0, 400.0, 1000.0])
    at_upgrade = chain(at, 1.0, LAM, MU)
    expected = np.where(
        t < at,
        chain(t, 1.0, LAM, MU),
        chain(np.maximum(t - at, 0.0), at_upgrade, better, MU),
    )
    np.testing.assert_allclose(rbd.point_availability(t), expected, atol=2e-6)

    def uptime(length, up, lam):
        total = lam + MU
        return MU / total * length + (up - MU / total) * (
            -np.expm1(-total * length) / total
        )

    failures = LAM * uptime(at, 1.0, LAM) + better * uptime(
        700.0, at_upgrade, better
    )
    assert rbd.expected_events(1000.0).node_failures["a"] == pytest.approx(
        failures, abs=2e-6
    )
    # The capital cost is charged at the upgrade, whatever the unit's
    # state, and discounted from there.
    rate = 1e-3
    cost = rbd.expected_cost(1000.0, discount_rate=rate)
    assert cost.by_category["capital"] == pytest.approx(
        5000.0 * math.exp(-rate * at), rel=1e-8
    )
    assert rbd.expected_cost(1000.0).mean == pytest.approx(
        5000.0 + 10.0 * failures, rel=1e-6
    )
    assert rbd.expected_cost(250.0).by_category["capital"] == 0.0
    value = rbd.expected_npv(1000.0, discount_rate=rate, periods=4)
    # (But for a rounding of the discounting integral in a later period.)
    rows = [
        r
        for r in value.cash_flows
        if r["category"] == "capital" and r["undiscounted"]
    ]
    assert [(r["period"], r["component"]) for r in rows] == [(2, "a")]
    assert rows[0]["undiscounted"] == pytest.approx(-5000.0, rel=1e-8)
    assert value.cost_by_category["capital"] == pytest.approx(
        5000.0 * math.exp(-rate * at), rel=1e-8
    )


def test_a_weibull_unit_that_fails_once_upgraded_in_closed_form():
    # Its repairs outlast the horizon: up before the upgrade if its first
    # life is, and after it if that life reached the upgrade and the new
    # unit has lasted since.
    old, new, at = W([300, 2.2]), W([900, 3.0]), 180.0
    rbd = one(old, FIXED([10_000.0]), upgrade={"at": at, "reliability": new})
    t = np.array([0.0, 60.0, 179.0, 180.0, 181.0, 300.0, 520.0])
    expected = np.where(
        t < at, old.sf(t), old.sf(at) * new.sf(np.maximum(t - at, 0.0))
    )
    np.testing.assert_allclose(rbd.point_availability(t), expected, atol=2e-6)
    assert rbd.expected_events(520.0).node_failures["a"] == pytest.approx(
        float(old.ff(at) + old.sf(at) * new.ff(520.0 - at)), abs=2e-6
    )


def test_an_upgrade_from_a_state_is_counted_from_now():
    better, at = 1 / 800.0, 300.0
    rbd = one(upgrade={"at": at, "reliability": E([better])})
    t = np.array([10.0, 299.0, 301.0, 400.0, 1000.0])
    # In a repair at 0: down, back at the repair rate.
    state = {"a": NodeState(alive=False, down_for=3.0)}
    at_upgrade = chain(at, 0.0, LAM, MU)
    expected = np.where(
        t < at,
        chain(t, 0.0, LAM, MU),
        chain(np.maximum(t - at, 0.0), at_upgrade, better, MU),
    )
    np.testing.assert_allclose(
        rbd.point_availability(t, state=state), expected, atol=2e-6
    )
    with pytest.raises(ValueError) as error:
        rbd.point_availability(t, state={"a": "stationary"})
    assert str(error.value) == (
        "Component 'a' is upgraded at a time, so it has no long-run state: "
        "give its age now (NodeState(age=...)), or leave it out (new)."
    )


def test_an_upgrade_is_saved_with_the_diagram():
    rbd = one(
        upgrade={"at": 300.0, "reliability": W([900, 2.5]), "cost": 70.0},
        repair_cost=2.0,
    )
    again = RepairableRBD.from_dict(json.loads(json.dumps(rbd.to_dict())))
    assert again._upgrades["a"].at == 300.0
    assert again._upgrades["a"].cost == 70.0
    t = np.linspace(0.0, 900.0, 19)
    np.testing.assert_allclose(
        again.point_availability(t), rbd.point_availability(t), atol=1e-12
    )
    assert again.expected_cost(900.0).mean == pytest.approx(
        rbd.expected_cost(900.0).mean, rel=1e-12
    )


NO_LONG_RUN = (
    "Component 'a' is upgraded at 300 (its units have another life from "
    "then), so the diagram has no long run with it: ask for its values "
    "over a horizon (expected_npv(horizon), expected_cost(T), "
    "mission_availability(T), point_availability(t)), or, for the long "
    "run, give the component the life it has after the upgrade."
)


def test_what_an_upgrade_is_not_taken_with_says_what_to_run():
    rbd = one(
        upgrade={"at": 300.0, "reliability": E([1e-3]), "cost": 1.0},
    )
    routes = rbd.analysis_routes()
    for call, name, message in (
        (lambda: rbd.mean_availability(), "mean_availability", NO_LONG_RUN),
        (lambda: rbd.expected_cost_rate(), "expected_cost_rate", NO_LONG_RUN),
    ):
        with pytest.raises(NotImplementedError) as error:
            call()
        assert str(error.value) == message
        assert (routes[name].route, routes[name].reason) == (
            "refused",
            message,
        )
    for name in ("point_availability", "expected_cost", "expected_npv"):
        assert routes[name].route == "numerical"
    for name in ("availability", "cost", "npv_distribution", "compare_npv"):
        assert routes[name].route == "simulated"
    with pytest.raises(NotImplementedError, match="expected_events"):
        rbd.spares_demand(100.0)


@pytest.mark.parametrize(
    "more, message",
    [
        (
            {"upgrade": {"reliability": E([1e-3])}},
            "Component 'a': an upgrade needs 'at': the time from which its "
            "units have the new life (at), and that life (reliability).",
        ),
        (
            {"upgrade": {"at": 0.0, "reliability": E([1e-3])}},
            "Component 'a': the upgrade's time (at) must be a positive, "
            "finite number, in the diagram's time from 0, got 0.0.",
        ),
        (
            {"upgrade": {"at": 5.0, "reliability": E([1e-3]), "when": 1}},
            "Component 'a': unknown upgrade key(s) ['when']; it takes at, "
            "reliability, cost.",
        ),
        (
            {
                "upgrade": {"at": 5.0, "reliability": E([1e-3])},
                "preventive": {"interval": 50.0},
            },
            "Component 'a' has an upgrade and a preventive task "
            "('preventive'), which are not taken together yet: give it "
            "one or the other.",
        ),
        (
            {"upgrade": {"at": 5.0, "reliability": 100.0}},
            "Component 'a': the upgrade's reliability must be a lifetime "
            "model (such as a fitted surpyval distribution), got 100.0.",
        ),
    ],
)
def test_an_upgrade_s_spec_is_checked(more, message):
    with pytest.raises(ValueError) as error:
        one(**more)
    assert str(error.value) == message


# -- retirement ---------------------------------------------------------------


def priced():
    return one(repair_cost=6.0, acquisition_cost=900.0)


def test_retiring_at_the_horizon_changes_nothing():
    rbd, options = priced(), dict(
        discount_rate=1e-3, value_rate=3.0, periods=4
    )
    plain = rbd.expected_npv(1000.0, **options)
    retired = rbd.expected_npv(1000.0, retire_at=1000.0, **options)
    assert retired.npv == plain.npv
    assert retired.cash_flows == plain.cash_flows
    assert retired.eac == plain.eac
    assert plain.retire_at is None and plain.salvage == 0.0


def test_retiring_early_is_the_valuation_to_the_retirement():
    rbd, rate = priced(), 1e-3
    retired = rbd.expected_npv(
        1000.0, discount_rate=rate, value_rate=3.0, periods=4, retire_at=600.0
    )
    short = rbd.expected_npv(600.0, discount_rate=rate, value_rate=3.0)
    assert retired.npv == pytest.approx(short.npv, rel=1e-9)
    assert retired.value == pytest.approx(short.value, rel=1e-9)
    assert retired.cost == pytest.approx(short.cost, rel=1e-9)
    assert retired.eac == pytest.approx(short.eac, rel=1e-9)
    assert retired.retire_at == 600.0
    # Nothing is earned or spent after the retirement: its period is the
    # last with anything in it.
    assert max(r["period"] for r in retired.cash_flows) == 3
    assert retired.period_ends == [250.0, 500.0, 750.0, 1000.0]
    assert sum(r["present_value"] for r in retired.cash_flows) == (
        pytest.approx(retired.npv, rel=1e-12)
    )


@pytest.mark.parametrize("retire_at", [1000.5, 0.0, -1.0, "600"])
def test_a_retirement_is_within_the_horizon(retire_at):
    with pytest.raises(ValueError, match="retire_at"):
        priced().expected_npv(1000.0, retire_at=retire_at)


# -- salvage ------------------------------------------------------------------


def test_a_constant_salvage_is_credited_at_the_end_and_discounted():
    rbd, rate = priced(), 1e-3
    plain = rbd.expected_npv(1000.0, discount_rate=rate, value_rate=3.0)
    for salvage, total in (
        (400.0, 400.0),
        ({"a": 250.0, None: 150.0}, 400.0),
        (-80.0, -80.0),  # a cost of disposal
    ):
        result = rbd.expected_npv(
            1000.0, discount_rate=rate, value_rate=3.0, salvage=salvage
        )
        worth = total * math.exp(-rate * 1000.0)
        assert result.salvage == pytest.approx(worth, rel=1e-12)
        assert result.undiscounted_salvage == pytest.approx(total, rel=1e-12)
        assert result.npv == pytest.approx(plain.npv + worth, rel=1e-12)
        assert result.cost == plain.cost
        assert result.undiscounted_npv == pytest.approx(
            plain.undiscounted_npv + total, rel=1e-12
        )
    rows = [r for r in result.cash_flows if r["category"] == "salvage"]
    assert [r["component"] for r in rows] == [None]
    assert rows[0]["present_value"] == pytest.approx(worth, rel=1e-12)
    # At a retirement, it is credited then.
    retired = rbd.expected_npv(
        1000.0,
        discount_rate=rate,
        value_rate=3.0,
        periods=4,
        retire_at=600.0,
        salvage=400.0,
    )
    assert retired.salvage == pytest.approx(
        400.0 * math.exp(-rate * 600.0), rel=1e-12
    )
    (row,) = [r for r in retired.cash_flows if r["category"] == "salvage"]
    assert row["period"] == 3


@pytest.mark.parametrize("repair", ["instant", "exponential"])
def test_a_salvage_by_age_of_a_memoryless_unit_in_closed_form(repair):
    # The unit in service at T has its age a with density
    # R(a) m(T - a), m the rate at which units are put in, and the age T
    # if it never failed; it is worth V exp(-k a) while it is up.
    V, k, T = 1000.0, 2e-3, 1000.0
    c = LAM + k

    def worth(age):
        return V * np.exp(-k * age)

    if repair == "instant":
        rbd = one(repair="instant", repair_cost=1.0)
        expected = V * (math.exp(-c * T) + LAM * -math.expm1(-c * T) / c)
    else:
        rbd = one(repair_cost=1.0)
        total = LAM + MU
        # Units are put in at the rate MU P(down): MU LAM / total
        # (1 - exp(-total u)).
        put_in = (
            MU
            * LAM
            / total
            * (
                -math.expm1(-c * T) / c
                - math.exp(-total * T)
                * math.expm1((total - c) * T)
                / (total - c)
            )
        )
        expected = V * (math.exp(-c * T) + put_in)
    result = rbd.expected_npv(T, discount_rate=1e-3, salvage={"a": worth})
    assert result.undiscounted_salvage == pytest.approx(expected, rel=1e-6)
    assert result.salvage == pytest.approx(
        expected * math.exp(-1e-3 * T), rel=1e-6
    )
    # A constant function of age is the constant, while the unit is up.
    flat = rbd.expected_npv(T, salvage={"a": lambda age: 0.0 * age + V})
    assert flat.undiscounted_salvage == pytest.approx(
        V * float(rbd.point_availability(T)), rel=1e-6
    )


def test_a_salvage_by_age_is_a_plain_component_s():
    def worth(age):
        return 100.0 - 0.0 * age

    replaced = one(
        W([300, 2.0]), preventive={"interval": 100.0}, repair_cost=1.0
    )
    with pytest.raises(NotImplementedError) as error:
        replaced.expected_npv(500.0, salvage={"a": worth})
    assert str(error.value) == (
        "Component 'a': a salvage by age needs the age of the unit in "
        "service at the end, which is worked out for a plain component "
        "from new (no maintenance, upgrade, tests, imperfect repair, "
        "standby, group, shutdown, crews to wait for or state at 0): give "
        "its salvage as a number."
    )
    assert replaced.expected_npv(500.0, salvage={"a": 100.0}).salvage == 100.0
    rbd = priced()
    with pytest.raises(NotImplementedError, match="give its salvage"):
        rbd.expected_npv(
            500.0, salvage={"a": worth}, state={"a": NodeState(age=5.0)}
        )
    with pytest.raises(ValueError, match="system's salvage must be a number"):
        rbd.expected_npv(500.0, salvage=worth)
    with pytest.raises(ValueError, match="finite number"):
        rbd.expected_npv(500.0, salvage={"a": math.inf})
    with pytest.raises(ValueError, match="'b'"):
        rbd.expected_npv(500.0, salvage={"b": 1.0})
    with pytest.raises(ValueError, match="array of ages"):
        rbd.expected_npv(500.0, salvage={"a": lambda age: 1.0})


# -- together -----------------------------------------------------------------


def test_is_the_upgrade_worth_it_and_when_to_retire():
    # A worn pump earning 40 an hour: upgraded at 2,000 h for 30,000 it
    # fails a quarter as often; retired at 6,000 h it is sold for 8,000.
    def pump(**more):
        return one(E([1 / 500.0]), E([1 / 50.0]), repair_cost=2_000.0, **more)

    rate, horizon = math.log(1.08) / 8760.0, 8000.0
    kept = pump().expected_npv(horizon, discount_rate=rate, value_rate=40.0)
    upgraded = pump(
        upgrade={
            "at": 2000.0,
            "reliability": E([1 / 2000.0]),
            "cost": 30_000.0,
        }
    ).expected_npv(horizon, discount_rate=rate, value_rate=40.0)

    # By hand: the availability and failure rate either side of the
    # upgrade are the two-state chains', which have all but settled.
    def worth(lam, start, stop, up):
        total = lam + 1 / 50.0
        steady = (1 / 50.0) / total
        # integral of exp(-rate t) A(t) over [start, stop]
        flat = (
            steady * (math.exp(-rate * start) - math.exp(-rate * stop)) / rate
        )
        swing = (
            (up - steady)
            * math.exp(-rate * start)
            * -math.expm1(-(total + rate) * (stop - start))
            / (total + rate)
        )
        return flat + swing

    old = worth(1 / 500.0, 0.0, horizon, 1.0)
    assert kept.npv == pytest.approx((40.0 - 2000.0 / 500.0) * old, rel=1e-6)
    before = worth(1 / 500.0, 0.0, 2000.0, 1.0)
    at_upgrade = chain(2000.0, 1.0, 1 / 500.0, 1 / 50.0)
    after = worth(1 / 2000.0, 2000.0, horizon, at_upgrade)
    by_hand = (
        (40.0 - 2000.0 / 500.0) * before
        + (40.0 - 2000.0 / 2000.0) * after
        - 30_000.0 * math.exp(-rate * 2000.0)
    )
    assert upgraded.npv == pytest.approx(by_hand, rel=1e-6)
    assert upgraded.npv > kept.npv
    retired = pump().expected_npv(
        horizon,
        discount_rate=rate,
        value_rate=40.0,
        retire_at=6000.0,
        salvage=8_000.0,
    )
    by_hand = (40.0 - 2000.0 / 500.0) * worth(
        1 / 500.0, 0.0, 6000.0, 1.0
    ) + 8_000.0 * math.exp(-rate * 6000.0)
    assert retired.npv == pytest.approx(by_hand, rel=1e-6)


def test_an_upgrade_s_time_is_a_lever():
    rbd = one(
        upgrade={"at": 300.0, "reliability": E([1 / 800.0]), "cost": 9.0},
        repair_cost=1.0,
    )
    (lever,) = [x for x in rbd.levers() if x.name == "upgrade.at"]
    assert (lever.key, lever.value, lever.bounds) == (
        "a",
        300.0,
        (0, math.inf),
    )
    later = rbd.with_levers({("a", "upgrade.at"): 400.0})
    assert later._upgrades["a"].at == 400.0
    window, step = 1000.0, 0.5
    found = rbd.parameter_sensitivity(window=window)["a"]["upgrade.at"]
    high = rbd.with_levers({("a", "upgrade.at"): 300.0 + step})
    low = rbd.with_levers({("a", "upgrade.at"): 300.0 - step})
    slope = (
        high.mission_availability(window) - low.mission_availability(window)
    ) / (2.0 * step)
    assert found == pytest.approx(slope, rel=0.02, abs=1e-9)
