"""A plan against what happened (``RepairableRBD.plan_vs_actual``): an
outage log priced by hand, the states read off a log by hand, and two
independent checks by simulation: the log pricing of simulated histories
averages to the plan's exact table (the two mean the same thing), and
what happened so far plus the rest valued from each history's state
averages to the plan's value (the tower property, which checks the
states read off the logs)."""

import json
import math
import warnings

import numpy as np
import pytest
import surpyval as surv

from repyability import (
    NodeState,
    PlanVsActual,
    RepairableRBD,
    Timeline,
)
from repyability.rbd import _npv, _plan_actual

E = surv.Exponential.from_params
W = surv.Weibull.from_params
LINE = [("s", "p"), ("p", "t")]
PAIR = [("s", "a"), ("s", "b"), ("a", "v"), ("b", "v"), ("v", "t")]


def pump(scale=1.0, **extra):
    """An exponential pump, MTTF 1000 and MTTR 10 (times ``scale``)."""
    spec = {
        "reliability": E([1e-3 / scale]),
        "repairability": E([0.1 / scale]),
        "repair_cost": 500.0,
        "replace_cost": 200.0,
        "downtime_cost": 20.0 / scale,
    }
    spec.update(extra)
    return RepairableRBD(LINE, {"p": spec}, downtime_cost_rate=100.0 / scale)


def span(a, b, r):
    """The integral of exp(-r t) from a to b."""
    return (math.exp(-r * a) - math.exp(-r * b)) / r


# -- a log priced by hand ---------------------------------------------------

R = 1e-4
HAND_LOG = [(300.0, 310.0), (1200.0, 1250.0), (1950.0, None)]


def hand_result():
    return pump().plan_vs_actual(
        {"p": Timeline.from_outages(HAND_LOG, end=2000.0)},
        2000.0,
        5000.0,
        discount_rate=R,
        value_rate=10.0,
        periods=[1000.0, 2000.0, 5000.0],
    )


def test_a_log_is_priced_by_hand():
    result = hand_result()
    # Failures at 300, 1200 and 1950; discount factors exp(-r t):
    # exp(-0.03) = 0.970446, exp(-0.12) = 0.886920, exp(-0.195) =
    # 0.822835. Down spans [300, 310], [1200, 1250], [1950, 2000]; up
    # spans [0, 300], [310, 1200], [1250, 1950], cut at 1000 for the
    # periods. A span's rate is worth rate x (exp(-r a) - exp(-r b)) / r:
    # e.g. [300, 310] counts 9.9853.
    d1 = span(300, 310, R)  # 9.9853
    d2 = span(1200, 1250, R) + span(1950, 2000, R)  # 44.135 + 41.009
    hand = {
        (1, "repair"): -500 * math.exp(-0.03),  # -485.22
        (1, "replace"): -200 * math.exp(-0.03),  # -194.09
        (1, "component_downtime"): -20 * d1,  # -199.71
        (1, "system_downtime"): -100 * d1,  # -998.53
        (1, "value"): 10 * (span(0, 300, R) + span(310, 1000, R)),
        (2, "repair"): -500 * (math.exp(-0.12) + math.exp(-0.195)),
        (2, "replace"): -200 * (math.exp(-0.12) + math.exp(-0.195)),
        (2, "component_downtime"): -20 * d2,
        (2, "system_downtime"): -100 * d2,
        (2, "value"): 10 * (span(1000, 1200, R) + span(1250, 1950, R)),
    }
    got = {
        (r["period"], r["category"]): r["present_value"]
        for r in result.actual_cash_flows
    }
    assert got == pytest.approx(hand, rel=1e-12)
    # Undiscounted: 1 failure then 2; down 10 then 100; up 990 then 900.
    nominal = {
        (r["period"], r["category"]): r["undiscounted"]
        for r in result.actual_cash_flows
    }
    assert nominal == pytest.approx(
        {
            (1, "repair"): -500.0,
            (1, "replace"): -200.0,
            (1, "component_downtime"): -200.0,
            (1, "system_downtime"): -1000.0,
            (1, "value"): 9900.0,
            (2, "repair"): -1000.0,
            (2, "replace"): -400.0,
            (2, "component_downtime"): -2000.0,
            (2, "system_downtime"): -10000.0,
            (2, "value"): 9000.0,
        }
    )
    assert result.actual_so_far == pytest.approx(sum(hand.values()))
    assert result.undiscounted_actual_so_far == pytest.approx(
        sum(nominal.values())
    )
    # Its components and the system: the pump (component None is the
    # system's own).
    assert {r["component"] for r in result.actual_cash_flows} == {"p", None}


def test_the_variance_is_actual_less_the_plan_s_table():
    result = hand_result()
    plan = pump().expected_npv(
        5000.0,
        discount_rate=R,
        value_rate=10.0,
        periods=[1000.0, 2000.0, 5000.0],
    )
    so_far = [r for r in plan.cash_flows if r["period"] <= 2]
    assert result.expected_cash_flows == so_far
    assert result.plan.npv == pytest.approx(plan.npv, rel=1e-12)
    actual = {
        (r["period"], r["category"]): r["present_value"]
        for r in result.actual_cash_flows
    }
    expected = {
        (r["period"], r["category"]): r["present_value"] for r in so_far
    }
    for r in result.variance_cash_flows:
        key = (r["period"], r["category"])
        assert r["present_value"] == pytest.approx(
            actual.get(key, 0.0) - expected.get(key, 0.0), abs=1e-9
        )
    assert result.variance_so_far == pytest.approx(
        result.actual_so_far - result.expected_so_far
    )
    assert sum(result.variance_by_category.values()) == pytest.approx(
        result.variance_so_far
    )
    # Three failures where the plan expected about two: the repairs cost
    # more than planned (a negative variance, against you).
    assert result.variance_by_category["repair"] < 0.0
    assert result.not_compared == []
    assert result.as_planned_so_far == 0.0


def test_the_plan_is_valued_again_from_the_state_now():
    result = hand_result()
    # Down for 50 at 2000, in a repair.
    assert result.states == {
        "p": NodeState(alive=False, down_for=50.0, maintenance=False)
    }
    assert result.derived_states == ["p"]
    rest = pump().expected_npv(
        3000.0,
        discount_rate=R,
        value_rate=10.0,
        periods=[3000.0],
        state=result.states,
    )
    assert result.rest.npv == pytest.approx(rest.npv, rel=1e-12)
    assert result.rest_npv == pytest.approx(
        rest.npv * math.exp(-R * 2000.0), rel=1e-12
    )
    assert result.revalued_npv == pytest.approx(
        result.actual_so_far + result.rest_npv
    )
    assert result.change == pytest.approx(
        result.variance_so_far + result.rest_npv - result.plan_rest_npv
    )
    assert result.plan_npv == pytest.approx(
        result.expected_so_far + result.plan_rest_npv
    )
    # The rest's rows on the plan's clock: period 3, from 2000 to 5000.
    assert {
        (r["period"], r["start"], r["end"]) for r in result.rest_cash_flows
    } == {(3, 2000.0, 5000.0)}
    assert sum(r["present_value"] for r in result.rest_cash_flows) == (
        pytest.approx(result.rest_npv)
    )


def test_a_period_now_falls_in_is_split():
    log = {"p": Timeline.from_outages(HAND_LOG, end=2000.0)}
    result = pump().plan_vs_actual(
        log, 2000.0, 5000.0, discount_rate=R, periods=2
    )
    assert result.period_ends == [2000.0, 2500.0, 5000.0]
    default = pump().plan_vs_actual(log, 2000.0, 5000.0)
    assert default.period_ends == [2000.0, 5000.0]


# -- what the log does not determine ----------------------------------------


def test_what_the_log_does_not_determine_is_compared_on_neither_side():
    tested = {
        "reliability": E([1e-3]),
        "repairability": E([0.1]),
        "repair_cost": 50.0,
        "acquisition_cost": 1000.0,
        "inspection": {"interval": 100.0, "cost": 5.0},
    }
    rbd = RepairableRBD(LINE, {"p": tested}, downtime_cost_rate=10.0)
    log = {"p": Timeline.from_outages([(130.0, 210.0)], end=250.0)}
    with pytest.raises(NotImplementedError, match="hidden failures"):
        rbd.plan_vs_actual(log, 250.0, 1000.0)
    # Its state given: tested at 200, found failed and repaired by 210.
    state = {"p": NodeState(age=40.0, since_scheduled=50.0)}
    result = rbd.plan_vs_actual(log, 250.0, 1000.0, state=state)
    assert {(x["category"], x["component"]) for x in result.not_compared} == {
        ("repair", "p"),
        ("inspection", "p"),
        ("acquisition", "p"),
    }
    assert result.derived_states == []
    assert result.states == state
    left_out = {"repair", "inspection", "acquisition"}
    for rows in (result.expected_cash_flows, result.actual_cash_flows):
        assert not left_out & {r["category"] for r in rows}
    # The system was down 80: priced, as the plan prices it.
    assert result.actual_so_far == pytest.approx(-800.0)
    # Carried as planned: the purchase and the plan's repairs and tests.
    assert result.as_planned_so_far == pytest.approx(
        sum(
            r["present_value"]
            for r in result.plan.cash_flows
            if r["period"] == 1 and r["category"] in left_out
        )
    )
    assert result.as_planned_so_far < -1000.0
    assert result.revalued_npv == pytest.approx(
        result.actual_so_far + result.as_planned_so_far + result.rest_npv
    )
    # Bought once: the rest from now has no purchase in it.
    assert "acquisition" not in result.rest.cost_by_category
    assert result.change == pytest.approx(
        result.variance_so_far + result.rest_npv - result.plan_rest_npv
    )


def test_each_kind_of_cost_left_out_says_why():
    unit = {"reliability": W([500.0, 1.5]), "repairability": E([0.5])}
    rbd = RepairableRBD(
        [("s", "a"), ("a", "b"), ("b", "c"), ("c", "d"), ("d", "t")],
        {
            "a": {
                **unit,
                "standby": {"units": 2, "k": 1},
                "repair_cost": 1.0,
                "replace_cost": 1.0,
            },
            "b": {
                **unit,
                "preventive": {"interval": 100.0, "cost": 2.0},
                "group": "g",
            },
            "c": {
                **unit,
                "preventive": {
                    "interval": 100.0,
                    "policy": "condition",
                    "threshold": 0.1,
                    "cost": 2.0,
                    "inspection_cost": 0.5,
                    "duration": E([1.0]),
                },
            },
            "d": {
                **unit,
                "reliability": E([1e-3]),
                "repair": {"model": "kijima1", "q": 0.5},
                "replace_after": 3,
                "replace_cost": 3.0,
                "repair_cost": 1.0,
            },
        },
        maintenance_groups={"g": {"setup_cost": 4.0}},
    )
    reasons = {
        (category, node): why
        for category, node, why in _plan_actual._not_compared(rbd)
    }
    assert reasons == {
        ("repair", "a"): _plan_actual._UNITS,
        ("replace", "a"): _plan_actual._UNITS,
        ("preventive", "b"): _plan_actual._INSTANT,
        ("inspection", "c"): _plan_actual._INSPECTIONS,
        ("replace", "d"): _plan_actual._REPLACED,
        ("setup", None): _plan_actual._SETUP,
    }


# -- the states read off a log ----------------------------------------------


def unit_rbd(**extra):
    spec = {"reliability": W([100.0, 2.5]), "repairability": W([5.0, 1.5])}
    spec.update(extra)
    return RepairableRBD(LINE, {"p": spec})


def state_of(rbd, log, start=None):
    return _plan_actual._derived_state(
        rbd, "p", log, log.end, start or NodeState()
    )


def test_states_are_read_off_a_log_by_hand():
    plain = unit_rbd()
    # Up since a repair ended at 12: age 50 - 12.
    assert state_of(plain, Timeline.from_outages([(10, 12)], end=50)) == (
        NodeState(age=38.0)
    )
    # Down since 10, in a repair.
    assert state_of(plain, Timeline.from_outages([(10, None)], end=50)) == (
        NodeState(alive=False, down_for=40.0)
    )
    # Never down: as old as the log, plus its age at the start.
    assert state_of(plain, Timeline([], end=20.0)) == NodeState(age=20.0)
    assert state_of(
        plain, Timeline([], end=20.0), NodeState(age=5.0)
    ) == NodeState(age=25.0)
    # Down from the start, and still: as long as the log, and before.
    assert state_of(
        plain,
        Timeline([], end=20.0, up=False),
        NodeState(alive=False, down_for=1.0),
    ) == NodeState(alive=False, down_for=21.0)
    # An instant repair at 30: renewed then.
    assert state_of(plain, Timeline([30.0, 30.0], end=50)) == NodeState(
        age=20.0
    )
    aged = unit_rbd(preventive={"interval": 60.0, "duration": E([1.0])})
    # In its planned replacement since 60.
    log = Timeline.from_outages([(60.0, None)], end=61.5, planned=[True])
    assert state_of(aged, log) == NodeState(
        alive=False, down_for=1.5, maintenance=True
    )
    block = unit_rbd(
        preventive={"interval": 60.0, "policy": "block", "duration": E([1.0])}
    )
    # Replaced at 60 and 120 (back at 62 and 121); 130 is 10 past 120.
    log = Timeline.from_outages(
        [(60.0, 62.0), (120.0, 121.0)], end=130.0, planned=[True, True]
    )
    assert state_of(block, log) == NodeState(age=9.0, since_scheduled=10.0)
    # Started 50 into its interval of 60: 20 later it is 10 into the next.
    assert state_of(
        block, Timeline([], end=20.0), NodeState(age=50, since_scheduled=50.0)
    ) == NodeState(age=70.0, since_scheduled=10.0)
    # On a scheduled time exactly: 0 past it.
    assert state_of(block, Timeline([], end=60.0)).since_scheduled == 0.0


def test_the_states_used_are_given_or_derived():
    result = unit_rbd(repair_cost=1.0).plan_vs_actual(
        {"p": Timeline([], end=20.0)},
        20.0,
        50.0,
        state={"p": NodeState(age=3)},
    )
    assert result.states == {"p": NodeState(age=3.0)}
    assert result.derived_states == []


EXPONENTIAL = {"reliability": E([1e-2]), "repairability": E([0.5])}


@pytest.mark.parametrize(
    "spec, match",
    [
        ({**EXPONENTIAL, "standby": {"units": 2, "k": 1}}, "standby group"),
        (
            {**EXPONENTIAL, "repair": {"model": "kijima1", "q": 1.0}},
            "virtual age",
        ),
        ({"inspection": {"interval": 50.0}}, "hidden failures"),
        ({"preventive": {"interval": 50.0}}, "take no time"),
    ],
)
def test_a_state_the_log_does_not_give_is_asked_for(spec, match):
    rbd = unit_rbd(repair_cost=1.0, **spec)
    with pytest.raises(NotImplementedError, match=match) as error:
        rbd.plan_vs_actual({"p": Timeline([], end=20.0)}, 20.0, 50.0)
    assert "'p'" in str(error.value)
    assert "state=" in str(error.value)
    route = rbd.analysis_routes()["plan_vs_actual"]
    assert route.reason == str(error.value)


def test_a_nested_rbd_takes_its_components_states():
    inner = RepairableRBD(
        [("s", "x"), ("x", "t")],
        {"x": {"reliability": E([1e-2]), "repairability": E([1.0])}},
    )
    outer = RepairableRBD(
        [("s", "p"), ("p", "q"), ("q", "t")],
        {
            "p": inner,
            "q": {"reliability": E([1e-3]), "repairability": E([1.0])},
        },
        downtime_cost_rate=1.0,
    )
    logs = {
        "p": Timeline.from_outages([(5.0, 6.0)], end=20.0),
        "q": Timeline([], end=20.0),
    }
    with pytest.raises(NotImplementedError, match="nested RBD"):
        outer.plan_vs_actual(logs, 20.0, 50.0)
    result = outer.plan_vs_actual(
        logs, 20.0, 50.0, state={"p": {"x": NodeState(age=14.0)}}
    )
    assert result.derived_states == ["q"]
    # The system down for the nested system's hour.
    assert result.actual_so_far == pytest.approx(-1.0)


def test_crews_and_common_causes_ask_for_the_states():
    unit = {
        "reliability": E([1e-2]),
        "repairability": E([1.0]),
        "repair_cost": 1.0,
    }
    crews = RepairableRBD(PAIR, {n: dict(unit) for n in "abv"}, repair_crews=1)
    logs = {n: Timeline([], end=10.0) for n in "abv"}
    with pytest.raises(NotImplementedError, match="crews"):
        crews.plan_vs_actual(logs, 10.0, 20.0)


# -- consistency with the plan, by simulation -------------------------------


def priced(rbd, sim, ends, rate, value_rate):
    """Each simulated history priced as a log: {(period, category,
    component): present values}."""
    out: dict = {}
    n = len(sim.system)
    for i in range(n):
        logs = {node: sim.components[node][i] for node in rbd.components}
        for r in _plan_actual._priced_log(
            rbd, logs, ends, rate, value_rate, set()
        ):
            key = (r["period"], r["category"], r["component"])
            out.setdefault(key, np.zeros(n))[i] = r["present_value"]
    return out


def redundant():
    pumps = {
        "reliability": W([200.0, 1.8]),
        "repairability": W([20.0, 1.5]),
        "repair_cost": 40.0,
        "replace_cost": 15.0,
        "downtime_cost": 0.5,
    }
    return RepairableRBD(
        PAIR,
        {
            "a": dict(pumps),
            "b": dict(pumps),
            "v": {
                "reliability": E([2e-3]),
                "repairability": E([0.2]),
                "repair_cost": 60.0,
            },
        },
        downtime_cost_rate=8.0,
    )


def age_replaced(policy="age"):
    return RepairableRBD(
        LINE,
        {
            "p": {
                "reliability": W([100.0, 2.5]),
                "repairability": W([6.0, 1.5]),
                "repair_cost": 30.0,
                "downtime_cost": 1.0,
                "preventive": {
                    "interval": 60.0,
                    "policy": policy,
                    "duration": W([2.0, 2.0]),
                    "cost": 10.0,
                },
            }
        },
        downtime_cost_rate=4.0,
    )


@pytest.mark.parametrize(
    "make", [redundant, age_replaced, lambda: age_replaced("block")]
)
def test_the_priced_logs_average_to_the_plan(make):
    rbd = make()
    rate, value, ends = 2e-3, 3.0, np.array([150.0, 300.0])
    sim = rbd.simulate_timelines(300.0, mc_samples=4000, seed=7)
    cells = priced(rbd, sim, ends, rate, value)
    plan = rbd.expected_npv(
        300.0, discount_rate=rate, value_rate=value, periods=ends
    )
    expected = {
        (r["period"], r["category"], r["component"]): r["present_value"]
        for r in plan.cash_flows
    }
    assert set(cells) == set(expected)
    for key, values in cells.items():
        se = values.std(ddof=1) / math.sqrt(values.size)
        assert abs(values.mean() - expected[key]) < 4.0 * se + 1e-9, key
    # The whole so far, in one number.
    total = sum(cells.values())
    se = total.std(ddof=1) / math.sqrt(total.size)
    assert abs(total.mean() - plan.npv) < 4.0 * se


@pytest.mark.parametrize(
    "make, n",
    [
        (redundant, 120),
        (age_replaced, 80),
        (lambda: age_replaced("block"), 80),
    ],
)
def test_so_far_and_the_rest_from_each_state_average_to_the_plan(make, n):
    # The tower property: the plan's value is the mean, over the ways the
    # first stretch could go, of what it yielded plus the rest valued from
    # where it left the components. It holds only if each state read off a
    # log is the one the plan's own model would be in.
    rbd = make()
    rate, value, now, horizon = 2e-3, 3.0, 150.0, 300.0
    ends = np.array([now, horizon])
    plan = rbd.expected_npv(
        horizon, discount_rate=rate, value_rate=value, periods=ends
    )
    plan_rest = sum(
        r["present_value"] for r in plan.cash_flows if r["period"] == 2
    )
    sim = rbd.simulate_timelines(now, mc_samples=n, seed=11)
    so_far = np.zeros(n)
    rest = np.zeros(n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(n):
            logs = {node: sim.components[node][i] for node in rbd.components}
            so_far[i] = sum(
                r["present_value"]
                for r in _plan_actual._priced_log(
                    rbd, logs, ends[:1], rate, value, set()
                )
            )
            states, _ = _plan_actual._states_now(rbd, logs, now, {}, None)
            rest[i] = (
                math.exp(-rate * now)
                * _npv.expected_npv(
                    rbd,
                    horizon - now,
                    discount_rate=rate,
                    value_rate=value,
                    periods=None,
                    working_nodes=None,
                    broken_nodes=None,
                    method="p",
                    state=states,
                ).npv
            )
    se = rest.std(ddof=1) / math.sqrt(n)
    assert abs(rest.mean() - plan_rest) < 4.0 * se
    whole = so_far + rest
    se = whole.std(ddof=1) / math.sqrt(n)
    assert abs(whole.mean() - plan.npv) < 4.0 * se


# -- the edges --------------------------------------------------------------


def test_at_the_horizon_there_is_no_rest():
    result = hand_result()
    end = pump().plan_vs_actual(
        {"p": Timeline.from_outages(HAND_LOG, end=2000.0)},
        2000.0,
        2000.0,
        discount_rate=R,
        value_rate=10.0,
    )
    assert end.rest is None
    assert end.rest_npv == 0.0 and end.rest_cash_flows == []
    assert end.revalued_npv == pytest.approx(result.actual_so_far)
    assert end.plan_rest_npv == 0.0


def test_with_nothing_yet_happened_the_rest_is_the_plan():
    result = pump().plan_vs_actual(
        {"p": Timeline([], end=1e-6)}, 1e-6, 5000.0, value_rate=10.0
    )
    assert result.actual_so_far == pytest.approx(1e-5, rel=1e-9)
    assert result.rest_npv == pytest.approx(result.plan_npv, rel=1e-6)
    assert result.change == pytest.approx(0.0, abs=1e-6 * abs(result.plan_npv))


@pytest.mark.parametrize("scale", [1e-6, 1e-3, 1e3, 1e6])
def test_amounts_do_not_depend_on_the_unit_of_time(scale):
    base = hand_result()
    log = [(a * scale, None if b is None else b * scale) for a, b in HAND_LOG]
    result = pump(scale).plan_vs_actual(
        {"p": Timeline.from_outages(log, end=2000.0 * scale)},
        2000.0 * scale,
        5000.0 * scale,
        discount_rate=R / scale,
        value_rate=10.0 / scale,
        periods=[1000.0 * scale, 2000.0 * scale, 5000.0 * scale],
    )
    for name in ("actual_so_far", "expected_so_far", "plan_npv", "rest_npv"):
        assert getattr(result, name) == pytest.approx(
            getattr(base, name), rel=1e-6
        ), name
    assert result.states["p"].down_for == pytest.approx(50.0 * scale)


def test_it_is_plain_data():
    result = hand_result()
    assert isinstance(result, PlanVsActual)
    out = json.loads(json.dumps(result.to_dict()))
    assert out["states"]["p"]["down_for"] == 50.0
    assert out["plan"]["npv"] == pytest.approx(result.plan_npv)
    assert len(out["variance_cash_flows"]) == len(result.variance_cash_flows)
    assert repr(result).startswith("PlanVsActual(now=2000")


def test_inputs_are_checked():
    rbd = pump()
    log = {"p": Timeline([], end=100.0)}
    with pytest.raises(ValueError, match="No log for component"):
        rbd.plan_vs_actual({}, 100.0, 200.0)
    with pytest.raises(ValueError, match="'q'"):
        rbd.plan_vs_actual({**log, "q": Timeline([], end=100.0)}, 100.0, 200.0)
    with pytest.raises(TypeError, match="Timeline"):
        rbd.plan_vs_actual({"p": "log"}, 100.0, 200.0)
    with pytest.raises(TypeError, match="mapping"):
        rbd.plan_vs_actual([log["p"]], 100.0, 200.0)
    with pytest.raises(ValueError, match="not at now"):
        rbd.plan_vs_actual(log, 90.0, 200.0)
    with pytest.raises(ValueError, match="past the horizon"):
        rbd.plan_vs_actual(log, 100.0, 50.0)
    with pytest.raises(ValueError, match="now must be"):
        rbd.plan_vs_actual(log, -1.0, 50.0)
    with pytest.raises(ValueError, match="no maintenance or test"):
        rbd.plan_vs_actual(
            {"p": Timeline.from_outages([(10, 20)], 100.0, planned=[True])},
            100.0,
            200.0,
        )
    with pytest.raises(ValueError, match="starts down"):
        rbd.plan_vs_actual(
            {"p": Timeline([], end=100.0, up=False)}, 100.0, 200.0
        )
    with pytest.raises(TypeError, match="state must be a dict"):
        rbd.plan_vs_actual(log, 100.0, 200.0, state="stationary")
    with pytest.raises(TypeError, match="start_state must be a dict"):
        rbd.plan_vs_actual(log, 100.0, 200.0, start_state="stationary")
    with pytest.raises(ValueError, match="periods"):
        rbd.plan_vs_actual(log, 100.0, 200.0, periods=[50.0, 150.0])


@pytest.fixture(scope="module")
def kinds():
    from repyability.tests.catalogue import repairable_kinds

    return repairable_kinds()


@pytest.mark.parametrize(
    "kind, words",
    [
        ("replaced on condition by level", "degradation level"),
        ("replaced on a calendar policy", "calendar policy"),
        ("degrading capacity", "stages"),
        ("fixed probability", "a probability"),
        ("common cause, revealed", "common-cause group"),
    ],
)
def test_each_kind_whose_state_the_log_does_not_give_is_named(
    kinds, kind, words
):
    # Where expected_npv refuses such a diagram already, the method says
    # that first (the routes test checks it); the reason is still here.
    rbd = kinds[kind]
    reason = _plan_actual._log_state_refusal(rbd, "a")
    assert words in reason and "'a'" in reason and "state=" in reason
    if kind == "replaced on a calendar policy":
        assert ("preventive", "a", _plan_actual._CALENDAR) in (
            _plan_actual._not_compared(rbd)
        )


def test_planned_outages_are_taken_where_something_planned_takes_time():
    rbd = RepairableRBD(
        LINE,
        {
            "p": {
                "reliability": E([1e-2]),
                "repairability": E([0.5]),
                "inspection": {"interval": 50.0, "duration": E([1.0])},
            }
        },
        downtime_cost_rate=1.0,
    )
    log = Timeline.from_outages([(50.0, 51.0)], end=60.0, planned=[True])
    state = {"p": NodeState(age=9.0, since_scheduled=10.0)}
    result = rbd.plan_vs_actual({"p": log}, 60.0, 100.0, state=state)
    assert result.actual_so_far == pytest.approx(-1.0)


def test_a_plan_that_started_from_a_state():
    rbd = unit_rbd(repair_cost=1.0)
    start = {"p": NodeState(age=40.0)}
    result = rbd.plan_vs_actual(
        {"p": Timeline([], end=20.0)}, 20.0, 50.0, start_state=start
    )
    assert result.states == {"p": NodeState(age=60.0)}
    assert result.plan.npv == pytest.approx(
        rbd.expected_npv(50.0, periods=[20.0, 50.0], state=start).npv
    )
