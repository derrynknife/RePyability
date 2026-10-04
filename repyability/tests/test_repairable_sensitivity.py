"""A repairable diagram's parameter sensitivity (#192): the derivative of
its availability (long-run, at times, over a window) and its cost rate in
each lever, and the change one more standby unit or repair crew makes.

Checked against closed forms for exponential units, against differences of
the system rebuilt by hand, and against the system's Birnbaum importance
times its components' own derivatives, which the structure function's
linearity in each component makes exact."""

import numpy as np
import pytest
import surpyval as surv

from repyability import BetaFactor, CCFGroup, RepairableRBD

E, W = surv.Exponential.from_params, surv.Weibull.from_params


def exponential(life, repair, **more):
    return {"reliability": E([life]), "repairability": E([repair]), **more}


def one(life=0.01, repair=0.1, **more):
    return RepairableRBD(
        [("s", "c"), ("c", "t")], {"c": exponential(life, repair)}, **more
    )


def a_b_then_c(**specs):
    """a and b in parallel, then c."""
    edges = [("s", "a"), ("s", "b"), ("a", "c"), ("b", "c"), ("c", "t")]
    return RepairableRBD(edges, specs)


def test_a_single_unit_in_closed_form():
    lam, mu = 0.01, 0.1
    k = lam + mu
    long_run = one(lam, mu).parameter_sensitivity()["c"]
    assert long_run["reliability.failure_rate"] == pytest.approx(
        -mu / k**2, rel=1e-7
    )
    assert long_run["repairability.failure_rate"] == pytest.approx(
        lam / k**2, rel=1e-7
    )
    # A(t) = mu / k + lam / k exp(-k t).
    t = np.array([1.0, 10.0, 60.0])
    decay = np.exp(-k * t)
    d_lam = -mu / k**2 + mu / k**2 * decay - lam / k * t * decay
    d_mu = lam / k**2 - lam / k**2 * decay - lam / k * t * decay
    at = one(lam, mu).parameter_sensitivity(x=t)["c"]
    # (The curves over time are numerical, to about 1e-7: their
    # derivatives to about 1e-4 of their size, 1e-3 early on.)
    np.testing.assert_allclose(at["reliability.failure_rate"], d_lam, 1e-3)
    np.testing.assert_allclose(at["repairability.failure_rate"], d_mu, 1e-3)
    # The cost rate: a repair at each failure, and the system's down time.
    cost = RepairableRBD(
        [("s", "c"), ("c", "t")],
        {"c": exponential(lam, mu, repair_cost=200.0)},
        downtime_cost_rate=50.0,
    )
    rate = cost.parameter_sensitivity(of="cost_rate")["c"]
    # 200 lam mu / k + 50 lam / k, in lam: 200 mu^2 / k^2 + 50 mu / k^2.
    assert rate["reliability.failure_rate"] == pytest.approx(
        200.0 * mu**2 / k**2 + 50.0 * mu / k**2, rel=1e-6
    )


def plant(**more):
    return a_b_then_c(
        a=exponential(0.01, 0.1),
        b={
            "reliability": W([200.0, 2.0]),
            "repairability": E([0.2]),
            "preventive": {
                "interval": 100.0,
                "policy": "block",
                "duration": E([0.5]),
            },
            **more,
        },
        c={"reliability": W([600.0, 1.5]), "repairability": E([0.5])},
    )


def by_hand(rbd, node, key, value, evaluate):
    """``evaluate`` of ``rbd`` rebuilt with ``node``'s spec's ``key`` (a
    path of keys) at ``value``."""
    args = rbd._init_args
    spec = dict(args["components"][node])
    *path, last = key
    inner = spec
    for part in path:
        inner[part] = dict(inner[part])
        inner = inner[part]
    inner[last] = value
    return evaluate(
        RepairableRBD(
            **{**args, "components": {**args["components"], node: spec}}
        )
    )


def test_the_long_run_is_the_birnbaum_importance_times_the_node():
    rbd = a_b_then_c(
        a=exponential(0.01, 0.1),
        b=exponential(0.02, 0.25),
        c=exponential(0.002, 0.5),
    )
    sensitivity = rbd.parameter_sensitivity()
    birnbaum = rbd.birnbaum_importance()
    for node, (lam, mu) in {
        "a": (0.01, 0.1),
        "b": (0.02, 0.25),
        "c": (0.002, 0.5),
    }.items():
        k = lam + mu
        assert sensitivity[node]["reliability.failure_rate"] == pytest.approx(
            birnbaum[node] * -mu / k**2, rel=1e-7
        )
        assert sensitivity[node][
            "repairability.failure_rate"
        ] == pytest.approx(birnbaum[node] * lam / k**2, rel=1e-7)


@pytest.mark.parametrize(
    "lever, key",
    [
        ("reliability.alpha", ("reliability",)),
        ("preventive.duration.failure_rate", ("preventive", "duration")),
    ],
)
def test_over_time_the_component_alone_is_the_system(lever, key):
    rbd = plant()
    model = {
        "reliability.alpha": lambda v: W([v, 2.0]),
        "preventive.duration.failure_rate": lambda v: E([v]),
    }[lever]
    value = {
        "reliability.alpha": 200.0,
        "preventive.duration.failure_rate": 0.5,
    }[lever]
    h = 1e-3 * value
    t = np.array([30.0, 150.0, 420.0])
    at = rbd.parameter_sensitivity(x=t, rel_step=1e-3)["b"][lever]
    up = by_hand(
        rbd, "b", key, model(value + h), lambda r: r.point_availability(t)
    )
    down = by_hand(
        rbd, "b", key, model(value - h), lambda r: r.point_availability(t)
    )
    np.testing.assert_allclose(
        at, (up - down) / (2 * h), rtol=1e-6, atol=1e-12
    )
    window = rbd.parameter_sensitivity(window=500.0, rel_step=1e-3)["b"][lever]
    up = by_hand(
        rbd,
        "b",
        key,
        model(value + h),
        lambda r: r.mission_availability(500.0),
    )
    down = by_hand(
        rbd,
        "b",
        key,
        model(value - h),
        lambda r: r.mission_availability(500.0),
    )
    assert window == pytest.approx((up - down) / (2 * h), rel=1e-5)


def test_a_lever_that_moves_the_curve_s_breaks_over_a_window():
    # The interval moves b's replacements: differenced on the system.
    rbd = plant()
    window = rbd.parameter_sensitivity(window=500.0, rel_step=1e-3)["b"][
        "preventive.interval"
    ]
    h = 1e-3 * 100.0
    up = by_hand(
        rbd,
        "b",
        ("preventive", "interval"),
        100.0 + h,
        lambda r: r.mission_availability(500.0),
    )
    down = by_hand(
        rbd,
        "b",
        ("preventive", "interval"),
        100.0 - h,
        lambda r: r.mission_availability(500.0),
    )
    assert window == pytest.approx((up - down) / (2 * h), rel=1e-9)


def test_calendar_levers_in_the_long_run():
    # Alone on its calendar, b's interval is differenced on the system; with
    # c's tests sharing it, its schedule is taken apart from theirs.
    alone = plant()
    direct = alone.parameter_sensitivity()["b"]["preventive.interval"]
    birnbaum = alone.birnbaum_importance()["b"]
    h = 1e-5 * 100.0
    unit = RepairableRBD(
        [("s", "b"), ("b", "t")], {"b": alone._init_args["components"]["b"]}
    )

    def own(value):
        return by_hand(
            unit,
            "b",
            ("preventive", "interval"),
            value,
            lambda r: r.mean_availability(),
        )

    apart = birnbaum * (own(100.0 + h) - own(100.0 - h)) / (2 * h)
    # Alone on its calendar the two agree: the others do not vary with it.
    assert direct == pytest.approx(apart, rel=1e-6)
    shared = a_b_then_c(
        a=exponential(0.01, 0.1),
        b=alone._init_args["components"]["b"],
        c={
            "reliability": E([0.002]),
            "repairability": E([0.5]),
            "inspection": {"interval": 50.0},
        },
    )
    values = shared.parameter_sensitivity()
    birnbaum = shared.birnbaum_importance()["b"]
    assert values["b"]["preventive.interval"] == pytest.approx(
        birnbaum * (own(100.0 + h) - own(100.0 - h)) / (2 * h), rel=1e-6
    )
    assert np.isfinite(values["c"]["inspection.interval"])


def test_schedules_that_repeat_together_too_rarely_are_refused():
    # A block interval moved off the tests' calendar (99.999 against 50)
    # repeats with it only after 100,000 tests: refused, before a grid too
    # large for memory is built.
    rbd = a_b_then_c(
        a=exponential(0.01, 0.1),
        b={
            "reliability": W([200.0, 2.0]),
            "repairability": E([0.2]),
            "preventive": {"interval": 99.999, "policy": "block"},
        },
        c={
            "reliability": E([0.002]),
            "repairability": E([0.5]),
            "inspection": {"interval": 50.0},
        },
    )
    with pytest.raises(NotImplementedError, match="too long a time"):
        rbd.mean_availability()


def test_discrete_levers():
    # One more crew: the pair repaired at once, instead of in turn.
    crewed = a_b_then_c(
        a=exponential(0.01, 0.1),
        b=exponential(0.01, 0.1),
        c=exponential(0.002, 0.5),
    )
    crewed = RepairableRBD(**{**crewed._init_args, "repair_crews": 1})
    more = RepairableRBD(**{**crewed._init_args, "repair_crews": 2})
    change = crewed.parameter_sensitivity()[None]["repair_crews"]
    assert change == pytest.approx(
        more.mean_availability() - crewed.mean_availability(), rel=1e-9
    )
    # One more standby unit.
    pair = {**exponential(0.01, 0.1), "standby": {"units": 2}}
    standby = a_b_then_c(
        a=exponential(0.01, 0.1), b=pair, c=exponential(0.002, 0.5)
    )
    larger = by_hand(
        standby, "b", ("standby", "units"), 3, lambda r: r.mean_availability()
    )
    assert standby.parameter_sensitivity()["b"][
        "standby.units"
    ] == pytest.approx(larger - standby.mean_availability(), rel=1e-9)
    # Over time, the component alone: the same as the system rebuilt.
    t = np.array([20.0, 200.0])
    over = standby.parameter_sensitivity(x=t)["b"]["standby.units"]
    larger = by_hand(
        standby,
        "b",
        ("standby", "units"),
        3,
        lambda r: r.point_availability(t),
    )
    np.testing.assert_allclose(
        over, larger - standby.point_availability(t), rtol=1e-9, atol=1e-14
    )


def test_a_common_cause_group_s_levers():
    rbd = RepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "c"), ("b", "c"), ("c", "t")],
        {
            "a": exponential(0.01, 0.1),
            "b": exponential(0.01, 0.1),
            "c": exponential(0.002, 0.5),
        },
        ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.1, basis="rate"))],
    )
    values = rbd.parameter_sensitivity()
    assert set(values) == {("a", "b"), "c"}
    group = values[("a", "b")]
    h = 1e-5 * 0.1

    def with_beta(beta):
        changed = RepairableRBD(
            **{
                **rbd._init_args,
                "ccf_groups": [
                    CCFGroup(["a", "b"], BetaFactor(beta, basis="rate"))
                ],
            }
        )
        return changed.mean_availability()

    assert group["ccf_beta"] == pytest.approx(
        (with_beta(0.1 + h) - with_beta(0.1 - h)) / (2 * h), rel=1e-6
    )
    # The members' life rate, moved for both at once.
    assert group["reliability.failure_rate"] < 0.0
    # Over time, the system rebuilt.
    over = rbd.parameter_sensitivity(x=[50.0])[("a", "b")]["ccf_beta"]
    assert np.isfinite(over)


def test_held_nodes_and_unit_costs():
    rbd = plant()
    held = rbd.parameter_sensitivity(working_nodes=["c"])
    assert all(v == 0.0 for v in held["c"].values())
    costs = {
        ("b", "preventive.interval"): 50.0,
        ("a", "repairability.failure_rate"): 2e3,
    }
    ranked = rbd.parameter_sensitivity(unit_costs=costs)
    plain = rbd.parameter_sensitivity()
    assert {(k, n) for k, v in ranked.items() for n in v} == set(costs)
    for (key, name), cost in costs.items():
        assert ranked[key][name] == pytest.approx(plain[key][name] / cost)


def test_both_quantities_at_once():
    rbd = plant()
    rbd = RepairableRBD(**{**rbd._init_args, "downtime_cost_rate": 10.0})
    both = rbd.parameter_sensitivity(of=("availability", "cost_rate"))
    assert set(both) == {"availability", "cost_rate"}
    assert both["availability"] == rbd.parameter_sensitivity()
    assert both["cost_rate"] == rbd.parameter_sensitivity(of="cost_rate")
    # Over a window: the expected cost per unit time.
    window = rbd.parameter_sensitivity(
        window=300.0, of="cost_rate", rel_step=1e-3
    )
    h = 1e-3 * 200.0

    def mean_rate(alpha):
        return by_hand(
            rbd,
            "b",
            ("reliability",),
            W([alpha, 2.0]),
            lambda r: r.expected_cost(300.0).mean / 300.0,
        )

    assert window["b"]["reliability.alpha"] == pytest.approx(
        (mean_rate(200.0 + h) - mean_rate(200.0 - h)) / (2 * h), rel=1e-6
    )


def test_what_it_refuses():
    rbd = plant()
    with pytest.raises(ValueError, match="not both"):
        rbd.parameter_sensitivity(x=1.0, window=10.0)
    with pytest.raises(ValueError, match="window, not x"):
        rbd.parameter_sensitivity(x=1.0, of="cost_rate")
    with pytest.raises(ValueError, match="state is where"):
        rbd.parameter_sensitivity(state="stationary")
    with pytest.raises(ValueError, match="rel_step"):
        rbd.parameter_sensitivity(rel_step=0.0)
    with pytest.raises(ValueError, match="of must be"):
        rbd.parameter_sensitivity(of="reliability")
    with pytest.raises(ValueError, match="not levers"):
        rbd.parameter_sensitivity(unit_costs={("b", "nothing"): 1.0})
    imperfect = RepairableRBD(
        [("s", "a"), ("a", "t")],
        {
            "a": {
                "reliability": W([100.0, 2.0]),
                "repairability": E([0.1]),
                "repair": {"model": "kijima1", "q": 0.5},
            }
        },
    )
    with pytest.raises(NotImplementedError):
        imperfect.parameter_sensitivity()
