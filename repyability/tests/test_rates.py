"""The rate at which a system's availability (or reliability) changes, split
among its components, and the Barlow-Proschan importance (#195).

With independent components the system is multilinear in them, so its rate
is the sum of each one's Birnbaum importance times its own rate: checked
against the system's own differences, closed forms, and jumps worked out
from the curves either side. The Barlow-Proschan importance is checked
against closed forms, against simulated lifetimes (which component's failure
ended each), and against the simulated failure criticality index."""

import numpy as np
import pytest
import surpyval as surv

from repyability import (
    BetaFactor,
    CCFGroup,
    NodeState,
    NonRepairableRBD,
    RepairableRBD,
)

E, W = surv.Exponential.from_params, surv.Weibull.from_params

BRIDGE = [
    ("s", "a"),
    ("s", "b"),
    ("a", "c"),
    ("b", "c"),
    ("a", "d"),
    ("c", "d"),
    ("b", "e"),
    ("c", "e"),
    ("d", "t"),
    ("e", "t"),
]
PAIR_THEN_C = [("s", "a"), ("s", "b"), ("a", "c"), ("b", "c"), ("c", "t")]


def lives():
    return {n: W([60 + 15 * i, 1.2 + 0.4 * i]) for i, n in enumerate("abcde")}


# -- a non-repairable diagram -------------------------------------------------


def test_the_parts_add_up_to_the_system_density():
    rbd = NonRepairableRBD(BRIDGE, lives())
    x = np.array([5.0, 40.0, 120.0])
    rate = rbd.reliability_rate(x)
    np.testing.assert_allclose(rate.rate, -rbd.df(x), rtol=1e-6)
    np.testing.assert_allclose(sum(rate.node_rate.values()), rate.rate)
    assert rate.jump_times.size == 0


def test_a_node_s_part_is_the_system_moved_by_its_life_alone():
    # The system's change with one node's life moved on and the others
    # held: its part.
    rbd = NonRepairableRBD(BRIDGE, lives())
    t, h = 40.0, 1e-4
    rate = rbd.reliability_rate(t)
    at = {n: float(m.sf(t)) for n, m in rbd.reliabilities.items()}
    for node in "abcde":
        model = rbd.reliabilities[node]
        ends = []
        for moved in (t + h, t - h):
            given = {**at, node: float(model.sf(moved))}
            ends.append(float(np.ravel(rbd.system_probability(given))[0]))
        assert rate.node_rate[node] == pytest.approx(
            (ends[0] - ends[1]) / (2 * h), rel=1e-6
        )


def test_a_repeated_node_is_one_component():
    shared = NonRepairableRBD(
        [
            ("in", "a"),
            ("a", "psu"),
            ("psu", "out"),
            ("in", "b"),
            ("b", "psu2"),
            ("psu2", "out"),
        ],
        {"a": W([10, 2]), "b": W([12, 2]), "psu": W([20, 1.5]), "psu2": "psu"},
    )
    rate = shared.reliability_rate([2.0, 5.0])
    assert set(rate.node_rate) == {"a", "b", "psu"}
    np.testing.assert_allclose(rate.rate, -shared.df([2.0, 5.0]), rtol=1e-6)


def test_a_common_cause_group_takes_its_part_together():
    rbd = NonRepairableRBD(
        PAIR_THEN_C,
        {"a": W([100, 2]), "b": W([100, 2]), "c": W([200, 1.5])},
        ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.1, basis="rate"))],
    )
    rate = rbd.reliability_rate([20.0, 50.0])
    assert set(rate.node_rate) == {"c", ("a", "b")}
    np.testing.assert_allclose(rate.rate, -rbd.df([20.0, 50.0]), rtol=1e-6)
    shares = rbd.barlow_proschan_importance(50.0)
    assert sum(shares.values()) == pytest.approx(1.0)


def test_held_nodes_take_no_part():
    rbd = NonRepairableRBD(BRIDGE, lives())
    rate = rbd.reliability_rate(40.0, working_nodes=["c"])
    assert rate.node_rate["c"] == 0.0
    assert rate.rate == pytest.approx(
        -rbd.df(40.0, working_nodes=["c"]), rel=1e-6
    )
    shares = rbd.barlow_proschan_importance(working_nodes=["c"])
    assert shares["c"] == 0.0
    assert sum(shares.values()) == pytest.approx(1.0)


def test_series_exponentials_share_by_their_rates():
    rates = {"a": 1.0, "b": 3.0, "c": 0.5}
    rbd = NonRepairableRBD(
        [("s", "a"), ("a", "b"), ("b", "c"), ("c", "t")],
        {n: E([r]) for n, r in rates.items()},
    )
    total = sum(rates.values())
    for x in (None, 0.3, [0.1, 2.0]):
        shares = rbd.barlow_proschan_importance(x)
        for node, rate in rates.items():
            np.testing.assert_allclose(shares[node], rate / total, rtol=1e-8)


def test_the_last_of_a_pair_to_fail_causes_its_failure():
    la, lb = 0.3, 0.1
    rbd = NonRepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        {"a": E([la]), "b": E([lb])},
    )
    # Over the whole life, a fails last with probability lb / (la + lb).
    shares = rbd.barlow_proschan_importance()
    assert shares["a"] == pytest.approx(lb / (la + lb), rel=1e-8)
    # By x: P(b < a <= x) over P(both by x).
    x = 4.0
    a_last = (1 - np.exp(-la * x)) - la / (la + lb) * (
        1 - np.exp(-(la + lb) * x)
    )
    both = (1 - np.exp(-la * x)) * (1 - np.exp(-lb * x))
    assert rbd.barlow_proschan_importance(x)["a"] == pytest.approx(
        a_last / both, rel=1e-7
    )


def test_barlow_proschan_against_simulated_lifetimes():
    rbd = NonRepairableRBD(
        PAIR_THEN_C, {"a": W([100, 2]), "b": W([100, 2]), "c": W([200, 1.5])}
    )
    rng = np.random.default_rng(7)
    n = 200_000
    a = 100 * rng.weibull(2.0, n)
    b = 100 * rng.weibull(2.0, n)
    c = 200 * rng.weibull(1.5, n)
    pumps = np.maximum(a, b)
    system = np.minimum(pumps, c)
    caused = {"c": c < pumps, "a": (c >= pumps) & (a > b)}
    caused["b"] = ~caused["c"] & ~caused["a"]
    for x in (None, 50.0):
        within = np.ones(n, bool) if x is None else system <= x
        shares = rbd.barlow_proschan_importance(x)
        for node, flags in caused.items():
            p = flags[within].mean()
            sd = np.sqrt(p * (1 - p) / within.sum())
            assert abs(shares[node] - p) < 4 * sd, (x, node)


def test_shapes_and_a_system_that_has_not_failed():
    rbd = NonRepairableRBD(BRIDGE, lives())
    shares = rbd.barlow_proschan_importance([[0.0, 10.0], [50.0, 100.0]])
    assert shares["a"].shape == (2, 2)
    assert np.isnan(shares["a"][0, 0])
    np.testing.assert_allclose(
        sum(v[~np.isnan(v)] for v in shares.values()), 1.0
    )
    with pytest.raises(ValueError, match="non-negative"):
        rbd.barlow_proschan_importance(-1.0)


# -- a repairable diagram ----------------------------------------------------


def unit(life, repair=None, **more):
    return {
        "reliability": life,
        "repairability": E([1.0]) if repair is None else repair,
        **more,
    }


def exponential_series(rates):
    nodes = list(rates)
    edges = [("s", nodes[0])]
    edges += list(zip(nodes, nodes[1:]))
    edges += [(nodes[-1], "t")]
    return RepairableRBD(edges, {n: unit(E([r])) for n, r in rates.items()})


def test_a_series_of_exponential_units_in_closed_form():
    rates = {"a": 0.1, "b": 0.3}
    rbd = exponential_series(rates)
    x = np.array([0.0, 0.5, 2.0, 7.0])

    def available(lam):
        return 1 / (1 + lam) + lam / (1 + lam) * np.exp(-(1 + lam) * x)

    def changing(lam):
        return -lam * np.exp(-(1 + lam) * x)

    rate = rbd.availability_rate(x)
    a, b = available(0.1), available(0.3)
    expected = {"a": b * changing(0.1), "b": a * changing(0.3)}
    for node, value in expected.items():
        np.testing.assert_allclose(rate.node_rate[node], value, atol=1e-6)
    np.testing.assert_allclose(rate.rate, sum(expected.values()), atol=1e-6)
    assert rate.jump_times.size == 0


def own_rate(rbd, x, **given):
    """The system's own rate: central differences of its point availability
    a step of its coarsest curve's grid either side (each curve is linear
    between its points)."""
    h = 0.012
    return (
        rbd.point_availability(np.asarray(x) + h, **given)
        - rbd.point_availability(np.asarray(x) - h, **given)
    ) / (2 * h)


def mixed():
    return RepairableRBD(
        PAIR_THEN_C,
        {
            "a": unit(W([10, 2.5])),
            "b": unit(E([0.1])),
            "c": unit(W([30, 1.5]), E([0.5])),
        },
    )


def test_the_parts_add_up_to_the_system_s_rate():
    rbd = mixed()
    x = np.array([0.7, 2.0, 4.5, 9.9, 15.0])
    rate = rbd.availability_rate(x)
    np.testing.assert_allclose(rate.rate, own_rate(rbd, x), rtol=3e-5)
    np.testing.assert_allclose(sum(rate.node_rate.values()), rate.rate)


def test_jumps_at_block_replacements_that_take_time():
    # Both a and c come off line at each block time, together.
    rbd = RepairableRBD(
        PAIR_THEN_C,
        {
            "a": unit(
                W([10, 2.5]),
                preventive={
                    "policy": "block",
                    "interval": 5.0,
                    "duration": E([4.0]),
                },
            ),
            "b": unit(E([0.1])),
            "c": unit(
                W([30, 1.5]),
                E([0.5]),
                preventive={
                    "policy": "block",
                    "interval": 5.0,
                    "duration": E([2.0]),
                },
            ),
        },
    )
    rate = rbd.availability_rate([3.0, 12.0])
    np.testing.assert_allclose(rate.jump_times, [5.0, 10.0])
    before = rbd.point_availability(np.nextafter(rate.jump_times, -np.inf))
    after = rbd.point_availability(rate.jump_times)
    np.testing.assert_allclose(rate.jumps, after - before, atol=1e-9)
    np.testing.assert_allclose(sum(rate.node_jumps.values()), rate.jumps)
    assert np.all(rate.node_jumps["b"] == 0.0)
    # Both a and c take part in each jump.
    assert np.all(rate.node_jumps["a"] < 0) and np.all(
        rate.node_jumps["c"] < 0
    )


def test_the_rates_and_jumps_make_up_the_change():
    # Integrated, the rate and the jumps give the change in availability.
    rbd = RepairableRBD(
        PAIR_THEN_C,
        {
            "a": unit(
                W([10, 2.5]),
                preventive={
                    "policy": "age",
                    "interval": 5.0,
                    "duration": E([4.0]),
                },
            ),
            "b": unit(E([0.1])),
            "c": unit(W([30, 1.5])),
        },
    )
    # (Up to 9: from 10, a unit's second replacement on age comes in a
    # burst, which the curve follows more coarsely than its rate needs.)
    end = 9.0
    jumps = rbd.availability_rate(end)
    assert jumps.jump_times.size >= 1
    # The rate between the jumps (just short of each, before it), by the
    # trapezoidal rule, and the jumps themselves.
    edges = np.concatenate([[0.0], jumps.jump_times, [end]])
    integral = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        x = np.linspace(lo, hi, 4001)
        if hi < end:
            x[-1] = np.nextafter(hi, -np.inf)
        rate = rbd.availability_rate(x).rate
        integral += np.sum(0.5 * (rate[1:] + rate[:-1]) * np.diff(x))
    change = rbd.point_availability(end) - rbd.point_availability(0.0)
    assert integral + jumps.jumps.sum() == pytest.approx(change, abs=2e-6)


def test_from_a_state_a_component_in_repair_pulls_the_system_up():
    rbd = mixed()
    down = {"c": NodeState(alive=False)}
    rate = rbd.availability_rate(0.5, state=down)
    assert rate.node_rate["c"] > 0.0
    assert rate.rate == pytest.approx(
        float(own_rate(rbd, 0.5, state=down)), rel=3e-5
    )


def test_held_components_take_no_part():
    rbd = mixed()
    rate = rbd.availability_rate([1.0, 5.0], broken_nodes=["b"])
    np.testing.assert_array_equal(rate.node_rate["b"], 0.0)
    np.testing.assert_allclose(
        rate.rate, own_rate(rbd, [1.0, 5.0], broken_nodes=["b"]), rtol=3e-5
    )


def test_dependent_components_are_refused_for_now():
    crews = RepairableRBD(
        PAIR_THEN_C, {n: unit(E([0.1])) for n in "abc"}, repair_crews=1
    )
    with pytest.raises(NotImplementedError, match="repair crews"):
        crews.availability_rate(1.0)
    with pytest.raises(NotImplementedError, match="repair crews"):
        crews.barlow_proschan_importance(window=10.0)
    grouped = RepairableRBD(
        PAIR_THEN_C,
        {n: unit(E([0.1])) for n in "abc"},
        ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.1))],
    )
    with pytest.raises(NotImplementedError, match="common-cause"):
        grouped.availability_rate(1.0)


def test_long_run_shares_of_the_failure_frequency():
    rbd = exponential_series({"a": 0.2, "b": 0.5})
    # Each fails at its rate while the other is up: 2/3 * 1/6 and
    # 5/6 * 1/3 of the frequency 7/18.
    shares = rbd.barlow_proschan_importance()
    assert shares == pytest.approx({"a": 2 / 7, "b": 5 / 7}, rel=1e-12)
    # A long window from new comes to the same.
    window = rbd.barlow_proschan_importance(window=1e4)
    assert window == pytest.approx(shares, rel=1e-6)
    with pytest.raises(ValueError, match="window"):
        rbd.barlow_proschan_importance(state={"a": NodeState(alive=False)})
    with pytest.raises(ValueError, match="positive"):
        rbd.barlow_proschan_importance(window=0.0)


def test_with_crews_and_groups_in_the_long_run():
    crews = RepairableRBD(
        PAIR_THEN_C,
        {"a": unit(E([0.1])), "b": unit(E([0.1])), "c": unit(E([0.02]))},
        repair_crews=1,
    )
    shares = crews.barlow_proschan_importance()
    assert sum(shares.values()) == pytest.approx(1.0)
    assert shares["a"] == pytest.approx(shares["b"])
    grouped = RepairableRBD(
        PAIR_THEN_C,
        {n: unit(E([0.1])) for n in "abc"},
        ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.1))],
    )
    shares = grouped.barlow_proschan_importance()
    assert set(shares) == {"a", "b", "c", ("a", "b")}
    assert sum(shares.values()) == pytest.approx(1.0)
    assert shares[("a", "b")] > 0.0


def test_window_shares_against_the_simulated_criticality():
    rbd = RepairableRBD(
        PAIR_THEN_C,
        {
            "a": unit(W([10, 2.5])),
            "b": unit(E([0.1])),
            "c": unit(W([30, 1.5]), E([0.5])),
        },
    )
    end = 40.0
    shares = rbd.barlow_proschan_importance(window=end)
    result = rbd.availability(end, mc_samples=4000, seed=3)
    simulated = result.criticalities.failure_criticality_index
    failures = rbd.expected_failures(end) * 4000
    for node, share in shares.items():
        p = simulated.per_system_failure[node]
        sd = np.sqrt(p * (1 - p) / failures)
        assert abs(share - p) < 4 * sd + 1e-3, (node, share, p)
