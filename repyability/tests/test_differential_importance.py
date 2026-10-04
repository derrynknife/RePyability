"""The differential importance measure (#193): each component's or
parameter's share of the change in the system when they all change
together, uniformly or in proportion.

Checked by its meaning: the systems rebuilt with every probability (or
parameter) moved, and with only a group of them moved, change in the ratio
of the group's shares, to first order. And against the measures it shares
out: the Birnbaum importance for a uniform change, the criticality
importance for a proportional one."""

import numpy as np
import pytest
import surpyval as surv
from surpyval import FixedEventProbability

from repyability import (
    BetaFactor,
    CCFGroup,
    FaultTree,
    NodeState,
    NonRepairableRBD,
    RepairableRBD,
)

E, W = surv.Exponential.from_params, surv.Weibull.from_params
F = FixedEventProbability.from_params

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
Q = {"a": 0.1, "b": 0.2, "c": 0.05, "d": 0.3, "e": 0.15}


def bridge(q):
    return NonRepairableRBD(BRIDGE, {n: F(p) for n, p in q.items()})


def changed(reliability, q, moved, step):
    """The central difference of ``reliability`` when each node in
    ``moved`` has its probability moved by ``step(node)``."""

    def at(sign):
        return reliability(
            {n: p + sign * step(n) if n in moved else p for n, p in q.items()}
        )

    return (at(1) - at(-1)) / 2


def bridge_reliability(q):
    return bridge(q).sf()


@pytest.mark.parametrize(
    "change, kind, step",
    [
        ("uniform", "failure", lambda n: 1e-5),
        ("proportional", "failure", lambda n: 1e-5 * Q[n]),
        # In proportion to the probability of working, which moves the
        # probability of failing the other way.
        ("proportional", "success", lambda n: -1e-5 * (1 - Q[n])),
    ],
)
def test_a_group_s_share_is_its_part_of_the_change(change, kind, step):
    shares = bridge(Q).differential_importance(change=change, kind=kind)
    assert sum(shares.values()) == pytest.approx(1.0, abs=1e-12)
    every = changed(bridge_reliability, Q, set(Q), step)
    for group in ({"a"}, {"c"}, {"a", "b"}, {"c", "d", "e"}):
        part = changed(bridge_reliability, Q, group, step)
        assert sum(shares[n] for n in group) == pytest.approx(
            part / every, rel=1e-7
        )


def test_uniform_and_proportional_share_out_birnbaum_and_criticality():
    rbd = bridge(Q)
    birnbaum = rbd.birnbaum_importance()
    total = sum(birnbaum.values())
    for node, share in rbd.differential_importance().items():
        assert share == pytest.approx(birnbaum[node] / total, rel=1e-12)
    for kind in ("failure", "success"):
        critical = rbd.criticality_importance(kind=kind)
        total = sum(critical.values())
        shares = rbd.differential_importance(change="proportional", kind=kind)
        for node, share in shares.items():
            assert share == pytest.approx(critical[node] / total, rel=1e-12)
    # A uniform change of the probabilities of working is the same change
    # of the probabilities of failing, the other way: the same shares.
    assert rbd.differential_importance(kind="success") == (
        rbd.differential_importance()
    )


def test_groups_add_up_their_members():
    rbd = bridge(Q)
    shares = rbd.differential_importance(change="proportional")
    grouped = rbd.differential_importance(
        change="proportional",
        groups={"left": ["a", "b"], "middle": ["c"], "right": ["d", "e"]},
    )
    assert list(grouped) == ["left", "middle", "right"]
    assert grouped["left"] == pytest.approx(shares["a"] + shares["b"])
    assert grouped["right"] == pytest.approx(shares["d"] + shares["e"])
    assert sum(grouped.values()) == pytest.approx(1.0)
    # A key may be in more than one group, and a group may hold one key.
    overlapping = rbd.differential_importance(
        change="proportional", groups={"all": list(Q), "a": ["a"]}
    )
    assert overlapping == pytest.approx({"all": 1.0, "a": shares["a"]})


def test_at_times_each_time_is_shared_out():
    rbd = NonRepairableRBD(
        BRIDGE, {n: W([100 + 20 * i, 1.5]) for i, n in enumerate(Q)}
    )
    x = np.array([5.0, 50.0, 200.0])
    shares = rbd.differential_importance(x)
    assert all(np.shape(v) == (3,) for v in shares.values())
    np.testing.assert_allclose(sum(shares.values()), 1.0, atol=1e-12)
    for i, t in enumerate(x):
        at = rbd.differential_importance(t)
        assert isinstance(at["a"], float)
        for node, share in at.items():
            assert share == pytest.approx(shares[node][i], rel=1e-10)


def test_held_nodes_take_no_part():
    rbd = bridge(Q)
    shares = rbd.differential_importance(working_nodes=["c"])
    assert shares["c"] == 0.0
    assert sum(shares.values()) == pytest.approx(1.0)
    # With c working, the bridge is a and b, each in series with its side.
    birnbaum = rbd.birnbaum_importance(working_nodes=["c"])
    total = sum(v for n, v in birnbaum.items() if n != "c")
    assert shares["a"] == pytest.approx(birnbaum["a"] / total)


def weibull_bridge(alphas, betas):
    return NonRepairableRBD(
        BRIDGE,
        {n: W([alphas[n], betas[n]]) for n in Q},
    )


def test_parameters_share_out_the_reliability_s_change():
    alphas = {n: 100.0 + 25 * i for i, n in enumerate(Q)}
    betas = {n: 1.2 + 0.3 * i for i, n in enumerate(Q)}
    x = 60.0
    rbd = weibull_bridge(alphas, betas)
    for change in ("uniform", "proportional"):
        shares = rbd.differential_importance(
            x, over="parameters", change=change
        )
        assert set(shares) == {(n, p) for n in Q for p in ("alpha", "beta")}
        assert sum(shares.values()) == pytest.approx(1.0)

        def moved(which, sign, change=change):
            def by(value):
                step = 1e-5 * (value if change == "proportional" else 1.0)
                return value + sign * step

            a = {
                n: by(v) if "alpha" in which else v for n, v in alphas.items()
            }
            b = {n: by(v) if "beta" in which else v for n, v in betas.items()}
            return weibull_bridge(a, b).sf(x)

        every = moved({"alpha", "beta"}, 1) - moved({"alpha", "beta"}, -1)
        scales = moved({"alpha"}, 1) - moved({"alpha"}, -1)
        grouped = rbd.differential_importance(
            x,
            over="parameters",
            change=change,
            groups={"scales": [(n, "alpha") for n in Q]},
        )
        assert grouped["scales"] == pytest.approx(scales / every, rel=1e-4)


def test_a_common_cause_group_s_parameters_are_its_own():
    rbd = NonRepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "c"), ("b", "c"), ("c", "t")],
        {"a": W([100, 2]), "b": W([100, 2]), "c": F(0.05)},
        ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.1))],
    )
    derivatives = rbd.parameter_sensitivity(30.0)
    values = {(("a", "b"), "alpha"): 100.0, (("a", "b"), "beta"): 2.0}
    values.update({(("a", "b"), "ccf_beta"): 0.1, ("c", "p"): 0.05})
    contributions = {
        (key, name): d * values[(key, name)]
        for key, by in derivatives.items()
        for name, d in by.items()
    }
    total = sum(contributions.values())
    shares = rbd.differential_importance(
        30.0, over="parameters", change="proportional"
    )
    assert shares == pytest.approx(
        {key: v / total for key, v in contributions.items()}
    )


def repairable(life_scale=1.0, repair_scale=1.0):
    def unit(alpha, beta, repair):
        return {
            "reliability": W([alpha * life_scale, beta * life_scale]),
            "repairability": E([repair * repair_scale]),
        }

    return RepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "c"), ("b", "c"), ("c", "t")],
        {
            "a": unit(100.0, 2.0, 0.5),
            "b": unit(150.0, 1.5, 0.25),
            "c": unit(400.0, 1.2, 1.0),
        },
    )


def test_a_repairable_diagram_s_components_over_time():
    rbd = repairable()
    long_run = rbd.differential_importance()
    birnbaum = rbd.birnbaum_importance()
    total = sum(birnbaum.values())
    assert long_run == pytest.approx(
        {n: v / total for n, v in birnbaum.items()}
    )
    critical = rbd.criticality_importance()
    total = sum(critical.values())
    assert rbd.differential_importance(change="proportional") == (
        pytest.approx({n: v / total for n, v in critical.items()})
    )
    x = [10.0, 100.0]
    over_time = rbd.differential_importance(x=x)
    birnbaum = rbd.birnbaum_importance(x=x)
    total = sum(birnbaum.values())
    for node, share in over_time.items():
        np.testing.assert_allclose(share, birnbaum[node] / total, rtol=1e-12)
    window = rbd.differential_importance(window=100.0, change="proportional")
    critical = rbd.criticality_importance(window=100.0)
    total = sum(critical.values())
    assert window == pytest.approx({n: v / total for n, v in critical.items()})
    state = rbd.differential_importance(
        x=5.0, state={"a": NodeState(alive=False)}
    )
    birnbaum = rbd.birnbaum_importance(
        x=5.0, state={"a": NodeState(alive=False)}
    )
    total = sum(birnbaum.values())
    assert state == pytest.approx({n: v / total for n, v in birnbaum.items()})


def test_a_repairable_diagram_s_lives_against_its_repairs():
    # Every life parameter moved in proportion, against every repair rate:
    # the share of the availability's change that lies in the repairs.
    rbd = repairable()
    shares = rbd.differential_importance(
        over="parameters",
        change="proportional",
        groups={
            "lives": [
                (n, f"reliability.{p}")
                for n in "abc"
                for p in ("alpha", "beta")
            ],
            "repairs": [(n, "repairability.failure_rate") for n in "abc"],
        },
    )
    step = 1e-4

    def availability(life, repair):
        return repairable(life, repair).mean_availability()

    every = availability(1 + step, 1 + step) - availability(1 - step, 1 - step)
    repairs = availability(1, 1 + step) - availability(1, 1 - step)
    assert shares["repairs"] == pytest.approx(repairs / every, rel=1e-5)
    assert shares["lives"] + shares["repairs"] == pytest.approx(1.0)


def test_rates_that_cancel_share_nothing_out():
    # Exponential units' failure and repair rates, moved in proportion,
    # leave their availabilities, and the system's, as they were.
    rbd = RepairableRBD(
        [("s", "a"), ("a", "b"), ("b", "t")],
        {
            n: {"reliability": E([rate]), "repairability": E([1.0])}
            for n, rate in (("a", 0.1), ("b", 0.25))
        },
    )
    shares = rbd.differential_importance(
        over="parameters", change="proportional"
    )
    assert len(shares) == 4
    assert all(np.isnan(v) for v in shares.values())
    # A uniform change does not cancel.
    shares = rbd.differential_importance(over="parameters")
    assert sum(shares.values()) == pytest.approx(1.0)


def test_one_more_crew_or_unit_is_no_derivative():
    rbd = RepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "c"), ("b", "c"), ("c", "t")],
        {
            n: {"reliability": E([0.01]), "repairability": E([0.5])}
            for n in "abc"
        },
        repair_crews=1,
    )
    shares = rbd.differential_importance(over="parameters")
    assert all(key is not None for key, _ in shares)
    assert sum(shares.values()) == pytest.approx(1.0)


TREE = FaultTree(
    {
        "top": ("or", ["valve", "flow"]),
        "flow": ("and", ["pump 1", "pump 2"]),
    },
    {"pump 1": 0.1, "pump 2": 0.2, "valve": 0.05},
)


def tree_probability(q):
    return FaultTree(
        {
            "top": ("or", ["valve", "flow"]),
            "flow": ("and", ["pump 1", "pump 2"]),
        },
        q,
    ).top_event_probability()


@pytest.mark.parametrize("change", ["uniform", "proportional"])
def test_a_fault_tree_s_events(change):
    q = {"pump 1": 0.1, "pump 2": 0.2, "valve": 0.05}
    shares = TREE.differential_importance(change=change)
    step = {
        "uniform": lambda e: 1e-5,
        "proportional": lambda e: 1e-5 * q[e],
    }[change]
    every = changed(tree_probability, q, set(q), step)
    pumps = changed(tree_probability, q, {"pump 1", "pump 2"}, step)
    grouped = TREE.differential_importance(
        change=change, groups={"pumps": ["pump 1", "pump 2"]}
    )
    assert grouped["pumps"] == pytest.approx(pumps / every, rel=1e-7)
    assert grouped["pumps"] == pytest.approx(
        shares["pump 1"] + shares["pump 2"]
    )
    measure = (
        TREE.birnbaum_importance()
        if change == "uniform"
        else TREE.criticality_importance()
    )
    total = sum(measure.values())
    assert shares == pytest.approx({e: v / total for e, v in measure.items()})


def test_a_fault_tree_over_time():
    tree = FaultTree(
        {"top": ("and", ["x", "y"])}, {"x": W([100, 2]), "y": E([0.01])}
    )
    shares = tree.differential_importance([10.0, 100.0])
    np.testing.assert_allclose(sum(shares.values()), 1.0)
    birnbaum = tree.birnbaum_importance([10.0, 100.0])
    total = sum(birnbaum.values())
    for event, share in shares.items():
        np.testing.assert_allclose(share, birnbaum[event] / total)


def test_bad_arguments_are_refused():
    rbd = bridge(Q)
    with pytest.raises(ValueError, match="change must be"):
        rbd.differential_importance(change="relative")
    with pytest.raises(ValueError, match="over must be"):
        rbd.differential_importance(over="nodes")
    with pytest.raises(ValueError, match="kind must be"):
        rbd.differential_importance(kind="up")
    with pytest.raises(ValueError, match="not among the keys"):
        rbd.differential_importance(groups={"g": ["a", "z"]})
    with pytest.raises(TypeError, match="groups must map"):
        rbd.differential_importance(groups=[["a", "b"]])
    with pytest.raises(ValueError, match="change must be"):
        repairable().differential_importance(change="relative")
    with pytest.raises(ValueError, match="change must be"):
        TREE.differential_importance(change="relative")
    with pytest.raises(ValueError, match="not among the keys"):
        TREE.differential_importance(groups={"g": ["pump 3"]})


def test_improving_shares_out_a_gain():
    # Moved in proportion the way that improves it, an exponential unit's
    # failure and repair rates are worth the same: its availability
    # depends on their ratio alone.
    rbd = RepairableRBD(
        [("s", "a"), ("a", "b"), ("b", "t")],
        {
            n: {"reliability": E([rate]), "repairability": E([1.0])}
            for n, rate in (("a", 0.1), ("b", 0.25))
        },
    )
    shares = rbd.differential_importance(
        over="parameters", change="proportional", improving=True
    )
    for n in "ab":
        assert shares[(n, "reliability.failure_rate")] == pytest.approx(
            shares[(n, "repairability.failure_rate")], rel=1e-6
        )
    assert all(0 <= v <= 1 for v in shares.values())
    assert sum(shares.values()) == pytest.approx(1.0)
    # Each one's size, shared out.
    derivatives = rbd.parameter_sensitivity()
    sizes = {
        (n, name): (
            abs(d) * (0.1 if n == "a" else 0.25)
            if name.startswith("reliability")
            else abs(d)
        )
        for n, by in derivatives.items()
        for name, d in by.items()
    }
    total = sum(sizes.values())
    assert shares == pytest.approx({k: v / total for k, v in sizes.items()})
    # The nodes' shares are the same either way.
    assert rbd.differential_importance(improving=True) == (
        rbd.differential_importance()
    )


def test_improving_parameters_of_a_non_repairable_diagram():
    rbd = weibull_bridge(
        {n: 100.0 + 25 * i for i, n in enumerate(Q)},
        {n: 1.2 + 0.3 * i for i, n in enumerate(Q)},
    )
    # By 150, past some nodes' scales, a larger shape lowers their
    # reliability, and raises the others'.
    plain = rbd.differential_importance(
        150.0, over="parameters", change="proportional"
    )
    improving = rbd.differential_importance(
        150.0, over="parameters", change="proportional", improving=True
    )
    assert any(v < 0 for v in plain.values())
    assert all(v >= 0 for v in improving.values())
    total = sum(abs(v) for v in plain.values())
    assert improving == pytest.approx(
        {k: abs(v) / total for k, v in plain.items()}
    )
