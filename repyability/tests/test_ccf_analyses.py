"""
Importance, parameter sensitivity, parameter uncertainty and redundancy
allocation with common-cause groups in a NonRepairableRBD (#140).

The importance measures are checked against a brute-force enumeration of
the groups' shock outcomes and every component's own failure; the
sensitivities against differences of diagrams rebuilt with the parameter
moved; each uncertainty draw against the diagram rebuilt with its models;
and each allocation against the larger group built by hand. A group with
``beta = 0`` gives what the diagram without it gives.
"""

import itertools
import math
import warnings

import numpy as np
import pytest
import scipy.stats as st
import surpyval as surv

from repyability import (
    MGL,
    BetaFactor,
    CCFGroup,
    ComponentOption,
    NonRepairableRBD,
)
from repyability.rbd.helper_classes import PerfectReliability

P = surv.FixedEventProbability.from_params
W = surv.Weibull.from_params
E = surv.Exponential.from_params

MEASURES = (
    "birnbaum_importance",
    "improvement_potential",
    "risk_achievement_worth",
    "risk_reduction_worth",
    "criticality_importance",
    "fussell_vesely",
)
PAIR = [("s", "a"), ("s", "b"), ("a", "c"), ("b", "c"), ("c", "t")]
VOTE = [("s", "a"), ("s", "b"), ("s", "c"), ("a", "t"), ("b", "t")]
VOTE += [("c", "t")]
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


@pytest.fixture(autouse=True)
def quiet():
    # Some cases evaluate a probability split beyond 0.1, which warns.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        yield


def enumerated(rbd, probabilities, group):
    """The importance measures by enumeration: every shock outcome of the
    group, and every component's own failure or not."""
    nodes = sorted(probabilities)
    members = list(group.members)
    q_own, shocks = group.model.decompose(members, probabilities[members[0]])
    outcomes = [(frozenset(s), float(np.ravel(w)[0])) for s, w in shocks]
    outcomes.append((frozenset(), 1.0 - sum(w for _, w in outcomes)))
    joint: dict = {}
    for struck, weight in outcomes:
        for own in itertools.product([False, True], repeat=len(nodes)):
            p = weight
            failed = set(struck)
            for node, fails in zip(nodes, own):
                q = float(q_own[0]) if node in members else probabilities[node]
                p *= q if fails else 1.0 - q
                if fails:
                    failed.add(node)
            key = tuple(node in failed for node in nodes)
            joint[key] = joint.get(key, 0.0) + p
    cuts = rbd.get_min_cut_sets()

    def down(key):
        failed = {n for n, f in zip(nodes, key) if f}
        return any(cut <= failed for cut in cuts)

    Q = sum(p for key, p in joint.items() if down(key))
    R = sum(p for key, p in joint.items() if not down(key))
    out = {}
    for i, node in enumerate(nodes):
        works = sum(p for key, p in joint.items() if not key[i])
        fails = sum(p for key, p in joint.items() if key[i])
        q1 = sum(p for k, p in joint.items() if not k[i] and down(k)) / works
        q0 = sum(p for k, p in joint.items() if k[i] and down(k)) / fails
        failed_cut = sum(
            p
            for key, p in joint.items()
            if any(
                node in cut and cut <= {n for n, f in zip(nodes, key) if f}
                for cut in cuts
            )
        )
        out[node] = {
            "birnbaum_importance": q0 - q1,
            "improvement_potential": fails * (q0 - q1),
            "risk_achievement_worth": q0 / Q,
            "risk_reduction_worth": Q / q1,
            "criticality_importance": (q0 - q1) * fails / Q,
            "fussell_vesely": failed_cut / Q,
            "success": (q0 - q1) * works / R,
        }
    return out


CASES = {
    "pair and valve": (
        PAIR,
        {"a": 0.2, "b": 0.2, "c": 0.05},
        CCFGroup(["a", "b"], BetaFactor(0.1)),
        None,
    ),
    "pair and valve, by rate": (
        PAIR,
        {"a": 0.2, "b": 0.2, "c": 0.05},
        CCFGroup(["a", "b"], BetaFactor(0.1, basis="rate")),
        None,
    ),
    "two of three": (
        VOTE,
        {"a": 0.1, "b": 0.1, "c": 0.1},
        CCFGroup(["a", "b", "c"], MGL(0.2, 0.3)),
        {"t": 2},
    ),
    "two of three, by rate": (
        VOTE,
        {"a": 0.1, "b": 0.1, "c": 0.1},
        CCFGroup(["a", "b", "c"], MGL(0.2, 0.3, basis="rate")),
        {"t": 2},
    ),
    "bridge": (
        BRIDGE,
        {"a": 0.3, "b": 0.3, "c": 0.1, "d": 0.2, "e": 0.25},
        CCFGroup(["a", "b"], BetaFactor(0.3)),
        None,
    ),
    "small probabilities": (
        PAIR,
        {"a": 1e-6, "b": 1e-6, "c": 1e-8},
        CCFGroup(["a", "b"], BetaFactor(0.1)),
        None,
    ),
}


def build(case):
    edges, probabilities, group, k = CASES[case]
    models = {n: P(q) for n, q in probabilities.items()}
    return (
        NonRepairableRBD(edges, models, k=k, ccf_groups=[group]),
        probabilities,
        group,
    )


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_importance_measures_are_the_enumerated_ones(case):
    rbd, probabilities, group = build(case)
    want = enumerated(rbd, probabilities, group)
    for measure in MEASURES:
        got = getattr(rbd, measure)()
        for node in probabilities:
            assert got[node] == pytest.approx(
                want[node][measure], rel=1e-10
            ), (measure, node)
    success = rbd.criticality_importance(kind="success")
    for node in probabilities:
        assert success[node] == pytest.approx(want[node]["success"], rel=1e-10)


def test_beta_zero_gives_the_independent_measures():
    models = {"a": W([100.0, 2.0]), "b": W([100.0, 2.0]), "c": E([0.002])}
    grouped = NonRepairableRBD(
        PAIR, models, ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.0))]
    )
    plain = NonRepairableRBD(PAIR, models)
    t = np.array([10.0, 60.0, 150.0])
    for measure in MEASURES:
        got, want = getattr(grouped, measure)(t), getattr(plain, measure)(t)
        for node in "abc":
            np.testing.assert_allclose(got[node], want[node], rtol=1e-12)
    for options in ({"fv_type": "p"}, {"method": "rare_event"}):
        got = grouped.fussell_vesely(t, **options)
        want = plain.fussell_vesely(t, **options)
        for node in "abc":
            np.testing.assert_allclose(got[node], want[node], rtol=1e-12)


def test_the_measures_at_several_times_are_each_times():
    rbd = NonRepairableRBD(
        PAIR,
        {"a": W([100.0, 2.0]), "b": W([100.0, 2.0]), "c": E([0.002])},
        ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.2, basis="rate"))],
    )
    t = np.array([5.0, 40.0, 120.0])
    for measure in MEASURES:
        together = getattr(rbd, measure)(t)
        for j, x in enumerate(t):
            alone = getattr(rbd, measure)(x)
            for node in "abc":
                assert together[node][j] == pytest.approx(alone[node])


def test_a_node_outside_the_groups_may_be_held():
    # With the valve held working, the pair's measures are the pair's own.
    group = CCFGroup(["a", "b"], BetaFactor(0.1))
    models = {"a": P(0.2), "b": P(0.2), "c": P(0.05)}
    rbd = NonRepairableRBD(PAIR, models, ccf_groups=[group])
    pair = NonRepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        {"a": P(0.2), "b": P(0.2)},
        ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.1))],
    )
    for measure in MEASURES:
        held = getattr(rbd, measure)(working_nodes=["c"])
        alone = getattr(pair, measure)()
        for node in "ab":
            assert held[node] == pytest.approx(alone[node], rel=1e-12)


def test_a_member_held_working_or_broken_is_refused():
    rbd, _, _ = build("pair and valve")
    for measure in MEASURES:
        with pytest.raises(NotImplementedError, match="CCF group member"):
            getattr(rbd, measure)(working_nodes=["a"])
        with pytest.raises(NotImplementedError, match="CCF group member"):
            getattr(rbd, measure)(broken_nodes=["b"])
    with pytest.raises(ValueError, match="kind"):
        rbd.criticality_importance(kind="both")
    with pytest.raises(ValueError, match="fv_type"):
        rbd.fussell_vesely(fv_type="x")
    with pytest.raises(ValueError, match="method"):
        rbd.fussell_vesely(method="bdd")


# -- parameter sensitivity ------------------------------------------------


def rebuilt(models, model, members=("a", "b"), edges=PAIR, k=None):
    return NonRepairableRBD(
        edges, models, k=k, ccf_groups=[CCFGroup(list(members), model)]
    )


@pytest.mark.parametrize(
    "model", [BetaFactor(0.1), BetaFactor(0.1, basis="rate")]
)
def test_a_groups_parameters_are_reported_together(model):
    t = np.array([50.0, 300.0])
    valve = E([1e-4])

    def sf(alpha=1000.0, shape=1.7, beta=model.beta):
        pump = W([alpha, shape])
        return rebuilt(
            {"a": pump, "b": pump, "c": valve},
            BetaFactor(beta, basis=model.basis),
        ).sf(t)

    rbd = rebuilt(
        {"a": W([1000.0, 1.7]), "b": W([1000.0, 1.7]), "c": valve}, model
    )
    sens = rbd.parameter_sensitivity(t)
    assert set(sens) == {("a", "b"), "c"}
    group = sens[("a", "b")]
    assert list(group) == ["alpha", "beta", "ccf_beta"]
    np.testing.assert_allclose(
        group["alpha"],
        (sf(alpha=1000.01) - sf(alpha=999.99)) / 0.02,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        group["beta"],
        (sf(shape=1.7 + 1e-6) - sf(shape=1.7 - 1e-6)) / 2e-6,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        group["ccf_beta"],
        (sf(beta=0.1 + 1e-6) - sf(beta=0.1 - 1e-6)) / 2e-6,
        rtol=1e-6,
    )
    # A node outside the group is as without groups: Birnbaum times dsf.
    rate = 1e-4
    dsf = -t * np.exp(-rate * t)
    birnbaum = rbd.birnbaum_importance(t)["c"]
    np.testing.assert_allclose(
        sens["c"]["failure_rate"], birnbaum * dsf, rtol=1e-6
    )


def test_beta_zero_sensitivity_is_the_members_together():
    pump, valve = W([1000.0, 1.7]), E([1e-4])
    models = {"a": pump, "b": pump, "c": valve}
    t = np.array([50.0, 300.0])
    grouped = rebuilt(models, BetaFactor(0.0)).parameter_sensitivity(t)
    plain = NonRepairableRBD(PAIR, models).parameter_sensitivity(t)
    for name in ("alpha", "beta"):
        np.testing.assert_allclose(
            grouped[("a", "b")][name],
            plain["a"][name] + plain["b"][name],
            rtol=1e-6,
        )
    np.testing.assert_allclose(
        grouped["c"]["failure_rate"], plain["c"]["failure_rate"], rtol=1e-9
    )
    # beta at the edge of [0, 1]: differenced forward.
    forward = (
        rebuilt(models, BetaFactor(1e-7)).sf(t)
        - rebuilt(models, BetaFactor(0.0)).sf(t)
    ) / 1e-7
    np.testing.assert_allclose(
        grouped[("a", "b")]["ccf_beta"], forward, rtol=1e-4
    )


def test_an_mgl_models_letters_are_named():
    def sf(q=0.05, beta=0.1, gamma=0.3):
        unit = P(q)
        return rebuilt(
            {n: unit for n in "abc"},
            MGL(beta, gamma),
            members="abc",
            edges=VOTE,
            k={"t": 2},
        ).sf()

    rbd = rebuilt(
        {n: P(0.05) for n in "abc"},
        MGL(0.1, 0.3),
        members="abc",
        edges=VOTE,
        k={"t": 2},
    )
    sens = rbd.parameter_sensitivity()[("a", "b", "c")]
    assert list(sens) == ["p", "ccf_beta", "ccf_gamma"]
    h = 1e-7
    assert sens["p"] == pytest.approx(
        (sf(q=0.05 + h) - sf(q=0.05 - h)) / (2 * h), rel=1e-6
    )
    assert sens["ccf_beta"] == pytest.approx(
        (sf(beta=0.1 + h) - sf(beta=0.1 - h)) / (2 * h), rel=1e-6
    )
    assert sens["ccf_gamma"] == pytest.approx(
        (sf(gamma=0.3 + h) - sf(gamma=0.3 - h)) / (2 * h), rel=1e-6
    )


def test_a_small_probabilitys_sensitivity_keeps_its_precision():
    # The system fails with beta q + (1 - beta q) ((1 - beta) q) ** 2.
    q, beta = 1e-9, 0.1
    rbd = rebuilt({"a": P(q), "b": P(q), "c": P(0.0)}, BetaFactor(beta))
    sens = rbd.parameter_sensitivity()[("a", "b")]
    d_q = beta + (1 - beta) ** 2 * (2 * q * (1 - beta * q) - beta * q**2)
    assert -sens["p"] == pytest.approx(d_q, rel=1e-8)
    d_beta = q - (1 - beta) * q**2 * (2 * (1 - beta * q) + q * (1 - beta))
    assert -sens["ccf_beta"] == pytest.approx(d_beta, rel=1e-8)


# -- parameter uncertainty ------------------------------------------------

RATE = {"failure_rate": st.uniform(0.001, 0.002)}


def pumps(model=BetaFactor(0.1, basis="rate")):
    group = CCFGroup(["a", "b"], model)
    pump = E([0.002])
    models = {"a": pump, "b": pump, "c": P(0.01)}
    return NonRepairableRBD(PAIR, models, ccf_groups=[group]), group


def test_each_draw_is_the_diagram_rebuilt_with_its_models():
    rbd, group = pumps()
    t = np.array([100.0, 500.0])
    for spec in (
        {("a", "b"): RATE},
        {("a", "b"): RATE, group: {"beta": st.beta(2, 18)}},
        {
            group: [
                BetaFactor(0.05, basis="rate"),
                BetaFactor(0.2, basis="rate"),
            ]
        },
    ):
        result = rbd.sf_uncertainty(t, spec, n_draws=40, seed=1)
        drawn, groups = rbd._uncertain_draws(spec, 40, 1)
        for i in range(40):
            pump = drawn["a"][i] if "a" in drawn else E([0.002])
            model = group.model if groups[0] is None else groups[0][i]
            want = rebuilt({"a": pump, "b": pump, "c": P(0.01)}, model).sf(t)
            np.testing.assert_array_equal(result.samples[i], want)
        assert result.nominal == pytest.approx(rbd.sf(t))


def test_the_nodes_draws_are_the_same_with_the_groups_uncertainty():
    rbd, group = pumps()
    alone, _ = rbd._uncertain_draws({("a", "b"): RATE}, 30, 5)
    both, groups = rbd._uncertain_draws(
        {("a", "b"): RATE, group: {"beta": st.uniform(0.05, 0.1)}}, 30, 5
    )
    assert [m.params[0] for m in alone["a"]] == [
        m.params[0] for m in both["a"]
    ]
    assert all(0.05 <= m.beta <= 0.15 for m in groups[0])


def test_the_mttf_and_times_of_each_draw():
    rbd, group = pumps()
    spec = {("a", "b"): RATE, group: {"beta": st.beta(2, 18)}}
    mean = rbd.mean_uncertainty(spec, n_draws=15, seed=2)
    times = rbd.time_to_reliability_uncertainty(0.9, spec, n_draws=15, seed=2)
    drawn, groups = rbd._uncertain_draws(spec, 15, 2)
    for i in range(15):
        pump = drawn["a"][i]
        diagram = rebuilt({"a": pump, "b": pump, "c": P(0.01)}, groups[0][i])
        assert mean.samples[i] == pytest.approx(diagram.mean(), rel=1e-12)
        assert times.samples[i] == pytest.approx(
            diagram.time_to_reliability(0.9), rel=1e-12
        )
    assert mean.nominal == pytest.approx(rbd.mean())
    b10 = rbd.bx_life_uncertainty(10, spec, n_draws=15, seed=2)
    np.testing.assert_array_equal(b10.samples, times.samples)


def test_beta_zero_uncertainty_is_independent():
    rbd, _ = pumps(BetaFactor(0.0))
    plain = NonRepairableRBD(
        PAIR, {"a": E([0.002]), "b": E([0.002]), "c": P(0.01)}
    )
    t = np.array([100.0, 500.0])
    grouped = rbd.sf_uncertainty(t, {("a", "b"): RATE}, n_draws=50, seed=3)
    alone = plain.sf_uncertainty(t, {("a", "b"): RATE}, n_draws=50, seed=3)
    np.testing.assert_allclose(grouped.samples, alone.samples, rtol=1e-13)


def test_a_probability_split_has_no_mttf_uncertainty():
    rbd, _ = pumps(BetaFactor(0.1))
    with pytest.raises(NotImplementedError, match="split a failure"):
        rbd.mean_uncertainty({("a", "b"): RATE}, n_draws=5)
    routes = rbd.analysis_routes()
    assert routes["mean_uncertainty"].route == "refused"
    for name in ("sf_uncertainty", "time_to_reliability_uncertainty"):
        assert routes[name].route == "simulated"
    # Its reliability's uncertainty is there.
    result = rbd.sf_uncertainty(100.0, {("a", "b"): RATE}, n_draws=5, seed=0)
    assert result.samples.shape == (5,)


def test_what_cannot_be_drawn_is_refused():
    rbd, group = pumps()
    for spec, match in (
        ({"a": RATE}, "together"),
        ({("a",): RATE}, "together"),
        ({group: "fit"}, "dict of distributions"),
        ({group: {}}, "no parameter distributions"),
        ({group: {"gamma": st.uniform(0, 1)}}, "not parameters"),
        ({group: {"beta": st.uniform(0, 2)}}, r"outside \[0, 1\]"),
        ({group: {"beta": object()}}, "quantile function"),
        ({group: []}, "empty"),
        ({group: [BetaFactor(0.1)]}, "splitting the rate"),
        ({group: [MGL(0.1, 0.2, basis="rate")]}, "for 2 members"),
        ({group: [E([0.1])]}, "BetaFactor or an MGL"),
    ):
        with pytest.raises(ValueError, match=match):
            rbd.sf_uncertainty(100.0, spec, n_draws=5, seed=0)
    # Another diagram's group (an equal one) is not this diagram's.
    other = CCFGroup(["a", "b"], BetaFactor(0.1, basis="rate"))
    with pytest.raises(ValueError, match="not a component node"):
        rbd.sf_uncertainty(100.0, {other: {"beta": st.uniform(0, 1)}})


# -- redundancy allocation ------------------------------------------------


def larger_group(copies, model, required=None):
    """Each of a and b as a block of copies, every copy in one group."""
    pump, valve = W([1000.0, 1.7]), E([2e-4])
    edges, models, members = [], {}, []
    for name, n in copies.items():
        for i in range(n):
            copy = f"{name}{i}"
            edges += [("s", copy), (copy, f"{name}_out")]
            models[copy] = pump
            members.append(copy)
        edges.append((f"{name}_out", "c"))
        models[f"{name}_out"] = PerfectReliability
    edges.append(("c", "t"))
    models["c"] = valve
    k = {f"{n}_out": k for n, k in (required or {}).items()}
    return NonRepairableRBD(
        edges, models, k=k or None, ccf_groups=[CCFGroup(members, model)]
    )


def pumps_and_valve(model):
    pump, valve = W([1000.0, 1.7]), E([2e-4])
    return NonRepairableRBD(
        PAIR,
        {"a": pump, "b": pump, "c": valve},
        ccf_groups=[CCFGroup(["a", "b"], model)],
    )


@pytest.mark.parametrize(
    "model", [BetaFactor(0.1), BetaFactor(0.15, basis="rate")]
)
def test_copies_join_the_group(model):
    rbd = pumps_and_valve(model)
    front = rbd.redundancy_front({"a": 1.0, "b": 1.0}, budget=7, t=300.0)
    assert len(front) > 2
    for design in front:
        want = larger_group(design.units, model).sf(300.0)
        assert design.reliability == pytest.approx(float(want), rel=1e-13)
    best = rbd.allocate_redundancy({"a": 1.0, "b": 1.0}, budget=5, t=300.0)
    greedy = rbd.allocate_redundancy(
        {"a": 1.0, "b": 1.0}, budget=5, t=300.0, method="greedy"
    )
    assert greedy.reliability == pytest.approx(best.reliability)
    assert best.reliability == pytest.approx(
        float(larger_group(best.units, model).sf(300.0)), rel=1e-13
    )
    cheapest = rbd.allocate_redundancy(
        {"a": 1.0, "b": 1.0}, target=0.999 * best.reliability, t=300.0
    )
    assert cheapest.reliability >= 0.999 * best.reliability
    assert cheapest.cost <= best.cost


def test_the_shared_shock_caps_what_copies_can_reach():
    rbd = pumps_and_valve(BetaFactor(0.1))
    with pytest.raises(ValueError, match="unreachable"):
        rbd.allocate_redundancy({"a": 1.0, "b": 1.0}, target=0.99, t=300.0)


def test_k_of_n_copies_of_a_member():
    model = BetaFactor(0.1)
    rbd = pumps_and_valve(model)
    best = rbd.allocate_redundancy(
        {"a": 1.0}, budget=4, t=300.0, required={"a": 2}
    )
    want = larger_group({"a": best.units["a"], "b": 1}, model, {"a": 2})
    assert best.reliability == pytest.approx(float(want.sf(300.0)), rel=1e-13)


def test_beta_zero_allocation_is_independent():
    pump, valve = W([1000.0, 1.7]), E([2e-4])
    models = {"a": pump, "b": pump, "c": valve}
    grouped = pumps_and_valve(BetaFactor(0.0))
    plain = NonRepairableRBD(PAIR, models)
    costs = {"a": 1.0, "b": 2.0, "c": 1.5}
    for options in ({"budget": 8}, {"target": 0.99, "budget": 12}):
        got = grouped.allocate_redundancy(costs, t=300.0, **options)
        want = plain.allocate_redundancy(costs, t=300.0, **options)
        assert got.units == want.units
        assert got.reliability == pytest.approx(want.reliability, rel=1e-13)


def test_what_cannot_join_a_group_is_refused():
    rbd = pumps_and_valve(BetaFactor(0.1))
    mgl = pumps_and_valve(MGL(0.1))
    choice = [ComponentOption("pump", W([1000.0, 1.7]), 1.0)]
    for call, match in (
        (
            lambda: mgl.allocate_redundancy({"a": 1.0}, budget=3, t=300.0),
            "MGL",
        ),
        (
            lambda: rbd.allocate_redundancy({"a": choice}, budget=3, t=300.0),
            "no component options",
        ),
        (
            lambda: rbd.allocate_redundancy(
                {"a": 1.0}, budget=3, t=300.0, strategy="cold"
            ),
            "active",
        ),
        (
            lambda: rbd.allocate_redundancy(
                {"a": 1.0, "c": 1.0}, budget=4, t=300.0, strategy="choose"
            ),
            "active",
        ),
    ):
        with pytest.raises(NotImplementedError, match=match):
            call()
    # A node outside an MGL group is copied as ever.
    best = mgl.allocate_redundancy({"c": 1.0}, budget=3, t=300.0)
    block = 1.0 - (1.0 - float(E([2e-4]).sf(300.0))) ** best.units["c"]
    pump = W([1000.0, 1.7])
    want = NonRepairableRBD(
        PAIR,
        {"a": pump, "b": pump, "c": P(1.0 - block)},
        ccf_groups=[CCFGroup(["a", "b"], MGL(0.1))],
    ).sf(300.0)
    assert best.reliability == pytest.approx(float(want), rel=1e-12)


def test_reliability_redundancy_with_a_group_elsewhere():
    rbd = pumps_and_valve(BetaFactor(0.1))

    def cost(r, n):
        return n * (1.0 + (-1.0 / math.log(r)))

    best = rbd.allocate_reliability_redundancy(
        {"c": cost}, budget=20, bounds=(0.5, 0.9999), t=300.0
    )
    r, n = best.component_reliability["c"], best.units["c"]
    pump = W([1000.0, 1.7])
    want = NonRepairableRBD(
        PAIR,
        {"a": pump, "b": pump, "c": P((1.0 - r) ** n)},
        ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.1))],
    ).sf(300.0)
    assert best.reliability == pytest.approx(float(want), rel=1e-12)
    with pytest.raises(NotImplementedError, match="common-cause group"):
        rbd.allocate_reliability_redundancy(
            {"a": cost}, budget=20, bounds=(0.5, 0.9999), t=300.0
        )


def test_the_routes_say_how_the_groups_are_worked_out():
    routes = pumps_and_valve(BetaFactor(0.1)).analysis_routes()
    for name in MEASURES:
        assert routes[name].route == "exact"
        assert "common-cause" in routes[name].reason
    assert routes["parameter_sensitivity"].route == "numerical"
    assert routes["allocate_redundancy"].route == "exact"
    assert "join its group" in routes["allocate_redundancy"].reason
    assert routes["allocate_reliability_redundancy"].route == "numerical"
    assert routes["sf_uncertainty"].route == "simulated"
