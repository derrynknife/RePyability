"""Common-cause groups worked out module by module, or as shock events
(#219).

Each group is conditioned on within the smallest module holding its
members, and where a module's groups would multiply their outcomes they
are written out as shock events instead: either way the values are those
of conditioning on every combination of every group's outcomes (what
``ccf.shock_outcomes`` enumerates), which the tests take as the reference,
in time that grows with the number of groups rather than doubling.
"""

import itertools
import random
import time

import numpy as np
import pytest
import surpyval as surv

from repyability import (
    MGL,
    AlphaFactor,
    BetaFactor,
    CCFGroup,
    FaultTree,
    NonRepairableRBD,
)
from repyability.rbd import _ccf_modules
from repyability.rbd.ccf import _as_independent, shock_outcomes

# The diagrams here reach large probabilities of failing on purpose, to try
# the arithmetic: past the rare-event range a group's model is meant for,
# which it warns about.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Common-cause group:UserWarning"
)

KEYS = ("works", "fails", "up_ok", "down_ok", "up_bad", "down_bad")


def chosen_pairs(rbd) -> list:
    """Pairs of nodes of each kind there is: two members of one group,
    members of two groups, a member and a node outside the groups, and two
    outside; the first of each."""
    group_of = {m: k for k, g in enumerate(rbd.ccf_groups) for m in g.members}
    kinds: dict = {}
    for i, j in itertools.combinations(rbd.nodes, 2):
        a, b = group_of.get(i), group_of.get(j)
        kind = (
            "outside"
            if a is None and b is None
            else (
                "mixed"
                if a is None or b is None
                else "within" if a == b else "across"
            )
        )
        kinds.setdefault(kind, (i, j))
    return list(kinds.values())


def every_combination(rbd, x, pairs=()):
    """What ``_ccf_modules.Evaluation.states`` gives of each node and of
    each pair of ``pairs``, and the Fussell-Vesely numerators, conditioned
    on every combination of every group's outcomes."""
    p = {n: np.atleast_1d(rbd.reliabilities[n].sf(x)) for n in rbd.nodes}
    q = {n: np.atleast_1d(rbd.reliabilities[n].ff(x)) for n in rbd.nodes}
    grouped = {m for g in rbd.ccf_groups for m in g.members}
    sums: dict = {key: {} for key in KEYS}
    R = Q = 0.0
    fv: dict = {}
    states = {(i, j): {} for i, j in pairs}
    for w, given, failing in shock_outcomes(rbd.ccf_groups, p, q):
        up, down = rbd._system_probabilities(given, failing)
        R, Q = R + w * up, Q + w * down

        def held(hold: dict):
            """The system with each node of ``hold`` failed (1.0) or
            working (0.0)."""
            down = {n: s * np.ones_like(up) for n, s in hold.items()}
            return rbd._system_probabilities(
                {**given, **{n: 1.0 - d for n, d in down.items()}},
                {**failing, **down},
            )

        for node in rbd.nodes:
            works, fails = given[node], failing[node]
            u1, d1 = held({node: 0.0})
            u0, d0 = held({node: 1.0})
            on, off = (works, fails) if node in grouped else (1.0, 1.0)
            for key, value in zip(
                KEYS,
                (works, fails, on * u1, on * d1, off * u0, off * d0),
            ):
                sums[key][node] = sums[key].get(node, 0.0) + w * value
        for i, j in pairs:
            for a, b in itertools.product((True, False), repeat=2):
                chance = w
                for node, up_ in ((i, a), (j, b)):
                    if node in grouped:
                        chance = chance * (given if up_ else failing)[node]
                u, d = held({i: 0.0 if a else 1.0, j: 0.0 if b else 1.0})
                was = states[i, j].get((a, b), (0.0, 0.0, 0.0))
                states[i, j][a, b] = (
                    was[0] + chance,
                    was[1] + chance * u,
                    was[2] + chance * d,
                )
        pairs_p, failures, size = rbd._node_pairs(given, failing)
        structure = rbd._set_structure()
        for node, value in structure.failed_cut_sets(
            pairs_p, failures, shape=size
        ).items():
            fv[node] = fv.get(node, 0.0) + w * value
    return R, Q, sums, fv, states


def joint_of(states: dict) -> np.ndarray:
    """The joint importance from a pair's states (see
    ``every_combination``): the system given each combination."""
    given = {combo: up / chance for combo, (chance, up, _) in states.items()}
    return (
        given[True, True]
        - given[True, False]
        - given[False, True]
        + given[False, False]
    )


def agree(rbd, x, rel=1e-10):
    pairs = chosen_pairs(rbd)
    R, Q, sums, fv, states = every_combination(rbd, x, pairs)
    up, down = rbd._sf_and_ff(x, None, None)
    np.testing.assert_allclose(up, R, rtol=rel)
    np.testing.assert_allclose(down, Q, rtol=rel)
    evaluation = rbd._ccf_evaluation(
        *rbd._importance_inputs(x, set(), set()), set(), set()
    )
    grouped = {m for g in rbd.ccf_groups for m in g.members}
    for node in rbd.nodes:
        ours = evaluation.states((node,))
        (works, up_ok, down_ok), (fails, up_bad, down_bad) = (
            ours[True,],
            ours[False,],
        )
        if node not in grouped:
            assert works == 1.0 and fails == 1.0
            works, fails = sums["works"][node], sums["fails"][node]
        for key, value in zip(
            KEYS, (works, fails, up_ok, down_ok, up_bad, down_bad)
        ):
            np.testing.assert_allclose(
                value, sums[key][node], rtol=rel, err_msg=key
            )
    joint = rbd.joint_importance(x)
    for pair in pairs:
        np.testing.assert_allclose(
            joint[pair],
            joint_of(states[pair]),
            rtol=1e-8,
            atol=1e-13,
            err_msg=str(pair),
        )
    p, q = rbd._importance_inputs(x, set(), set())
    shares = rbd._ccf_fv_numerators(p, q, set(), set(), "c", "exact")
    for node in rbd.nodes:
        np.testing.assert_allclose(
            shares[node], fv.get(node, 0.0), rtol=rel, atol=1e-300
        )


def trains(kinds: int, trains: int, model) -> NonRepairableRBD:
    """Redundant trains, each a series of ``kinds`` components, and a
    common-cause group of each kind across the trains."""
    edges, nodes = [], {}
    for j in range(trains):
        before = "s"
        for i in range(kinds):
            name = f"c{i}_{j}"
            edges.append((before, name))
            before = name
            rate = 1e-3 * (1 + i % 3)
            nodes[name] = surv.Exponential.from_params([rate])
        edges.append((before, "t"))
    groups = [
        CCFGroup([f"c{i}_{j}" for j in range(trains)], model(i))
        for i in range(kinds)
    ]
    return NonRepairableRBD(edges, nodes, ccf_groups=groups)


def paired_tree(n: int, model=lambda: BetaFactor(0.1)) -> FaultTree:
    """The issue's tree: an OR of ANDs of pairs, a group for each pair."""
    gates = {f"P{i}": ("and", [f"a{i}", f"b{i}"]) for i in range(n)}
    gates["TOP"] = ("or", list(gates))
    events = {**{f"a{i}": 1e-3 for i in range(n)}}
    events.update({f"b{i}": 1e-3 for i in range(n)})
    groups = [CCFGroup([f"a{i}", f"b{i}"], model()) for i in range(n)]
    return FaultTree(gates, events, ccf_groups=groups)


X = np.array([10.0, 200.0, 900.0])


@pytest.mark.parametrize(
    "kinds, model",
    [
        (3, lambda i: BetaFactor(0.1 + 0.02 * i)),
        (7, lambda i: BetaFactor(0.1 + 0.02 * i)),  # written out
        (7, lambda i: BetaFactor(0.1, basis="rate")),
        (3, lambda i: MGL(0.1, 0.3, shocks="independent")),
        (2, lambda i: MGL(0.1, 0.3)),
        (3, lambda i: MGL(0.1, 0.3)),  # exclusive, as independent causes
        (3, lambda i: MGL(0.1, 0.3, basis="rate")),
        (3, lambda i: MGL(0.1, 0.3) if i % 2 else BetaFactor(0.15)),
        (3, lambda i: MGL(0.1, 0.0)),  # no independent equivalent
        (3, lambda i: AlphaFactor([0.9, 0.07, 0.03])),
        (3, lambda i: AlphaFactor([0.9, 0.07, 0.03], testing="non-staggered")),
        (3, lambda i: AlphaFactor([0.9, 0.07, 0.03], shocks="independent")),
        (3, lambda i: AlphaFactor([0.9, 0.07, 0.03], basis="rate")),
        (
            3,
            lambda i: AlphaFactor([0.9, 0.1, 0.0]),
        ),  # no independent equivalent
    ],
)
def test_trains_agree_with_every_combination(kinds, model):
    agree(trains(kinds, 3, model), X)


def test_groups_that_would_multiply_are_written_out():
    rbd = trains(7, 3, lambda i: BetaFactor(0.1))
    plan = rbd._ccf_plan(rbd.ccf_groups)
    assert sorted(plan.expanded) == list(range(7))
    evaluation = rbd._ccf_evaluation(
        *rbd._importance_inputs(X, set(), set()), set(), set()
    )
    assert sorted(evaluation.causes) == list(range(7))
    # Exclusive causes with no independent equivalent are conditioned on.
    rbd = trains(4, 3, lambda i: MGL(0.1, 0.0))
    assert rbd._ccf_plan(rbd.ccf_groups).expanded
    evaluation = rbd._ccf_evaluation(
        *rbd._importance_inputs(X, set(), set()), set(), set()
    )
    assert not evaluation.causes


def random_diagram(rng: random.Random):
    """A random diagram of layers joined at random (so with bridges), with
    a few groups of alike members."""
    n = rng.randint(4, 10)
    nodes = [f"n{i}" for i in range(n)]
    layers, i = [], 0
    while i < n:
        k = rng.randint(1, 3)
        layers.append(nodes[i : i + k])  # noqa: E203
        i += k
    edges = [("s", v) for v in layers[0]]
    for before, after in zip(layers, layers[1:]):
        for u in before:
            edges += [
                (u, v)
                for v in rng.sample(after, rng.randint(1, min(2, len(after))))
            ]
        for v in after:
            if not any(e[1] == v for e in edges):
                edges.append((rng.choice(before), v))
    edges += [(v, "t") for v in layers[-1]]
    pool = nodes[:]
    rng.shuffle(pool)
    groups = []
    for _ in range(rng.randint(1, 3)):
        size = rng.choice([2, 2, 3])
        if len(pool) < size:
            break
        members = [pool.pop() for _ in range(size)]
        models = [
            BetaFactor(rng.uniform(0.05, 0.4)),
            BetaFactor(rng.uniform(0.05, 0.4), basis="rate"),
        ]
        if size == 3:
            models += [
                MGL(rng.uniform(0.05, 0.3), rng.uniform(0.1, 0.5)),
                MGL(0.2, 0.3, shocks="independent"),
                AlphaFactor([0.9, 0.07, 0.03], testing="non-staggered"),
            ]
        groups.append(CCFGroup(members, rng.choice(models)))
    rate = {v: rng.uniform(0.002, 0.03) for v in nodes}
    for group in groups:
        for member in group.members[1:]:
            rate[member] = rate[group.members[0]]
    models = {v: surv.Exponential.from_params([rate[v]]) for v in nodes}
    return NonRepairableRBD(edges, models, ccf_groups=groups)


@pytest.mark.parametrize("seed", range(25))
def test_random_diagrams_agree_with_every_combination(seed):
    agree(random_diagram(random.Random(seed)), X)


def test_written_out_groups_in_random_diagrams(monkeypatch):
    # Every group written out, however few outcomes its module's groups
    # have between them.
    monkeypatch.setattr(_ccf_modules, "COMBINATIONS", 1)
    for seed in range(15):
        rbd = random_diagram(random.Random(100 + seed))
        agree(rbd, X)


def test_groups_in_separate_modules_cost_a_sum_each():
    # The issue's tree: a group in each of 40 modules, which every
    # combination of outcomes would take 2 ** 40 evaluations for. Each
    # beta-factor cause written as a repeated event gives the same.
    tree = paired_tree(40)
    start = time.perf_counter()
    value = tree.top_event_probability()
    assert time.perf_counter() - start < 5.0
    gates, events = {}, {}
    for i in range(40):
        gates[f"A{i}"] = ("or", [f"a{i}", f"c{i}"])
        gates[f"B{i}"] = ("or", [f"b{i}", f"c{i}"])
        gates[f"P{i}"] = ("and", [f"A{i}", f"B{i}"])
        events[f"a{i}"] = events[f"b{i}"] = 0.9e-3
        events[f"c{i}"] = 1e-4
    gates["TOP"] = ("or", [f"P{i}" for i in range(40)])
    assert value == pytest.approx(
        FaultTree(gates, events).top_event_probability(), rel=1e-13
    )


@pytest.mark.parametrize(
    "model",
    [
        lambda i: BetaFactor(0.1),
        lambda i: MGL(0.1, 0.3),
        lambda i: MGL(0.1, 0.3, shocks="independent"),
    ],
)
def test_many_groups_across_trains_are_quick(model):
    rbd = trains(30, 3, model)
    start = time.perf_counter()
    rbd.ff(X)
    rbd.birnbaum_importance(X)
    assert time.perf_counter() - start < 30.0
    # A tree of the same trains agrees with the diagram.
    small = trains(9, 3, model)
    tree = FaultTree.from_rbd(small)
    assert tree.top_event_probability(200.0) == pytest.approx(
        float(np.ravel(small.ff(200.0))[0]), rel=1e-12
    )
    ours = tree.birnbaum_importance(200.0)
    theirs = small.birnbaum_importance(200.0)
    for event in tree.events:
        assert ours[event] == pytest.approx(
            float(np.ravel(theirs[event])[0]), rel=1e-9
        )
    joint, pairs = tree.joint_importance(200.0), small.joint_importance(200.0)
    assert set(joint) == set(pairs)
    for pair in chosen_pairs(small):
        assert joint[pair] == pytest.approx(
            float(np.ravel(pairs[pair])[0]), rel=1e-9
        )


@pytest.mark.parametrize("Q", [1e-6, 1e-3, 0.05, 0.3, 0.9])
@pytest.mark.parametrize(
    "model, members",
    [
        (MGL(0.1, 0.3), ("a", "b", "c")),
        (MGL(0.2, 0.5, 0.4), ("a", "b", "c", "d")),
        (AlphaFactor([0.9, 0.07, 0.03]), ("a", "b", "c")),
        (
            AlphaFactor([0.8, 0.1, 0.06, 0.04], testing="non-staggered"),
            ("a", "b", "c", "d"),
        ),
    ],
)
def test_exclusive_causes_as_independent_ones(model, members, Q):
    # Independent causes failing the same sets of members as often as the
    # exclusive shocks do.
    Q = np.array([Q])
    q_independent, r_independent, shocks = model._split(members, Q, 1 - Q)
    split = _as_independent(
        members,
        q_independent,
        r_independent,
        shocks,
        model._cause_sets(members),
    )
    assert split is not None
    _, _, fired = split
    unions = {frozenset(): 1.0}
    for struck, fires, holds in fired:
        after: dict = {}
        for union, p in unions.items():
            after[union] = after.get(union, 0.0) + p * holds
            after[union | struck] = after.get(union | struck, 0.0) + p * fires
        unions = after
    expected = {frozenset(s): p for s, p in shocks}
    expected[frozenset()] = 1.0 - sum(expected.values())
    for subset in itertools.chain.from_iterable(
        itertools.combinations(members, k) for k in range(len(members) + 1)
    ):
        key = frozenset(subset)
        assert unions.get(key, 0.0) == pytest.approx(
            expected.get(key, 0.0), rel=1e-9, abs=1e-300
        )


def test_exclusive_causes_with_no_independent_equivalent():
    # Without the triple shock, two pair causes would strike all three
    # together, which the exclusive shocks never do.
    model = MGL(0.1, 0.0)
    members = ("a", "b", "c")
    Q = np.array([0.01])
    assert model._fired(members, Q, 1 - Q) is None
