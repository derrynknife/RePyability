"""Common-cause groups in a RepairableRBD worked out module by module
(#218).

Given the combination of its members up or down, a group's members are
independent of every other node, so each group is conditioned on only
within the smallest module holding its members: groups in separate
modules cost a sum, where splitting each time by every combination of
every group's states doubled the memory per group (ten groups killed the
process). The reference here is that split (``_with_ccf_groups``, which
the capacity distribution and the allocations still use), under which the
nodes are independent: the values worked out module by module must be
its, to rounding.
"""

import random
import time

import numpy as np
import pytest
import surpyval as surv

from repyability import (
    MGL,
    BetaFactor,
    CCFGroup,
    PerfectReliability,
    RepairableRBD,
)
from repyability.rbd import _ccf_chain, _ccf_groups, _ccf_modules, _long_run
from repyability.rbd.rbd import RBD

E = surv.Exponential.from_params


def hidden(rate, interval, **more):
    return {
        "reliability": E([rate]),
        "repairability": "instant",
        "inspection": {"interval": interval, **more},
    }


def revealed(rate, repair):
    return {"reliability": E([rate]), "repairability": E([repair])}


def split(rbd, working=(), broken=()):
    """The long-run points split by every combination of every group's
    states, as before #218."""
    times, weights = _long_run._long_run_grid(rbd)
    p = rbd._probabilities_with_overrides(
        _long_run._availabilities_at(rbd, times), working, broken
    )
    q = rbd._failures_with_overrides(
        _long_run._unavailabilities_at(rbd, times), working, broken
    )
    p, q, weights, index = _ccf_groups._with_ccf_groups(
        rbd, times, p, q, weights
    )
    return p, q, weights, index


def members_measure(rbd, measure, p, q, weights, index, kind="failure"):
    """A member's measure over the split points: conditioned on its state
    at each time, as before #218."""
    pairs_p, pairs_q, _ = rbd._node_pairs(p, q)
    up, down = rbd._system_probabilities(pairs_p, pairs_q)
    times = int(index.max()) + 1

    def per_time(values):
        return np.bincount(index, weights=weights * values, minlength=times)

    R, Q = weights @ up, weights @ down
    R_t, Q_t = per_time(up), per_time(down)
    out = {}
    for group in rbd.ccf_groups:
        for member in group.members:
            on, off = pairs_p[member], pairs_q[member]
            works, fails = per_time(on), per_time(off)
            r1, q1 = per_time(on * up) / works, per_time(on * down) / works
            r0, q0 = per_time(off * up) / fails, per_time(off * down) / fails
            change = np.where(Q_t <= R_t, q0 - q1, r1 - r0)
            weight = works + fails
            out[member] = {
                "birnbaum": weight @ change,
                "improvement": fails @ change,
                "raw": (weight @ q0) / Q,
                "rrw": Q / (weight @ q1),
                "criticality": (
                    (fails @ change) / Q
                    if kind == "failure"
                    else (works @ change) / R
                ),
            }[measure]
    return out


BASE = {
    "birnbaum": RBD._birnbaum_importance,
    "improvement": RBD._improvement_potential,
    "raw": RBD._risk_achievement_worth,
    "rrw": RBD._risk_reduction_worth,
}

PUBLIC = {
    "birnbaum": "birnbaum_importance",
    "improvement": "improvement_potential",
    "raw": "risk_achievement_worth",
    "rrw": "risk_reduction_worth",
}


def agree(rbd, working=(), broken=(), rel=1e-11):
    with np.errstate(divide="ignore", invalid="ignore"):
        _agree(rbd, set(working), set(broken), rel)


def _agree(rbd, working, broken, rel):
    p, q, weights, index = split(rbd, working, broken)
    U = float(weights @ RBD._system_unreliability(rbd, p, q))
    assert rbd.mean_unavailability(working, broken) == pytest.approx(
        U, rel=rel
    )
    assert rbd.mean_availability(working, broken) == pytest.approx(
        float(weights @ RBD.system_probability(rbd, p)), rel=rel, abs=1e-15
    )
    for name, base in BASE.items():
        want = {k: float(v[0]) for k, v in base(rbd, p, weights, q).items()}
        want.update(
            {
                k: float(v)
                for k, v in members_measure(
                    rbd, name, p, q, weights, index
                ).items()
            }
        )
        got = getattr(rbd, PUBLIC[name])(working, broken)
        for node in rbd.nodes:
            assert got[node] == pytest.approx(
                want[node], rel=1e-9, abs=1e-13 * max(1.0, abs(want[node]))
            ), (name, node)
    for kind in ("failure", "success"):
        want = {
            k: float(v[0])
            for k, v in RBD._criticality_importance(
                rbd, p, kind, weights, q
            ).items()
        }
        want.update(
            {
                k: float(v)
                for k, v in members_measure(
                    rbd, "criticality", p, q, weights, index, kind
                ).items()
            }
        )
        got = rbd.criticality_importance(working, broken, kind=kind)
        for node in rbd.nodes:
            assert got[node] == pytest.approx(
                want[node], rel=1e-9, abs=1e-13
            ), ("criticality", kind, node)
    for fv_type in ("c", "p"):
        for method in ("exact", "rare_event"):
            want = RBD._fussell_vesely(rbd, p, fv_type, method, weights, q)
            got = rbd.fussell_vesely(fv_type, working, broken, method)
            for node in rbd.nodes:
                assert got[node] == pytest.approx(
                    float(want[node][0]), rel=1e-9, abs=1e-13
                ), ("fv", fv_type, method, node)


def pairs(n, **options):
    """The issue's system: pairs of tested units in series, a group of
    each pair."""
    edges, nodes, prev = [], {}, "s"
    for i in range(n):
        j = f"j{i}"
        edges += [
            (prev, f"a{i}"),
            (prev, f"b{i}"),
            (f"a{i}", j),
            (f"b{i}", j),
        ]
        nodes[f"a{i}"] = nodes[f"b{i}"] = hidden(2e-6 * (1 + i % 3), 8760.0)
        nodes[j] = PerfectReliability
        prev = j
    edges.append((prev, "t"))
    groups = [CCFGroup([f"a{i}", f"b{i}"], BetaFactor(0.1)) for i in range(n)]
    return RepairableRBD(edges, nodes, ccf_groups=groups, **options)


def trains(kinds, count, life=lambda k: revealed(1e-3 * (1 + k), 0.05)):
    """Redundant trains of ``kinds`` components in series, a group of each
    kind across the trains: the groups meet in one module."""
    edges, nodes = [], {}
    for t in range(count):
        before = "s"
        for k in range(kinds):
            name = f"k{k}t{t}"
            edges.append((before, name))
            before = name
            nodes[name] = life(k)
        edges.append((before, "t"))
    groups = [
        CCFGroup(
            [f"k{k}t{t}" for t in range(count)],
            BetaFactor(0.1 + 0.05 * k) if k % 2 else MGL(0.1, 0.3),
        )
        for k in range(kinds)
    ]
    return RepairableRBD(edges, nodes, ccf_groups=groups)


def test_the_issues_pairs_agree_with_every_combination():
    agree(pairs(4))


def test_the_issues_pairs_cost_a_sum_each():
    # Ten groups were killed for memory; forty take well under a second
    # each, and stay as the pairs alone give them.
    rbd = pairs(40)
    start = time.perf_counter()
    U = rbd.mean_unavailability()
    importance = rbd.birnbaum_importance()
    rbd.mean_time_between_failures()
    assert time.perf_counter() - start < 30.0
    small = pairs(3)
    assert importance["a0"] < 1.0
    assert U > small.mean_unavailability()


@pytest.mark.parametrize("kinds", [2, 3])
def test_groups_meeting_in_one_module_agree(kinds):
    agree(trains(kinds, 3))


def test_tested_trains_agree():
    agree(
        trains(
            2,
            3,
            life=lambda k: hidden(1e-4 * (1 + k), 500.0, offset=100.0 * k),
        )
    )


def nested():
    """A pair group inside one train, and a group across the trains that
    reaches into it."""
    return RepairableRBD(
        [
            ("s", "p1"),
            ("s", "p2"),
            ("p1", "j"),
            ("p2", "j"),
            ("j", "v1"),
            ("v1", "t"),
            ("s", "q"),
            ("q", "v2"),
            ("v2", "t"),
        ],
        {
            "p1": revealed(1e-3, 0.1),
            "p2": revealed(1e-3, 0.1),
            "j": PerfectReliability,
            "q": revealed(2e-3, 0.1),
            "v1": revealed(5e-4, 0.2),
            "v2": revealed(5e-4, 0.2),
        },
        ccf_groups=[
            CCFGroup(["p1", "p2"], BetaFactor(0.1)),
            CCFGroup(["v1", "v2"], BetaFactor(0.2)),
        ],
    )


def test_nested_modules_agree():
    agree(nested())
    agree(nested(), working=["q"])
    agree(nested(), broken=["q"])


def bridge():
    edges = [
        ("s", "a"),
        ("s", "b"),
        ("a", "c"),
        ("b", "c"),
        ("a", "d"),
        ("b", "e"),
        ("c", "d"),
        ("c", "e"),
        ("d", "t"),
        ("e", "t"),
    ]
    return RepairableRBD(
        edges,
        {n: revealed(1e-3 * (1 + "abcde".index(n) % 2), 0.1) for n in "abcde"},
        ccf_groups=[
            CCFGroup(["a", "e"], BetaFactor(0.15)),
            CCFGroup(["b", "d"], BetaFactor(0.1)),
        ],
    )


def test_groups_only_the_core_joins_agree():
    agree(bridge())
    agree(bridge(), working=["c"])


def random_diagram(rng: random.Random):
    """Layers joined at random (so with bridges), of revealed or tested
    units, with a few groups of alike members."""
    n = rng.randint(4, 9)
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
    tested = rng.random() < 0.5
    pool = nodes[:]
    rng.shuffle(pool)
    groups = []
    for _ in range(rng.randint(1, 3)):
        size = rng.choice([2, 2, 3])
        if len(pool) < size:
            break
        members = [pool.pop() for _ in range(size)]
        models = [BetaFactor(rng.uniform(0.05, 0.4))]
        if size == 3:
            models.append(MGL(rng.uniform(0.05, 0.3), rng.uniform(0.1, 0.5)))
        groups.append(CCFGroup(members, rng.choice(models)))
    rate = {v: rng.uniform(1e-4, 3e-3) for v in nodes}
    for group in groups:
        for member in group.members[1:]:
            rate[member] = rate[group.members[0]]

    def unit(v):
        if tested:
            return hidden(
                rate[v], 400.0, offset=float(rng.randint(0, 3) * 100)
            )
        return revealed(rate[v], 0.05)

    models = {v: unit(v) for v in nodes}
    for group in groups:
        for member in group.members[1:]:
            models[member] = dict(models[group.members[0]])
    return RepairableRBD(edges, models, ccf_groups=groups)


@pytest.mark.parametrize("seed", range(12))
def test_random_diagrams_agree(seed):
    agree(random_diagram(random.Random(seed)))


def test_over_time_agrees_with_every_combination():
    # The groups' system over time, against the times split by every
    # combination (as GroupsSystem evaluated it before #218).
    for rbd in (nested(), trains(2, 3), bridge()):
        curve = _ccf_groups._groups_curve(rbd, 2000.0)
        x = np.array([1.0, 30.0, 400.0, 1500.0])
        values = {n: c.at(x) for n, c in curve.curves.items()}
        system = curve.system
        importance, up, down, rate, own = system.evaluate(values, x)
        probabilities, weights, index, _ = system._split(values, x)

        def total(v):
            v = np.broadcast_to(np.asarray(v, dtype=float), weights.shape)
            return np.bincount(index, weights * v, len(x))

        base, works, fails, _, _ = rbd._importances(probabilities)
        np.testing.assert_allclose(down, total(fails), rtol=1e-12)
        np.testing.assert_allclose(up, total(works), rtol=1e-12)
        for node in values:
            np.testing.assert_allclose(
                importance[node], total(base[node]), rtol=1e-9, atol=1e-15
            )
        for member in system.served:
            np.testing.assert_allclose(
                own[member], total(probabilities[member]), rtol=1e-12
            )
        hits = 0.0
        unreliable = rbd._system_unreliability(probabilities)
        for group, causes in zip(system.groups, system.causes):
            for struck, cause in causes:
                hit = dict(probabilities)
                for position in struck:
                    hit[group.members[position]] = np.zeros_like(weights)
                rise = rbd._system_unreliability(hit) - unreliable
                hits = hits + cause * total(rise)
        np.testing.assert_allclose(rate, hits, rtol=1e-9, atol=1e-18)
        for number in range(len(system.groups)):
            given = system.given(values, x, number)
            count = len(system.groups[number].down)
            chances: list = [None] * len(system.groups)
            chances[number] = np.ones((len(x), count))
            p, w, at, combos = system._split(values, x, chances)
            want = np.bincount(
                at * count + combos[number],
                w * rbd._system_unreliability(p),
                len(x) * count,
            ).reshape(len(x), count)
            np.testing.assert_allclose(given, want, rtol=1e-12)


def test_an_owner_with_too_many_combinations_refuses_first(monkeypatch):
    # Kinds of components across trains meet in one module: their groups'
    # combinations multiply there, and past the limit it refuses before
    # working anything out, saying what to do.
    monkeypatch.setattr(_ccf_modules, "POINTS", 1000)
    rbd = trains(4, 3)
    with pytest.raises(NotImplementedError, match="meet in one part"):
        rbd.mean_unavailability()
    with pytest.raises(
        NotImplementedError, match=r"availability\(\) or cost\(\)"
    ):
        rbd.birnbaum_importance()
    # In separate modules the same groups each cost their own.
    pairs(12).mean_unavailability()


def test_the_split_refuses_before_building_it(monkeypatch):
    # The capacity distribution still takes every combination at once: it
    # refuses where they would be too many, rather than run out of memory.
    monkeypatch.setattr(_ccf_chain, "SPLIT_VALUES", 10_000)
    rbd = pairs(6, capacity={f"{x}{i}": 1.0 for x in "ab" for i in range(6)})
    with pytest.raises(
        NotImplementedError,
        match="The capacity distribution takes every combination",
    ):
        rbd.capacity_distribution()
    # The long-run values do not split, so do not refuse.
    rbd.mean_unavailability()


@pytest.mark.parametrize("basis", ["probability", "rate"])
def test_groups_with_limited_crews_refuse_rather_than_drop_the_groups(basis):
    # The crews' chain does not take common causes in: the long-run values
    # refused only over time, and gave the system without its groups in
    # the long run; then each component's own, which a shared failure
    # changes too, as one member waits for the crew (#251).
    units = {n: revealed(1e-3, 0.1) for n in "abc"}
    edges = [("s", "a"), ("s", "b"), ("a", "c"), ("b", "c"), ("c", "t")]
    rbd = RepairableRBD(
        edges,
        units,
        ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.1, basis=basis))],
        repair_crews=1,
        downtime_cost_rate=5.0,
    )
    message = "crews' Markov chain does not take common causes in"
    for method in (
        "mean_availability",
        "mean_unavailability",
        "node_availability",
        "system_failure_frequency",
        "mean_time_between_failures",
        "mean_up_time",
        "birnbaum_importance",
        "risk_achievement_worth",
        "criticality_importance",
        "fussell_vesely",
        "barlow_proschan_importance",
        "parameter_sensitivity",
        "expected_cost_rate",
        "capacity_distribution",
        "point_availability",
        "mission_availability",
    ):
        args = (100.0,) if method.startswith(("point", "mission")) else ()
        with pytest.raises(NotImplementedError, match=message):
            getattr(rbd, method)(*args)
        assert rbd.analysis_routes()[method].route == "refused", method
    # The simulations take both.
    assert rbd.analysis_routes()["availability"].route == "simulated"
    rbd.availability(100.0, mc_samples=20, seed=1)
