"""Phased missions (#100): phases with their own durations and diagrams over
the same non-repairable components, exact (the Esary-Ziehms segments) and
simulated."""

import itertools

import numpy as np
import pytest
import surpyval as surv

from repyability import NonRepairableRBD, PhasedMission
from repyability.rbd import phased_mission

E, W = surv.Exponential.from_params, surv.Weibull.from_params
F = surv.FixedEventProbability.from_params

UNITS = {
    "a": W([100.0, 2.0]),
    "b": W([80.0, 1.5]),
    "c": E([0.01]),
    "d": W([150.0, 3.0]),
    "e": E([0.004]),
}


def rbd(edges, k=None, **models):
    nodes = {n for edge in edges for n in edge} - {"s", "t"}
    chosen = {n: models.get(n, UNITS.get(n)) for n in nodes}
    return NonRepairableRBD(edges, chosen, k=k)


SERIES = rbd([("s", "a"), ("a", "b"), ("b", "t")])
PARALLEL = rbd([("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")])
BRIDGE = rbd(
    [
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
)
VOTE = rbd(
    [
        ("s", "a"),
        ("s", "b"),
        ("s", "c"),
        ("a", "v"),
        ("b", "v"),
        ("c", "v"),
        ("v", "t"),
    ],
    k={"v": 2},
    v=F(0.0),
)
DIAGRAMS = {
    "series": SERIES,
    "parallel": PARALLEL,
    "bridge": BRIDGE,
    "vote": VOTE,
}


@pytest.mark.parametrize("name", DIAGRAMS)
def test_one_phase_is_the_diagram_at_its_duration(name):
    diagram = DIAGRAMS[name]
    mission = PhasedMission([("only", 40.0, diagram)])
    assert mission.reliability() == pytest.approx(
        float(diagram.sf(40.0)), rel=1e-12
    )
    assert mission.unreliability() == pytest.approx(
        float(diagram.ff(40.0)), rel=1e-10
    )
    assert mission.phase_failure_probabilities() == {
        "only": pytest.approx(float(diagram.ff(40.0)), rel=1e-10)
    }


@pytest.mark.parametrize("name", DIAGRAMS)
def test_phases_with_the_same_diagram_are_one_phase(name):
    diagram = DIAGRAMS[name]
    mission = PhasedMission(
        [
            ("one", 10.0, diagram),
            ("two", 25.0, diagram),
            ("three", 5.0, diagram),
        ]
    )
    assert mission.duration == 40.0
    assert mission.reliability() == pytest.approx(
        float(diagram.sf(40.0)), rel=1e-12
    )
    # Failing in a phase: working at its start and not at its end.
    failing = mission.phase_failure_probabilities()
    sf = diagram.sf(np.array([0.0, 10.0, 35.0, 40.0]))
    np.testing.assert_allclose(
        list(failing.values()), sf[:-1] - sf[1:], rtol=1e-9, atol=1e-15
    )


def enumerated(mission):
    """The mission's reliability and phase failure probabilities from every
    combination of the phases each component fails in."""
    nodes = list(mission.components)
    ends = [phase.end for phase in mission.phases]
    count = len(ends)
    chances = {}
    for node, model in mission.components.items():
        sf = np.concatenate(([1.0], np.ravel(model.sf(np.array(ends)))))
        chances[node] = np.append(sf[:-1] - sf[1:], sf[-1])
    success, failing = 0.0, np.zeros(count)
    for phases in itertools.product(range(count + 1), repeat=len(nodes)):
        chance = np.prod([chances[n][f] for n, f in zip(nodes, phases)])
        # Component n works at the end of phase j if it fails in a later
        # one (phases numbered from 0; ``count`` for never).
        status = {n: f for n, f in zip(nodes, phases)}
        for j, phase in enumerate(mission.phases):
            works = {n: status.get(n, count) > j for n in phase.rbd.G.nodes}
            works.update(
                {n: status[o] > j for n, o in phase.rbd.repeated.items()}
            )
            works[phase.rbd.input_node] = works[phase.rbd.output_node] = True
            if not phase.rbd.is_system_working(works, "p"):
                failing[j] += chance
                break
        else:
            success += chance
    return success, failing


def test_small_missions_by_enumerating_the_phases_components_fail_in():
    # Take-off needs both engines and the gear; cruise either engine and
    # the navigation unit; landing the gear, and the engines two out of
    # three with an auxiliary.
    engine, gear, nav, aux = (
        W([200.0, 1.8]),
        W([500.0, 2.5]),
        E([0.002]),
        E([0.01]),
    )
    models = {"e1": engine, "e2": engine, "g": gear, "n": nav, "x": aux}
    takeoff = NonRepairableRBD(
        [("s", "e1"), ("e1", "e2"), ("e2", "g"), ("g", "t")],
        {k: models[k] for k in ("e1", "e2", "g")},
    )
    cruise = NonRepairableRBD(
        [("s", "e1"), ("s", "e2"), ("e1", "n"), ("e2", "n"), ("n", "t")],
        {k: models[k] for k in ("e1", "e2", "n")},
    )
    landing = NonRepairableRBD(
        [
            ("s", "g"),
            ("g", "e1"),
            ("g", "e2"),
            ("g", "x"),
            ("e1", "v"),
            ("e2", "v"),
            ("x", "v"),
            ("v", "t"),
        ],
        {**{k: models[k] for k in ("g", "e1", "e2", "x")}, "v": F(0.0)},
        k={"v": 2},
    )
    mission = PhasedMission(
        [
            ("take-off", 0.5, takeoff),
            ("cruise", 20.0, cruise),
            ("landing", 0.5, landing),
        ]
    )
    success, failing = enumerated(mission)
    assert mission.reliability() == pytest.approx(success, rel=1e-12)
    assert list(mission.phase_failure_probabilities().values()) == (
        pytest.approx(list(failing), rel=1e-9)
    )
    # Not the product of the phases' reliabilities.
    alone = np.prod(
        [
            float(takeoff.sf(0.5)),
            float(cruise.sf(20.5)) / float(cruise.sf(0.5)),
            float(landing.sf(21.0)) / float(landing.sf(20.5)),
        ]
    )
    assert abs(alone - success) > 1e-4
    # And the simulation agrees.
    interval = mission.reliability_interval(200_000, seed=3)
    assert interval.lower < success < interval.upper
    simulated = mission.phase_failure_probabilities(
        method="simulate", mc_samples=200_000, seed=3
    )
    for estimate, exact in zip(simulated.values(), failing):
        assert abs(estimate - exact) < 4.0 * np.sqrt(exact / 200_000) + 1e-4


def test_components_of_every_kind():
    # A repeated node, a nested RBD as a component, a fixed probability
    # (which fails at the start, if at all), and a phase of no duration.
    inner = NonRepairableRBD(
        [("s", "p"), ("s", "q"), ("p", "t"), ("q", "t")],
        {"p": E([0.02]), "q": E([0.02])},
    )
    first = NonRepairableRBD(
        [
            ("s", "ps"),
            ("ps", "a"),
            ("a", "t"),
            ("s", "ps2"),
            ("ps2", "sub"),
            ("sub", "t"),
        ],
        {"ps": W([300.0, 2.0]), "ps2": "ps", "a": UNITS["a"], "sub": inner},
    )
    check = NonRepairableRBD(
        [("s", "sw"), ("sw", "a"), ("a", "t")],
        {"sw": F(0.05), "a": UNITS["a"]},
    )
    second = NonRepairableRBD(
        [("s", "sub"), ("sub", "ps"), ("ps", "t")],
        {"sub": inner, "ps": W([300.0, 2.0])},
    )
    mission = PhasedMission(
        [
            ("first", 10.0, first),
            ("check", 0.0, check),
            ("second", 30.0, second),
        ]
    )
    assert [p.end for p in mission.phases] == [10.0, 10.0, 40.0]
    success, failing = enumerated(mission)
    assert mission.reliability() == pytest.approx(success, rel=1e-12)
    assert list(mission.phase_failure_probabilities().values()) == (
        pytest.approx(list(failing), rel=1e-9, abs=1e-15)
    )
    interval = mission.reliability_interval(100_000, seed=5)
    assert interval.lower < success < interval.upper


def test_a_reliable_mission_keeps_its_unreliability_precise():
    tiny = E([1e-9])
    pair = NonRepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        {"a": tiny, "b": tiny},
    )
    one = NonRepairableRBD([("s", "a"), ("a", "t")], {"a": tiny})
    mission = PhasedMission([("both", 10.0, pair), ("one", 10.0, one)])
    # It succeeds if "a" lasts 20 hours: then the pair works in the first
    # phase too. So it fails with 1 - exp(-2e-8), which the complement of
    # the reliability would round to a few digits.
    expected = -np.expm1(-2e-8)
    assert mission.unreliability() == pytest.approx(expected, rel=1e-9)
    assert mission.reliability() == pytest.approx(1.0 - expected, abs=1e-16)
    failing = mission.phase_failure_probabilities()
    assert failing["both"] == pytest.approx(-np.expm1(-1e-8) ** 2, rel=1e-6)


def test_the_simulation_is_seeded_and_runs_to_a_tolerance():
    mission = PhasedMission([("up", 30.0, PARALLEL), ("on", 30.0, BRIDGE)])
    a = mission.reliability(method="simulate", mc_samples=5_000, seed=11)
    b = mission.reliability(method="simulate", mc_samples=5_000, seed=11)
    assert a == b
    exact = mission.reliability()
    paired = mission.reliability_interval(20_000, seed=2, antithetic=True)
    assert paired.lower < exact < paired.upper
    run = mission.reliability_interval(
        2_000, seed=4, tolerance=0.004, max_samples=200_000
    )
    assert run.upper - run.estimate <= 0.004
    assert run.n_samples % 2_000 == 0 and run.n_samples > 2_000
    assert run.lower < exact < run.upper
    with pytest.warns(RuntimeWarning, match="did not converge"):
        mission.reliability_interval(
            1_000, seed=4, tolerance=1e-6, max_samples=3_000
        )


def test_too_large_a_decomposition_refuses(monkeypatch):
    monkeypatch.setattr(phased_mission, "MAX_NODES", 5)
    mission = PhasedMission([("up", 30.0, BRIDGE), ("on", 30.0, VOTE)])
    with pytest.raises(NotImplementedError, match="method='simulate'"):
        mission.reliability()
    assert 0.0 < mission.reliability(method="simulate", seed=1) < 1.0
    # The path sets' decomposition has its own limit.
    monkeypatch.setattr(phased_mission, "METHOD", "paths")
    monkeypatch.setattr(phased_mission, "MAX_STATES", 3)
    listed = PhasedMission([("up", 30.0, BRIDGE), ("on", 30.0, VOTE)])
    with pytest.raises(NotImplementedError, match="method='simulate'"):
        listed.reliability()
    monkeypatch.setattr(phased_mission, "METHOD", "neither")
    with pytest.raises(ValueError, match="METHOD"):
        PhasedMission([("up", 30.0, BRIDGE)]).reliability()


# -- the decision diagram (#142) -------------------------------------------


@pytest.mark.parametrize(
    "phases",
    [
        [("up", 30.0, BRIDGE), ("on", 30.0, VOTE)],
        [("a", 10.0, SERIES), ("b", 20.0, BRIDGE), ("c", 5.0, PARALLEL)],
        [("one", 15.0, VOTE), ("two", 0.0, SERIES), ("three", 40.0, BRIDGE)],
    ],
    ids=["bridge then vote", "series, bridge, parallel", "with an instant"],
)
def test_the_diagram_is_the_path_sets_decomposition(phases, monkeypatch):
    diagram = PhasedMission(phases)
    monkeypatch.setattr(phased_mission, "METHOD", "paths")
    paths = PhasedMission(phases)
    assert diagram.reliability() == pytest.approx(
        paths.reliability(), rel=1e-12
    )
    assert diagram.unreliability() == pytest.approx(
        paths.unreliability(), rel=1e-11
    )
    got = diagram.phase_failure_probabilities()
    for name, value in paths.phase_failure_probabilities().items():
        assert got[name] == pytest.approx(value, rel=1e-10, abs=1e-16)


def bridges(count, prefix):
    """``count`` bridges, each feeding the next: meshed, so its path sets
    multiply (four to the ``count``)."""
    edges, models = [], {}
    before = ["s"]
    for i in range(count):
        a, b, c, d, e = (f"{prefix}{x}{i}" for x in "abcde")
        for node in before:
            edges += [(node, a), (node, b)]
        edges += [(a, c), (b, c), (a, d), (c, d), (b, e), (c, e)]
        before = [d, e]
        for j, node in enumerate((a, b, c, d, e)):
            models[node] = W([200.0 + 25.0 * j, 1.5])
    edges += [(node, "t") for node in before]
    return NonRepairableRBD(edges, models)


def test_a_meshed_mission_beyond_the_path_sets():
    # Eight bridges in a chain have 65,536 path sets; with a second phase
    # over the same components, their decomposition over the segments
    # takes minutes, where the decision diagram takes a fraction of a
    # second.
    chain = bridges(8, "x")
    tail = NonRepairableRBD(
        [("s", "xd7"), ("s", "xe7"), ("xd7", "t"), ("xe7", "t")],
        {"xd7": chain.reliabilities["xd7"], "xe7": chain.reliabilities["xe7"]},
    )
    mission = PhasedMission([("climb", 20.0, chain), ("hold", 30.0, tail)])
    success = mission.reliability()
    assert success + mission.unreliability() == pytest.approx(1.0, abs=1e-15)
    interval = mission.reliability_interval(50_000, seed=11)
    assert interval.lower < success < interval.upper
    # One phase is the diagram's own reliability at its end.
    alone = PhasedMission([("climb", 20.0, chain)])
    assert alone.reliability() == pytest.approx(
        float(chain.sf(20.0)), rel=1e-12
    )


@pytest.mark.parametrize(
    "phases, message",
    [
        ([], "at least one phase"),
        ([("a", 1.0)], r"\(name, duration, rbd\)"),
        ([("a", 1.0, SERIES), ("a", 2.0, SERIES)], "Two phases"),
        ([("a", -1.0, SERIES)], "finite number"),
        ([("a", np.inf, SERIES)], "finite number"),
        ([("a", True, SERIES)], "finite number"),
        ([("a", 1.0, "diagram")], "NonRepairableRBD"),
        (
            [
                ("a", 1.0, SERIES),
                ("b", 1.0, rbd([("s", "a"), ("a", "t")], a=E([0.5]))),
            ],
            "different model",
        ),
    ],
)
def test_the_phases_are_checked(phases, message):
    with pytest.raises(ValueError, match=message):
        PhasedMission(phases)


def test_the_methods_are_checked():
    mission = PhasedMission([("up", 1.0, SERIES)])
    with pytest.raises(ValueError, match="'exact' or 'simulate'"):
        mission.reliability(method="guess")
    with pytest.raises(ValueError, match="only to method='simulate'"):
        mission.reliability(mc_samples=10)
    with pytest.raises(ValueError):
        mission.reliability_interval(0)
    assert repr(mission) == "PhasedMission('up' (1))"


def test_series_and_parallel_phases_in_closed_form():
    # (#101) Both units needed for 10 hours, then either for 40: both last
    # 10, and then not both fail in the next 40. The other way round, the
    # mission succeeds exactly when both last 50.
    a, b = UNITS["a"], UNITS["b"]
    ra, rb = (np.ravel(m.sf(np.array([10.0, 50.0]))) for m in (a, b))
    both = PhasedMission(
        [("series", 10.0, SERIES), ("parallel", 40.0, PARALLEL)]
    )
    expected = (
        ra[0] * rb[0] * (1.0 - (1.0 - ra[1] / ra[0]) * (1.0 - rb[1] / rb[0]))
    )
    assert both.reliability() == pytest.approx(expected, rel=1e-12)
    either = PhasedMission(
        [("parallel", 10.0, PARALLEL), ("series", 40.0, SERIES)]
    )
    assert either.reliability() == pytest.approx(ra[1] * rb[1], rel=1e-12)
    assert either.phase_failure_probabilities()["parallel"] == pytest.approx(
        (1.0 - ra[0]) * (1.0 - rb[0]), rel=1e-10
    )
