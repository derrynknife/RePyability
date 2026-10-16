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


# -- importance and cut sets (#316) ----------------------------------------


MEASURES = (
    "birnbaum_importance",
    "improvement_potential",
    "risk_achievement_worth",
    "risk_reduction_worth",
    "criticality_importance",
    "fussell_vesely",
)


def fails(mission, status):
    """Whether the mission fails with its components failing in the phases
    ``status`` gives (numbered from 0; ``len(phases)`` for never)."""
    count = len(mission.phases)
    for j, phase in enumerate(mission.phases):
        works = {n: status.get(n, count) > j for n in phase.rbd.G.nodes}
        works.update({n: status[o] > j for n, o in phase.rbd.repeated.items()})
        works[phase.rbd.input_node] = works[phase.rbd.output_node] = True
        if not phase.rbd.is_system_working(works, "p"):
            return True
    return False


def by_brute_force(mission):
    """Every measure, and the minimal cut sets, by conditioning on every
    combination of the phases the components fail in: each assignment's
    probability from the components' models at the phases' ends, a
    component held by keeping the assignments with it so (its own
    probability left out)."""
    nodes = list(mission.components)
    ends = np.array([phase.end for phase in mission.phases])
    count = len(ends)
    chances, failed = {}, {}
    for node, model in mission.components.items():
        sf = np.concatenate(([1.0], np.ravel(model.sf(ends))))
        chances[node] = np.append(sf[:-1] - sf[1:], sf[-1])
        failed[node] = np.ravel(model.ff(ends))
    assignments = [
        dict(zip(nodes, phases))
        for phases in itertools.product(range(count + 1), repeat=len(nodes))
    ]
    outcome = {
        tuple(status.values()): fails(mission, status)
        for status in assignments
    }

    def probabilities(held=None):
        success = failure = 0.0
        for status in assignments:
            if held is not None and status[held[0]] != held[1]:
                continue
            chance = np.prod(
                [
                    chances[n][f]
                    for n, f in status.items()
                    if held is None or n != held[0]
                ]
            )
            if outcome[tuple(status.values())]:
                failure += chance
            else:
                success += chance
        return success, failure

    # The minimal cut sets: the failing assignments' events (each failed
    # component, by its phase), less those implied by another's: the same
    # components failed by those phases or earlier ones, or fewer of them.
    cuts = [
        {n: f for n, f in status.items() if f < count}
        for status in assignments
        if outcome[tuple(status.values())]
    ]
    minimal = [
        cut
        for cut in cuts
        if not any(
            other != cut
            and all(n in cut and cut[n] <= j for n, j in other.items())
            for other in cuts
        )
    ]
    cut_sets = {
        frozenset((n, mission.phases[j].name) for n, j in cut.items())
        for cut in minimal
    }
    success, failure = probabilities()
    out: dict = {name: {} for name in MEASURES}
    out["criticality_success"] = {}
    out["rare_event"] = {}
    # The sums of products above are good to about 1e-13 of 1, and the
    # ratios to that over the unreliability; the mission's own values
    # keep a small measure's relative precision, which these do not.
    out["tolerance"] = 1e-13 / failure
    for node in nodes:
        r_works, f_works = probabilities((node, count))
        r_failed, f_failed = probabilities((node, 0))
        out["birnbaum_importance"][node] = r_works - r_failed
        out["improvement_potential"][node] = r_works - success
        out["risk_achievement_worth"][node] = f_failed / failure
        out["risk_reduction_worth"][node] = failure / f_works
        out["criticality_importance"][node] = (r_works - success) / failure
        out["criticality_success"][node] = (success - r_failed) / success
        naming = [cut for cut in minimal if node in cut]
        occurred = 0.0
        for status in assignments:
            if any(
                all(status[n] <= j for n, j in cut.items()) for cut in naming
            ):
                occurred += np.prod([chances[n][f] for n, f in status.items()])
        out["fussell_vesely"][node] = occurred / failure
        out["rare_event"][node] = (
            sum(
                np.prod([failed[n][j] for n, j in cut.items()])
                for cut in naming
            )
            / failure
        )
    return out, cut_sets


def check_against(mission, expected, cut_sets, rel=1e-12):
    close = {"rel": rel, "abs": expected["tolerance"]}
    for name in MEASURES:
        got = getattr(mission, name)()
        assert list(got) == list(mission.components)
        for node, value in expected[name].items():
            assert got[node] == pytest.approx(value, **close), (name, node)
    for node, value in expected["criticality_success"].items():
        got = mission.criticality_importance(kind="success")[node]
        assert got == pytest.approx(value, **close), node
    for node, value in expected["rare_event"].items():
        got = mission.fussell_vesely(method="rare_event")[node]
        assert got == pytest.approx(value, **close), node
    assert set(mission.minimal_cut_sets()) == cut_sets


@pytest.mark.parametrize("name", DIAGRAMS)
def test_a_one_phase_mission_has_the_diagram_s_measures(name):
    diagram = DIAGRAMS[name]
    mission = PhasedMission([("only", 40.0, diagram)])
    for measure in MEASURES:
        got = getattr(mission, measure)()
        expected = getattr(diagram, measure)(40.0)
        assert set(got) == set(expected)
        for node, value in expected.items():
            assert got[node] == pytest.approx(float(value), rel=1e-12)
    for kind in ("failure", "success"):
        got = mission.criticality_importance(kind=kind)
        expected = diagram.criticality_importance(40.0, kind=kind)
        for node, value in expected.items():
            assert got[node] == pytest.approx(float(value), rel=1e-12)
    got = mission.fussell_vesely(method="rare_event")
    expected = diagram.fussell_vesely(40.0, method="rare_event")
    for node, value in expected.items():
        assert got[node] == pytest.approx(float(value), rel=1e-12)
    assert mission.minimal_cut_sets() == [
        frozenset((node, "only") for node in cut)
        for cut in diagram.minimal_cut_sets()
    ]


def two_phase_mission():
    """Take-off on both engines and the gear, then cruise on either engine
    and the navigation unit."""
    engine, gear, nav = W([200.0, 1.8]), W([500.0, 2.5]), E([0.002])
    takeoff = NonRepairableRBD(
        [("s", "e1"), ("e1", "e2"), ("e2", "g"), ("g", "t")],
        {"e1": engine, "e2": engine, "g": gear},
    )
    cruise = NonRepairableRBD(
        [("s", "e1"), ("s", "e2"), ("e1", "n"), ("e2", "n"), ("n", "t")],
        {"e1": engine, "e2": engine, "n": nav},
    )
    return PhasedMission(
        [("take-off", 0.5, takeoff), ("cruise", 20.0, cruise)]
    )


def test_a_two_phase_mission_by_enumerating_the_failure_phases():
    mission = two_phase_mission()
    expected, cut_sets = by_brute_force(mission)
    check_against(mission, expected, cut_sets)
    # The gear is needed only to take off: a cut set at the take-off,
    # and a failure in the cruise does no harm.
    assert mission.minimal_cut_sets() == [
        frozenset({("e1", "take-off")}),
        frozenset({("e2", "take-off")}),
        frozenset({("g", "take-off")}),
        frozenset({("n", "cruise")}),
        frozenset({("e1", "cruise"), ("e2", "cruise")}),
    ]
    assert mission.fussell_vesely()["g"] < mission.fussell_vesely()["n"]
    for name in MEASURES:
        assert getattr(mission, name)()["g"] > 0.0


def test_missions_of_more_phases_and_kinds_by_enumerating():
    # Three phases: the landing's two-of-three engines with an auxiliary,
    # and the navigation unit unused after the cruise.
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
    expected, cut_sets = by_brute_force(mission)
    check_against(mission, expected, cut_sets, rel=1e-11)
    # A repeated node, a nested diagram and a fixed probability.
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
    mixed = PhasedMission([("first", 10.0, first), ("check", 0.0, check)])
    expected, cut_sets = by_brute_force(mixed)
    check_against(mixed, expected, cut_sets)


@pytest.mark.parametrize(
    "phases",
    [
        [("up", 30.0, BRIDGE), ("on", 30.0, VOTE)],
        [("a", 10.0, SERIES), ("b", 20.0, BRIDGE), ("c", 5.0, PARALLEL)],
    ],
    ids=["bridge then vote", "series, bridge, parallel"],
)
def test_the_measures_are_the_same_from_the_path_sets(phases, monkeypatch):
    diagram = PhasedMission(phases)
    monkeypatch.setattr(phased_mission, "METHOD", "paths")
    paths = PhasedMission(phases)
    for name in MEASURES:
        got, expected = getattr(diagram, name)(), getattr(paths, name)()
        assert list(got) == list(expected)
        for node in got:
            assert got[node] == pytest.approx(expected[node], rel=1e-10)
    assert diagram.minimal_cut_sets() == paths.minimal_cut_sets()
    assert diagram.analysis_routes()["fussell_vesely"].route == "exact"
    assert paths.analysis_routes()["fussell_vesely"].route == "exact"


def test_a_reliable_mission_keeps_its_measures_precise():
    # As in ``test_a_reliable_mission_keeps_its_unreliability_precise``:
    # the mission succeeds if "a" lasts 20 hours, whatever "b" does.
    tiny = E([1e-9])
    pair = NonRepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        {"a": tiny, "b": tiny},
    )
    one = NonRepairableRBD([("s", "a"), ("a", "t")], {"a": tiny})
    mission = PhasedMission([("both", 10.0, pair), ("one", 10.0, one)])
    failure = -np.expm1(-2e-8)
    assert mission.birnbaum_importance() == {"a": 1.0, "b": 0.0}
    potential = mission.improvement_potential()
    assert potential["a"] == pytest.approx(failure, rel=1e-9)
    assert potential["b"] == 0.0
    raw = mission.risk_achievement_worth()
    assert raw["a"] == pytest.approx(1.0 / failure, rel=1e-9)
    assert raw["b"] == pytest.approx(1.0, rel=1e-9)
    rrw = mission.risk_reduction_worth()
    assert rrw["a"] == np.inf and rrw["b"] == pytest.approx(1.0, rel=1e-9)
    assert mission.criticality_importance() == pytest.approx(
        {"a": 1.0, "b": 0.0}, rel=1e-9
    )
    # Given a success, "a" failed from the start would have failed it.
    assert mission.criticality_importance(kind="success") == pytest.approx(
        {"a": 1.0, "b": 0.0}, rel=1e-9
    )
    # "a" failed by the first phase fails the pair only with "b", but by
    # the second phase alone: the later, weaker event is the cut set.
    assert mission.minimal_cut_sets() == [frozenset({("a", "one")})]
    for method in ("exact", "rare_event"):
        assert mission.fussell_vesely(method=method) == pytest.approx(
            {"a": 1.0, "b": 0.0}, rel=1e-9
        )


def test_the_rare_event_sum_passes_the_exact_union():
    # On a bridge whose units are likely to fail, the cut sets overlap:
    # the sum of their probabilities passes the union's, and 1.
    likely = NonRepairableRBD(BRIDGE.G.edges, {n: F(0.8) for n in "abcde"})
    mission = PhasedMission([("up", 1.0, likely), ("on", 1.0, likely)])
    exact = mission.fussell_vesely()
    summed = mission.fussell_vesely(method="rare_event")
    assert exact["a"] == pytest.approx(
        float(likely.fussell_vesely(1.0)["a"]), rel=1e-12
    )
    assert summed["a"] == pytest.approx(
        float(likely.fussell_vesely(1.0, method="rare_event")["a"]), rel=1e-12
    )
    assert exact["a"] < 1.0 < summed["a"]


def test_the_measures_are_checked_and_refuse_a_large_diagram(monkeypatch):
    mission = two_phase_mission()
    with pytest.raises(ValueError, match="'failure' or 'success'"):
        mission.criticality_importance(kind="both")
    with pytest.raises(ValueError, match="'exact' or 'rare_event'"):
        mission.fussell_vesely(method="simulate")
    monkeypatch.setattr(phased_mission, "MAX_NODES", 3)
    large = two_phase_mission()
    for name in MEASURES:
        with pytest.raises(NotImplementedError, match="too large"):
            getattr(large, name)()
    # The cut sets need no diagram; the sum of their probabilities needs
    # the mission's unreliability, which does.
    assert len(large.minimal_cut_sets()) == 5
    with pytest.raises(NotImplementedError, match="too large"):
        large.fussell_vesely(method="rare_event")


def test_the_cut_sets_diagram_has_its_own_limit(monkeypatch):
    # A mission whose own diagram fits in the limit but whose cut sets'
    # does not refuses the exact Fussell-Vesely importance alone.
    mission = PhasedMission([("up", 30.0, PARALLEL), ("on", 30.0, SERIES)])
    mission.reliability()
    monkeypatch.setattr(phased_mission, "MAX_NODES", 4)
    with pytest.raises(NotImplementedError, match="rare_event"):
        mission.fussell_vesely()
    assert mission.analysis_routes()["fussell_vesely"].route == "refused"
    assert mission.analysis_routes()["birnbaum_importance"].route == "exact"
    assert mission.fussell_vesely(method="rare_event")["a"] > 0.0
