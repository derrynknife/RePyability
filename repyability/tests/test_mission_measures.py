"""A phased mission saved and loaded, and the measures it shares with the
other models: differential, joint and Barlow-Proschan importance, the
expected time it runs, its levers and parameter sensitivity, and the
importance measures of a mission with blocks on standby, from its Markov
chain."""

import json
import math
import warnings

import numpy as np
import pytest
import surpyval as surv
from scipy import integrate

from repyability import (
    BlockSetting,
    Lever,
    MissionPhase,
    NonRepairableRBD,
    PhasedMission,
)
from repyability.rbd import _mission_chain
from repyability.tests.catalogue import mission_kinds
from repyability.tests.test_analysis_routes import mission as bridge_vote
from repyability.tests.test_phased_mission import DIAGRAMS

E, W = surv.Exponential.from_params, surv.Weibull.from_params
F = surv.FixedEventProbability.from_params
P, B = MissionPhase, BlockSetting
PAIR = [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")]
SERIES = [("s", "a"), ("a", "b"), ("b", "t")]
MEASURES = (
    "birnbaum_importance",
    "improvement_potential",
    "risk_achievement_worth",
    "risk_reduction_worth",
    "criticality_importance",
    "differential_importance",
    "joint_importance",
)


def kinds():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {**mission_kinds(), "per phase": bridge_vote()}


KINDS = kinds()


def within(estimate, error, exact, z=4.0):
    return abs(estimate - exact) <= z * error + 1e-12


def replaced(mission, models):
    """A mission built from a diagram per phase, with ``models`` in
    place of its components'."""
    phases = []
    for phase in mission.phases:
        given = phase.rbd._init_args
        changed = {
            n: models.get(n, m) for n, m in given["reliabilities"].items()
        }
        phases.append(
            (
                phase.name,
                phase.duration,
                NonRepairableRBD(**{**given, "reliabilities": changed}),
            )
        )
    return PhasedMission(phases)


# -- saving ------------------------------------------------------------------


@pytest.mark.parametrize("kind", sorted(KINDS))
def test_every_mission_round_trips_through_json(kind):
    mission = KINDS[kind]
    text = mission.to_json()
    again = PhasedMission.from_json(text)
    assert json.loads(again.to_json()) == json.loads(text)
    assert [(p.name, p.duration) for p in again.phases] == [
        (p.name, p.duration) for p in mission.phases
    ]
    if mission.analysis_routes()["reliability"].route == "refused":
        # Simulated: the same draws give the same missions.
        assert again.reliability(
            "simulate", mc_samples=500, seed=3
        ) == mission.reliability("simulate", mc_samples=500, seed=3)
        return
    assert again.reliability() == mission.reliability()
    assert (
        again.phase_failure_probabilities()
        == mission.phase_failure_probabilities()
    )
    assert again.birnbaum_importance() == mission.birnbaum_importance()
    assert dict(again.joint_importance()) == dict(mission.joint_importance())


def test_a_mission_saves_its_settings_and_names():
    rbd = NonRepairableRBD(
        [("s", ("p", 1)), ("s", ("p", 2)), (("p", 1), "t"), (("p", 2), "t")],
        {("p", 1): E([0.01]), ("p", 2): E([0.02])},
    )
    warm = B(state="standby", dormancy_factor=0.3, start_probability=0.9)
    mission = PhasedMission.from_diagram(
        rbd,
        [
            P(("leg", 1), 5.0, blocks={("p", 2): warm}),
            P("leg 2", 10.0, blocks={("p", 1): B(model=E([0.05]))}),
        ],
    )
    d = mission.to_dict()
    assert d["type"] == "PhasedMission" and "repyability_version" in d
    again = PhasedMission.from_dict(json.loads(json.dumps(d)))
    assert [s.name for s in again._settings] == [("leg", 1), "leg 2"]
    assert again._settings[0].blocks[("p", 2)] == warm
    assert again._settings[1].blocks[("p", 1)].model.params == (
        pytest.approx([0.05])
    )
    assert again.reliability() == mission.reliability()


def test_saving_refuses_what_it_cannot_load():
    with pytest.raises(ValueError, match="PhasedMission"):
        PhasedMission.from_dict({"type": "Network"})
    with pytest.raises(ValueError, match="PhasedMission"):
        PhasedMission.from_dict([])


def test_a_mission_is_written_to_a_file(tmp_path):
    mission = KINDS["settings"]
    path = tmp_path / "mission.json"
    assert mission.to_json(path) is None
    assert PhasedMission.from_json(path).reliability() == (
        mission.reliability()
    )


# -- one phase is the diagram ------------------------------------------------


@pytest.mark.parametrize("name", sorted(DIAGRAMS))
def test_one_phase_s_measures_are_the_diagram_s(name):
    diagram, t = DIAGRAMS[name], 40.0
    mission = PhasedMission([("only", t, diagram)])
    for change, kind in (
        ("uniform", "failure"),
        ("proportional", "failure"),
        ("proportional", "success"),
    ):
        expected = diagram.differential_importance(t, change=change, kind=kind)
        found = mission.differential_importance(change=change, kind=kind)
        assert found == pytest.approx(expected, rel=1e-9, abs=1e-14)
    joint = diagram.joint_importance(t)
    found = mission.joint_importance()
    assert set(found) == set(joint)
    for pair, value in joint.items():
        assert found[pair] == pytest.approx(value, rel=1e-9, abs=1e-14)
    shares = diagram.barlow_proschan_importance(t)
    assert mission.barlow_proschan_importance() == pytest.approx(
        shares, rel=1e-7, abs=1e-12
    )
    times = np.array([10.0, 25.0])
    over = diagram.barlow_proschan_importance(times)
    for node, values in mission.barlow_proschan_importance(times).items():
        np.testing.assert_allclose(values, over[node], rtol=1e-7, atol=1e-12)
    expected_time, _ = integrate.quad(
        lambda x: float(diagram.sf(x)), 0.0, t, epsabs=0, epsrel=1e-12
    )
    assert mission.expected_mission_time() == pytest.approx(
        expected_time, rel=1e-8
    )
    sensitivity = mission.parameter_sensitivity()
    for node, values in diagram.parameter_sensitivity(t).items():
        assert sensitivity[node] == pytest.approx(values, rel=1e-5, abs=1e-12)
    # The duration moves the reliability by minus the density.
    assert sensitivity[None]["phase.only.duration"] == pytest.approx(
        -float(diagram.df(t)), rel=1e-5
    )


# -- two phases by hand ------------------------------------------------------


def test_series_then_parallel_by_hand():
    la, lb, t1, t2 = 0.01, 0.03, 5.0, 20.0
    total = la + lb
    series = NonRepairableRBD(SERIES, {"a": E([la]), "b": E([lb])})
    parallel = NonRepairableRBD(PAIR, {"a": E([la]), "b": E([lb])})
    mission = PhasedMission([("one", t1, series), ("two", t2, parallel)])
    first = math.exp(-total * t1)

    def reliability(la, lb, t1, t2):
        return math.exp(-(la + lb) * t1) * (
            math.exp(-la * t2) + math.exp(-lb * t2) - math.exp(-(la + lb) * t2)
        )

    r = reliability(la, lb, t1, t2)
    assert mission.reliability() == pytest.approx(r, rel=1e-12)
    birnbaum = {"a": math.exp(-lb * t1), "b": math.exp(-la * t1)}
    assert mission.birnbaum_importance() == pytest.approx(birnbaum, rel=1e-12)
    whole = birnbaum["a"] + birnbaum["b"]
    assert mission.differential_importance() == pytest.approx(
        {node: value / whole for node, value in birnbaum.items()}, rel=1e-12
    )
    # Both are needed in the first phase: complements.
    assert mission.joint_importance()["a", "b"] == pytest.approx(1.0)
    # Each one's failure in the first phase, and in the second after the
    # other's.
    parts = {
        node: rate * (1 - first) / total
        + first
        * (
            (1 - math.exp(-rate * t2))
            - rate * (1 - math.exp(-total * t2)) / total
        )
        for node, rate in (("a", la), ("b", lb))
    }
    assert sum(parts.values()) == pytest.approx(1 - r, rel=1e-12)
    assert mission.barlow_proschan_importance() == pytest.approx(
        {node: part / (1 - r) for node, part in parts.items()}, rel=1e-8
    )
    expected = (1 - first) / total + first * (
        (1 - math.exp(-la * t2)) / la
        + (1 - math.exp(-lb * t2)) / lb
        - (1 - math.exp(-total * t2)) / total
    )
    assert mission.expected_mission_time() == pytest.approx(expected, rel=1e-9)
    sensitivity = mission.parameter_sensitivity()
    h = 1e-7
    for key, name, moved in (
        ("a", "failure_rate", lambda h: reliability(la + h, lb, t1, t2)),
        ("b", "failure_rate", lambda h: reliability(la, lb + h, t1, t2)),
        (
            None,
            "phase.one.duration",
            lambda h: reliability(la, lb, t1 + h, t2),
        ),
        (
            None,
            "phase.two.duration",
            lambda h: reliability(la, lb, t1, t2 + h),
        ),
    ):
        derivative = (moved(h) - moved(-h)) / (2 * h)
        assert sensitivity[key][name] == pytest.approx(derivative, rel=1e-6)


def test_a_failure_at_a_phase_s_start_takes_no_part():
    # Parallel, then series: a unit lost in the first phase fails the
    # second at its start, a failure of no unit then; the shares are of
    # the failures within the phases.
    la, lb, t1, t2 = 0.02, 0.01, 10.0, 10.0
    parallel = NonRepairableRBD(PAIR, {"a": E([la]), "b": E([lb])})
    series = NonRepairableRBD(SERIES, {"a": E([la]), "b": E([lb])})
    mission = PhasedMission([("one", t1, parallel), ("two", t2, series)])
    both = math.exp(-(la + lb) * t1)
    # Within the first: both lost. Within the second: from both alive.
    within_first = (1 - math.exp(-la * t1)) * (1 - math.exp(-lb * t1))
    within_second = both * (1 - math.exp(-(la + lb) * t2))
    a = integrate.quad(
        lambda t: la * math.exp(-la * t) * (1 - math.exp(-lb * t)), 0, t1
    )[0] + both * la / (la + lb) * (1 - math.exp(-(la + lb) * t2))
    shares = mission.barlow_proschan_importance()
    assert shares["a"] == pytest.approx(
        a / (within_first + within_second), rel=1e-8
    )
    assert sum(shares.values()) == pytest.approx(1.0)
    assert mission.unreliability() > within_first + within_second


# -- against finite differences ----------------------------------------------


def test_differential_and_joint_importance_against_differences():
    # A component working throughout with probability x, else failed from
    # the start, is a probability of failing 1 - x: the reliability is
    # linear in x, its slope the Birnbaum importance, and the mixed
    # difference of two the joint importance.
    mission = bridge_vote()
    birnbaum = mission.birnbaum_importance()
    joint = mission.joint_importance()
    h = 0.25

    def at(**x):
        return replaced(mission, {n: F(1.0 - v) for n, v in x.items()})

    slopes = {}
    for node in mission.components:
        up = at(**{node: 0.5 + h}).reliability()
        down = at(**{node: 0.5 - h}).reliability()
        slopes[node] = (up - down) / (2 * h)
        assert slopes[node] == pytest.approx(birnbaum[node], abs=1e-12)
    total = sum(slopes.values())
    assert mission.differential_importance() == pytest.approx(
        {node: slope / total for node, slope in slopes.items()}, abs=1e-12
    )
    for i, j in (("a", "b"), ("a", "d"), ("c", "v")):
        mixed = (
            at(**{i: 1.0, j: 1.0}).reliability()
            - at(**{i: 1.0, j: 0.0}).reliability()
            - at(**{i: 0.0, j: 1.0}).reliability()
            + at(**{i: 0.0, j: 0.0}).reliability()
        )
        assert joint[i, j] == pytest.approx(mixed, abs=1e-12)


@pytest.mark.parametrize("kind", ["settings", "standby_exact", "per phase"])
def test_parameter_sensitivity_against_with_levers(kind):
    mission = KINDS[kind]
    sensitivity = mission.parameter_sensitivity()
    for lever in mission.levers():
        h = 1e-3 * abs(lever.value) or 1e-3
        up = mission.with_levers({lever: lever.value + h}).reliability()
        if lever.value - h >= lever.bounds[0]:
            down = mission.with_levers({lever: lever.value - h}).reliability()
            difference = (up - down) / (2 * h)
        else:
            difference = (up - mission.reliability()) / h
        assert sensitivity[lever.key][lever.name] == pytest.approx(
            difference, rel=1e-4, abs=1e-10
        ), lever


# -- levers ------------------------------------------------------------------


@pytest.mark.parametrize("kind", sorted(KINDS))
def test_the_levers_are_those_the_sensitivity_reports(kind):
    mission = KINDS[kind]
    levers = mission.levers()
    assert all(isinstance(lever, Lever) for lever in levers)
    assert [lever.name for lever in levers if lever.key is None] == [
        f"phase.{phase.name}.duration" for phase in mission.phases
    ]
    if mission.analysis_routes()["parameter_sensitivity"].route == "refused":
        with pytest.raises(NotImplementedError, match="failure_drivers"):
            mission.parameter_sensitivity()
        return
    reported = mission.parameter_sensitivity()
    assert [(lever.key, lever.name) for lever in levers] == [
        (key, name) for key, values in reported.items() for name in values
    ]


@pytest.mark.parametrize("kind", sorted(KINDS))
def test_a_lever_s_range_is_where_its_mission_builds(kind):
    mission = KINDS[kind]
    for lever in mission.levers():
        low, high = lever.bounds
        assert low <= lever.value <= high, lever
        mission.with_levers({lever: lever.value})
        for end, past in ((low, -1.0), (high, 1.0)):
            if math.isfinite(end):
                with pytest.raises(ValueError, match=f"{lever.name!r} of"):
                    mission.with_levers({lever: end + past * 1e-6})


def test_with_levers_moves_a_component_in_every_phase():
    mission = bridge_vote()
    moved = mission.with_levers(
        {("c", "failure_rate"): 0.02, (None, "phase.on.duration"): 10.0}
    )
    assert [p.rbd.reliabilities["c"].params[0] for p in moved.phases] == [
        0.02,
        0.02,
    ]
    assert moved.duration == 40.0
    # Moved from a mission drawn as one diagram, it keeps its settings.
    drawn = KINDS["settings"]
    again = drawn.with_levers({("c", "failure_rate"): 0.02})
    assert again._settings == drawn._settings
    assert again.reliability() < drawn.reliability()
    with pytest.raises(ValueError, match="no lever 'alpah'.*'alpha'"):
        mission.with_levers({("a", "alpah"): 1.0})
    with pytest.raises(ValueError, match="phases' durations"):
        mission.with_levers({("z", "alpha"): 1.0})


# -- blocks on standby: the measures from the chain ---------------------------


def test_a_cold_spare_s_measures_by_hand():
    lam, t, p = 0.01, 50.0, 0.9
    rbd = NonRepairableRBD(PAIR, {"a": E([lam]), "b": E([lam])})
    spare = B(state="standby", start_probability=p)
    mission = PhasedMission.from_diagram(
        rbd, [P("only", t, blocks={"b": spare})]
    )
    r = math.exp(-lam * t) * (1 + p * lam * t)
    assert mission.reliability() == pytest.approx(r, rel=1e-9)
    # a failed from the start: b called on at once; b held: it starts
    # surely and never fails.
    birnbaum = {"a": 1 - p * math.exp(-lam * t), "b": 1 - math.exp(-lam * t)}
    assert mission.birnbaum_importance() == pytest.approx(birnbaum, rel=1e-9)
    assert mission.risk_achievement_worth()["a"] == pytest.approx(
        (1 - p * math.exp(-lam * t)) / (1 - r), rel=1e-9
    )
    assert mission.improvement_potential() == pytest.approx(
        {"a": 1 - r, "b": 1 - r}, rel=1e-9
    )
    # Either held working carries the mission: substitutes.
    assert mission.joint_importance()["a", "b"] == pytest.approx(-1.0)
    table = mission.importances()
    assert set(table.refused) == {
        "fussell_vesely",
        "barlow_proschan_importance",
    }
    assert "failure_drivers" in table.refused["fussell_vesely"]


def plant():
    """Start-up, duty, standby: two pumps, one on standby with warm
    dormancy and an imperfect switch; a valve needed only on duty (see
    ``test_mission_standby_chain``)."""
    rbd = NonRepairableRBD(
        [
            ("s", "p1"),
            ("s", "p2"),
            ("p1", "valve"),
            ("p2", "valve"),
            ("valve", "t"),
        ],
        {"p1": E([0.004]), "p2": E([0.006]), "valve": E([0.001])},
    )
    spare = B(state="standby", dormancy_factor=0.2, start_probability=0.95)
    phases = [
        P("start-up", 2.0, blocks={"p1": "guaranteed", "p2": spare}),
        P("duty", 100.0, blocks={"p2": spare}),
        P("standby", 50.0, blocks={"valve": "not_needed", "p1": spare}),
    ]
    return rbd, phases


def test_the_standby_measures_against_the_simulation():
    rbd, phases = plant()
    mission = PhasedMission.from_diagram(rbd, phases)
    n = 40_000
    held = mission._held()
    # Held working: the simulation's drivers hold a block never failing.
    potential = mission.improvement_potential()
    played = mission.failure_drivers("simulate", mc_samples=n, seed=7)
    for node, value in potential.items():
        assert within(played.totals[node], played.standard_errors[node], value)
    assert potential == pytest.approx(
        mission.failure_drivers().totals, rel=1e-9, abs=1e-15
    )
    # Held failed: a block dead on arrival (a probability of failing 1),
    # simulated.
    for node in ("p2", "valve"):
        models = {**rbd._init_args["reliabilities"], node: F(1.0)}
        dead = PhasedMission.from_diagram(
            NonRepairableRBD(rbd._init_args["edges"], models), phases
        )
        estimate = dead.reliability("simulate", mc_samples=n, seed=11)
        error = math.sqrt(max(estimate * (1 - estimate), 1e-12) / n)
        assert within(estimate, error, held[node][2])


def test_the_chain_s_holds_are_the_decision_diagram_s(monkeypatch):
    # Without standby the chain gives the decision diagram's values, a
    # component held working or failed too.
    rbd = NonRepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "c"), ("b", "c"), ("c", "t")],
        {"a": E([0.01]), "b": E([0.03]), "c": E([0.002])},
    )
    phases = [
        P("p", 10.0, blocks={"b": "guaranteed"}),
        P("q", 20.0, blocks={"c": "not_needed"}),
        P("r", 5.0),
    ]
    by_diagram = PhasedMission.from_diagram(rbd, phases)
    by_chain = PhasedMission.from_diagram(rbd, phases)
    monkeypatch.setattr(by_chain, "_standby", lambda: ["b"])
    assert by_chain.reliability() == pytest.approx(
        by_diagram.reliability(), rel=1e-10
    )
    for name in MEASURES:
        found = dict(getattr(by_chain, name)())
        expected = dict(getattr(by_diagram, name)())
        assert found == pytest.approx(expected, rel=1e-8, abs=1e-14), name
    assert by_chain.criticality_importance("success") == pytest.approx(
        by_diagram.criticality_importance("success"), rel=1e-8
    )


def test_a_chain_that_cannot_be_worked_out_refuses_the_measures():
    rbd = NonRepairableRBD(PAIR, {"a": W([100.0, 2.0]), "b": E([0.01])})
    mission = PhasedMission.from_diagram(
        rbd, [P("only", 10.0, blocks={"b": "standby"})]
    )
    report = mission.analysis_routes()
    for name in MEASURES + ("parameter_sensitivity",):
        assert report[name].route == "refused"
        with pytest.raises(NotImplementedError, match="failure_drivers"):
            getattr(mission, name)()
    assert report["expected_mission_time"].route == "simulated"
    simulated = mission.expected_mission_time(mc_samples=2000, seed=1)
    assert 9.0 < simulated <= 10.0


def test_expected_mission_time_simulated_against_exact():
    rbd, phases = plant()
    mission = PhasedMission.from_diagram(rbd, phases)
    exact = mission.expected_mission_time()
    n = 20_000
    runs = mission._simulated_runs(n, 5)
    ran = np.minimum(runs.time, mission.duration)
    assert mission.expected_mission_time(
        "simulate", mc_samples=n, seed=5
    ) == pytest.approx(float(ran.mean()))
    assert within(float(ran.mean()), ran.std() / math.sqrt(n), exact)


def test_the_chain_s_failed_hold_starts_failed():
    rbd = NonRepairableRBD(PAIR, {"a": E([0.01]), "b": E([0.02])})
    mission = PhasedMission.from_diagram(
        rbd, [P("only", 10.0, blocks={"b": "standby"})]
    )
    values = _mission_chain.run(mission, held={"a": False})
    assert values.through[-1] == pytest.approx(math.exp(-0.2), rel=1e-10)
