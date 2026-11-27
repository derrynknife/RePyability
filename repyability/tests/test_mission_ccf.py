"""Common-cause groups in a phased mission (see ``_mission_ccf``): each
shared cause a component of its own, in series with every member it
fails, through every phase.

The exact values are checked against closed forms written from the
causes' lives, against ``NonRepairableRBD`` with the same group (which
conditions on the group's outcomes instead), and against every state of
the segments enumerated; the simulation against the exact values.
"""

import itertools
import math

import numpy as np
import pytest
import surpyval as surv

from repyability import (
    MGL,
    AlphaFactor,
    BetaFactor,
    BlockSetting,
    CCFGroup,
    MissionPhase,
    NonRepairableRBD,
    PhasedMission,
    Shocks,
)
from repyability.rbd.helper_classes import PerfectReliability

UNIT = surv.Weibull.from_params([100, 2])
PARALLEL = [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")]
SERIES = [("s", "a"), ("a", "b"), ("b", "t")]
PAIR = NonRepairableRBD(PARALLEL, {"a": UNIT, "b": UNIT})
BOTH = NonRepairableRBD(SERIES, {"a": UNIT, "b": UNIT})
SAMPLES = 50_000


def H(t: float) -> float:
    """The cumulative hazard of ``UNIT``."""
    return (t / 100.0) ** 2


def beta_group(beta: float) -> CCFGroup:
    return CCFGroup(["a", "b"], BetaFactor(beta, basis="rate"))


def two_phases(groups=None) -> PhasedMission:
    return PhasedMission(
        [("up", 30.0, BOTH), ("on", 20.0, PAIR)], ccf_groups=groups
    )


def within_four_errors(mission: PhasedMission, seed: int) -> None:
    """The simulated reliability of ``SAMPLES`` seeded missions against
    the exact one."""
    exact = mission.reliability()
    simulated = mission.reliability(
        method="simulate", mc_samples=SAMPLES, seed=seed
    )
    error = math.sqrt(exact * (1.0 - exact) / SAMPLES)
    assert abs(simulated - exact) < 4.0 * error, (simulated, exact, error)


# ----------------------------------------------------------------------
# One phase: the diagram's own values with the group
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "model",
    [
        BetaFactor(0.2, basis="rate"),
        MGL(0.2, basis="rate"),
        AlphaFactor([0.9, 0.1], basis="rate"),
        Shocks(0.001, 0.5, 0.0005),
    ],
    ids=["beta", "mgl", "alpha", "shocks"],
)
def test_one_phase_is_the_diagram_with_the_group(model):
    group = CCFGroup(["a", "b"], model)
    diagram = NonRepairableRBD(
        PARALLEL, {"a": UNIT, "b": UNIT}, ccf_groups=[group]
    )
    mission = PhasedMission([("only", 40.0, PAIR)], ccf_groups=[group])
    assert mission.reliability() == pytest.approx(
        float(diagram.sf(40.0)), rel=1e-13
    )
    assert mission.unreliability() == pytest.approx(
        float(diagram.ff(40.0)), rel=1e-12
    )
    times = np.array([0.0, 10.0, 25.0, 40.0])
    assert mission.reliability_over_time(times) == pytest.approx(
        np.asarray(diagram.sf(times), dtype=float), rel=1e-13
    )


def test_three_members_with_partial_causes_are_the_diagram_with_the_group():
    # An MGL group of three by rate has causes that fail two of the three.
    edges = [("s", n) for n in "abc"] + [(n, "v") for n in "abc"]
    edges.append(("v", "t"))
    models = {"a": UNIT, "b": UNIT, "c": UNIT, "v": PerfectReliability}
    group = CCFGroup(["a", "b", "c"], MGL(0.2, 0.4, basis="rate"))
    plain = NonRepairableRBD(edges, models, k={"v": 2})
    diagram = NonRepairableRBD(edges, models, k={"v": 2}, ccf_groups=[group])
    mission = PhasedMission(
        [("out", 25.0, plain), ("back", 15.0, plain)], ccf_groups=[group]
    )
    assert mission.reliability() == pytest.approx(
        float(diagram.sf(40.0)), rel=1e-13
    )
    assert len(mission.components) == 3 + 4  # each pair's cause, and all's


# ----------------------------------------------------------------------
# Two phases, two units, by hand
# ----------------------------------------------------------------------


@pytest.mark.parametrize("beta", [0.05, 0.2, 0.7])
def test_two_phases_by_hand(beta):
    # Both to 30 (own causes and the shared one), then either to 50: the
    # shared cause not struck by 50, and not both own causes in (30, 50].
    mission = two_phases([beta_group(beta)])
    own = 1.0 - beta
    up = math.exp(-beta * H(30.0)) * math.exp(-2.0 * own * H(30.0))
    later = 1.0 - math.exp(-own * (H(50.0) - H(30.0)))
    through = (
        math.exp(-beta * H(50.0))
        * math.exp(-2.0 * own * H(30.0))
        * (1.0 - later**2)
    )
    failing = mission.phase_failure_probabilities()
    assert failing["up"] == pytest.approx(1.0 - up, rel=1e-13)
    assert failing["on"] == pytest.approx(up - through, rel=1e-12)
    assert mission.reliability() == pytest.approx(through, rel=1e-13)


def test_no_shared_cause_is_no_group():
    # beta = 0: the members' lives as given, and no cause; to the last bit.
    plain = two_phases()
    mission = two_phases([beta_group(0.0)])
    assert list(mission.components) == ["a", "b"]
    assert mission.reliability() == plain.reliability()
    assert (
        mission.phase_failure_probabilities()
        == plain.phase_failure_probabilities()
    )


def test_only_the_shared_cause_is_one_unit():
    # beta = 1: the members fail together, as one unit of their life.
    mission = two_phases([beta_group(1.0)])
    assert list(mission.components) == [("cause", ("a", "b"))]
    assert mission.reliability() == pytest.approx(float(UNIT.sf(50.0)))
    one = NonRepairableRBD([("s", "a"), ("a", "t")], {"a": UNIT})
    single = PhasedMission([("up", 30.0, one), ("on", 20.0, one)])
    assert mission.phase_failure_probabilities() == pytest.approx(
        single.phase_failure_probabilities(), rel=1e-13
    )


def test_shocks_by_hand_with_members_of_different_lives():
    # Lethal shocks at 0.002 on members of different lives: the pair for
    # 30, then b alone for 20.
    a = surv.Exponential.from_params([0.01])
    b = surv.Weibull.from_params([80, 1.5])
    pair = NonRepairableRBD(PARALLEL, {"a": a, "b": b})
    alone = NonRepairableRBD([("s", "b"), ("b", "t")], {"b": b})
    mission = PhasedMission(
        [("both", 30.0, pair), ("one", 20.0, alone)],
        ccf_groups=CCFGroup(["a", "b"], Shocks(0.0, lethal_rate=0.002)),
    )
    expected = math.exp(-0.002 * 50.0) * float(b.sf(50.0))
    assert mission.reliability() == pytest.approx(expected, rel=1e-13)
    first = math.exp(-0.002 * 30.0) * (
        1.0 - float(a.ff(30.0)) * float(b.ff(30.0))
    )
    assert mission.phase_failure_probabilities()["both"] == pytest.approx(
        1.0 - first, rel=1e-12
    )


def test_a_cause_that_strikes_in_a_phase_fails_its_members_in_the_next():
    # The first phase needs neither member (a third unit alone), the
    # second both: a cause struck in the first fails the mission at the
    # second's start, and every state of the segments says the same.
    c = surv.Exponential.from_params([0.001])
    first = NonRepairableRBD([("s", "c"), ("c", "t")], {"c": c})
    mission = PhasedMission(
        [("out", 30.0, first), ("on", 20.0, BOTH)],
        ccf_groups=[beta_group(0.3)],
    )
    survive = {
        "a": [math.exp(-0.7 * H(30.0)), math.exp(-0.7 * (H(50.0) - H(30.0)))],
        "b": [math.exp(-0.7 * H(30.0)), math.exp(-0.7 * (H(50.0) - H(30.0)))],
        "z": [math.exp(-0.3 * H(30.0)), math.exp(-0.3 * (H(50.0) - H(30.0)))],
        "c": [math.exp(-0.03), math.exp(-0.02)],
    }
    names = list(survive)
    total = 0.0
    for state in itertools.product([True, False], repeat=2 * len(names)):
        lasted = {
            name: (state[2 * i], state[2 * i] and state[2 * i + 1])
            for i, name in enumerate(names)
        }
        chance = 1.0
        for i, name in enumerate(names):
            first_p, second_p = survive[name]
            chance *= first_p if state[2 * i] else 1.0 - first_p
            chance *= second_p if state[2 * i + 1] else 1.0 - second_p
        if lasted["c"][0] and all(lasted[n][1] for n in ("a", "b", "z")):
            total += chance
    assert mission.reliability() == pytest.approx(total, rel=1e-12)
    cut_sets = mission.minimal_cut_sets()
    assert frozenset({(("cause", ("a", "b")), "on")}) in cut_sets


# ----------------------------------------------------------------------
# The simulation
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "groups, seed",
    [
        ([CCFGroup(["a", "b"], BetaFactor(0.2, basis="rate"))], 1),
        ([CCFGroup(["a", "b"], Shocks(0.002, 0.5, 0.001))], 2),
    ],
    ids=["beta", "shocks"],
)
def test_the_simulation_agrees_with_the_exact_values(groups, seed):
    within_four_errors(two_phases(groups), seed)


def test_a_model_with_a_hazard_ratio_draws_with_it():
    # The draws of a cause's life (a share of the members' hazard).
    shared = UNIT.with_params(UNIT.params, hr=0.3)
    one = NonRepairableRBD([("s", "a"), ("a", "t")], {"a": shared})
    np.random.seed(3)
    lives = one.random(SAMPLES)
    exact = float(shared.sf(100.0))
    error = math.sqrt(exact * (1.0 - exact) / SAMPLES)
    assert abs(np.mean(lives > 100.0) - exact) < 4.0 * error


# ----------------------------------------------------------------------
# One diagram, measures, levers, saving
# ----------------------------------------------------------------------


def test_a_mission_drawn_as_one_diagram_takes_the_groups():
    group = beta_group(0.2)
    drawn = PhasedMission.from_diagram(
        BOTH,
        [
            MissionPhase("up", 30.0),
            MissionPhase("on", 20.0, blocks={"a": "not_needed"}),
        ],
        ccf_groups=group,
    )
    # Both to 30, then b alone to 50: the own failures of a to 30 and of
    # b to 50, and the shared cause, which b still answers to, to 50.
    expected = (
        math.exp(-0.8 * H(30.0))
        * math.exp(-0.8 * H(50.0))
        * math.exp(-0.2 * H(50.0))
    )
    assert drawn.reliability() == pytest.approx(expected, rel=1e-13)
    assert drawn.ccf_groups == [group]


def test_the_measures_name_each_cause():
    mission = two_phases([beta_group(0.2)])
    cause = ("cause", ("a", "b"))
    worth = mission.birnbaum_importance()
    assert list(worth) == ["a", cause, "b"]
    # The cause held working, the mission is that of the members' own
    # failures; held failed, it fails.
    own = UNIT.with_params(UNIT.params, hr=0.8)
    alone = PhasedMission(
        [
            ("up", 30.0, NonRepairableRBD(SERIES, {"a": own, "b": own})),
            ("on", 20.0, NonRepairableRBD(PARALLEL, {"a": own, "b": own})),
        ]
    )
    assert worth[cause] == pytest.approx(alone.reliability(), rel=1e-12)
    drivers = mission.failure_drivers()
    assert drivers.totals[cause] == pytest.approx(
        alone.reliability() - mission.reliability(), rel=1e-10
    )


def test_a_groups_levers_are_under_its_members():
    mission = two_phases([beta_group(0.2)])
    levers = [(lever.key, lever.name) for lever in mission.levers()]
    assert levers == [
        (("a", "b"), "alpha"),
        (("a", "b"), "beta"),
        (("a", "b"), "ccf_beta"),
        (None, "phase.up.duration"),
        (None, "phase.on.duration"),
    ]
    sens = mission.parameter_sensitivity()
    assert [
        (key, name) for key, names in sens.items() for name in names
    ] == levers
    step = 1e-6
    high = mission.with_levers({(("a", "b"), "ccf_beta"): 0.2 + step})
    low = mission.with_levers({(("a", "b"), "ccf_beta"): 0.2 - step})
    assert sens[("a", "b")]["ccf_beta"] == pytest.approx(
        (high.reliability() - low.reliability()) / (2 * step), rel=1e-6
    )
    stiffer = mission.with_levers({(("a", "b"), "alpha"): 120.0})
    longer = surv.Weibull.from_params([120, 2])
    assert stiffer.reliability() == pytest.approx(
        PhasedMission(
            [
                (
                    "up",
                    30.0,
                    NonRepairableRBD(SERIES, {"a": longer, "b": longer}),
                ),
                (
                    "on",
                    20.0,
                    NonRepairableRBD(PARALLEL, {"a": longer, "b": longer}),
                ),
            ],
            ccf_groups=[beta_group(0.2)],
        ).reliability(),
        rel=1e-13,
    )
    with pytest.raises(ValueError):
        mission.with_levers({(("a", "b"), "ccf_beta"): 1.5})


def test_shocks_levers_are_each_members_and_the_groups():
    mission = two_phases([CCFGroup(["a", "b"], Shocks(0.002, 0.5))])
    levers = [(lever.key, lever.name) for lever in mission.levers()]
    assert levers[:7] == [
        ("a", "alpha"),
        ("a", "beta"),
        ("b", "alpha"),
        ("b", "beta"),
        (("a", "b"), "ccf_rate"),
        (("a", "b"), "ccf_p"),
        (("a", "b"), "ccf_lethal_rate"),
    ]
    sens = mission.parameter_sensitivity()
    assert sens[("a", "b")]["ccf_rate"] < 0.0


@pytest.mark.parametrize("drawn", [False, True], ids=["phases", "diagram"])
def test_a_mission_with_groups_saves_and_loads(drawn):
    groups = [CCFGroup(["a", "b"], Shocks(0.002, 0.5, 0.001))]
    if drawn:
        mission = PhasedMission.from_diagram(
            PAIR,
            [MissionPhase("up", 30.0), MissionPhase("on", 20.0)],
            ccf_groups=groups,
        )
    else:
        mission = two_phases([beta_group(0.2)])
    again = PhasedMission.from_json(mission.to_json())
    assert again.reliability() == mission.reliability()
    assert repr(again.ccf_groups) == repr(mission.ccf_groups)
    assert list(again.components) == list(mission.components)


def test_no_group_changes_nothing():
    # The mission as before: its components, phases and values.
    assert two_phases().ccf_groups == []
    assert two_phases().phases[0].rbd is BOTH
    assert "ccf_groups" not in two_phases().to_dict()


# ----------------------------------------------------------------------
# Refusals
# ----------------------------------------------------------------------


def test_a_probability_split_is_refused_naming_the_rate():
    with pytest.raises(ValueError, match="basis='rate'"):
        two_phases([CCFGroup(["a", "b"], BetaFactor(0.2))])


def test_groups_are_checked():
    with pytest.raises(ValueError, match="not a component of the mission"):
        two_phases([CCFGroup(["a", "z"], BetaFactor(0.2, basis="rate"))])
    with pytest.raises(ValueError, match="more than one CCF group"):
        two_phases([beta_group(0.2), beta_group(0.1)])
    other = NonRepairableRBD(
        PARALLEL, {"a": UNIT, "b": surv.Weibull.from_params([90, 2])}
    )
    with pytest.raises(ValueError, match="not symmetric"):
        PhasedMission([("only", 10.0, other)], ccf_groups=[beta_group(0.2)])
    with pytest.raises(ValueError, match="CCFGroup"):
        two_phases(["a group"])


def test_a_rate_split_needs_a_model_with_a_hazard_ratio():
    nested = NonRepairableRBD([("s", "x"), ("x", "t")], {"x": UNIT})
    diagram = NonRepairableRBD(PARALLEL, {"a": nested, "b": nested})
    with pytest.raises(ValueError, match="hazard ratio.*Shocks"):
        PhasedMission([("only", 10.0, diagram)], ccf_groups=[beta_group(0.2)])
    # Shocks take members of any model.
    shocked = PhasedMission(
        [("only", 10.0, diagram)],
        ccf_groups=[CCFGroup(["a", "b"], Shocks(0.0, lethal_rate=0.01))],
    )
    assert shocked.reliability() == pytest.approx(
        math.exp(-0.1) * (1.0 - float(UNIT.ff(10.0)) ** 2), rel=1e-13
    )


def test_a_phases_own_groups_are_refused_naming_the_missions():
    grouped = NonRepairableRBD(
        PARALLEL, {"a": UNIT, "b": UNIT}, ccf_groups=[beta_group(0.2)]
    )
    with pytest.raises(ValueError, match="ccf_groups="):
        PhasedMission([("only", 10.0, grouped)])
    with pytest.raises(ValueError, match="ccf_groups="):
        PhasedMission.from_diagram(grouped, [MissionPhase("only", 10.0)])


@pytest.mark.parametrize(
    "setting",
    [
        "standby",
        "guaranteed",
        BlockSetting(model=surv.Weibull.from_params([40, 2])),
    ],
    ids=["standby", "guaranteed", "model"],
)
def test_a_member_keeps_its_nodes_life_in_every_phase(setting):
    with pytest.raises(ValueError, match="member of a common-cause group"):
        PhasedMission.from_diagram(
            PAIR,
            [MissionPhase("up", 30.0, blocks={"a": setting})],
            ccf_groups=[beta_group(0.2)],
        )


def test_a_member_that_another_node_repeats_is_struck_in_each_place():
    # b repeats a: one component in two places, a cause after each.
    edges = [("s", "a"), ("a", "c"), ("c", "t"), ("s", "d"), ("d", "b")]
    edges.append(("b", "t"))
    c = surv.Weibull.from_params([100, 2])
    diagram = NonRepairableRBD(edges, {"a": UNIT, "b": "a", "c": c, "d": c})
    group = CCFGroup(["a", "c"], BetaFactor(0.25, basis="rate"))
    mission = PhasedMission([("only", 40.0, diagram)], ccf_groups=[group])
    grouped = NonRepairableRBD(
        edges, {"a": UNIT, "b": "a", "c": c, "d": c}, ccf_groups=[group]
    )
    assert mission.reliability() == pytest.approx(
        float(grouped.sf(40.0)), rel=1e-13
    )
    with pytest.raises(ValueError, match="repeats node 'a'"):
        PhasedMission(
            [("only", 40.0, diagram)],
            ccf_groups=[CCFGroup(["b", "c"], BetaFactor(0.25, basis="rate"))],
        )
