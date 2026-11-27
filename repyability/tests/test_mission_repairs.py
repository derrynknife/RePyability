"""Repair during a phased mission's phases (see ``_mission_repairs`` and
``_mission_chain``).

The exact values, from the mission's Markov chain, are checked against
chains written out here by hand, closed forms where a phase is a series,
and ``RepairableRBD.first_failure_sf`` for one phase; the simulation
against the chain, and its order of events at one instant against the
order stated.
"""

import math

import numpy as np
import pytest
import surpyval as surv
from scipy.linalg import expm

from repyability import (
    BetaFactor,
    CCFGroup,
    MissionPhase,
    NonRepairableRBD,
    PhasedMission,
    RepairableRBD,
)
from repyability.rbd import _mission_chain, _mission_repairs, _mission_runs
from repyability.rbd.helper_classes import PerfectReliability
from repyability.tests.catalogue import mission_kinds


def E(rate: float):
    return surv.Exponential.from_params([rate])


LA, LB, MA, MB = 0.01, 0.02, 0.5, 0.25
A, B = E(LA), E(LB)
PARALLEL = [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")]
SERIES = [("s", "a"), ("a", "b"), ("b", "t")]
PAIR = NonRepairableRBD(PARALLEL, {"a": A, "b": B})
BOTH = NonRepairableRBD(SERIES, {"a": A, "b": B})
REPAIRS = {"a": E(MA), "b": E(MB)}
SAMPLES = 50_000


def within_four_errors(mission: PhasedMission, seed: int) -> None:
    """The simulated reliability of ``SAMPLES`` seeded missions against
    the exact one."""
    exact = mission.reliability()
    simulated = mission.reliability(
        method="simulate", mc_samples=SAMPLES, seed=seed
    )
    error = math.sqrt(exact * (1.0 - exact) / SAMPLES)
    assert abs(simulated - exact) < 4.0 * error, (simulated, exact, error)


def pair_chain(repair_a: float = MA, repair_b: float = MB) -> np.ndarray:
    """The generator of the pair while one unit at least is up, its
    states (both up, a down, b down), the rows summing to less than 0 by
    the rate at which the other unit fails too."""
    return np.array(
        [
            [-(LA + LB), LA, LB],
            [repair_a, -(repair_a + LB), 0.0],
            [repair_b, 0.0, -(repair_b + LA)],
        ]
    )


# ----------------------------------------------------------------------
# Exact: the chain
# ----------------------------------------------------------------------


def test_one_phase_is_the_first_failure_of_the_repairable_diagram():
    mission = PhasedMission([("only", 40.0, PAIR)], repairability=REPAIRS)
    diagram = RepairableRBD(
        PARALLEL,
        {
            "a": {"reliability": A, "repairability": REPAIRS["a"]},
            "b": {"reliability": B, "repairability": REPAIRS["b"]},
        },
    )
    assert mission.reliability() == pytest.approx(
        float(diagram.first_failure_sf(40.0)), rel=1e-12
    )
    times = np.array([0.0, 5.0, 20.0, 40.0])
    assert mission.reliability_over_time(times) == pytest.approx(
        np.asarray(diagram.first_failure_sf(times), dtype=float), rel=1e-12
    )
    # Repairs help: the pair without them lasts less often.
    assert mission.reliability() > float(PAIR.sf(40.0))


def test_two_phases_two_units_by_the_chain_written_out():
    # Either for 30, then both for 20: through the first phase in each
    # live state, then both needed at once, and for the rest.
    mission = PhasedMission(
        [("either", 30.0, PAIR), ("both", 20.0, BOTH)], repairability=REPAIRS
    )
    at_30 = np.array([1.0, 0.0, 0.0]) @ expm(pair_chain() * 30.0)
    failing = mission.phase_failure_probabilities()
    assert failing["either"] == pytest.approx(1.0 - at_30.sum(), rel=1e-11)
    # With a unit under repair at 30, the second phase starts down.
    through = at_30[0] * math.exp(-(LA + LB) * 20.0)
    assert failing["both"] == pytest.approx(at_30.sum() - through, rel=1e-11)
    assert mission.reliability() == pytest.approx(through, rel=1e-12)
    assert mission.unreliability() == pytest.approx(1.0 - through, rel=1e-11)
    # Just after the boundary, the mission has lost the states that
    # start the second phase down.
    curve = mission.reliability_over_time([30.0, 40.0])
    assert curve[0] == pytest.approx(at_30[0], rel=1e-12)
    assert curve[1] == pytest.approx(
        at_30[0] * math.exp(-(LA + LB) * 10.0), rel=1e-12
    )


def test_a_series_phase_then_the_pair_in_closed_form():
    # No repair comes before a series phase's failure; the pair then
    # starts new (exponential lives), with its repairs.
    mission = PhasedMission(
        [("both", 10.0, BOTH), ("either", 30.0, PAIR)], repairability=REPAIRS
    )
    pair = (np.array([1.0, 0.0, 0.0]) @ expm(pair_chain() * 30.0)).sum()
    assert mission.reliability() == pytest.approx(
        math.exp(-(LA + LB) * 10.0) * pair, rel=1e-12
    )
    assert mission.phase_failure_probabilities()["both"] == pytest.approx(
        -math.expm1(-(LA + LB) * 10.0), rel=1e-12
    )


def test_a_repair_goes_on_through_a_phase_that_does_not_use_the_block():
    # a alone, then b alone (a under repair or not), then both: a's
    # state at the last phase's start is its availability's, from 0.
    c = E(0.001)
    first = NonRepairableRBD([("s", "a"), ("a", "t")], {"a": A})
    alone = NonRepairableRBD([("s", "c"), ("c", "t")], {"c": c})
    last = NonRepairableRBD(
        [("s", "a"), ("a", "c"), ("c", "t")], {"a": A, "c": c}
    )
    mission = PhasedMission(
        [("a", 10.0, first), ("c", 15.0, alone), ("both", 5.0, last)],
        repairability={"a": E(MA)},
    )
    # a up at 10 without failing; then free to fail and come back to 25,
    # and up to 30. c, whose life runs from the mission's start, to 30.
    total = LA + MA
    up_at = MA / total + LA / total * math.exp(-total * 15.0)
    expected = (
        math.exp(-LA * 10.0)
        * up_at
        * math.exp(-LA * 5.0)
        * math.exp(-0.001 * 30.0)
    )
    assert mission.reliability() == pytest.approx(expected, rel=1e-12)


def test_no_repair_is_the_mission_without():
    plain = PhasedMission([("either", 30.0, PAIR), ("both", 20.0, BOTH)])
    for repairs in (None, {}):
        mission = PhasedMission(
            [("either", 30.0, PAIR), ("both", 20.0, BOTH)],
            repairability=repairs,
        )
        assert mission.repairability == {}
        assert mission.reliability() == plain.reliability()
        assert "repairability" not in mission.to_dict()
    # A repair that never ends is no repair: the chain's values are the
    # decision diagram's.
    slow = PhasedMission(
        [("either", 30.0, PAIR), ("both", 20.0, BOTH)],
        repairability={"a": E(1e-300), "b": E(1e-300)},
    )
    assert slow.reliability() == pytest.approx(plain.reliability(), rel=1e-12)
    assert slow.phase_failure_probabilities()["either"] == pytest.approx(
        plain.phase_failure_probabilities()["either"], rel=1e-11
    )


def test_a_block_that_is_not_repaired_fails_once():
    # Only a is repaired: with b down the pair is a alone.
    mission = PhasedMission([("only", 40.0, PAIR)], repairability={"a": E(MA)})
    # States: both up, a down (b up), b down (a up, for good).
    q = np.array(
        [
            [-(LA + LB), LA, LB],
            [MA, -(MA + LB), 0.0],
            [0.0, 0.0, -LA],
        ]
    )
    live = (np.array([1.0, 0.0, 0.0]) @ expm(q * 40.0)).sum()
    assert mission.reliability() == pytest.approx(live, rel=1e-12)


# ----------------------------------------------------------------------
# The measures, from the chain
# ----------------------------------------------------------------------


def test_the_measures_hold_a_block_in_the_chain():
    mission = PhasedMission(
        [("either", 30.0, PAIR), ("both", 20.0, BOTH)], repairability=REPAIRS
    )
    # a held working: b alone decides, repaired through the first phase
    # (the pair works whatever b does) and needed up at 30, then to 50.
    total = LB + MB
    b_up = MB / total + LB / total * math.exp(-total * 30.0)
    works = b_up * math.exp(-LB * 20.0)
    # a failed from the start, and never repaired: b must last alone to
    # 30, and the second phase then starts down.
    worth = mission.birnbaum_importance()
    assert worth["a"] == pytest.approx(works - 0.0, rel=1e-11)
    drivers = mission.failure_drivers()
    assert drivers.totals["a"] == pytest.approx(
        works - mission.reliability(), rel=1e-10
    )
    for what in (
        mission.fussell_vesely,
        mission.barlow_proschan_importance,
        mission.minimal_cut_sets,
    ):
        with pytest.raises(
            NotImplementedError, match="repaired during its phases"
        ):
            what()


def test_levers_and_sensitivities_take_the_repairs():
    mission = PhasedMission(
        [("either", 30.0, PAIR), ("both", 20.0, BOTH)], repairability=REPAIRS
    )
    levers = [(lever.key, lever.name) for lever in mission.levers()]
    assert levers == [
        ("a", "failure_rate"),
        ("a", "repairability.failure_rate"),
        ("b", "failure_rate"),
        ("b", "repairability.failure_rate"),
        (None, "phase.either.duration"),
        (None, "phase.both.duration"),
    ]
    sens = mission.parameter_sensitivity()
    assert [
        (key, name) for key, names in sens.items() for name in names
    ] == levers
    name = "repairability.failure_rate"
    assert sens["a"][name] > 0.0  # a faster repair helps
    step = 1e-6
    high = mission.with_levers({("a", name): MA + step})
    low = mission.with_levers({("a", name): MA - step})
    assert sens["a"][name] == pytest.approx(
        (high.reliability() - low.reliability()) / (2 * step), rel=1e-5
    )
    assert high.repairability["a"].params[0] == pytest.approx(MA + step)


@pytest.mark.parametrize("drawn", [False, True], ids=["phases", "diagram"])
def test_a_mission_with_repairs_saves_and_loads(drawn):
    if drawn:
        mission = PhasedMission.from_diagram(
            PAIR,
            [MissionPhase("up", 30.0), MissionPhase("on", 20.0)],
            repairability=REPAIRS,
        )
    else:
        mission = PhasedMission(
            [("either", 30.0, PAIR), ("both", 20.0, BOTH)],
            repairability=REPAIRS,
        )
    again = PhasedMission.from_json(mission.to_json())
    assert again.reliability() == mission.reliability()
    assert list(again.repairability) == ["a", "b"]


# ----------------------------------------------------------------------
# The simulation against the chain
# ----------------------------------------------------------------------


def three_phases() -> PhasedMission:
    """Two of three, then all three, then one, with a unit that is not
    repaired."""
    edges = [("s", n) for n in "abc"] + [(n, "v") for n in "abc"]
    edges.append(("v", "t"))
    models = {"a": A, "b": B, "c": E(0.015), "v": PerfectReliability}
    diagram = NonRepairableRBD(edges, models, k={"v": 2})
    return PhasedMission.from_diagram(
        diagram,
        [
            MissionPhase("out", 30.0),
            MissionPhase("pass", 2.0, k={"v": 3}),
            MissionPhase("back", 30.0, k={"v": 1}),
        ],
        repairability=REPAIRS,
    )


@pytest.mark.parametrize("seed", [1, 2])
def test_the_simulation_agrees_with_the_chain(seed):
    mission = PhasedMission(
        [("either", 30.0, PAIR), ("both", 5.0, BOTH), ("back", 20.0, PAIR)],
        repairability=REPAIRS,
    )
    within_four_errors(mission, seed)


def test_the_simulation_agrees_with_the_chain_phase_by_phase():
    mission = three_phases()
    exact = mission.phase_failure_probabilities()
    with np.errstate(all="ignore"):
        played = mission._simulated(SAMPLES, 3)
    for j, phase in enumerate(mission.phases):
        p = exact[phase.name]
        error = math.sqrt(p * (1.0 - p) / SAMPLES)
        assert abs(np.mean(played == j) - p) < 4.0 * error, phase.name
    within_four_errors(mission, 4)


def test_simulated_drivers_and_times_agree_with_the_chain():
    mission = three_phases()
    exact = mission.failure_drivers()
    played = mission.failure_drivers(
        method="simulate", mc_samples=SAMPLES, seed=5
    )
    for node, total in exact.totals.items():
        assert (
            abs(played.totals[node] - total)
            < 4.0 * played.standard_errors[node]
        )
    time = mission.expected_mission_time()
    runs = mission._simulated_runs(SAMPLES, 6)
    ran = np.minimum(runs.time, mission.duration)
    assert abs(ran.mean() - time) < 4.0 * ran.std() / math.sqrt(SAMPLES)


def test_antithetic_and_paired_runs_take_the_repairs():
    mission = three_phases()
    exact = mission.reliability()
    interval = mission.reliability_interval(SAMPLES, seed=7, antithetic=True)
    assert abs(interval.estimate - exact) < 4.0 * interval.standard_error
    slower = mission.with_levers({("a", "repairability.failure_rate"): 0.05})
    difference = mission.compare(
        slower, mc_samples=SAMPLES, seed=8, method="simulate"
    )
    exactly = mission.reliability() - slower.reliability()
    assert abs(difference.estimate - exactly) < 4.0 * difference.standard_error
    assert mission.compare(slower).estimate == pytest.approx(exactly)


def test_other_lives_are_simulated_and_the_routes_say_so():
    unit = surv.Weibull.from_params([60, 1.5])
    pair = NonRepairableRBD(PARALLEL, {"a": unit, "b": unit})
    both = NonRepairableRBD(SERIES, {"a": unit, "b": unit})
    mission = PhasedMission(
        [("both", 10.0, both), ("either", 30.0, pair)],
        repairability={"a": surv.LogNormal.from_params([1.0, 0.5])},
    )
    report = mission.analysis_routes()
    assert report["reliability"].route == "refused"
    assert "method='simulate'" in report["reliability"].reason
    assert report["summary"].route == "simulated"
    assert report["reliability_interval"].route == "simulated"
    with pytest.raises(NotImplementedError, match="constant failure rate"):
        mission.reliability()
    # Through the series phase no repair matters: its failures are the
    # diagram's.
    played = mission._simulated(SAMPLES, 9)
    p = 1.0 - float(both.sf(10.0))
    error = math.sqrt(p * (1.0 - p) / SAMPLES)
    assert abs(np.mean(played == 0) - p) < 4.0 * error
    # And repairs help the pair that follows.
    without = PhasedMission([("both", 10.0, both), ("either", 30.0, pair)])
    assert np.mean(played == 2) > without.reliability()
    summary = mission.summary(mc_samples=2000, seed=1)
    assert summary.method == "simulated"


def test_an_exponential_life_with_a_repair_of_another_shape_is_simulated():
    mission = PhasedMission(
        [("only", 40.0, PAIR)],
        repairability={"a": surv.Weibull.from_params([2.0, 1.5])},
    )
    with pytest.raises(NotImplementedError, match="repair.*exponential"):
        mission.reliability()
    # Between no repair and an exponential repair of a tenth of its mean
    # time.
    played = mission.reliability(
        method="simulate", mc_samples=SAMPLES, seed=10
    )
    mean = float(mission.repairability["a"].mean())
    faster = PhasedMission(
        [("only", 40.0, PAIR)], repairability={"a": E(10.0 / mean)}
    )
    assert float(PAIR.sf(40.0)) < played < faster.reliability()


def test_too_many_states_are_refused_naming_the_simulation(monkeypatch):
    mission = three_phases()
    monkeypatch.setattr(_mission_chain, "MAX_STATES", 3)
    with pytest.raises(NotImplementedError, match="MAX_STATES.*simulate"):
        mission.reliability()


# ----------------------------------------------------------------------
# The order of events at one instant
# ----------------------------------------------------------------------


class Given:
    """Lives and repairs as given: each block's, in order, the same in
    every mission."""

    def __init__(self, lives, repairs, size=1):
        self.lives, self.repairs, self.size = lives, repairs, size

    def life(self, node, k):
        return np.full(self.size, self.lives[node][k])

    def repair(self, node, k):
        return np.full(self.size, self.repairs[node][k])


def played(mission, lives, repairs):
    """One mission with each block's lives (its first, then those after
    its repairs) and repairs as given: the phase it fails in (``None``
    for none) and when."""
    first = {node: np.array([life[0]]) for node, life in lives.items()}
    runs = _mission_repairs.play(mission, first, 1, Given(lives, repairs))
    phase = int(runs.phase[0])
    return (
        None if phase == len(mission.phases) else mission.phases[phase].name
    ), float(runs.time[0])


FOREVER = math.inf


def test_a_block_back_exactly_at_a_phases_start_is_back_too_late():
    # Either to 10, then both: a fails at 4.
    mission = PhasedMission(
        [("either", 10.0, PAIR), ("both", 10.0, BOTH)],
        repairability={"a": E(MA)},
    )
    lives = {"a": [4.0, FOREVER], "b": [FOREVER]}
    # Back at 10, the instant the second phase reads its diagram: failed.
    assert played(mission, lives, {"a": [6.0, 1.0]}) == ("both", 10.0)
    # Back just before: through.
    assert played(mission, lives, {"a": [5.5, 1.0]}) == (None, FOREVER)
    # Back after: the phase started down.
    assert played(mission, lives, {"a": [7.0, 1.0]}) == ("both", 10.0)


def test_a_failure_comes_before_a_restoration_at_the_same_instant():
    mission = PhasedMission(
        [("either", 20.0, PAIR)], repairability={"a": E(MA)}
    )
    # a fails at 2 and is back at 5, when b fails: the pair is down at 5.
    lives = {"a": [2.0, FOREVER], "b": [5.0]}
    assert played(mission, lives, {"a": [3.0, 1.0]}) == ("either", 5.0)
    # Back before b fails, a carries the pair.
    assert played(mission, lives, {"a": [2.5, 1.0]}) == (None, FOREVER)


def test_a_failure_at_a_phases_end_is_in_the_phase():
    mission = PhasedMission(
        [("both", 10.0, BOTH), ("either", 10.0, PAIR)],
        repairability={"a": E(MA)},
    )
    # a fails as the first phase ends: the phase fails, as it does
    # without repairs.
    lives = {"a": [10.0, FOREVER], "b": [FOREVER]}
    assert played(mission, lives, {"a": [1.0, 1.0]}) == ("both", 10.0)
    plain = PhasedMission([("both", 10.0, BOTH), ("either", 10.0, PAIR)])
    runs = _mission_runs.play(
        plain, {"a": np.array([10.0]), "b": np.array([np.inf])}, 1
    )
    assert int(runs.phase[0]) == 0


def test_a_block_fails_and_comes_back_more_than_once():
    mission = PhasedMission(
        [("either", 30.0, PAIR), ("both", 5.0, BOTH)],
        repairability=REPAIRS,
    )
    # a: down 3-4, 9-10, then up; b: down 5-7, then fails at 32.
    lives = {"a": [3.0, 5.0, FOREVER], "b": [5.0, 25.0, FOREVER]}
    repairs = {"a": [1.0, 1.0, 1.0], "b": [2.0, 1.0]}
    assert played(mission, lives, repairs) == ("both", 32.0)
    # b down 5-7 and a down again from 6: the pair is down at 6.
    lives = {"a": [3.0, 2.0, FOREVER], "b": [5.0, 25.0, FOREVER]}
    assert played(mission, lives, repairs) == ("either", 6.0)


def test_a_zero_length_phase_reads_its_diagram_at_that_instant():
    mission = PhasedMission(
        [("either", 10.0, PAIR), ("check", 0.0, BOTH), ("on", 10.0, PAIR)],
        repairability=REPAIRS,
    )
    at_10 = np.array([1.0, 0.0, 0.0]) @ expm(pair_chain() * 10.0)
    failing = mission.phase_failure_probabilities()
    assert failing["check"] == pytest.approx(at_10[1] + at_10[2], rel=1e-11)
    lives = {"a": [4.0, FOREVER], "b": [FOREVER]}
    assert played(mission, lives, {"a": [7.0, 1.0], "b": [1.0]}) == (
        "check",
        10.0,
    )


# ----------------------------------------------------------------------
# Missions without repair, as before
# ----------------------------------------------------------------------

#: Seeded simulations of the catalogue's missions that have no repair
#: and no group, as the simulation gave them before repairs: the number
#: of missions, of 2,000 with seed 11, that fail in each phase or none.
BEFORE = {
    "models": ([499, 829, 672], "0x1.f898a8b7d72b0p+14"),
    "settings": ([1280, 203, 517], "0x1.162833e4a7480p+14"),
    "standby": ([353, 466, 1181], "0x1.30b9b6b47e632p+14"),
    "standby_exact": ([457, 1072, 471], "0x1.2baef424f0410p+15"),
    "standby_models": ([453, 445, 1102], "0x1.35eb0cb94823bp+14"),
}


@pytest.mark.parametrize("name", sorted(BEFORE))
def test_missions_without_repairs_play_as_before(name):
    mission = mission_kinds()[name]
    runs = mission._simulated_runs(2000, 11)
    counts = np.bincount(runs.phase, minlength=len(mission.phases) + 1)
    assert counts.tolist() == BEFORE[name][0]
    finite = runs.time[np.isfinite(runs.time)]
    assert float(finite.sum()).hex() == BEFORE[name][1]


# ----------------------------------------------------------------------
# Refusals
# ----------------------------------------------------------------------


def test_repairability_is_checked():
    phases = [("either", 30.0, PAIR)]
    with pytest.raises(ValueError, match="must be a dict"):
        PhasedMission(phases, repairability=[("a", E(MA))])
    with pytest.raises(ValueError, match="not a component of the mission"):
        PhasedMission(phases, repairability={"z": E(MA)})
    with pytest.raises(TypeError, match="distribution itself"):
        PhasedMission(phases, repairability={"a": surv.Exponential})
    with pytest.raises(TypeError, match="model of its repair times"):
        PhasedMission(phases, repairability={"a": 2.0})
    fixed = NonRepairableRBD(
        PARALLEL, {"a": surv.FixedEventProbability.from_params(0.1), "b": B}
    )
    with pytest.raises(ValueError, match="probability of failing"):
        PhasedMission([("only", 1.0, fixed)], repairability={"a": E(MA)})


def test_a_repaired_block_is_in_no_group():
    unit = surv.Weibull.from_params([100, 2])
    pair = NonRepairableRBD(PARALLEL, {"a": unit, "b": unit})
    group = CCFGroup(["a", "b"], BetaFactor(0.1, basis="rate"))
    with pytest.raises(ValueError, match="member of a common-cause group"):
        PhasedMission(
            [("only", 10.0, pair)],
            ccf_groups=[group],
            repairability={"a": E(MA)},
        )


def test_repairs_take_no_block_on_standby_and_plain_repaired_blocks():
    with pytest.raises(ValueError, match="no block on standby"):
        PhasedMission.from_diagram(
            PAIR,
            [MissionPhase("up", 30.0, blocks={"b": "standby"})],
            repairability={"a": E(MA)},
        )
    with pytest.raises(ValueError, match="which is repaired"):
        PhasedMission.from_diagram(
            PAIR,
            [MissionPhase("up", 30.0, blocks={"a": "guaranteed"})],
            repairability={"a": E(MA)},
        )
    # Another block may be set, and a repaired one not needed.
    mission = PhasedMission.from_diagram(
        BOTH,
        [
            MissionPhase("up", 30.0, blocks={"b": "guaranteed"}),
            MissionPhase("on", 20.0, blocks={"a": "not_needed"}),
        ],
        repairability={"a": E(MA)},
    )
    assert mission.reliability() == pytest.approx(
        math.exp(-LA * 30.0) * math.exp(-LB * 20.0), rel=1e-12
    )
    within_four_errors(mission, 12)
