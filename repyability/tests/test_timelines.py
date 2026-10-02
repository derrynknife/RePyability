"""Timelines (``repyability.timelines``): up/down histories, their
measures, merges as a diagram's structure, and
``RepairableRBD.simulate_timelines``.

The merges are checked against a replay that takes every change in turn,
in the order the merges take them (by time, then by the inputs' order and
each input's own order), and decides the unit after each; the simulated
timelines against ``availability``, which runs the same simulations.
"""

import dataclasses
import warnings

import numpy as np
import pytest
import surpyval as surv

from repyability import (
    RBD,
    BetaFactor,
    CCFGroup,
    NonRepairableRBD,
    PerfectReliability,
    RepairableRBD,
    Timeline,
    Timelines,
    TimelineSimulation,
)
from repyability import timelines as tl
from repyability.rbd import _timeline_runs, modular
from repyability.timelines import k_out_of_n, parallel, series

W = surv.Weibull.from_params
E = surv.Exponential.from_params
L = surv.LogNormal.from_params
X = surv.ExactEventTime.from_params
BRIDGE = [
    ("s", "a"),
    ("s", "b"),
    ("a", "c"),
    ("b", "c"),
    ("a", "d"),
    ("c", "e"),
    ("b", "e"),
    ("d", "t"),
    ("e", "t"),
]
PAR3 = [("s", "a"), ("s", "b"), ("s", "c"), ("a", "t"), ("b", "t"), ("c", "t")]


def random_timeline(rng, end=15.0, grid=True, planned=False, name=None):
    """A random history: on a grid of whole numbers (so with changes of
    several inputs at one time, and blips) or anywhere."""
    n = int(rng.integers(0, 7))
    if grid:
        times = np.sort(rng.integers(0, int(end), n)).astype(float)
    else:
        times = np.sort(rng.uniform(0.0, end, n))
    up = bool(rng.integers(0, 2))
    flags = None
    if planned:
        # Only a change down can be planned.
        down = (np.arange(n) % 2) == (0 if up else 1)
        flags = down & (rng.random(n) < 0.4)
    return Timeline(times, end, up=up, planned=flags, name=name)


def replay(inputs, decide):
    """The unit decided by ``decide`` (a function of the inputs' states)
    from ``inputs`` ({label: Timeline}), each change taken in turn: by
    time, then by the inputs' order and each one's own order. Its start,
    and each change's time, cause and whether it is planned."""
    labels = list(inputs)
    events = sorted(
        (t, rank, j, label)
        for rank, label in enumerate(labels)
        for j, t in enumerate(inputs[label].changes.tolist())
    )
    state = {label: inputs[label].up for label in labels}
    start = up = decide(state)
    out = []
    for t, _, j, label in events:
        state[label] = not state[label]
        now = decide(state)
        if now != up:
            planned = bool(inputs[label].planned[j]) and not now
            out.append((t, label, planned))
            up = now
    return start, out


def as_replayed(timeline):
    return timeline.up, [
        (t, c, p)
        for t, c, p in zip(
            timeline.changes.tolist(),
            timeline.causes,
            timeline.planned.tolist(),
        )
    ]


# A timeline and its measures


def test_a_timeline_is_its_changes_over_a_window():
    unit = Timeline([1.0, 3.0, 5.0], end=10.0, name="pump")
    assert unit.up and unit.end == 10.0 and unit.name == "pump"
    assert unit.changes.tolist() == [1.0, 3.0, 5.0] and len(unit) == 3
    assert unit.uptime == 3.0 and unit.downtime == 7.0
    assert unit.availability == 0.3
    assert (unit.failures, unit.restorations, unit.planned_outages) == (
        2,
        1,
        0,
    )
    assert unit.first_failure == 1.0
    assert unit.up_intervals.tolist() == [[0.0, 1.0], [3.0, 5.0]]
    assert unit.down_intervals.tolist() == [[1.0, 3.0], [5.0, 10.0]]
    assert unit.state(0.5) is True and unit.state(1.0) is False
    assert unit.state([0.0, 2.0, 4.0, 9.0]).tolist() == [
        True,
        False,
        True,
        False,
    ]
    assert unit.causes == ["pump"] * 3
    assert unit.failures_by_cause() == {"pump": 2}
    assert unit.restorations_by_cause() == {"pump": 1}
    assert repr(unit) == "Timeline([1, 3, 5], end=10, up=True, name='pump')"


def test_a_timeline_can_start_down():
    unit = Timeline([4.0], end=10.0, up=False)
    assert not unit.up and unit.uptime == 6.0
    assert unit.failures == 0 and unit.restorations == 1
    assert unit.first_failure == np.inf


def test_a_change_at_the_end_is_outside_the_window():
    assert Timeline([2.0, 10.0], end=10.0).changes.tolist() == [2.0]


def test_two_changes_at_once_are_an_instant_repair():
    unit = Timeline([5.0, 5.0], end=10.0)
    assert unit.uptime == 10.0 and unit.failures == 1
    assert unit.restorations == 1
    assert unit.down_intervals.tolist() == [[5.0, 5.0]]
    assert unit.state(5.0) is True


def test_planned_outages_are_counted_apart_from_failures():
    unit = Timeline(
        [1.0, 2.0, 5.0, 6.0], end=10.0, planned=[True, False, False, False]
    )
    assert unit.failures == 1 and unit.planned_outages == 1
    assert unit.first_failure == 5.0
    assert unit.failures_by_cause() == {None: 1}


@pytest.mark.parametrize(
    "changes, end, options, error, match",
    [
        ([3.0, 1.0], 10.0, {}, ValueError, "increasing"),
        ([-1.0], 10.0, {}, ValueError, r"\[0, end\]"),
        ([11.0], 10.0, {}, ValueError, r"\[0, end\]"),
        ([np.nan], 10.0, {}, ValueError, "finite"),
        ([np.inf], 10.0, {}, ValueError, "finite"),
        ([], 0.0, {}, ValueError, "positive"),
        ([], -1.0, {}, ValueError, "positive"),
        ([], np.inf, {}, ValueError, "positive"),
        ([], np.nan, {}, ValueError, "positive"),
        ([], "10", {}, TypeError, "number"),
        ([], True, {}, TypeError, "number"),
        ([1.0, 2.0], 10.0, {"planned": [True]}, ValueError, "one for each"),
        ([1.0, 2.0], 10.0, {"planned": [False, True]}, ValueError, "down"),
        ([1.0], 10.0, {"up": False, "planned": [True]}, ValueError, "down"),
    ],
)
def test_a_timeline_checks_its_changes(changes, end, options, error, match):
    with pytest.raises(error, match=match):
        Timeline(changes, end, **options)


def test_a_timeline_from_an_outage_log():
    log = Timeline.from_outages(
        [(1.0, 2.0), (5.0, 6.0), (8.0, None)],
        end=10.0,
        planned=[False, True, False],
        name="pump",
    )
    assert log.changes.tolist() == [1.0, 2.0, 5.0, 6.0, 8.0]
    assert log.planned.tolist() == [False, False, True, False, False]
    assert log.downtime == 4.0 and log.name == "pump"
    assert (log.failures, log.planned_outages, log.restorations) == (2, 1, 2)
    # An outage running past the end runs to it; one at 0 starts down.
    assert Timeline.from_outages([(4.0, 20.0)], 10.0).downtime == 6.0
    first = Timeline.from_outages([(0.0, 2.0)], 10.0)
    assert first.up and first.changes.tolist() == [0.0, 2.0]
    assert first.state(0.0) is False and first.uptime == 8.0


@pytest.mark.parametrize(
    "outages, options, match",
    [
        ([(2.0, 1.0)], {}, "out of order"),
        ([(1.0, 3.0), (2.0, 4.0)], {}, "out of order"),
        ([(11.0, 12.0)], {}, "after the window's end"),
        ([(1.0, 2.0)], {"planned": [True, False]}, "one for each outage"),
    ],
)
def test_an_outage_log_is_checked(outages, options, match):
    with pytest.raises(ValueError, match=match):
        Timeline.from_outages(outages, 10.0, **options)


def test_a_timeline_from_durations_up_and_down_in_turn():
    unit = Timeline.from_durations([10, 20, 30], [1, 2], end=50)
    assert unit.changes.tolist() == [10.0, 11.0, 31.0, 33.0]
    # Each duration ends in a change: a life in a failure, a repair in a
    # restoration.
    down_first = Timeline.from_durations([5.0], [3.0], end=50, up=False)
    assert not down_first.up and down_first.changes.tolist() == [3.0, 8.0]
    # Durations that run out leave the unit as it is; inf is for ever.
    forever = Timeline.from_durations([4.0, np.inf], [1.0, 1.0], end=50)
    assert forever.changes.tolist() == [4.0, 5.0]
    assert Timeline.from_durations([60.0], [1.0], end=50).changes.size == 0
    for bad in ([-1.0], [np.nan]):
        with pytest.raises(ValueError, match="non-negative"):
            Timeline.from_durations(bad, [1.0], end=50)


def test_timelines_compare_by_history_cause_and_name():
    a = Timeline([1.0, 2.0], 10.0, name="a")
    assert a == Timeline([1.0, 2.0], 10.0, name="a")
    assert a != Timeline([1.0, 2.0], 10.0, name="b")
    assert a != Timeline([1.0, 3.0], 10.0, name="a")
    assert a != Timeline([1.0, 2.0], 12.0, name="a")
    with pytest.raises(TypeError):
        hash(a)


# Many histories at once


def test_timelines_measure_each_history():
    rng = np.random.default_rng(3)
    histories = [
        random_timeline(rng, grid=bool(i % 2), planned=True) for i in range(40)
    ]
    many = Timelines(histories, name="pump")
    assert len(many) == 40 and many.name == "pump" and many.end == 15.0
    for measure in (
        "uptime",
        "downtime",
        "availability",
        "failures",
        "planned_outages",
        "restorations",
        "first_failure",
    ):
        assert getattr(many, measure).tolist() == [
            getattr(h, measure) for h in histories
        ], measure
    assert many.up.tolist() == [h.up for h in histories]
    times = [0.0, 3.0, 7.5, 14.0]
    assert np.array_equal(
        many.state(times), np.array([h.state(times) for h in histories])
    )
    assert many.state(7.5).tolist() == [h.state(7.5) for h in histories]
    for s, history in enumerate(histories):
        assert many[s] == Timeline(
            history.changes,
            15.0,
            up=history.up,
            planned=history.planned,
            name="pump",
        )


def test_the_fraction_up_over_time():
    rng = np.random.default_rng(4)
    histories = [random_timeline(rng, grid=False) for _ in range(30)]
    many = Timelines(histories)
    time, fraction = many.availability_curve()
    assert time[0] == 0.0 and time[-1] == 15.0
    for t in np.linspace(0.0, 14.9, 60):
        expected = np.mean([h.state(t) for h in histories])
        assert many.point_availability(t) == pytest.approx(expected)
    # Between changes the fraction is constant, and the curve's mean over
    # the window is the histories' mean availability.
    spans = np.diff(time)
    assert np.sum(fraction[:-1] * spans) / 15.0 == pytest.approx(
        many.availability.mean()
    )
    assert many.point_availability([1.0, 2.0]).shape == (2,)


def test_timelines_index_slice_and_iterate():
    histories = [
        Timeline([1.0], 10.0),
        Timeline([2.0, 3.0], 10.0),
        Timeline([], 10.0, up=False),
    ]
    many = Timelines(histories)
    assert many[-1] == histories[2] and many[1] == histories[1]
    assert list(many) == histories
    assert many[1:] == Timelines(histories[1:])
    assert len(many[::2]) == 2
    with pytest.raises(IndexError):
        many[3]
    with pytest.raises(TypeError):
        many[1.0]
    assert repr(many) == "Timelines(3 histories, end=10, 3 changes)"


@pytest.mark.parametrize(
    "items, match",
    [
        ([], "one or more"),
        (["not a timeline"], "one or more"),
        ([Timeline([], 10.0), Timeline([], 12.0)], "one window"),
    ],
)
def test_timelines_need_timelines_over_one_window(items, match):
    with pytest.raises(ValueError, match=match):
        Timelines(items)


# Merges


@pytest.mark.parametrize("grid", [True, False])
def test_k_out_of_n_is_the_changes_replayed_in_turn(grid):
    rng = np.random.default_rng(11 if grid else 12)
    for _ in range(400):
        n = int(rng.integers(1, 6))
        inputs = {
            f"u{i}": random_timeline(rng, grid=grid, planned=True)
            for i in range(n)
        }
        k = int(rng.integers(1, n + 1))
        merged = k_out_of_n(k, inputs)
        expected = replay(inputs, lambda state: sum(state.values()) >= k)
        assert as_replayed(merged) == expected, (k, inputs)
        if k == n:
            assert series(inputs) == merged
        if k == 1:
            assert parallel(inputs) == merged


def test_one_by_one_inputs_keep_their_causes():
    a = Timeline([1.0, 4.0], 10.0, name="a")
    b = Timeline([2.0, 3.0], 10.0, name="b")
    c = Timeline([5.0, 6.0], 10.0, name="c")
    d = Timeline([3.0, 5.0], 10.0, name="d")
    pair = parallel(a, b)
    assert pair.causes == ["b", "b"] and pair.uptime == 9.0
    trio = series(a, b, c, name="line")
    assert trio.causes == ["a", "a", "c", "c"]
    assert trio.name == "line"
    assert trio.failures_by_cause() == {"a": 1, "b": 0, "c": 1}
    # Merged again, a merged timeline's changes keep their components.
    assert series(a | b, c).causes == ["b", "b", "c", "c"]
    assert series(parallel(a, c), b).causes == ["b", "b"]
    assert parallel(series(a, b), d).causes == ["d", "a"]
    # Given as a mapping, each input is a cause of its own.
    assert series({"left": a | b, "right": c}).causes == [
        "left",
        "left",
        "right",
        "right",
    ]


def test_merges_of_many_histories_merge_each():
    rng = np.random.default_rng(5)
    many = {
        f"u{i}": Timelines(
            [random_timeline(rng, planned=True) for _ in range(25)]
        )
        for i in range(3)
    }
    one = random_timeline(rng, name="shared")
    merged = k_out_of_n(2, *many.values(), one)
    assert isinstance(merged, Timelines) and len(merged) == 25
    for s in range(25):
        assert merged[s] == k_out_of_n(2, *(m[s] for m in many.values()), one)
    single = series(*(m[0] for m in many.values()))
    assert isinstance(single, Timeline)


def test_operators_merge_and_complement():
    a = Timeline([1.0, 4.0], 10.0, planned=[True, False], name="a")
    b = Timeline([2.0, 6.0], 10.0, name="b")
    assert a & b == series(a, b) and a | b == parallel(a, b)
    many = Timelines([a, a])
    assert many & b == series(many, b) and many | b == parallel(many, b)
    off = ~a
    assert not off.up and off.changes.tolist() == [1.0, 4.0]
    assert off.planned.tolist() == [False, False] and off.name == "a"
    assert (~many)[0] == off
    # A planned change down that takes the merge down is planned there too.
    line = a & b
    assert line.changes.tolist() == [1.0, 6.0] and line.causes == ["a", "b"]
    assert line.planned.tolist() == [True, False]
    assert line.planned_outages == 1 and line.failures == 0


def test_merges_check_their_inputs():
    a = Timeline([1.0], 10.0)
    with pytest.raises(ValueError, match="one window"):
        series(a, Timeline([1.0], 12.0))
    with pytest.raises(ValueError, match="as many histories"):
        series(Timelines([a, a]), Timelines([a, a, a]))
    with pytest.raises(ValueError, match="at least one"):
        parallel()
    with pytest.raises(ValueError, match="at least one"):
        parallel({})
    with pytest.raises(TypeError, match="Timeline or Timelines"):
        series(a, [1.0, 2.0])
    for k in (0, 3, True, 1.5):
        with pytest.raises(ValueError, match="whole number from 1 to"):
            k_out_of_n(k, a, a)


# A diagram's timeline


def structures():
    rep = NonRepairableRBD(
        [
            ("s", "x"),
            ("x", "a"),
            ("a", "t"),
            ("s", "b"),
            ("b", "a2"),
            ("a2", "t"),
        ],
        {"x": E([1.0]), "a": E([1.0]), "b": E([1.0]), "a2": "a"},
    )
    return {
        "series and parallel": RBD(
            [("s", "a"), ("s", "b"), ("a", "c"), ("b", "c"), ("c", "t")]
        ),
        "bridge": RBD(BRIDGE),
        "vote": RBD(
            [("s", u) for u in "abc"]
            + [(u, "v") for u in "abc"]
            + [("v", "t")],
            k={"v": 2},
        ),
        "bridge into a vote": RBD(
            BRIDGE[:-2]
            + [("d", "f"), ("e", "f"), ("f", "g"), ("f", "h"), ("f", "i")]
            + [("g", "w"), ("h", "w"), ("i", "w"), ("w", "t")],
            k={"w": 2},
        ),
        "repeated node": rep,
    }


def replayed_system(rbd, inputs):
    """The system replayed from its components' timelines: each change in
    turn, in the order of the diagram's nodes."""
    ordered = {node: inputs[node] for node in rbd.nodes if node in inputs}
    rest = {node: True for node in rbd.nodes if node not in inputs}

    def decide(state):
        return rbd.is_system_working({**rest, **state}, "p")

    return replay(ordered, decide)


@pytest.mark.parametrize("name", sorted(structures()))
@pytest.mark.parametrize("path_members", [tl._PATH_MEMBERS, 0])
def test_a_systems_timeline_is_its_components_replayed(
    name, path_members, monkeypatch
):
    # With no path members allowed, a core is decided at each of its
    # members' changes rather than merged path set by path set.
    monkeypatch.setattr(tl, "_PATH_MEMBERS", path_members)
    rbd = structures()[name]
    relevant = rbd._decomposition().nodes
    rng = np.random.default_rng(len(name) + path_members)
    for trial in range(150):
        inputs = {
            node: random_timeline(rng, grid=trial % 2 == 0, planned=True)
            for node in rbd.nodes
            if node in relevant
        }
        system = rbd.system_timeline(inputs)
        assert as_replayed(system) == replayed_system(rbd, inputs), inputs
        assert system.name is None


def test_a_core_by_its_decision_diagram(monkeypatch):
    monkeypatch.setattr(modular, "CORE_METHOD", "bdd")
    rbd = RBD(BRIDGE)
    assert rbd._decomposition().from_graph
    rng = np.random.default_rng(8)
    for trial in range(150):
        inputs = {
            node: random_timeline(rng, grid=trial % 2 == 0, planned=True)
            for node in rbd.nodes
        }
        system = rbd.system_timeline(inputs)
        assert as_replayed(system) == replayed_system(rbd, inputs)


def test_a_systems_timelines_merge_each_history():
    rng = np.random.default_rng(9)
    rbd = RBD(BRIDGE)
    inputs = {
        node: Timelines([random_timeline(rng) for _ in range(20)])
        for node in "abcd"
    }
    inputs["e"] = random_timeline(rng)
    many = rbd.system_timeline(inputs)
    assert isinstance(many, Timelines) and len(many) == 20
    for s in range(20):
        one = {
            n: (t[s] if isinstance(t, Timelines) else t)
            for n, t in inputs.items()
        }
        assert many[s] == rbd.system_timeline(one)
    assert set(many.failures_by_cause()) == set("abcde")


def test_a_systems_timeline_counts_its_failures_by_component():
    rbd = RBD([("s", "a"), ("s", "b"), ("a", "v"), ("b", "v"), ("v", "t")])
    plant = rbd.system_timeline(
        {
            "a": Timeline.from_outages([(10, 30)], end=100),
            "b": Timeline.from_outages([(20, 25), (60, 70)], end=100),
            "v": Timeline.from_outages(
                [(80, 81), (90, 95)], end=100, planned=[False, True]
            ),
        }
    )
    assert plant.down_intervals.tolist() == [
        [20.0, 25.0],
        [80.0, 81.0],
        [90.0, 95.0],
    ]
    assert plant.causes == ["b", "b", "v", "v", "v", "v"]
    assert plant.failures_by_cause() == {"a": 0, "b": 1, "v": 1}
    assert plant.restorations_by_cause() == {"a": 0, "b": 1, "v": 2}
    assert plant.planned_outages == 1


def test_junctions_and_irrelevant_components_need_no_timeline():
    vote = NonRepairableRBD(
        [("s", u) for u in "abc"] + [(u, "v") for u in "abc"] + [("v", "t")],
        {"a": E([1.0]), "b": E([1.0]), "c": E([1.0]), "v": PerfectReliability},
        k={"v": 2},
    )
    units = {
        "a": Timeline([1.0, 5.0], 10.0),
        "b": Timeline([2.0, 3.0], 10.0),
        "c": Timeline([], 10.0),
    }
    assert vote.system_timeline(units).down_intervals.tolist() == [[2.0, 3.0]]
    bypassed = RBD([("s", "a"), ("a", "t"), ("a", "b"), ("b", "t")])
    alone = bypassed.system_timeline({"a": Timeline([4.0], 10.0)})
    with_b = bypassed.system_timeline(
        {"a": Timeline([4.0], 10.0), "b": Timeline([1.0], 10.0)}
    )
    assert alone == with_b and alone.changes.tolist() == [4.0]


def test_a_systems_timeline_checks_its_components():
    rbd = structures()["repeated node"]
    up = Timeline([], 10.0)
    full = {"x": up, "a": up, "b": up}
    with pytest.raises(TypeError, match="mapping"):
        rbd.system_timeline([up, up, up])
    with pytest.raises(ValueError, match="repeats node 'a'"):
        rbd.system_timeline({**full, "a2": up})
    with pytest.raises(ValueError, match="input or output node 's'"):
        rbd.system_timeline({**full, "s": up})
    with pytest.raises(ValueError, match="Unknown node 'z'"):
        rbd.system_timeline({**full, "z": up})
    with pytest.raises(ValueError, match=r"No timeline for node\(s\) \['b'\]"):
        rbd.system_timeline({"x": up, "a": up})
    with pytest.raises(ValueError, match="one window"):
        rbd.system_timeline({**full, "b": Timeline([], 12.0)})


# Simulated timelines


def unit(scale, shape, repair=L([1.0, 0.5]), **extra):
    return {"reliability": W([scale, shape]), "repairability": repair, **extra}


def simulated_systems():
    inner = RepairableRBD(
        [("s", "x"), ("s", "y"), ("x", "t"), ("y", "t")],
        {"x": unit(40, 2.0), "y": unit(45, 2.0)},
    )
    return {
        "pair": RepairableRBD(
            [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
            {"a": unit(50, 1.5), "b": unit(60, 1.5)},
        ),
        "bridge": RepairableRBD(
            BRIDGE, {n: unit(60 + 5 * i, 1.5) for i, n in enumerate("abcde")}
        ),
        "nested": RepairableRBD(
            [("s", "m"), ("s", "c"), ("m", "t"), ("c", "t")],
            {"m": inner, "c": unit(60, 1.5)},
        ),
        "instant repairs": RepairableRBD(
            [("s", "a"), ("a", "b"), ("b", "t")],
            {"a": unit(30, 1.5, "instant"), "b": unit(40, 1.5)},
        ),
        "a repair crew": RepairableRBD(
            PAR3, {n: unit(50, 2.0) for n in "abc"}, repair_crews=1
        ),
        "maintained and tested": RepairableRBD(
            PAR3,
            {
                "a": unit(
                    50,
                    2.0,
                    preventive={"interval": 20.0, "duration": E([1.0])},
                ),
                "b": unit(55, 2.0),
                "c": unit(
                    60, 2.0, inspection={"interval": 15.0, "duration": X(0.5)}
                ),
            },
        ),
        "a standby group": RepairableRBD(
            PAR3,
            {
                "a": unit(50, 2.0),
                "b": unit(55, 2.0, standby={"units": 2}),
                "c": unit(60, 2.0),
            },
        ),
    }


STREAMED = {"pair", "bridge", "nested", "instant repairs"}


@pytest.mark.parametrize("name", sorted(simulated_systems()))
@pytest.mark.parametrize("how", ["plain", "antithetic", "broken", "working"])
def test_simulated_timelines_are_the_simulations_availability_runs(name, how):
    rbd = simulated_systems()[name]
    nodes = list(rbd.components)
    options = {
        "plain": {},
        "antithetic": {"antithetic": True},
        "broken": {"broken_nodes": [nodes[0]]},
        "working": {"working_nodes": [nodes[-1]]},
    }[how]
    runs = rbd.simulate_timelines(300.0, mc_samples=60, seed=3, **options)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = rbd.availability(
            300.0, mc_samples=60, seed=3, engine="python", **options
        )
    assert isinstance(runs, TimelineSimulation)
    assert runs.method == ("streams" if name in STREAMED else "event loop")
    assert runs.n_simulations == 60 and runs.time_simulated_to == 300.0
    assert runs.antithetic == bool(options.get("antithetic"))
    system = runs.system
    assert len(system) == 60 and system.end == 300.0
    # The same simulations, to the last bit of each one's uptime.
    assert np.array_equal(system.uptime, result.uptimes)
    assert system.failures.sum() == result.system_failures
    assert system.restorations.sum() == result.system_restorations
    assert system.planned_outages.sum() == result.system_planned_outages
    time, fraction = system.availability_curve()
    assert np.array_equal(time, result.timeline)
    np.testing.assert_allclose(fraction, result.availability, atol=1e-12)
    assert list(runs.components) == list(rbd.components)
    for node, history in runs.components.items():
        assert history.name == node and len(history) == 60
        assert history.uptime.sum() == pytest.approx(
            result.node_uptime[node], rel=1e-12
        )
    for node in options.get("broken_nodes", []) + options.get(
        "working_nodes", []
    ):
        held = runs.components[node]
        assert np.all(held.up == (how == "working"))
        assert not held.failures.any() and not held.restorations.any()
    # The system's failures, by the component that caused them, are the
    # simulation's own (its failure criticality index).
    by = {n: int(v.sum()) for n, v in system.failures_by_cause().items()}
    total = sum(by.values())
    share = result.criticalities.failure_criticality_index.per_system_failure
    for node in rbd.components:
        mine = by.get(node, 0) / total if total else 0.0
        assert mine == pytest.approx(float(share.get(node, 0.0)), abs=1e-12)


def test_a_nested_components_timeline_is_its_own_systems():
    rbd = simulated_systems()["nested"]
    runs = rbd.simulate_timelines(300.0, mc_samples=40, seed=5)
    nested = runs.components["m"]
    assert set(nested.failures_by_cause()) == {"x", "y"}
    assert runs.system == rbd.system_timeline(runs.components)


def test_simulated_timelines_are_seeded():
    rbd = simulated_systems()["pair"]
    np.random.seed(5)
    before = np.random.get_state()
    first = rbd.simulate_timelines(200.0, mc_samples=30, seed=1)
    after = np.random.get_state()
    assert all(np.array_equal(x, y) for x, y in zip(before, after))
    again = rbd.simulate_timelines(200.0, mc_samples=30, seed=1)
    assert first.system == again.system
    assert all(first.components[n] == again.components[n] for n in "ab")
    other = rbd.simulate_timelines(200.0, mc_samples=30, seed=2)
    assert first.system != other.system
    # A result is a read-only mapping of its fields, like the others.
    assert first["system"] is first.system and "components" in first
    # Unseeded runs draw from numpy's global RNG, as availability's do.
    np.random.seed(7)
    unseeded = rbd.simulate_timelines(200.0, mc_samples=30)
    np.random.seed(7)
    assert (
        unseeded.system == rbd.simulate_timelines(200.0, mc_samples=30).system
    )


def test_simulated_timelines_default_to_a_thousand_simulations():
    rbd = RepairableRBD([("s", "a"), ("a", "t")], {"a": unit(50, 1.5)})
    assert rbd.simulate_timelines(20.0, seed=1).n_simulations == 1000


@pytest.mark.parametrize(
    "arguments, error, match",
    [
        ({"t_simulation": 0.0}, ValueError, "t_simulation"),
        ({"t_simulation": np.inf}, ValueError, "t_simulation"),
        ({"t_simulation": "100"}, ValueError, "t_simulation"),
        ({"t_simulation": True}, ValueError, "t_simulation"),
        ({"mc_samples": 0}, ValueError, "mc_samples"),
        ({"mc_samples": 2.5}, ValueError, "mc_samples"),
        ({"mc_samples": 7, "antithetic": True}, ValueError, "even"),
        ({"broken_nodes": ["z"]}, ValueError, "z"),
        ({"working_nodes": ["a"], "broken_nodes": ["a"]}, ValueError, "both"),
    ],
)
def test_simulated_timelines_check_their_arguments(arguments, error, match):
    rbd = simulated_systems()["pair"]
    call = {"t_simulation": 100.0, "mc_samples": 10, "seed": 1, **arguments}
    with pytest.raises(error, match=match):
        rbd.simulate_timelines(**call)


def test_simulated_timelines_take_no_common_causes_as_yet():
    rbd = RepairableRBD(
        PAR3,
        {
            n: {"reliability": E([0.002]), "repairability": E([0.5])}
            for n in "abc"
        },
        ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.1))],
    )
    with pytest.raises(NotImplementedError, match="common-cause groups"):
        rbd.simulate_timelines(100.0, mc_samples=10, seed=1)


def test_a_component_that_changes_state_without_end_is_refused():
    rbd = RepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        {
            "a": {"reliability": X(0.0), "repairability": "instant"},
            "b": unit(50, 1.5),
        },
    )
    with pytest.raises(ValueError, match="too many to keep as timelines"):
        rbd.simulate_timelines(10.0, mc_samples=20, seed=1)


@pytest.mark.parametrize("batch", [1, 4096])
def test_a_simulation_with_more_changes_than_planned_is_drawn_further(
    batch, monkeypatch
):
    # Planned for a single failure and repair each, nearly every simulation
    # has more, and is drawn further on its own: to the same timelines.
    # (A chunk's rows change only how the draws are computed.)
    rbd = simulated_systems()["pair"]
    plain = rbd.simulate_timelines(300.0, mc_samples=20, seed=4)
    planned = type(rbd)._stream_plan

    def one_row(self, t_simulation, entropy, antithetic):
        plan, complete = planned(self, t_simulation, entropy, antithetic)
        plan.specs = {
            name: dataclasses.replace(spec, rows=1)
            for name, spec in plan.specs.items()
        }
        return plan, complete

    monkeypatch.setattr(type(rbd), "_stream_plan", one_row)
    monkeypatch.setattr(_timeline_runs, "_BATCH", batch)
    again = rbd.simulate_timelines(300.0, mc_samples=20, seed=4)
    assert again.system == plain.system
    assert all(again.components[n] == plain.components[n] for n in "ab")
