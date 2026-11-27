"""Event trees: each sequence one fault tree with NOT gates, exact where the
headings share events, checked against every state of the events
enumerated in exact arithmetic, against the product of the branches
(which differs where events are shared), and against the definitions of
the approximations and the importances."""

import inspect
import itertools
import json
from fractions import Fraction

import numpy as np
import pytest
import surpyval as surv

from repyability import EventTree, FaultTree
from repyability.rbd import routes
from repyability.tests.catalogue import event_tree_kinds
from repyability.tests.test_analysis_routes import behaves_as_reported

E = surv.Exponential.from_params
W = surv.Weibull.from_params
X = np.array([10.0, 50.0])


def heading(top, inputs, events, kind="or"):
    return FaultTree({top: (kind, inputs)}, events)


def shared_tree(sequences=None):
    """The hand-worked tree: a loss of cooling (0.1 a year) through
    detection (a probability), injection and recirculation, whose pumps
    both need the one power supply."""
    return EventTree(
        0.1,
        [
            ("detection", 0.01),
            (
                "injection",
                heading(
                    "no injection",
                    ["pump A", "power"],
                    {"pump A": 0.05, "power": 0.01},
                ),
            ),
            (
                "recirculation",
                heading(
                    "no recirculation",
                    ["pump B", "power"],
                    {"pump B": 0.1, "power": 0.01},
                ),
            ),
        ],
        sequences,
    )


def enumerated(tree: EventTree, rare_event=False):
    """Each path's probability by summing, in exact arithmetic, over every
    state of the events, each heading failed when its tree's top event
    occurs (a probability heading when its own event does)."""
    events = list(tree.events)
    q = {e: Fraction(str(tree.events[e])) for e in events}
    out = {path: Fraction(0) for path in tree.sequences}
    for state in itertools.product((False, True), repeat=len(events)):
        occurred = {e for e, s in zip(events, state) if s}
        weight = Fraction(1)
        for e, s in zip(events, state):
            weight *= q[e] if s else 1 - q[e]
        failed = {}
        for name, model in tree.headings.items():
            if isinstance(model, FaultTree):
                failed[name] = model.occurs(occurred & set(model.events))
            else:
                failed[name] = name in occurred
        for path in tree.sequences:
            if all(
                branch is None
                or (rare_event and branch == "success")
                or failed[name] == (branch == "failure")
                for name, branch in zip(tree.headings, path)
            ):
                out[path] += weight
    return out


def test_independent_headings_give_the_product_of_the_branches():
    tree = event_tree_kinds()["independent headings"]
    exact = tree.sequence_probabilities().probabilities
    product = tree.sequence_probabilities(
        approximation="independent"
    ).probabilities
    q = {
        name: model.top_event_probability()
        for name, model in tree.headings.items()
    }
    for path, value in exact.items():
        by_hand = 1.0
        for name, branch in zip(tree.headings, path):
            by_hand *= q[name] if branch == "failure" else 1.0 - q[name]
        assert value == pytest.approx(by_hand, rel=1e-15, abs=1e-15)
        assert product[path] == pytest.approx(by_hand, rel=1e-15)


@pytest.mark.parametrize("partial", [False, True])
def test_shared_events_are_one_event_in_every_sequence(partial):
    sequences = (
        {
            ("success", "success", "success"): "ok",
            ("success", "success", "failure"): "damage",
            ("success", "failure", None): "damage",
            ("failure", None, None): "damage",
        }
        if partial
        else None
    )
    tree = shared_tree(sequences)
    assert tree.shared_events == {"power"}
    exact = tree.sequence_probabilities().probabilities
    by_hand = enumerated(tree)
    assert sum(by_hand.values()) == 1
    for path, value in exact.items():
        assert value == pytest.approx(float(by_hand[path]), rel=1e-14)


def test_the_product_of_the_branches_is_wrong_where_events_are_shared():
    # The hand-worked sequence: detected, then injection and
    # recirculation both lost. Exactly 0.1 * 0.99 * (0.01 + 0.99 * 0.05 *
    # 0.1) = 1.480050e-3 a year; the product of the branches, 0.1 * 0.99 *
    # 0.0595 * 0.109 = 6.420645e-4, is 57% too low, as one power failure
    # takes out both pumps.
    tree = shared_tree()
    path = ("success", "failure", "failure")
    exact = tree.sequence_frequencies().frequencies[path]
    naive = tree.sequence_frequencies(approximation="independent")
    assert exact == pytest.approx(0.1 * 0.99 * (0.01 + 0.99 * 0.005), 1e-14)
    assert naive.frequencies[path] == pytest.approx(
        0.1 * 0.99 * 0.0595 * 0.109, rel=1e-14
    )
    assert naive.frequencies[path] / exact == pytest.approx(
        0.0595 * 0.109 / (0.01 + 0.99 * 0.005), rel=1e-14
    )
    # Injection lost and recirculation working: the product counts power
    # failures that the exact value knows also fail recirculation.
    path = ("success", "failure", "success")
    exact = tree.sequence_frequencies().frequencies[path]
    assert exact == pytest.approx(0.1 * 0.99 * 0.99 * 0.05 * 0.9, 1e-14)
    assert naive.frequencies[path] == pytest.approx(
        0.1 * 0.99 * 0.0595 * 0.891, rel=1e-14
    )


@pytest.mark.parametrize("name", sorted(event_tree_kinds()))
def test_the_frequencies_sum_to_the_initiator(name):
    tree = event_tree_kinds()[name]
    sequences = tree.sequence_frequencies(X)
    outcomes = tree.outcome_frequencies(X)
    total = sum(sequences.frequencies.values())
    np.testing.assert_allclose(total, tree.frequency, rtol=1e-14)
    np.testing.assert_allclose(
        sum(outcomes.frequencies.values()), tree.frequency, rtol=1e-14
    )
    probabilities = tree.sequence_probabilities(X).probabilities
    np.testing.assert_allclose(sum(probabilities.values()), 1.0, rtol=1e-14)
    for outcome, paths in outcomes.sequences.items():
        np.testing.assert_allclose(
            outcomes.frequencies[outcome],
            sum(sequences.frequencies[p] for p in paths),
            rtol=1e-15,
        )


def test_the_rare_event_values_leave_out_the_successes():
    tree = shared_tree()
    rare = tree.sequence_probabilities(approximation="rare_event")
    by_hand = enumerated(tree, rare_event=True)
    for path, value in rare.probabilities.items():
        assert value == pytest.approx(float(by_hand[path]), rel=1e-14)
    assert rare.probabilities["success", "success", "success"] == 1.0
    assert rare.approximation == "rare_event"
    sequence = tree.sequence_tree(
        ("success", "failure", "failure"), approximation="rare_event"
    )
    assert set(sequence.events) == {"pump A", "pump B", "power"}
    assert tree.sequence_tree(("success",) * 3, "rare_event") is None


def test_a_heading_not_asked_and_outcomes_grouped():
    tree = event_tree_kinds()["protection layers (LOPA)"]
    frequencies = tree.outcome_frequencies()
    assert list(frequencies.frequencies) == ["safe", "contained", "release"]
    assert frequencies.frequencies["release"] == pytest.approx(
        0.2 * 0.1 * 0.01 * 0.05, rel=1e-14
    )
    assert frequencies.frequencies["safe"] == pytest.approx(
        0.2 * (0.9 + 0.1 * 0.99), rel=1e-14
    )
    assert frequencies.sequences["safe"] == [
        ("success", None, None),
        ("failure", "success", None),
    ]
    # A path asking only the first heading's success takes no tree of the
    # others.
    assert set(tree.sequence_tree(("success", None, None)).events) == {"alarm"}


def test_the_paths_must_partition_the_initiator():
    two = [("a", 0.1), ("b", 0.2)]
    with pytest.raises(
        ValueError, match=r"overlap.*\('success', 'success'\) is on both"
    ):
        EventTree(
            1.0,
            two,
            {
                ("success", None): 1,
                ("failure", None): 2,
                (None, "success"): 3,
            },
        )
    with pytest.raises(
        ValueError, match=r"cover 3 of .* 4 .*\('failure', 'failure'\) is on"
    ):
        EventTree(
            1.0,
            two,
            {
                ("success", None): 1,
                ("failure", "success"): 2,
            },
        )
    with pytest.raises(ValueError, match="tuple of 2 entries"):
        EventTree(1.0, two, {("success", "fail"): 1})
    with pytest.raises(ValueError, match="not one of the event tree"):
        EventTree(1.0, two).sequence_tree(("success",))


def test_an_event_with_two_models_is_refused_by_name():
    with pytest.raises(
        ValueError,
        match="Event 'power' has different models in headings 'a' and 'b'",
    ):
        EventTree(
            1.0,
            [
                ("a", heading("A", ["x", "power"], {"x": 0.1, "power": 0.01})),
                ("b", heading("B", ["y", "power"], {"y": 0.1, "power": 0.02})),
            ],
        )
    # One model, given twice alike, is one event.
    tree = EventTree(
        1.0,
        [
            ("a", heading("A", ["x", "s"], {"x": 0.1, "s": W([10, 2])})),
            ("b", heading("B", ["y", "s"], {"y": 0.1, "s": W([10, 2])})),
        ],
    )
    assert tree.shared_events == {"s"}


def test_what_cannot_be_merged_is_refused_saying_what_to_do():
    plain = heading("A", ["x", "y"], {"x": 0.1, "y": 0.2})
    with pytest.raises(ValueError, match="Heading 'x' is a probability"):
        EventTree(1.0, [("a", plain), ("x", 0.1)])
    with pytest.raises(ValueError, match="given more than once"):
        EventTree(1.0, [("a", 0.1), ("a", 0.2)])
    with pytest.raises(ValueError, match=r"probability .* in \[0, 1\]"):
        EventTree(1.0, [("a", 1.5)])
    with pytest.raises(ValueError, match="initiator must be"):
        EventTree(-1.0, [("a", 0.1)])
    with pytest.raises(ValueError, match="approximation must be"):
        EventTree(1.0, [("a", 0.1)]).sequence_probabilities(
            approximation="exact"
        )
    dynamic = FaultTree(
        {"A": ("pand", ["x", "y"])}, {"x": E([1]), "y": E([1])}
    )
    with pytest.raises(NotImplementedError, match="dynamic gates"):
        EventTree(1.0, [("a", dynamic)])
    from repyability import BetaFactor, CCFGroup

    grouped = FaultTree(
        {"A": ("and", ["x", "y"])},
        {"x": 0.1, "y": 0.1},
        ccf_groups=[CCFGroup(["x", "y"], BetaFactor(0.1))],
    )
    with pytest.raises(NotImplementedError, match="shared event"):
        EventTree(1.0, [("a", grouped)])
    repaired = heading(
        "A", ["x"], {"x": {"reliability": E([0.1]), "repairability": E([1])}}
    )
    with pytest.raises(NotImplementedError, match="one-event tree"):
        EventTree(1.0, [("a", repaired), ("b", 0.1)])
    forcing = FaultTree(
        {"A": ("or", ["x", "s"]), "f": ("fdep", "x", ["s"])},
        {"x": 0.1, "s": 0.2},
    )
    with pytest.raises(NotImplementedError, match="FDEP gate 'f'"):
        EventTree(
            1.0,
            [
                ("a", forcing),
                ("b", heading("B", ["s", "y"], {"s": 0.2, "y": 0.1})),
            ],
        )


@pytest.mark.parametrize(
    "name",
    [
        "a shared support event",
        "a heading with a NOT gate",
        "protection layers (LOPA)",
    ],
)
def test_the_importances_against_finite_differences(name):
    tree = event_tree_kinds()[name]
    h = 1e-6
    for outcome in dict.fromkeys(tree.sequences.values()):
        birnbaum = tree.birnbaum_importance(outcome)
        fv = tree.fussell_vesely(outcome)
        total = tree.outcome_frequencies().frequencies[outcome]
        for event, q in tree.events.items():
            moved = [_with_event(tree, event, q + s * h) for s in (1, -1)]
            up, down = (
                t.outcome_frequencies().frequencies[outcome] for t in moved
            )
            slope = (up - down) / (2 * h)
            assert birnbaum[event] == pytest.approx(slope, rel=1e-6, abs=1e-12)
            # The outcome is linear in each event's probability, so the
            # share its occurring takes part in is q B / F.
            assert fv[event] == pytest.approx(
                q * birnbaum[event] / total, rel=1e-12, abs=1e-15
            )


def _with_event(tree: EventTree, event, q):
    """``tree`` with ``event``'s probability ``q`` (in every heading)."""
    headings = []
    for name, model in tree.headings.items():
        if not isinstance(model, FaultTree):
            headings.append((name, q if name == event else model))
            continue
        events = {e: (q if e == event else m) for e, m in model.events.items()}
        headings.append((name, FaultTree(model.gates, events, top=model.top)))
    return EventTree(tree.frequency, headings, tree.sequences)


def test_an_outcome_not_in_the_tree_is_refused_naming_them():
    tree = event_tree_kinds()["protection layers (LOPA)"]
    with pytest.raises(ValueError, match=r"give one of \['safe'"):
        tree.birnbaum_importance("fire")


def test_time_varying_and_repaired_events():
    tree = event_tree_kinds()["lifetime models"]
    p = tree.sequence_probabilities(X).probabilities
    assert p["failure", "failure"].shape == (2,)
    with pytest.raises(ValueError, match="t is required"):
        tree.sequence_probabilities()
    one = tree.sequence_probabilities(50.0).probabilities
    assert one["failure", "failure"] == pytest.approx(
        p["failure", "failure"][1], rel=1e-15
    )
    repaired = event_tree_kinds()["repaired events, an initiator tree"]
    assert repaired.frequency == repaired.initiator.system_failure_frequency()
    # A sequence's probability at t is its tree's point unavailability.
    sequence = repaired.sequence_tree(("success", "failure"))
    np.testing.assert_allclose(
        repaired.sequence_probabilities(X).probabilities["success", "failure"],
        sequence.point_unavailability(X),
        rtol=1e-15,
    )


@pytest.mark.parametrize("name", sorted(event_tree_kinds()))
def test_an_event_tree_goes_through_json(name):
    tree = event_tree_kinds()[name]
    again = EventTree.from_json(tree.to_json())
    assert again.sequences == tree.sequences
    assert list(again.headings) == list(tree.headings)
    for path, value in tree.sequence_frequencies(X).frequencies.items():
        np.testing.assert_array_equal(
            again.sequence_frequencies(X).frequencies[path], value
        )
    out = tree.outcome_frequencies(X).to_dict()
    json.dumps(out)
    assert set(out) == {
        "frequencies",
        "sequences",
        "initiator_frequency",
        "approximation",
    }
    json.dumps(tree.sequence_probabilities(X).to_dict())
    json.dumps(tree.analysis_routes().to_dict())
    with pytest.raises(ValueError, match="Not an event tree"):
        EventTree.from_dict({"type": "FaultTree"})


EVENT_TREE_CALLS = {
    "sequence_probabilities": lambda tree: tree.sequence_probabilities(X),
    "sequence_frequencies": lambda tree: tree.sequence_frequencies(X),
    "outcome_frequencies": lambda tree: tree.outcome_frequencies(X),
    "birnbaum_importance": lambda tree: tree.birnbaum_importance(
        next(iter(tree.sequences.values())), X
    ),
    "fussell_vesely": lambda tree: tree.fussell_vesely(
        next(iter(tree.sequences.values())), X
    ),
}


@pytest.mark.parametrize("name", sorted(event_tree_kinds()))
def test_every_event_tree_method_does_what_the_report_says(name):
    tree = event_tree_kinds()[name]
    report = tree.analysis_routes()
    assert set(report) == set(EVENT_TREE_CALLS)
    for method, call in EVENT_TREE_CALLS.items():
        assert report[method].route == routes.EXACT
        behaves_as_reported(report[method], lambda: call(tree))


def test_the_event_tree_report_covers_every_public_analysis():
    not_analyses = {
        "analysis_routes",
        "from_dict",
        "from_json",
        "to_dict",
        "to_json",
        "sequence_tree",
    }
    public = {
        name
        for name, _ in inspect.getmembers(EventTree, callable)
        if not name.startswith("_")
    }
    tree = next(iter(event_tree_kinds().values()))
    assert set(tree.analysis_routes()) == public - not_analyses
