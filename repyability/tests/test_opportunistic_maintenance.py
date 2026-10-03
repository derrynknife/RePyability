"""Opportunistic maintenance (#108): the components of a maintenance group
are renewed early, from their opportunity age, at a stop of the group,
which another member's failure or scheduled replacement opens (and, if
asked, any system outage), and the group's set-up cost is charged once per
stop."""

import numpy as np
import pytest
import surpyval as surv

from repyability import RepairableRBD
from repyability.rbd import routes as r

E, W = surv.Exponential.from_params, surv.Weibull.from_params
X = surv.ExactEventTime.from_params

SERIES = [("s", "a"), ("a", "b"), ("b", "t")]


def worn(interval=60.0, opportunity=None, group="train", **more):
    """A wearing unit under age replacement, in ``group``."""
    preventive = {"interval": interval, "cost": 20.0}
    if opportunity is not None:
        preventive["opportunity"] = opportunity
    spec = {
        "reliability": W([50.0, 3.0]),
        "repairability": E([0.5]),
        "repair_cost": 100.0,
        "preventive": preventive,
        **more,
    }
    if group is not None:
        spec["group"] = group
    return spec


def pair(opportunity=None, setup=0.0, group="train", **options):
    """Two wearing units in series, each opportunity ``opportunity``."""
    return RepairableRBD(
        SERIES,
        {
            "a": worn(opportunity=opportunity, group=group),
            "b": worn(opportunity=opportunity, group=group),
        },
        maintenance_groups=(
            {group: {"setup_cost": setup}} if group is not None else None
        ),
        **options,
    )


def states(rbd, until):
    """The system's changes of state, stepped through one by one."""
    rbd.initialize_event_queue(until)
    changes = [rbd.next_event()]
    while changes[-1][0] < until:
        changes.append(rbd.next_event())
    return changes


def timeline_rbd():
    # "a" fails at age 30, repaired in 5; "b" never fails in the window but
    # is replaced at age 100, taking 2, from age 20 at a stop of its group.
    return RepairableRBD(
        SERIES,
        {
            "a": {
                "reliability": X(30.0),
                "repairability": X(5.0),
                "repair_cost": 100.0,
                "group": "train",
            },
            "b": {
                "reliability": X(1000.0),
                "repairability": X(5.0),
                "preventive": {
                    "interval": 100.0,
                    "opportunity": 20.0,
                    "duration": X(2.0),
                    "cost": 20.0,
                },
                "group": "train",
            },
        },
        maintenance_groups={"train": {"setup_cost": 500.0}},
    )


def test_a_timeline_worked_by_hand():
    # "a" fails at 30, 65, 100, 135 and 170 (back 5 later each time). Each
    # failure stops the train, and "b", at least 20 old each time (put back
    # at 32, 67, ...), is renewed then, down 2 inside a's 5. Alone, "b"
    # would have been replaced once, at 100.
    rbd = timeline_rbd()
    assert states(rbd, 200.0) == [
        (30.0, False),
        (35.0, True),
        (65.0, False),
        (70.0, True),
        (100.0, False),
        (105.0, True),
        (135.0, False),
        (140.0, True),
        (170.0, False),
        (175.0, True),
        (200.0, True),
    ]
    result = rbd.availability(
        200.0, mc_samples=2, seed=1, control_variate=False
    )
    assert result.mean_availability_interval().estimate == pytest.approx(
        175.0 / 200.0
    )
    assert result.opportunistic_renewals == {"a": 0, "b": 10}
    # Five stops, each charged one set-up; five repairs of "a" and five
    # replacements of "b".
    assert result.cost.by_category == {
        "repair": 500.0,
        "replace": 0.0,
        "preventive": 100.0,
        "inspection": 0.0,
        "component_downtime": 0.0,
        "system_downtime": 0.0,
        "setup": 2500.0,
    }
    used = rbd.spares_demand(200.0, method="simulate", mc_samples=2, seed=1)
    assert used["b"].probabilities[5] == 1.0


def test_a_member_due_at_the_stop_keeps_its_own_replacement():
    # Two units that never fail in the window, replaced every 50 (taking
    # 1), from age 30 at a stop: both are due at the same instants, so each
    # is replaced on its own schedule, at one stop, with one set-up (at 50,
    # 101 and 152).
    unit = {
        "reliability": X(1000.0),
        "repairability": X(5.0),
        "preventive": {
            "interval": 50.0,
            "opportunity": 30.0,
            "duration": X(1.0),
            "cost": 20.0,
        },
        "group": "train",
    }
    rbd = RepairableRBD(
        SERIES,
        {"a": unit, "b": unit},
        maintenance_groups={"train": {"setup_cost": 500.0}},
    )
    result = rbd.availability(
        200.0, mc_samples=2, seed=1, control_variate=False
    )
    assert result.opportunistic_renewals == {"a": 0, "b": 0}
    assert result.cost.by_category["preventive"] == 6 * 20.0
    assert result.cost.by_category["setup"] == 3 * 500.0
    assert result.mean_availability_interval().estimate == pytest.approx(
        197.0 / 200.0
    )


def same_runs(one, other, t=2000.0, **options):
    """Whether two diagrams simulate the same, draw for draw."""
    a = one.availability(t, mc_samples=200, seed=3, **options)
    b = other.availability(t, mc_samples=200, seed=3, **options)
    return (
        np.array_equal(a.timeline, b.timeline)
        and np.array_equal(a.availability, b.availability)
        and np.array_equal(a.uptimes, b.uptimes)
    )


def test_an_opportunity_at_the_interval_is_plain_age_replacement():
    # Never renewed early, each unit is replaced at age 60 as on its own,
    # draw for draw; the set-up cost alone is added.
    plain = pair(group=None)
    grouped = pair(opportunity=60.0)
    assert same_runs(grouped, plain)
    a = grouped.cost(2000.0, mc_samples=200, seed=3)
    b = plain.cost(2000.0, mc_samples=200, seed=3)
    assert np.array_equal(a.samples, b.samples)
    assert grouped.availability(
        2000.0, mc_samples=20, seed=3
    ).opportunistic_renewals == {"a": 0, "b": 0}
    priced = pair(opportunity=60.0, setup=500.0)
    assert same_runs(priced, plain)
    c = priced.cost(2000.0, mc_samples=200, seed=3)
    for category, value in b.by_category.items():
        if category != "setup":
            assert c.by_category[category] == value
    assert c.by_category["setup"] > 0.0
    assert c.mean == pytest.approx(b.mean + c.by_category["setup"])
    # Without opportunities the exact values are those of the units alone.
    assert priced.mean_availability() == plain.mean_availability()


def test_a_group_with_nothing_to_renew_changes_nothing():
    # One member alone in its group: its own stops renew no other, so it
    # is simulated draw for draw as with no group.
    alone = RepairableRBD(
        SERIES,
        {"a": worn(opportunity=20.0), "b": worn(group=None)},
        maintenance_groups={"train": {"setup_cost": 0.0}},
    )
    plain = pair(group=None)
    assert same_runs(alone, plain)
    result = alone.availability(2000.0, mc_samples=50, seed=3)
    assert result.opportunistic_renewals == {"a": 0, "b": 0}


def test_grouping_pays_with_a_large_set_up_cost():
    # In series, each stop costs 500 to set up. Renewing the other unit at
    # a stop, from age 40, shares the set-up and the outage: the cost rate
    # falls well below that of replacing each unit on its own (each
    # failure or replacement its own stop, with its own set-up), whose
    # exact value is known.
    grouped = pair(opportunity=40.0, setup=500.0)
    separate = pair(setup=500.0)
    difference = grouped.compare(
        separate, 20_000.0, mc_samples=100, seed=5, quantity="cost"
    )
    assert difference.upper < 0.0
    simulated = separate.cost(20_000.0, mc_samples=100, seed=5)
    assert simulated.cost_rate == pytest.approx(
        separate.expected_cost_rate(), rel=0.02
    )
    renewed = grouped.availability(20_000.0, mc_samples=20, seed=5)
    assert min(renewed.opportunistic_renewals.values()) > 0


def test_seeds_antithetic_pairs_and_compare():
    rbd = pair(opportunity=40.0, setup=500.0)
    a = rbd.cost(2000.0, mc_samples=100, seed=8)
    b = rbd.cost(2000.0, mc_samples=100, seed=8)
    assert np.array_equal(a.samples, b.samples)
    assert not np.array_equal(
        a.samples, rbd.cost(2000.0, mc_samples=100, seed=9).samples
    )
    paired = rbd.cost(2000.0, mc_samples=100, seed=8, antithetic=True)
    assert paired.mean == pytest.approx(a.mean, rel=0.05)
    same = rbd.compare(rbd, 2000.0, mc_samples=50, seed=8, quantity="cost")
    assert same.estimate == 0.0 and same.standard_error == 0.0
    # Run in two processes, the simulations are the same ones.
    split = rbd.cost(2000.0, mc_samples=100, seed=8, n_jobs=2)
    assert np.array_equal(np.sort(split.samples), np.sort(a.samples))


def test_a_system_outage_is_a_stop_when_asked():
    # "x", outside the group, fails at 40, 85, 130 and 175 (back 5 later),
    # taking the series system down. With system_down, each outage renews
    # the two members, from age 30 (taking 1), with one set-up; without, they
    # are only replaced at age 100, together, at one stop.
    member = {
        "reliability": X(1000.0),
        "repairability": X(5.0),
        "preventive": {
            "interval": 100.0,
            "opportunity": 30.0,
            "duration": X(1.0),
            "cost": 20.0,
        },
        "group": "train",
    }
    edges = [("s", "x"), ("x", "a"), ("a", "b"), ("b", "t")]
    components = {
        "x": {"reliability": X(40.0), "repairability": X(5.0)},
        "a": member,
        "b": member,
    }

    def run(system_down):
        rbd = RepairableRBD(
            edges,
            components,
            maintenance_groups={
                "train": {"setup_cost": 500.0, "system_down": system_down}
            },
        )
        return rbd, rbd.availability(
            200.0, mc_samples=2, seed=1, control_variate=False
        )

    rbd, result = run(True)
    assert result.opportunistic_renewals == {"x": 0, "a": 8, "b": 8}
    assert result.cost.by_category["setup"] == 4 * 500.0
    # The members' outages fall inside x's.
    assert result.mean_availability_interval().estimate == pytest.approx(
        180.0 / 200.0
    )
    assert states(rbd, 200.0)[:4] == [
        (40.0, False),
        (45.0, True),
        (85.0, False),
        (90.0, True),
    ]
    rbd, result = run(False)
    assert result.opportunistic_renewals == {"x": 0, "a": 0, "b": 0}
    assert result.cost.by_category["setup"] == 500.0
    assert result.mean_availability_interval().estimate == pytest.approx(
        179.0 / 200.0
    )


def test_a_nested_diagram_renews_its_group_too():
    # The worked timeline, as the one node of a parent diagram: the parent
    # steps it event by event, stops included.
    parent = RepairableRBD(
        [("s", "train"), ("train", "t")], {"train": timeline_rbd()}
    )
    result = parent.availability(
        200.0, mc_samples=2, seed=1, control_variate=False
    )
    assert result.mean_availability_interval().estimate == pytest.approx(
        175.0 / 200.0
    )


def test_repair_crews_serve_the_early_renewals():
    # With one crew, an early renewal that takes time waits for the crew
    # repairing the failure that opened the stop.
    unit = worn(opportunity=40.0)
    unit["preventive"]["duration"] = E([1.0])
    rbd = RepairableRBD(
        SERIES,
        {"a": unit, "b": unit},
        maintenance_groups={"train": {"setup_cost": 500.0}},
        repair_crews=1,
    )
    free = RepairableRBD(
        SERIES,
        {"a": unit, "b": unit},
        maintenance_groups={"train": {"setup_cost": 500.0}},
    )
    crewed = rbd.availability(
        5000.0, mc_samples=100, seed=2, control_variate=False
    )
    assert crewed.opportunistic_renewals["a"] > 0
    assert np.array_equal(
        crewed.availability,
        rbd.availability(5000.0, mc_samples=100, seed=2).availability,
    )
    unlimited = free.availability(
        5000.0, mc_samples=100, seed=2, control_variate=False
    )
    assert (
        crewed.mean_availability_interval().estimate
        < unlimited.mean_availability_interval().estimate
    )


def test_the_exact_cost_rate_charges_a_set_up_per_stop():
    # No member renewed early: each failure and each replacement is its own
    # stop, charged one set-up.
    priced = pair(setup=500.0)
    plain = pair(group=None)
    frequencies = [priced._node_frequencies(node) for node in "ab"]
    actions = sum(
        failure + maintained for failure, maintained, _ in frequencies
    )
    assert priced.expected_cost_rate() == pytest.approx(
        plain.expected_cost_rate() + 500.0 * actions, rel=1e-12
    )
    # Held working, a member makes no stops.
    assert priced.expected_cost_rate(working_nodes=["b"]) == pytest.approx(
        plain.expected_cost_rate(working_nodes=["b"])
        + 500.0 * sum(frequencies[0][:2]),
        rel=1e-12,
    )
    # A set-up cost alone prices the system.
    bare = {"reliability": W([50.0, 3.0]), "repairability": E([0.5])}
    for setup in (500.0, 0.0):
        rbd = RepairableRBD(
            SERIES,
            {"a": dict(bare, group="train"), "b": dict(bare, group="train")},
            maintenance_groups={"train": {"setup_cost": setup}},
        )
        assert rbd.has_costs == bool(setup)
    assert rbd.expected_cost_rate() == 0.0


def test_replacements_on_a_clock_share_a_set_up():
    # Block replacements at the same instants make one stop: the exact rate,
    # charging a set-up for each member's, refuses.
    unit = worn(preventive={"interval": 60.0, "policy": "block"})
    rbd = RepairableRBD(
        SERIES,
        {"a": unit, "b": unit},
        maintenance_groups={"train": {"setup_cost": 500.0}},
    )
    with pytest.raises(NotImplementedError, match="on a clock"):
        rbd.expected_cost_rate()
    report = rbd.analysis_routes()
    assert report["expected_cost_rate"].route == r.REFUSED
    assert report["mean_availability"].route == r.NUMERICAL
    # One block member alone is fine.
    one = RepairableRBD(
        SERIES,
        {"a": unit, "b": worn()},
        maintenance_groups={"train": {"setup_cost": 500.0}},
    )
    assert one.expected_cost_rate() > 0.0


def test_the_exact_methods_refuse_early_renewals():
    rbd = pair(opportunity=40.0, setup=500.0)
    for method in (
        rbd.mean_availability,
        rbd.system_failure_frequency,
        rbd.expected_cost_rate,
        lambda: rbd.point_availability([10.0]),
        rbd.birnbaum_importance,
    ):
        with pytest.raises(NotImplementedError, match="renewed early"):
            method()
    with pytest.raises(NotImplementedError, match="renewed early"):
        rbd.spares_demand(100.0)
    report = rbd.analysis_routes()
    assert report["mean_availability"].route == r.REFUSED
    assert "renewed early" in report["mean_availability"].reason
    assert report["availability"].route == r.SIMULATED
    assert report["availability"].engine == "python"
    used = rbd.spares_demand(500.0, method="simulate", mc_samples=50, seed=1)
    assert used["a"].mean > 0.0


@pytest.mark.parametrize(
    "components, groups, message",
    [
        ({"a": worn(group=["x"])}, None, "hashable"),
        (
            {
                "a": {
                    "reliability": E([0.01]),
                    "repairability": "instant",
                    "inspection": {"interval": 10.0},
                    "group": "train",
                }
            },
            None,
            "hidden failures",
        ),
        (
            {"a": worn(preventive=None, standby={"units": 2})},
            None,
            "standby group",
        ),
        ({"a": worn(opportunity=20.0, group=None)}, None, "no maintenance"),
        ({"a": worn()}, {"pumps": {}}, "that no component is in"),
        ({"a": worn()}, {"train": {"setup": 1.0}}, "options must be a dict"),
        ({"a": worn()}, {"train": 5.0}, "options must be a dict"),
        ({"a": worn()}, {"train": {"setup_cost": -1.0}}, "setup_cost"),
        ({"a": worn()}, {"train": {"system_down": 1}}, "True or False"),
    ],
)
def test_groups_are_checked(components, groups, message):
    with pytest.raises(ValueError, match=message):
        RepairableRBD(
            [("s", "a"), ("a", "t")], components, maintenance_groups=groups
        )


@pytest.mark.parametrize(
    "preventive, message",
    [
        ({"interval": 60.0, "opportunity": 70.0}, "from 0 to the interval"),
        ({"interval": 60.0, "opportunity": -1.0}, "from 0 to the interval"),
        ({"interval": 60.0, "opportunity": True}, "from 0 to the interval"),
        ({"interval": 60.0, "opportunity": "soon"}, "from 0 to the interval"),
        (
            {"interval": 60.0, "policy": "block", "opportunity": 30.0},
            "only to the 'age' policy",
        ),
    ],
)
def test_the_opportunity_is_checked(preventive, message):
    with pytest.raises(ValueError, match=message):
        RepairableRBD(
            [("s", "a"), ("a", "t")], {"a": worn(preventive=preventive)}
        )


def test_groups_are_saved():
    rbd = RepairableRBD(
        SERIES,
        {
            "a": worn(opportunity=40.0, group=("train", 1)),
            "b": worn(opportunity=45.0, group=("train", 1)),
        },
        maintenance_groups={
            ("train", 1): {"setup_cost": 500.0, "system_down": True}
        },
    )
    loaded = RepairableRBD.from_json(rbd.to_json())
    assert loaded._maintenance == rbd._maintenance
    assert loaded._preventive["b"].opportunity == 45.0
    a = rbd.cost(500.0, mc_samples=50, seed=2)
    b = loaded.cost(500.0, mc_samples=50, seed=2)
    assert np.array_equal(a.samples, b.samples)
