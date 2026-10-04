"""Choosing when redundant components are tested, with their intervals
(#184): ``optimal_inspection_intervals(offset_shares=...)``.

The reference is every plan on the grid, built as a diagram of its own with
the offsets in its schedules, and its exact long-run cost rate and
availability.
"""

import itertools

import pytest
import surpyval as surv

from repyability import BetaFactor, CCFGroup, RepairableRBD

CALENDAR = [730.0, 2190.0, 4380.0, 8760.0, 17520.0]


def valve(interval=8760.0, offset=0.0, rate=2e-6):
    return {
        "reliability": surv.Exponential.from_params([rate]),
        "repairability": "instant",
        "inspection": {"interval": interval, "cost": 500.0, "offset": offset},
    }


def valves(schedules, beta=0.1, rate=2e-6):
    """Valves in parallel (1-out-of-n), each ``(interval, offset)``."""
    names = [f"v{i}" for i in range(1, len(schedules) + 1)]
    return RepairableRBD(
        [("s", n) for n in names] + [(n, "t") for n in names],
        {
            n: valve(*schedule, rate=rate)
            for n, schedule in zip(names, schedules)
        },
        ccf_groups=[CCFGroup(names, BetaFactor(beta))] if beta else None,
    )


def cheapest(plans, target):
    """The cheapest of ``{key: rbd}`` whose PFDavg is at most ``target``,
    the most available of those that cost the same: its key, cost rate and
    availability."""
    meeting = {
        key: (rbd.expected_cost_rate(), 1 - rbd.mean_availability())
        for key, rbd in plans.items()
        if 1 - rbd.mean_availability() <= target
    }
    key = min(meeting, key=meeting.get)
    return key, meeting[key][0], 1 - meeting[key][1]


@pytest.mark.parametrize("target", [2e-4, 5e-4, 1e-3])
@pytest.mark.parametrize("beta", [0.0, 0.1])
def test_the_grid_is_searched(target, beta):
    shares = [0.0, 0.25, 0.5, 0.75]
    plan = valves([(8760.0, 0.0)] * 2, beta).optimal_inspection_intervals(
        allowed=CALENDAR, min_availability=1 - target, offset_shares=shares
    )
    # The first valve's tests stay from 0: shifting both changes nothing.
    plans = {
        (a, b, share): valves([(a, 0.0), (b, share * b)], beta)
        for a, b, share in itertools.product(CALENDAR, CALENDAR, shares)
    }
    _, cost, availability = cheapest(plans, target)
    assert plan.cost_rate == pytest.approx(cost, rel=1e-12)
    assert plan.availability == pytest.approx(availability, rel=1e-15)
    # The plan is what it says.
    a, b = plan.intervals["v1"], plan.intervals["v2"]
    assert plan.offsets["v1"] == 0.0
    chosen = plans[a, b, plan.offsets["v2"] / b]
    assert chosen.expected_cost_rate() == plan.cost_rate
    assert chosen.mean_availability() == plan.availability


def test_staggering_beats_testing_together():
    # The issue's 1oo2 trip with a 10% common cause: yearly tests six
    # months apart halve the PFDavg of yearly tests together.
    together = valves([(8760.0, 0.0)] * 2)
    apart = valves([(8760.0, 0.0), (8760.0, 4380.0)])
    assert 1 - together.mean_availability() == pytest.approx(9.573e-4, 1e-3)
    assert 1 - apart.mean_availability() == pytest.approx(4.925e-4, 1e-3)
    plain = together.optimal_inspection_intervals(
        allowed=CALENDAR, min_availability=1 - 5e-4
    )
    staggered = together.optimal_inspection_intervals(
        allowed=CALENDAR, min_availability=1 - 5e-4, offset_shares="stagger"
    )
    # A plan has the offsets it keeps, searched or not (#222).
    assert plain.offsets == {"v1": 0.0, "v2": 0.0}
    assert staggered.intervals == {"v1": 8760.0, "v2": 8760.0}
    assert staggered.offsets == {"v1": 0.0, "v2": 4380.0}
    assert staggered.cost_rate < 0.7 * plain.cost_rate


def test_three_valves_are_spread_over_the_interval():
    trio = valves([(8760.0, 0.0)] * 3, beta=0.2, rate=1e-5)
    plan = trio.optimal_inspection_intervals(
        allowed=[8760.0], min_availability=0.9, offset_shares="stagger"
    )
    # Only the offsets are free: the cheapest plan is the most available
    # one at the same cost, its tests a third of a year apart.
    assert sorted(plan.offsets.values()) == pytest.approx(
        [0.0, 2920.0, 5840.0]
    )
    best = max(
        (
            valves(
                [(8760.0, 0.0)] + [(8760.0, s * 8760.0) for s in shares],
                beta=0.2,
                rate=1e-5,
            ).mean_availability()
            for shares in itertools.product([0, 1 / 3, 2 / 3], repeat=2)
        )
    )
    assert plan.availability == pytest.approx(best, rel=1e-15)


def test_with_other_tests_fixed_the_first_offset_is_chosen_too():
    # v1 and v3 are tested together and stay so; v2's tests are best half a
    # year from theirs, which keeping its first offset at 0 would miss.
    trio = valves([(8760.0, 0.0)] * 3, beta=0.2, rate=1e-5)
    plan = trio.optimal_inspection_intervals(
        nodes=["v2"],
        allowed=[8760.0],
        max_cost_rate=1e9,
        offset_shares=[0.0, 0.5],
    )
    assert plan.offsets == {"v2": 4380.0}
    assert plan.availability == pytest.approx(
        valves(
            [(8760.0, 0.0), (8760.0, 4380.0), (8760.0, 0.0)],
            beta=0.2,
            rate=1e-5,
        ).mean_availability(),
        rel=1e-15,
    )


def test_a_cost_cap():
    pair = valves([(8760.0, 0.0)] * 2)
    plan = pair.optimal_inspection_intervals(
        allowed=CALENDAR, max_cost_rate=0.12, offset_shares=[0.0, 0.5]
    )
    plans = {
        (a, b, s): valves([(a, 0.0), (b, s * b)])
        for a, b, s in itertools.product(CALENDAR, CALENDAR, [0.0, 0.5])
    }
    affordable = {
        key: rbd.mean_availability()
        for key, rbd in plans.items()
        if rbd.expected_cost_rate() <= 0.12
    }
    assert plan.availability == pytest.approx(
        max(affordable.values()), rel=1e-15
    )


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"offset_shares": "stagger"}, "give allowed"),
        ({"allowed": CALENDAR, "offset_shares": "even"}, "or 'stagger'"),
        ({"allowed": CALENDAR, "offset_shares": [0.0, 1.0]}, r"in \[0, 1\)"),
        ({"allowed": CALENDAR, "offset_shares": [-0.1]}, r"in \[0, 1\)"),
        ({"allowed": CALENDAR, "offset_shares": [True]}, r"in \[0, 1\)"),
        ({"allowed": CALENDAR, "offset_shares": []}, "no share"),
        ({"allowed": CALENDAR, "offset_shares": {"v1": [0.0]}}, "no share"),
    ],
)
def test_offsets_are_checked(kwargs, match):
    with pytest.raises(ValueError, match=match):
        valves([(8760.0, 0.0)] * 2).optimal_inspection_intervals(
            min_availability=0.999, **kwargs
        )


def test_a_single_share_is_a_number_too():
    pair = valves([(8760.0, 0.0)] * 2)
    plan = pair.optimal_inspection_intervals(
        allowed=[8760.0],
        min_availability=0.99,
        offset_shares={"v1": 0, "v2": 0.5},
    )
    assert plan.offsets == {"v1": 0.0, "v2": 4380.0}


def test_offsets_is_the_old_name_of_offset_shares():
    # Its values are shares, where with_intervals' offsets are times (#222).
    pair = valves([(8760.0, 0.0), (8760.0, 0.0)])
    plan = pair.optimal_inspection_intervals(
        allowed=[8760.0], min_availability=0.99, offset_shares=[0.0, 0.5]
    )
    with pytest.warns(FutureWarning, match="renamed offset_shares"):
        old = pair.optimal_inspection_intervals(
            allowed=[8760.0], min_availability=0.99, offsets=[0.0, 0.5]
        )
    assert old == plan
    with pytest.raises(ValueError, match="offset_shares alone"):
        pair.optimal_inspection_intervals(
            allowed=[8760.0], offsets=[0.0], offset_shares=[0.0]
        )


def test_unknown_nodes_are_named():
    pair = valves([(8760.0, 0.0), (8760.0, 0.0)])
    with pytest.raises(ValueError, match=r"names \['V3'\]"):
        pair.optimal_inspection_intervals(
            allowed=[8760.0],
            offset_shares={"v1": [0.0], "v2": [0.5], "V3": [0.25]},
        )
    with pytest.raises(ValueError, match=r"names \['V3'\]"):
        pair.optimal_inspection_intervals(
            allowed={"v1": [8760.0], "v2": [8760.0], "V3": [8760.0]}
        )
    with pytest.raises(ValueError, match="Its components are"):
        pair.optimal_inspection_intervals(["V2"], allowed=[8760.0])
    with pytest.raises(ValueError, match="Its components are"):
        pair.with_intervals({"v1": 8760.0}, offsets={"V2": 0.0})


def test_an_offset_needs_a_test_schedule():
    # Also when the node's interval is given (#222).
    seal = {
        "reliability": surv.Weibull.from_params([6000.0, 2.5]),
        "repairability": surv.Exponential.from_params([1 / 8.0]),
        "replace_cost": 3000.0,
        "preventive": {"interval": 3000.0, "cost": 800.0},
    }
    rbd = RepairableRBD([("s", "seal"), ("seal", "t")], {"seal": seal})
    for intervals in ({"seal": 3000.0}, {}):
        with pytest.raises(ValueError, match="offset but no test schedule"):
            rbd.with_intervals(intervals, offsets={"seal": 10.0})
