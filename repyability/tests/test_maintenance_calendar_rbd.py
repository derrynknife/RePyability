"""A ``RepairableRBD`` component replaced on a calendar policy (#319,
UC-13, the brief's deliverable 6): a ``"preventive"`` spec with
``"policy": "calendar"`` and the ``CalendarPolicy`` of
``NonRepairable.optimal_calendar_policy``, simulated by the Python event
loop, each replacement charged at its opportunity's price."""

import math

import numpy as np
import pytest
import surpyval as surv

from repyability import (
    Calendar,
    NodeState,
    NonRepairable,
    Opportunity,
    RepairableRBD,
)
from repyability.maintenance import _replacement


def solve(part, calendar, **options):
    """The calendar policy, on a grid of a given step where a check needs
    one (the public method sizes its own)."""
    return _replacement.optimal_calendar_policy(part, calendar, **options)


def price(part, calendar, rule, **options):
    """A rule's cost rate, likewise."""
    return _replacement.calendar_policy_cost(part, calendar, rule, **options)


LIFE = surv.Weibull.from_params([1000.0, 2.5])
PLANT = Calendar(
    [
        Opportunity("weekly", 3.0, period=7.0),
        Opportunity("quarterly", 2.0, period=91.0),
        Opportunity("annual", 1.0, period=364.0),
    ],
    failure=20.0,
    anytime=4.0,
)
SHUTS = Calendar(
    [
        Opportunity("quarterly", 3.0, period=91.0),
        Opportunity("annual", 1.0, period=364.0),
    ],
    failure=20.0,
    anytime=None,
)


@pytest.fixture(scope="module")
def plant():
    return solve(NonRepairable(LIFE), PLANT, step=1.0)


def bearing(policy, **more):
    spec = {
        "reliability": LIFE,
        "repairability": "instant",
        "replace_cost": 20.0,
        "preventive": {"policy": "calendar", "calendar": policy},
    }
    return RepairableRBD([("s", "b"), ("b", "t")], {"b": {**spec, **more}})


@pytest.mark.parametrize("method", ["dp", "integration"])
def test_a_unit_is_next_replaced_where_its_threshold_is_reached(method):
    policy = solve(NonRepairable(LIFE), SHUTS, step=1.0, method=method)
    # New at 0: the next annual shut. Put in at an annual shut (300 days
    # on, it is young at the next): the one after. Old: at once.
    annual = 1.0, "annual"
    assert policy._next_planned(0.0, 0.0) == (364.0, *annual)
    assert policy._next_planned(300.0, 300.0) == (728.0, *annual)
    assert policy._next_planned(-500.0, 0.0) == (0.0, *annual)
    assert policy._next_planned(100.0, 150.0) == (364.0, *annual)


def test_the_diagram_costs_the_policys_rate(plant):
    # The brief's test 3 against the real engine: over 40 years, the mean
    # cost a day is the policy's rate (from new, a few units' worth of
    # start-up, well inside the error).
    rbd = bearing(plant)
    years = 40
    result = rbd.cost(
        t_simulation=364.0 * years, mc_samples=300, seed=2, engine="python"
    )
    days = 364.0 * years
    assert abs(result.mean / days - plant.cost_rate) < 3 * (
        result.mean_se / days
    )


def test_the_compiled_engine_leaves_it_to_python(plant):
    rbd = bearing(plant)
    route = rbd.analysis_routes()["availability"]
    assert route.engine == "python"
    assert "calendar policy" in route.engine_reason
    with pytest.raises(NotImplementedError, match="calendar policy"):
        rbd.availability(t_simulation=100.0, mc_samples=2, engine="numba")


def test_the_exact_methods_refuse_it(plant):
    rbd = bearing(plant)
    with pytest.raises(NotImplementedError, match="calendar policy"):
        rbd.expected_cost_rate()
    with pytest.raises(NotImplementedError, match="calendar policy"):
        rbd.mean_availability()
    assert rbd.has_costs


def test_a_calendar_policy_is_not_saved(plant):
    with pytest.raises(NotImplementedError, match="calendar"):
        bearing(plant).to_json()


def test_its_times_are_absolute(plant):
    rbd = bearing(plant)
    with pytest.raises(ValueError, match="no phase"):
        rbd.availability(
            t_simulation=100.0,
            mc_samples=2,
            state={"b": NodeState(age=10.0, phase=5.0)},
        )
    # An age is taken: an old unit is replaced at the first chance.
    rbd.availability(
        t_simulation=100.0, mc_samples=2, state={"b": NodeState(age=900.0)}
    )


@pytest.mark.parametrize(
    "preventive, match",
    [
        ({"policy": "calendar"}, "needs a calendar"),
        ({"policy": "calendar", "calendar": "weekly"}, "needs a calendar"),
        ({"policy": "calendar", "interval": 30.0, "calendar": 0}, "interval"),
        ({"policy": "calendar", "cost": 3.0, "calendar": 0}, "no cost"),
        ({"interval": 30.0, "calendar": 0}, "only to the 'calendar'"),
        (
            {"policy": "calendar", "opportunity": 3.0, "calendar": 0},
            "no maintenance group",
        ),
        (
            {"policy": "calendar", "opportunity": -1.0, "calendar": 0},
            "an age at least 0",
        ),
    ],
)
def test_the_spec_is_checked(plant, preventive, match):
    # 0 stands for the policy itself.
    if preventive.get("calendar") == 0:
        preventive = {**preventive, "calendar": plant}
    with pytest.raises(ValueError, match=match):
        RepairableRBD(
            [("s", "b"), ("b", "t")],
            {
                "b": {
                    "reliability": LIFE,
                    "repairability": "instant",
                    "preventive": preventive,
                }
            },
        )


def test_a_failure_cost_is_the_diagrams(plant):
    # The calendar's failure price prices the policy; the diagram charges
    # its own replace_cost at a failure, as for any component.
    free = bearing(plant, replace_cost=0.0)
    priced = bearing(plant)
    options = dict(t_simulation=3640.0, mc_samples=50, seed=4)
    assert free.cost(**options).mean < priced.cost(**options).mean
    assert math.isfinite(free.cost(**options).mean)


# -- in a maintenance group (#349) --------------------------------------------
#
# A calendar component is a member like any other: its planned
# replacements are stops of its group, and a stop renews it early from its
# opportunity age, charged the calendar's price for a planned replacement
# on the day (its replace_cost where the calendar allows none), the new
# unit's next replacement counted from then. The policy is not
# re-optimised for the group's stops.

X = surv.ExactEventTime.from_params
SERIES = [("s", "a"), ("a", "b"), ("b", "t")]


@pytest.fixture(scope="module")
def shuts():
    """The policy over SHUTS (no replacement on any day): a new unit is
    replaced at the annual shut, from 88 days old there."""
    return solve(NonRepairable(LIFE), SHUTS, step=1.0)


def states(rbd, until):
    """The system's changes of state, stepped through one by one."""
    rbd.initialize_event_queue(until)
    changes = [rbd.next_event()]
    while changes[-1][0] < until:
        changes.append(rbd.next_event())
    return changes


def members(policy, opportunity, life=300.0, repair_cost=100.0):
    """A train's members: "a" fails at ``life`` (back 5 later), each
    failure a stop of the train; "b", never failing in the window, is on
    ``policy`` and renewed early from ``opportunity`` old, at 20 a
    failure."""
    return {
        "a": {
            "reliability": X(life),
            "repairability": X(5.0),
            "repair_cost": repair_cost,
            "group": "train",
        },
        "b": {
            "reliability": X(1000.0),
            "repairability": X(5.0),
            "replace_cost": 20.0,
            "preventive": {
                "policy": "calendar",
                "calendar": policy,
                "opportunity": opportunity,
            },
            "group": "train",
        },
    }


def grouped(policy, opportunity, **more):
    """The train of ``members`` in series, set up at 500 a stop."""
    return RepairableRBD(
        SERIES,
        members(policy, opportunity, **more),
        maintenance_groups={"train": {"setup_cost": 500.0}},
    )


def test_a_calendar_replacement_is_a_stop_of_its_group(shuts):
    # "a", never failing in the window, is replaced on its calendar at the
    # annual shuts, 364 and 728, at 1 each; "b", never failing, replaced
    # at 1000 old on its own (taking 2) and from 20 old at a stop, is
    # renewed at each, down 2: one set-up a stop.
    assert shuts._next_planned(0.0, 0.0) == (364.0, 1.0, "annual")
    rbd = RepairableRBD(
        SERIES,
        {
            "a": {
                "reliability": X(1000.0),
                "repairability": X(5.0),
                "replace_cost": 20.0,
                "preventive": {"policy": "calendar", "calendar": shuts},
                "group": "train",
            },
            "b": {
                "reliability": X(1000.0),
                "repairability": X(5.0),
                "preventive": {
                    "interval": 1000.0,
                    "opportunity": 20.0,
                    "duration": X(2.0),
                    "cost": 20.0,
                },
                "group": "train",
            },
        },
        maintenance_groups={"train": {"setup_cost": 500.0}},
    )
    assert states(rbd, 800.0) == [
        (364.0, False),
        (366.0, True),
        (728.0, False),
        (730.0, True),
        (800.0, True),
    ]
    runs = rbd.simulate_timelines(800.0, mc_samples=1, seed=1)
    assert list(runs.components["a"][0].changes) == []
    assert list(runs.components["b"][0].changes) == [
        364.0,
        366.0,
        728.0,
        730.0,
    ]
    assert list(runs.system[0].causes) == ["b", "b", "b", "b"]
    result = rbd.availability(
        800.0, mc_samples=2, seed=1, control_variate=False
    )
    assert result.mean_availability_interval().estimate == pytest.approx(
        796.0 / 800.0
    )
    # Renewals summed over the two simulations; costs a simulation.
    assert result.opportunistic_renewals == {"a": 0, "b": 4}
    assert result.cost.by_category == {
        "repair": 0.0,
        "replace": 0.0,
        "preventive": 2 * 1.0 + 2 * 20.0,
        "inspection": 0.0,
        "component_downtime": 0.0,
        "system_downtime": 0.0,
        "setup": 2 * 500.0,
    }


def test_a_stop_renews_the_calendar_component_from_its_opportunity_age(
    shuts,
):
    # "a" fails at 300 and 605. Alone, "b" is replaced at the annual shuts,
    # 364 (364 old) and 728. From 250 old at a stop, it is renewed at 300
    # and 605 (300 and 305 old), days the shuts price no replacement on,
    # so each is charged its replace_cost; and its next planned
    # replacement counts from its new start: not 364 (64 old, under the
    # threshold) but 728. From 320 old it is renewed at neither.
    assert shuts._planned_price(300.0) == math.inf
    assert shuts._next_planned(300.0, 300.0)[0] == 728.0
    assert shuts._next_planned(605.0, 605.0)[0] == 728.0
    for opportunity, renewed, preventive, stops in (
        (250.0, 2, 20.0 + 20.0 + 1.0, 3),
        (320.0, 0, 1.0 + 1.0, 4),
    ):
        rbd = grouped(shuts, opportunity)
        assert states(rbd, 800.0) == [
            (300.0, False),
            (305.0, True),
            (605.0, False),
            (610.0, True),
            (800.0, True),
        ]
        result = rbd.availability(
            800.0, mc_samples=2, seed=1, control_variate=False
        )
        assert result.opportunistic_renewals == {"a": 0, "b": 2 * renewed}
        assert result.cost.by_category["preventive"] == preventive
        assert result.cost.by_category["repair"] == 2 * 100.0
        assert result.cost.by_category["setup"] == stops * 500.0


def test_an_early_renewal_is_charged_the_calendars_price_on_the_day(plant):
    # "a" fails every 15 (at 10, 25, 40, ...) and "b", from 0 old, is
    # renewed at each stop: at the plant's price on the day, a weekly shut
    # (70, 175) at 3, any other day at 4. Never 20, its failure price.
    rbd = grouped(plant, 0.0, life=10.0, repair_cost=0.0)
    stops = np.arange(10.0, 200.0, 15.0)
    prices = PLANT.planned_cost(stops, step=plant.step)
    assert set(prices) == {3.0, 4.0} and prices[4] == 3.0
    result = rbd.availability(
        200.0, mc_samples=2, seed=1, control_variate=False
    )
    assert result.opportunistic_renewals == {"a": 0, "b": 2 * stops.size}
    assert result.cost.by_category["preventive"] == prices.sum()
    assert result.cost.by_category["setup"] == stops.size * 500.0


def test_the_grouped_rule_against_a_plain_simulation():
    # Two instant units in a train with a set-up of 50 a stop: "a"
    # (Weibull, 50 by 3) repaired at 100, replaced at 60 old for 20, and
    # from 30 old at a stop; "b" (Weibull, 100 by 2.5) replaced at 10 at
    # a failure, on a policy over shuts every 70 at 2 (4 on any day), and
    # from 40 old at a stop, at the calendar's price on the day. The rules
    # written out plainly, with draws of their own, give the same mean
    # cost and early renewals.
    a_life = surv.Weibull.from_params([50.0, 3.0])
    b_life = surv.Weibull.from_params([100.0, 2.5])
    calendar = Calendar(
        [Opportunity("shut", 2.0, period=70.0)], failure=10.0, anytime=4.0
    )
    policy = NonRepairable(b_life).optimal_calendar_policy(calendar)
    rbd = RepairableRBD(
        SERIES,
        {
            "a": {
                "reliability": a_life,
                "repairability": "instant",
                "repair_cost": 100.0,
                "preventive": {
                    "interval": 60.0,
                    "cost": 20.0,
                    "opportunity": 30.0,
                },
                "group": "train",
            },
            "b": {
                "reliability": b_life,
                "repairability": "instant",
                "replace_cost": 10.0,
                "preventive": {
                    "policy": "calendar",
                    "calendar": policy,
                    "opportunity": 40.0,
                },
                "group": "train",
            },
        },
        maintenance_groups={"train": {"setup_cost": 50.0}},
    )
    horizon, n = 2000.0, 200

    def plain(rng):
        put, life = {}, {}

        def renew(node, t):
            put[node] = t
            life[node] = (
                50.0 * rng.weibull(3.0)
                if node == "a"
                else 100.0 * rng.weibull(2.5)
            )

        renew("a", 0.0)
        renew("b", 0.0)
        cost, early = 0.0, {"a": 0, "b": 0}
        while True:
            failure = {node: put[node] + life[node] for node in "ab"}
            next_b = policy.next_replacement(0.0, put["b"])
            planned = {"a": put["a"] + 60.0, "b": next_b.time}
            t, node = min((min(failure[n], planned[n]), n) for n in "ab")
            if t >= horizon:
                return cost, early
            cost += 50.0
            failed = failure[node] <= planned[node]
            if node == "a":
                cost += 100.0 if failed else 20.0
                renew("a", t)
                if t - put["b"] >= 40.0:
                    cost += calendar.planned_cost([t], step=policy.step)[0]
                    renew("b", t)
                    early["b"] += 1
            else:
                cost += 10.0 if failed else next_b.price
                renew("b", t)
                if t - put["a"] >= 30.0:
                    cost += 20.0
                    renew("a", t)
                    early["a"] += 1

    rng = np.random.default_rng(7)
    runs = [plain(rng) for _ in range(n)]
    costs = np.array([cost for cost, _ in runs])
    result = rbd.availability(
        horizon, mc_samples=n, seed=3, control_variate=False
    )
    se = math.hypot(result.cost.mean_se, costs.std(ddof=1) / math.sqrt(n))
    assert abs(result.cost.mean - costs.mean()) < 3 * se
    for node in "ab":
        counts = np.array([early[node] for _, early in runs])
        assert counts.mean() > 5
        se = math.sqrt(2) * counts.std(ddof=1) / math.sqrt(n)
        simulated = result.opportunistic_renewals[node] / n
        assert abs(simulated - counts.mean()) < 3 * se


def test_the_group_is_simulated_whole_in_python(shuts):
    rbd = grouped(shuts, 250.0)
    route = rbd.analysis_routes()["availability"]
    assert route.engine == "python"
    assert "maintenance groups" in route.engine_reason
    with pytest.raises(NotImplementedError, match="maintenance groups"):
        rbd.availability(t_simulation=100.0, mc_samples=2, engine="numba")
    # A conditional run simulates its modules alone: a calendar member
    # goes with its group, here every component.
    with pytest.raises(NotImplementedError, match="every component"):
        rbd.availability(t_simulation=100.0, mc_samples=2, conditional=True)
    three = RepairableRBD(
        [("s", "a"), ("a", "b"), ("b", "c"), ("c", "t")],
        {
            **members(shuts, 250.0),
            "c": {"reliability": LIFE, "repairability": "instant"},
        },
        maintenance_groups={"train": {"setup_cost": 500.0}},
    )
    assert three._conditional_modules(set(), set()) == ["a", "b"]
    with pytest.raises(NotImplementedError, match="held"):
        three._conditional_modules({"a"}, set())
    for method in (rbd.expected_cost_rate, rbd.mean_availability):
        with pytest.raises(NotImplementedError, match="calendar policy"):
            method()
    with pytest.raises(NotImplementedError, match="calendar policy"):
        rbd.spares_demand(100.0)
    for name in ("expected_cost_rate", "mean_availability", "spares_demand"):
        assert rbd.analysis_routes()[name].route == "refused"
