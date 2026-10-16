"""A ``RepairableRBD`` component replaced on a calendar policy (#319,
UC-13, the brief's deliverable 6): a ``"preventive"`` spec with
``"policy": "calendar"`` and the ``CalendarPolicy`` of
``NonRepairable.optimal_calendar_policy``, simulated by the Python event
loop, each replacement charged at its opportunity's price."""

import math

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
        ({"policy": "calendar", "opportunity": 3.0, "calendar": 0}, "only"),
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
