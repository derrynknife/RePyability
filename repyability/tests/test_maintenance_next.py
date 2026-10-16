"""When to replace a part in service next
(``NonRepairable.optimal_next_replacement``; #319, UC-12 of
``repyability/maintenance/README.md``)."""

import json
import math

import numpy as np
import pytest
import surpyval as surv

from repyability import (
    Calendar,
    MaintenancePolicy,
    NextReplacement,
    NonRepairable,
    Opportunity,
)

LIFE = surv.Weibull.from_params([365.0, 2.0])
BEARING = NonRepairable(LIFE, planned_cost=150.0, unplanned_cost=400.0)
OUTAGES = [(150.0, 100.0), (330.0, 100.0)]
# The shuts that keep coming: every 180 days, at $100 (#348).
SHUTS = Calendar([Opportunity("shut", 100.0, period=180.0)])


@pytest.fixture(scope="module")
def uc12():
    return BEARING.optimal_next_replacement(200.0, scheduled_outages=OUTAGES)


@pytest.fixture(scope="module")
def shuts_policy():
    return BEARING.optimal_calendar_policy(SHUTS)


def by_kind(found):
    return {
        (choice.kind, choice.time if choice.kind == "outage" else None): (
            choice
        )
        for choice in found.choices
    }


def test_the_bearing_at_200_days(uc12):
    # UC-12: valued at the baseline, the replacement policy's cost rate.
    assert isinstance(uc12, NextReplacement)
    assert uc12.cost_rate == BEARING.optimal_replacement_policy().cost_rate
    assert uc12.cost_rate == pytest.approx(1.117, abs=5e-4)
    choices = by_kind(uc12)
    assert choices["now", None].cost == 150.0
    assert choices["outage", 150.0].cost == pytest.approx(110.0, abs=0.05)
    assert choices["outage", 330.0].cost == pytest.approx(156.5, abs=0.05)
    assert choices["planned", None].cost == pytest.approx(133.9, abs=0.05)
    assert choices["planned", None].time == pytest.approx(97.6, abs=0.05)
    assert choices["failure", None].cost == pytest.approx(186.1, abs=0.05)
    assert not uc12.now
    assert uc12.best is choices["outage", 150.0]
    evens = {key: choice.break_even_age for key, choice in choices.items()}
    assert evens["outage", 150.0] == pytest.approx(288.4, abs=0.05)
    assert evens["outage", 330.0] == pytest.approx(189.3, abs=0.05)
    assert evens["planned", None] == pytest.approx(297.6, abs=0.05)
    assert evens["failure", None] == pytest.approx(133.3, abs=0.05)
    assert uc12.replace_from_age == evens["planned", None]


@pytest.mark.parametrize("age", [0.0, 100.0, 200.0, 290.0])
def test_with_no_outages_it_is_the_age_replacement(age):
    # Valued at the age replacement's own cost rate, the best planned
    # replacement of a part is at the optimal age, and replacing now is the
    # best from that age.
    policy = BEARING.optimal_replacement_policy()
    found = BEARING.optimal_next_replacement(age)
    assert found.cost_rate == policy.cost_rate
    planned = by_kind(found)["planned", None]
    assert planned.time == pytest.approx(policy.interval - age, abs=1e-3)
    assert found.replace_from_age == pytest.approx(policy.interval, rel=1e-4)
    assert found.best is planned
    older = BEARING.optimal_next_replacement(policy.interval + 10.0)
    assert older.now


def test_the_age_replacement_policy_given_is_the_default(uc12):
    policy = BEARING.optimal_replacement_policy()
    assert isinstance(policy, MaintenancePolicy)
    given = BEARING.optimal_next_replacement(
        200.0, scheduled_outages=OUTAGES, policy=policy
    )
    assert given.to_dict() == uc12.to_dict()


def test_a_calendar_with_no_opportunities_gives_the_default_baseline(uc12):
    # A calendar with no opportunities, replacement on any day at the
    # planned price and a failure at the unplanned one, is age
    # replacement. Its policy's cost rate is exact for its grid (a step
    # of about the mean life over 1,000: 0.32 days here), within about
    # the square of the step of the continuous one, so the baselines
    # agree to 1e-6 relative, and with them the costs and break-even
    # ages to 1e-3 (they agree to 1e-4).
    policy = BEARING.optimal_calendar_policy(Calendar())
    found = BEARING.optimal_next_replacement(
        200.0, scheduled_outages=OUTAGES, policy=policy
    )
    assert found.cost_rate == policy.cost_rate
    assert found.cost_rate == pytest.approx(uc12.cost_rate, rel=1e-6)
    assert found.best.kind == uc12.best.kind
    assert found.best.time == uc12.best.time
    assert found.replace_from_age == pytest.approx(
        uc12.replace_from_age, abs=1e-3
    )
    for mine, theirs in zip(found.choices, uc12.choices):
        assert (mine.kind, mine.price) == (theirs.kind, theirs.price)
        assert mine.time == pytest.approx(theirs.time, abs=1e-3)
        assert mine.cost == pytest.approx(theirs.cost, abs=1e-3)
        if theirs.break_even_age is None:
            assert mine.break_even_age is None
        else:
            assert mine.break_even_age == pytest.approx(
                theirs.break_even_age, abs=1e-3
            )


def test_the_shuts_policy_keeps_the_bearing_longer(uc12, shuts_policy):
    # UC-12 with the shuts' policy as the baseline (#348): a day of the
    # bearing's service is worth 0.990 rather than 1.117, so keeping it is
    # dearer and replacing now beats each later choice from a younger
    # age; the outage in 150 days from 256.1 days old, not 288.4.
    found = BEARING.optimal_next_replacement(
        200.0, scheduled_outages=OUTAGES, policy=shuts_policy
    )
    assert found.cost_rate == shuts_policy.cost_rate
    assert found.cost_rate == pytest.approx(0.990, abs=5e-4)
    choices = by_kind(found)
    assert choices["now", None].cost == 150.0
    assert choices["outage", 150.0].cost == pytest.approx(124.6, abs=0.05)
    assert choices["outage", 330.0].cost == pytest.approx(178.5, abs=0.05)
    assert choices["planned", None].cost == pytest.approx(142.8, abs=0.05)
    assert choices["planned", None].time == pytest.approx(63.9, abs=0.05)
    assert choices["failure", None].cost == pytest.approx(210.4, abs=0.05)
    assert not found.now
    assert found.best is choices["outage", 150.0]
    evens = {key: choice.break_even_age for key, choice in choices.items()}
    assert evens["outage", 150.0] == pytest.approx(256.13, abs=0.05)
    assert evens["outage", 330.0] == pytest.approx(152.95, abs=0.05)
    assert evens["planned", None] == pytest.approx(263.86, abs=0.05)
    assert evens["failure", None] == pytest.approx(86.35, abs=0.05)
    assert found.replace_from_age == evens["planned", None]
    # Every later choice costs more, and is beaten from a younger age.
    for key, theirs in by_kind(uc12).items():
        if key != ("now", None):
            assert choices[key].cost > theirs.cost + 1.0
            assert evens[key] < theirs.break_even_age - 1.0
    # A break-even age is where the choice costs what replacing now does.
    at_even = BEARING.optimal_next_replacement(
        evens["outage", 150.0], scheduled_outages=OUTAGES, policy=shuts_policy
    )
    assert by_kind(at_even)["outage", 150.0].cost == pytest.approx(
        150.0, abs=1e-6
    )
    assert by_kind(at_even)["outage", 150.0].break_even_age == pytest.approx(
        evens["outage", 150.0], abs=1e-6
    )


def simulated(age, until, price, rate, n=400_000, seed=0):
    """A choice's cost simulated plainly: keep the Weibull(365, 2) part from
    ``age`` to ``until`` from now, then replace it at ``price``, or at 400
    if it fails first, less the time it serves at ``rate``; the mean and its
    standard error."""
    rng = np.random.default_rng(seed)
    held = (age / 365.0) ** 2
    left = 365.0 * np.sqrt(held - np.log(rng.random(n))) - age
    cost = np.where(left > until, price, 400.0)
    cost = cost - rate * np.minimum(left, until)
    return cost.mean(), cost.std() / math.sqrt(n)


@pytest.mark.parametrize("baseline", ["age replacement", "shuts"])
@pytest.mark.parametrize("age", [50.0, 250.0])
def test_each_choice_is_its_simulated_cost(age, baseline, shuts_policy):
    policy = shuts_policy if baseline == "shuts" else None
    found = BEARING.optimal_next_replacement(
        age, scheduled_outages=OUTAGES, policy=policy
    )
    for choice in found.choices[1:]:
        mean, error = simulated(
            age, choice.time, choice.price, found.cost_rate
        )
        assert abs(choice.cost - mean) < 4 * error, choice


def test_an_exponential_life_runs_to_failure():
    # Memoryless: keeping to u costs price * exp(-rate u), running to
    # failure nothing, and replacing now never pays at any age.
    rate = 1 / 200.0
    unit = NonRepairable(
        surv.Exponential.from_params([rate]),
        planned_cost=150.0,
        unplanned_cost=400.0,
    )
    found = unit.optimal_next_replacement(300.0, scheduled_outages=OUTAGES)
    assert found.cost_rate == pytest.approx(400.0 * rate, rel=1e-12)
    for time, price in OUTAGES:
        assert by_kind(found)["outage", time].cost == pytest.approx(
            price * math.exp(-rate * time), rel=1e-6
        )
    assert found.best.kind == "failure"
    assert found.best.cost == pytest.approx(0.0, abs=1e-6)
    assert by_kind(found)["planned", None].time == math.inf
    assert found.replace_from_age == math.inf


def test_replacing_now_dearer_than_planned_is_never_best():
    found = BEARING.optimal_next_replacement(1000.0, now_cost=200.0)
    assert not found.now
    assert found.best.kind == "planned"
    assert by_kind(found)["planned", None].break_even_age == math.inf


def test_a_cheap_chance_now_replaces_a_younger_part():
    # Replacing now in an outage today ($100) beats waiting for one in 150
    # days from a younger age than at the planned price.
    dear = BEARING.optimal_next_replacement(100.0, scheduled_outages=OUTAGES)
    cheap = BEARING.optimal_next_replacement(
        100.0, scheduled_outages=OUTAGES, now_cost=100.0
    )
    assert (
        by_kind(cheap)["outage", 150.0].break_even_age
        < by_kind(dear)["outage", 150.0].break_even_age
    )


def test_the_outages_come_in_time_order():
    found = BEARING.optimal_next_replacement(
        100.0, scheduled_outages=OUTAGES[::-1]
    )
    assert [c.time for c in found.choices if c.kind == "outage"] == [
        150.0,
        330.0,
    ]


def test_the_answer_is_the_same_in_hours():
    days = BEARING.optimal_next_replacement(200.0, scheduled_outages=OUTAGES)
    hourly = NonRepairable(
        surv.Weibull.from_params([365.0 * 24, 2.0]),
        planned_cost=150.0,
        unplanned_cost=400.0,
    )
    hours = hourly.optimal_next_replacement(
        200.0 * 24, scheduled_outages=[(t * 24, p) for t, p in OUTAGES]
    )
    assert hours.cost_rate == pytest.approx(days.cost_rate / 24, rel=1e-6)
    assert hours.replace_from_age == pytest.approx(
        days.replace_from_age * 24, rel=1e-6
    )
    for day, hour in zip(days.choices, hours.choices):
        assert hour.cost == pytest.approx(day.cost, rel=1e-6)
        assert hour.time == pytest.approx(day.time * 24, rel=1e-6)


def test_it_goes_through_json():
    found = BEARING.optimal_next_replacement(200.0, scheduled_outages=OUTAGES)
    plain = json.loads(json.dumps(found.to_dict()))
    assert plain["best"]["kind"] == "outage"
    assert plain["choices"][0]["break_even_age"] is None
    assert len(plain["choices"]) == 5


@pytest.mark.parametrize("age", [-1.0, math.inf, True, "200"])
def test_an_age_is_at_least_0(age):
    with pytest.raises(ValueError, match="age"):
        BEARING.optimal_next_replacement(age)


@pytest.mark.parametrize(
    "outages, error, match",
    [
        ("150", TypeError, "pairs"),
        ([(150.0,)], TypeError, "pair"),
        ([(0.0, 100.0)], ValueError, "time"),
        ([("150", 100.0)], ValueError, "time"),
        ([(150.0, -1.0)], ValueError, "price"),
        ([(150.0, math.inf)], ValueError, "price"),
    ],
)
def test_the_outages_are_checked(outages, error, match):
    with pytest.raises(error, match=match):
        BEARING.optimal_next_replacement(100.0, scheduled_outages=outages)


def test_the_other_inputs_are_checked():
    with pytest.raises(TypeError):
        BEARING.optimal_next_replacement(100.0, shuts=OUTAGES)
    with pytest.raises(ValueError, match="now_cost"):
        BEARING.optimal_next_replacement(100.0, now_cost=-1.0)
    with pytest.raises(ValueError, match="costs not set"):
        NonRepairable(LIFE, unplanned_cost=400.0).optimal_next_replacement(1.0)
    with pytest.raises(TypeError):
        BEARING.optimal_next_replacement(100.0, None)


@pytest.mark.parametrize(
    "policy", [0.99, "0.99", SHUTS, {"cost_rate": 0.99}, (297.6, 1.117)]
)
def test_a_policy_is_a_policy(policy):
    with pytest.raises(TypeError, match="policy must be the policy"):
        BEARING.optimal_next_replacement(100.0, policy=policy)


def test_a_bare_cost_rate_says_to_give_the_policy():
    with pytest.raises(TypeError, match="not its cost rate"):
        BEARING.optimal_next_replacement(100.0, policy=0.99)


@pytest.mark.parametrize("rate", [-1.0, math.inf, math.nan])
def test_a_policy_has_a_cost_rate(rate):
    with pytest.raises(ValueError, match="cost rate of policy"):
        BEARING.optimal_next_replacement(
            100.0, policy=MaintenancePolicy(297.6, rate)
        )


def test_a_life_with_no_closed_form_is_refused():
    estimate = surv.KaplanMeier.fit(LIFE.random(50))
    unit = NonRepairable(estimate, planned_cost=150.0, unplanned_cost=400.0)
    with pytest.raises(NotImplementedError, match="closed form"):
        unit.optimal_next_replacement(100.0)
