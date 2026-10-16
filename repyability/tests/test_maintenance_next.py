"""When to replace a part in service next
(``NonRepairable.optimal_next_replacement``; #319, UC-12 of
``repyability/maintenance/README.md``)."""

import json
import math

import numpy as np
import pytest
import surpyval as surv

from repyability import NextReplacement, NonRepairable, replacement_policies

LIFE = surv.Weibull.from_params([365.0, 2.0])
BEARING = NonRepairable(LIFE, planned_cost=150.0, unplanned_cost=400.0)
SHUTS = [(150.0, 100.0), (330.0, 100.0)]


@pytest.fixture(scope="module")
def uc12():
    return BEARING.optimal_next_replacement(
        200.0, shuts=SHUTS, shut_every=180.0, shut_cost=100.0
    )


def by_kind(found):
    return {
        (choice.kind, choice.time if choice.kind == "shut" else None): choice
        for choice in found.choices
    }


def test_the_bearing_at_200_days(uc12):
    # UC-12's hand-worked answer, valued at UC-7's best long-run cost.
    assert isinstance(uc12, NextReplacement)
    assert uc12.cost_rate == pytest.approx(
        replacement_policies(
            LIFE,
            shut_every=180.0,
            shut_cost=100.0,
            planned_cost=150.0,
            failure_cost=400.0,
        ).best.cost_rate,
        rel=1e-12,
    )
    choices = by_kind(uc12)
    assert choices["now", None].cost == 150.0
    assert choices["shut", 150.0].cost == pytest.approx(124.6, abs=0.05)
    assert choices["shut", 330.0].cost == pytest.approx(178.5, abs=0.05)
    assert choices["planned", None].cost == pytest.approx(142.8, abs=0.05)
    assert choices["planned", None].time == pytest.approx(63.9, abs=0.1)
    assert choices["failure", None].cost == pytest.approx(210.4, abs=0.05)
    assert not uc12.now
    assert uc12.best is choices["shut", 150.0]
    evens = {key: choice.break_even_age for key, choice in choices.items()}
    assert evens["shut", 150.0] == pytest.approx(256.1, abs=0.05)
    assert evens["shut", 330.0] == pytest.approx(153.0, abs=0.05)
    assert evens["planned", None] == pytest.approx(263.9, abs=0.05)
    assert evens["failure", None] == pytest.approx(86.4, abs=0.05)
    assert uc12.replace_from_age == evens["planned", None]


@pytest.mark.parametrize("age", [0.0, 100.0, 200.0, 290.0])
def test_with_no_shuts_it_is_the_age_replacement(age):
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


@pytest.mark.parametrize("age", [50.0, 250.0])
def test_each_choice_is_its_simulated_cost(age):
    found = BEARING.optimal_next_replacement(age, shuts=SHUTS)
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
    found = unit.optimal_next_replacement(300.0, shuts=SHUTS)
    assert found.cost_rate == pytest.approx(400.0 * rate, rel=1e-12)
    for time, price in SHUTS:
        assert by_kind(found)["shut", time].cost == pytest.approx(
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
    # Replacing now in a shut today ($100) beats waiting for one in 150
    # days from a younger age than at the planned price.
    dear = BEARING.optimal_next_replacement(100.0, shuts=SHUTS)
    cheap = BEARING.optimal_next_replacement(
        100.0, shuts=SHUTS, now_cost=100.0
    )
    assert (
        by_kind(cheap)["shut", 150.0].break_even_age
        < by_kind(dear)["shut", 150.0].break_even_age
    )


def test_the_shuts_come_in_time_order():
    found = BEARING.optimal_next_replacement(100.0, shuts=SHUTS[::-1])
    assert [c.time for c in found.choices if c.kind == "shut"] == [
        150.0,
        330.0,
    ]


def test_the_answer_is_the_same_in_hours():
    days = BEARING.optimal_next_replacement(200.0, shuts=SHUTS)
    hourly = NonRepairable(
        surv.Weibull.from_params([365.0 * 24, 2.0]),
        planned_cost=150.0,
        unplanned_cost=400.0,
    )
    hours = hourly.optimal_next_replacement(
        200.0 * 24, shuts=[(t * 24, p) for t, p in SHUTS]
    )
    assert hours.cost_rate == pytest.approx(days.cost_rate / 24, rel=1e-6)
    assert hours.replace_from_age == pytest.approx(
        days.replace_from_age * 24, rel=1e-6
    )
    for day, hour in zip(days.choices, hours.choices):
        assert hour.cost == pytest.approx(day.cost, rel=1e-6)
        assert hour.time == pytest.approx(day.time * 24, rel=1e-6)


def test_it_goes_through_json():
    found = BEARING.optimal_next_replacement(200.0, shuts=SHUTS)
    plain = json.loads(json.dumps(found.to_dict()))
    assert plain["best"]["kind"] == "shut"
    assert plain["choices"][0]["break_even_age"] is None
    assert len(plain["choices"]) == 5


@pytest.mark.parametrize("age", [-1.0, math.inf, True, "200"])
def test_an_age_is_at_least_0(age):
    with pytest.raises(ValueError, match="age"):
        BEARING.optimal_next_replacement(age)


@pytest.mark.parametrize(
    "shuts, error, match",
    [
        ("150", TypeError, "pairs"),
        ([(150.0,)], TypeError, "pair"),
        ([(0.0, 100.0)], ValueError, "time"),
        ([("150", 100.0)], ValueError, "time"),
        ([(150.0, -1.0)], ValueError, "price"),
        ([(150.0, math.inf)], ValueError, "price"),
    ],
)
def test_the_shuts_are_checked(shuts, error, match):
    with pytest.raises(error, match=match):
        BEARING.optimal_next_replacement(100.0, shuts=shuts)


def test_the_other_inputs_are_checked():
    with pytest.raises(ValueError, match="together"):
        BEARING.optimal_next_replacement(100.0, shut_every=180.0)
    with pytest.raises(ValueError, match="now_cost"):
        BEARING.optimal_next_replacement(100.0, now_cost=-1.0)
    with pytest.raises(ValueError, match="costs not set"):
        NonRepairable(LIFE, unplanned_cost=400.0).optimal_next_replacement(1.0)
    with pytest.raises(TypeError):
        BEARING.optimal_next_replacement(100.0, None)


def test_a_life_with_no_closed_form_is_refused():
    estimate = surv.KaplanMeier.fit(LIFE.random(50))
    unit = NonRepairable(estimate, planned_cost=150.0, unplanned_cost=400.0)
    with pytest.raises(NotImplementedError, match="closed form"):
        unit.optimal_next_replacement(100.0)
