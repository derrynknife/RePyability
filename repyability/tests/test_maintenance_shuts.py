"""The best replacement policy with shut, between-shut and failure prices
(``replacement_policies``; #319, UC-7 of
``repyability/maintenance/README.md``)."""

import json
import math

import numpy as np
import pytest
import surpyval as surv
from surpyval import ExactEventTime

from repyability import (
    RepairableRBD,
    ReplacementPolicies,
    replacement_policies,
)
from repyability.rbd._condition_replacement import condition_cycle

LIFE = surv.Weibull.from_params([365.0, 2.0])
PRICES = {"shut_cost": 100.0, "planned_cost": 150.0, "failure_cost": 400.0}


@pytest.fixture(scope="module")
def bearing():
    return replacement_policies(LIFE, shut_every=180.0, **PRICES)


def simulated(
    scale,
    shape,
    every,
    shuts,
    older_than,
    prices,
    n=20_000,
    horizon=200_000.0,
    seed=0,
):
    """The long-run cost a unit of time of replacing at every ``shuts``-th
    shut if at least ``older_than`` old (and at failure), simulated
    plainly: a Weibull part, shuts every ``every``."""
    rng = np.random.default_rng(seed)
    period = shuts * every
    installed = np.zeros(n)
    cost = np.zeros(n)
    live = np.ones(n, dtype=bool)
    while live.any():
        i = np.flatnonzero(live)
        start = installed[i]
        life = scale * (-np.log(rng.random(i.size))) ** (1 / shape)
        failure = start + life
        # The first decision shut at which the part is old enough.
        first = np.floor(start / period) * period + period
        due = np.maximum(
            np.ceil((start + older_than) / period) * period, first
        )
        ends = np.minimum(failure, due)
        paid = np.where(
            failure <= due, prices["failure_cost"], prices["shut_cost"]
        )
        over = ends >= horizon
        cost[i] += np.where(over, 0.0, paid)
        installed[i] = np.where(over, horizon, ends)
        live[i[over]] = False
    return cost.sum() / (n * horizon)


def test_the_bearing_with_its_shuts(bearing):
    # UC-7: the families as worked out by hand.
    assert isinstance(bearing, ReplacementPolicies)
    assert bearing.run_to_failure.cost_rate == pytest.approx(
        400.0 / (365.0 * math.gamma(1.5)), rel=1e-12
    )
    assert bearing.between_shuts.age == pytest.approx(297.64, abs=0.05)
    assert bearing.between_shuts.cost_rate == pytest.approx(1.1170, abs=1e-4)
    assert bearing.every_shut.shuts == 1
    assert bearing.every_shut.cost_rate == pytest.approx(1.0555, abs=1e-4)
    best = bearing.best
    assert (best.policy, best.shuts) == ("at a shut if older", 1)
    assert best.older_than == pytest.approx(138.0, abs=2.0)
    assert best.cost_rate == pytest.approx(0.9903, abs=1e-4)
    assert bearing.saving == pytest.approx(
        bearing.run_to_failure.cost_rate - best.cost_rate
    )


def test_every_shut_is_block_replacement(bearing):
    # The same policy through RePyability's block replacement.
    block = RepairableRBD(
        [("s", "b"), ("b", "t")],
        {
            "b": {
                "reliability": LIFE,
                "repairability": "instant",
                "replace_cost": 400.0,
                "preventive": {
                    "interval": 180.0,
                    "policy": "block",
                    "cost": 100.0,
                },
            }
        },
    )
    assert bearing.every_shut.cost_rate == pytest.approx(
        block.expected_cost_rate(), rel=1e-6
    )


@pytest.mark.parametrize(
    "shuts, older_than", [(1, 138.0), (2, 300.0), (1, 0.0)]
)
def test_the_shut_policies_are_their_simulation(shuts, older_than):
    instant = ExactEventTime.from_params([0.0])
    cycle = condition_cycle(
        LIFE, instant, None, shuts * 180.0, None, older_than=older_than
    )
    exact = (100.0 * cycle.replaced + 400.0 * cycle.failures) / cycle.length
    plain = simulated(365.0, 2.0, 180.0, shuts, older_than, PRICES)
    # The simulation stops at a horizon, which biases it a little low.
    assert plain == pytest.approx(exact, rel=5e-3)


def test_the_age_rule_is_the_threshold_rule_for_a_rising_hazard():
    instant = ExactEventTime.from_params([0.0])
    by_age = condition_cycle(
        LIFE, instant, None, 180.0, None, older_than=138.0
    )
    threshold = 1.0 - float(LIFE.sf(318.0)) / float(LIFE.sf(138.0))
    by_chance = condition_cycle(LIFE, instant, None, 180.0, threshold)
    assert by_age.length == pytest.approx(by_chance.length, rel=1e-3)
    assert by_age.failures == pytest.approx(by_chance.failures, rel=1e-3)


def test_with_no_ageing_no_planned_replacement_pays():
    found = replacement_policies(
        surv.Exponential.from_params([1 / 200.0]), shut_every=300.0, **PRICES
    )
    assert found.best.policy == "run to failure"
    assert found.between_shuts.age == math.inf
    assert found.every_shut.cost_rate > found.run_to_failure.cost_rate
    assert found.saving == 0.0


def test_a_shut_dearer_than_a_failure_never_pays():
    found = replacement_policies(
        LIFE,
        shut_every=180.0,
        shut_cost=500.0,
        planned_cost=600.0,
        failure_cost=400.0,
    )
    assert found.best.policy == "run to failure"


def test_the_answer_does_not_depend_on_the_unit_of_time(bearing):
    hours = replacement_policies(
        surv.Weibull.from_params([365.0 * 24, 2.0]),
        shut_every=180.0 * 24,
        **PRICES,
    )
    assert hours.best.cost_rate * 24 == pytest.approx(
        bearing.best.cost_rate, rel=1e-6
    )
    assert hours.best.older_than / 24 == pytest.approx(
        bearing.best.older_than, rel=1e-3
    )
    assert hours.between_shuts.age / 24 == pytest.approx(
        bearing.between_shuts.age, rel=1e-6
    )


def test_the_result_goes_through_json(bearing):
    plain = json.loads(json.dumps(bearing.to_dict()))
    assert plain["best"]["policy"] == "at a shut if older"
    assert set(plain) == {
        "best",
        "run_to_failure",
        "between_shuts",
        "every_shut",
        "at_a_shut_if_older",
        "saving",
    }


@pytest.mark.parametrize(
    "keywords, error, match",
    [
        ({"shut_every": 0}, ValueError, "shut_every"),
        ({"shut_every": "180"}, ValueError, "shut_every"),
        ({"shut_cost": -1}, ValueError, "shut_cost"),
        ({"planned_cost": math.inf}, ValueError, "planned_cost"),
        ({"failure_cost": True}, ValueError, "failure_cost"),
        ({"life": surv.Weibull}, TypeError, "life"),
        ({"life": None}, TypeError, "life"),
    ],
)
def test_the_inputs_are_checked(keywords, error, match):
    arguments = {"life": LIFE, "shut_every": 180.0, **PRICES} | keywords
    life = arguments.pop("life")
    with pytest.raises(error, match=match):
        replacement_policies(life, **arguments)
