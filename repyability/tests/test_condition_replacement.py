"""Replacement on condition (#96): a component inspected at every multiple
of an interval, and replaced when it is more likely than a threshold to
fail before the next inspection, given its age."""

import numpy as np
import pytest
import surpyval as surv

from repyability import RepairableRBD
from repyability.rbd import routes as r
from repyability.rbd.repairable_rbd import _failures_between

E, W, L = (
    surv.Exponential.from_params,
    surv.Weibull.from_params,
    surv.LogNormal.from_params,
)
X = surv.ExactEventTime.from_params


def single(spec, **options):
    return RepairableRBD([("s", "c"), ("c", "t")], {"c": spec}, **options)


def on_condition(threshold, interval=20.0, **more):
    return {
        "interval": interval,
        "policy": "condition",
        "threshold": threshold,
        **more,
    }


WORN = {"reliability": W([100.0, 3.0]), "repairability": L([1.5, 0.5])}


def test_a_timeline_worked_by_hand():
    # A unit that fails at age 50, inspected every 20: at age 20 it cannot
    # fail before the next inspection (at age 40 < 50), at age 40 it
    # surely will, so it is replaced then, taking 2. Put back at 42, it is
    # 18 at the inspection at 60 and 38 at the one at 80, and so on: down
    # [40, 42], [80, 82], [120, 122], [160, 162] in 200, never failing.
    rbd = single(
        {
            "reliability": X(50.0),
            "repairability": X(5.0),
            "replace_cost": 100.0,
            "preventive": on_condition(
                0.5, duration=X(2.0), cost=10.0, inspection_cost=1.0
            ),
        }
    )
    result = rbd.availability(200.0, mc_samples=2, seed=1)
    assert result.mean_availability_interval().estimate == pytest.approx(
        192.0 / 200.0
    )
    rbd.initialize_event_queue(200.0)
    changes = [rbd.next_event()]
    while changes[-1][0] < 200.0:
        changes.append(rbd.next_event())
    assert changes == [
        (40.0, False),
        (42.0, True),
        (80.0, False),
        (82.0, True),
        (120.0, False),
        (122.0, True),
        (160.0, False),
        (162.0, True),
        (200.0, True),
    ]
    # Four replacements, and nine inspections (at 20, 40, ..., 180: the
    # one at 200 is outside the window); no failure.
    cost = rbd.cost(200.0, mc_samples=2, seed=1)
    assert cost.by_category["preventive"] == 4 * 10.0
    assert cost.by_category["inspection"] == 9 * 1.0
    assert cost.by_category["replace"] == 0.0
    used = rbd.spares_demand(200.0, method="simulate", mc_samples=2, seed=1)
    assert used["c"].probabilities[4] == 1.0
    # Never replaced, it fails at 50, 105, 160 (a repair takes 5).
    never = single(
        dict(
            WORN,
            **{
                "reliability": X(50.0),
                "repairability": X(5.0),
                "preventive": on_condition(1.0),
            },
        )
    )
    never.initialize_event_queue(200.0)
    changes = [never.next_event()]
    while changes[-1][0] < 200.0:
        changes.append(never.next_event())
    assert changes[:4] == [
        (50.0, False),
        (55.0, True),
        (105.0, False),
        (110.0, True),
    ]


def test_a_failure_at_an_inspection_comes_first():
    # A unit that fails at age 40 exactly, never replaced (a threshold of
    # 1), is inspected at 20 and fails at 40, when it would be inspected;
    # the new unit put in at once is not inspected there. So inspections
    # at 20, 60, 100, 140 and 180, and failures at 40, 80, 120 and 160.
    spec = {
        "reliability": X(40.0),
        "repairability": "instant",
        "replace_cost": 5.0,
        "preventive": on_condition(1.0, cost=2.0, inspection_cost=1.0),
    }
    cost = single(spec).cost(200.0, mc_samples=2, seed=1)
    assert cost.by_category["replace"] == 4 * 5.0
    assert cost.by_category["inspection"] == 5 * 1.0
    assert cost.by_category["preventive"] == 0.0
    # With any threshold below 1 it is replaced at every inspection: at age
    # 20 it is sure to fail before the next. Never failing, it is replaced
    # at 20, 40, ..., 180.
    spec["preventive"] = on_condition(0.5, cost=2.0, inspection_cost=1.0)
    cost = single(spec).cost(200.0, mc_samples=2, seed=1)
    assert cost.by_category["replace"] == 0.0
    assert cost.by_category["inspection"] == 9 * 1.0
    assert cost.by_category["preventive"] == 9 * 2.0


def test_a_threshold_of_zero_is_block_replacement():
    # Any chance of failing before the next inspection replaces the unit:
    # every inspection replaces it, as block replacement does, draw for
    # draw.
    maintenance = {"duration": E([0.5]), "cost": 7.0}
    spec = dict(WORN, replace_cost=50.0)
    condition = single(dict(spec, preventive=on_condition(0.0, **maintenance)))
    block = single(
        dict(
            spec,
            preventive={"interval": 20.0, "policy": "block", **maintenance},
        )
    )
    a = condition.cost(2000.0, mc_samples=200, seed=3)
    b = block.cost(2000.0, mc_samples=200, seed=3)
    assert np.array_equal(a.samples, b.samples)
    a = condition.availability(2000.0, mc_samples=200, seed=3)
    b = block.availability(2000.0, mc_samples=200, seed=3)
    assert np.array_equal(a.timeline, b.timeline)
    assert np.array_equal(a.availability, b.availability)


def test_a_threshold_of_one_is_run_to_failure():
    # Nothing is more likely than certain: the unit is never replaced, and
    # fails and is repaired draw for draw as with no maintenance. Only the
    # inspections are charged.
    spec = dict(WORN, repair_cost=100.0)
    inspected = single(
        dict(spec, preventive=on_condition(1.0, inspection_cost=0.0))
    )
    unmaintained = single(spec)
    difference = inspected.compare(
        unmaintained, 2000.0, mc_samples=200, seed=3
    )
    assert difference.estimate == 0.0 and difference.standard_error == 0.0
    difference = inspected.compare(
        unmaintained, 2000.0, mc_samples=200, seed=3, quantity="cost"
    )
    assert difference.estimate == 0.0 and difference.standard_error == 0.0


@pytest.mark.parametrize("threshold, replaced", [(0.2, False), (0.1, True)])
def test_a_constant_hazard_does_not_age(threshold, replaced):
    # With a constant failure rate, the chance of failing before the next
    # inspection is 1 - exp(-rate * interval) = 0.18 at every age: the
    # unit is replaced at every inspection or at none, never early.
    rate, interval = 0.01, 20.0
    assert -np.expm1(-rate * interval) == pytest.approx(0.1813, abs=1e-4)
    spec = {
        "reliability": E([rate]),
        "repairability": E([0.5]),
        "preventive": on_condition(threshold, interval, cost=1.0),
    }
    cost = single(spec).cost(1000.0, mc_samples=100, seed=5)
    replacements = cost.by_category["preventive"]
    if replaced:
        # One at each inspection it is up for: nearly all 49.
        assert 45.0 < replacements <= 49.0
    else:
        assert replacements == 0.0
    # Memoryless: replacing it or not, it is up as often.
    availability = single(spec).availability(20_000.0, mc_samples=20, seed=5)
    assert availability.mean_availability_interval().estimate == (
        pytest.approx(0.5 / (0.5 + 0.01), abs=2e-3)
    )


def simulate_directly(interval, threshold, until, rng):
    """One component under the policy, simulated from its definition, in
    closed forms: a Weibull(100, 3) life, a LogNormal(1.5, 0.5) repair and
    an Exponential(0.5) maintenance time. Its up time, failures and
    replacements in [0, until]."""

    def sf(x):
        return np.exp(-((x / 100.0) ** 3))

    t, up, failures, replacements = 0.0, 0.0, 0, 0
    while True:
        renewed = t
        failure = renewed + 100.0 * (-np.log(rng.uniform())) ** (1.0 / 3.0)
        inspection = (np.floor(renewed / interval) + 1.0) * interval
        replaced = None
        while inspection < min(failure, until):
            age = inspection - renewed
            if 1.0 - sf(age + interval) / sf(age) > threshold:
                replaced = inspection
                break
            inspection += interval
        end = replaced if replaced is not None else failure
        if end >= until:
            return up + until - renewed, failures, replacements
        up += end - renewed
        if replaced is None:
            failures += 1
            t = end + np.exp(1.5 + 0.5 * rng.standard_normal())
        else:
            replacements += 1
            t = end + rng.exponential(1.0 / 0.5)
        if t >= until:
            return up, failures, replacements


def test_the_long_run_values_match_a_direct_simulation():
    rbd = single(
        {
            "reliability": W([100.0, 3.0]),
            "repairability": L([1.5, 0.5]),
            "repair_cost": 1.0,
            "preventive": on_condition(0.2, duration=E([0.5]), cost=1.0),
        }
    )
    until = 20_000.0
    result = rbd.availability(until, mc_samples=60, seed=5)
    estimate = result.mean_availability_interval()
    rng = np.random.default_rng(7)
    direct = np.array(
        [simulate_directly(20.0, 0.2, until, rng) for _ in range(200)]
    )
    up = direct[:, 0] / until
    spread = np.hypot(estimate.standard_error, up.std() / np.sqrt(len(up)))
    assert abs(estimate.estimate - up.mean()) < 4.0 * spread
    # Each failure costs 1 to repair, and each replacement 1.
    for category, column in (("repair", 1), ("preventive", 2)):
        counts = direct[:, column]
        spread = counts.std() * np.sqrt(1.0 / 60 + 1.0 / len(counts))
        assert abs(result.cost.by_category[category] - counts.mean()) < (
            4.0 * spread
        )


def test_the_chance_of_failing_before_the_next_inspection():
    life = W([100.0, 3.0])
    ages = np.array([0.0, 20.0, 40.0, 60.0])
    expected = 1.0 - life.sf(ages[1:]) / life.sf(ages[:-1])
    np.testing.assert_allclose(
        _failures_between(life, ages), expected, rtol=1e-12
    )
    # From the cumulative hazard, precise when small; from the survival
    # function for a model without one; certain once it cannot survive.
    tiny = _failures_between(life, np.array([0.0, 1e-3]))[0]
    assert tiny == pytest.approx((1e-5) ** 3, rel=1e-9)

    class NoHazard:
        def sf(self, x):
            return life.sf(x)

    np.testing.assert_allclose(
        _failures_between(NoHazard(), ages), expected, rtol=1e-12
    )
    assert _failures_between(
        X(50.0), np.array([40.0, 60.0, 80.0])
    ).tolist() == [
        1.0,
        1.0,
    ]


def test_the_values_over_time_are_numerical_and_the_spares_simulated():
    # The long-run values are numerical (see test_condition_long_run.py),
    # and so are those over time (#161, see test_condition_over_time.py);
    # the spares are counted by simulation.
    rbd = single(dict(WORN, replace_cost=10.0, preventive=on_condition(0.1)))
    assert 0.0 < float(rbd.point_availability([10.0])[0]) <= 1.0
    assert 0.0 < float(np.ravel(rbd.mission_availability(10.0))[0]) <= 1.0
    with pytest.raises(NotImplementedError, match="replaced on condition"):
        rbd.spares_demand(100.0)
    report = rbd.analysis_routes()
    assert report["mean_availability"].route == r.NUMERICAL
    assert "replacement on condition" in report["mean_availability"].reason
    assert report["point_availability"].route == r.NUMERICAL
    assert report["spares_demand"].route == r.REFUSED
    assert report["availability"].route == r.SIMULATED
    assert report["availability"].engine == "python"
    # Held working, it needs no maintenance at all.
    held = RepairableRBD(
        [("s", "a"), ("s", "c"), ("a", "t"), ("c", "t")],
        {"a": WORN, "c": dict(WORN, preventive=on_condition(0.1))},
    )
    simulated = held.availability(
        500.0, working_nodes=["c"], mc_samples=10, seed=1
    )
    assert simulated.mean_availability_interval().estimate == 1.0


@pytest.mark.parametrize(
    "preventive, message",
    [
        ({"interval": 10.0, "policy": "condition"}, "needs a threshold"),
        (on_condition(1.5), r"in \[0, 1\]"),
        (on_condition(-0.1), r"in \[0, 1\]"),
        (on_condition(True), r"in \[0, 1\]"),
        (on_condition("half"), r"in \[0, 1\]"),
        ({"interval": 10.0, "threshold": 0.5}, "only to the 'condition'"),
        (
            {"interval": 10.0, "policy": "block", "inspection_cost": 1.0},
            "only to the 'condition'",
        ),
        ({"interval": 10.0, "policy": "cbm"}, "'age', 'block' or 'condition'"),
    ],
)
def test_the_spec_is_checked(preventive, message):
    with pytest.raises(ValueError, match=message):
        single(dict(WORN, preventive=preventive))


def test_the_policy_is_saved():
    rbd = single(
        dict(
            WORN,
            replace_cost=10.0,
            preventive=on_condition(
                0.15,
                duration=E([0.5]),
                cost=4.0,
                inspection_cost=L([0.0, 0.3]),
            ),
        )
    )
    loaded = RepairableRBD.from_json(rbd.to_json())
    schedule = loaded._preventive["c"]
    assert (schedule.policy, schedule.threshold) == ("condition", 0.15)
    a = rbd.cost(500.0, mc_samples=50, seed=2)
    b = loaded.cost(500.0, mc_samples=50, seed=2)
    assert np.array_equal(a.samples, b.samples)
