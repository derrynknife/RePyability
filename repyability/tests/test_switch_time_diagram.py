"""The best time to switch blocks of a diagram from one way of running
them to another (``RepairableRBD.best_switch_time``), by simulation.

A one-block diagram is the one unit ``best_switch_time`` works out
numerically, so its answer is checked against that one; a block that the
load does not touch, and that hardly fails, leaves the answer where the
loaded block's economics put it; and the search is seeded and its
intervals narrow with the runs.
"""

import math

import numpy as np
import pytest
import surpyval as surv

from repyability import (
    RepairableRBD,
    SimulatedSwitchTime,
    best_switch_time,
    operating_schedule_value,
)
from repyability.rbd import routes

E = surv.Exponential.from_params
STEPS = surv.StepSchedule.from_changepoints
HORIZON = 200.0
RATE = 0.001
COST = 80.0
# Run hot (load 1.5, earning 1.5 an hour) or cool (load 1, earning 1).
ROWS = [[1.5], [1.0]]
VALUES = [1.5, 1.0]


@pytest.fixture(scope="module")
def model():
    """A Weibull life that wears out (shape about 3) and lives about a
    sixteenth as long at load 2 as at load 1."""
    rng = np.random.default_rng(3)
    load = np.repeat([1.0, 2.0], 40)
    x = 100.0 * rng.weibull(3.0, load.size) / load**2
    return surv.WeibullPH.fit(x, Z=load[:, None])


def _diagram(model, other=None):
    components = {
        "a": {
            "reliability": model,
            "repairability": E([1.0]),
            "replace_cost": COST,
            "covariates": STEPS([0.0], [[1.0]]),
        }
    }
    edges = [("s", "a"), ("a", "t")]
    if other is not None:
        components["b"] = other
        edges = [("s", "a"), ("a", "b"), ("b", "t")]
    return RepairableRBD(edges, components)


def _search(rbd, **more):
    options = dict(discount_rate=RATE, mc_samples=400, seed=11, scan=9)
    options.update(more)
    return rbd.best_switch_time({"a": ROWS}, VALUES, HORIZON, **options)


@pytest.fixture(scope="module")
def one_block(model):
    return _search(_diagram(model))


def test_one_block_agrees_with_the_one_unit_answer(model, one_block):
    """Run hot while the unit is new, then cool: the numerical answer for
    one unit switches at about 21 hours. The diagram of that unit finds
    the same choice, each end's NPV within its interval, and the gain
    over switching at once within the interval on the difference."""
    one = best_switch_time(
        model,
        ROWS,
        VALUES,
        HORIZON,
        failure_cost=COST,
        down_time=E([1.0]),
        discount_rate=RATE,
        tolerance=1e-4,
    )
    got = one_block
    assert isinstance(got, SimulatedSwitchTime)
    assert one.choice == got.choice == "switch"
    n = got.n_simulations

    def within(samples, want):
        se = np.std(samples, ddof=1) / math.sqrt(n)
        assert abs(samples.mean() - want) < 3.5 * se

    within(got.never_switch_samples, one.never_switch_npv)
    within(got.switch_at_once_samples, one.switch_at_once_npv)
    gain = got.difference_interval("switch at once", confidence=0.99)
    assert gain.lower <= one.npv - one.switch_at_once_npv <= gain.upper
    assert gain.lower > 0.0
    # The time found is as good, in the one-unit numbers, as the best to
    # within the noise of the difference.
    at_found = operating_schedule_value(
        model,
        [0.0, got.switch_time],
        ROWS,
        VALUES,
        HORIZON,
        failure_cost=COST,
        down_time=E([1.0]),
        discount_rate=RATE,
    )
    assert one.npv - at_found.npv < 2.0 * gain.standard_error
    # The scan audits the answer: every time valued, in order, ends
    # included, its best the one returned.
    assert got.times[0] == 0.0 and got.times[-1] == HORIZON
    assert got.times == sorted(got.times)
    assert got.npv == max(got.npvs)
    assert got.npvs[got.times.index(got.switch_time)] == got.npv


def test_a_block_the_load_does_not_touch_leaves_the_answer(model, one_block):
    """In series with a block that is not on the schedule and hardly
    ever fails, the loaded block's runs are the same (its streams are
    named as before), so the switch time is the one-block diagram's."""
    other = {"reliability": E([1e-6]), "repairability": E([10.0])}
    got = _search(_diagram(model, other))
    assert got.switch_time == one_block.switch_time
    assert got.choice == one_block.choice
    assert got.npv == pytest.approx(one_block.npv, rel=1e-3)


def test_seeded_searches_repeat_and_narrow_with_more_runs(model):
    rbd = _diagram(model)
    few = _search(rbd, mc_samples=60, scan=3)
    again = _search(rbd, mc_samples=60, scan=3)
    assert few.times == again.times and few.npvs == again.npvs
    np.testing.assert_array_equal(few.samples, again.samples)
    more = _search(rbd, mc_samples=240, scan=3)
    for against in ("never switch", "switch at once"):
        narrow = more.difference_interval(against)
        wide = few.difference_interval(against)
        assert narrow.upper - narrow.lower < wide.upper - wide.lower
    assert few.to_dict()["choice"] == few.choice


def test_value_rates_split_at_the_switch(model):
    """With no costs and a block that never fails, the NPV is the value
    alone: ``v1`` up to the switch and ``v2`` after, discounted."""
    rbd = RepairableRBD(
        [("s", "a"), ("a", "t")],
        {
            "a": {
                "reliability": model,
                "repairability": E([1.0]),
                "covariates": STEPS([0.0], [[1e-9]]),
            }
        },
    )
    # A load so light the unit outlives the horizon in every run.
    got = rbd.best_switch_time(
        {"a": [[-40.0], [-40.0]]},
        [2.0, 1.0],
        HORIZON,
        discount_rate=RATE,
        mc_samples=20,
        seed=1,
        scan=3,
    )
    assert got.choice == "never switch"
    want = 2.0 * -math.expm1(-RATE * HORIZON) / RATE
    assert got.npv == pytest.approx(want, rel=1e-12)
    at_once = 1.0 * -math.expm1(-RATE * HORIZON) / RATE
    assert got.switch_at_once_npv == pytest.approx(at_once, rel=1e-12)


def test_refusals_say_what_to_give(model):
    plain = RepairableRBD(
        [("s", "a"), ("a", "t")],
        {"a": {"reliability": E([0.01]), "repairability": E([1.0])}},
    )
    with pytest.raises(routes.Refused, match="covariate schedule"):
        plain.best_switch_time({"a": ROWS}, VALUES, HORIZON)
    assert plain.analysis_routes()["best_switch_time"].route == (
        routes.REFUSED
    )
    rbd = _diagram(model, {"reliability": E([0.01]), "repairability": E([1])})
    assert rbd.analysis_routes()["best_switch_time"].route == (
        routes.SIMULATED
    )
    with pytest.raises(ValueError, match="'b' does not run on a covariate"):
        rbd.best_switch_time({"b": ROWS}, VALUES, HORIZON)
    with pytest.raises(ValueError, match="two rows"):
        rbd.best_switch_time({"a": [[1.0]]}, VALUES, HORIZON)
    with pytest.raises(ValueError, match="value_rates must be two"):
        rbd.best_switch_time({"a": ROWS}, [1.0], HORIZON)
    with pytest.raises(ValueError, match="covariates must be a dict"):
        rbd.best_switch_time(ROWS, VALUES, HORIZON)
    with pytest.raises(ValueError, match="scan"):
        rbd.best_switch_time({"a": ROWS}, VALUES, HORIZON, scan=2)
    with pytest.raises(ValueError, match="against"):
        _search(rbd, mc_samples=5, scan=3).difference_interval("sometimes")
