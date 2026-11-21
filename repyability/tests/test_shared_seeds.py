"""A result made of several runs on one seed, called without a seed, draws
that seed from numpy's global RNG (``rbd._streams.shared_seed``), so
``np.random.seed`` makes the call repeatable, and the result is the seeded
call of the seed drawn."""

import numpy as np
import pytest
import surpyval as surv
from surpyval import JointFrailty

from repyability import RepairableRBD, RepairOrReplace
from repyability.rbd import _contract
from repyability.rbd._streams import shared_seed

E = surv.Exponential.from_params
STEPS = surv.StepSchedule.from_changepoints
LINE = [("s", "p"), ("p", "t")]


def unseeded(call, k):
    """``call()`` with numpy's global RNG seeded with ``k`` (restored
    after), and the seed an unseeded call draws from it."""
    kept = np.random.get_state()
    try:
        np.random.seed(k)
        drawn = shared_seed(None)
        np.random.seed(k)
        return call(), drawn
    finally:
        np.random.set_state(kept)


def pump(rate=1e-3):
    return RepairableRBD(
        LINE,
        {
            "p": {
                "reliability": E([rate]),
                "repairability": E([0.1]),
                "repair_cost": 500.0,
            }
        },
    )


def test_npv_distribution_draws_its_seed_from_the_global_rng():
    rbd = pump()
    out, drawn = unseeded(
        lambda: rbd.npv_distribution(500.0, value_rate=1.0, mc_samples=20), 3
    )
    assert out.seed == drawn
    seeded = rbd.npv_distribution(
        500.0, value_rate=1.0, mc_samples=20, seed=drawn
    )
    np.testing.assert_array_equal(out.samples, seeded.samples)


def test_compare_npv_draws_its_seed_from_the_global_rng():
    rbd, other = pump(), pump(2e-3)
    out, drawn = unseeded(
        lambda: rbd.compare_npv(other, 500.0, value_rate=1.0, mc_samples=20),
        4,
    )
    assert out.seed == drawn
    seeded = rbd.compare_npv(
        other, 500.0, value_rate=1.0, mc_samples=20, seed=drawn
    )
    assert out.mean == seeded.mean


def test_best_switch_time_draws_its_seed_from_the_global_rng():
    rng = np.random.default_rng(3)
    load = np.repeat([1.0, 2.0], 40)
    x = 100.0 * rng.weibull(3.0, load.size) / load**2
    model = surv.WeibullPH.fit(x, Z=load[:, None])
    rbd = RepairableRBD(
        [("s", "a"), ("a", "t")],
        {
            "a": {
                "reliability": model,
                "repairability": E([1.0]),
                "replace_cost": 80.0,
                "covariates": STEPS([0.0], [[1.0]]),
            }
        },
    )
    rows, values = {"a": [[1.5], [1.0]]}, [1.5, 1.0]
    options = dict(discount_rate=1e-3, mc_samples=20, scan=3)
    out, drawn = unseeded(
        lambda: rbd.best_switch_time(rows, values, 200.0, **options), 5
    )
    assert out.seed == drawn
    seeded = rbd.best_switch_time(rows, values, 200.0, seed=drawn, **options)
    np.testing.assert_array_equal(out.samples, seeded.samples)


@pytest.fixture(scope="module")
def fleet():
    truth = JointFrailty.from_params((100.0, 2.0), (400.0, 2.0), 0.5, 1.0)
    model = JointFrailty.fit(**truth.random(60, end=250.0, random_state=2))
    return RepairOrReplace(
        model, repair_cost=1.0, replacement_cost=10.0, failure_cost=2.0
    )


def test_unit_decisions_draw_their_seed_from_the_global_rng(fleet):
    ages = {0: 120.0, 1: 120.0}
    out, drawn = unseeded(lambda: fleet.unit_decisions(ages, mc_samples=30), 6)
    seeded = fleet.unit_decisions(ages, mc_samples=30, seed=drawn)
    # repr, as a rule a unit cannot take has a cost rate of nan.
    assert repr(out) == repr(seeded)
    one, drawn = unseeded(
        lambda: fleet.unit_decision(120.0, unit=0, mc_samples=30), 7
    )
    seeded = fleet.unit_decision(120.0, unit=0, mc_samples=30, seed=drawn)
    assert repr(one) == repr(seeded)


def test_a_plan_request_draws_its_seed_from_the_global_rng():
    spec = {
        "schema": _contract.SCHEMA,
        "diagram": pump().to_dict(),
        "analysis": "npv_distribution",
        "inputs": {"horizon": 500.0},
    }
    request, drawn = unseeded(lambda: _contract._checked(spec), 8)
    assert request["inputs"]["seed"] == drawn
