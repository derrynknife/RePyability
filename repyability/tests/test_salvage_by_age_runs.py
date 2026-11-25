"""A salvage by age in the simulated runs: each run credits ``worth(age)``
of each unit in service at its end that is up (the Python loop records
the age, ``_replicate``), discounted from the end."""

import math

import numpy as np
import pytest
import surpyval as surv

from repyability.maintenance import AgePolicy, Prices, UnitPlan
from repyability.rbd import _npv_runs

LAM, H = 1e-3, 3000.0


def worth(age):
    return 1000.0 - 0.2 * np.asarray(age)


def plan(**more):
    spec = dict(
        life=surv.Exponential.from_params([LAM]),
        repair="instant",
        prices=Prices(repair=50.0),
        salvage=worth,
    )
    spec.update(more)
    return UnitPlan(**spec)


def test_a_memoryless_unit_against_the_closed_form():
    # Repaired at once it is always up, and the age of the unit in
    # service at H is min(Exp(LAM), H): E[age] = (1 - e^{-LAM H}) / LAM.
    runs = plan().diagram().npv_distribution(H, mc_samples=4000, seed=3)
    mean_age = (1.0 - math.exp(-LAM * H)) / LAM
    hand = 1000.0 - 0.2 * mean_age
    sd_age = math.sqrt(
        # Var of min(X, H) for X ~ Exp(LAM).
        (2.0 / LAM**2) * (1.0 - math.exp(-LAM * H) * (1.0 + LAM * H))
        - mean_age**2
    )
    se = 0.2 * sd_age / math.sqrt(4000)
    assert abs(runs.salvage - hand) < 4.0 * se
    assert abs(runs.mean - (hand - 50.0 * LAM * H)) < 4.0 * float(
        np.std(runs.samples, ddof=1) / math.sqrt(4000)
    )


def test_a_plain_unit_against_expected_npv():
    # Repairs that take time: a unit down at H is credited nothing; the
    # exact salvage by age of a plain unit, discounted.
    unit = plan(
        life=surv.Weibull.from_params([800.0, 2.0]),
        repair=surv.Exponential.from_params([0.01]),
    )
    rate = 1e-4
    runs = unit.diagram().npv_distribution(
        H, discount_rate=rate, mc_samples=4000, seed=7
    )
    exact = unit.expected_npv(H, discount_rate=rate)
    assert abs(runs.mean - exact.npv) < 4.0 * float(
        np.std(runs.samples, ddof=1) / math.sqrt(4000)
    )
    assert abs(runs.salvage - exact.salvage) < 0.02 * exact.salvage


def test_a_unit_with_a_task_is_credited_too():
    # Age replacement (taking an hour, so it is priced in the runs):
    # no exact salvage by age, so the runs' is checked against the mean
    # worth of each run's age, which no replacement leaves above 600 h.
    unit = plan(
        life=surv.Weibull.from_params([800.0, 2.0]),
        tasks=(AgePolicy(interval=600.0, cost=10.0, duration=1.0),),
    )
    runs = unit.diagram().npv_distribution(H, mc_samples=500, seed=2)
    assert 1000.0 - 0.2 * 600.0 <= runs.salvage <= 1000.0


def test_seeded_runs_without_a_salvage_are_unchanged():
    with_age = plan().diagram().npv_distribution(H, mc_samples=300, seed=4)
    without = (
        plan(salvage=None)
        .diagram()
        .npv_distribution(H, mc_samples=300, seed=4)
    )
    credit = with_age.samples - without.samples
    assert np.all((credit >= 1000.0 - 0.2 * H) & (credit <= 1000.0))


def test_the_compiled_engine_is_refused_for_it():
    with pytest.raises(NotImplementedError, match="Python engine"):
        plan().diagram().npv_distribution(
            H, mc_samples=10, seed=1, engine="numba"
        )


def test_a_service_and_a_switch_credit_it_as_a_negative_cost():
    unit = plan()
    rbd = unit.diagram()
    ledger, _ = _npv_runs.priced_chunk(
        rbd,
        H,
        np.array([H]),
        _npv_runs._npv.Money(rbd, 0.0, 0.0),
        50,
        5,
        "auto",
        0,
        None,
    )
    cells = [k for k in ledger.cells if k[1] == "salvage"]
    assert cells and np.all(ledger.cells[cells[0]][0] < 0.0)
