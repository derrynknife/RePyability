"""surpyval's fitted degradation processes as lives, and replacement on
condition by the measured level (#271)."""

import numpy as np
import pytest
import surpyval as surv
from scipy import stats

from repyability import (
    NodeState,
    NonRepairable,
    NonRepairableRBD,
    RepairableRBD,
)
from repyability.rbd._degradation import level_after
from repyability.rbd.serialisation import rbd_from_json, rbd_to_json
from repyability.tests.catalogue import wear

EDGES = [("s", "a"), ("a", "t")]
REPAIR = surv.Exponential.from_params([1 / 20.0])


def drift():
    """A Wiener process fitted to 10 units' readings; it fails at 8."""
    g = np.random.default_rng(3)
    hours = np.tile(np.arange(1.0, 11.0) * 30.0, 10)
    units = np.repeat(np.arange(10), 10)
    worn = np.concatenate(
        [
            np.cumsum(0.6 + 0.15 * np.sqrt(30.0) * g.standard_normal(10))
            for _ in range(10)
        ]
    )
    return surv.WienerProcess.fit(hours, worn, units, threshold=8.0)


def levelled(model, level, interval=60.0, repair_cost=0.0, cost=0.0):
    """One component replaced on condition at ``level``, repaired in no
    time."""
    spec = {
        "reliability": model,
        "repairability": "instant",
        "repair_cost": repair_cost,
        "preventive": {
            "policy": "condition",
            "interval": interval,
            "level": level,
            "cost": cost,
        },
    }
    return RepairableRBD(EDGES, {"a": spec})


def reference(model, level, interval, horizon, n, dt=0.25, seed=0):
    """Each simulation's failures and replacements, from the process's
    definition: its path stepped on a grid of ``dt`` (a Wiener step crossing
    the threshold within it with its Brownian bridge's probability), failed
    at the threshold and repaired as new at once, and inspected at every
    multiple of ``interval``. Exact but for failures falling on the grid."""
    g = np.random.default_rng(seed)
    threshold, new = float(model.threshold), float(model.y0)
    a, b = (float(p) for p in model.params)
    wiener = type(model).__name__.startswith("Wiener")
    y = np.full(n, new)
    failures, replacements = np.zeros(n), np.zeros(n)
    every = round(interval / dt)
    for k in range(1, round(horizon / dt) + 1):
        if wiener:
            step = y + a * dt + b * np.sqrt(dt) * g.standard_normal(n)
            gap, after = threshold - y, np.maximum(threshold - step, 0.0)
            bridge = np.exp(-2.0 * gap * after / (b * b * dt))
            failed = (step >= threshold) | (g.random(n) < bridge)
        else:
            step = y + g.gamma(a * dt, 1.0 / b, n)
            failed = step >= threshold
        failures += failed
        y = np.where(failed, new, step)
        if k % every == 0:
            replaced = y >= level
            replacements += replaced
            y = np.where(replaced, new, y)
    return failures, replacements


# A degradation process as a life ------------------------------------------


@pytest.mark.parametrize("make", [wear, drift])
def test_a_degradation_life_is_its_first_passage(make):
    model = make()
    rbd = NonRepairableRBD(EDGES, {"a": model})
    x = np.array([100.0, 300.0, 600.0])
    np.testing.assert_allclose(rbd.sf(x), np.ravel(model.sf(x)), rtol=1e-12)
    assert rbd.mean() == pytest.approx(float(model.mean()), rel=1e-6)


@pytest.mark.parametrize("make", [wear, drift])
def test_a_repaired_degradation_life_is_exact(make):
    model = make()
    rbd = RepairableRBD(
        EDGES, {"a": {"reliability": model, "repairability": REPAIR}}
    )
    life = float(model.mean())
    assert rbd.mean_availability() == pytest.approx(
        life / (life + 20.0), rel=1e-6
    )
    run = rbd.availability(
        t_simulation=20_000.0, mc_samples=200, seed=2, control_variate=False
    )
    interval = run.mean_availability_interval(0.9999)
    assert interval.lower < rbd.mean_availability() < interval.upper


@pytest.mark.parametrize("make", [wear, drift])
def test_a_degradation_life_is_replaced_at_an_age(make):
    """Age replacement of a degradation life: its mean cycle is the
    integral of its survival to the age, summed here on a fine grid, and
    the best age is the cheapest of a scan of ages."""
    model = make()
    unit = NonRepairable(model)
    unit.set_costs_planned_and_unplanned(cp=1.0, cu=5.0)
    age = float(np.ravel(model.qf(np.array([0.3])))[0])
    ages = np.linspace(0.0, age, 200_001)
    R = np.ravel(model.sf(ages))
    cycle = float(np.sum(0.5 * (R[1:] + R[:-1]) * np.diff(ages)))
    assert unit.avg_replacement_time(age) == pytest.approx(cycle, rel=1e-8)

    best = unit.find_optimal_replacement()
    scan = np.linspace(0.2 * best, 2.0 * best, 1801)
    rates = np.array([unit._cost_rate(a) for a in scan])
    assert best == pytest.approx(scan[int(np.argmin(rates))], rel=2e-3)
    assert unit._cost_rate(best) <= rates.min() * (1 + 1e-9)
    assert unit._cost_rate(best) < unit._run_to_failure_rate()


@pytest.mark.parametrize("make", [wear, drift])
def test_a_repaired_degradation_life_replaced_at_an_age_is_exact(make):
    """In a diagram, replaced in no time at an age T or repaired at a
    failure (mean 20): a cycle is up for the integral of R to T and down
    for 20 with probability F(T), so the long-run availability is
    U / (U + 20 F(T))."""
    model = make()
    age = float(np.ravel(model.qf(np.array([0.3])))[0])
    rbd = RepairableRBD(
        EDGES,
        {
            "a": {
                "reliability": model,
                "repairability": REPAIR,
                "preventive": {"policy": "age", "interval": age},
            }
        },
    )
    ages = np.linspace(0.0, age, 200_001)
    R = np.ravel(model.sf(ages))
    up = float(np.sum(0.5 * (R[1:] + R[:-1]) * np.diff(ages)))
    down = 20.0 * (1.0 - R[-1])  # F(T) = 0.3
    assert rbd.mean_availability() == pytest.approx(up / (up + down), rel=1e-7)
    assert 0.0 < rbd.point_availability(2.5 * age) < 1.0
    run = rbd.availability(
        t_simulation=20_000.0, mc_samples=200, seed=2, control_variate=False
    )
    interval = run.mean_availability_interval(0.9999)
    assert interval.lower < rbd.mean_availability() < interval.upper


def test_an_accelerated_process_is_refused():
    g = np.random.default_rng(1)
    hours = np.tile(np.arange(1.0, 6.0) * 10.0, 6)
    units = np.repeat(np.arange(6), 5)
    stress = np.repeat([1.0, 1.0, 2.0, 2.0, 3.0, 3.0], 5)
    worn = np.concatenate(
        [np.cumsum(g.gamma(s, 0.25, 5)) for s in (1, 1, 2, 2, 3, 3)]
    )
    model = surv.GammaProcess.fit(
        hours, worn, units, threshold=8.0, Z=stress[:, None]
    )
    with pytest.raises(ValueError, match="stress covariates"):
        NonRepairableRBD(EDGES, {"a": model})


# The level after an interval, given no failure ------------------------------


def test_a_gamma_level_is_the_truncated_increment():
    model = wear()
    alpha, beta = (float(p) for p in model.params)
    y, t, threshold = 3.0, 90.0, float(model.threshold)
    g = np.random.default_rng(4)
    drawn = y + g.gamma(alpha * t, 1.0 / beta, 200_000)
    kept = drawn[drawn < threshold]
    ours = [level_after(model, y, t, u) for u in g.random(4000)]
    assert stats.ks_2samp(kept, ours).pvalue > 1e-3
    assert max(ours) < threshold


def test_a_wiener_level_is_the_path_killed_at_the_threshold():
    model = drift()
    mu, sigma = (float(p) for p in model.params)
    y, t, threshold = 5.0, 60.0, float(model.threshold)
    g = np.random.default_rng(5)
    # Paths on a fine grid, killed where a step or its bridge crosses.
    n, steps = 40_000, 600
    dt = t / steps
    path = np.full(n, y)
    alive = np.ones(n, bool)
    for _ in range(steps):
        step = path + mu * dt + sigma * np.sqrt(dt) * g.standard_normal(n)
        after = np.maximum(threshold - step, 0.0)
        bridge = np.exp(-2.0 * (threshold - path) * after / (sigma**2 * dt))
        alive &= (step < threshold) & (g.random(n) >= bridge)
        path = step
    ours = [level_after(model, y, t, u) for u in g.random(4000)]
    assert stats.ks_2samp(path[alive], ours).pvalue > 1e-3
    assert max(ours) < threshold


# Replacement on condition by level -----------------------------------------


@pytest.mark.parametrize("make, level", [(wear, 7.0), (drift, 6.0)])
def test_level_replacement_agrees_with_the_process(make, level):
    model = make()
    horizon, n = 2000.0, 150
    failures, replacements = reference(model, level, 60.0, horizon, 8000)
    for counted, costs in (
        (failures, {"repair_cost": 1.0}),
        (replacements, {"cost": 1.0}),
    ):
        run = levelled(model, level, **costs).cost(
            t_simulation=horizon, mc_samples=n, seed=1
        )
        spread = np.sqrt(counted.var() / counted.size + run.samples.var() / n)
        assert run.samples.mean() == pytest.approx(
            counted.mean(), abs=4.0 * spread
        )


def test_a_lower_level_replaces_more_and_fails_less():
    model = wear()
    counts = [
        levelled(model, level, repair_cost=1.0, cost=1000.0)
        .cost(t_simulation=3000.0, mc_samples=60, seed=3)
        .samples
        for level in (7.5, 5.0)
    ]
    failures = [np.mean(c % 1000.0) for c in counts]
    replacements = [np.mean(c // 1000.0) for c in counts]
    assert failures[1] < failures[0]
    assert replacements[1] > replacements[0]


def test_a_level_is_kept_by_saving():
    rbd = levelled(wear(), 6.5, repair_cost=3.0, cost=1.0)
    again = rbd_from_json(rbd_to_json(rbd))
    first = rbd.cost(t_simulation=1000.0, mc_samples=20, seed=7)
    assert again.cost(
        t_simulation=1000.0, mc_samples=20, seed=7
    ).samples.tolist() == (first.samples.tolist())


def test_level_replacement_from_a_state_needs_the_level():
    # From a state (see test_level_replacement.py): an age alone does not
    # give the level, in the exact values or the simulations; a measured
    # level and the long-run state are taken.
    rbd = levelled(wear(), 6.0, repair_cost=1.0, cost=1.0)
    with pytest.raises(ValueError, match="give the level measured now"):
        rbd.point_availability([100.0], state={"a": NodeState(age=50.0)})
    with pytest.raises(ValueError, match="give the level measured now"):
        rbd.availability(
            t_simulation=100.0,
            mc_samples=2,
            seed=0,
            state={"a": NodeState(age=50.0)},
        )
    for state in (NodeState(level=4.0), NodeState(stationary=True)):
        value = rbd.point_availability([100.0], state={"a": state})
        assert 0.0 < float(value[0]) <= 1.0


@pytest.mark.parametrize(
    "preventive, life, match",
    [
        ({"policy": "age", "level": 6.0}, wear, "only to the 'condition'"),
        ({"policy": "condition", "level": 6.0}, None, "Wiener or gamma"),
        (
            {"policy": "condition", "level": 6.0, "threshold": 0.1},
            wear,
            "not both",
        ),
        ({"policy": "condition", "level": 8.0}, wear, "below the"),
        ({"policy": "condition", "level": "6"}, wear, "below the"),
    ],
)
def test_a_level_is_checked(preventive, life, match):
    model = life() if life is not None else surv.Weibull.from_params([500, 2])
    spec = {
        "reliability": model,
        "repairability": REPAIR,
        "preventive": {"interval": 60.0, **preventive},
    }
    with pytest.raises(ValueError, match=match):
        RepairableRBD(EDGES, {"a": spec})
