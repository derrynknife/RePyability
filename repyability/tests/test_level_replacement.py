"""Replacement on condition by a measured degradation level, worked out
numerically (#271, ``rbd/_level_replacement.py``): the level's law against
the simulation's draws, closed forms and coincidences, the simulation
itself, and the grid's accuracy and units."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import surpyval as surv
from scipy import stats
from scipy.integrate import quad
from surpyval.degradation.process_models import (
    GammaProcessModel,
    WienerProcessModel,
)

import repyability.rbd._level_replacement as level_module
from repyability import NodeState, RepairableRBD
from repyability.rbd._block_replacement import (
    block_availability,
    block_cycle,
)
from repyability.rbd._degradation import level_after, level_below
from repyability.rbd._level_replacement import LevelModel, level_model

E = surv.Exponential.from_params
EDGES = [("s", "a"), ("a", "t")]
REPAIR, DURATION = E([1 / 5.0]), E([1 / 3.0])
#: Mean lives of about 80, inspected every 60: a unit is often replaced at
#: its first or second inspection, and often fails between them.
PROCESSES = {
    "gamma": lambda: GammaProcessModel(alpha=0.5, beta=1.0, threshold=40.0),
    "wiener": lambda: WienerProcessModel(mu=0.5, sigma=2.0, threshold=40.0),
}


def unit(model, level=25.0, interval=60.0, repair=REPAIR, duration=DURATION):
    """One component replaced on condition at ``level``, priced."""
    spec = {
        "reliability": model,
        "repairability": repair,
        "repair_cost": 10.0,
        "preventive": {
            "policy": "condition",
            "interval": interval,
            "level": level,
            "duration": duration,
            "cost": 2.0,
            "inspection_cost": 1.0,
        },
    }
    return RepairableRBD(EDGES, {"a": spec})


def plain_unit(model, repair=REPAIR):
    return RepairableRBD(
        EDGES, {"a": {"reliability": model, "repairability": repair}}
    )


# The level's law: one statement for the draws and the grid -----------------


@pytest.mark.parametrize("name", sorted(PROCESSES))
def test_the_draws_invert_the_law_the_grid_takes(name):
    model = PROCESSES[name]()
    y, t = 12.0, 45.0
    survive = float(level_below(model, y, t, model.threshold))
    # At the threshold, the law is the survival surpyval gives from y.
    assert survive == pytest.approx(
        float(np.ravel(model.sf(t, y0=y))[0]), rel=1e-9
    )
    for u in (1e-6, 0.1, 0.5, 0.9, 1 - 1e-6):
        z = level_after(model, y, t, u)
        assert float(level_below(model, y, t, z)) / survive == pytest.approx(
            u, abs=1e-9
        )


@pytest.mark.parametrize("name", sorted(PROCESSES))
def test_the_kernel_is_the_draws_distribution_from_a_cell(name):
    # A cell's units, spread evenly over it, moved by the simulation's own
    # draws (given that they have not failed): they land in the cells as
    # the kernel's row says, once divided by the row's survival.
    model = PROCESSES[name]()
    built = LevelModel(model, REPAIR, None, 60.0, 25.0)
    edges, kernel = built.levels.edges, built.levels.kernel
    assert np.allclose(
        kernel.sum(axis=1), built.levels.survival[:, -1], rtol=1e-9
    )
    g = np.random.default_rng(11)
    cell = int(np.searchsorted(edges, 5.0)) - 1
    n = 20_000
    start = g.uniform(edges[cell], edges[cell + 1], n)
    landed = np.array(
        [level_after(model, y, 60.0, u) for y, u in zip(start, g.random(n))]
    )
    counts = np.bincount(
        np.clip(np.searchsorted(edges, landed, side="right") - 1, 0, None),
        minlength=len(edges),
    )
    expected = kernel[cell] / kernel[cell].sum() * n
    # Pooled into a few bins of enough expected units each.
    order = np.argsort(-expected)
    bins = np.array_split(order[expected[order] > 0], 8)
    observed = np.array([counts[b].sum() for b in bins])
    wanted = np.array([expected[b].sum() for b in bins])
    wanted *= observed.sum() / wanted.sum()
    assert stats.chisquare(observed, wanted).pvalue > 1e-3


# Closed forms and coincidences ---------------------------------------------


@pytest.mark.parametrize("name", sorted(PROCESSES))
def test_a_level_at_the_threshold_is_the_plain_unit(name):
    # A level just short of the threshold all but never replaces: a unit
    # unfailed at an inspection is almost never that close to failing.
    model = PROCESSES[name]()
    rbd = unit(model, level=40.0 * (1 - 1e-9), duration="instant")
    plain = plain_unit(model)
    # To the accuracy of the grid over an interval, which falls with its
    # step (as for replacement on condition by age, whose grid it is):
    # about 1e-6 in the availability and 2e-5 in the failure rate here.
    # Just after an inspection, the units kept just short of the
    # threshold fail at once, which the level grid follows only to the
    # square of its cells' width: about 1e-4 here, the worst case.
    assert rbd.mean_availability() == pytest.approx(
        plain.mean_availability(), abs=3e-6
    )
    t = np.array([30.0, 59.0, 61.0, 150.0, 400.0])
    np.testing.assert_allclose(
        rbd.point_availability(t), plain.point_availability(t), atol=2e-4
    )
    np.testing.assert_allclose(
        rbd.expected_failures(t),
        plain.expected_failures(t),
        rtol=5e-5,
        atol=2e-4,
    )


class _FirstPassage:
    """A degradation process's first passage as a plain life, for the
    block-replacement code (which takes a surpyval distribution)."""

    dist = SimpleNamespace(name="FirstPassage")

    def __init__(self, model):
        self.model = model

    def sf(self, x):
        return self.model.sf(x)

    def ff(self, x):
        return self.model.ff(x)

    def qf(self, p):
        return self.model.qf(p)

    def mean(self):
        return self.model.mean()


@pytest.mark.parametrize("duration", [None, DURATION])
def test_a_gamma_level_at_its_start_is_block_replacement(duration):
    # A gamma level only rises, so one at or below y0 is past the level at
    # every inspection: each inspection of a working unit replaces it,
    # and the unit is under block replacement at the interval.
    model = PROCESSES["gamma"]()
    built = LevelModel(model, REPAIR, duration, 60.0, float(model.y0))
    assert len(built.levels.edges) == 1  # no level is kept
    ours = built.availability(1000.0)
    block = block_availability(
        _FirstPassage(model), REPAIR, duration, 60.0, 1000.0
    )
    np.testing.assert_allclose(ours.smooth, block.smooth, atol=1e-12)
    np.testing.assert_allclose(ours.replaced, block.replaced, atol=1e-12)
    np.testing.assert_allclose(ours.failures, block.failures, atol=1e-12)
    cycle = built.cycle()
    other = block_cycle(_FirstPassage(model), REPAIR, duration, 60.0)
    # The up time here by the trapezoid rule on the grid, there by its
    # exact integral: to the grid's accuracy.
    assert cycle.up / cycle.length == pytest.approx(
        other.up / other.length, rel=1e-7
    )
    assert cycle.failures / cycle.length == pytest.approx(
        other.failures / other.length, rel=1e-9
    )
    assert cycle.replaced / cycle.length == pytest.approx(
        1.0 / other.length, rel=1e-9
    )
    assert (cycle.before, cycle.after) == pytest.approx(
        (other.before, other.after), abs=1e-12
    )


def test_replaced_at_every_inspection_in_no_time_by_hand():
    # Repairs and replacements in no time, every inspection replacing: an
    # interval is a renewal process of first passages from y0 = 0 over
    # 40, F(t) = P(Gamma(alpha t, beta) >= 40) = 1 - P(alpha t, 40 beta),
    # renewed at its start. Its failures in an interval are the renewal
    # function M(T) = F(T) + int_0^T M(T - s) dF(s); with F(60) about
    # 0.0024 here, two terms leave out under 1e-8.
    model = GammaProcessModel(alpha=0.5, beta=1.0, threshold=40.0)

    def F(t):
        return float(stats.gamma.sf(40.0, 0.5 * t)) if t > 0 else 0.0

    def f(t):
        h = 1e-4
        return (F(t + h) - F(t - h)) / (2 * h)

    def M(t):
        return F(t) + quad(lambda s: F(t - s) * f(s), 0.0, t, limit=200)[0]

    T = 60.0
    rbd = unit(
        model, level=0.0, interval=T, repair="instant", duration="instant"
    )
    # Ten inspections, each renewing, and half an interval after the last.
    events = rbd.expected_events(10.5 * T)
    by_hand = 10 * M(T) + M(T / 2)
    assert float(events.node_failures["a"]) == pytest.approx(by_hand, rel=1e-5)
    assert float(events.node_preventive["a"]) == pytest.approx(10.0, rel=1e-9)
    assert float(events.node_inspections["a"]) == pytest.approx(10, rel=1e-9)
    assert rbd.mean_availability() == pytest.approx(1.0, abs=1e-12)


# The grid's accuracy, and units ---------------------------------------------


@pytest.mark.parametrize("name", sorted(PROCESSES))
def test_halving_the_cells_changes_little(name, monkeypatch):
    # The level grid's error falls as its cells' width squared: halving
    # them moves the values by about three quarters of their error.
    model = PROCESSES[name]()
    t = np.array([60.5, 90.0, 120.5, 300.5, 600.0])

    def values():
        rbd = unit(model)
        return (
            rbd.point_availability(t),
            rbd.mean_availability(),
            rbd.expected_cost_rate(),
        )

    coarse = values()
    monkeypatch.setattr(level_module, "_PER_SPREAD", 64)
    fine = values()
    np.testing.assert_allclose(coarse[0], fine[0], atol=1e-5)
    assert coarse[1] == pytest.approx(fine[1], abs=2e-6)
    assert coarse[2] == pytest.approx(fine[2], rel=3e-5)


def scaled(name, c=1.0, k=1.0, level=25.0):
    """The diagram with time multiplied by ``c`` and levels by ``k``."""
    if name == "gamma":
        model = GammaProcessModel(
            alpha=0.5 / c, beta=1.0 / k, threshold=40.0 * k, y0=2.0 * k
        )
    else:
        model = WienerProcessModel(
            mu=0.5 * k / c,
            sigma=2.0 * k / math.sqrt(c),
            threshold=40.0 * k,
            y0=2.0 * k,
        )
    return unit(
        model,
        level=level * k,
        interval=60.0 * c,
        repair=E([1 / (5.0 * c)]),
        duration=E([1 / (3.0 * c)]),
    )


@pytest.mark.parametrize("name", sorted(PROCESSES))
@pytest.mark.parametrize(
    "c, k", [(1e-6, 1.0), (1e6, 1.0), (1.0, 1e-6), (1.0, 1e6)]
)
def test_results_do_not_depend_on_the_units(name, c, k):
    # Neither the grid over an interval nor the grid of levels is sized in
    # absolute units, so a unit of time or of level changes nothing.
    base, other = scaled(name), scaled(name, c, k)
    t = np.array([30.0, 60.5, 200.0, 500.0])
    np.testing.assert_allclose(
        other.point_availability(t * c),
        base.point_availability(t),
        rtol=1e-9,
    )
    assert other.mean_availability() == pytest.approx(
        base.mean_availability(), rel=1e-9
    )
    assert other.expected_cost_rate() * c == pytest.approx(
        base.expected_cost_rate(), rel=1e-7
    )


# What is still refused ----------------------------------------------------


def test_a_wiener_process_with_no_drift_is_refused():
    model = WienerProcessModel(mu=-0.01, sigma=2.0, threshold=40.0)
    rbd = unit(model)
    assert rbd.analysis_routes()["mean_availability"].route == "refused"
    with pytest.raises(NotImplementedError, match="does not drift"):
        rbd.mean_availability()
    with pytest.raises(NotImplementedError, match="does not drift"):
        rbd.point_availability([10.0])


def test_a_level_too_fine_for_the_grid_is_refused():
    # Worn 40 an hour, give or take about 3 over an interval of 60 hours,
    # replaced at 2000: the grid of levels would need some 20,000 cells.
    model = GammaProcessModel(alpha=1e4, beta=250.0, threshold=4000.0)
    rbd = unit(model, level=2000.0, interval=60.0)
    reason = rbd.analysis_routes()["mean_availability"].reason
    assert "cells" in reason
    with pytest.raises(NotImplementedError, match="cells"):
        rbd.mean_availability()


def test_the_model_is_built_once_and_kept():
    rbd = unit(PROCESSES["gamma"]())
    assert level_model(rbd, "a") is level_model(rbd, "a")
    rbd.mean_availability()
    rbd.point_availability([100.0])
    assert level_model(rbd, "a").cycle() is level_model(rbd, "a").cycle()


def test_a_state_is_refused_where_a_unit_is_not_new():
    rbd = unit(PROCESSES["gamma"]())
    with pytest.raises(NotImplementedError, match="leave it out"):
        rbd.expected_cost(100.0, state={"a": NodeState(age=10.0)})


# Against the simulation, run plainly ----------------------------------------

#: Plain runs: by default a run's means would be these exact values.
PLAIN = dict(control_variate=False, conditional=False)


@pytest.mark.parametrize("name", sorted(PROCESSES))
def test_the_values_over_time_agree_with_the_simulation(name):
    # The availability just before and after the inspections (replacements
    # take the unit down for a while), between them and later on, within
    # the simulation's 99.9% band at each time: with 2,000 simulations, a
    # band some 0.02 wide about 0.5 and 0.01 about 0.95.
    rbd = unit(PROCESSES[name]())
    run = rbd.availability(
        600.0, mc_samples=2000, seed=3, curve_points=1200, **PLAIN
    )
    at = np.array([30.0, 59.5, 60.5, 65.0, 119.5, 120.5, 300.5, 599.5])
    index = np.rint(at * 2).astype(int)
    assert np.allclose(run.timeline[index], at)
    lower, upper = run.availability_interval(0.999)
    exact = rbd.point_availability(at)
    assert np.all(lower[index] <= exact), (exact, lower[index])
    assert np.all(exact <= upper[index]), (exact, upper[index])
    # A replacement that takes time shows: down just after an inspection.
    assert exact[2] < exact[1] - 0.2


@pytest.mark.parametrize("name", sorted(PROCESSES))
def test_the_events_and_costs_agree_with_the_simulation(name):
    # Failures, replacements and inspections over a window, counted in
    # the simulation by pricing each at a different power of 1000, and
    # the cost and availability over it.
    model = PROCESSES[name]()
    counted = RepairableRBD(
        EDGES,
        {
            "a": {
                "reliability": model,
                "repairability": REPAIR,
                "repair_cost": 1.0,
                "preventive": {
                    "policy": "condition",
                    "interval": 60.0,
                    "level": 25.0,
                    "duration": DURATION,
                    "cost": 1e3,
                    "inspection_cost": 1e6,
                },
            }
        },
    )
    window, n = 2000.0, 600
    events = counted.expected_events(window)
    samples = counted.cost(window, mc_samples=n, seed=21, **PLAIN).samples
    inspections = np.floor(samples / 1e6)
    replacements = np.floor((samples - 1e6 * inspections) / 1e3)
    failures = samples - 1e6 * inspections - 1e3 * replacements
    for exact, drawn in (
        (events.node_failures["a"], failures),
        (events.node_preventive["a"], replacements),
        (events.node_inspections["a"], inspections),
    ):
        error = drawn.std() / math.sqrt(n)
        assert abs(drawn.mean() - float(exact)) <= 4.0 * error + 1e-9


@pytest.mark.parametrize("name", sorted(PROCESSES))
def test_the_long_run_agrees_with_the_simulation(name):
    # Over a long window from new the simulation's mean availability and
    # cost rate are the long run's, but for the start, which the exact
    # values over the window take in: they lie between, and both within
    # the simulation's interval.
    rbd = unit(PROCESSES[name]())
    window, n = 6000.0, 60
    run = rbd.availability(window, mc_samples=n, seed=4, **PLAIN)
    interval = run.mean_availability_interval(0.999)
    assert interval.lower <= rbd.mean_availability() <= interval.upper
    assert (
        interval.lower <= rbd.mission_availability(window) <= (interval.upper)
    )
    costs = rbd.cost(window, mc_samples=n, seed=5, **PLAIN).mean_interval(
        0.999
    )
    rate = rbd.expected_cost_rate()
    assert costs.lower <= rate * window <= costs.upper
    assert costs.lower <= rbd.expected_cost(window).mean <= costs.upper


def test_grids_too_large_or_unsettled_are_refused(monkeypatch):
    model = PROCESSES["gamma"]()
    monkeypatch.setattr(level_module, "_MAX_SURVIVAL", 1000)
    with pytest.raises(NotImplementedError, match="too large to follow"):
        unit(model).mean_availability()
    monkeypatch.undo()
    # Followed for at most three intervals: no long run yet.
    monkeypatch.setattr(level_module, "_MAX_VALUES", 1)
    with pytest.raises(NotImplementedError, match="has not settled"):
        LevelModel(model, REPAIR, DURATION, 60.0, 25.0).cycle()
