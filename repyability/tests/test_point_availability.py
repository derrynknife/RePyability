"""Exact point and mission availability from new (issue #117).

``RepairableRBD.point_availability`` gives the probability that the system
is up at each time, every component new at 0, and ``mission_availability``
its mean over ``[0, t]``. The references are closed forms (exponential up
and down times; a component always up but for its fixed-time replacements;
sums of gamma maintenance times; periodic tests), the long-run values the
curves settle at (``mean_availability``) with the start-up term renewal
theory predicts, grids four times finer, and the simulation
(``availability``) on the benchmark diagrams.
"""

import math

import numpy as np
import pytest
import scipy.stats as st
import surpyval as surv
from scipy.integrate import quad
from scipy.special import gamma as gamma_function

import repyability.rbd._point_availability as point_availability
import repyability.rbd.repairable_rbd as repairable_rbd
from repyability import NodeState, RepairableRBD
from repyability.rbd._model_utils import lfp_extras
from repyability.tests.test_performance_equivalence import (
    instrument_air,
    repairable_rbds,
)

E = surv.Exponential.from_params
W = surv.Weibull.from_params
LN = surv.LogNormal.from_params
GAMMA = surv.Gamma.from_params
FIXED = surv.ExactEventTime.from_params


def unit(life, repair, **extra):
    return {"reliability": life, "repairability": repair, **extra}


def alone(spec):
    return RepairableRBD([("s", "c"), ("c", "t")], {"c": spec})


def pair(spec_a, spec_b, parallel=True):
    edges = (
        [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")]
        if parallel
        else [("s", "a"), ("a", "b"), ("b", "t")]
    )
    return RepairableRBD(edges, {"a": spec_a, "b": spec_b})


def age(interval, duration=None):
    preventive = {"interval": interval, "policy": "age"}
    if duration is not None:
        preventive["duration"] = duration
    return preventive


def block(interval, duration=None):
    preventive = {"interval": interval, "policy": "block"}
    if duration is not None:
        preventive["duration"] = duration
    return preventive


def exponential(lam, mu, t):
    """Up at ``t``: a unit with failure rate ``lam`` and repair rate
    ``mu``, up at 0."""
    t = np.asarray(t, dtype=float)
    return mu / (lam + mu) + lam / (lam + mu) * np.exp(-(lam + mu) * t)


def mean_of(curve, t):
    """The mean of ``curve`` over ``[0, t]``, by adaptive quadrature."""
    return quad(curve, 0.0, t, limit=500, epsabs=1e-13, epsrel=1e-13)[0] / t


def exponential_sum(lam, mu):
    """A unit's point availability as ``{decay rate: coefficient}``."""
    return {0.0: mu / (lam + mu), lam + mu: lam / (lam + mu)}


def times(p, q):
    out: dict = {}
    for c1, a1 in p.items():
        for c2, a2 in q.items():
            out[c1 + c2] = out.get(c1 + c2, 0.0) + a1 * a2
    return out


def plus(p, q, scale=1.0):
    out = dict(p)
    for c, a in q.items():
        out[c] = out.get(c, 0.0) + scale * a
    return out


def value(p, t):
    return sum(a * np.exp(-c * t) for c, a in p.items())


def mean(p, T):
    """The mean over ``[0, T]`` of a sum of exponentials, exactly."""
    total = sum(
        a * (T if c == 0.0 else -np.expm1(-c * T) / c) for c, a in p.items()
    )
    return total / T


# -- Closed forms -------------------------------------------------------------


def test_one_component_with_exponential_times():
    lam, mu = 0.1, 1.0
    rbd = alone(unit(E([lam]), E([mu])))
    t = np.array([0.0, 0.3, 1.0, 2.5, 10.0, 100.0])
    assert np.abs(
        rbd.point_availability(t) - exponential(lam, mu, t)
    ).max() < (8e-7)
    windows = np.array([0.5, 1.0, 10.0, 1000.0])
    exact = (
        mu / (lam + mu)
        + lam
        / (lam + mu) ** 2
        * (1.0 - np.exp(-(lam + mu) * windows))
        / windows
    )
    assert np.abs(rbd.mission_availability(windows) - exact).max() < 4e-7


@pytest.mark.parametrize("parallel", [True, False], ids=["parallel", "series"])
def test_time_scales_far_apart(parallel):
    # Failure rates from 1e-2 to 1e-5 and repairs of an hour to twenty:
    # each unit on its own grid, the system exact at every time.
    rates = [(0.01, 0.5), (0.001, 0.05), (1e-5, 1.0)]
    specs = [unit(E([lam]), E([mu])) for lam, mu in rates]
    names = ["a", "b", "c"]
    if parallel:
        edges = [("s", n) for n in names] + [(n, "t") for n in names]
    else:
        edges = [("s", "a"), ("a", "b"), ("b", "c"), ("c", "t")]
    rbd = RepairableRBD(edges, dict(zip(names, specs)))
    system = {0.0: 1.0}
    for lam, mu in rates:
        up = exponential_sum(lam, mu)
        system = times(system, plus({0.0: 1.0}, up, -1.0) if parallel else up)
    if parallel:
        system = plus({0.0: 1.0}, system, -1.0)
    # From 200 h: the slowest unit's grid step is 20 h and more, far longer
    # than its 1 h repairs, which are smoothed over its first step (see the
    # docstring).
    t = np.array([200.0, 1000.0, 5000.0, 60000.0, 250000.0])
    assert np.abs(rbd.point_availability(t) - value(system, t)).max() < 3e-7
    for window in [1000.0, 20000.0, 300000.0]:
        assert rbd.mission_availability(window) == pytest.approx(
            mean(system, window), abs=1.2e-7
        )


def test_the_start_up_term():
    # From new, a unit that wears out fails less early on: its expected
    # uptime over [0, T] is A T + b + o(1), with
    # b = A (E[C^2] / 2 E[C] - E[U^2] / 2 E[U]) for up times U and cycles C.
    shape, scale, mu, sigma = 2.5, 100.0, 1.0, 0.6
    rbd = alone(unit(W([scale, shape]), LN([mu, sigma])))
    A = rbd.mean_availability()
    up = scale * gamma_function(1 + 1 / shape)
    up2 = scale**2 * gamma_function(1 + 2 / shape)
    down = math.exp(mu + sigma**2 / 2)
    down2 = math.exp(2 * mu + 2 * sigma**2)
    cycle, cycle2 = up + down, up2 + 2 * up * down + down2
    b = A * (cycle2 / (2 * cycle) - up2 / (2 * up))
    for T in [1e4, 1e5, 1e6]:
        start_up = T * (rbd.mission_availability(T) - A)
        assert start_up == pytest.approx(b, rel=1e-6)
    # And the curve itself settles at the long-run value.
    assert rbd.point_availability(20 * cycle) == pytest.approx(A, abs=1e-10)


def test_the_first_order_system_start_up_term():
    # The check from issue #117: a system's mission average approaches
    # A + sum_i I_i b_i / T, I_i each unit's Birnbaum importance at the
    # long-run availabilities (to first order: the rest are products of
    # the units' small start-up terms). Reliafy's simulation of this
    # sample gave 0.9996857014 +- 4.3e-7 over 600,000 h.
    rbd = instrument_air()
    T = 600_000.0
    A = rbd.mean_availability()
    predicted = 0.0
    for node, component in rbd.components.items():
        life, repair = component.reliability, component.time_to_replace
        if life.dist.name == "Exponential":
            (rate,) = np.ravel(life.params)
            up, up2 = 1 / rate, 2 / rate**2
        else:
            scale, shape = np.ravel(life.params)
            up = scale * gamma_function(1 + 1 / shape)
            up2 = scale**2 * gamma_function(1 + 2 / shape)
        mu, sigma = np.ravel(repair.params)
        down = math.exp(mu + sigma**2 / 2)
        down2 = math.exp(2 * mu + 2 * sigma**2)
        cycle, cycle2 = up + down, up2 + 2 * up * down + down2
        b = up / cycle * (cycle2 / (2 * cycle) - up2 / (2 * up))
        importance = rbd.mean_availability(
            working_nodes=[node]
        ) - rbd.mean_availability(broken_nodes=[node])
        predicted += importance * b / T
    exact = rbd.mission_availability(T) - A
    assert predicted == pytest.approx(3.81e-6, abs=0.01e-6)
    assert exact == pytest.approx(predicted, rel=0.03)
    assert abs(A + exact - 0.9996857014) < 4.3e-7


# -- Age replacement ----------------------------------------------------------


def test_age_replacement_up_to_and_at_the_age():
    life, repair = W([60.0, 2.5]), LN([0.5, 0.5])
    plain = alone(unit(life, repair))
    maintained = alone(
        unit(life, repair, preventive=age(30.0, LN([0.0, 0.3])))
    )
    # Before the age nothing is due: the same as without the schedule.
    t = np.array([1.0, 10.0, 20.0, 29.9])
    assert (
        np.abs(
            maintained.point_availability(t) - plain.point_availability(t)
        ).max()
        < 1e-7
    )
    # At it, the unit that has not failed goes down for its maintenance.
    survive = float(np.ravel(life.sf(np.array([30.0])))[0])
    assert maintained.point_availability(30.0) == pytest.approx(
        plain.point_availability(30.0) - survive, abs=1e-7
    )
    assert maintained.point_availability(5000.0) == pytest.approx(
        maintained.mean_availability(), abs=1e-10
    )


def test_later_age_replacements_are_exact():
    # A unit replaced at age 10 before it can fail is maintained from
    # 10 (n + 1) + T_n for one more maintenance, T_n the sum of n of them:
    # with gamma maintenance times, T_n is gamma too.
    shape, rate, interval = 3.0, 2.0, 10.0
    rbd = alone(
        unit(
            FIXED([2 * interval]),
            E([1.0]),
            preventive=age(interval, GAMMA([shape, rate])),
        )
    )

    def exact(t):
        down = 0.0
        for n in range(12):
            s = t - (n + 1) * interval
            if s < 0:
                break
            done = st.gamma.cdf(s, n * shape, scale=1 / rate) if n else 1.0
            down += done - st.gamma.cdf(s, (n + 1) * shape, scale=1 / rate)
        return 1.0 - down

    t = np.concatenate(
        [
            interval * (n + 1) + n * 1.5 + np.linspace(-2.0, 6.0, 41)
            for n in range(6)
        ]
    )
    exact_values = np.array([exact(x) for x in t])
    assert np.abs(rbd.point_availability(t) - exact_values).max() < 1e-6
    assert rbd.mission_availability(55.0) == pytest.approx(
        mean_of(np.vectorize(exact), 55.0), abs=1e-8
    )


def test_units_replaced_together_go_down_together(monkeypatch):
    # Two units in parallel, both new at 0 and replaced at the same age:
    # their later maintenances nearly coincide, which is when the pair is
    # down. The mission average agrees with a grid four times finer.
    spec = unit(
        W([8000.0, 3.0]),
        LN([math.log(24.0), 0.5]),
        preventive=age(4000.0, LN([0.0, 0.3])),
    )
    rbd = pair(spec, dict(spec))
    windows = [10_000.0, 40_000.0]
    default = rbd.mission_availability(windows)
    monkeypatch.setattr(repairable_rbd, "_POINT_STEPS", 4000)
    fine = rbd.mission_availability(windows)
    assert np.abs(default - fine).max() < 2e-8
    # Without the maintenance of units that each reach their age kept off
    # the grid, the pair's downtime would be some 40% short.
    assert 1.0 - default[1] == pytest.approx(4.723e-5, rel=0.001)


def test_age_replacement_matches_the_simulation():
    spec = unit(
        W([40.0, 3.0]),
        LN([0.0, 0.5]),
        preventive=age(20.0, LN([math.log(0.5), 0.3])),
    )
    rbd = pair(spec, dict(spec))
    T = 100.0
    result = rbd.availability(
        T, mc_samples=20_000, seed=5, control_variate=False
    )
    window = result.mean_availability_interval(confidence=0.999)
    assert window.lower <= rbd.mission_availability(T) <= window.upper
    lower, upper = result.availability_interval(confidence=0.999)
    t = np.array([10.0, 20.05, 20.3, 40.4, 41.0, 60.9, 75.0])
    index = np.searchsorted(result.timeline, t, side="right") - 1
    exact = rbd.point_availability(t)
    assert np.all(
        (lower[index] <= exact + 1e-9) & (exact <= upper[index] + 1e-9)
    )


# -- Block replacement --------------------------------------------------------


def test_block_replacement_of_a_unit_that_is_never_down_otherwise():
    # Instant repair: the unit is up but for its replacements, from each
    # block time k T for d, whatever its life.
    T, d = 100.0, 4.0
    rbd = alone(
        unit(W([150.0, 2.0]), "instant", preventive=block(T, FIXED([d])))
    )
    t = np.array([50.0, 99.9, 100.0, 103.9, 104.0, 250.0, 302.0, 304.5])
    assert rbd.point_availability(t).tolist() == pytest.approx(
        [1, 1, 0, 0, 1, 1, 0, 1], abs=1e-12
    )
    windows = np.array([100.0, 102.0, 450.0, 1000.0])
    down = np.array([0.0, 2.0, 4 * d, 9 * d + 0.0])
    assert rbd.mission_availability(windows) == pytest.approx(
        1.0 - down / windows, abs=1e-12
    )


def test_block_replacement_from_new():
    life, repair = W([100.0, 2.5]), LN([1.5, 0.5])
    plain = alone(unit(life, repair))
    rbd = alone(unit(life, repair, preventive=block(60.0, LN([0.5, 0.3]))))
    # The first interval is the unit's own: nothing is due before 60.
    t = np.array([5.0, 30.0, 59.0])
    assert (
        np.abs(rbd.point_availability(t) - plain.point_availability(t)).max()
        < 1e-7
    )
    # At a block time, the unit that is up goes down to be replaced.
    assert rbd.point_availability(60.0) == pytest.approx(0.0, abs=1e-12)
    # Later it repeats from one interval to the next, about its long-run
    # profile, whose mean is the long-run availability.
    late = 60.0 * 40 + np.linspace(0.0, 60.0, 60_001)[:-1] + 0.0005
    values = rbd.point_availability(late)
    assert np.abs(rbd.point_availability(late + 60.0) - values).max() < 1e-12
    assert values.mean() == pytest.approx(rbd.mean_availability(), abs=1e-7)
    A = rbd.mean_availability()
    start_up = [T * (rbd.mission_availability(T) - A) for T in (1e4, 1e6)]
    assert start_up[0] == pytest.approx(start_up[1], abs=1e-4)


def test_block_replaced_units_in_step():
    # Units replaced at the same block times are replaced together: a
    # parallel pair of units always up but for their replacements is down
    # whenever they are.
    T, d = 100.0, 4.0
    spec = unit(E([0.01]), "instant", preventive=block(T, FIXED([d])))
    rbd = pair(spec, dict(spec))
    assert rbd.mission_availability(1000.0) == pytest.approx(
        1.0 - 9 * d / 1000.0, abs=1e-12
    )


# -- Hidden failures ----------------------------------------------------------


def test_hidden_failures_found_by_periodic_tests():
    lam, tau = 0.01, 10.0
    rbd = alone(unit(E([lam]), "instant", inspection={"interval": tau}))
    t = np.array([0.0, 5.0, 9.99, 10.0, 17.5, 1234.5])
    expected = np.exp(-lam * np.mod(t, tau))
    assert np.abs(rbd.point_availability(t) - expected).max() < 1e-12
    k, r = 123, 4.0
    exact = (k * (1 - math.exp(-lam * tau)) + (1 - math.exp(-lam * r))) / (
        lam * (k * tau + r)
    )
    assert rbd.mission_availability(k * tau + r) == pytest.approx(
        exact, abs=1e-12
    )


def test_mission_repeats_its_calendar(monkeypatch):
    # Past the time its units settle, the system's availability repeats
    # with the calendar (a common period of 50 here): integrating one
    # period and repeating it gives what integrating them all does.
    rbd = RepairableRBD(
        [("s", "a"), ("a", "b"), ("b", "t")],
        {
            "a": unit(
                W([100.0, 2.5]),
                LN([1.5, 0.5]),
                preventive=block(25.0, LN([0.5, 0.3])),
            ),
            "b": unit(E([0.002]), "instant", inspection={"interval": 10.0}),
        },
    )
    windows = np.array([333.0, 2000.0, 4321.5])
    repeated = rbd.mission_availability(windows)
    monkeypatch.setattr(
        repairable_rbd, "_settling", lambda curves: (np.inf, None)
    )
    assert rbd.mission_availability(windows) == pytest.approx(
        repeated, abs=1e-12
    )


# -- Structure ----------------------------------------------------------------


def test_forced_nodes():
    lam, mu = 0.1, 1.0
    spec = unit(E([lam]), E([mu]))
    rbd = pair(spec, unit(E([0.05]), E([0.5])))
    t = np.array([0.5, 2.0, 30.0])
    assert rbd.point_availability(t, broken_nodes=["b"]) == pytest.approx(
        exponential(lam, mu, t), abs=2e-7
    )
    assert rbd.point_availability(t, working_nodes=["b"]).tolist() == [1, 1, 1]
    assert rbd.mission_availability(
        30.0, broken_nodes=["a", "b"]
    ) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        rbd.point_availability(t, working_nodes=["x"])
    with pytest.raises(ValueError):
        rbd.mission_availability(30.0, working_nodes=["a"], broken_nodes=["a"])


def test_a_nested_rbd_is_its_own_structure_function():
    a = unit(W([100.0, 2.0]), LN([0.5, 0.5]))
    b = unit(W([80.0, 1.5]), E([0.2]))
    c = unit(E([0.02]), E([0.5]), preventive=block(40.0, FIXED([1.0])))
    inner = RepairableRBD(
        [("s", "a"), ("a", "b"), ("b", "t")], {"a": a, "b": b}
    )
    nested = RepairableRBD(
        [("s", "ab"), ("s", "c"), ("ab", "t"), ("c", "t")],
        {"ab": inner, "c": c},
    )
    flat = RepairableRBD(
        [("s", "a"), ("a", "b"), ("b", "t"), ("s", "c"), ("c", "t")],
        {"a": a, "b": b, "c": c},
    )
    t = np.array([0.0, 3.0, 40.0, 41.0, 150.0, 2000.0])
    assert (
        np.abs(nested.point_availability(t) - flat.point_availability(t)).max()
        < 1e-12
    )
    for method in ["p", "c"]:
        assert nested.mission_availability(
            [100.0, 3000.0], method=method
        ) == pytest.approx(
            flat.mission_availability([100.0, 3000.0]), abs=1e-12
        )


@pytest.mark.parametrize(
    "name",
    sorted(set(repairable_rbds()) - {"inspected"}),
)
def test_matches_the_simulation(name):
    rbd = repairable_rbds()[name]
    T = 200.0
    result = rbd.availability(
        T, mc_samples=4000, seed=11, control_variate=False
    )
    window = result.mean_availability_interval(confidence=0.999)
    assert window.lower <= rbd.mission_availability(T) <= window.upper
    lower, upper = result.availability_interval(confidence=0.999)
    t = np.linspace(0.0, T, 9)
    index = np.searchsorted(result.timeline, t, side="right") - 1
    exact = rbd.point_availability(t)
    assert np.all(
        (lower[index] <= exact + 1e-9) & (exact <= upper[index] + 1e-9)
    )


# -- Models -------------------------------------------------------------------


def test_instant_repair_is_always_up():
    rbd = alone(unit(W([100.0, 2.0]), "instant"))
    assert rbd.point_availability([0.0, 10.0, 1e4]).tolist() == [1, 1, 1]
    assert rbd.mission_availability(100.0) == 1.0


def test_units_dead_on_arrival_and_units_that_never_fail():
    dead = alone(unit(W([100.0, 2.0], f0=0.1), E([0.5])))
    assert dead.point_availability(0.0) == pytest.approx(0.9)
    assert dead.point_availability(1e4) == pytest.approx(
        dead.mean_availability(), abs=1e-10
    )
    # With p < 1 some unit eventually never fails: the unit ends up for good.
    lasting = alone(unit(W([100.0, 2.0], **lfp_extras(0.8)), E([0.5])))
    values = lasting.point_availability([0.0, 50.0, 500.0, 5000.0])
    assert values[0] == 1.0 and values[1] < values[2] < values[3]
    assert values[3] == pytest.approx(1.0, abs=1e-6)


def test_a_lifetime_known_exactly():
    # Failing at exactly 10 and repaired at rate 1: up to 10, then down
    # until the first repair ends (the next unit fails after 20).
    rbd = alone(unit(FIXED([10.0]), E([1.0])))
    t = np.array([2.0, 9.9, 10.5, 12.0, 15.0, 19.9])
    expected = np.where(t < 10.0, 1.0, 1.0 - np.exp(-(t - 10.0)))
    assert np.abs(rbd.point_availability(t) - expected).max() < 1e-7


# -- The grid -----------------------------------------------------------------


def test_the_error_falls_as_the_square_of_the_step(monkeypatch):
    rbd = alone(unit(W([100.0, 2.5]), LN([1.0, 0.6])))
    t = np.array([3.0, 40.0, 95.5, 250.0, 700.0])
    windows = np.array([50.0, 300.0, 5000.0])
    points, missions = [], []
    for steps in [500, 2000, 8000]:
        monkeypatch.setattr(repairable_rbd, "_POINT_STEPS", steps)
        points.append(rbd.point_availability(t))
        missions.append(rbd.mission_availability(windows))
    # Against the finest: the default grid is some 16 times closer than
    # one of a quarter of its steps.
    for values in [points, missions]:
        coarse = np.abs(values[0] - values[2]).max()
        default = np.abs(values[1] - values[2]).max()
        assert coarse / default > 8.0
    assert np.abs(points[1] - points[2]).max() < 5e-8
    assert np.abs(missions[1] - missions[2]).max() < 5e-9


def test_later_maintenance_on_a_grid_of_its_own(monkeypatch):
    # The fine grid of ``ChainDips`` at half and double its resolution:
    # the error falls as the fourth power of its step.
    rbd = alone(
        unit(
            FIXED([20.0]),
            E([1.0]),
            preventive=age(10.0, GAMMA([3.0, 2.0])),
        )
    )
    t = 10.0 * np.arange(2, 7)[:, None] + np.linspace(0.0, 8.0, 33)[None, :]
    values = []
    for steps in [10, 20, 40]:
        monkeypatch.setattr(point_availability, "_CHAIN_STEPS", steps)
        values.append(rbd.point_availability(t.ravel()))
    ratio = (
        np.abs(values[1] - values[0]).max()
        / np.abs(values[2] - values[1]).max()
    )
    assert ratio > 10.0


# -- Arguments ----------------------------------------------------------------


def test_shapes_and_errors():
    rbd = alone(unit(E([0.1]), E([1.0])))
    assert isinstance(rbd.point_availability(1.0), float)
    assert isinstance(rbd.mission_availability(1.0), float)
    grid = np.array([[0.0, 1.0], [2.0, 3.0]])
    assert rbd.point_availability(grid).shape == (2, 2)
    assert rbd.mission_availability(grid).shape == (2, 2)
    assert rbd.point_availability([]).shape == (0,)
    assert rbd.mission_availability(0.0) == rbd.point_availability(0.0) == 1.0
    for bad in [-1.0, np.nan, np.inf, [1.0, -0.5]]:
        with pytest.raises(ValueError):
            rbd.point_availability(bad)
        with pytest.raises(ValueError):
            rbd.mission_availability(bad)


def test_what_it_does_not_cover():
    # Hidden failures found by tests that take time, as for the long run.
    inspected = alone(
        unit(
            W([60.0, 1.5]),
            "instant",
            inspection={"interval": 10.0, "duration": E([1.0])},
        )
    )
    with pytest.raises(NotImplementedError, match="hidden failures"):
        inspected.point_availability(5.0)
    # Block replacement of units some of which are dead on arrival.
    dead = alone(
        unit(W([100.0, 2.0], f0=0.1), E([0.5]), preventive=block(50.0))
    )
    with pytest.raises(NotImplementedError, match="dead on arrival"):
        dead.mission_availability(100.0)
    # A probability rather than a lifetime distribution.
    fixed = alone(unit(surv.FixedEventProbability.from_params(0.1), E([1.0])))
    with pytest.raises(NotImplementedError, match="not a distribution"):
        fixed.point_availability(5.0)


def test_identical_components_share_one_curve(monkeypatch):
    # The same life and repair (equal models, not one object), the same
    # state at 0: one curve, giving what a curve each gives.
    def spec(scale=100.0):
        return unit(W([scale, 1.6]), LN([0.5, 0.6]))

    rbd = RepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "c"), ("b", "c"), ("c", "t")],
        {"a": spec(), "b": spec(), "c": spec(120.0)},
    )
    built = []
    plain = RepairableRBD._unit_curve

    def counted(self, node, *args, **kwargs):
        built.append(node)
        return plain(self, node, *args, **kwargs)

    monkeypatch.setattr(RepairableRBD, "_unit_curve", counted)
    x = np.array([0.0, 30.0, 300.0])
    shared = (
        rbd.point_availability(x),
        rbd.mission_availability(1000.0),
        rbd.expected_failures(1000.0),
    )
    assert built == ["a", "c"] * 3
    # Another state at 0 is another curve.
    built.clear()
    rbd.point_availability(x, state={"b": NodeState(age=50.0)})
    assert built == ["a", "b", "c"]
    monkeypatch.setattr(RepairableRBD, "_curve_twin", lambda *a: None)
    apart = (
        rbd.point_availability(x),
        rbd.mission_availability(1000.0),
        rbd.expected_failures(1000.0),
    )
    np.testing.assert_array_equal(shared[0], apart[0])
    assert shared[1:] == apart[1:]


def test_a_curve_is_followed_about_as_far_as_it_takes_to_settle():
    # Its distance from the long-run value falls cycle by cycle: the curve
    # is followed until it has kept within 1e-10 for a quarter of its
    # length, not four times as far (as before) when it has not yet.
    rbd = alone(unit(W([80.0, 1.6]), LN([0.5, 0.6])))
    curve = rbd._unit_curve("c", 5000.0)
    off = np.abs(curve.at(curve.times) - curve.long_run)
    settled = curve.times[np.flatnonzero(off >= 1e-10)[-1]]
    tail = curve.times[len(curve.times) - len(curve.times) // 4 :]
    assert np.all(np.abs(curve.at(tail) - curve.long_run) < 1e-10)
    assert curve.times[-1] < 2.0 * settled
