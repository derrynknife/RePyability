"""Load-sharing groups and blocks of identical units on a ``RepairableRBD``
(a component spec's ``"load_sharing"`` and ``"identical"``).

A load-sharing group's units share a load: while ``s`` work, each carries
``load / s`` and ages at its life's ``phi(load / s)``, and each failed
unit is repaired on its own. With an exponential baseline and repairs the
group is a birth-death chain, whose long-run and transient values are
worked out by hand here and checked against the exact methods and the
simulation. A block of identical units is the diagram with its units
drawn out, as one node.
"""

import math

import numpy as np
import pytest
import surpyval as surv
from scipy.linalg import expm

from repyability import LoadSharingModel, RepairableRBD
from repyability.maintenance import UnitPlan

#: Standard errors a simulated estimate may be from the exact value.
Z = 4.0

SERIES = [("s", "g"), ("g", "t")]


def _aft(family, params):
    """A surpyval AFT model of ``family`` with the load as its covariate,
    given ``params`` (fitted to small data, then set through its saved
    form)."""
    x = np.array([50.0, 100.0, 150.0, 12.5, 25.0, 37.5])
    load = np.array([[1.0], [1.0], [1.0], [2.0], [2.0], [2.0]])
    saved = family.fit(x, Z=load).to_dict()
    saved["params"] = [float(p) for p in params]
    return surv.from_dict(saved)


#: Baseline rate 0.005 and phi(z) = 2 ** z: with a load of 2, each of two
#: working units fails at 0.01, and a unit working alone at 0.02.
LAM0, BETA = 0.005, math.log(2.0)
LAM2, LAM1 = 0.01, 0.02
MU = 0.1


def pair(**options):
    """Two pumps sharing a load of 2, either enough, repaired at ``MU``."""
    spec = {
        "reliability": _aft(surv.ExponentialAFT, [LAM0, BETA]),
        "repairability": surv.Exponential.from_params([MU]),
        "load_sharing": {"units": 2, "k": 1, "load": 2.0},
        **options.pop("spec", {}),
    }
    return RepairableRBD(SERIES, {"g": spec}, **options)


def pair_generator(crews=None):
    """The pair's chain on (2 working, 1 working, none), by hand."""
    both = MU if crews == 1 else 2.0 * MU
    q = np.array(
        [
            [-2.0 * LAM2, 2.0 * LAM2, 0.0],
            [MU, -(MU + LAM1), LAM1],
            [0.0, both, -both],
        ]
    )
    return q


def pair_long_run(crews=None):
    """The pair's long-run availability and failure frequency in closed
    form: pi1 = (2 lam2 / mu) pi2, pi0 = (lam1 / (c mu)) pi1, the group
    failing from one working unit at lam1."""
    c = 1.0 if crews == 1 else 2.0
    p2 = 1.0
    p1 = 2.0 * LAM2 / MU * p2
    p0 = LAM1 / (c * MU) * p1
    total = p2 + p1 + p0
    return 1.0 - p0 / total, p1 / total * LAM1


# -- the exact values ---------------------------------------------------------


def test_the_pair_takes_its_rates_from_the_load_it_shares():
    rbd = pair()
    arrangement = rbd._standby["g"]
    assert arrangement.rates == pytest.approx((4.0, 2.0))
    # Each unit draws its life at no acceleration.
    assert rbd.components["g"].reliability.dist.name == "Exponential"
    assert float(rbd.components["g"].reliability.mean()) == pytest.approx(
        1.0 / LAM0
    )


@pytest.mark.parametrize("crews", [None, 1])
def test_the_pair_long_run_is_the_closed_form(crews):
    rbd = pair(repair_crews=crews)
    availability, frequency = pair_long_run(crews)
    # 1 - 0.02 / 1.22 with a crew for each unit.
    if crews is None:
        assert availability == pytest.approx(1.0 - 0.02 / 1.22, rel=1e-12)
    assert rbd.mean_availability() == pytest.approx(availability, rel=1e-10)
    assert rbd.mean_unavailability() == pytest.approx(
        1.0 - availability, rel=1e-9
    )
    assert rbd.system_failure_frequency() == pytest.approx(
        frequency, rel=1e-10
    )
    assert rbd.analysis_routes()["mean_availability"].route == "exact"


def test_the_pair_over_time_is_its_chain_from_new():
    rbd = pair()
    times = np.array([1.0, 10.0, 50.0, 200.0, 2000.0])
    q = pair_generator()
    start = np.array([1.0, 0.0, 0.0])
    expected = [1.0 - (start @ expm(q * t))[2] for t in times]
    assert rbd.point_availability(times) == pytest.approx(expected, rel=1e-8)


def test_the_load_shared_makes_the_survivor_fail_sooner():
    # The same pair with no load effect is two independent units in
    # parallel, which are down together less often.
    shared = pair().mean_availability()
    independent = RepairableRBD(
        SERIES,
        {
            "g": {
                "reliability": surv.Exponential.from_params([LAM2]),
                "repairability": surv.Exponential.from_params([MU]),
                "identical": {"units": 2, "k": 1},
            }
        },
    ).mean_availability()
    q = LAM2 / (LAM2 + MU)
    assert independent == pytest.approx(1.0 - q * q, rel=1e-12)
    assert shared < independent


def test_no_load_effect_is_the_identical_block():
    # phi = 1 (no coefficient): the units age at their baseline rate
    # whatever the load, as identical, independent units.
    no_effect = pair(
        spec={"reliability": _aft(surv.ExponentialAFT, [LAM2, 0])}
    )
    q = LAM2 / (LAM2 + MU)
    assert no_effect.mean_availability() == pytest.approx(1 - q * q, rel=1e-12)


# -- the simulation -----------------------------------------------------------


def close(estimate, exact, error):
    assert abs(estimate - exact) <= Z * error + 1e-12, (estimate, exact)


def mean_se(result, window):
    """The standard error of a plain run's mean availability over the
    window, from its simulations' uptimes."""
    shares = np.asarray(result.uptimes, dtype=float) / window
    return float(np.std(shares, ddof=1) / np.sqrt(len(shares)))


@pytest.mark.parametrize("crews", [None, 1])
def test_the_simulation_follows_the_pair_chain(crews):
    rbd = pair(repair_crews=crews)
    window = 400.0
    result = rbd.availability(
        t_simulation=window, mc_samples=4000, seed=3, control_variate=False
    )
    exact = rbd.mission_availability(window)
    close(result.mean_availability, exact, mean_se(result, window))
    # The chain from new, by hand: the mean over the window of P(up).
    q = pair_generator(crews)
    grid = np.linspace(0.0, window, 2001)
    start = np.array([1.0, 0.0, 0.0])
    up = [1.0 - (start @ expm(q * t))[2] for t in grid]
    assert exact == pytest.approx(np.trapezoid(up, grid) / window, rel=1e-6)


def test_the_simulation_counts_each_unit_failure_as_a_repair():
    rbd = pair(spec={"repair_cost": 10.0})
    window = 400.0
    result = rbd.cost(
        t_simulation=window, mc_samples=3000, seed=4, control_variate=False
    )
    exact = rbd.expected_cost(window)
    close(result.mean, exact.mean, result.mean_se)


def test_a_weibull_pair_with_no_repair_lives_as_the_load_sharing_model():
    # Repairs that never end: the group's first life is the non-repairable
    # LoadSharingModel's, so the availability over a window is the mean of
    # its reliability.
    life = _aft(surv.WeibullAFT, [60.0, 2.0, -0.5])
    rbd = RepairableRBD(
        SERIES,
        {
            "g": {
                "reliability": life,
                "repairability": surv.ExactEventTime.from_params([1e9]),
                "load_sharing": {"units": 2, "load": 2.0},
            }
        },
    )
    window = 150.0
    group = LoadSharingModel([life, life], load=2.0, k=1)
    grid = np.linspace(0.0, window, 3001)
    expected = np.trapezoid(group.sf(grid), grid) / window
    result = rbd.availability(t_simulation=window, mc_samples=4000, seed=7)
    # The group's lives are about 1e-4 of a probability off (numerical).
    close(result.mean_availability, float(expected), mean_se(result, window))
    assert rbd.analysis_routes()["mean_availability"].route == "refused"


# -- blocks of identical units ------------------------------------------------


def drawn_out(life, repair, units, k):
    """The block's units drawn out, in parallel to a k-of-n vote."""
    names = [f"u{i}" for i in range(units)]
    return RepairableRBD(
        [("s", n) for n in names] + [(n, "t") for n in names],
        {n: {"reliability": life, "repairability": repair} for n in names},
        k={"t": k} if k > 1 else None,
    )


def block(life, repair, units, k, **more):
    return RepairableRBD(
        SERIES,
        {
            "g": {
                "reliability": life,
                "repairability": repair,
                "identical": {"units": units, "k": k},
                **more,
            }
        },
    )


@pytest.mark.parametrize("units, k", [(3, 1), (3, 2), (2, 2), (1, 1)])
def test_an_exponential_block_is_its_units_drawn_out(units, k):
    life = surv.Exponential.from_params([0.01])
    repair = surv.Exponential.from_params([0.2])
    one, many = block(life, repair, units, k), drawn_out(
        life, repair, units, k
    )
    assert one.mean_availability() == pytest.approx(
        many.mean_availability(), rel=1e-10
    )
    assert one.system_failure_frequency() == pytest.approx(
        many.system_failure_frequency(), rel=1e-9
    )

    # Up while at least k of n units are, each up with its closed form,
    # mu / (lam + mu) in the long run.
    def binomial(a):
        return sum(
            math.comb(units, j) * a**j * (1 - a) ** (units - j)
            for j in range(k, units + 1)
        )

    assert one.mean_availability() == pytest.approx(
        binomial(0.2 / 0.21), rel=1e-10
    )
    times = np.array([5.0, 50.0, 500.0])
    each = 0.2 / 0.21 + 0.01 / 0.21 * np.exp(-0.21 * times)
    assert one.point_availability(times) == pytest.approx(
        [binomial(a) for a in each], rel=1e-10
    )


@pytest.mark.parametrize("units, k", [(3, 1), (2, 2)])
def test_a_weibull_block_simulates_as_its_units_drawn_out(units, k):
    life = surv.Weibull.from_params([100.0, 2.0])
    repair = surv.Weibull.from_params([10.0, 1.5])
    window = 300.0
    result = block(life, repair, units, k).availability(
        t_simulation=window, mc_samples=3000, seed=11, control_variate=False
    )
    exact = drawn_out(life, repair, units, k).mission_availability(window)
    close(result.mean_availability, exact, mean_se(result, window))


def test_a_weibull_block_is_simulated_with_its_units_as_its_twin():
    life = surv.Weibull.from_params([100.0, 2.0])
    repair = surv.Weibull.from_params([10.0, 1.5])
    rbd = block(life, repair, 3, 2)
    assert rbd.analysis_routes()["mission_availability"].route == "refused"
    window = 300.0
    controlled = rbd.availability(
        t_simulation=window, mc_samples=500, seed=2, control_variate=True
    )
    exact = drawn_out(life, repair, 3, 2).mission_availability(window)
    # Its twin is the block itself, so the controlled estimate is exact.
    assert controlled.mean_availability == pytest.approx(exact, rel=1e-6)


# -- saving and plans ---------------------------------------------------------


def test_a_load_sharing_diagram_saves_and_loads():
    rbd = pair(spec={"repair_cost": 5.0})
    again = RepairableRBD.from_dict(rbd.to_dict())
    assert again._standby == rbd._standby
    assert again.mean_availability() == pytest.approx(rbd.mean_availability())


def test_a_block_reads_back_as_a_plan():
    life = surv.Exponential.from_params([0.01])
    repair = surv.Exponential.from_params([0.2])
    rbd = block(life, repair, 3, 2)
    plan = rbd.plans["g"]
    assert plan.identical == {"units": 3, "k": 2}
    assert UnitPlan.from_spec(plan.spec()) == plan


# -- refusals ----------------------------------------------------------------


def spec(**more):
    return {
        "reliability": _aft(surv.ExponentialAFT, [LAM0, BETA]),
        "repairability": surv.Exponential.from_params([MU]),
        **more,
    }


@pytest.mark.parametrize(
    "component, match",
    [
        (
            {
                "reliability": surv.Exponential.from_params([0.01]),
                "repairability": surv.Exponential.from_params([MU]),
                "load_sharing": {"load": 2.0},
            },
            "accelerated-failure-time",
        ),
        (spec(load_sharing={"units": 2}), "needs its 'load'"),
        (spec(load_sharing={"units": 2, "load": -1.0}), "positive"),
        (spec(load_sharing={"units": 2, "k": 3, "load": 1.0}), "more than"),
        (spec(load_sharing={"units": 2, "lod": 1.0}), "unknown"),
        (spec(load_sharing=[2, 1]), "must be a dict"),
        (
            spec(
                load_sharing={"load": 1.0},
                preventive={"interval": 10.0},
            ),
            "preventive schedule",
        ),
        (
            {**spec(load_sharing={"load": 1.0}), "repairability": "instant"},
            "repair time",
        ),
        (spec(identical={"units": 3}), "needs its 'units' and its 'k'"),
        (spec(identical={"units": 0, "k": 1}), "whole number"),
        (
            spec(identical={"units": 2, "k": 1}, standby={"units": 2}),
            "one group of units at most",
        ),
    ],
)
def test_what_a_group_does_not_take_is_refused(component, match):
    with pytest.raises(ValueError, match=match):
        RepairableRBD(SERIES, {"g": component})


def test_a_load_sharing_group_is_no_common_cause_member():
    from repyability import BetaFactor, CCFGroup

    unit = {
        "reliability": surv.Exponential.from_params([0.01]),
        "repairability": surv.Exponential.from_params([MU]),
    }
    with pytest.raises(ValueError, match="standby group"):
        RepairableRBD(
            [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
            {"a": {**unit, "identical": {"units": 2, "k": 1}}, "b": unit},
            ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.1))],
        )
