"""Availability allocation (#107): ``RepairableRBD.availability_allocation``
and ``RepairableRBD.mttf_mttr_allocation``."""

import math

import numpy as np
import pytest
import surpyval as surv

from repyability import AvailabilityAllocation, RepairableRBD
from repyability.rbd._model_utils import lfp_extras

PLANT = [("s", "p1"), ("s", "p2"), ("p1", "v"), ("p2", "v"), ("v", "t")]
SERIES = [("s", "a"), ("a", "b"), ("b", "c"), ("c", "t")]
METHODS = ["cost_based", "improvement", "equal"]
PLANT_NODES = ("p1", "p2", "v")


def unit(mttf, mttr):
    return {
        "reliability": surv.Exponential.from_params([1 / mttf]),
        "repairability": surv.Exponential.from_params([1 / mttr]),
    }


def weibull_unit(mttf, mttr):
    """A Weibull life (shape 2) and a Weibull repair (shape 1.5) with these
    means: scaling a model's scale parameter scales its mean."""
    life = mttf / math.gamma(1.5)
    repair = mttr / math.gamma(1 + 1 / 1.5)
    return {
        "reliability": surv.Weibull.from_params([life, 2.0]),
        "repairability": surv.Weibull.from_params([repair, 1.5]),
    }


def plant(make=unit, mttf=None, mttr=None, **extra):
    """Two pumps in parallel (MTTF 10, MTTR 1) in series with a valve (MTTF
    50, MTTR 2), with any MTTFs and MTTRs replaced."""
    means = {"p1": [10.0, 1.0], "p2": [10.0, 1.0], "v": [50.0, 2.0]}
    for node, value in (mttf or {}).items():
        means[node][0] = value
    for node, value in (mttr or {}).items():
        means[node][1] = value
    components = {node: make(*pair) for node, pair in means.items()}
    components.update(extra)
    return RepairableRBD(PLANT, components)


def valve(interval=2000.0):
    """A valve whose failures are hidden, found by a proof test."""
    return {
        "reliability": surv.Exponential.from_params([1e-4]),
        "repairability": "instant",
        "inspection": {"interval": interval},
    }


def proof_tested(pump):
    """Two proof-tested valves in parallel, tested together, in series with
    a pump."""
    return RepairableRBD(
        [("s", "v1"), ("s", "v2"), ("v1", "pump"), ("v2", "pump")]
        + [("pump", "t")],
        {"v1": valve(), "v2": valve(), "pump": pump},
    )


# -- one component -----------------------------------------------------------


@pytest.mark.parametrize("method", METHODS + ["minimum_effort"])
def test_one_component_gets_exactly_the_target(method):
    rbd = RepairableRBD([("s", "c"), ("c", "t")], {"c": unit(100.0, 5.0)})
    need = rbd.availability_allocation(0.99, method)
    assert isinstance(need, AvailabilityAllocation)
    assert need.availability["c"] == pytest.approx(0.99, rel=1e-12)
    # MTTF / (MTTF + MTTR) = 0.99 at the current MTTR, or MTTR at the MTTF.
    assert need.mttf["c"] == pytest.approx(5.0 * 99.0, rel=1e-9)
    assert need.mttr["c"] == pytest.approx(100.0 / 99.0, rel=1e-9)
    assert need.system_availability == pytest.approx(0.99, rel=1e-12)
    for spec in (unit(need.mttf["c"], 5.0), unit(100.0, need.mttr["c"])):
        again = RepairableRBD([("s", "c"), ("c", "t")], {"c": spec})
        assert again.mean_availability() == pytest.approx(0.99, rel=1e-12)


def test_one_component_levers_in_closed_form():
    rbd = RepairableRBD([("s", "c"), ("c", "t")], {"c": unit(100.0, 5.0)})
    repairs = rbd.mttf_mttr_allocation(0.99, levers="mttr")
    assert repairs.mttf["c"] == 100.0
    assert repairs.mttr["c"] == pytest.approx(100.0 / 99.0, rel=1e-8)
    lives = rbd.mttf_mttr_allocation(0.99, levers="mttf")
    assert lives.mttf["c"] == pytest.approx(495.0, rel=1e-8)
    assert lives.mttr["c"] == 5.0
    # At equal feasibilities the MTTF rises by the factor the MTTR falls by,
    # k, with (MTTR / MTTF) / k**2 = 1 / 99.
    k = math.sqrt(0.05 * 99.0)
    both = rbd.mttf_mttr_allocation(0.99)
    assert both.mttf["c"] == pytest.approx(100.0 * k, rel=1e-6)
    assert both.mttr["c"] == pytest.approx(5.0 / k, rel=1e-6)
    assert both.availability["c"] == pytest.approx(0.99, rel=1e-12)


# -- the plant ---------------------------------------------------------------


def test_plant_matches_the_probability_allocation():
    rbd = plant()
    need = rbd.availability_allocation(0.98)
    current = {
        n: a for n, a in rbd.node_availability().items() if n in PLANT_NODES
    }
    expected = rbd.cost_based_allocation(0.98, current)
    assert need.availability == pytest.approx(expected, rel=1e-12)
    # The figures quoted in the issue.
    assert need.availability["p1"] == pytest.approx(0.9318, abs=5e-5)
    assert need.availability["v"] == pytest.approx(0.9846, abs=5e-5)
    assert need.mttf["p1"] == pytest.approx(13.66, abs=0.005)
    assert need.mttr["p1"] == pytest.approx(0.732, abs=0.0005)
    assert need.mttf["v"] == pytest.approx(127.7, abs=0.05)
    assert need.mttr["v"] == pytest.approx(0.783, abs=0.0005)
    assert need.system_availability == pytest.approx(0.98, rel=1e-12)
    assert dict(need)["mttf"] is need.mttf


@pytest.mark.parametrize("make", [unit, weibull_unit])
@pytest.mark.parametrize("method", METHODS)
def test_applying_the_allocation_meets_the_target(method, make):
    need = plant(make).availability_allocation(0.98, method)
    # Either lever alone: the MTTFs at the current MTTRs, or the MTTRs at
    # the current MTTFs.
    longer = plant(make, mttf=need.mttf)
    shorter = plant(make, mttr=need.mttr)
    assert longer.mean_availability() == pytest.approx(0.98, rel=1e-10)
    assert shorter.mean_availability() == pytest.approx(0.98, rel=1e-10)


@pytest.mark.parametrize("make", [unit, weibull_unit])
@pytest.mark.parametrize("levers", ["both", "mttf", "mttr"])
def test_applying_the_design_meets_the_target(levers, make):
    rbd = plant(make)
    design = rbd.mttf_mttr_allocation(0.98, levers=levers)
    assert rbd.res.success
    applied = plant(make, mttf=design.mttf, mttr=design.mttr)
    assert applied.mean_availability() == pytest.approx(0.98, rel=1e-10)
    assert design.system_availability == pytest.approx(0.98, rel=1e-10)
    if levers == "mttr":
        assert design.mttf == pytest.approx({"p1": 10, "p2": 10, "v": 50})
    if levers == "mttf":
        assert design.mttr == pytest.approx({"p1": 1, "p2": 1, "v": 2})


def test_cheap_repair_changes_only_the_mttrs():
    everywhere = dict.fromkeys(["p1", "p2", "v"])
    design = plant().mttf_mttr_allocation(
        0.98,
        mttf_feasibility=dict.fromkeys(everywhere, 0.1),
        mttr_feasibility=dict.fromkeys(everywhere, 0.9),
    )
    assert design.mttf == {"p1": 10.0, "p2": 10.0, "v": 50.0}
    assert all(design.mttr[n] < now for n, now in [("p1", 1), ("v", 2)])
    # And the reverse.
    design = plant().mttf_mttr_allocation(
        0.98,
        mttf_feasibility=dict.fromkeys(everywhere, 0.9),
        mttr_feasibility=dict.fromkeys(everywhere, 0.1),
    )
    assert design.mttr == {"p1": 1.0, "p2": 1.0, "v": 2.0}
    assert all(design.mttf[n] > now for n, now in [("p1", 10), ("v", 50)])


def test_equal_feasibilities_use_both_levers_alike():
    design = plant().mttf_mttr_allocation(0.98)
    for node, (mttf, mttr) in {"p1": (10, 1), "v": (50, 2)}.items():
        assert design.mttf[node] / mttf == pytest.approx(
            mttr / design.mttr[node], rel=1e-5
        )
        assert design.mttf[node] > mttf


def test_limits_are_kept():
    # The valve, which every path goes through, can reach 80 / 81.
    design = plant().mttf_mttr_allocation(
        0.98, max_mttf={"v": 80.0}, min_mttr={"v": 1.0, "p1": 1.0}
    )
    assert 50.0 < design.mttf["v"] < 80.0
    assert 1.0 < design.mttr["v"] < 2.0
    assert design.mttr["p1"] == 1.0  # held by its limit
    applied = plant(mttf=design.mttf, mttr=design.mttr)
    assert applied.mean_availability() == pytest.approx(0.98, rel=1e-10)


# -- series systems in closed form -------------------------------------------


def series(means):
    return RepairableRBD(
        SERIES, {node: unit(*pair) for node, pair in zip("abc", means)}
    )


def test_equal_apportionment_in_series():
    means = [(100.0, 5.0), (40.0, 1.0), (300.0, 20.0)]
    need = series(means).availability_allocation(0.95, "equal")
    share = 0.95 ** (1 / 3)
    for node, (mttf, mttr) in zip("abc", means):
        assert need.availability[node] == pytest.approx(share, rel=1e-10)
        assert need.mttf[node] == pytest.approx(
            mttr * share / (1 - share), rel=1e-8
        )
        assert need.mttr[node] == pytest.approx(
            mttf * (1 - share) / share, rel=1e-8
        )


def test_identical_components_in_series():
    rbd = series([(50.0, 2.0)] * 3)
    design = rbd.mttf_mttr_allocation(0.95)
    share = 0.95 ** (1 / 3)
    k = math.sqrt((2.0 / 50.0) * share / (1 - share))
    for node in "abc":
        assert design.availability[node] == pytest.approx(share, rel=1e-9)
        assert design.mttf[node] == pytest.approx(50.0 * k, rel=1e-6)
        assert design.mttr[node] == pytest.approx(2.0 / k, rel=1e-6)


def test_maintainability_allocation_in_series():
    """With the MTTFs held, the cheapest MTTRs: each MTTR is cut to
    ``MTTR_0 * exp(-v)`` at a cost ``exp(s * expm1(v))``, and in series the
    system's log-odds rises at the component's unavailability ``U`` per unit
    of ``v`` (over ``1 - A_sys``). At the optimum the marginal cost per unit
    of that rise is the same for every component; and the components that
    fail most often get the shortest repairs."""
    means = [(20.0, 2.0), (50.0, 2.0), (100.0, 2.0)]
    rbd = series(means)
    design = rbd.mttf_mttr_allocation(0.95, levers="mttr")
    s = 0.5
    ratios = []
    for node, (mttf, mttr) in zip("abc", means):
        v = math.log(mttr / design.mttr[node])
        unavailability = 1 - design.availability[node]
        ratios.append(s * math.exp(v + s * math.expm1(v)) / unavailability)
    assert ratios == pytest.approx([ratios[0]] * 3, rel=1e-5)
    assert design.mttr["a"] < design.mttr["b"] < design.mttr["c"]


def test_minimum_effort_in_series():
    rbd = series([(100.0, 5.0), (40.0, 1.0), (300.0, 20.0)])
    current = {n: a for n, a in rbd.node_availability().items() if n in "abc"}
    need = rbd.availability_allocation(0.9, "minimum_effort")
    expected = rbd.minimum_effort_allocation(0.9, current)
    assert need.availability == pytest.approx(expected, rel=1e-12)
    # With c held, a and b make up the rest of the target.
    need = rbd.availability_allocation(0.9, "minimum_effort", fixed=["c"])
    assert need.availability["c"] == current["c"]
    assert set(need.mttf) == {"a", "b"}
    assert need.system_availability == pytest.approx(0.9, rel=1e-12)


# -- components that keep their availability ---------------------------------


def test_held_components_keep_their_availability():
    nested = RepairableRBD([("s", "x"), ("x", "t")], {"x": unit(30.0, 3.0)})
    cured = {
        "reliability": surv.Weibull.from_params(
            [20.0, 2.0], **lfp_extras(0.7)
        ),
        "repairability": surv.Exponential.from_params([0.5]),
    }
    rbd = RepairableRBD(
        [("s", n) for n in ["a", "b", "c", "d", "e", "f", "g"]]
        + [("s", "h"), ("h", "t")]
        + [(n, "h") for n in ["a", "b", "c", "d", "e", "f", "g"]],
        {
            "a": unit(10.0, 1.0),
            "b": {
                **unit(10.0, 1.0),
                "preventive": {"interval": 5.0, "duration": "instant"},
            },
            "c": valve(20.0),
            "d": {**unit(10.0, 1.0), "repairability": "instant"},
            "e": nested,
            "f": cured,
            "g": unit(10.0, 1.0),
            "h": unit(100.0, 1.0),
        },
    )
    before = rbd.node_availability()
    need = rbd.availability_allocation(0.995, fixed=["g"])
    assert set(need.mttf) == set(need.mttr) == {"a", "h"}
    for node in "bcdefg":
        assert need.availability[node] == before[node]
    assert need.system_availability == pytest.approx(0.995, rel=1e-12)
    design = rbd.mttf_mttr_allocation(0.995, fixed=["g"])
    assert set(design.mttf) == {"a", "h"}
    for node in "bcdefg":
        assert design.availability[node] == before[node]
    assert design.system_availability == pytest.approx(0.995, rel=1e-10)


@pytest.mark.parametrize("method", METHODS)
def test_inspected_components_enter_over_their_calendar(method):
    """Two valves tested together are down together more often than two
    independent ones: the pump's share is worked out over the test
    calendar, as mean_availability works it out, not from the valves' mean
    availabilities."""
    rbd = proof_tested(unit(100.0, 10.0))
    need = rbd.availability_allocation(0.95, method)
    for pump in (
        unit(need.mttf["pump"], 10.0),
        unit(100.0, need.mttr["pump"]),
    ):
        assert proof_tested(pump).mean_availability() == pytest.approx(
            0.95, rel=1e-10
        )
    valves = rbd.node_availability()["v1"]
    independent = 0.95 / (1 - (1 - valves) ** 2)
    assert need.availability["pump"] > independent + 1e-3
    if method != "equal":
        assert rbd.res.success


@pytest.mark.parametrize("levers", ["both", "mttf", "mttr"])
def test_designs_over_the_calendar(levers):
    rbd = proof_tested(unit(100.0, 10.0))
    design = rbd.mttf_mttr_allocation(0.95, levers=levers)
    pump = unit(design.mttf["pump"], design.mttr["pump"])
    assert proof_tested(pump).mean_availability() == pytest.approx(
        0.95, rel=1e-10
    )


def test_minimum_effort_over_the_calendar():
    """In series with an inspected valve, the other components make up the
    target over the valve's test calendar."""
    rbd = RepairableRBD(
        [("s", "v"), ("v", "a"), ("a", "b"), ("b", "t")],
        {"v": valve(), "a": unit(100.0, 5.0), "b": unit(40.0, 1.0)},
    )
    assert rbd.mean_availability() < 0.87
    need = rbd.availability_allocation(0.87, "minimum_effort")
    applied = RepairableRBD(
        [("s", "v"), ("v", "a"), ("a", "b"), ("b", "t")],
        {
            "v": valve(),
            "a": unit(need.mttf["a"], 5.0),
            "b": unit(need.mttf["b"], 1.0),
        },
    )
    assert applied.mean_availability() == pytest.approx(0.87, rel=1e-10)


# -- edges and errors --------------------------------------------------------


def test_a_target_already_met():
    rbd = plant()
    now = rbd.mean_availability()
    need = rbd.availability_allocation(0.9)
    assert need.availability == pytest.approx(
        {n: a for n, a in rbd.node_availability().items() if n in PLANT_NODES}
    )
    assert need.system_availability == pytest.approx(now)
    design = rbd.mttf_mttr_allocation(0.9)
    assert design.mttf == {"p1": 10.0, "p2": 10.0, "v": 50.0}
    assert design.mttr == {"p1": 1.0, "p2": 1.0, "v": 2.0}
    assert rbd.res.message == "The target is already met."


def test_perfect_components():
    need = plant().availability_allocation(1.0, "improvement")
    assert need.mttf == {"p1": math.inf, "p2": math.inf, "v": math.inf}
    assert need.mttr == {"p1": 0.0, "p2": 0.0, "v": 0.0}


def test_integer_node_names():
    rbd = RepairableRBD(
        [("s", 1), ("s", 2), (1, 3), (2, 3), (3, "t")],
        {1: unit(10.0, 1.0), 2: unit(10.0, 1.0), 3: unit(50.0, 2.0)},
    )
    need = rbd.availability_allocation(0.98, feasibility={3: 0.2})
    assert set(need.mttf) == {1, 2, 3}
    design = rbd.mttf_mttr_allocation(0.98, min_mttr={3: 1.0})
    assert design.mttr[3] > 1.0


def test_the_reliability_allocation_is_unchanged():
    """The allocation methods of the RBD itself still score plain node
    probabilities with the structure function, calendar or not."""
    rbd = proof_tested(unit(100.0, 10.0))
    new = rbd.improvement_allocation(0.9, {"v1": 0.9, "v2": 0.9, "pump": 0.9})
    probability = rbd.system_probability(
        {n: np.atleast_1d(p) for n, p in new.items()}
    )
    assert float(probability[0]) == pytest.approx(0.9, rel=1e-12)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"method": "arinc"}, "method must be one of"),
        ({"weights": {"v": 1.0}}, "weights applies to method='improvement'"),
        (
            {"method": "equal", "feasibility": {"v": 0.2}},
            "feasibility applies to method='cost_based'",
        ),
        (
            {"method": "improvement", "max_availabilities": {"v": 0.99}},
            "max_availabilities applies",
        ),
        ({"target": 1.5}, "target must be a number in"),
        ({"target": "high"}, "target must be a number in"),
        ({"fixed": ["pump"]}, "not components of this RBD"),
        ({"fixed": ["v"], "feasibility": {"v": 0.2}}, "keep their"),
        ({"feasibility": {"x": 0.2}}, "not intermediate nodes"),
        ({"fixed": ["p1", "p2", "v"]}, "No component can be allocated"),
        ({"fixed": ["v"], "target": 0.99}, "cannot be reached"),
        ({"method": "minimum_effort"}, "series system"),
        (
            {"max_availabilities": {"v": 0.97}, "target": 0.99},
            "cannot be reached",
        ),
    ],
)
def test_availability_allocation_errors(kwargs, message):
    kwargs = {"target": 0.98, **kwargs}
    with pytest.raises(ValueError, match=message):
        plant().availability_allocation(**kwargs)


def test_weights_need_every_component_allocated():
    with pytest.raises(KeyError):
        plant().availability_allocation(
            0.98, "improvement", weights={"p1": 1.0, "p2": 1.0}
        )


def test_minimum_effort_held_back():
    rbd = series([(100.0, 5.0), (40.0, 1.0), (300.0, 20.0)])
    with pytest.raises(ValueError, match="hold the system to at most"):
        rbd.availability_allocation(0.95, "minimum_effort", fixed=["c"])


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"levers": "repair"}, "levers must be"),
        ({"target": -0.1}, "target must be a number in"),
        ({"max_mttf": {"v": 40.0}}, "max_mttf.*at least the component's"),
        ({"min_mttr": {"v": 3.0}}, "min_mttr.*between 0 and"),
        ({"min_mttr": {"v": -1.0}}, "min_mttr.*between 0 and"),
        ({"mttr_feasibility": {"v": 1.0}}, r"mttr_feasibility\['v'\]"),
        ({"mttf_feasibility": {"v": "easy"}}, r"mttf_feasibility\['v'\]"),
        ({"fixed": ["v"], "max_mttf": {"v": 60.0}}, "keep their"),
        (
            {
                "levers": "mttr",
                "min_mttr": {"p1": 1.0, "p2": 1.0, "v": 2.0},
            },
            "No MTTF or MTTR can change",
        ),
        (
            {"max_mttf": {"v": 60.0}, "min_mttr": {"v": 1.9}},
            "cannot be reached",
        ),
        ({"target": 1.0}, "cannot be reached"),
    ],
)
def test_mttf_mttr_allocation_errors(kwargs, message):
    kwargs = {"target": 0.98, **kwargs}
    with pytest.raises(ValueError, match=message):
        plant().mttf_mttr_allocation(**kwargs)


def test_a_design_cut_short_still_meets_the_target(monkeypatch):
    import repyability.rbd.repairable_rbd as module

    real = module.minimize

    def one_step(*args, **kwargs):
        kwargs["options"] = {**kwargs.get("options", {}), "maxiter": 1}
        return real(*args, **kwargs)

    monkeypatch.setattr(module, "minimize", one_step)
    rbd = plant()
    with pytest.warns(UserWarning, match="stopped before converging"):
        design = rbd.mttf_mttr_allocation(0.98)
    assert not rbd.res.success
    applied = plant(mttf=design.mttf, mttr=design.mttr)
    assert applied.mean_availability() >= 0.98 * (1 - 1e-12)
