"""The diagrams of every kind the analyses are checked on (#239).

One registry, for the routes test (every analysis does what
``analysis_routes()`` says), for the engines' agreement and for the
cross-checks of the exact and numerical routes against simulation
(``test_catalogue.py``). A new node kind, spec key or diagram option is
added here once, and ``test_catalogue.test_the_catalogue_has_every_kind``
fails until it is.
"""

import numpy as np
import surpyval as surv

from repyability import (
    MGL,
    BetaFactor,
    CCFGroup,
    DegradingNode,
    LoadSharingModel,
    NonRepairableRBD,
    PerfectReliability,
    PerfectUnreliability,
    RegressionNode,
    RepairableRBD,
    RepeatedNode,
    RepeatedStandbyNode,
    StandbyModel,
)
from repyability.rbd import bdd, modular
from repyability.tests.test_performance_equivalence import (
    binomial_first,
    instrument_air,
    pumps_with_capacities,
    repairable_rbds,
)

W = surv.Weibull.from_params
E = surv.Exponential.from_params
L = surv.LogNormal.from_params
FIXED = surv.FixedEventProbability.from_params
EDGES = [("s", "a"), ("s", "b"), ("a", "c"), ("b", "c"), ("c", "t")]
BRIDGE = [
    ("s", "a"),
    ("s", "b"),
    ("a", "c"),
    ("b", "c"),
    ("a", "d"),
    ("c", "d"),
    ("b", "e"),
    ("c", "e"),
    ("d", "t"),
    ("e", "t"),
]


JUNCTION = [
    ("s", "a"),
    ("s", "b"),
    ("a", "j"),
    ("b", "j"),
    ("j", "c"),
    ("c", "t"),
]
VOTE = [
    ("s", "a"),
    ("s", "b"),
    ("s", "d"),
    ("a", "j"),
    ("b", "j"),
    ("d", "j"),
    ("j", "c"),
    ("c", "t"),
]


def aft():
    """A pump's life at its load (surpyval's ExponentialAFT): 100 at load
    1, 25 at load 2."""
    x = np.array([50.0, 100.0, 150.0, 12.5, 25.0, 37.5])
    load = np.array([[1.0], [1.0], [1.0], [2.0], [2.0], [2.0]])
    return surv.ExponentialAFT.fit(x, Z=load)


def fitted_life():
    """A Weibull fitted to 30 failures: a fit with a parameter covariance."""
    return surv.Weibull.fit(np.random.default_rng(7).weibull(2, 30) * 100)


def shares_life():
    """A Weibull fitted to 30 failures, 20 units that never failed and 5
    dead on arrival: a fit of the share that ever fails (``lfp_p``) and
    the share dead on arrival (``f0``) too (#267)."""
    x = np.r_[
        np.random.default_rng(7).weibull(2, 30) * 100,
        np.full(20, 300.0),
        np.zeros(5),
    ]
    c = np.r_[np.zeros(30), np.ones(20), np.zeros(5)].astype(int)
    return surv.Weibull.fit(x, c=c, lfp=True, zi=True)


def crow_amsaa():
    """A Crow-AMSAA process fitted to 60 failures of a repairable unit
    (#269): minimal repair of a Weibull life."""
    gaps = np.random.default_rng(7).weibull(1.5, 60) * 300
    return surv.CrowAMSAA.fit(np.cumsum(gaps))


def mixture_life():
    """A two-mode population, infant mortality and wear-out (#227)."""
    g = np.random.default_rng(7)
    mix = surv.MixtureModel(surv.Weibull, 2)
    mix.fit(np.r_[20 * g.weibull(1, 15), 150 * g.weibull(3, 45)])
    return mix


def too_meshed(build):
    """``build()``, its core given up on as too meshed to work out (see
    ``modular.GraphStructure``), as a far larger one would be (#172)."""
    limit, method = bdd.STEP_LIMIT, modular.CORE_METHOD
    bdd.STEP_LIMIT, modular.CORE_METHOD = 2, "bdd"
    try:
        rbd = build()
    finally:
        bdd.STEP_LIMIT, modular.CORE_METHOD = limit, method
    assert rbd.structure_check["is_too_meshed"]
    return rbd


def nonrepairable_kinds():
    unit = W([100, 2])
    rest = {"b": W([80, 1.5]), "c": E([0.002])}
    return {
        "plain": NonRepairableRBD(EDGES, {"a": unit, **rest}),
        "convolved standby": NonRepairableRBD(
            EDGES, {"a": StandbyModel([unit, unit]), **rest}
        ),
        "simulated standby": NonRepairableRBD(
            EDGES,
            {
                "a": StandbyModel([unit] * 3, k=2, dormancy_factor=0.5),
                **rest,
            },
        ),
        "repeated standby": NonRepairableRBD(
            EDGES, {"a": RepeatedStandbyNode(unit, 2), **rest}
        ),
        "repeated": NonRepairableRBD(
            EDGES, {"a": RepeatedNode(unit, 2, "parallel"), **rest}
        ),
        "common cause": NonRepairableRBD(
            EDGES,
            {"a": unit, "b": unit, "c": E([0.002])},
            ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.1))],
        ),
        "common cause by rate": NonRepairableRBD(
            EDGES,
            {"a": unit, "b": unit, "c": E([0.002])},
            ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.1, basis="rate"))],
        ),
        "fixed": NonRepairableRBD(
            EDGES, {"a": FIXED(0.1), "b": FIXED(0.2), "c": FIXED(0.05)}
        ),
        "capacities": NonRepairableRBD(
            EDGES,
            {"a": unit, **rest},
            capacity={"a": 5.0, "b": 5.0, "c": 10.0},
        ),
        "unreplayable": NonRepairableRBD(
            EDGES, {"a": binomial_first([100, 2]), **rest}
        ),
        "nested": NonRepairableRBD(
            EDGES,
            {
                "a": NonRepairableRBD(
                    [("s", "x"), ("x", "t")],
                    {"x": StandbyModel([unit] * 3, k=2, dormancy_factor=0.5)},
                ),
                **rest,
            },
        ),
        "junction in series": NonRepairableRBD(
            JUNCTION, {"a": unit, **rest, "j": PerfectReliability}
        ),
        "common cause, MGL": NonRepairableRBD(
            EDGES,
            {"a": unit, "b": unit, "c": E([0.002])},
            ccf_groups=[CCFGroup(["a", "b"], MGL(0.1))],
        ),
        "a branch that never works": NonRepairableRBD(
            EDGES, {"a": unit, "b": PerfectUnreliability, "c": E([0.002])}
        ),
        "junction vote": NonRepairableRBD(
            VOTE,
            {
                "a": unit,
                "b": unit,
                "d": unit,
                "c": E([0.002]),
                "j": PerfectReliability,
            },
            k={"j": 2},
        ),
        "load sharing": NonRepairableRBD(
            EDGES, {"a": LoadSharingModel([aft(), aft()], load=2.0), **rest}
        ),
        "regression": NonRepairableRBD(
            EDGES, {"a": RegressionNode(aft(), covariates=[1.0]), **rest}
        ),
        "fitted": NonRepairableRBD(EDGES, {"a": fitted_life(), **rest}),
        "fitted shares": NonRepairableRBD(
            EDGES, {"a": shares_life(), **rest}
        ),
        "mixture": NonRepairableRBD(EDGES, {"a": mixture_life(), **rest}),
        "degrading": NonRepairableRBD(
            EDGES,
            {
                "a": DegradingNode([(10.0, unit), (5.0, E([0.01]))]),
                **rest,
            },
            capacity={"a": 10.0, "b": 10.0, "c": 10.0},
        ),
        "too meshed": too_meshed(
            lambda: NonRepairableRBD(
                BRIDGE,
                {"a": unit, "b": unit, "c": E([0.002]), "d": unit, "e": unit},
            )
        ),
        "too meshed, with capacities": too_meshed(
            lambda: NonRepairableRBD(
                BRIDGE,
                {"a": unit, "b": unit, "c": E([0.002]), "d": unit, "e": unit},
                capacity={"a": 5.0, "b": 5.0, "c": 5.0, "d": 5.0, "e": 5.0},
            )
        ),
    }


def systems_of_every_kind():
    """The repairable systems every simulation is checked on: those of
    ``test_performance_equivalence``, with draws that cannot be streamed
    too."""
    systems = dict(repairable_rbds())
    systems["instrument air"] = instrument_air()
    systems["capacities"] = pumps_with_capacities()
    # A maintenance time that cannot be streamed, likewise.
    systems["unstreamable maintenance"] = RepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        {
            name: {
                "reliability": W([70, 2]),
                "repairability": E([0.8]),
                "preventive": {
                    "interval": 30.0,
                    "duration": binomial_first([2, 1.5]),
                    "cost": 5.0,
                },
            }
            for name in "ab"
        },
    )
    # A component whose draws cannot be streamed draws from numpy's global
    # RNG, seeded for each simulation; the others still stream.
    systems["unstreamable"] = RepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        {
            "a": {"reliability": W([70, 1.5]), "repairability": E([0.8])},
            "b": {
                "reliability": binomial_first([40, 2]),
                "repairability": E([0.5]),
            },
        },
    )
    return systems


def repairable_kinds():
    life, repair = W([500, 1.5]), E([0.5])

    def unit(**more):
        return {"reliability": life, "repairability": repair, **more}

    def system(a, **options):
        return RepairableRBD(
            EDGES, {"a": a, "b": unit(), "c": unit()}, **options
        )

    out = dict(systems_of_every_kind())
    out.update(
        {
            "age replacement, priced": system(
                unit(preventive={"interval": 300.0}, replace_cost=10.0),
            ),
            "block replacement": system(
                unit(preventive={"interval": 300.0, "policy": "block"})
            ),
            "block replacement, in no time": system(
                unit(
                    repairability="instant",
                    preventive={"interval": 300.0, "policy": "block"},
                )
            ),
            "replaced on condition": system(
                unit(
                    preventive={
                        "interval": 100.0,
                        "policy": "condition",
                        "threshold": 0.1,
                        "inspection_cost": 1.0,
                    },
                    replace_cost=10.0,
                )
            ),
            "opportunistic maintenance": RepairableRBD(
                EDGES,
                {
                    node: unit(
                        preventive={"interval": 300.0, "opportunity": 200.0},
                        group="train",
                        replace_cost=10.0,
                    )
                    for node in "abc"
                },
                maintenance_groups={
                    "train": {"setup_cost": 50.0, "system_down": True}
                },
            ),
            "grouped, no opportunities": RepairableRBD(
                EDGES,
                {
                    "a": unit(preventive={"interval": 300.0}, group="train"),
                    "b": unit(group="train"),
                    "c": unit(),
                },
                maintenance_groups={"train": {"setup_cost": 50.0}},
            ),
            "grouped block replacements": RepairableRBD(
                EDGES,
                {
                    node: unit(
                        preventive={"interval": 300.0, "policy": "block"},
                        group="train",
                    )
                    for node in "ab"
                }
                | {"c": unit()},
                maintenance_groups={"train": {"setup_cost": 50.0}},
            ),
            "imperfect repair": system(
                unit(
                    repair={"model": "kijima1", "q": 0.5},
                    repair_cost=1.0,
                    replace_cost=10.0,
                )
            ),
            "minimal repair in no time": system(
                {
                    "reliability": life,
                    "repairability": "instant",
                    "repair": {"model": "kijima1", "q": 1.0},
                    "repair_cost": 1.0,
                    "replace_cost": 10.0,
                }
            ),
            "imperfect repair, replaced and maintained": system(
                unit(
                    repair={"model": "kijima2", "q": 0.8},
                    replace_after=3,
                    preventive={"interval": 300.0},
                    replace_cost=10.0,
                )
            ),
            "tested, constant rate": system(
                {
                    "reliability": E([0.002]),
                    "repairability": "instant",
                    "inspection": {"interval": 100.0},
                }
            ),
            "tested, Weibull": system(
                unit(repairability="instant", inspection={"interval": 100.0})
            ),
            "tested, taking time": system(
                unit(
                    repairability="instant",
                    inspection={"interval": 100.0, "duration": E([2.0])},
                )
            ),
            "tested, missing": system(
                unit(
                    inspection={
                        "interval": 100.0,
                        "coverage": 0.6,
                        "full_test": 300.0,
                    },
                )
            ),
            "tested, as long as the interval": system(
                unit(
                    repairability="instant",
                    inspection={"interval": 100.0, "duration": E([0.01])},
                )
            ),
            "simulated standby life": system(
                {
                    "reliability": StandbyModel(
                        [life] * 3, k=2, dormancy_factor=0.5
                    ),
                    "repairability": repair,
                }
            ),
            "fixed probability": system(
                {"reliability": FIXED(0.1), "repairability": repair}
            ),
            "mixture life": system(
                unit(reliability=mixture_life(), replace_cost=10.0)
            ),
            "fitted Crow-AMSAA process": system(
                unit(reliability=crow_amsaa(), repair_cost=1.0)
            ),
            "mixture life, maintained": system(
                unit(
                    reliability=mixture_life(),
                    preventive={"interval": 120.0},
                    replace_cost=10.0,
                )
            ),
            "mixture life, tested": system(
                unit(
                    reliability=mixture_life(),
                    repairability="instant",
                    inspection={"interval": 50.0},
                )
            ),
            "mixture life, imperfect repair": system(
                unit(
                    reliability=mixture_life(),
                    repair={"model": "kijima1", "q": 0.5},
                    repair_cost=1.0,
                )
            ),
            "degrading capacity": system(
                {
                    "reliability": DegradingNode(
                        [(100.0, E([0.004])), (50.0, E([0.004]))]
                    ),
                    "repairability": repair,
                },
                capacity={"b": 100.0, "c": 100.0},
            ),
            "one repair crew": system(
                unit(priority=1), repair_crews=1, downtime_cost_rate=5.0
            ),
            "enough repair crews": system(unit(), repair_crews=3),
            "too meshed": too_meshed(
                lambda: RepairableRBD(
                    BRIDGE, {n: unit(repair_cost=2.0) for n in "abcde"}
                )
            ),
            "too meshed, downtime priced": too_meshed(
                lambda: RepairableRBD(
                    BRIDGE,
                    {n: unit(repair_cost=2.0) for n in "abcde"},
                    downtime_cost_rate=5.0,
                    capacity={n: 5.0 for n in "abcde"},
                )
            ),
            "standby group": system(
                {
                    "reliability": E([0.002]),
                    "repairability": E([0.5]),
                    "standby": {"units": 3, "switching_probability": 0.95},
                    "repair_cost": 3.0,
                },
                downtime_cost_rate=5.0,
            ),
            "standby group, Weibull": system(
                unit(standby={"dormancy_factor": 0.5}, repair_cost=3.0),
                downtime_cost_rate=5.0,
            ),
            "common cause, tested": RepairableRBD(
                EDGES,
                {
                    "a": {
                        "reliability": E([0.002]),
                        "repairability": "instant",
                        "inspection": {
                            "interval": 100.0,
                            "coverage": 0.8,
                            "full_test": 300.0,
                        },
                    },
                    "b": {
                        "reliability": E([0.002]),
                        "repairability": "instant",
                        "inspection": {
                            "interval": 100.0,
                            "offset": 50.0,
                            "coverage": 0.8,
                            "full_test": 300.0,
                        },
                    },
                    "c": unit(repair_cost=2.0),
                },
                ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.1))],
                downtime_cost_rate=5.0,
            ),
            "common cause, revealed": RepairableRBD(
                EDGES,
                {
                    node: {
                        "reliability": E([0.002]),
                        "repairability": E([0.5]),
                    }
                    for node in "ab"
                }
                | {
                    "c": unit(
                        preventive={"interval": 300.0, "policy": "block"}
                    )
                },
                ccf_groups=[CCFGroup(["a", "b"], MGL(0.2))],
            ),
            "common cause, Weibull": system(
                unit(), ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.1))]
            ),
            "common cause, timed tests and repairs": RepairableRBD(
                EDGES,
                {
                    node: {
                        "reliability": E([0.002]),
                        "repairability": surv.ExactEventTime.from_params(
                            [8.0]
                        ),
                        "inspection": {
                            "interval": 100.0,
                            "offset": offset,
                            "duration": surv.ExactEventTime.from_params([2.0]),
                        },
                    }
                    for node, offset in (("a", 0.0), ("b", 50.0))
                }
                | {"c": unit(repair_cost=2.0)},
                ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.1))],
                downtime_cost_rate=5.0,
            ),
            "common cause, one repair crew": RepairableRBD(
                EDGES,
                {
                    node: {
                        "reliability": E([0.002]),
                        "repairability": E([0.5]),
                    }
                    for node in "abc"
                },
                ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.1))],
                repair_crews=1,
            ),
            "common cause, timed block replacement": RepairableRBD(
                EDGES,
                {
                    node: {
                        "reliability": E([0.002]),
                        "repairability": E([0.5]),
                    }
                    for node in "ab"
                }
                | {
                    "c": unit(
                        preventive={
                            "interval": 300.0,
                            "policy": "block",
                            "duration": E([2.0]),
                        }
                    )
                },
                ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.1))],
            ),
            "junction, bought": RepairableRBD(
                JUNCTION,
                {
                    "a": unit(acquisition_cost=100.0),
                    "b": unit(acquisition_cost=100.0),
                    "c": unit(repair_cost=2.0),
                    "j": PerfectReliability,
                },
                downtime_cost_rate=5.0,
            ),
            "fitted life": system(unit(reliability=fitted_life())),
            "standby group, two needed": system(
                {
                    "reliability": E([0.002]),
                    "repairability": E([0.5]),
                    "standby": {"units": 3, "k": 2},
                }
            ),
            "repair crews around a nested RBD": RepairableRBD(
                EDGES,
                {
                    "a": RepairableRBD(
                        [("s", "x"), ("s", "y"), ("x", "t"), ("y", "t")],
                        {"x": unit(), "y": unit()},
                    ),
                    "b": unit(),
                    "c": unit(),
                },
                repair_crews=1,
            ),
            "one repair crew, exponential": RepairableRBD(
                EDGES,
                {
                    node: {
                        "reliability": E([0.002]),
                        "repairability": E([0.5]),
                        "priority": priority,
                        "repair_cost": 3.0,
                    }
                    for node, priority in zip("abc", (1, 0, 0))
                },
                repair_crews=1,
                downtime_cost_rate=5.0,
                capacity={"a": 5.0, "b": 5.0, "c": 10.0},
            ),
        }
    )
    return out
