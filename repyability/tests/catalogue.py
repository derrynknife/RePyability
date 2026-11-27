"""The diagrams of every kind the analyses are checked on (#239).

One registry, for the routes test (every analysis does what
``analysis_routes()`` says), for the engines' agreement and for the
cross-checks of the exact and numerical routes against simulation
(``test_catalogue.py``). A new node kind, spec key or diagram option is
added here once, and ``test_catalogue.test_the_catalogue_has_every_kind``
fails until it is.
"""

import copy
import functools

import numpy as np
import surpyval as surv

from repyability import (
    MGL,
    AlphaFactor,
    BetaFactor,
    CCFGroup,
    DegradingNode,
    FaultTree,
    JointNode,
    LoadSharingModel,
    MarkovComponent,
    NonRepairable,
    NonRepairableRBD,
    Opportunities,
    Opportunity,
    PerfectReliability,
    PerfectUnreliability,
    RegressionNode,
    RepairableRBD,
    RepeatedNode,
    RepeatedStandbyNode,
    Shocks,
    StandbyModel,
)
from repyability.maintenance import AgePolicy, Prices, UnitPlan
from repyability.rbd import bdd, modular
from repyability.rbd._usage import has_at_usage
from repyability.schedule import DutySchedule
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


def _fitted_once(fit):
    """``fit``, run once: each call gets its own copy of the fitted model,
    as the diagrams are built many times a run and a test may change the
    model it is given."""
    once = functools.cache(fit)

    @functools.wraps(fit)
    def fitted():
        return copy.deepcopy(once())

    return fitted


@_fitted_once
def aft():
    """A pump's life at its load (surpyval's ExponentialAFT): 100 at load
    1, 25 at load 2."""
    x = np.array([50.0, 100.0, 150.0, 12.5, 25.0, 37.5])
    load = np.array([[1.0], [1.0], [1.0], [2.0], [2.0], [2.0]])
    return surv.ExponentialAFT.fit(x, Z=load)


@_fitted_once
def weibull_aft():
    """A pump's Weibull life at its load (surpyval's WeibullAFT)."""
    x = np.array([50.0, 100.0, 150.0, 12.5, 25.0, 37.5])
    load = np.array([[1.0], [1.0], [1.0], [2.0], [2.0], [2.0]])
    return surv.WeibullAFT.fit(x, Z=load)


@_fitted_once
def ph():
    """A pump's life at its load (surpyval's ExponentialPH): 500 at load
    1, 125 at load 2."""
    x = np.array([250.0, 500.0, 750.0, 62.5, 125.0, 187.5])
    load = np.array([[1.0], [1.0], [1.0], [2.0], [2.0], [2.0]])
    return surv.ExponentialPH.fit(x, Z=load)


def load_steps():
    """Load 1 for the first 200, and 2 after (a covariate schedule on the
    diagram's clock)."""
    return surv.StepSchedule.from_changepoints([0.0, 200.0], [[1.0], [2.0]])


@_fitted_once
def joint_model():
    """A Weibull joint model (surpyval's WeibullJointPH) of a fleet of 60
    pumps, each with a marker read every 10 hours and a load: a pump's
    marker starts near 2 and rises by about 0.02 an hour, and a higher
    marker or load shortens its life."""
    from surpyval import ItemLifetime, SurpyvalData, WeibullJointPH

    rng = np.random.default_rng(3)
    items = []
    for k in range(60):
        start, rate = rng.normal(2.0, 0.4), rng.normal(0.02, 0.005)
        load = float(k % 2)
        life = 100 * rng.weibull(2.0) * np.exp(-0.9 * (start - 2) - 0.3 * load)
        end = min(life, 120.0)
        ages = np.arange(0.0, end, 10.0)
        y = start + rate * ages + rng.normal(0, 0.2, ages.size)
        outcome = "observed" if life < 120 else "right_censored"
        items.append(
            ItemLifetime(
                k,
                **{outcome: end},
                markers=dict(zip(ages, y)),
                covariates={"load": load},
            )
        )
    return WeibullJointPH.fit(SurpyvalData.from_items(items))


#: A pump of the joint model's fleet, 40 hours old, at load 1, whose
#: marker has risen faster than most.
PUMP = {"Z": {"load": 1.0}, "markers": {0: 2.1, 20: 2.5, 40: 3.0}, "age": 40}


def joint_unit():
    """The pump ``PUMP`` of ``joint_model``, as a JointNode."""
    return JointNode(joint_model(), **PUMP)


@_fitted_once
def fitted_life():
    """A Weibull fitted to 30 failures: a fit with a parameter covariance."""
    return surv.Weibull.fit(np.random.default_rng(7).weibull(2, 30) * 100)


@_fitted_once
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


def wear():
    """A gamma degradation process fitted to the wear of 12 units, each
    measured 10 times; it fails at a wear of 8 (#271)."""
    g = np.random.default_rng(7)
    hours = np.tile(np.arange(1.0, 11.0) * 30.0, 12)
    units = np.repeat(np.arange(12), 10)
    worn = np.concatenate(
        [np.cumsum(g.gamma(2.0, 0.25, 10)) for _ in range(12)]
    )
    return surv.GammaProcess.fit(hours, worn, units, threshold=8.0)


def crow_amsaa():
    """A Crow-AMSAA process fitted to 60 failures of a repairable unit
    (#269): minimal repair of a Weibull life."""
    gaps = np.random.default_rng(7).weibull(1.5, 60) * 300
    return surv.CrowAMSAA.fit(np.cumsum(gaps))


def ara_process():
    """An ARA process (arithmetic reduction of age, memory 1) with a
    Weibull life and half its age taken away at each repair, which the
    simulation steps with surpyval (a repair surpyval fits that is not
    Kijima's)."""
    return surv.ARA.from_params([300.0, 1.5], 0.5)


@_fitted_once
def bayes_life():
    """A Bayesian Weibull fit, its posterior 400 draws about alpha 500
    and beta 1.5 (from another sampler, say): its life is the posterior
    predictive (``PosteriorPredictive``)."""
    g = np.random.default_rng(7)
    draws = np.c_[
        500 * g.lognormal(0, 0.1, 400), 1.5 * g.lognormal(0, 0.1, 400)
    ]
    return surv.bayes.BayesianParametric.from_draws(surv.Weibull, draws)


def usage_life():
    """A part that wears by hours and by starts (one start worth 10
    hours), at 0.1 starts per hour: surpyval's life at a usage rate
    (surpyval 0.28 and later; None before)."""
    if not has_at_usage():
        return None
    model = surv.WeibullEquivalence.from_params(
        [2000.0, 2.0, 10.0], clocks=("hours", "starts")
    )
    return model.at_usage(0.1)


def _if_usage(name: str, build) -> dict:
    """``{name: build()}``, or nothing with a surpyval that has no life at
    a usage rate."""
    return {name: build()} if has_at_usage() else {}


@_fitted_once
def bayes_shares_life():
    """A Bayesian Weibull fit with an offset and a limited failure
    population: 400 draws about an offset of 20, alpha 500, beta 1.5 and
    a share that ever fails of 0.8, each draw its own model."""
    g = np.random.default_rng(7)
    draws = np.c_[
        g.uniform(10.0, 30.0, 400),
        500 * g.lognormal(0, 0.1, 400),
        1.5 * g.lognormal(0, 0.1, 400),
        g.uniform(0.7, 0.9, 400),
    ]
    return surv.bayes.BayesianParametric.from_draws(
        surv.Weibull, draws, ["gamma", "alpha", "beta", "lfp_p"]
    )


@_fitted_once
def royston_parmar_life():
    """A Royston-Parmar fit (a spline on the log cumulative hazard) to 40
    lives about a Weibull of alpha 500 and beta 1.5: a surpyval model
    taken as a life by its functions (#93)."""
    x = 500 * np.random.default_rng(7).weibull(1.5, 40)
    return surv.RoystonParmar.fit(x)


@_fitted_once
def life_at_load():
    """The pump of ``aft()`` at a load of 1.5: a regression at one unit's
    covariates (``regression.at(Z)``), taken as a life by its functions
    (#93)."""
    return aft().at([[1.5]])


@_fitted_once
def mixture_life():
    """A two-mode population, infant mortality and wear-out (#227)."""
    g = np.random.default_rng(7)
    mix = surv.MixtureModel(surv.Weibull, 2)
    mix.fit(np.r_[20 * g.weibull(1, 15), 150 * g.weibull(3, 45)])
    return mix


def markov_unit(capacity=None, start=None):
    """A Markov component: a unit that degrades, is restored from its
    degraded state or fails from it, and is repaired from failure."""
    return MarkovComponent(
        ["working", "degraded", "failed"],
        {
            ("working", "degraded"): 0.01,
            ("degraded", "working"): 0.2,
            ("degraded", "failed"): 0.05,
            ("failed", "working"): 0.5,
        },
        up=None if capacity else ["working", "degraded"],
        capacity=capacity,
        start=start,
    )


def markov_life():
    """A Markov component with no way back from its failed state: a
    life."""
    return MarkovComponent(
        ["working", "degraded", "failed"],
        {
            ("working", "degraded"): 0.01,
            ("degraded", "working"): 0.02,
            ("degraded", "failed"): 0.05,
        },
        up=["working", "degraded"],
    )


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
        # A life given as a Markov chain: it degrades, is restored from
        # its degraded state or fails from it, and stays failed.
        "Markov life": NonRepairableRBD(EDGES, {"a": markov_life(), **rest}),
        "simulated standby": NonRepairableRBD(
            EDGES,
            {
                "a": StandbyModel([unit] * 3, k=2, dormancy_factor=0.5),
                **rest,
            },
        ),
        "exact warm standby": NonRepairableRBD(
            EDGES,
            {
                # Warm, two operating, a spare that may fail to start:
                # exact for exponential units (#339).
                "a": StandbyModel(
                    [E([0.01])] * 3,
                    k=2,
                    dormancy_factor=0.5,
                    start_probability=0.9,
                ),
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
        "shocks": NonRepairableRBD(
            EDGES,
            {"a": unit, "b": surv.Weibull.from_params([400.0, 1.5]), **rest},
            ccf_groups=[CCFGroup(["a", "b"], Shocks(1e-3, 0.5, 2e-4))],
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
        "nested, worked out": NonRepairableRBD(
            EDGES,
            {
                "a": NonRepairableRBD([("s", "x"), ("x", "t")], {"x": unit}),
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
        "common cause, alpha factor": NonRepairableRBD(
            VOTE,
            {
                "a": unit,
                "b": unit,
                "d": unit,
                "c": E([0.002]),
                "j": PerfectReliability,
            },
            k={"j": 2},
            ccf_groups=[
                CCFGroup(
                    ["a", "b", "d"],
                    AlphaFactor([0.95, 0.04, 0.01], testing="non-staggered"),
                )
            ],
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
        "joint": NonRepairableRBD(EDGES, {"a": joint_unit(), **rest}),
        "fitted": NonRepairableRBD(EDGES, {"a": fitted_life(), **rest}),
        "fitted shares": NonRepairableRBD(EDGES, {"a": shares_life(), **rest}),
        "degradation process": NonRepairableRBD(EDGES, {"a": wear(), **rest}),
        "mixture": NonRepairableRBD(EDGES, {"a": mixture_life(), **rest}),
        "Bayesian fit": NonRepairableRBD(EDGES, {"a": bayes_life(), **rest}),
        "Bayesian fit with shares": NonRepairableRBD(
            EDGES, {"a": bayes_shares_life(), **rest}
        ),
        "Royston-Parmar fit": NonRepairableRBD(
            EDGES, {"a": royston_parmar_life(), **rest}
        ),
        **_if_usage(
            "life at a usage rate",
            lambda: NonRepairableRBD(EDGES, {"a": usage_life(), **rest}),
        ),
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


def mission_kinds():
    """The phased missions drawn as one diagram whose analyses are checked
    on: with blocks guaranteed, not needed, at a phase's covariates and a
    phase's vote count (exact), on a phase's own model (exact, and with a
    block on standby), with a block on standby (simulated), and with
    blocks on standby, warm and switching imperfectly, every life of a
    constant rate (exact, by the Markov chain), and with a common-cause
    group that splits its members' failure rate, and one of shocks at a
    rate of their own (exact, each cause a component), and with blocks
    repaired during the phases (exact by the chain where every life and
    repair is exponential, else simulated)."""
    from repyability import BlockSetting, MissionPhase, PhasedMission

    unit = W([100, 2])
    edges = [
        ("s", "a"),
        ("s", "b"),
        ("s", "c"),
        ("a", "v"),
        ("b", "v"),
        ("c", "v"),
        ("v", "d"),
        ("d", "t"),
    ]
    models = {
        "a": unit,
        "b": unit,
        "c": E([0.01]),
        "v": PerfectReliability,
        "d": RegressionNode(aft(), covariates=[1.0]),
    }
    vote = NonRepairableRBD(edges, models, k={"v": 2})
    settings = [
        MissionPhase(
            "up",
            20.0,
            blocks={"a": "guaranteed", "d": BlockSetting(covariates=[2.0])},
            k={"v": 3},
        ),
        MissionPhase("on", 30.0, blocks={"c": "not_needed"}),
    ]
    standby = [
        MissionPhase("up", 20.0, blocks={"c": "standby"}),
        MissionPhase("on", 30.0, k={"v": 1}),
    ]
    # A block on a model of the phase's own, exact and with standby.
    start_up = BlockSetting(model=W([40, 0.8]))
    models = [
        MissionPhase("up", 20.0, blocks={"b": start_up}),
        MissionPhase("on", 30.0, blocks={"c": BlockSetting(model=E([0.03]))}),
    ]
    standby_models = [
        MissionPhase("up", 20.0, blocks={"c": "standby", "b": start_up}),
        MissionPhase("on", 30.0, k={"v": 1}),
    ]
    constant = NonRepairableRBD(
        edges,
        {
            "a": E([0.01]),
            "b": E([0.02]),
            "c": E([0.01]),
            "v": PerfectReliability,
            "d": RegressionNode(aft(), covariates=[1.0]),
        },
        k={"v": 2},
    )
    warm = BlockSetting(
        state="standby", dormancy_factor=0.2, start_probability=0.9
    )
    chain = [
        MissionPhase(
            "up",
            20.0,
            blocks={"c": warm, "a": "guaranteed"},
            k={"v": 2},
        ),
        MissionPhase(
            "on",
            30.0,
            blocks={"b": "standby", "d": BlockSetting(covariates=[2.0])},
            k={"v": 1},
        ),
    ]
    plain = [
        MissionPhase("up", 20.0, k={"v": 3}),
        MissionPhase("on", 30.0, blocks={"c": "not_needed"}),
    ]
    repairs = {"a": E([0.5]), "c": E([0.2])}
    return {
        "repair_exact": PhasedMission.from_diagram(
            constant, plain, repairability=repairs
        ),
        "repair": PhasedMission.from_diagram(
            vote, plain, repairability={"a": W([3.0, 1.5])}
        ),
        "common_cause": PhasedMission.from_diagram(
            vote,
            plain,
            ccf_groups=CCFGroup(["a", "b"], BetaFactor(0.1, basis="rate")),
        ),
        "shocks": PhasedMission.from_diagram(
            vote, plain, ccf_groups=CCFGroup(["a", "c"], Shocks(0.002, 0.5))
        ),
        "standby_exact": PhasedMission.from_diagram(constant, chain),
        "settings": PhasedMission.from_diagram(vote, settings),
        "standby": PhasedMission.from_diagram(vote, standby),
        "models": PhasedMission.from_diagram(vote, models),
        "standby_models": PhasedMission.from_diagram(vote, standby_models),
    }


def fault_tree_kinds():
    """The fault trees of every kind the tree's analyses are checked on
    (#314): events given as probabilities, as lifetime models (surpyval
    distributions, a fit with its covariance) and as both, a repeated
    event, common-cause groups of each model and basis, a NOT gate, a
    functional dependency, dynamic gates integrated or shared, and events
    repaired or tested (#809), repeated or under a NOT gate, beside fixed
    probabilities and lives that are not repaired, and in common-cause
    groups (#831)."""
    unit = W([100, 2])
    gates = {"top": ("or", ["pair", "c"]), "pair": ("and", ["a", "b"])}
    rest = {"b": W([80, 1.5]), "c": E([0.002])}
    alike = {"a": unit, "b": unit, "c": E([0.002])}
    repaired = {
        "a": {"reliability": unit, "repairability": E([0.5])},
        "b": {"reliability": E([0.01]), "repairability": L([1.0, 0.5])},
        "c": {"reliability": E([0.002]), "repairability": E([0.1])},
    }
    return {
        "fixed": FaultTree(gates, {"a": 0.1, "b": 0.2, "c": 0.05}),
        "lifetimes": FaultTree(gates, {"a": unit, **rest}),
        "mixed": FaultTree(gates, {"a": unit, "b": 0.2, "c": L([4.0, 0.5])}),
        "fitted": FaultTree(gates, {"a": fitted_life(), **rest}),
        "repeated": FaultTree(
            {
                "top": ("and", ["A", "B"]),
                "A": ("or", ["s", "a"]),
                "B": ("or", ["s", "b"]),
            },
            {"s": E([0.001]), "a": unit, "b": W([80, 1.5])},
        ),
        "common cause": FaultTree(
            gates, alike, ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.1))]
        ),
        "common cause by rate": FaultTree(
            gates,
            alike,
            ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.1, basis="rate"))],
        ),
        "common cause, MGL": FaultTree(
            {
                "top": ("or", ["vote", "d"]),
                "vote": ("vote", 2, ["a", "b", "c"]),
            },
            {"a": unit, "b": unit, "c": unit, "d": E([0.002])},
            ccf_groups=[
                CCFGroup(["a", "b", "c"], MGL(0.1, 0.3, basis="rate"))
            ],
        ),
        "common cause, fixed": FaultTree(
            gates,
            {"a": 0.05, "b": 0.05, "c": 0.01},
            ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.1))],
        ),
        # A NOT gate (non-coherent): exact, its prime implicants for cut
        # sets, Birnbaum's the one importance measure.
        "not gate": FaultTree(
            {
                "top": ("or", ["pair", "c and not a"]),
                "pair": ("and", ["a", "b"]),
                "c and not a": ("and", ["c", "not a"]),
                "not a": ("not", ["a"]),
            },
            {"a": unit, **rest},
        ),
        # A functional dependency: the static tree it is written out as.
        "functional dependency": FaultTree(
            {**gates, "power": ("fdep", "p", ["a", "b"])},
            {"a": unit, **rest, "p": E([0.0005])},
        ),
        # Dynamic gates, each an independent module: integrated.
        "dynamic": FaultTree(
            {
                "top": ("or", ["order", "spare"]),
                "order": ("pand", ["a", "b"]),
                "spare": ("spare", 0.2, ["c", "d"]),
            },
            {"a": unit, **rest, "d": W([150, 1.2])},
        ),
        # A dynamic gate sharing an event with the rest: simulated on
        # request (method="simulate"), refused exactly.
        "dynamic, shared": FaultTree(
            {
                "top": ("or", ["order", "pair"]),
                "order": ("seq", ["a", "b"]),
                "pair": ("and", ["b", "c"]),
            },
            {"a": unit, **rest},
        ),
        # Repaired events (#809's kind): the values over time.
        "repaired": FaultTree(gates, repaired),
        # Hidden failures found by periodic tests, one test taking time.
        "tested": FaultTree(
            gates,
            {
                **repaired,
                "a": {
                    "reliability": E([0.004]),
                    "repairability": E([0.5]),
                    "inspection": {"interval": 50.0, "duration": 1.0},
                },
                "b": {
                    "reliability": E([0.002]),
                    "repairability": "instant",
                    "inspection": {"interval": 80.0},
                },
            },
        ),
        # A repaired event feeding two gates: exact from the tree, and no
        # RepairableRBD (to_repairable_rbd refuses).
        "repeated, repaired": FaultTree(
            {
                "top": ("and", ["A", "B"]),
                "A": ("or", ["s", "a"]),
                "B": ("or", ["s", "b"]),
            },
            {"s": repaired["c"], "a": repaired["a"], "b": repaired["b"]},
        ),
        # A NOT gate over repaired events: Q(t) exact, w(t) refused.
        "repaired, not gate": FaultTree(
            {
                "top": ("or", ["pair", "c and not a"]),
                "pair": ("and", ["a", "b"]),
                "c and not a": ("and", ["c", "not a"]),
                "not a": ("not", ["a"]),
            },
            repaired,
        ),
        # Repaired events beside a fixed probability (#831): a long run.
        "repaired, with a probability": FaultTree(
            gates, {"a": repaired["a"], "b": repaired["b"], "c": 0.002}
        ),
        # Beside a life that is not repaired too: no long run.
        "repaired, with a probability and a life": FaultTree(
            gates, {"a": repaired["a"], "b": W([400, 1.5]), "c": 0.002}
        ),
        # Common cause over repaired events: their groups' chains.
        "repaired, common cause": FaultTree(
            gates,
            {"a": repaired["c"], "b": repaired["c"], "c": repaired["b"]},
            ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.1, basis="rate"))],
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


def calendar_policy(life):
    """A calendar policy for ``life`` (#319): shuts every 60 at 2, and a
    failure at 10."""
    shuts = Opportunities([Opportunity("shut", period=60.0)], any_day=False)
    part = NonRepairable(life, unplanned_cost=10.0, opportunity_cost=2.0)
    return part.optimal_opportunity_policy(shuts)


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
            # One unit's maintenance plan as a node: the node its spec
            # gives (UC-63 of the maintenance README).
            "a maintenance plan": system(
                UnitPlan(
                    life,
                    repair,
                    tasks=(AgePolicy(interval=300.0, cost=4.0),),
                    prices=Prices(repair=2.0, downtime=1.0),
                )
            ),
            # A life that is a diagram: a subassembly (a series pair)
            # replaced whole at its failure, taken by its functions.
            "a life that is a diagram": system(
                unit(
                    reliability=NonRepairableRBD(
                        [("s", "x"), ("x", "y"), ("y", "t")],
                        {"x": W([400.0, 2.0]), "y": W([400.0, 2.0])},
                    )
                )
            ),
            # A plan valued alone, with its own money: its salvage, its
            # prices' growth and its retirement (UC-65 of the maintenance
            # README), the one node of its diagram.
            "a plan valued alone": UnitPlan(
                life,
                repair,
                tasks=(AgePolicy(interval=300.0, cost=4.0),),
                prices=Prices(repair=2.0, downtime=1.0),
                salvage=3.0,
                escalation=1e-3,
                retire_at=250.0,
            ).diagram(),
            # Replaced at the times of a plan, and after the last no more
            # (block replacement at any list of times).
            "replaced at given times": system(
                unit(
                    preventive={
                        "replace_at": [150.0, 400.0, 700.0],
                        "duration": W([4, 2]),
                        "cost": 20.0,
                    },
                    repair_cost=5.0,
                ),
            ),
            # Upgraded at a time: its units have another life from then,
            # at a capital cost (worked out over a horizon, and simulated).
            "upgraded at a time": system(
                unit(
                    upgrade={
                        "at": 250.0,
                        "reliability": W([900, 2.0]),
                        "cost": 50.0,
                    },
                    repair_cost=5.0,
                ),
            ),
            # A shelf of spares (the spec's "spares"): a repair that
            # finds it empty waits for a spare, which only the
            # simulations take, the exact methods refusing the node.
            "a shelf of spares": system(
                unit(
                    spares={"stock": 1, "lead_time": 200.0, "repairs": False},
                    repair_cost=5.0,
                )
            ),
            # A sub-system's component on a shelf of its own, in the
            # sub-system's loop: the system's exact methods refuse it
            # through the sub-system's.
            "a sub-system's shelf of spares": system(
                RepairableRBD(
                    [("s", "x"), ("x", "t")],
                    {"x": unit(spares={"stock": 0, "lead_time": 100.0})},
                )
            ),
            # Upgraded, and block-replaced before and after (the
            # upgrade as one more replacement): over a horizon.
            "upgraded at a time, block-replaced": system(
                unit(
                    preventive={
                        "policy": "block",
                        "interval": 200.0,
                        "cost": 3.0,
                        "duration": surv.ExactEventTime.from_params([2.0]),
                    },
                    upgrade={
                        "at": 300.0,
                        "reliability": W([900, 2.0]),
                        "cost": 50.0,
                    },
                    repair_cost=5.0,
                ),
            ),
            # Upgraded and age-replaced: the unit up at the upgrade
            # replaced as at its age, over a horizon (UC-72).
            "upgraded at a time, age-replaced": system(
                unit(
                    preventive={
                        "interval": 200.0,
                        "cost": 3.0,
                        "duration": surv.ExactEventTime.from_params([2.0]),
                    },
                    upgrade={
                        "at": 300.0,
                        "reliability": W([900, 2.0]),
                        "cost": 50.0,
                    },
                    repair_cost=5.0,
                ),
            ),
            # Upgraded with tests: the unit in service replaced at the
            # upgrade, and the new one tested on the same calendar.
            "upgraded at a time, tested": system(
                unit(
                    repairability="instant",
                    inspection={"interval": 100.0, "cost": 1.0},
                    upgrade={"at": 250.0, "reliability": W([900, 2.0])},
                    repair_cost=5.0,
                ),
            ),
            # Upgraded, the unit in service kept until it fails or is
            # due: each unit with the life in service when it is put in,
            # over a horizon.
            "upgraded at a time, the unit in service kept": system(
                unit(
                    preventive={"interval": 200.0, "cost": 3.0},
                    upgrade={
                        "at": 250.0,
                        "reliability": W([900, 2.0]),
                        "replace_in_service": False,
                    },
                    repair_cost=5.0,
                ),
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
            "replaced on condition by level": system(
                unit(
                    reliability=wear(),
                    preventive={
                        "interval": 60.0,
                        "policy": "condition",
                        "level": 6.0,
                        "inspection_cost": 1.0,
                    },
                    replace_cost=10.0,
                )
            ),
            "replaced on a calendar policy": system(
                unit(
                    preventive={
                        "policy": "calendar",
                        "calendar": calendar_policy(life),
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
            # A calendar component in a maintenance group (#349): renewed
            # early at the age-replaced member's stops, its own
            # replacements stops that renew that member early.
            "calendar policy in a maintenance group": RepairableRBD(
                EDGES,
                {
                    "a": unit(
                        preventive={
                            "policy": "calendar",
                            "calendar": calendar_policy(life),
                            "opportunity": 100.0,
                        },
                        group="train",
                        replace_cost=10.0,
                    ),
                    "b": unit(
                        preventive={"interval": 300.0, "opportunity": 200.0},
                        group="train",
                        replace_cost=10.0,
                    ),
                    "c": unit(),
                },
                maintenance_groups={"train": {"setup_cost": 50.0}},
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
            "minimal repair, repairs that take time": system(
                unit(repair={"model": "kijima1", "q": 1.0}, repair_cost=1.0)
            ),
            "imperfect repair, replaced and maintained": system(
                unit(
                    repair={"model": "kijima2", "q": 0.8},
                    replace_after=3,
                    preventive={"interval": 300.0},
                    replace_cost=10.0,
                )
            ),
            # Preventive maintenance that restores a unit only partly
            # (Kijima): by age, on the calendar with every third task
            # replacing it, on condition, with imperfect repair too, and
            # on a duty.
            "maintenance restores partly, by age": system(
                unit(
                    preventive={
                        "interval": 300.0,
                        "cost": 2.0,
                        "restoration": {"model": "kijima1", "q": 0.3},
                    },
                    repair_cost=1.0,
                    replace_cost=10.0,
                )
            ),
            "maintenance restores partly, every third replaces": system(
                unit(
                    preventive={
                        "interval": 300.0,
                        "policy": "block",
                        "duration": E([0.5]),
                        "cost": 2.0,
                        "restoration": {"model": "kijima2", "q": 0.5},
                        "replace_every": 3,
                        "replacement_cost": 8.0,
                    },
                    repair_cost=1.0,
                )
            ),
            "maintenance restores partly, on condition": system(
                unit(
                    preventive={
                        "interval": 100.0,
                        "policy": "condition",
                        "threshold": 0.1,
                        "inspection_cost": 1.0,
                        "restoration": {"model": "kijima1", "q": 0.5},
                    },
                    replace_cost=10.0,
                )
            ),
            "imperfect repair, maintenance restores partly": system(
                unit(
                    repair={"model": "kijima1", "q": 0.5},
                    preventive={
                        "interval": 300.0,
                        "restoration": {"model": "kijima1", "q": 0.0},
                    },
                    repair_cost=1.0,
                )
            ),
            "operated part of the time, maintenance restores partly": system(
                unit(
                    duty=0.25,
                    preventive={
                        "interval": 300.0,
                        "restoration": {"model": "kijima2", "q": 0.4},
                    },
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
            # A probability of failing per demand with the demands' rate
            # (a pump that may fail to start): an exponential life of
            # rate d p, which every route takes.
            "demand": system(
                unit(reliability=FIXED(0.1), demand_rate=0.05, repair_cost=5.0)
            ),
            # A life that wears out with a demand mode beside it (a pump
            # that wears and may fail to start): competing risks, renewed
            # together, a life of the summed hazard, which every route
            # takes as it takes any life.
            "life with a demand mode": system(
                unit(
                    demand={"probability": 0.1, "rate": 0.05},
                    repair_cost=5.0,
                )
            ),
            # A life that wears out with demands at set times, or on a
            # plan of rates that changes, beside it: competing risks
            # renewed together, on the diagram's clock, so simulated.
            "life with demands at set times": system(
                unit(
                    demand={
                        "probability": 0.1,
                        "times": [50.0, 120.0, 200.0, 260.0],
                    },
                    repair_cost=5.0,
                )
            ),
            "life with a demand plan": system(
                unit(
                    demand={
                        "probability": 0.1,
                        "plan": {"times": [0.0, 150.0], "rates": [0.1, 0.02]},
                    },
                    repair_cost=5.0,
                )
            ),
            # Demands at set times (UC-73): a life that ends only at them,
            # on the calendar, exact over a window and simulated.
            "demand times": system(
                unit(
                    reliability=FIXED(0.1),
                    demand_times=[50.0, 120.0, 200.0, 260.0],
                    repair_cost=5.0,
                )
            ),
            # Demands at a rate that changes, counted in the unit's
            # operating time on a duty: an exponential life in demands
            # on the composed plan, simulated (its repairs take time).
            "demand plan": system(
                unit(
                    reliability=FIXED(0.1),
                    demand_plan={"times": [0.0, 150.0], "rates": [0.1, 0.02]},
                    duty=0.5,
                    demand_basis="operating",
                    repair_cost=5.0,
                )
            ),
            "mixture life": system(
                unit(reliability=mixture_life(), replace_cost=10.0)
            ),
            "Bayesian fit life": system(
                unit(reliability=bayes_life(), replace_cost=10.0)
            ),
            "Bayesian fit with shares life": system(
                unit(reliability=bayes_shares_life(), replace_cost=10.0)
            ),
            "Royston-Parmar life": system(
                unit(reliability=royston_parmar_life(), replace_cost=10.0)
            ),
            "life at a regression's covariates": system(
                unit(reliability=life_at_load(), replace_cost=10.0)
            ),
            **_if_usage(
                "life at a usage rate",
                lambda: system(
                    unit(reliability=usage_life(), replace_cost=10.0)
                ),
            ),
            "fitted Crow-AMSAA process": system(
                unit(reliability=crow_amsaa(), repair_cost=1.0)
            ),
            # Minimal repair, replaced (#402): a renewal cycle, exact for
            # a power law, by quadrature for Cox-Lewis's intensity.
            "fitted Crow-AMSAA process, replaced at its third failure": (
                system(
                    unit(
                        reliability=crow_amsaa(),
                        replace_after=3,
                        repair_cost=1.0,
                        replace_cost=10.0,
                    )
                )
            ),
            "Cox-Lewis process, replaced at its second failure or an age": (
                system(
                    unit(
                        reliability=surv.CoxLewis.from_params(
                            [np.log(0.001), 0.005]
                        ),
                        replace_after=2,
                        preventive={
                            "interval": 400.0,
                            "duration": E([0.5]),
                            "cost": 5.0,
                        },
                        repair_cost=1.0,
                        replace_cost=10.0,
                    )
                )
            ),
            "fitted ARA process": system(
                unit(reliability=ara_process(), repair_cost=1.0)
            ),
            "degradation process life": system(unit(reliability=wear())),
            "covariate schedule, proportional hazards, maintained": system(
                unit(
                    reliability=ph(),
                    covariates=load_steps(),
                    preventive={"interval": 300.0},
                    replace_cost=10.0,
                )
            ),
            "covariate schedule, accelerated failure time": system(
                unit(
                    reliability=aft(),
                    covariates=surv.StepSchedule.cyclic(
                        [0.0, 60.0], [[1.0], [2.0]], 100.0
                    ),
                )
            ),
            "operated part of the time": system(
                unit(duty=0.25, preventive={"interval": 300.0})
            ),
            # Tasks counted in operating time, put on the calendar
            # through the block's duty.
            "operated part of the time, maintained by running hours": system(
                unit(
                    duty=0.25,
                    preventive={
                        "interval": 75.0,
                        "basis": "operating",
                        "cost": 2.0,
                    },
                    replace_cost=10.0,
                )
            ),
            # A duty plan that changes over time: lives and tasks in
            # operating time put on the calendar through it (simulated).
            "on a duty plan that changes, maintained by running hours": system(
                unit(
                    duty=DutySchedule(
                        "hours", (0.0, 40.0, 100.0), (1.0, 0.0, 0.5)
                    ),
                    preventive={
                        "interval": 75.0,
                        "basis": "operating",
                        "duration": E([2.0]),
                        "cost": 2.0,
                    },
                    replace_cost=10.0,
                )
            ),
            # Repaired and maintained in no time on such a plan: its
            # operating-time process seen through the plan (exact over
            # time).
            "on a duty plan that changes, renewed in no time": system(
                unit(
                    duty=DutySchedule(
                        "hours", (0.0, 40.0, 100.0), (1.0, 0.0, 0.5)
                    ),
                    repairability="instant",
                    preventive={
                        "interval": 75.0,
                        "basis": "operating",
                        "cost": 2.0,
                    },
                    replace_cost=10.0,
                )
            ),
            "operated part of the time, tested by running hours": system(
                unit(
                    duty=0.5,
                    repairability="instant",
                    inspection={
                        "interval": 50.0,
                        "offset": 10.0,
                        "basis": "operating",
                    },
                )
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
            # A component given as a Markov chain: working, degraded
            # (restored, or failing) and failed (repaired), priced.
            "Markov component": system(
                {
                    "reliability": markov_unit(),
                    "repair_cost": 5.0,
                    "downtime_cost": 2.0,
                },
                downtime_cost_rate=3.0,
            ),
            # One with a capacity in each state, started in its
            # degraded state or its failed one.
            "Markov component, capacities": system(
                markov_unit(
                    capacity={"working": 100.0, "degraded": 60.0, "failed": 0},
                    start={"degraded": 0.5, "failed": 0.5},
                ),
                capacity={"b": 100.0, "c": 150.0},
            ),
            # A life given as a Markov chain, repaired at a model's
            # times: a plain unit of that life.
            "Markov life, repaired": system(
                unit(reliability=markov_life(), repair_cost=2.0)
            ),
            "one repair crew": system(
                unit(priority=1), repair_crews=1, downtime_cost_rate=5.0
            ),
            "enough repair crews": system(unit(), repair_crews=3),
            "paid repair crews": system(
                unit(priority=1, repair_cost=2.0),
                repair_crews=1,
                crew_cost_rate=0.5,
            ),
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
            "cold standby pair, Weibull, one crew": RepairableRBD(
                [("s", "a"), ("a", "t")],
                {
                    "a": {
                        "reliability": W([1000.0, 2.5]),
                        "repairability": W([300.0, 1.5]),
                        "standby": {"units": 2, "k": 1},
                        "repair_cost": 3.0,
                    }
                },
                repair_crews=1,
                downtime_cost_rate=5.0,
            ),
            "cold standby pair, Weibull, a crew each": RepairableRBD(
                [("s", "a"), ("a", "t")],
                {
                    "a": {
                        "reliability": W([1000.0, 2.5]),
                        "repairability": W([300.0, 1.5]),
                        "standby": {"units": 2, "k": 1},
                        "repair_cost": 3.0,
                    }
                },
                downtime_cost_rate=5.0,
            ),
            "cold standby pair, Weibull, failing switch": RepairableRBD(
                [("s", "a"), ("a", "t")],
                {
                    "a": {
                        "reliability": W([1000.0, 2.5]),
                        "repairability": W([300.0, 1.5]),
                        "standby": {
                            "units": 2,
                            "k": 1,
                            "switching_probability": 0.9,
                        },
                        "repair_cost": 3.0,
                    }
                },
                repair_crews=1,
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
                    for node in "abc"
                },
                ccf_groups=[CCFGroup(["a", "b"], MGL(0.2))],
            ),
            "common cause, alpha factor": RepairableRBD(
                VOTE,
                {
                    node: {
                        "reliability": E([0.002]),
                        "repairability": E([0.5]),
                    }
                    for node in "abdc"
                }
                | {"j": PerfectReliability},
                k={"j": 2},
                ccf_groups=[
                    CCFGroup(["a", "b", "d"], AlphaFactor([0.95, 0.04, 0.01]))
                ],
            ),
            "shocks, revealed": RepairableRBD(
                EDGES,
                {
                    "a": {
                        "reliability": W([400.0, 1.5]),
                        "repairability": E([0.5]),
                    },
                    "b": {
                        "reliability": W([300.0, 2.0]),
                        "repairability": W([3.0, 2.0]),
                    },
                    "c": {
                        "reliability": E([0.002]),
                        "repairability": E([0.5]),
                    },
                },
                ccf_groups=[CCFGroup(["a", "b"], Shocks(1e-3, 0.5, 2e-4))],
            ),
            "shocks, tested": RepairableRBD(
                EDGES,
                {
                    **{
                        node: {
                            "reliability": W([600.0, 1.5]),
                            "repairability": "instant",
                            "inspection": {
                                "interval": 100.0,
                                "coverage": 0.8,
                                "full_test": 300.0,
                            },
                        }
                        for node in "ab"
                    },
                    "c": {
                        "reliability": E([0.002]),
                        "repairability": E([0.5]),
                    },
                },
                ccf_groups=[CCFGroup(["a", "b"], Shocks(1e-3, 0.5))],
            ),
            # Exponential members, of lives and repairs of their own: the
            # shocks' causes at their own rates in the groups' chains.
            "shocks, exponential": RepairableRBD(
                EDGES,
                {
                    "a": {
                        "reliability": E([0.003]),
                        "repairability": E([0.5]),
                    },
                    "b": {
                        "reliability": E([0.001]),
                        "repairability": E([0.2]),
                    },
                    "c": {
                        "reliability": E([0.0005]),
                        "repairability": E([0.5]),
                    },
                },
                ccf_groups=[CCFGroup(["a", "b"], Shocks(2e-3, 0.6, 5e-4))],
            ),
            "shocks, exponential, tested": RepairableRBD(
                EDGES,
                {
                    **{
                        node: {
                            "reliability": E([rate]),
                            "repairability": "instant",
                            "inspection": {"interval": interval},
                        }
                        for node, rate, interval in (
                            ("a", 0.002, 100.0),
                            ("b", 0.0005, 150.0),
                        )
                    },
                    "c": {
                        "reliability": E([0.0005]),
                        "repairability": E([0.5]),
                    },
                },
                ccf_groups=[CCFGroup(["a", "b"], Shocks(1e-3, 0.5, 2e-4))],
            ),
            # Members maintained on a schedule, repaired imperfectly and in
            # a maintenance group, struck by shocks too (simulated).
            "shocks, maintained": RepairableRBD(
                EDGES,
                {
                    "a": {
                        "reliability": W([400.0, 2.0]),
                        "repairability": E([0.5]),
                        "preventive": {
                            "interval": 250.0,
                            "opportunity": 150.0,
                        },
                        "group": "train",
                    },
                    "b": {
                        "reliability": W([300.0, 2.0]),
                        "repairability": E([0.5]),
                        "repair": {"model": "kijima1", "q": 0.5},
                    },
                    "c": {
                        "reliability": W([500.0, 1.5]),
                        "repairability": E([0.5]),
                        "preventive": {
                            "interval": 300.0,
                            "opportunity": 200.0,
                        },
                        "group": "train",
                    },
                },
                maintenance_groups={"train": {"setup_cost": 5.0}},
                ccf_groups=[CCFGroup(["a", "b"], Shocks(1e-3, 0.5))],
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
            # A repair of any length (#106; a test of any length is in
            # test_ccf_general.py).
            "common cause, tests and repairs of any length": RepairableRBD(
                EDGES,
                {
                    node: {
                        "reliability": E([0.002]),
                        "repairability": W([8.0, 2.0]),
                        "inspection": {"interval": 100.0, "offset": offset},
                    }
                    for node, offset in (("a", 0.0), ("b", 50.0))
                }
                | {"c": unit(repair_cost=2.0)},
                ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.1))],
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
            # Shutdowns: the line shut every 300 for 5 (from 100), every
            # block's work done there.
            "shutdown of the line, tasks at it": RepairableRBD(
                EDGES,
                {
                    "a": unit(
                        preventive={"shutdown": "line", "cost": 2.0},
                        repair_cost=3.0,
                    ),
                    "b": unit(
                        preventive={"shutdown": "line", "older_than": 400.0}
                    ),
                    "c": {
                        "reliability": E([0.001]),
                        "repairability": E([0.5]),
                        "inspection": {"shutdown": "line", "cost": 1.0},
                    },
                },
                shutdowns={
                    "line": {
                        "period": 300.0,
                        "offset": 100.0,
                        "duration": 5.0,
                        "cost": 20.0,
                    }
                },
                downtime_cost_rate=5.0,
            ),
            "shutdown of some blocks, idle ones paused": RepairableRBD(
                EDGES,
                {
                    "a": unit(preventive={"shutdown": "unit"}),
                    "b": unit(repair_cost=2.0),
                    "c": unit(),
                },
                shutdowns={
                    "unit": {
                        "period": 200.0,
                        "duration": 10.0,
                        "scope": ["a", "b"],
                        "idle_ageing": False,
                    }
                },
            ),
            "shutdowns at given times, restoring": RepairableRBD(
                EDGES,
                {
                    "a": unit(
                        preventive={
                            "shutdown": "outage",
                            "restoration": {"model": "kijima1", "q": 0.5},
                            "cost": 2.0,
                        }
                    ),
                    "b": unit(),
                    "c": unit(),
                },
                shutdowns={
                    "outage": {
                        "times": [150.0, 400.0, 700.0],
                        "duration": 8.0,
                        "scope": ["a"],
                        "cost": 30.0,
                    }
                },
            ),
            # Outages at given times whose job leaves a block as if 50
            # old (simulated).
            "outages restoring to an age": RepairableRBD(
                EDGES,
                {
                    "a": unit(
                        preventive={
                            "shutdown": "outage",
                            "restore_to": 50.0,
                            "cost": 2.0,
                        }
                    ),
                    "b": unit(),
                    "c": unit(),
                },
                shutdowns={
                    "outage": {
                        "times": [150.0, 400.0, 700.0],
                        "duration": 8.0,
                        "scope": ["a"],
                    }
                },
            ),
            # Outages at given times, each renewing a block: exact over
            # a horizon (no long run).
            "outages at given times, renewing": RepairableRBD(
                EDGES,
                {
                    "a": unit(
                        preventive={"shutdown": "outage", "cost": 2.0},
                        repair_cost=5.0,
                    ),
                    "b": unit(),
                    "c": unit(),
                },
                shutdowns={
                    "outage": {
                        "times": [150.0, 400.0, 700.0],
                        "duration": 8.0,
                        "scope": ["a", "c"],
                        "cost": 30.0,
                    }
                },
            ),
            # A block renewed at two shutdowns, a short stop on a period
            # and longer outages at given times, each renewal taking its
            # own shut's length: exact over a horizon (no long run).
            "a block renewed at several shutdowns": RepairableRBD(
                EDGES,
                {
                    "a": unit(
                        preventive={
                            "shutdown": ["stop", "outage"],
                            "cost": 2.0,
                        },
                        repair_cost=5.0,
                    ),
                    "b": unit(),
                    "c": unit(),
                },
                shutdowns={
                    "stop": {"period": 100.0, "duration": 1.0, "scope": ["a"]},
                    "outage": {
                        "times": [250.0, 640.0],
                        "duration": 12.0,
                        "scope": ["a", "c"],
                        "cost": 30.0,
                    },
                },
            ),
            # Outages at given times that renew a block under age
            # replacement if it is old enough then (simulated).
            "outages renewing opportunistically": RepairableRBD(
                EDGES,
                {
                    "a": unit(
                        preventive={
                            "policy": "age",
                            "interval": 300.0,
                            "cost": 2.0,
                        },
                        repair_cost=5.0,
                    ),
                    "b": unit(),
                    "c": unit(),
                },
                shutdowns={
                    "outage": {
                        "times": [150.0, 400.0, 700.0],
                        "duration": 8.0,
                        "scope": ["a", "b"],
                        "cost": 30.0,
                        "opportunistic": {"a": 100.0},
                    }
                },
            ),
            "shutdown in no time": RepairableRBD(
                EDGES,
                {
                    "a": unit(preventive={"shutdown": "check"}),
                    "b": unit(),
                    "c": {
                        "reliability": E([0.001]),
                        "repairability": E([0.5]),
                        "inspection": {"shutdown": "check"},
                    },
                },
                shutdowns={
                    "check": {"period": 250.0, "offset": 40.0, "duration": 0}
                },
            ),
            "standby group, two needed": system(
                {
                    "reliability": E([0.002]),
                    "repairability": E([0.5]),
                    "standby": {"units": 3, "k": 2},
                }
            ),
            "load-sharing group": system(
                {
                    "reliability": aft(),
                    "repairability": E([0.5]),
                    "load_sharing": {"units": 3, "k": 2, "load": 2.0},
                    "repair_cost": 3.0,
                },
                downtime_cost_rate=5.0,
            ),
            "load-sharing group, Weibull": system(
                {
                    "reliability": weibull_aft(),
                    "repairability": E([0.5]),
                    "load_sharing": {"units": 2, "load": 2.0},
                }
            ),
            "identical units, parallel": system(
                {
                    "reliability": E([0.002]),
                    "repairability": E([0.5]),
                    "identical": {"units": 3, "k": 1},
                    "repair_cost": 3.0,
                },
                downtime_cost_rate=5.0,
            ),
            "identical units, series, Weibull": system(
                unit(identical={"units": 2, "k": 2}, repair_cost=3.0)
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
            # Planned outages (#319, UC-25): maintenance that takes time
            # in a nested pair and at block times, its downtime split into
            # planned and unplanned exactly; and with a repair crew, whose
            # chain does not tell them apart (simulated).
            "planned outages, nested and in a pair": RepairableRBD(
                EDGES,
                {
                    "a": RepairableRBD(
                        [("s", "x"), ("s", "y"), ("x", "t"), ("y", "t")],
                        {
                            "x": unit(
                                preventive={
                                    "interval": 250.0,
                                    "duration": E([0.25]),
                                }
                            ),
                            "y": unit(),
                        },
                    ),
                    "b": unit(
                        preventive={
                            "interval": 300.0,
                            "policy": "block",
                            "duration": surv.ExactEventTime.from_params([6.0]),
                            "cost": 3.0,
                        },
                        downtime_cost=1.0,
                    ),
                    "c": unit(),
                },
                downtime_cost_rate=5.0,
            ),
            # A nested RBD that carries costs (its repairs, a task, its
            # own downtime and crews), counted in the system's.
            "a nested RBD that carries costs": RepairableRBD(
                EDGES,
                {
                    "a": RepairableRBD(
                        [("s", "x"), ("s", "y"), ("x", "t"), ("y", "t")],
                        {
                            "x": unit(
                                repair_cost=4.0,
                                preventive={
                                    "interval": 250.0,
                                    "duration": E([0.25]),
                                    "cost": 2.0,
                                },
                            ),
                            "y": unit(repair_cost=3.0, downtime_cost=0.5),
                        },
                        downtime_cost_rate=6.0,
                        repair_crews=2,
                        crew_cost_rate=0.01,
                    ),
                    "b": unit(repair_cost=3.0),
                    "c": unit(),
                },
                downtime_cost_rate=5.0,
            ),
            "planned outages, one repair crew": system(
                unit(
                    preventive={
                        "interval": 300.0,
                        "duration": E([0.5]),
                    }
                ),
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
            # One crew, and a call-out crew once two jobs wait.
            "a call-out crew": RepairableRBD(
                EDGES,
                {
                    node: {
                        "reliability": E([0.002]),
                        "repairability": E([0.05]),
                        "priority": priority,
                        "repair_cost": 3.0,
                    }
                    for node, priority in zip("abc", (1, 0, 0))
                },
                repair_crews=1,
                call_out={"when_waiting": 2, "fee": 40.0, "rate": 6.0},
                downtime_cost_rate=5.0,
            ),
        }
    )
    return out


def safety_function_kinds():
    """The safety functions of every kind their results are checked on: a
    single channel of undetected failures, voted channels with
    diagnostics, repair and common cause of each kind, a proof test that
    can miss, staggered tests, detected failures that trip, a fitted life,
    and votes in series."""
    from repyability import SafetyChannel, SafetyFunction

    def channel(du=2e-6, dd=3e-6, safe=5e-6, **options):
        options.setdefault("repair", 8.0)
        options.setdefault("proof_test_interval", 8760.0)
        return SafetyChannel(du, dangerous_detected=dd, safe=safe, **options)

    def vote(k, n, unit, **options):
        return SafetyFunction({f"c{i}": unit for i in range(n)}, k, **options)

    betas = {"beta": 0.1, "beta_detected": 0.05, "beta_safe": 0.1}
    return {
        "1oo1, undetected failures alone": vote(
            1, 1, channel(dd=0.0, safe=0.0, repair="instant")
        ),
        "1oo2": vote(1, 2, channel()),
        "2oo3, common cause": vote(2, 3, channel(), **betas),
        "2oo4, no formula": vote(2, 4, channel(), beta=0.1),
        "a proof test that can miss": vote(
            1,
            2,
            channel(proof_test_coverage=0.9, full_test=5 * 8760.0),
            beta=0.1,
        ),
        "staggered tests": SafetyFunction(
            {"a": channel(), "b": channel(proof_test_offset=4380.0)}, 1
        ),
        "detected failures trip": vote(
            2, 3, channel(), detected_trips=True, **betas
        ),
        "a fitted life": vote(
            1, 2, channel(du=W([4e5, 1.5]), undetected_repair="instant")
        ),
        "votes in series": SafetyFunction.series(
            vote(2, 3, channel(), beta=0.1),
            SafetyFunction({"valve": channel(du=5e-6)}, 1),
        ),
    }


def event_tree_kinds():
    """The event trees of every kind their analyses are checked on:
    protection layers alone (LOPA), with paths that do not ask a heading;
    independent heading trees; trees sharing a support event; a heading
    tree with a NOT gate; lifetime models; and repaired events with an
    initiator tree of repaired events, and a repaired heading beside a
    protection layer."""
    from repyability import EventTree

    def tree(top, inputs, events, kind="or"):
        return FaultTree({top: (kind, inputs)}, events)

    lopa = EventTree(
        0.2,
        [("alarm", 0.1), ("relief", 0.01), ("dike", 0.05)],
        {
            ("success", None, None): "safe",
            ("failure", "success", None): "safe",
            ("failure", "failure", "success"): "contained",
            ("failure", "failure", "failure"): "release",
        },
    )
    independent = EventTree(
        0.1,
        [
            (
                "injection",
                tree("no injection", ["a", "b"], {"a": 0.02, "b": 0.03}),
            ),
            (
                "cooling",
                tree("no cooling", ["c", "d"], {"c": 0.1, "d": 0.2}, "and"),
            ),
        ],
    )
    shared = EventTree(
        0.1,
        [
            ("detection", 0.01),
            (
                "injection",
                tree(
                    "no injection",
                    ["pump A", "power"],
                    {"pump A": 0.05, "power": 0.01},
                ),
            ),
            (
                "recirculation",
                tree(
                    "no recirculation",
                    ["pump B", "power"],
                    {"pump B": 0.1, "power": 0.01},
                ),
            ),
        ],
        {
            ("success", "success", "success"): "ok",
            ("success", "success", "failure"): "damage",
            ("success", "failure", None): "damage",
            ("failure", None, None): "damage",
        },
    )
    not_gate = EventTree(
        0.3,
        [
            (
                "isolation",
                tree(
                    "no isolation",
                    ["valve", "power"],
                    {"valve": 0.05, "power": 0.02},
                ),
            ),
            (
                "bypass",
                FaultTree(
                    {
                        "no bypass": ("and", ["line", "powered"]),
                        "powered": ("not", ["power"]),
                    },
                    {"line": 0.1, "power": 0.02},
                ),
            ),
        ],
    )
    lifetimes = EventTree(
        0.05,
        [
            (
                "first",
                tree(
                    "first fails",
                    ["x", "shared"],
                    {"x": W([100, 2]), "shared": E([0.002])},
                ),
            ),
            (
                "second",
                tree(
                    "second fails",
                    ["y", "shared"],
                    {"y": W([80, 1.5]), "shared": E([0.002])},
                ),
            ),
        ],
    )

    def spec(rate, repair=0.5):
        return {"reliability": E([rate]), "repairability": E([repair])}

    repaired = EventTree(
        tree(
            "trip", ["feed", "grid"], {"feed": spec(0.01), "grid": spec(0.002)}
        ),
        [
            (
                "backup",
                tree(
                    "no backup",
                    ["diesel", "bus"],
                    {"diesel": spec(0.02), "bus": spec(0.001)},
                ),
            ),
            (
                "battery",
                tree(
                    "no battery",
                    ["cells", "bus"],
                    {"cells": spec(0.005), "bus": spec(0.001)},
                ),
            ),
        ],
    )
    # A repaired heading beside a protection layer (#831).
    layered = EventTree(
        0.2,
        [
            ("alarm", 0.1),
            (
                "backup",
                tree(
                    "no backup",
                    ["diesel", "start"],
                    {"diesel": spec(0.02), "start": 0.01},
                ),
            ),
        ],
    )
    return {
        "protection layers (LOPA)": lopa,
        "a repaired heading beside a protection layer": layered,
        "independent headings": independent,
        "a shared support event": shared,
        "a heading with a NOT gate": not_gate,
        "lifetime models": lifetimes,
        "repaired events, an initiator tree": repaired,
    }
