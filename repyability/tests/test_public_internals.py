"""The public versions of what was reachable only through private calls:
each component's actions per unit time, the repair crews' jobs and chain,
a replaced component's renewal cycle, each simulation's samples and the
entropy a seed makes, a diagram's junctions and core, a model's
parameters, and a fault tree's conditional probabilities.

Each gives what the private call it stands for gives, on the catalogue's
diagrams (the seeded samples to the last bit), and a value worked out by
hand where there is one. The private calls keep working as they were."""

import functools
import json
import warnings

import numpy as np
import pytest
import surpyval as surv

from repyability import (
    ConditionalProbabilities,
    CoreStructure,
    CrewJob,
    CrewStates,
    FaultTree,
    ModelParameters,
    NodeActionRates,
    NonRepairableRBD,
    PerfectReliability,
    RepairableRBD,
    ReplacementCycle,
    SimulationSamples,
    model_parameters,
    seed_entropy,
)
from repyability.rbd import _costs, _crews, _event_loop, _long_run, _streams
from repyability.rbd._common import _squeeze_values
from repyability.rbd._events import _Unit
from repyability.rbd._model_utils import parametric_spec
from repyability.rbd.modular import _SINK, _Reduction
from repyability.rbd.routes import Refused
from repyability.tests import catalogue
from repyability.tests.catalogue import BRIDGE


@functools.lru_cache(maxsize=None)
def repairable_kinds():
    """The catalogue's repairable diagrams, built once for this module."""
    return catalogue.repairable_kinds()


@functools.lru_cache(maxsize=None)
def nonrepairable_kinds():
    return catalogue.nonrepairable_kinds()


@functools.lru_cache(maxsize=None)
def fault_tree_kinds():
    return catalogue.fault_tree_kinds()


E = surv.Exponential.from_params
W = surv.Weibull.from_params
PAIR = [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")]


def quietly(call):
    """``call()``, or the error it raises, with warnings silenced."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return call()
        except (NotImplementedError, ValueError) as error:
            return error


def pumps(crews=None, **pump):
    pump = pump or {"reliability": E([0.1]), "repairability": E([1.0])}
    return RepairableRBD(PAIR, {"a": pump, "b": pump}, repair_crews=crews)


# -- node_action_rates ------------------------------------------------------


def private_actions(rbd, working=(), broken=()):
    """Each component's actions per unit time as the private call gives
    them, with each component's long-run availability as
    ``expected_cost_rate`` works it out."""
    working, broken = set(working), set(broken)
    if rbd._crews_couple():
        probabilities, weights = _crews._chain_probabilities(
            rbd, working, broken
        )
        availability = {
            n: float(weights @ probabilities[n]) for n in rbd.nodes
        }
    else:
        availability = _squeeze_values(
            rbd._probabilities_with_overrides(
                rbd.node_availability(), working, broken
            )
        )
    return {
        node: (
            (0.0, 0.0)
            if node in working | broken
            else _costs._node_actions(rbd, node, availability[node])
        )
        for node in rbd.components
    }


@pytest.mark.parametrize("name", sorted(repairable_kinds()))
def test_the_action_rates_are_the_private_call_s(name):
    rbd = repairable_kinds()[name]
    public = quietly(rbd.node_action_rates)
    if isinstance(public, Exception):
        return  # refused: the routes test checks it says so
    private = quietly(lambda: private_actions(rbd))
    assert list(public) == list(rbd.components)
    for node, rates in public.items():
        assert isinstance(rates, NodeActionRates)
        assert (rates.corrective, rates.preventive) == tuple(
            map(float, private[node])
        ), node


def test_one_component_fails_once_a_life_and_a_repair():
    lam, mu = 0.1, 2.0
    rbd = RepairableRBD(
        [("s", "a"), ("a", "t")],
        {"a": {"reliability": E([lam]), "repairability": E([mu])}},
    )
    rates = rbd.node_action_rates()["a"]
    assert rates.corrective == pytest.approx(1 / (1 / lam + 1 / mu))
    assert rates.preventive == 0.0


def test_a_held_component_takes_no_actions():
    rbd = pumps()
    held = rbd.node_action_rates(working_nodes=["a"])
    assert held["a"] == NodeActionRates(0.0, 0.0)
    assert held["b"] == rbd.node_action_rates()["b"]
    assert rbd.node_action_rates(broken_nodes=["b"])["b"].corrective == 0.0


def test_with_one_crew_a_pump_fails_only_while_up():
    # The two pumps' chain: both up, one in repair, one waiting.
    rbd = pumps(crews=1)
    up = 1.1 / 1.22
    assert rbd.node_action_rates()["a"].corrective == pytest.approx(0.1 * up)
    public = rbd.node_action_rates(broken_nodes=["b"])
    private = private_actions(rbd, broken=["b"])
    assert public["a"].corrective == private["a"][0]


def test_unknown_held_nodes_are_refused():
    with pytest.raises(ValueError, match="pump"):
        pumps().node_action_rates(working_nodes=["pump"])


# -- crew_jobs and crew_states ----------------------------------------------


@pytest.mark.parametrize("name", sorted(repairable_kinds()))
def test_the_crew_jobs_are_the_private_call_s(name):
    rbd = repairable_kinds()[name]
    private = _crews._crew_served(rbd)
    public = rbd.crew_jobs()
    assert len(public) == len(private)
    for job, key in zip(public, private):
        assert isinstance(job, CrewJob)
        if isinstance(key, _Unit):
            assert (job.node, job.unit) == (key.node, key.unit)
        else:
            assert (job.node, job.unit) == (key, None)


def test_each_unit_of_a_standby_group_is_a_job():
    unit = {"reliability": E([0.01]), "repairability": E([0.5])}
    rbd = RepairableRBD(
        [("s", "m"), ("m", "g"), ("g", "t")],
        {"m": unit, "g": {**unit, "standby": {"units": 3}}},
        repair_crews=1,
    )
    assert rbd.crew_jobs() == [
        CrewJob("m", None),
        CrewJob("g", 0),
        CrewJob("g", 1),
        CrewJob("g", 2),
    ]


@pytest.mark.parametrize("name", sorted(repairable_kinds()))
def test_the_crew_states_are_the_private_call_s(name):
    rbd = repairable_kinds()[name]
    public = quietly(rbd.crew_states)
    if isinstance(public, Exception):
        return  # refused: the routes test checks it says so
    availability, probabilities = _crews._chain_probabilities(
        rbd, set(), set()
    )
    assert isinstance(public, CrewStates)
    assert np.array_equal(public.probabilities, probabilities)
    assert public.node_availability.keys() == availability.keys()
    for node, values in availability.items():
        assert np.array_equal(public.node_availability[node], values)


def test_two_pumps_and_one_crew_by_hand():
    # States: both up; one pump in repair (either); one in repair and the
    # other waiting (either way), in the ratio 1 : lambda / mu (each) :
    # lambda^2 / mu^2 (each).
    states = pumps(crews=1).crew_states()
    assert sorted(states.probabilities) == pytest.approx(
        sorted(np.array([1.0, 0.1, 0.1, 0.01, 0.01]) / 1.22)
    )
    up = states.probabilities @ states.node_availability["a"]
    assert up == pytest.approx(1.1 / 1.22)
    assert up == pytest.approx(pumps(crews=1).node_availability()["a"])


def test_held_nodes_need_no_crew():
    rbd = pumps(crews=1)
    states = rbd.crew_states(working_nodes=["b"])
    assert np.all(states.node_availability["b"] == 1.0)
    # The other pump has the crew to itself.
    assert states.probabilities @ states.node_availability[
        "a"
    ] == pytest.approx(1.0 / 1.1)


def test_crews_that_never_keep_a_job_waiting_have_no_chain():
    for rbd in (pumps(), pumps(crews=2)):
        with pytest.raises(Refused, match="no crews' chain"):
            rbd.crew_states()


# -- replacement_cycle ------------------------------------------------------


def replaced_nodes(rbd):
    return [
        node
        for node, schedule in rbd._preventive.items()
        if schedule.policy in ("block", "condition")
    ]


@pytest.mark.parametrize("name", sorted(repairable_kinds()))
def test_the_replacement_cycle_is_the_private_call_s(name):
    rbd = repairable_kinds()[name]
    for node in replaced_nodes(rbd):
        public = quietly(lambda: rbd.replacement_cycle(node))
        if isinstance(public, Exception):
            continue  # refused: the routes test checks it says so
        private = _long_run._block_cycle(rbd, node)
        assert isinstance(public, ReplacementCycle)
        for field in public:
            assert np.array_equal(
                public[field], getattr(private, field)
            ), field


def test_instant_block_replacement_of_an_exponential_life():
    # Repaired and replaced in no time, it is always up and fails at its
    # rate: lambda T failures an interval.
    lam, T = 0.002, 300.0
    rbd = RepairableRBD(
        [("s", "a"), ("a", "t")],
        {
            "a": {
                "reliability": E([lam]),
                "repairability": "instant",
                "preventive": {"interval": T, "policy": "block"},
            }
        },
    )
    cycle = rbd.replacement_cycle("a")
    assert cycle.length == pytest.approx(T)
    assert cycle.up == pytest.approx(T)
    assert cycle.failures == pytest.approx(lam * T, rel=1e-6)
    assert cycle.before == pytest.approx(1.0)


def test_a_component_with_no_cycle_is_refused():
    with pytest.raises(Refused, match="No component here"):
        pumps().replacement_cycle("a")
    rbd = repairable_kinds()["block replacement"]
    (block,) = replaced_nodes(rbd)
    other = next(n for n in rbd.components if n != block)
    with pytest.raises(Refused, match=f"Component {other!r} is not"):
        rbd.replacement_cycle(other)
    with pytest.raises(ValueError, match="Unknown node"):
        rbd.replacement_cycle("pump")


# -- simulation_samples and seed_entropy ------------------------------------


def test_the_entropy_is_the_private_call_s():
    for seed in (0, 1, 7, 2**32 - 1, [3, 4]):
        assert seed_entropy(seed) == _streams.entropy_of(seed)
    np.random.seed(5)
    drawn = seed_entropy()
    np.random.seed(5)
    assert drawn == _streams.entropy_of(None)


@pytest.mark.parametrize("name", sorted(repairable_kinds()))
def test_the_samples_are_the_private_run_s_to_the_last_bit(name):
    rbd = repairable_kinds()[name]
    public = quietly(lambda: rbd.simulation_samples(200.0, 30, seed=3))
    if isinstance(public, Exception):
        return  # refused: the routes test checks it says so
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tally = _event_loop._run(
            rbd,
            200.0,
            set(),
            set(),
            "c",
            30,
            False,
            None,
            entropy=_streams.entropy_of(3),
            common=True,
        )
    assert isinstance(public, SimulationSamples)
    assert np.array_equal(
        public.availability, np.asarray(tally.uptimes, dtype=float) / 200.0
    )
    if rbd.has_costs:
        assert np.array_equal(
            public.cost, np.asarray(tally.cost_samples, dtype=float)
        )
    else:
        assert public.cost is None


def test_a_seed_and_its_entropy_give_the_same_run():
    rbd = repairable_kinds()["costed_pairs"]
    by_seed = rbd.simulation_samples(100.0, 50, seed=11)
    by_entropy = rbd.simulation_samples(100.0, 50, entropy=seed_entropy(11))
    assert np.array_equal(by_seed.availability, by_entropy.availability)
    assert np.array_equal(by_seed.cost, by_entropy.cost)
    # A simulation is the same however many the run has.
    first = rbd.simulation_samples(100.0, 20, seed=11)
    assert np.array_equal(first.availability, by_seed.availability[:20])


def test_held_nodes_are_held_in_every_simulation():
    rbd = pumps()
    held = rbd.simulation_samples(100.0, 50, seed=2, broken_nodes=["a"])
    alone = RepairableRBD(
        [("s", "b"), ("b", "t")],
        {"b": {"reliability": E([0.1]), "repairability": E([1.0])}},
    ).simulation_samples(100.0, 50, seed=2)
    # Pump b draws from its own stream in both.
    assert np.array_equal(held.availability, alone.availability)


def test_common_random_numbers_pair_two_designs():
    # The same draws in both: the paired differences vary less
    # than two independent runs' would.
    fast = pumps(reliability=E([0.1]), repairability=E([1.0]))
    slow = pumps(reliability=E([0.1]), repairability=E([0.5]))
    a = fast.simulation_samples(100.0, 400, seed=4)
    b = slow.simulation_samples(100.0, 400, seed=4)
    other = slow.simulation_samples(100.0, 400, seed=5)
    assert np.mean(a.availability - b.availability) > 0.0
    assert np.var(a.availability - b.availability) < 0.75 * np.var(
        a.availability - other.availability
    )


def test_a_seed_and_an_entropy_together_are_refused():
    with pytest.raises(ValueError, match="not both"):
        pumps().simulation_samples(10.0, 5, seed=1, entropy=2)
    for entropy in (-1, 2**62, 1.5, True):
        with pytest.raises(ValueError, match="entropy must be"):
            pumps().simulation_samples(10.0, 5, entropy=entropy)


# -- junctions and core_structure -------------------------------------------


def every_diagram():
    return {**nonrepairable_kinds(), **repairable_kinds()}


@pytest.mark.parametrize("name", sorted(every_diagram()))
def test_the_junctions_are_the_private_call_s(name):
    rbd = every_diagram()[name]
    assert rbd.junctions() == rbd._junctions()
    assert isinstance(rbd.junctions(), frozenset)


def private_core(rbd):
    """The core as the private reduction gives it, the repeated nodes
    pinned."""
    aliases = rbd._component_aliases()
    reduction = _Reduction(
        rbd.G,
        rbd.input_node,
        rbd.output_node,
        pinned=set(aliases) | set(aliases.values()),
    )
    reduction.run()
    if len(reduction.alive) <= 1:
        return None
    return reduction.core_graph()


@pytest.mark.parametrize("name", sorted(every_diagram()))
def test_the_core_is_the_private_reduction_s(name):
    rbd = every_diagram()[name]
    public = rbd.core_structure()
    private = private_core(rbd)
    if private is None:
        assert public is None
        return
    assert isinstance(public, CoreStructure)
    assert public.output == _SINK
    assert set(public.graph.nodes) == set(private.nodes)
    assert set(public.graph.edges) == set(private.edges)
    for v in private.nodes:
        assert public.graph.nodes[v]["k"] == private.nodes[v]["k"]
    json.dumps(public.to_dict())


def test_a_bridge_is_its_own_core():
    rbd = NonRepairableRBD(BRIDGE, {n: E([0.01]) for n in "abcde"})
    core = rbd.core_structure()
    stands_for = [
        data["components"]
        for v, data in core.graph.nodes(data=True)
        if data["components"]
    ]
    assert sorted(map(sorted, stands_for)) == [[n] for n in "abcde"]
    assert core.graph.in_degree(core.output) == 2
    (source,) = [v for v in core.graph if core.graph.in_degree(v) == 0]
    assert core.graph.out_degree(source) == 2


def test_a_bridge_s_arms_reduce_into_modules():
    # Each corner of the bridge a series pair: the core's vertices stand
    # for the pairs.
    edges = [
        ("s", "a1"),
        ("a1", "a2"),
        ("a2", "c"),
        ("s", "b"),
        ("b", "c"),
        ("a2", "d"),
        ("c", "d"),
        ("b", "e"),
        ("c", "e"),
        ("d", "t"),
        ("e", "t"),
    ]
    rbd = NonRepairableRBD(
        edges, {n: E([0.01]) for n in ("a1", "a2", "b", "c", "d", "e")}
    )
    stands_for = {
        frozenset(data["components"])
        for _, data in rbd.core_structure().graph.nodes(data=True)
    }
    assert frozenset({"a1", "a2"}) in stands_for


def test_a_series_parallel_diagram_has_no_core():
    rbd = NonRepairableRBD(PAIR, {"a": E([0.1]), "b": E([0.1])})
    assert rbd.core_structure() is None


def test_a_repeated_component_stays_in_the_core():
    # Two parallel paths sharing a supply drawn twice: with the copies
    # independent it would reduce; repeated, it cannot.
    edges = [
        ("s", "p1"),
        ("p1", "a"),
        ("s", "p2"),
        ("p2", "b"),
        ("a", "t"),
        ("b", "t"),
    ]
    models = {"p1": E([0.01]), "a": E([0.1]), "b": E([0.1])}
    plain = NonRepairableRBD(edges, {**models, "p2": E([0.01])})
    assert plain.core_structure() is None
    shared = NonRepairableRBD(edges, {**models, "p2": "p1"})
    core = shared.core_structure()
    assert core is not None
    names = set().union(
        *(d["components"] for _, d in core.graph.nodes(data=True))
    )
    assert {"p1", "p2"} <= names


# -- model_parameters --------------------------------------------------------


def every_model():
    for rbd in nonrepairable_kinds().values():
        yield from rbd.reliabilities.values()
    for rbd in repairable_kinds().values():
        for component in rbd.components.values():
            yield getattr(component, "reliability", None)
            yield getattr(component, "time_to_replace", None)
    for tree in fault_tree_kinds().values():
        for event in tree.events:
            yield tree._event_model(event)


def test_the_parameters_are_the_private_call_s():
    found = 0
    for model in every_model():
        private = (
            None
            if model is None or isinstance(model, str)
            else parametric_spec(model)
        )
        public = model_parameters(model)
        if private is None:
            assert public is None
            continue
        found += 1
        assert isinstance(public, ModelParameters)
        assert public.cls is private.cls
        assert public.params == private.params
        assert public.names == private.names
        assert public.extras == private.extras
        assert public.bounds == private.bounds
        moved = [1.1 * p for p in public.params]
        t = np.array([1.0, 50.0, 500.0])
        assert np.array_equal(
            public.build(moved).sf(t), private.build(moved).sf(t)
        )
        json.dumps(public.to_dict())
    assert found > 10


def test_a_limited_failure_population_rebuilds_with_its_share():
    model = W([100.0, 2.0], lfp_p=0.3)
    found = model_parameters(model)
    assert found.cls is surv.Weibull
    assert found.names == ["alpha", "beta", "lfp_p"]
    t = np.array([10.0, 100.0, 1e4])
    assert np.allclose(found.build(found.params).sf(t), model.sf(t))
    assert found.build([100.0, 2.0, 0.6]).ff(1e6) == pytest.approx(0.6)
    assert found.to_dict()["cls"] == "Weibull"


def test_what_has_no_parameters_gives_none():
    for model in (None, "a name", PerfectReliability, pumps()):
        assert model_parameters(model) is None


# -- conditional_probabilities -----------------------------------------------


@pytest.mark.parametrize("name", sorted(fault_tree_kinds()))
def test_the_conditionals_are_the_private_call_s(name):
    tree = fault_tree_kinds()[name]
    t = np.array([30.0, 300.0])
    public = quietly(lambda: tree.conditional_probabilities(t))
    if isinstance(public, Exception):
        return  # refused: the routes test checks it says so
    _, top, occurred, not_occurred, _, _, _ = tree._conditioned(t)
    assert isinstance(public, ConditionalProbabilities)
    assert np.array_equal(public.top, np.broadcast_to(top, t.shape))
    for event in tree.events:
        assert np.array_equal(public.given_occurred[event], occurred[event])
        assert np.array_equal(
            public.given_not_occurred[event], not_occurred[event]
        )
    # The importance measures are made of them.
    birnbaum = tree.birnbaum_importance(t)
    for event in tree.events:
        assert np.allclose(
            public.given_occurred[event] - public.given_not_occurred[event],
            birnbaum[event],
            rtol=1e-12,
            atol=0.0,
        )


def test_the_conditionals_by_hand():
    tree = FaultTree(
        {
            "top": ("or", ["valve", "flow"]),
            "flow": ("and", ["pump 1", "pump 2"]),
        },
        {"pump 1": 0.1, "pump 2": 0.1, "valve": 0.05},
    )
    given = tree.conditional_probabilities()
    assert given.top == pytest.approx(1 - 0.95 * 0.99)
    assert given.given_occurred["valve"] == 1.0
    assert given.given_not_occurred["valve"] == pytest.approx(0.01)
    assert given.given_occurred["pump 1"] == pytest.approx(1 - 0.95 * 0.9)
    assert given.given_not_occurred["pump 1"] == pytest.approx(0.05)
    assert isinstance(given.top, float)
    json.dumps(given.to_dict())


# -- the results through JSON ------------------------------------------------


def test_the_results_go_through_json():
    rbd = pumps(crews=1)
    for result in (
        rbd.node_action_rates(),
        rbd.crew_jobs(),
        rbd.crew_states(),
        rbd.simulation_samples(10.0, 5, seed=1),
        repairable_kinds()["block replacement"].replacement_cycle("a"),
    ):
        json.dumps(
            {k: v.to_dict() for k, v in result.items()}
            if isinstance(result, dict)
            else (
                [v.to_dict() for v in result]
                if isinstance(result, list)
                else result.to_dict()
            )
        )
