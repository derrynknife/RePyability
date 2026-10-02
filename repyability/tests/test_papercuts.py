"""Small things a new user meets first (#134): a diagram's ``repr``,
perfect junction nodes left out of the importance measures and
allocations, a number where a model belongs, ``remaining_life``'s
arguments, ages as plain numbers in a state, JSON files read and written
as surpyval's are, a ``FaultTree`` answering to an RBD's names,
``NonRepairable``'s signature, and ``RegressionNode``'s scalars and
covariate count.
"""

import inspect
import io
import json

import numpy as np
import pytest
import surpyval as surv

from repyability import (
    RBD,
    FaultTree,
    NodeState,
    NonRepairable,
    NonRepairableRBD,
    PerfectReliability,
    RegressionNode,
    RepairableRBD,
    SimulationChunk,
)

W = surv.Weibull.from_params([100.0, 2.0])
E = surv.Exponential.from_params
VOTE = [
    ("s", "a"),
    ("s", "b"),
    ("s", "c"),
    ("a", "v"),
    ("b", "v"),
    ("c", "v"),
    ("v", "t"),
]


def two_of_three() -> NonRepairableRBD:
    return NonRepairableRBD(
        VOTE,
        {"a": W, "b": W, "c": W, "v": PerfectReliability},
        k={"v": 2},
    )


def test_a_diagram_says_what_it_is():
    assert repr(two_of_three()) == (
        "NonRepairableRBD(4 nodes: 'a', 'b', 'c', 'v'; input 's', output "
        "'t'; k-out-of-n 'v': 2; junction(s) 'v')"
    )
    unit = {"reliability": E([0.01]), "repairability": E([0.5])}
    maintained = {**unit, "preventive": {"interval": 100.0}}
    rbd = RepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        {"a": unit, "b": maintained},
        repair_crews=1,
    )
    assert repr(rbd) == (
        "RepairableRBD(2 nodes: 'a', 'b'; input 's', output 't'; "
        "1 maintained; 1 repair crew(s))"
    )
    many = RBD([("s", n) for n in range(12)] + [(n, "t") for n in range(12)])
    assert repr(many).startswith("RBD(12 nodes: 0, 1, 2, 3, 4, 5, 6, 7, ...")


@pytest.mark.parametrize(
    "measure",
    [
        "birnbaum_importance",
        "improvement_potential",
        "risk_achievement_worth",
        "risk_reduction_worth",
        "criticality_importance",
        "fussell_vesely",
    ],
)
def test_a_junction_is_no_component_to_rank(measure):
    values = getattr(two_of_three(), measure)(50.0)
    assert set(values) == {"a", "b", "c"}


def test_a_junction_is_always_working():
    rbd = two_of_three()
    # Each pump is pivotal when exactly one of the other two works: half of
    # their states, with the vote always working.
    assert rbd.structural_importance() == {"a": 0.5, "b": 0.5, "c": 0.5}
    given = rbd.importances_given_state(10.0, {"a": NodeState(age=40.0)})
    assert all(set(values) == {"a", "b", "c"} for values in given.values())


def test_allocations_hold_a_junction_at_one():
    rbd = two_of_three()
    current = {n: float(W.sf(50.0)) for n in "abc"}
    for allocation in (
        rbd.equal_allocation(0.95),
        rbd.simple_allocation(0.95),
        rbd.improvement_allocation(0.95, current),
        rbd.cost_based_allocation(0.95, current),
    ):
        assert set(allocation) == {"a", "b", "c"}
        # p = 3 p**2 - 2 p**3 meets 0.95 at p = 0.8646...
        np.testing.assert_allclose(list(allocation.values()), 0.86465, 1e-4)
        values = {n: np.atleast_1d(p) for n, p in allocation.items()}
        values["v"] = np.ones(1)
        assert rbd.system_probability(values)[0] == pytest.approx(0.95)


def test_a_number_is_not_a_model():
    with pytest.raises(TypeError, match=r"from_params\(0.1\)"):
        NonRepairableRBD([("s", "a"), ("a", "t")], {"a": 0.9})
    with pytest.raises(TypeError, match="no survival function"):
        NonRepairableRBD([("s", "a"), ("a", "t")], {"a": "pump"})
    for key, value, hint in (
        ("repairability", 50, r"Exponential.from_params\(\[1 / 50\]\)"),
        ("reliability", 1000, r"ExactEventTime.from_params\(\[1000\]\)"),
    ):
        spec = {"reliability": E([0.01]), "repairability": E([0.5])}
        spec[key] = value
        with pytest.raises(TypeError, match=hint):
            RepairableRBD([("s", "a"), ("a", "t")], {"a": spec})
    with pytest.raises(ValueError, match="needs a 'repairability'"):
        RepairableRBD(
            [("s", "a"), ("a", "t")], {"a": {"reliability": E([0.01])}}
        )


def test_remaining_life_takes_the_target_first_and_ages_as_numbers():
    rbd = NonRepairableRBD([("s", "c"), ("c", "t")], {"c": W})
    with pytest.raises(TypeError, match="first argument is the reliability"):
        rbd.remaining_life({"c": NodeState(age=40.0)})
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        rbd.remaining_life(1.5)
    assert rbd.remaining_life(0.9, {"c": 40}) == rbd.remaining_life(
        0.9, {"c": NodeState(age=40.0)}
    )
    assert rbd.sf_given_state(10.0, {"c": 40.0}) == pytest.approx(
        W.sf(50.0) / W.sf(40.0)
    )
    with pytest.raises(TypeError, match="or a number, its age"):
        rbd.sf_given_state(10.0, {"c": "old"})


def test_json_goes_to_and_from_files(tmp_path):
    rbd = two_of_three()
    path = tmp_path / "plant.json"
    assert rbd.to_json(path) is None
    for source in (path, str(path), path.read_text(), io.StringIO()):
        if isinstance(source, io.StringIO):
            rbd.to_json(source)
            source.seek(0)
        restored = NonRepairableRBD.from_json(source)
        assert restored.sf(50.0) == pytest.approx(rbd.sf(50.0))
    assert json.loads(rbd.to_json())["type"] == "NonRepairableRBD"
    with pytest.raises(FileNotFoundError, match="neither a JSON document"):
        NonRepairableRBD.from_json(str(tmp_path / "missing.json"))
    tree = FaultTree({"top": ("and", ["a", "b"])}, {"a": 0.1, "b": 0.2})
    tree.to_json(tmp_path / "tree.json")
    assert FaultTree.from_json(tmp_path / "tree.json").ff() == pytest.approx(
        0.02
    )
    unit = {"reliability": E([0.1]), "repairability": E([1.0])}
    repairable = RepairableRBD([("s", "a"), ("a", "t")], {"a": unit})
    chunk = repairable.simulate_chunk(10.0, 0, 20, seed=1)
    chunk.to_json(tmp_path / "chunk.json")
    loaded = SimulationChunk.from_json(tmp_path / "chunk.json")
    assert loaded.n_simulations == 20


def test_a_fault_tree_answers_to_an_rbds_names():
    tree = FaultTree({"top": ("and", ["a", "b"])}, {"a": 0.1, "b": 0.2})
    assert tree.ff() == pytest.approx(tree.top_event_probability())
    assert tree.sf() == pytest.approx(0.98)
    # In its own right: a tiny probability of working keeps its precision.
    certain = FaultTree(
        {"top": ("or", ["a", "b"])}, {"a": 1.0 - 1e-12, "b": 1.0 - 1e-12}
    )
    assert certain.sf() == pytest.approx(1e-24, rel=1e-6)
    assert tree.get_min_cut_sets() == {frozenset({"a", "b"})}
    assert tree.get_min_path_sets() == {frozenset({"a"}), frozenset({"b"})}
    rbd = RBD([("s", "a"), ("s", "b"), ("a", "c"), ("b", "c"), ("c", "t")])
    assert rbd.minimal_cut_sets() == [
        frozenset({"c"}),
        frozenset({"a", "b"}),
    ]
    assert rbd.minimal_path_sets() == [
        frozenset({"a", "c"}),
        frozenset({"b", "c"}),
    ]
    # The same code on either.
    plant = NonRepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "c"), ("b", "c"), ("c", "t")],
        {"a": W, "b": W, "c": W},
    )
    converted = FaultTree.from_rbd(plant)
    assert converted.minimal_cut_sets() == plant.minimal_cut_sets()
    assert converted.get_min_cut_sets() == plant.get_min_cut_sets()
    assert converted.ff(50.0) == pytest.approx(plant.ff(50.0))
    assert converted.sf(50.0) == pytest.approx(plant.sf(50.0))


def test_non_repairable_shows_a_plain_default():
    signature = inspect.signature(NonRepairable)
    assert signature.parameters["time_to_replace"].default is None
    unit = NonRepairable(W)
    assert unit.time_to_replace.mean() == 0.0


def regression_model():
    rng = np.random.default_rng(0)
    load = rng.uniform(0.5, 2.0, size=200)
    times = rng.weibull(2.0, size=200) * 80.0 / np.exp(0.4 * (load - 1.0))
    return surv.WeibullAFT.fit(times + 1e-3, Z=load.reshape(-1, 1))


def test_a_regression_node_gives_scalars_and_counts_covariates():
    model = regression_model()
    node = RegressionNode(model, covariates=[1.0])
    assert np.ndim(node.sf(50.0)) == 0
    assert np.ndim(node.ff(50.0)) == 0
    assert node.sf(50.0) == pytest.approx(node.sf(np.array([50.0]))[0])
    assert node.sf(np.array([10.0, 50.0])).shape == (2,)
    with pytest.raises(ValueError, match="fitted with 1 covariate"):
        RegressionNode(model, covariates=[1.0, 2.0])
