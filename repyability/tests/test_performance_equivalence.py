"""The performance fast paths must not change any result.

Each optimisation keeps the code it replaced as a fallback (or, for the exact
engine, is checked against a verbatim copy of the original recursion), and
these tests hold the fast path to that reference:

- the exact engine replays a Shannon decomposition recorded once per RBD
  instead of re-deriving it on every call: results must be *identical*;
- the samplers draw the global RNG's uniforms in one block, in the order the
  draw-at-a-time code consumed them, and apply each model's quantile function
  to the block: the samples must match the per-draw code, and the global RNG
  must end in exactly the same state;
- the per-sample loops of the warm-standby and load-sharing simulations run
  for every sample at once: same arithmetic, same tie-breaks;
- the repairable simulation's event queue drops ``queue.PriorityQueue``'s
  locking but keeps its heap, so events come out in the same order, and its
  components (nested RBDs' included) draw from one stream of uniforms, in the
  order they drew from the global RNG;
- composite nodes (standby, repeated, load-sharing, regression, nested RBDs)
  replay their ``random(1)`` for a whole block of samples, so an RBD
  containing them is batched too;
- minimal cut sets are read off the exact engine's decomposition instead of
  Berge's algorithm (the minimal cut sets are unique, so they must match
  exactly), and that decomposition runs on an explicit stack.

Floating-point samples are compared to 1e-12 relative (the fast paths do the
same operations, so they agree to the last bit here; the tolerance only
guards against a platform's vectorised math differing in the last bit). A
wrong assignment of uniforms to draws would show up as an O(1) difference.
Counts and RNG states are compared exactly.
"""

import dataclasses
import queue
from collections.abc import Mapping
from typing import Any, Dict

import numpy as np
import pytest
import surpyval as surv
from surpyval import FixedEventProbability

from repyability import (
    BetaFactor,
    CCFGroup,
    LoadSharingModel,
    NonRepairableRBD,
    RepairableRBD,
    StandbyModel,
)
from repyability.non_repairable import NonRepairable
from repyability.rbd import (
    _sampling,
    non_repairable_rbd,
    repairable_rbd,
    standby_node,
)
from repyability.rbd.helper_classes import (
    PerfectReliability,
    PerfectUnreliability,
)
from repyability.rbd.rbd import (
    minimal_cut_sets_from_path_sets,
    probability_any_set_satisfied,
)
from repyability.rbd.regression_node import RegressionNode
from repyability.rbd.repeated_node import RepeatedNode
from repyability.rbd.repeated_standby_node import RepeatedStandbyNode
from repyability.utils.wrappers import numpy_seed

W = surv.Weibull.from_params
RTOL = 1e-12


def rng_state():
    state = np.random.get_state()
    return state[1].copy(), state[2], state[3], state[4]


def assert_same_rng_state(a, b):
    assert np.array_equal(a[0], b[0]) and a[1:] == b[1:]


def no_fast_path(monkeypatch):
    """Force every sampler back onto its original draw-at-a-time code."""
    monkeypatch.setattr(non_repairable_rbd, "row_sampler", lambda model: None)
    for module in (standby_node, repairable_rbd):
        monkeypatch.setattr(module, "inverse_sampler", lambda model: None)


def assert_same(a, b, path="result"):
    """Recursively compare result objects: floats to RTOL, the rest exactly."""
    if dataclasses.is_dataclass(a):
        assert type(a) is type(b), path
        for f in dataclasses.fields(a):
            assert_same(
                getattr(a, f.name), getattr(b, f.name), f"{path}.{f.name}"
            )
    elif isinstance(a, Mapping):
        assert set(a) == set(b), path
        for key in a:
            assert_same(a[key], b[key], f"{path}[{key!r}]")
    elif isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        assert a.shape == b.shape, path
        np.testing.assert_allclose(a, b, rtol=RTOL, atol=0, err_msg=path)
    elif isinstance(a, float):
        np.testing.assert_allclose(a, b, rtol=RTOL, atol=0, err_msg=path)
    else:
        assert a == b, path


# -- the RBDs the fast paths are exercised on --------------------------------

BRIDGE = [
    ("s", "a"),
    ("s", "b"),
    ("a", "c"),
    ("b", "c"),
    ("a", "d"),
    ("c", "d"),
    ("b", "e"),
    ("d", "t"),
    ("e", "t"),
]


def series_of_parallels(n_sub=4, width=3):
    edges, models, previous = [], {}, ["s"]
    for i in range(n_sub):
        layer = [f"n{i}_{j}" for j in range(width)]
        edges += [(p, node) for p in previous for node in layer]
        for j, node in enumerate(layer):
            models[node] = W([1000 + 100 * i, 1.2 + 0.1 * j])
        previous = layer
    edges += [(p, "t") for p in previous]
    return NonRepairableRBD(edges, models)


def rbds():
    """RBDs covering every structural feature and plain model type."""
    return {
        "bridge": NonRepairableRBD(
            BRIDGE,
            {n: W([700 + 100 * i, 1.5]) for i, n in enumerate("abcde")},
        ),
        "mixed_models": NonRepairableRBD(
            BRIDGE,
            {
                "a": surv.Exponential.from_params([1 / 800]),
                "b": surv.LogNormal.from_params([6.5, 0.6]),
                "c": surv.Gamma.from_params([3.0, 0.004]),
                "d": W([700, 2.2], gamma=40),  # offset (3-parameter)
                "e": surv.Normal.from_params([900, 150]),
            },
        ),
        "series_of_parallels": series_of_parallels(),
        "koon": NonRepairableRBD(
            [
                ("s", "a"),
                ("s", "b"),
                ("s", "c"),
                ("a", "v"),
                ("b", "v"),
                ("c", "v"),
                ("v", "t"),
            ],
            {
                "a": W([500, 2]),
                "b": W([600, 1.5]),
                "c": W([700, 3]),
                "v": surv.Exponential.from_params([1 / 5000]),
            },
            k={"v": 2},
        ),
        "repeated_node": NonRepairableRBD(
            [(1, 2), (2, 3), (1, 4), (4, 5), (3, 6), (5, 6)],
            {2: W([400, 2]), 3: W([500, 1.3]), 4: 2, 5: W([650, 1.8])},
        ),
        "perfect_and_exact_nodes": NonRepairableRBD(
            [
                ("s", "a"),
                ("a", "b"),
                ("s", "p"),
                ("p", "x"),
                ("b", "t"),
                ("x", "t"),
                ("s", "u"),
                ("u", "t"),
            ],
            {
                "a": W([300, 2]),
                "b": W([900, 1.1]),
                "p": PerfectReliability,
                "u": PerfectUnreliability,
                "x": surv.ExactEventTime.from_params(250.0),
            },
        ),
        "ccf": NonRepairableRBD(
            [("s", "a"), ("s", "b"), ("a", "c"), ("b", "c"), ("c", "t")],
            {"a": W([500, 2]), "b": W([500, 2]), "c": W([2000, 1.2])},
            ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.1))],
        ),
    }


# -- the exact engine --------------------------------------------------------


def reference_probability_any_set_satisfied(
    sets, element_probabilities, array_shape
):
    """The exact engine before its decomposition was recorded as a plan:
    a verbatim copy of the original memoised recursion."""
    sets = [frozenset(s) for s in sets]
    memo: Dict[frozenset, np.ndarray] = {}

    def recurse(state: frozenset) -> np.ndarray:
        if not state:
            return np.zeros(array_shape)
        if frozenset() in state:
            return np.ones(array_shape)
        if state in memo:
            return memo[state]
        counts: Dict[Any, int] = {}
        for s in state:
            for element in s:
                counts[element] = counts.get(element, 0) + 1
        pivot = max(counts, key=lambda e: counts[e])
        p = element_probabilities[pivot]
        state_active = frozenset(s - {pivot} for s in state)
        state_inactive = frozenset(s for s in state if pivot not in s)
        result = p * recurse(state_active) + (1 - p) * recurse(state_inactive)
        memo[state] = result
        return result

    return recurse(frozenset(sets))


@pytest.mark.parametrize("name", sorted(rbds()))
@pytest.mark.parametrize("method", ["p", "c"])
def test_exact_engine_is_identical_to_the_original_recursion(name, method):
    rbd = rbds()[name]
    rng = np.random.default_rng(0)
    # The recorded plan is reused across calls with different probabilities.
    for shape in (1, 7):
        probs = {n: rng.uniform(0.0, 1.0, shape) for n in rbd.nodes}
        if method == "p":
            sets = rbd.get_min_path_sets(include_in_out_nodes=False)
            expected = reference_probability_any_set_satisfied(
                sets, probs, shape
            )
        else:
            sets = rbd.get_min_cut_sets(include_in_out_nodes=False)
            unreliability = {k: 1 - v for k, v in probs.items()}
            expected = 1 - reference_probability_any_set_satisfied(
                sets, unreliability, shape
            )
        assert np.array_equal(rbd.system_probability(probs, method), expected)


def test_probability_any_set_satisfied_is_unchanged():
    rng = np.random.default_rng(1)
    elements = list("abcdefg")
    for trial in range(40):
        sets = [
            frozenset(rng.choice(elements, size=rng.integers(0, 4)))
            for _ in range(rng.integers(0, 6))
        ]
        probs = {e: rng.uniform(0, 1, 3) for e in elements}
        assert np.array_equal(
            probability_any_set_satisfied(sets, probs, 3),
            reference_probability_any_set_satisfied(sets, probs, 3),
        ), sets


@pytest.mark.parametrize("name", sorted(rbds()))
def test_structure_function_matches_its_definition(name):
    rbd = rbds()[name]
    path_sets = rbd.get_min_path_sets(include_in_out_nodes=False)
    rng = np.random.default_rng(2)
    for _ in range(200):
        status = {n: bool(rng.random() < 0.6) for n in rbd.nodes}
        expected = any(all(status[c] for c in p) for p in path_sets)
        assert rbd.is_system_working(status, "p") is expected
        assert rbd.is_system_working(status, "c") is expected


# -- inverse-transform sampling ----------------------------------------------

PLAIN_MODELS = {
    "weibull": W([100, 2.0]),
    "offset_weibull": W([100, 2.0], gamma=15),
    "exponential": surv.Exponential.from_params([0.01]),
    "lognormal": surv.LogNormal.from_params([4.0, 0.7]),
    "gamma": surv.Gamma.from_params([3.0, 0.05]),
    "normal": surv.Normal.from_params([100, 10]),
    "exact": surv.ExactEventTime.from_params(42.0),
}


@pytest.mark.parametrize("name", sorted(PLAIN_MODELS))
def test_inverse_sampler_reproduces_surpyval(name):
    model = PLAIN_MODELS[name]
    sampler = _sampling.inverse_sampler(model)
    assert sampler is not None
    np.random.seed(3)
    expected = np.concatenate([model.random(1) for _ in range(50)])
    after_single_draws = rng_state()
    np.random.seed(3)
    assert np.array_equal(sampler(np.random.random_sample(50)), expected)
    assert_same_rng_state(rng_state(), after_single_draws)


@pytest.mark.parametrize(
    "model",
    [
        FixedEventProbability.from_params(0.1),
        W([100, 2], p=0.9),  # limited failure population
        W([100, 2], f0=0.1),  # zero-inflated
        surv.KaplanMeier.fit(np.array([1.0, 2.0, 3.0, 4.0])),
        StandbyModel([W([100, 2])] * 2, k=1),
        PerfectReliability,
        NonRepairableRBD([("s", "a"), ("a", "t")], {"a": W([100, 2])}),
    ],
)
def test_inverse_sampler_declines_what_it_cannot_reproduce(model):
    assert _sampling.inverse_sampler(model) is None


def test_uniform_stream_matches_single_draws_across_blocks():
    models = [PLAIN_MODELS[k] for k in ("weibull", "lognormal", "exact")]
    order = np.random.default_rng(4).integers(0, 3, size=100)
    np.random.seed(5)
    expected = [models[i].random(1).item() for i in order]
    after_single_draws = rng_state()

    np.random.seed(5)
    stream = _sampling.UniformStream(block_size=7)  # force many refills
    samplers = [_sampling.inverse_sampler(m) for m in models]
    drawn = [stream.draw(samplers[i]) for i in order]
    stream.close()
    assert drawn == expected
    assert_same_rng_state(rng_state(), after_single_draws)


# -- NonRepairableRBD.random -------------------------------------------------


@pytest.mark.parametrize("name", sorted(rbds()))
def test_random_is_identical_to_the_event_loop(name):
    rbd = rbds()[name]
    assert rbd._random_vectorised(1) is not None  # the fast path applies
    fast = rbd.random(2000, seed=6)
    with numpy_seed(6):
        reference = rbd._random_by_events(2000)
    np.testing.assert_allclose(fast, reference, rtol=RTOL, atol=0)

    # Unseeded, the global RNG ends where the event loop leaves it.
    np.random.seed(7)
    rbd.random(300)
    after_fast = rng_state()
    np.random.seed(7)
    rbd._random_by_events(300)
    assert_same_rng_state(rng_state(), after_fast)


def test_random_mean_and_interval_use_the_fast_path_unchanged(monkeypatch):
    rbd = rbds()["bridge"]
    fast_mean = rbd.mean(3000, seed=8)
    fast_interval = rbd.mean_time_to_failure_interval(3000, seed=9)
    no_fast_path(monkeypatch)
    assert rbd._random_vectorised(1) is None
    assert rbd.mean(3000, seed=8) == pytest.approx(fast_mean, rel=RTOL)
    assert_same(rbd.mean_time_to_failure_interval(3000, seed=9), fast_interval)


def test_random_falls_back_for_nodes_it_cannot_reproduce():
    # A zero-inflated model draws through np.random.binomial, which cannot be
    # replayed from a block of uniforms.
    rbd = NonRepairableRBD(
        [("s", "a"), ("a", "z"), ("z", "t")],
        {"a": W([900, 1.4]), "z": W([300, 2.0], f0=0.05)},
    )
    np.random.seed(10)
    before = rng_state()
    assert rbd._random_vectorised(5) is None
    assert_same_rng_state(rng_state(), before)  # nothing was drawn
    with numpy_seed(11):
        reference = rbd._random_by_events(200)
    assert np.array_equal(rbd.random(200, seed=11), reference)


def test_random_rewinds_and_falls_back_on_nan(monkeypatch):
    # The event loop's ordering of NaN times cannot be reproduced, so a NaN
    # draw must hand over to it with the RNG rewound.
    monkeypatch.setattr(
        _sampling,
        "inverse_sampler",
        lambda model: (lambda u: np.where(u < 0.5, np.nan, u)),
    )
    rbd = rbds()["bridge"]
    np.random.seed(12)
    before = rng_state()
    assert rbd._random_vectorised(50) is None
    assert_same_rng_state(rng_state(), before)


# -- standby and load sharing ------------------------------------------------

COLD_UNITS = [
    W([100, 2.0]),
    surv.LogNormal.from_params([4.4, 0.5]),
    W([90, 1.5], gamma=3),
    surv.Gamma.from_params([4, 0.05]),
    W([120, 1.7]),
]


@pytest.mark.parametrize("k", [2, 3, 4])
def test_cold_standby_k_is_identical_to_the_queue_loop(k, monkeypatch):
    model = StandbyModel(COLD_UNITS, k=k, n_sims=50, seed=0)
    fast = model.random(3000, seed=13)
    np.random.seed(14)
    model.random(200)
    after_fast = rng_state()

    no_fast_path(monkeypatch)
    np.testing.assert_allclose(
        model.random(3000, seed=13), fast, rtol=RTOL, atol=0
    )
    np.random.seed(14)
    model.random(200)
    assert_same_rng_state(rng_state(), after_fast)


def test_cold_standby_fit_is_unchanged(monkeypatch):
    t = np.linspace(0, 500, 26)
    fast = StandbyModel(COLD_UNITS, k=2, n_sims=2000, seed=15).sf(t)
    no_fast_path(monkeypatch)
    slow = StandbyModel(COLD_UNITS, k=2, n_sims=2000, seed=15).sf(t)
    np.testing.assert_allclose(fast, slow, rtol=RTOL, atol=0)


def budgets_with_ties(size, n):
    # Small integers make exact ties between units common, which is where a
    # vectorised loop could pick a different unit than the per-sample loop.
    return np.random.default_rng(16).integers(1, 5, (size, n)).astype(float)


@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("dormancy", [0.1, 0.5, 1.0])
@pytest.mark.parametrize("ties", [False, True])
def test_warm_standby_is_identical_per_sample(k, dormancy, ties):
    model = StandbyModel(
        [W([100, 2.0])] * 4, k=k, dormancy_factor=dormancy, n_sims=10, seed=0
    )
    budgets = (
        budgets_with_ties(3000, 4)
        if ties
        else np.random.default_rng(17).weibull(1.5, (3000, 4)) * 100
    )
    np.testing.assert_allclose(
        model._warm_lifetimes(budgets),
        model._warm_lifetimes_by_sample(budgets),
        rtol=RTOL,
        atol=0,
    )


class FixedDraws:
    """A stand-in unit whose ``random`` returns preset values."""

    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

    def random(self, size):
        return self.values[:size]


def test_warm_standby_with_infinite_budgets_uses_the_per_sample_loop(
    monkeypatch,
):
    model = StandbyModel(
        [W([100, 2.0])] * 3, k=1, dormancy_factor=0.3, n_sims=10, seed=0
    )
    budgets = budgets_with_ties(20, 3)
    budgets[3, 1] = np.inf
    model.reliabilities = [FixedDraws(column) for column in budgets.T]
    monkeypatch.setattr(
        StandbyModel,
        "_warm_lifetimes",
        lambda self, b: pytest.fail(
            "the vectorised loop needs finite budgets"
        ),
    )
    assert np.array_equal(
        model._random_warm(20), model._warm_lifetimes_by_sample(budgets)
    )


@pytest.fixture(scope="module")
def weibull_aft():
    rng = np.random.default_rng(1)
    load = rng.uniform(0.5, 2.0, size=400)
    x = rng.weibull(2.0, size=400) * 80.0 / np.exp(0.4 * (load - 1)) + 1e-3
    return surv.WeibullAFT.fit(x, Z=load.reshape(-1, 1))


@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("ties", [False, True])
def test_load_sharing_is_identical_per_sample(weibull_aft, k, ties):
    model = LoadSharingModel(
        [weibull_aft] * 3, load=2.0, k=k, n_sims=10, seed=0
    )
    tau = (
        budgets_with_ties(3000, 3).T
        if ties
        else np.random.default_rng(19).weibull(2.0, (3, 3000)) * 50
    )
    np.testing.assert_allclose(
        model._lifetimes(tau),
        model._lifetimes_by_sample(tau),
        rtol=RTOL,
        atol=0,
    )


@pytest.mark.filterwarnings("ignore:divide by zero:RuntimeWarning")
@pytest.mark.parametrize("bad", ["inf_threshold", "zero_phi"])
def test_load_sharing_degenerate_inputs_use_the_per_sample_loop(
    weibull_aft, bad, monkeypatch
):
    model = LoadSharingModel([weibull_aft] * 3, load=2.0, n_sims=10, seed=0)
    tau = budgets_with_ties(20, 3).T
    if bad == "inf_threshold":
        tau[1, 5] = np.inf
    else:
        model._phi_table = model._phi_table.copy()
        model._phi_table[0, -1] = 0.0
    model._baselines = [FixedDraws(row) for row in tau]
    monkeypatch.setattr(
        LoadSharingModel,
        "_lifetimes",
        lambda self, t: pytest.fail("the vectorised loop needs finite times"),
    )
    assert np.array_equal(
        model.random(20, seed=0), model._lifetimes_by_sample(tau)
    )


def test_load_sharing_fit_is_unchanged(weibull_aft, monkeypatch):
    t = np.linspace(0, 200, 21)
    fast = LoadSharingModel([weibull_aft] * 3, load=2.0, n_sims=2000, seed=20)
    monkeypatch.setattr(
        LoadSharingModel,
        "_lifetimes",
        lambda self, tau: self._lifetimes_by_sample(tau),
    )
    slow = LoadSharingModel([weibull_aft] * 3, load=2.0, n_sims=2000, seed=20)
    np.testing.assert_allclose(fast.sf(t), slow.sf(t), rtol=RTOL, atol=0)


# -- the repairable simulation -----------------------------------------------


def repairable_pair(scale, reliability=None):
    """A RepairableRBD of two repairable units in parallel, to nest."""
    return RepairableRBD(
        [("s", "p"), ("s", "q"), ("p", "t"), ("q", "t")],
        {
            "p": {
                "reliability": (
                    W([scale, 2]) if reliability is None else reliability
                ),
                "repairability": surv.Exponential.from_params([0.5]),
            },
            "q": {
                "reliability": W([scale + 5, 2]),
                "repairability": surv.LogNormal.from_params([0.2, 0.4]),
            },
        },
    )


def repairable_rbds():
    L = surv.LogNormal.from_params
    # A model object used for several nodes, even at different levels, has
    # one event state; the stand-ins must share it the same way.
    pump = NonRepairable(W([50, 2]), surv.Exponential.from_params([0.5]))
    shared = NonRepairable(W([60, 1.5]), L([0.3, 0.5]))
    return {
        "costed_pairs": RepairableRBD(
            [
                ("s", "a"),
                ("s", "b"),
                ("a", "c"),
                ("b", "c"),
                ("a", "d"),
                ("b", "d"),
                ("c", "t"),
                ("d", "t"),
            ],
            {
                n: {
                    "reliability": W([100 + 20 * i, 1.5]),
                    "repairability": L([1.5, 0.5]),
                    "repair_cost": 100.0,
                    "downtime_cost": 5.0,
                }
                for i, n in enumerate("abcd")
            },
            downtime_cost_rate=50.0,
        ),
        "instant_repair_random_cost": RepairableRBD(
            [("s", "x"), ("x", "y"), ("y", "z"), ("z", "t")],
            {
                "x": {
                    "reliability": surv.Exponential.from_params([0.02]),
                    "repairability": "instant",
                    "replace_cost": surv.Gamma.from_params([4.0, 0.04]),
                },
                "y": {
                    "reliability": W([80, 2.5], gamma=5),
                    "repairability": surv.Exponential.from_params([0.5]),
                    "repair_cost": 40.0,
                },
                "z": {
                    "reliability": L([4.0, 0.7]),
                    "repairability": W([3, 1.2]),
                },
            },
            downtime_cost_rate=10.0,
        ),
        "koon": RepairableRBD(
            [
                ("s", "a"),
                ("s", "b"),
                ("s", "c"),
                ("a", "v"),
                ("b", "v"),
                ("c", "v"),
                ("v", "t"),
            ],
            {
                n: {
                    "reliability": W([60 + 10 * i, 1.8]),
                    "repairability": surv.Exponential.from_params([0.3]),
                }
                for i, n in enumerate("abcv")
            },
            k={"v": 2},
        ),
        "nonrepairable_objects": RepairableRBD(
            [("s", "a"), ("a", "b"), ("b", "t")],
            {
                "a": NonRepairable(
                    W([50, 2]), surv.Exponential.from_params([1.0])
                ),
                "b": NonRepairable(
                    surv.Exponential.from_params([0.01]), L([0.5, 0.3])
                ),
            },
        ),
        "nested_one_level": RepairableRBD(
            [("s", "a"), ("a", "sub"), ("sub", "t")],
            {
                "a": {
                    "reliability": W([70, 1.5]),
                    "repairability": surv.Exponential.from_params([0.8]),
                    "repair_cost": 50.0,
                },
                "sub": repairable_pair(40),
            },
            downtime_cost_rate=5.0,
        ),
        "nested_two_levels": RepairableRBD(
            [("s", "a"), ("a", "mid"), ("mid", "t")],
            {
                "a": {
                    "reliability": W([120, 1.5]),
                    "repairability": surv.Exponential.from_params([0.8]),
                },
                "mid": RepairableRBD(
                    [("s", "m"), ("s", "deep"), ("m", "t"), ("deep", "t")],
                    {
                        "m": {
                            "reliability": W([70, 2]),
                            "repairability": surv.Exponential.from_params(
                                [0.4]
                            ),
                        },
                        "deep": RepairableRBD(
                            [("s", "u"), ("u", "v"), ("v", "t")],
                            {
                                "u": {
                                    "reliability": W([90, 1.5]),
                                    "repairability": "instant",
                                },
                                "v": NonRepairable(
                                    surv.Exponential.from_params([0.01]),
                                    W([2, 1.5]),
                                ),
                            },
                        ),
                    },
                ),
            },
        ),
        "nested_koon": RepairableRBD(
            [
                ("s", "x"),
                ("s", "y"),
                ("s", "z"),
                ("x", "v"),
                ("y", "v"),
                ("z", "v"),
                ("v", "t"),
            ],
            {
                "x": repairable_pair(50),
                "y": repairable_pair(55),
                "z": {
                    "reliability": W([60, 1.8]),
                    "repairability": surv.Exponential.from_params([0.3]),
                },
                "v": {
                    "reliability": W([500, 1.2]),
                    "repairability": surv.Exponential.from_params([1.0]),
                },
            },
            k={"v": 2},
        ),
        "one_model_two_nodes": RepairableRBD(
            [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
            {"a": pump, "b": pump},
        ),
        "one_model_two_levels": RepairableRBD(
            [
                ("s", "a"),
                ("s", "x"),
                ("s", "y"),
                ("a", "t"),
                ("x", "t"),
                ("y", "t"),
            ],
            {
                "a": shared,
                "x": RepairableRBD([("s", "p"), ("p", "t")], {"p": shared}),
                "y": RepairableRBD([("s", "p"), ("p", "t")], {"p": shared}),
            },
        ),
    }


NESTED = ["nested_koon", "nested_one_level", "nested_two_levels"]


def simulate_both(monkeypatch, rbd, **kwargs):
    fast = rbd.availability(**kwargs)
    with monkeypatch.context() as m:
        m.setattr(repairable_rbd, "inverse_sampler", lambda model: None)
        assert rbd._streamed_components(_sampling.UniformStream()) is None
        reference = rbd.availability(**kwargs)
    return fast, reference


@pytest.mark.parametrize("name", sorted(repairable_rbds()))
def test_repairable_simulation_is_identical(name, monkeypatch):
    rbd = repairable_rbds()[name]
    assert rbd._streamed_components(_sampling.UniformStream()) is not None
    first = rbd.nodes[0]
    for kwargs in (
        dict(t_simulation=300.0, N=60, seed=21),
        dict(t_simulation=200.0, N=30, seed=22, method="c"),
        dict(t_simulation=200.0, N=30, seed=23, working_nodes=[first]),
        dict(t_simulation=200.0, N=30, seed=24, broken_nodes=[first]),
    ):
        fast, reference = simulate_both(monkeypatch, rbd, **kwargs)
        assert_same(fast, reference)


@pytest.mark.parametrize("name", sorted(repairable_rbds()))
def test_repairable_simulation_leaves_the_global_rng_unchanged(
    name, monkeypatch
):
    rbd = repairable_rbds()[name]
    np.random.seed(25)
    fast = rbd.availability(150.0, N=25)
    after_fast = rng_state()
    with monkeypatch.context() as m:
        m.setattr(repairable_rbd, "inverse_sampler", lambda model: None)
        np.random.seed(25)
        reference = rbd.availability(150.0, N=25)
    assert_same(fast, reference)
    assert_same_rng_state(rng_state(), after_fast)


@pytest.mark.parametrize("forced", ["working_nodes", "broken_nodes"])
@pytest.mark.parametrize("name", NESTED)
def test_forcing_a_nested_rbd_is_identical(name, forced, monkeypatch):
    rbd = repairable_rbds()[name]
    for node, component in rbd.components.items():
        if isinstance(component, RepairableRBD):
            fast, reference = simulate_both(
                monkeypatch,
                rbd,
                t_simulation=200.0,
                N=30,
                seed=26,
                **{forced: [node]},
            )
            assert_same(fast, reference)


def test_nested_rbd_that_cannot_be_streamed_falls_back_entirely(monkeypatch):
    # A zero-inflated model draws through np.random.binomial, which the
    # stream cannot reproduce. Streaming the other components' draws would
    # change their order, so nothing is streamed.
    rbd = RepairableRBD(
        [("s", "a"), ("a", "sub"), ("sub", "t")],
        {
            "a": {
                "reliability": W([70, 1.5]),
                "repairability": surv.Exponential.from_params([0.8]),
            },
            "sub": repairable_pair(40, W([40, 2], f0=0.1)),
        },
    )
    assert rbd._streamed_components(_sampling.UniformStream()) is None
    fast, reference = simulate_both(
        monkeypatch, rbd, t_simulation=200.0, N=30, seed=27
    )
    assert_same(fast, reference)


def test_subclassed_components_keep_their_own_event_methods():
    # A subclass may draw its events its own way, so the simulation calls
    # the component itself rather than streaming its draws.
    E = surv.Exponential.from_params
    calls = []

    class LoggedUnit(NonRepairable):
        def next_event(self):
            calls.append("unit")
            return super().next_event()

    class LoggedRBD(RepairableRBD):
        def next_event(self, method="p"):
            calls.append("rbd")
            return super().next_event(method)

    unit = {"reliability": W([40, 2]), "repairability": E([0.5])}
    for component, kind in (
        (LoggedUnit(W([40, 2]), E([0.5])), "unit"),
        (LoggedRBD([("s", "p"), ("p", "t")], {"p": unit}), "rbd"),
    ):
        rbd = RepairableRBD(
            [("s", "a"), ("a", "sub"), ("sub", "t")],
            {
                "a": {"reliability": W([70, 1.5]), "repairability": E([0.8])},
                "sub": component,
            },
        )
        assert rbd._streamed_components(_sampling.UniformStream()) is None
        calls.clear()
        rbd.availability(200.0, N=5, seed=30)
        assert kind in calls


def step_by_hand(rbd, t_simulation):
    """Every system event of one of ``rbd``'s simulations, stepped through
    with its public event API."""
    rbd.initialize_event_queue(t_simulation)
    events = [rbd.next_event()]
    while events[-1][0] < t_simulation:
        events.append(rbd.next_event())
    return events


def test_nested_rbds_step_by_hand_as_before_after_a_simulation():
    # availability() hands nested RBDs their stand-ins only for the run:
    # afterwards they draw from their components again.
    used = repairable_rbds()["nested_two_levels"]
    used.availability(300.0, N=10, seed=28)
    fresh = repairable_rbds()["nested_two_levels"]
    np.random.seed(29)
    expected = step_by_hand(fresh, 2000.0)
    after_fresh = rng_state()
    np.random.seed(29)
    assert step_by_hand(used, 2000.0) == expected
    assert_same_rng_state(rng_state(), after_fresh)
    assert len(expected) > 10


def test_event_queue_pops_in_priority_queue_order():
    # Equal times are where two heaps could disagree; use many.
    rng = np.random.default_rng(27)
    ours = repairable_rbd._EventQueue()
    theirs: queue.PriorityQueue = queue.PriorityQueue()
    for step in range(2000):
        if rng.random() < 0.6 or theirs.empty():
            event = repairable_rbd.Event(
                float(rng.integers(0, 20)), step, bool(rng.random() < 0.5)
            )
            ours.put(event)
            theirs.put(event)
        else:
            assert ours.get() is theirs.get()
        assert ours.qsize() == theirs.qsize()
        assert ours.empty() == theirs.empty()


# -- composite nodes inside an RBD --------------------------------------------


@pytest.fixture(scope="module")
def composites(weibull_aft):
    """One of each composite node type, covering every sampling branch.
    (Building the cold-standby convolution evaluates LogNormal densities at
    zero, which only warns.)"""
    with np.errstate(divide="ignore", invalid="ignore"):
        return composite_nodes(weibull_aft)


def other_aft():
    """A second Weibull-AFT unit, unlike ``weibull_aft``."""
    rng = np.random.default_rng(2)
    load = rng.uniform(0.5, 2.0, size=400)
    x = rng.weibull(1.5, size=400) * 120.0 / np.exp(0.2 * (load - 1)) + 1e-3
    return surv.WeibullAFT.fit(x, Z=load.reshape(-1, 1))


def composite_nodes(aft):
    L = surv.LogNormal.from_params
    return {
        "standby_warm": StandbyModel(
            [W([300, 2.0]), W([280, 1.7]), L([5.5, 0.4])],
            k=1,
            dormancy_factor=0.3,
            n_sims=50,
            seed=1,
        ),
        "standby_warm_k2": StandbyModel(
            [W([300, 2.0])] * 4, k=2, dormancy_factor=0.6, n_sims=50, seed=1
        ),
        "standby_cold_k1": StandbyModel(
            [W([300, 2.0]), L([5.5, 0.4]), W([250, 1.2], gamma=5)], k=1
        ),
        "standby_cold_k1_switching": StandbyModel(
            [W([300, 2.0])] * 3, k=1, switching_probability=0.9
        ),
        "standby_cold_k1_switch_list": StandbyModel(
            [W([300, 2.0])] * 3, k=1, switching_probability=[0.95, 0.8]
        ),
        "standby_cold_k2": StandbyModel(COLD_UNITS, k=2, n_sims=50, seed=1),
        "repeated_parallel": RepeatedNode(W([500, 2.0]), 3, "parallel"),
        "repeated_series": RepeatedNode(L([6.0, 0.5]), 2, "series"),
        "repeated_standby": RepeatedStandbyNode(W([300, 2.0]), 3),
        "repeated_standby_switching": RepeatedStandbyNode(
            W([300, 2.0]), 3, switching_probability=0.85
        ),
        "load_sharing": LoadSharingModel(
            [aft] * 3, load=2.0, n_sims=50, seed=1
        ),
        # Different units, so their order matters.
        "load_sharing_mixed": LoadSharingModel(
            [aft, aft, other_aft()], load=2.0, n_sims=50, seed=1
        ),
        "regression": RegressionNode(aft, covariates=[1.2]),
        "nested_rbd": NonRepairableRBD(
            [("s", "q"), ("s", "sb"), ("q", "t"), ("sb", "t")],
            {"q": W([800, 2]), "sb": StandbyModel([W([300, 2.0])] * 2, k=1)},
        ),
        "perfect": PerfectReliability,
        "plain": W([400, 1.5]),
    }


def rbd_with(node):
    return NonRepairableRBD(
        [("s", "a"), ("a", "x"), ("s", "b"), ("b", "x"), ("x", "t")],
        {"a": W([900, 1.4]), "b": W([800, 1.6]), "x": node},
    )


COMPOSITE_NAMES = [
    "standby_warm",
    "standby_warm_k2",
    "standby_cold_k1",
    "standby_cold_k1_switching",
    "standby_cold_k1_switch_list",
    "standby_cold_k2",
    "repeated_parallel",
    "repeated_series",
    "repeated_standby",
    "repeated_standby_switching",
    "load_sharing",
    "load_sharing_mixed",
    "regression",
    "nested_rbd",
    "perfect",
    "plain",
]


@pytest.mark.parametrize("name", COMPOSITE_NAMES)
def test_row_sampler_replays_random_1(composites, name):
    # The block replay of ``random(1)``: same values, same RNG consumption
    # (the block's width is how many uniforms one call takes).
    model = composites[name]
    sampler = _sampling.row_sampler(model)
    assert sampler is not None
    np.random.seed(30)
    expected = np.array(
        [float(np.asarray(model.random(1)).reshape(-1)[0]) for _ in range(300)]
    )
    after_single_draws = rng_state()
    np.random.seed(30)
    got = sampler.draw(np.random.random_sample((300, sampler.width)))
    np.testing.assert_allclose(got, expected, rtol=RTOL, atol=0)
    assert_same_rng_state(rng_state(), after_single_draws)


@pytest.mark.parametrize("name", COMPOSITE_NAMES)
def test_rbd_with_a_composite_node_is_identical(composites, name):
    rbd = rbd_with(composites[name])
    assert rbd._random_vectorised(1) is not None  # the fast path applies
    fast = rbd.random(1500, seed=31)
    with numpy_seed(31):
        reference = rbd._random_by_events(1500)
    np.testing.assert_allclose(fast, reference, rtol=RTOL, atol=0)

    np.random.seed(32)
    rbd.random(200)
    after_fast = rng_state()
    np.random.seed(32)
    rbd._random_by_events(200)
    assert_same_rng_state(rng_state(), after_fast)


def test_every_composite_node_in_one_rbd(composites):
    nodes = composites
    edges = [("s", n) for n in nodes] + [(n, "t") for n in nodes]
    rbd = NonRepairableRBD(edges, nodes)
    assert rbd._random_vectorised(1) is not None
    fast = rbd.random(800, seed=33)
    with numpy_seed(33):
        reference = rbd._random_by_events(800)
    np.testing.assert_allclose(fast, reference, rtol=RTOL, atol=0)


def test_kaplan_meier_node_is_batched_without_the_global_rng(monkeypatch):
    with numpy_seed(34):
        km = surv.KaplanMeier.fit(W([700, 2]).random(200))
    rbd = rbd_with(km)
    assert rbd._random_vectorised(1) is not None
    # surpyval draws Kaplan-Meier samples from a fresh, OS-seeded generator,
    # never numpy's global RNG, so the global stream -- and with it every
    # other node's draws -- must be untouched by batching them.
    np.random.seed(35)
    rbd.random(300)
    after_fast = rng_state()
    np.random.seed(35)
    rbd._random_by_events(300)
    assert_same_rng_state(rng_state(), after_fast)
    # Pinning the Kaplan-Meier draws makes the rest comparable exactly.
    monkeypatch.setattr(
        type(km), "random", lambda self, size, *a, **k: np.full(size, 450.0)
    )
    fast = rbd.random(500, seed=36)
    with numpy_seed(36):
        reference = rbd._random_by_events(500)
    np.testing.assert_allclose(fast, reference, rtol=RTOL, atol=0)


@pytest.mark.parametrize("where", ["standby", "nested"])
def test_nan_inside_a_composite_node_falls_back(
    weibull_aft, where, monkeypatch
):
    # A NaN draw anywhere, however deep, must hand the whole batch to the
    # event loop with the RNG rewound.
    def nan_sampler(model):
        return lambda u: np.where(u < 0.3, np.nan, 100.0 + u)

    if where == "standby":
        monkeypatch.setattr(standby_node, "inverse_sampler", nan_sampler)
        node = StandbyModel(COLD_UNITS, k=2, n_sims=5, seed=0)
    else:
        node = rbd_with(W([500, 2]))
        monkeypatch.setattr(_sampling, "inverse_sampler", nan_sampler)
    rbd = rbd_with(node)
    np.random.seed(37)
    before = rng_state()
    assert rbd._random_vectorised(100) is None
    assert_same_rng_state(rng_state(), before)


def test_nan_in_an_irrelevant_node_still_falls_back(monkeypatch):
    # b lies on no minimal path (a -> t bypasses it), so its lifetime never
    # reaches the system's max-min, but the event loop still queues its NaN
    # time: the batch must still hand over to it.
    bad = W([500, 2.0])
    rbd = NonRepairableRBD(
        [("s", "a"), ("a", "t"), ("a", "b"), ("b", "t")],
        {"a": W([400, 2.0]), "b": bad},
    )
    assert rbd.structure_check["irrelevant_nodes"] == {"b"}
    real = _sampling.inverse_sampler
    monkeypatch.setattr(
        _sampling,
        "inverse_sampler",
        lambda m: (lambda u: np.full(len(u), np.nan)) if m is bad else real(m),
    )
    np.random.seed(39)
    before = rng_state()
    assert rbd._random_vectorised(50) is None
    assert_same_rng_state(rng_state(), before)


# -- minimal cut sets --------------------------------------------------------


def reference_minimal_cut_sets(path_sets):
    """The previous implementation (Berge's algorithm), verbatim."""

    def keep_minimal(sets):
        minimal = []
        for candidate in sorted(set(sets), key=len):
            if not any(kept <= candidate for kept in minimal):
                minimal.append(candidate)
        return minimal

    transversals = [frozenset()]
    for path_set in path_sets:
        path_set = frozenset(path_set)
        candidates = []
        for transversal in transversals:
            if transversal & path_set:
                candidates.append(transversal)
            else:
                for component in path_set:
                    candidates.append(transversal | {component})
        transversals = keep_minimal(candidates)
    return set(transversals)


def test_cut_sets_match_berge_on_random_hypergraphs():
    rng = np.random.default_rng(38)
    for _ in range(3000):
        elements = list(range(rng.integers(1, 9))) + ["x", ("t", 1)]
        path_sets = [
            frozenset(
                elements[i]
                for i in rng.choice(
                    len(elements),
                    rng.integers(0, len(elements)),
                    replace=False,
                )
            )
            for _ in range(rng.integers(0, 8))
        ]
        assert minimal_cut_sets_from_path_sets(
            path_sets
        ) == reference_minimal_cut_sets(path_sets), path_sets


@pytest.mark.parametrize("name", sorted(rbds()))
@pytest.mark.parametrize("in_out", [False, True])
def test_rbd_cut_sets_match_berge(name, in_out):
    rbd = rbds()[name]
    path_sets = rbd.get_min_path_sets(include_in_out_nodes=in_out)
    expected = reference_minimal_cut_sets(path_sets)
    assert rbd.get_min_cut_sets(include_in_out_nodes=in_out) == expected
    # Cached, but each call gets its own copy.
    got = rbd.get_min_cut_sets(include_in_out_nodes=in_out)
    got.add(frozenset({"not a node"}))
    assert rbd.get_min_cut_sets(include_in_out_nodes=in_out) == expected


def test_multi_stage_redundancy_cut_sets():
    # Where Berge's intermediate families explode (729 path sets), and the
    # answer is simply each stage.
    rbd = series_of_parallels(6, 3)
    stages = {frozenset(f"n{i}_{j}" for j in range(3)) for i in range(6)}
    assert rbd.get_min_cut_sets() == stages


# -- the decomposition on an explicit stack -----------------------------------


@pytest.mark.parametrize(
    "edges_of",
    [
        lambda n: [("s", f"c{i}") for i in range(n)]
        + [(f"c{i}", "t") for i in range(n)],
        lambda n: [("s", "c0")]
        + [(f"c{i}", f"c{i + 1}") for i in range(n - 1)]
        + [(f"c{n - 1}", "t")],
    ],
    ids=["parallel", "series"],
)
def test_deep_decomposition_is_identical_to_the_recursion(edges_of):
    # Deep enough to exercise the stack, shallow enough for the recursive
    # reference.
    n = 300
    rbd = NonRepairableRBD(
        edges_of(n),
        {f"c{i}": W([1000 + i, 1.5]) for i in range(n)},
    )
    probs = rbd._base_node_probabilities(np.array([400.0]), set(), set())
    sets = rbd.get_min_path_sets(include_in_out_nodes=False)
    assert np.array_equal(
        rbd.system_probability(probs),
        reference_probability_any_set_satisfied(sets, probs, 1),
    )


def test_wide_system_needs_no_recursion():
    # Wider than Python's recursion limit: the recursive decomposition could
    # not handle this.
    n = 1100
    q = 0.01
    units = [f"c{i}" for i in range(n)]
    rbd = NonRepairableRBD(
        [("s", u) for u in units] + [(u, "t") for u in units],
        {u: FixedEventProbability.from_params(q) for u in units},
    )
    assert float(rbd.sf()) == pytest.approx(1 - q**n)
    assert rbd.get_min_cut_sets() == {frozenset(units)}
