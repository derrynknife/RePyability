"""The API's rough edges smoothed (#179, #180, #184): one way to name the
structure methods, numbers wherever a model or count is taken, arrays
wherever siblings take them, results whose values are properties, and the
MGL shocks as PRA codes combine them."""

import itertools
import warnings

import numpy as np
import pytest
import surpyval as surv
from scipy.integrate import quad
from surpyval import FixedEventProbability

from repyability import (
    MGL,
    CCFGroup,
    Network,
    NodeState,
    NonRepairableRBD,
    PhasedMission,
    RepairableRBD,
    StandbyModel,
    Timeline,
    demonstrated_mtbf,
    demonstrated_reliability,
    demonstration_pass_probability,
    demonstration_sample_size,
    demonstration_test_multiple,
    mtbf_pass_probability,
    mtbf_test_time,
)
from repyability._version import __version__
from repyability.rbd.ccf import with_parameters
from repyability.rbd.uncertainty import draw_ccf_models
from repyability.utils.deprecation import NEXT_REMOVAL, REMOVAL_AFTER_NEXT

W, E = surv.Weibull.from_params, surv.Exponential.from_params
F = FixedEventProbability.from_params

BRIDGE = [
    ("s", "a"),
    ("s", "b"),
    ("a", "c"),
    ("b", "c"),
    ("a", "d"),
    ("b", "e"),
    ("c", "d"),
    ("c", "e"),
    ("d", "t"),
    ("e", "t"),
]


# -- structure methods by name --------------------------------------------


def test_paths_and_cuts_are_p_and_c():
    rbd = NonRepairableRBD(BRIDGE, {n: W([100, 2]) for n in "abcde"})
    t = np.array([20.0, 60.0])
    assert np.array_equal(rbd.sf(t, method="paths"), rbd.sf(t, method="p"))
    assert np.array_equal(rbd.sf(t, method="cuts"), rbd.sf(t, method="c"))
    probabilities = {n: 0.9 for n in "abcde"}
    assert rbd.system_probability(
        probabilities, method="cuts"
    ) == rbd.system_probability(probabilities, method="c")
    assert rbd.is_system_working(
        {"a": True, "b": False, "c": False, "d": True, "e": False}, "paths"
    )
    unit = {"reliability": W([100, 2]), "repairability": E([0.5])}
    repairable = RepairableRBD(BRIDGE, {n: dict(unit) for n in "abcde"})
    assert repairable.mean_availability(
        method="cuts"
    ) == repairable.mean_availability(method="c")
    runs = [
        repairable.availability(
            50.0, mc_samples=20, seed=1, method=m, engine="python"
        )
        for m in ("paths", "p")
    ]
    assert np.array_equal(runs[0].uptimes, runs[1].uptimes)
    with pytest.raises(ValueError, match="'paths'"):
        rbd.sf(20.0, method="path")


# -- numbers where a model is taken ---------------------------------------


def test_a_network_takes_a_probability_of_failing():
    links = {"a": ("s", "x"), "b": ("s", "y"), "c": ("x", "y")}
    links.update({"d": ("x", "t"), "e": ("y", "t")})
    plain = Network({n: (u, v, 0.1) for n, (u, v) in links.items()}, "s", "t")
    fixed = Network(
        {n: (u, v, F(0.1)) for n, (u, v) in links.items()}, "s", "t"
    )
    assert plain.sf() == fixed.sf() == pytest.approx(0.97848)
    assert plain.is_fixed
    node = Network(
        {"a": ("s", "x", 0.1), "d": ("x", "t", 0.1)},
        "s",
        "t",
        nodes={"x": 0.05},
    )
    assert node.sf() == pytest.approx(0.9 * 0.9 * 0.95, rel=1e-12)
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        Network({"a": ("s", "t", 1.5)}, "s", "t")
    with pytest.raises(ValueError, match="sf and ff"):
        Network({"a": ("s", "t", True)}, "s", "t")


# -- demonstration tests ---------------------------------------------------


@pytest.mark.parametrize(
    "function, arrays, fixed",
    [
        (demonstration_sample_size, {"reliability": [0.9, 0.95]}, {}),
        (demonstrated_reliability, {"n": [59, 93], "failures": [0, 1]}, {}),
        (
            demonstration_test_multiple,
            {"reliability": [0.9, 0.95]},
            {"n": 20, "shape": 2.0},
        ),
        (
            demonstration_pass_probability,
            {"reliability": [0.95, 0.99]},
            {"n": 59},
        ),
        (mtbf_test_time, {"mtbf": [1000.0, 2000.0], "failures": [0, 2]}, {}),
        (demonstrated_mtbf, {"test_time": [2995.7, 6295.8]}, {}),
        (
            mtbf_pass_probability,
            {"mtbf": [1000.0, 3000.0]},
            {"test_time": 2995.7},
        ),
    ],
)
def test_the_demonstration_functions_take_arrays(function, arrays, fixed):
    together = function(**{k: np.array(v) for k, v in arrays.items()}, **fixed)
    one_by_one = [
        function(**dict(zip(arrays, values)), **fixed)
        for values in zip(*arrays.values())
    ]
    assert together.shape == (2,)
    assert together.tolist() == one_by_one


def test_a_count_is_not_a_bool():
    with pytest.raises(ValueError, match="failures"):
        demonstration_sample_size(0.9, 0.9, failures=True)
    with pytest.raises(ValueError, match="n must"):
        demonstrated_reliability(True)
    with pytest.raises(ValueError, match="failure_terminated"):
        demonstrated_mtbf(1000.0, failures=1, failure_terminated="yes")


def test_a_failure_terminated_test():
    # 2r degrees of freedom: the plan is a time-terminated test allowing
    # one failure fewer, and the bound the time at the r-th failure.
    for r in (1, 2, 5):
        time = mtbf_test_time(1000.0, 0.9, r, failure_terminated=True)
        assert time == mtbf_test_time(1000.0, 0.9, r - 1)
        assert demonstrated_mtbf(
            time, 0.9, r, failure_terminated=True
        ) == pytest.approx(1000.0, rel=1e-12)
        assert mtbf_pass_probability(
            1000.0, time, r, failure_terminated=True
        ) == pytest.approx(0.1, rel=1e-9)
    with pytest.raises(ValueError, match="at least 1"):
        mtbf_test_time(1000.0, failures=0, failure_terminated=True)


# -- arrays of targets -----------------------------------------------------


def test_several_targets_at_once():
    rbd = NonRepairableRBD(BRIDGE, {n: W([100, 2]) for n in "abcde"})
    targets = np.array([0.9, 0.5, 0.1])
    times = rbd.time_to_reliability(targets)
    assert times.tolist() == [rbd.time_to_reliability(r) for r in targets]
    assert rbd.bx_life([10, 50]).tolist() == [
        rbd.bx_life(10),
        rbd.bx_life(50),
    ]
    state = {"a": NodeState(age=30.0)}
    assert rbd.remaining_life([0.9, 0.5], state).tolist() == [
        rbd.remaining_life(0.9, state),
        rbd.remaining_life(0.5, state),
    ]
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        rbd.remaining_life([0.5, 1.5])


def test_the_mean_residual_life():
    unit = W([100, 2])
    rbd = NonRepairableRBD(BRIDGE, {n: unit for n in "abcde"})
    assert rbd.mean_residual_life() == pytest.approx(rbd.mean(), rel=1e-9)
    state = {"a": NodeState(age=40.0), "c": NodeState(age=10.0)}
    expected = quad(
        lambda t: float(rbd.sf_given_state(t, state)), 0, np.inf, limit=200
    )[0]
    assert rbd.mean_residual_life(state) == pytest.approx(expected, rel=1e-8)
    # A failed node is no part of what is left.
    alone = NonRepairableRBD([("s", "c"), ("c", "t")], {"c": unit})
    after = quad(lambda t: unit.sf(40 + t) / unit.sf(40), 0, np.inf)[0]
    assert alone.mean_residual_life(
        {"c": NodeState(age=40.0)}
    ) == pytest.approx(after, rel=1e-9)
    grouped = NonRepairableRBD(
        [("s", "x"), ("s", "y"), ("x", "t"), ("y", "t")],
        {"x": unit, "y": unit},
        ccf_groups=[CCFGroup(["x", "y"], MGL(0.1))],
    )
    with pytest.raises(NotImplementedError, match="common-cause"):
        grouped.mean_residual_life()
    fixed = NonRepairableRBD([("s", "c"), ("c", "t")], {"c": F(0.1)})
    with pytest.raises(ValueError, match="lifetimes"):
        fixed.mean_residual_life()


# -- phased missions -------------------------------------------------------


def test_a_mission_has_sf_and_ff():
    pair = NonRepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        {"a": W([100, 2]), "b": W([100, 2])},
    )
    mission = PhasedMission([("one", 30.0, pair), ("two", 20.0, pair)])
    assert mission.sf() == mission.reliability()
    assert mission.ff() == mission.unreliability()


def test_equal_standby_models_are_one_model_through_a_mission():
    unit = W([100, 2])

    def phase(k=1):
        return NonRepairableRBD(
            [("s", "x"), ("x", "t")],
            {"x": StandbyModel([unit, unit, unit], k=k)},
        )

    mission = PhasedMission([("p", 10.0, phase()), ("q", 20.0, phase())])
    single = PhasedMission([("pq", 30.0, phase())])
    assert mission.sf() == pytest.approx(single.sf(), rel=1e-12)
    with pytest.raises(ValueError, match="different model"):
        PhasedMission([("p", 10.0, phase()), ("q", 20.0, phase(k=2))])


# -- results ---------------------------------------------------------------


def test_the_spares_demand_values_are_properties():
    rbd = RepairableRBD(
        [("s", "pump"), ("pump", "t")],
        {"pump": {"reliability": E([0.01]), "repairability": "instant"}},
    )
    demand = rbd.spares_demand(1000.0)["pump"]
    assert demand.mean == pytest.approx(10.0, rel=1e-5)
    assert demand.std == pytest.approx(np.sqrt(10.0), rel=1e-4)
    with pytest.warns(FutureWarning, match=r"write mean, not mean\(\)"):
        assert demand.mean() == demand.mean
    with pytest.warns(FutureWarning, match="0.13"):
        assert demand.std() == demand.std


def test_the_calls_go_in_the_release_after_next():
    # When the version reaches NEXT_REMOVAL, what 0.12 deprecates must go.
    version = tuple(map(int, __version__.split(".")))
    assert version < tuple(map(int, NEXT_REMOVAL.split(".")))


def test_what_0_13_deprecates_goes_in_0_14():
    # optimal_inspection_intervals(offsets=), renamed offset_shares (#222);
    # CapacityDistribution.mean() called, now a property (#235); and the
    # simulation options of StandbyModel, LoadSharingModel and
    # DegradingNode's exact mean, which it ignores (#233).
    version = tuple(map(int, __version__.split(".")))
    assert version < tuple(map(int, REMOVAL_AFTER_NEXT.split(".")))


def test_an_uncertainty_result_shows_a_summary():
    pump = surv.Weibull.fit(np.linspace(200, 1800, 50))
    rbd = NonRepairableRBD([("s", "pump"), ("pump", "t")], {"pump": pump})
    result = rbd.sf_uncertainty(500, {"pump": "fit"}, n_draws=500, seed=0)
    text = repr(result)
    assert text.startswith("UncertaintyResult(nominal=")
    assert "interval_90=" in text and "n_draws=500" in text
    assert len(text) < 200
    several = repr(rbd.sf_uncertainty([100, 500], n_draws=50, seed=0))
    assert several.count("[") == 4  # nominal, median and the interval's two


def test_the_fitted_nodes_are_the_default_uncertainty():
    pump = surv.Weibull.fit(np.linspace(200, 1800, 50))
    seal = surv.Weibull.fit(np.linspace(300, 2500, 40))
    valve = F(0.01)
    rbd = NonRepairableRBD(
        [("s", "p1"), ("s", "p2"), ("p1", "q"), ("p2", "q")]
        + [("q", "valve"), ("valve", "t")],
        {"p1": pump, "p2": pump, "q": seal, "valve": valve},
    )
    assert rbd._fitted_uncertainty() == {("p1", "p2"): "fit", "q": "fit"}
    default = rbd.sf_uncertainty(400, n_draws=200, seed=2)
    given = rbd.sf_uncertainty(
        400, {("p1", "p2"): "fit", "q": "fit"}, n_draws=200, seed=2
    )
    np.testing.assert_array_equal(default.samples, given.samples)
    certain = NonRepairableRBD([("s", "v"), ("v", "t")], {"v": valve})
    with pytest.raises(ValueError, match="No node's model is a surpyval fit"):
        certain.sf_uncertainty(1.0)
    with pytest.raises(ValueError, match="Give the uncertain nodes"):
        rbd.sf_uncertainty(400, {})


def test_a_run_controlled_by_itself_is_exact():
    unit = {"reliability": W([100, 2]), "repairability": E([0.5])}
    rbd = RepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        {"a": dict(unit, repair_cost=3.0), "b": dict(unit, repair_cost=3.0)},
    )
    run = rbd.availability(
        200.0, mc_samples=100, seed=1, control_variate=True, engine="python"
    )
    interval = run.mean_availability_interval()
    assert interval.method == "exact"
    assert interval.standard_error == 0.0
    assert interval.estimate == rbd.mission_availability(200.0)
    cost = rbd.cost(
        200.0, mc_samples=100, seed=1, control_variate=True, engine="python"
    )
    assert cost.mean_interval().method == "exact"
    # A twin that leaves something out still estimates.
    crewed = RepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        {"a": dict(unit), "b": dict(unit)},
        repair_crews=1,
    )
    controlled = crewed.availability(
        200.0, mc_samples=100, seed=1, control_variate=True, engine="python"
    ).mean_availability_interval()
    assert controlled.method == "control_variate"
    assert controlled.standard_error > 0.0


# -- outage logs -----------------------------------------------------------


def test_an_outage_log_is_merged_on_request():
    records = [(500, 520), (100, 104), (102, 110), (520, 530)]
    with pytest.raises(ValueError, match="merge=True"):
        Timeline.from_outages(records, end=1000)
    joined = Timeline.from_outages(
        records, end=1000, merge=True, planned=[True, False, True, True]
    )
    assert joined.failures == 1  # (100, 110): one record was a failure
    assert joined.downtime == 40.0
    assert joined == Timeline.from_outages(
        [(100, 110), (500, 530)], end=1000, planned=[False, True]
    )
    touching = Timeline.from_outages([(10, 20), (20, 30)], end=100)
    assert touching.failures == 2
    assert (
        Timeline.from_outages(
            [(10, 20), (20, 30)], end=100, merge=True
        ).failures
        == 1
    )


# -- the MGL shocks (#180) -------------------------------------------------


def brute_force(model, Q, k):
    """A 3-member group's k-out-of-3 failure probability, its causes
    independent basic events: each specific set's ``Q_k``, as the
    exclusive split gives them."""
    q1, shocks = MGL(*model.letters).decompose(list("abc"), Q)
    events = [frozenset(x) for x in "abc"] + [s for s, _ in shocks]
    probabilities = [float(q1[0])] * 3 + [float(p[0]) for _, p in shocks]
    failing = 0.0
    for row in itertools.product([False, True], repeat=len(events)):
        failed: set = set()
        weight = 1.0
        for event, p, fires in zip(events, probabilities, row):
            weight *= p if fires else 1.0 - p
            if fires:
                failed |= event
        if 3 - len(failed) < k:
            failing += weight
    return failing


def vote(model, k):
    parallel = [("s", x) for x in "abc"] + [(x, "t") for x in "abc"]
    return NonRepairableRBD(
        parallel,
        {x: W([1000, 1.5]) for x in "abc"},
        k={"t": k},
        ccf_groups=[CCFGroup(list("abc"), model)],
    )


@pytest.mark.parametrize("k", [1, 2, 3])
def test_independent_shocks_are_independent_basic_events(k):
    model = MGL(0.2, 0.3, shocks="independent")
    Q = float(W([1000, 1.5]).ff(100.0))
    rbd = vote(model, k)
    assert rbd.ff(100.0) == pytest.approx(brute_force(model, Q, k), rel=1e-12)
    # And differ from the exclusive shocks at second order.
    exclusive = vote(MGL(0.2, 0.3), k).ff(100.0)
    assert abs(rbd.ff(100.0) - exclusive) < 10 * Q**2


def test_a_single_shock_is_the_same_either_way():
    for model in (MGL(0.2), MGL(0.2, 1.0)):
        size = model.group_size
        members = list("abc")[:size]
        rbd = [
            NonRepairableRBD(
                [("s", x) for x in members] + [(x, "t") for x in members],
                {x: W([1000, 1.5]) for x in members},
                ccf_groups=[
                    CCFGroup(
                        members,
                        MGL(*model.letters, shocks=shocks),
                    )
                ],
            ).ff(np.array([50.0, 300.0]))
            for shocks in ("exclusive", "independent")
        ]
        np.testing.assert_allclose(rbd[0], rbd[1], rtol=1e-12)


def test_the_shocks_are_kept():
    model = MGL(0.2, 0.3, shocks="independent")
    assert repr(model) == "MGL(0.2, 0.3, shocks='independent')"
    assert model != MGL(0.2, 0.3) and model == MGL(
        0.2, 0.3, shocks="independent"
    )
    assert with_parameters(model, {"beta": 0.1}).shocks == "independent"
    rbd = vote(model, 2)
    back = NonRepairableRBD.from_json(rbd.to_json())
    assert back.ccf_groups[0].model == model
    assert back.ff(100.0) == rbd.ff(100.0)
    assert MGL(0.2, basis="rate").shocks == "independent"
    with pytest.raises(ValueError, match="independently"):
        MGL(0.2, basis="rate", shocks="exclusive")
    with pytest.raises(ValueError, match="shocks must be"):
        MGL(0.2, shocks="joint")
    group = CCFGroup(list("abc"), model)
    with pytest.raises(ValueError, match="independent shocks"):
        draw_ccf_models(group, [MGL(0.2, 0.3)], 5, np.random.default_rng(0))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        drawn = draw_ccf_models(
            group,
            {"beta": surv.Beta.from_params([2, 8])},
            5,
            np.random.default_rng(0),
        )
    assert all(m.shocks == "independent" for m in drawn)
